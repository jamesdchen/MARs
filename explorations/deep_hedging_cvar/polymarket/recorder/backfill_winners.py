#!/usr/bin/env python3
"""Fill `winner` / `is_winning_outcome` in the recorder's book files (and the resolution columns of
its markets files) once gamma reports the markets as resolved.

    python backfill_winners.py --out ./out [--days 2026-10-15,2026-10-16] [--dry-run]

* winner = 'yes' when outcome 0 (Up) won, 'no' when outcome 1 (Down) won (data/raw convention);
  is_winning_outcome = this row's token is the winning token.
* Parts of finished hours are compacted first (same code and lock as the recorder). Only compacted
  hourly files are rewritten (never the in-progress _parts of the current hour), each
  atomically (temp file + rename) under the same lock the recorder's compaction uses; the part
  manifest in the file metadata is preserved. Safe to run from cron while the recorder is running.
* Resolutions are cached in <out>/state/resolutions.json, so reruns only query new markets.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from polyrec.discovery import fetch_market_final, jl, resolution  # noqa: E402
from polyrec.http import Http  # noqa: E402
from polyrec.metadata import market_row  # noqa: E402
from polyrec.schema import BOOK_SCHEMA, MARKETS_SCHEMA  # noqa: E402
from polyrec.sinks import BookRowBuffer, TableSink, file_lock, hour_key  # noqa: E402

RES_COLS = ["closed_time", "outcome_price_up", "winner_up", "final_price", "uma_status", "tick_size"]


def final_files(out: Path, name: str, days):
    base = out / name
    if not base.exists():
        return []
    return sorted(f for d in sorted(base.iterdir()) if d.is_dir() and (days is None or d.name in days)
                  for f in d.glob(f"{name}_*.parquet"))


async def fetch_resolutions(slugs, cache: dict, workers: int = 6) -> dict:
    todo = [s for s in slugs if s not in cache]
    sem = asyncio.Semaphore(workers)
    async with Http() as http:
        async def one(s):
            async with sem:
                try:
                    g = await fetch_market_final(http, s)
                except Exception as e:  # noqa: BLE001
                    print(f"  {s}: {e}", file=sys.stderr)
                    return
                if g is None:
                    return
                win, _ = resolution(g)
                if win is None:
                    return
                toks = [str(t) for t in jl(g.get("clobTokenIds") or "[]")]
                keep = ("id", "outcomes", "outcomePrices", "clobTokenIds", "closed", "closedTime",
                        "umaResolutionStatus", "eventMetadata", "orderPriceMinTickSize")
                cache[s] = {"winner_idx": win, "winning_token": toks[win] if len(toks) == 2 else None,
                            "gamma": {k: g.get(k) for k in keep}, "at": int(time.time())}
        await asyncio.gather(*(one(s) for s in todo))
    return cache


def rewrite(path: Path, table: pa.Table):
    md = pq.read_schema(path).metadata
    if md:
        table = table.replace_schema_metadata(md)
    tmp = path.with_name(f".{path.name}.backfill")
    pq.write_table(table, tmp, compression="zstd", compression_level=9,
                   use_dictionary=["market_slug", "coin", "contract_length", "token_id", "outcome"]
                   if "market_slug" in table.column_names else True)
    tmp.replace(path)


def fill_book_file(path: Path, resolved: dict) -> int:
    """Fill null winner / is_winning_outcome of resolved markets in one books file; returns rows filled.
    resolved: slug -> {"winner_idx": 0|1, "winning_token": token_id}."""
    if not resolved:
        return 0
    t = pq.read_table(path)
    slugs = pa.array(list(resolved), pa.string())
    win_tok = pa.array([resolved[s]["winning_token"] for s in resolved], pa.string())
    win_yes = pa.array(["yes" if resolved[s]["winner_idx"] == 0 else "no" for s in resolved], pa.string())
    idx = pc.index_in(t["market_slug"], slugs)
    fill = pc.and_(pc.is_valid(idx), pc.is_null(t["winner"]))
    n = int(pc.sum(fill.cast(pa.int64())).as_py() or 0)
    if not n:
        return 0
    w = pc.if_else(fill, pc.take(win_yes, idx), t["winner"])
    iw = pc.if_else(fill, pc.equal(t["token_id"], pc.take(win_tok, idx)), t["is_winning_outcome"])
    t = t.set_column(t.schema.get_field_index("winner"), "winner", w.cast(pa.string()))
    t = t.set_column(t.schema.get_field_index("is_winning_outcome"), "is_winning_outcome", iw.cast(pa.bool_()))
    rewrite(path, t.cast(BOOK_SCHEMA))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", default=None, help="comma-separated UTC days (default: all)")
    ap.add_argument("--recent", type=int, default=None, help="only the last N UTC days (cron)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    days = a.days.split(",") if a.days else None
    if a.recent:
        days = [time.strftime("%Y-%m-%d", time.gmtime(time.time() - k * 86400)) for k in range(a.recent)]
    state = out / "state"
    state.mkdir(exist_ok=True)
    cache_p = state / "resolutions.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    # parts of finished hours (recorder stopped before compacting them) -> compact first
    cur = hour_key(time.time())
    for sink in (BookRowBuffer(out, "backfill"), TableSink(out, "markets", MARKETS_SCHEMA, "backfill")):
        for h in sink.part_hours():
            if h < cur and (days is None or h[:10] in days):
                sink.compact(h)
    book_files = final_files(out, "books", days)
    need = set()
    for f in book_files:
        t = pq.read_table(f, columns=["market_slug", "winner"])
        need |= set(pc.unique(t.filter(pc.is_null(t["winner"]))["market_slug"]).to_pylist())
    mk_files = final_files(out, "markets", days)
    for f in mk_files:
        t = pq.read_table(f, columns=["slug", "winner_up"])
        need |= set(pc.unique(t.filter(pc.is_null(t["winner_up"]))["slug"]).to_pylist())
    print(f"{len(need)} markets without a winner; {len([s for s in need if s in cache])} cached", flush=True)
    asyncio.run(fetch_resolutions(sorted(need), cache))
    if not a.dry_run:
        tmp = cache_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cache_p)
    resolved = {s: cache[s] for s in need if s in cache}
    print(f"resolved now: {len(resolved)}; still open: {len(need) - len(resolved)}")
    if a.dry_run or not resolved:
        return 0

    n_rows = n_files = 0
    for f in book_files:
        with file_lock(out, "books"):
            k = fill_book_file(f, resolved)
        n_rows += k
        n_files += k > 0
    n_mk = 0
    for f in mk_files:
        with file_lock(out, "markets"):
            df = pq.read_table(f).to_pandas()
            hit = df.winner_up.isna() & df.slug.isin(list(resolved))
            if not hit.any():
                continue
            for i in df.index[hit]:
                r = df.loc[i]
                new = market_row(r.slug, r.coin, r.contract_length, resolved[r.slug]["gamma"],
                                 r.n_taker, r.n_fills, r.truncated)
                for c in RES_COLS:
                    if c == "tick_size" and not (new[c] == new[c]):      # NaN: keep the finalize-time value
                        continue
                    df.at[i, c] = new[c]
            rewrite(f, pa.Table.from_pandas(df, schema=MARKETS_SCHEMA, preserve_index=False))
            n_mk += int(hit.sum())
    print(f"filled {n_rows} book rows in {n_files} files; {n_mk} market rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
