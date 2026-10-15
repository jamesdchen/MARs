#!/usr/bin/env python3
"""Merge recorder output into a data/raw-shaped directory and rebuild the env dataset with the
unmodified data/build.py, data/load.py and data/features.py.

    python merge.py --rec ./out [--rec /mnt/vps2/out] --dest ./merged [--splits split_config.json]
                    [--dry-run] [--no-build] [--no-features] [--python /usr/bin/python3]

What it does
  1. Selects recorded markets that are usable: book rows + resolved metadata (run backfill_winners.py
     / `python -m polyrec finalize` first), full coverage of [open, end] by the recorder (coverage log:
     max single gap / total gap), contract length and coin known to build.py, trade legs that pass the
     checks data/features.py asserts, the market's UTC day assigned to a split, and Binance 1 s
     underlying available for [open - 1 h, end] (fetched with data/fetch_underlying.py's own functions).
  2. Writes <dest>/raw/{updown_5m,updown_15m,updown_4h,markets,trades,underlying}.parquet: the ORIGINAL
     data/raw rows first, untouched, then the new markets, in exactly the data/raw schemas (the
     recorder's extra `reconciled` column is dropped; winner/is_winning_outcome filled from metadata).
     Book files are streamed row group by row group, so memory stays bounded by one hourly file.
  3. Resolves the day -> split map (polyrec/splits.py: frozen original days, explicit days, a fixed
     8-day cycle for the rest, and a lock file so no day ever moves; the original test day can never
     become train), then runs data/build.py with RAW/OUT/SPLIT_DAYS pointed at <dest> (module globals
     patched in a subprocess; the files in data/ are not modified), data/load.py verify(), and
     data/features.py (needs scipy: system python by default).
  4. Checks that the original test block (2026-08-24) is byte-identical to data/build's, when present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
from polyrec import coverage as cov  # noqa: E402
from polyrec import splits as sp  # noqa: E402
from polyrec.schema import BOOK_SCHEMA_RAW, MARKETS_SCHEMA, TRADES_SCHEMA  # noqa: E402
from polyrec.sinks import dataset_files, read_dataset  # noqa: E402
from polyrec.trades import trade_sanity  # noqa: E402

CLEN = {"5m": 300, "15m": 900, "4h": 14400}


def build_constants():
    sys.path.insert(0, str(DATA))
    import build  # data/build.py (import only; main() is guarded)
    return dict(SPLIT_DAYS=dict(build.SPLIT_DAYS), COINS=list(build.COINS), CLEN=dict(build.CLEN))


def day_of(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------- selection
def load_recorded(recs: list[Path]):
    metas, trades, qcs, events = [], [], [], []
    for r in recs:
        m = read_dataset(r, "markets", MARKETS_SCHEMA)
        if m is not None:
            metas.append(m.to_pandas())
        t = read_dataset(r, "trades")
        if t is not None:
            trades.append(t.cast(TRADES_SCHEMA))
        q = read_dataset(r, "market_qc")
        if q is not None:
            qcs.append(q.to_pandas())
        events.extend(cov.read_events(r))
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame(columns=MARKETS_SCHEMA.names)
    # one row per slug: prefer resolved, then the latest
    meta["_res"] = meta.winner_up.notna()
    meta = meta.sort_values(["slug", "_res"]).drop_duplicates("slug", keep="last").drop(columns="_res")
    tr = pa.concat_tables(trades).unify_dictionaries() if trades else TRADES_SCHEMA.empty_table()
    return meta.set_index("slug"), tr, cov.live_intervals(events)


def book_slugs(recs: list[Path]) -> dict:
    """slug -> (n_rows, first captured_at s, last captured_at s)."""
    out = {}
    for r in recs:
        for f in dataset_files(r, "books"):
            t = pq.read_table(f, columns=["market_slug", "captured_at"])
            df = t.to_pandas()
            g = df.groupby("market_slug").captured_at.agg(["size", "min", "max"])
            for s, row in g.iterrows():
                n, a, b = out.get(s, (0, np.inf, -np.inf))
                out[s] = (n + int(row["size"]), min(a, row["min"].timestamp()), max(b, row["max"].timestamp()))
    return out


def select(recs, meta, trades, intervals, consts, assignment, orig_slugs, args):
    rows = []
    tr_df = trades.select(["slug", "tx", "is_taker", "size"]).to_pandas() if trades.num_rows else None
    sanity = trade_sanity(tr_df) if tr_df is not None and len(tr_df) else {}
    for slug, (n, first, last) in sorted(book_slugs(recs).items()):
        r = {"slug": slug, "n_rows": n, "reason": ""}
        m = meta.loc[slug] if slug in meta.index else None
        if slug in orig_slugs:
            r["reason"] = "already in data/raw"
        elif m is None:
            r["reason"] = "no metadata (run: python -m polyrec finalize)"
        elif m.contract_length not in consts["CLEN"] or m.coin not in consts["COINS"]:
            r["reason"] = f"coin/length {m.coin}/{m.contract_length} unknown to build.py"
        elif pd.isna(m.winner_up):
            r["reason"] = "unresolved (run backfill_winners.py)"
        else:
            end = m.end.timestamp()
            o = m.event_start.timestamp()
            r.update(open=o, end=end, coin=m.coin, contract_length=m.contract_length)
            ok, tot, mx = cov.market_coverage(intervals.get(slug, []), o, end, args.max_gap, args.max_total_gap)
            r.update(cov_total_gap=tot, cov_max_gap=mx)
            d0, d1 = day_of(o), day_of(end - 1)
            r.update(day=d0)
            if abs(end - o - consts["CLEN"][m.contract_length]) > 0.5:
                r["reason"] = "eventStartTime != end - contract length"
            elif not ok:
                r["reason"] = f"coverage gap total {tot:.1f}s max {mx:.1f}s"
            elif first > o:
                r["reason"] = "first book row after open"
            elif assignment.get(d0, "exclude") == "exclude" or assignment.get(d1, "exclude") == "exclude":
                r["reason"] = f"day {d0} excluded by split config"
            elif slug in sanity and not sanity[slug]["ok"]:
                r["reason"] = f"trade legs fail features.py checks {sanity[slug]}"
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- underlying
def fetch_underlying(dest: Path, coins: list, days: list, workers: int = 4) -> pd.DataFrame:
    sys.path.insert(0, str(DATA))
    import fetch_underlying as fu   # data/fetch_underlying.py, cache redirected into dest
    from concurrent.futures import ThreadPoolExecutor
    fu.CACHE = dest / "cache" / "binance"
    fu.CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(c, d) for c in coins for d in days]
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(lambda cd: fu.get_zip(*cd), jobs))
    frames = [fu.parse(c, d, p) for (c, d), p in zip(jobs, paths) if p is not None]
    if not frames:
        return pd.DataFrame(columns=["coin", "open_time", "close", "source"])
    df = pd.concat(frames, ignore_index=True).sort_values(["coin", "open_time"])
    df["close"] = df.groupby("coin").close.ffill()
    return df.dropna(subset=["close"])


def underlying_ok(ul: pd.DataFrame, coin: str, lo: int, hi: int) -> bool:
    g = ul[(ul.coin == coin) & (ul.open_time >= lo) & (ul.open_time < hi)]
    return len(g) == hi - lo


# ----------------------------------------------------------------------------- writers
def write_books(dest_raw: Path, recs, chosen: pd.DataFrame, meta: pd.DataFrame):
    winners = {s: int(meta.loc[s].winner_up) for s in chosen.slug}
    token_up = {s: meta.loc[s].token_up for s in chosen.slug}
    keep = set(chosen.slug)
    counts = {}
    for cl in ("5m", "15m", "4h"):
        dst = dest_raw / f"updown_{cl}.parquet"
        tmp = dst.with_suffix(".tmp")
        with pq.ParquetWriter(tmp, BOOK_SCHEMA_RAW, compression="zstd", compression_level=9) as w:
            orig = pq.ParquetFile(DATA / "raw" / f"updown_{cl}.parquet")
            assert orig.schema_arrow.equals(BOOK_SCHEMA_RAW), "data/raw schema changed?"
            for i in range(orig.num_row_groups):
                w.write_table(orig.read_row_group(i))
            n_new = 0
            for r in recs:
                for f in dataset_files(r, "books"):
                    t = pq.read_table(f)
                    m = pc.and_(pc.is_in(t["market_slug"], pa.array(sorted(keep))), pc.equal(t["contract_length"], cl))
                    t = t.filter(m)
                    if not t.num_rows:
                        continue
                    df = t.drop_columns(["reconciled"]).to_pandas()
                    df = df.drop_duplicates(["token_id", "captured_at"], keep="last")
                    df = df[df.seconds_to_close >= 0]
                    wu = df.market_slug.map(winners)
                    df["winner"] = np.where(wu == 1, "yes", "no")
                    up = df.token_id == df.market_slug.map(token_up)
                    df["is_winning_outcome"] = np.where(wu == 1, up, ~up)
                    df = df.sort_values(["market_slug", "token_id", "captured_at"])
                    w.write_table(pa.Table.from_pandas(df, schema=BOOK_SCHEMA_RAW, preserve_index=False))
                    n_new += len(df)
            counts[cl] = n_new
        tmp.replace(dst)
    return counts


def write_trades(dest_raw: Path, trades: pa.Table, chosen_slugs: list):
    orig = pq.read_table(DATA / "raw" / "trades.parquet")
    new = trades.filter(pc.is_in(pc.cast(trades["slug"], pa.string()), pa.array(sorted(chosen_slugs)))) \
        if trades.num_rows else TRADES_SCHEMA.empty_table()
    t = pa.concat_tables([orig.cast(TRADES_SCHEMA), new.cast(TRADES_SCHEMA)]).unify_dictionaries()
    t = t.combine_chunks()
    pq.write_table(t, dest_raw / "trades.parquet", compression="zstd", compression_level=19,
                   use_dictionary=["slug"], row_group_size=1 << 20)
    return new.num_rows


def write_markets(dest_raw: Path, meta: pd.DataFrame, chosen_slugs: list):
    orig = pd.read_parquet(DATA / "raw" / "markets.parquet")
    new = meta.loc[chosen_slugs].reset_index()[orig.columns]
    allm = pd.concat([orig, new], ignore_index=True)
    allm["winner_up"] = allm.winner_up.astype(np.int64)
    pa_t = pa.Table.from_pandas(allm, schema=MARKETS_SCHEMA, preserve_index=False)
    pq.write_table(pa_t, dest_raw / "markets.parquet", compression="zstd")


def write_underlying(dest_raw: Path, new_ul: pd.DataFrame):
    orig = pd.read_parquet(DATA / "raw" / "underlying.parquet")
    orig["coin"] = orig.coin.astype(str)
    orig["source"] = orig.source.astype(str)
    if len(new_ul):
        new_ul = new_ul.copy()
        key = set(zip(orig.coin, orig.open_time // 86400))
        new_ul = new_ul[[k not in key for k in zip(new_ul.coin, new_ul.open_time // 86400)]]
    df = pd.concat([orig, new_ul[orig.columns]], ignore_index=True).sort_values(["coin", "open_time"], kind="stable")
    tbl = pa.table({"coin": pa.array(df.coin.values).dictionary_encode(),
                    "open_time": pa.array(df.open_time.values.astype(np.int64)),
                    "close": pa.array(df.close.values.astype(np.float64)),
                    "source": pa.array(df.source.values).dictionary_encode()})
    pq.write_table(tbl, dest_raw / "underlying.parquet", compression="zstd", compression_level=19,
                   use_byte_stream_split=["close"], row_group_size=1 << 20)


# ----------------------------------------------------------------------------- build / features
RUN_BUILD = """
import json, sys
from pathlib import Path
sys.path.insert(0, {data!r})
import build
build.RAW = Path({raw!r}); build.OUT = Path({out!r})
build.SPLIT_DAYS = json.loads({split!r})
build.main()
import load
load.verify(Path({out!r}))
"""
RUN_FEATURES = """
import sys
from pathlib import Path
sys.path.insert(0, {data!r})
import load, features
features.BUILD = Path({out!r}); features.RAW = Path({raw!r})
features.load = lambda: load.load(Path({out!r}))
features.main()
"""


def run_py(python: str, code: str):
    print(f"$ {python} -c <{code.strip().splitlines()[-1]}>", flush=True)
    subprocess.run([python, "-c", code], check=True, cwd=str(DATA))


def check_original_test_block(build_dir: Path) -> dict:
    """The original test markets come first in the new test split: their rows must be byte-identical."""
    orig_meta_p = DATA / "build" / "meta.json"
    if not orig_meta_p.exists():
        return {"checked": False, "reason": "data/build/meta.json missing"}
    om = json.loads(orig_meta_p.read_text())
    nm = json.loads((build_dir / "meta.json").read_text())
    ot, nt = om["test_rows"], nm["test_rows"]
    n = ot["row_hi"] - ot["row_lo"]
    ticks = np.memmap(build_dir / "ticks.bin", dtype=np.float32, mode="r", shape=(nm["N"], nm["F"]))
    sha = hashlib.sha256(np.ascontiguousarray(ticks[nt["row_lo"]: nt["row_lo"] + n]).tobytes()).hexdigest()
    return {"checked": True, "identical": sha == ot["sha256"], "rows": n}


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", action="append", required=True, help="recorder out_dir (repeatable)")
    ap.add_argument("--dest", default=str(HERE / "merged"))
    ap.add_argument("--splits", default=str(HERE / "split_config.json"))
    ap.add_argument("--lock", default=None, help="split lock file (default: next to --splits)")
    ap.add_argument("--reassign", action="store_true", help="allow changing a locked non-test day")
    ap.add_argument("--max-gap", type=float, default=5.0, help="max single coverage gap within [open, end] (s)")
    ap.add_argument("--max-total-gap", type=float, default=15.0)
    ap.add_argument("--allow-missing-underlying", action="store_true",
                    help="keep markets without Binance data (build marks them; features.py cannot run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--python", default="/usr/bin/python3", help="interpreter for build/features (needs scipy)")
    a = ap.parse_args(argv)
    t0 = time.time()
    recs = [Path(r) for r in a.rec]
    dest = Path(a.dest)
    consts = build_constants()
    cfg = json.loads(Path(a.splits).read_text())
    lock_path = Path(a.lock) if a.lock else Path(a.splits).with_name(cfg.get("lock_file", "split_lock.json"))
    lock = sp.load_lock(lock_path)

    meta, trades, intervals = load_recorded(recs)
    orig_slugs = set(pd.read_parquet(DATA / "raw" / "markets.parquet", columns=["slug"]).slug)
    rec_days = sorted({day_of(t.timestamp()) for t in meta.event_start.dropna()} |
                      {day_of(t.timestamp() - 1) for t in meta.end.dropna()})
    assignment = sp.resolve(rec_days, cfg, consts["SPLIT_DAYS"], lock, a.reassign)
    sel = select(recs, meta, trades, intervals, consts, assignment, orig_slugs, a)
    cand = sel[sel.reason == ""].copy()

    # underlying for the candidate markets
    new_ul = pd.DataFrame(columns=["coin", "open_time", "close", "source"])
    if len(cand) and not a.dry_run:
        lo = min(cand.open) - 86400
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(day_of(lo), day_of(max(cand.end)), freq="D")]
        orig_days = {d for d in consts["SPLIT_DAYS"]}
        days = [d for d in days if d not in orig_days]
        print(f"fetching Binance 1s underlying for {sorted(cand.coin.unique())} x {days}", flush=True)
        new_ul = fetch_underlying(dest, sorted(cand.coin.unique()), days)
        for i, r in cand.iterrows():
            if not underlying_ok(new_ul, r.coin, int(r.open) - 3600, int(r.end)):
                if not a.allow_missing_underlying:
                    sel.loc[i, "reason"] = "no Binance underlying yet (published next day)"
        cand = sel[sel.reason == ""].copy()

    summary = {
        "recorded_markets": int(len(sel)), "selected": int(len(cand)),
        "excluded_by_reason": sel[sel.reason != ""].reason.str.split(" \\(|:|\\{", regex=True).str[0]
        .value_counts().to_dict(),
        "selected_by_day_split": {d: {"split": assignment.get(d), "markets": int(n)}
                                  for d, n in cand.groupby("day").size().items()} if len(cand) else {},
        "assignment": {d: s for d, s in assignment.items()},
    }
    print(json.dumps(summary, indent=1))
    if a.dry_run:
        sel.to_csv(sys.stdout, index=False)
        return 0
    if not len(cand):
        print("nothing to merge")
        return 1

    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    chosen = sorted(cand.slug)
    counts = write_books(raw, recs, cand, meta)
    n_tr = write_trades(raw, trades, chosen)
    write_markets(raw, meta, chosen)
    write_underlying(raw, new_ul)
    used_days = sorted(set(cand.day) | {day_of(e - 1) for e in cand.end})
    build_days = sp.to_build_split_days({d: s for d, s in assignment.items()
                                         if d in consts["SPLIT_DAYS"] or d in used_days})
    sel.to_csv(dest / "merge_selection.csv", index=False)
    summary.update(new_book_rows=counts, new_trade_legs=int(n_tr), build_split_days=build_days)
    if not a.no_build:
        out = dest / "build"
        out.mkdir(exist_ok=True)
        run_py(a.python, RUN_BUILD.format(data=str(DATA), raw=str(raw), out=str(out), split=json.dumps(build_days)))
        summary["original_test_block"] = check_original_test_block(out)
        print("original test block:", summary["original_test_block"])
        if summary["original_test_block"].get("identical") is False:
            raise SystemExit("original test rows changed: refusing (check data/raw vs merged inputs)")
        if not a.no_features:
            run_py(a.python, RUN_FEATURES.format(data=str(DATA), raw=str(raw), out=str(out)))
    sp.save_lock(lock_path, lock, {d: s for d, s in assignment.items() if d in build_days})
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (dest / "merge_report.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"merged {len(chosen)} new markets into {dest} in {summary['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
