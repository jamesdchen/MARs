#!/usr/bin/env python3
"""Check recorder output against data/raw conventions.

    python verify_output.py --out ./out [--json report.json]

1. Every books file loads with the same reader as data/raw (pandas.read_parquet / pyarrow) and its
   schema equals data/raw/updown_5m.parquet's field for field (plus the trailing `reconciled` column).
2. Row checks from data/raw/README.md ("Verifying the data yourself") and data/build.py's assumptions:
   ladders ordered (bids descending, asks ascending), price/size lists aligned, sizes > 0, prices in
   (0, 1), best_bid/best_ask = first level or null, crossed == best_bid >= best_ask (False when a side
   is empty), seconds_to_close == market_end_at - captured_at, whole-second captured_at, unique
   (token_id, captured_at), constant market_end_at per market, two tokens per market, winner filled
   consistently where present.
3. Up/Down mirror agreement at equal timestamps (top of book), crossed / one-sided rates.
4. Trades / markets files: schema equals data/raw/trades.parquet / markets.parquet.
Run merge.py (it calls data/load.py verify()) for the dataset-level checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA_RAW = HERE.parent / "data" / "raw"
sys.path.insert(0, str(HERE))
from polyrec.schema import BOOK_SCHEMA, BOOK_SCHEMA_RAW, MARKETS_SCHEMA, TRADES_SCHEMA  # noqa: E402
from polyrec.sinks import dataset_files  # noqa: E402


def check(cond, msg, errors):
    if not bool(cond):
        errors.append(msg)
    return bool(cond)


def ladder_ok(px: np.ndarray, desc: bool) -> bool:
    if len(px) < 2:
        return True
    d = np.diff(px)
    return bool((d < 0).all() if desc else (d > 0).all())


def verify_books(out: Path, errors: list) -> dict:
    files = dataset_files(out, "books")
    check(files, "no books files", errors)
    if not files:
        return {}
    ref = pq.read_schema(DATA_RAW / "updown_5m.parquet") if (DATA_RAW / "updown_5m.parquet").exists() else BOOK_SCHEMA_RAW
    for f in files:
        s = pq.read_schema(f).remove_metadata()
        check(s.equals(BOOK_SCHEMA), f"{f.name}: schema differs from BOOK_SCHEMA", errors)
        check(pa_fields(s)[:17] == pa_fields(ref), f"{f.name}: first 17 fields differ from data/raw", errors)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)   # the data/raw reader
    n = len(df)
    bp, ap = df.bid_prices.values, df.ask_prices.values
    bs, asz = df.bid_sizes.values, df.ask_sizes.values
    nb = np.fromiter((len(x) for x in bp), int, n)
    na = np.fromiter((len(x) for x in ap), int, n)
    check((nb == np.fromiter((len(x) for x in bs), int, n)).all(), "bid price/size lengths differ", errors)
    check((na == np.fromiter((len(x) for x in asz), int, n)).all(), "ask price/size lengths differ", errors)
    check(all(ladder_ok(x, True) for x in bp), "bid ladder not strictly descending", errors)
    check(all(ladder_ok(x, False) for x in ap), "ask ladder not strictly ascending", errors)
    allp = np.concatenate([np.concatenate(bp) if n else [], np.concatenate(ap) if n else []])
    alls = np.concatenate([np.concatenate(bs) if n else [], np.concatenate(asz) if n else []])
    check(((allp > 0) & (allp < 1)).all(), "price outside (0, 1)", errors)
    check((alls > 0).all() and np.isfinite(alls).all(), "non-positive or non-finite size", errors)
    bb = np.array([x[0] if len(x) else np.nan for x in bp])
    ba = np.array([x[0] if len(x) else np.nan for x in ap])
    check(np.array_equal(np.nan_to_num(bb, nan=-1), df.best_bid.fillna(-1).values), "best_bid != bid_prices[0]", errors)
    check(np.array_equal(np.nan_to_num(ba, nan=-1), df.best_ask.fillna(-1).values), "best_ask != ask_prices[0]", errors)
    crossed = (df.best_bid >= df.best_ask).fillna(False).values
    check((df.crossed.values == crossed).all(), "crossed != best_bid >= best_ask", errors)
    stc = (df.market_end_at - df.captured_at).dt.total_seconds().values
    check(np.allclose(stc, df.seconds_to_close.values), "seconds_to_close inconsistent", errors)
    check((df.captured_at.astype("int64") % 1_000_000 == 0).all(), "captured_at not whole seconds", errors)
    check(not df.duplicated(["token_id", "captured_at"]).any(), "duplicate (token_id, captured_at)", errors)
    g = df.groupby("market_slug")
    check((g.market_end_at.nunique() == 1).all(), "market_end_at varies within a market", errors)
    check((g.token_id.nunique() == 2).all(), "market without exactly two tokens", errors)
    check((df.groupby("token_id").outcome.nunique() == 1).all(), "token with several outcomes", errors)
    w = df[df.winner.notna()]
    if len(w):
        check(w.winner.isin(["yes", "no"]).all(), "winner not yes/no", errors)
        upwin = (w.winner == "yes")
        check(w.is_winning_outcome.notna().all(), "winner set but is_winning_outcome null", errors)
        is_first = w.outcome.isin(["Up", "Yes"])
        check((w.is_winning_outcome == (is_first == upwin)).all(), "is_winning_outcome inconsistent with winner", errors)

    # Up/Down mirror at equal timestamps (top of book): Up bid = 1 - Down ask
    up = df[df.outcome == "Up"].set_index(["market_slug", "captured_at"])
    dn = df[df.outcome == "Down"].set_index(["market_slug", "captured_at"])
    common = up.index.intersection(dn.index)
    u, d = up.loc[common], dn.loc[common]
    m_bid = np.isclose(u.best_bid.fillna(-9).values, (1 - d.best_ask).fillna(-9).values, atol=1e-9)
    m_ask = np.isclose(u.best_ask.fillna(-9).values, (1 - d.best_bid).fillna(-9).values, atol=1e-9)
    per_len = df.groupby("contract_length").agg(rows=("crossed", "size"), crossed=("crossed", "mean"))
    one_sided = df.groupby("contract_length").apply(
        lambda x: float((x.best_bid.isna() | x.best_ask.isna()).mean()), include_groups=False)
    return {
        "files": len(files), "rows": n, "markets": int(df.market_slug.nunique()), "tokens": int(df.token_id.nunique()),
        "rows_reconciled": int(df.reconciled.sum()),
        "rows_by_length": per_len.rows.to_dict(), "crossed_rate_by_length": per_len.crossed.round(5).to_dict(),
        "one_sided_rate_by_length": one_sided.round(4).to_dict(),
        "median_bid_levels": float(np.median(nb)), "median_ask_levels": float(np.median(na)),
        "up_down_same_second": int(len(common)),
        "up_down_top_mirror_rate": float((m_bid & m_ask).mean()) if len(common) else None,
        "rows_per_token_second": float(n / max(1, df.token_id.nunique()) /
                                       max(1.0, (df.captured_at.max() - df.captured_at.min()).total_seconds())),
        "winner_filled_rate": float(df.winner.notna().mean()),
        "bytes_on_disk": int(sum(f.stat().st_size for f in files)),
    }


def pa_fields(schema):
    return [(f.name, str(f.type)) for f in schema]


def verify_table(out: Path, name: str, schema, ref_path: Path, errors: list) -> dict:
    files = dataset_files(out, name)
    if not files:
        return {"files": 0}
    ref = pq.read_schema(ref_path) if ref_path.exists() else schema
    rows = 0
    for f in files:
        s = pq.read_schema(f)
        check(pa_fields(s) == pa_fields(ref), f"{f.name}: schema differs from {ref_path.name}", errors)
        rows += pq.ParquetFile(f).metadata.num_rows
        pd.read_parquet(f)
    return {"files": len(files), "rows": rows, "bytes_on_disk": int(sum(f.stat().st_size for f in files))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    errors: list = []
    rep = {"books": verify_books(out, errors),
           "trades": verify_table(out, "trades", TRADES_SCHEMA, DATA_RAW / "trades.parquet", errors),
           "markets": verify_table(out, "markets", MARKETS_SCHEMA, DATA_RAW / "markets.parquet", errors)}
    raw = sorted((out / "raw").rglob("*.jsonl.zst")) if (out / "raw").exists() else []
    rep["raw_log"] = {"files": len(raw), "bytes_on_disk": int(sum(f.stat().st_size for f in raw))}
    rep["errors"] = [e for e in errors if e]
    print(json.dumps(rep, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
