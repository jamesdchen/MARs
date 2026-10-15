#!/usr/bin/env python3
"""DECISIONS.md D10 sensitivity (reported, never used to judge): each method's P&L with every
fill on a `book_valid = 0` row removed.

    python final/v2/d10_book_valid.py --results-dir final/v2/results --split test \
        --build-dir data/build_jun15 --out final/v2/d10_book_valid.json

Post-hoc, from run_test.py's frozen outputs only (fills.parquet, markets.parquet); no frozen code
is changed. A fill of market m at UTC second t sits on global row row_offset(m) + (t - open_ts(m))
(final/reduce.py's convention); `book_valid` is read from that row of ticks.bin. Each fill
contributes side * qty * (S - price) - fee + rebate to the market's settled P&L (Up frame,
S = settlement = winner_up; inventory settles at S), so

    P&L excluding book_valid=0 fills = P&L - sum of those fills' contributions.

This is a static counterfactual (the later quotes and inventory path are not re-simulated). The
identity sum(contributions) + rewards == P&L is checked per run and its max error is printed.
The mirror variant (Up/Down swapped data) is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINAL))
from common import CODE_ROOT, Dataset, resolve  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="run_test.py --results-dir")
    ap.add_argument("--split", default="test", help="results sub-folder (test / val)")
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rdir = Path(a.results_dir) / a.split
    ds = Dataset(resolve(a.build_dir, CODE_ROOT, CODE_ROOT))
    bv = np.asarray(ds.ticks[:, ds.ci["book_valid"]]) > 0.5
    fl = pd.read_parquet(rdir / "fills.parquet")
    mk = pd.read_parquet(rdir / "markets.parquet")
    keys = ["method", "variant"]
    for k in keys:
        if k not in fl.columns or k not in mk.columns:
            raise SystemExit(f"{k!r} column missing in the results tables")
    M = ds.markets
    fl = fl[~fl["variant"].str.endswith("_mirror")].copy()
    mk = mk[~mk["variant"].str.endswith("_mirror")].copy()
    m = fl["market"].to_numpy(np.int64)
    row = M["row_offset"][m].astype(np.int64) + (fl["t"].to_numpy(np.int64) - M["open_ts"][m].astype(np.int64))
    if (row < M["row_offset"][m]).any() or (row >= M["row_offset"][m] + M["n_rows"][m]).any():
        raise SystemExit("fill rows outside their market")
    S = M["winner_up"][m].astype(float)
    fl["book_valid"] = bv[row]
    fl["contrib"] = fl["side"] * fl["qty"] * (S - fl["price"]) - fl["fee"] + fl["rebate"]
    pnl_col = "pnl" if "pnl" in mk.columns else None
    if pnl_col is None:
        raise SystemExit(f"markets.parquet has no pnl column: {list(mk.columns)}")
    rew_col = "rewards" if "rewards" in mk.columns else None
    out, worst = [], 0.0
    for (meth, var), g in mk.groupby(keys):
        f = fl[(fl["method"] == meth) & (fl["variant"] == var)]
        pnl = float(g[pnl_col].sum())
        ident = float(f["contrib"].sum()) + (float(g[rew_col].sum()) if rew_col else 0.0) - pnl
        worst = max(worst, abs(ident))
        bad = f[~f["book_valid"]]
        out.append(dict(method=meth, variant=var, pnl=pnl, n_fills=int(len(f)), n_fills_book_invalid=int(len(bad)),
                        pnl_of_book_invalid_fills=float(bad["contrib"].sum()),
                        pnl_excluding_book_invalid=pnl - float(bad["contrib"].sum()),
                        n_markets_with_invalid_fills=int(bad["market"].nunique()), identity_error=ident))
    for var in sorted({r["variant"] for r in out}):
        rs = [r for r in out if r["variant"] == var and r["method"].startswith("rl/")]
        if rs:
            out.append(dict(method="rl (seed mean)", variant=var, n_seeds=len(rs),
                            **{k: float(np.mean([r[k] for r in rs])) for k in
                               ("pnl", "n_fills", "n_fills_book_invalid", "pnl_of_book_invalid_fills",
                                "pnl_excluding_book_invalid")}))
    doc = dict(rule="DECISIONS.md D10: P&L with every fill on a book_valid=0 row removed (static); "
                    "reported, not used to judge", results=str(rdir), build_dir=str(a.build_dir),
               n_rows_book_invalid=int((~bv).sum()), max_identity_error=worst, runs=out)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}: {len(out)} runs; max |sum(fill contributions)+rewards-P&L| = {worst:.2e}")
    for r in out:
        if r["variant"] == "central_lat1_x1":
            print(f"{r['method']:28s} P&L {r['pnl']:+9.2f}  ex book_valid=0 {r['pnl_excluding_book_invalid']:+9.2f}"
                  f"  ({r['n_fills_book_invalid']}/{r['n_fills']} fills)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
