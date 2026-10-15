#!/usr/bin/env python3
"""Coverage / fill-realism statistics for COVERAGE.md (run after build.py).

Prints the numbers and writes build/coverage_stats.json.
"At/through" test: each taker transaction at second t (open < t <= end) is
compared with the book prevailing BEFORE it, i.e. ticks row t-1 (latest
uncrossed snapshot captured <= t-1). Prices are the maker legs' resting prices
mapped to the Up frame. For an aggressor BUY we use its worst (max) price vs
best ask; for an aggressor SELL its worst (min) price vs best bid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from load import load

HERE = Path(__file__).resolve().parent
CL = {300: "5m", 900: "15m", 14400: "4h"}
EPS = 1e-6


def main():
    ticks, mk, meta = load()
    c = meta["column_index"]
    slugs = meta["slugs"]
    out = {}
    m = pd.DataFrame({k: mk[k] for k in mk.dtype.names})
    m["slug"] = slugs
    m["cl"] = m.contract_len_s.map(CL)
    m["has_trades"] = (m["flags"] & 1) > 0
    m["has_ul"] = (m["flags"] & 2) > 0
    m["has_strike"] = (m["flags"] & 8) > 0
    out["markets"] = {"total": len(m), "with_trades_in_window": int(m.has_trades.sum()),
                      "with_underlying": int(m.has_ul.sum()), "with_strike": int(m.has_strike.sum()),
                      "hype_perp_proxy": int(((m["flags"] & 4) > 0).sum())}
    # rows -> market
    mid = np.repeat(np.arange(len(m)), m.n_rows.values)
    tc = ticks[:, c["trade_count"]]
    rows = pd.DataFrame({"cl": m.cl.values[mid], "tc": tc,
                         "valid": ticks[:, c["book_valid"]], "xraw": ticks[:, c["crossed_raw"]],
                         "fromdn": ticks[:, c["book_from_down"]], "age": ticks[:, c["book_age"]],
                         "bid_empty": ticks[:, c["bid_sz_0"]] == 0, "ask_empty": ticks[:, c["ask_sz_0"]] == 0})
    g = rows.groupby("cl")
    out["seconds"] = pd.DataFrame({
        "rows": g.size(),
        "frac_with_taker_trade": g.tc.apply(lambda s: (s > 0).mean()),
        "book_valid": g.valid.mean(), "crossed_raw": g.xraw.mean(), "book_from_down": g.fromdn.mean(),
        "book_age_median": g.age.median(), "book_age_p90": g.age.quantile(0.9),
        "one_sided": g.apply(lambda d: (d.bid_empty | d.ask_empty).mean()),
    }).to_dict(orient="index")
    # trades per market (in window) and incl. pre-open
    trades = pd.read_parquet(HERE / "raw" / "trades.parquet")
    trades["slug"] = trades.slug.astype(str)
    tk = trades[trades.is_taker].merge(m[["slug", "open_ts", "end_ts", "cl"]], on="slug")
    tk["in_win"] = (tk.timestamp >= tk.open_ts) & (tk.timestamp <= tk.end_ts)
    per = tk.groupby("slug").agg(n_all=("tx", "size"), n_win=("in_win", "sum"))
    per = m[["slug", "cl"]].set_index("slug").join(per).fillna(0)
    out["taker_trades_per_market"] = per.groupby("cl").agg(
        median_in_window=("n_win", "median"), mean_in_window=("n_win", "mean"),
        median_incl_preopen=("n_all", "median"), markets_zero_in_window=("n_win", lambda s: int((s == 0).sum()))
    ).to_dict(orient="index")
    out["taker_trades_total"] = {"all": int(len(tk)), "in_window": int(tk.in_win.sum()),
                                 "pre_open": int((tk.timestamp < tk.open_ts).sum()),
                                 "after_end": int((tk.timestamp > tk.end_ts).sum())}
    # at/through analysis
    mk_legs = trades[~trades.is_taker].copy()
    mk_legs["pu"] = np.where(mk_legs.outcome_idx == 0, mk_legs.price, 1 - mk_legs.price).astype(np.float64)
    agg = mk_legs.groupby("tx").agg(pmax=("pu", "max"), pmin=("pu", "min"), vol=("size", "sum"))
    MAXLAG = 10
    w = tk[tk.in_win & (tk.timestamp >= tk.open_ts + MAXLAG)].set_index("tx").join(agg)
    mrow = dict(zip(m.slug, m.row_offset - m.open_ts))
    w["row"] = w.slug.map(mrow) + w.timestamp
    res = {}
    for lag in (0, 1, 2, 3, 4, 5, 6, 8, 10):
        lag_name = f"book_t_minus_{lag}"
        r = (w.row - lag).values.astype(np.int64)
        bb = ticks[r, c["bid_px_0"]].astype(np.float64)
        ba = ticks[r, c["ask_px_0"]].astype(np.float64)
        bsz = ticks[r, c["bid_sz_0"]]
        asz = ticks[r, c["ask_sz_0"]]
        valid = ticks[r, c["book_valid"]] == 1
        buy = w.aggressor_up.values == 1
        p = np.where(buy, w.pmax.values, w.pmin.values)
        best = np.where(buy, ba, bb)
        opp = np.where(buy, bb, ba)
        has_q = np.where(buy, asz > 0, bsz > 0)
        through = np.where(buy, p > best + EPS, p < best - EPS)
        at = np.abs(p - best) <= EPS
        better_than_opp = np.where(buy, p <= opp + EPS, p >= opp - EPS)   # printed at/through the SAME-side best
        inside = ~through & ~at & ~better_than_opp
        cat = np.select([~valid, ~has_q, at, through, inside, better_than_opp],
                        ["no_valid_book", "no_quote_on_hit_side", "at_best", "through_best", "inside_spread",
                         "at_or_beyond_own_side"], "other")
        df = pd.DataFrame({"cl": w.cl.values, "cat": cat, "vol": w.vol.values})
        tab = df.groupby(["cl", "cat"]).size().unstack(fill_value=0)
        frac = tab.div(tab.sum(axis=1), axis=0)
        tot = df.cat.value_counts(normalize=True)
        volf = df.groupby("cat").vol.sum() / df.vol.sum()
        ok = valid & has_q
        res[lag_name] = {"by_contract_frac": frac.round(4).to_dict(orient="index"),
                         "all_frac": tot.round(4).to_dict(), "all_volume_frac": volf.round(4).to_dict(),
                         "at_or_through_given_quote": float((at | through)[ok].mean()),
                         "n": int(len(df))}
    out["at_through"] = res
    print("lag  " + "  ".join(f"{cl:>28s}" for cl in ("5m", "15m", "4h")) + "   (at / through / inside / own-side)")
    for k, v in res.items():
        print(f"{k:>16s} " + "  ".join(
            "/".join(f"{v['by_contract_frac'][cl].get(x, 0):.3f}" for x in ("at_best", "through_best", "inside_spread", "at_or_beyond_own_side"))
            for cl in ("5m", "15m", "4h")) + f"   at|through={v['at_or_through_given_quote']:.3f}")
    out["mirror"] = meta["mirror_stats"]
    out["splits"] = meta["splits"]["counts"]
    (HERE / "build" / "coverage_stats.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "at_through"}, indent=1, default=float))


if __name__ == "__main__":
    main()
