#!/usr/bin/env python3
"""Category census of maker economics from cache/legs.parquet (build_legs.py).

  python census.py [--days 7] [--boot 1000]

Writes out/: category_summary.csv, subcat_summary.csv, by_price.csv, by_ttr.csv, by_price_ttr.parquet,
crypto_validation.csv; cache/market_level.parquet (per-condition sums, used for the bootstrap).

Per filled share (cents), size-weighted:
  spread   = s·(mid_pre − q)                 s = +1 maker buy, −1 maker sell
  markout_h= s·(mid(t+h) − mid_pre)          negative = adverse selection
  rebate   = rebateRate·feeRate·q(1−q)
  reward   = Σ_cond pool$/day·active days / Σ_cond maker shares   (CLOB /rewards/markets/current, 2026-10-15)
  edge_h   = spread + markout_h + rebate + reward
Per $ of maker notional: the same in bps of Σ q·size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
H = ["1m", "5m", "30m", "2h", "1d"]
PB = [0, .05, .15, .35, .65, .85, .95, 1.0001]
TB = [-1e9, 1, 6, 24, 24 * 7, 24 * 30, 1e9]
TBL = ["<1h", "1-6h", "6-24h", "1-7d", "7-30d", ">30d"]

KEY = """CASE WHEN category = 'sports_game' THEN (CASE WHEN inplay THEN 'sports_inplay' ELSE 'sports_pregame' END)
              ELSE category END"""


def sums(group: str) -> str:
    mk = ",\n".join(
        f"sum(s*(mid_{h}-mid_pre)*size) FILTER (WHERE mid_{h} IS NOT NULL AND mid_pre IS NOT NULL) AS mk_{h}_d, "
        f"sum(size) FILTER (WHERE mid_{h} IS NOT NULL AND mid_pre IS NOT NULL) AS sh_{h}, "
        f"sum(q*size) FILTER (WHERE mid_{h} IS NOT NULL AND mid_pre IS NOT NULL) AS us_{h}" for h in H)
    return f"""
      SELECT {group}, count(*) AS n_fills, sum(size) AS shares, sum(q*size) AS usd,
        count(DISTINCT wallet) AS n_makers, count(DISTINCT cond) AS n_markets,
        sum(s*(mid_pre-q)*size) FILTER (WHERE mid_pre IS NOT NULL) AS spread_d,
        sum(size) FILTER (WHERE mid_pre IS NOT NULL) AS sh_pre,
        sum(q*size) FILTER (WHERE mid_pre IS NOT NULL) AS us_pre,
        {mk},
        sum(s*(final-mid_pre)*size) FILTER (WHERE final IS NOT NULL AND mid_pre IS NOT NULL) AS mk_res_d,
        sum(s*(final-q)*size) FILTER (WHERE final IS NOT NULL) AS tot_res_d,
        sum(size) FILTER (WHERE final IS NOT NULL AND mid_pre IS NOT NULL) AS sh_res,
        sum(q*size) FILTER (WHERE final IS NOT NULL AND mid_pre IS NOT NULL) AS us_res,
        count(DISTINCT cond) FILTER (WHERE final IS NOT NULL) AS n_mkt_res,
        sum(s*(mark_end-q)*size) AS pnl_end_d,
        sum(rebate_ps*size) AS rebate_d, sum(maker_fee) AS maker_fee_d,
        sum(reward_ps*size) AS reward_d
      FROM legs GROUP BY ALL"""


def finish(df: pd.DataFrame, days: float) -> pd.DataFrame:
    d = df.copy()
    d["usd_day"] = d.usd / days
    d["spread_c"] = 100 * d.spread_d / d.sh_pre
    d["rebate_c"] = 100 * d.rebate_d / d.shares
    d["reward_c"] = 100 * d.reward_d / d.shares
    for h in H + ["res"]:
        d[f"mk_{h}_c"] = 100 * d[f"mk_{h}_d"] / d[f"sh_{h}"]
        d[f"edge_{h}_c"] = d.spread_c + d[f"mk_{h}_c"] + d.rebate_c + d.reward_c
    # per $ notional (bps): spread and markout on their own covered subsets
    sp_bps = 1e4 * d.spread_d / d.us_pre
    rr_bps = 1e4 * (d.rebate_d + d.reward_d) / d.usd
    for h in H + ["res"]:
        d[f"edge_{h}_bps"] = sp_bps + 1e4 * d[f"mk_{h}_d"] / d[f"us_{h}"] + rr_bps
    d["spread_bps"], d["rebrew_bps"] = sp_bps, rr_bps
    d["avg_px"] = d.usd / d.shares
    return d


def boot(mkt: pd.DataFrame, key: str, nb: int, seed=0) -> pd.DataFrame:
    """Poisson bootstrap over events (gamma event slug; markets of one event share news/outcomes) of edge per $
    (bps) at each horizon, with and without liquidity rewards."""
    rng = np.random.default_rng(seed)
    rows = []
    cols = ["spread_d", "rebate_d", "reward_d", "usd"] + [f"{p}_{h}" for h in ["30m", "2h", "1d", "res"]
                                                          for p in ("mk", "us")] + [f"{p}_{h}_d" for h in
                                                          ["30m", "2h", "1d", "res"] for p in ("mk",)]
    for k, g0 in mkt.groupby(key):
        g0 = g0.copy()
        for h in ["30m", "2h", "1d", "res"]:
            # spread and rebates/rewards restricted to the legs covered at horizon h (pro rata by notional)
            cov = g0[f"us_{h}"].fillna(0) / g0.usd
            g0[f"num_{h}"] = g0.spread_d.fillna(0) * cov + g0[f"mk_{h}_d"].fillna(0) + g0.rebate_d * cov
            g0[f"rw_{h}"] = g0.reward_d * cov
            g0[f"den_{h}"] = g0[f"us_{h}"].fillna(0)
        g = g0.groupby("event").sum(numeric_only=True)
        w = rng.poisson(1.0, size=(nb, len(g)))
        r = {key: k, "n_events": len(g)}
        for h in ["30m", "2h", "1d", "res"]:
            den = g[f"den_{h}"].to_numpy()
            for tag, num in (("", g[f"num_{h}"] + g[f"rw_{h}"]), ("_xr", g[f"num_{h}"])):
                num = num.to_numpy()
                e = 1e4 * (w @ num) / np.maximum(w @ den, 1e-9)
                r[f"edge{tag}_{h}_bps_lo"], r[f"edge{tag}_{h}_bps_hi"] = np.nanpercentile(e, [2.5, 97.5])
                r[f"edge{tag}_{h}_bps_boot"] = 1e4 * num.sum() / max(den.sum(), 1e-9)
        rows.append(r)
    return pd.DataFrame(rows)


def concentration(con, key: str) -> pd.DataFrame:
    w = con.execute(f"SELECT {key} AS k, wallet, sum(q*size) usd FROM legs GROUP BY 1, 2").df()
    out = []
    for k, g in w.groupby("k"):
        sh = (g.usd / g.usd.sum()).sort_values(ascending=False).to_numpy()
        out.append({"k": k, "hhi": float((sh ** 2).sum() * 1e4), "top5_share": float(sh[:5].sum()),
                    "top20_share": float(sh[:20].sum()),
                    "makers_for_90pct": int(np.searchsorted(np.cumsum(sh), 0.9) + 1)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='9GB'")
    # reward pool per condition over the window, spread over the condition's maker shares
    rw = pd.read_parquet(HERE / "cache" / "rewards_current.parquet")
    tk = pd.read_parquet(HERE / "cache" / "taker.parquet")
    con.register("rw", rw)
    con.register("tk", tk)
    con.execute(f"""
      CREATE TABLE pool AS
      SELECT l.cond, coalesce(rw.rate_per_day, 0) * least(coalesce(tk.n_days, 1), {a.days}) AS pool_usd,
             sum(l.size) AS msh
      FROM 'cache/legs.parquet' l LEFT JOIN rw ON l.cond = rw.condition_id LEFT JOIN tk ON l.cond = tk.cond
      GROUP BY l.cond, rw.rate_per_day, tk.n_days""")
    con.execute(f"""
      CREATE TABLE legs AS
      SELECT l.*, {KEY} AS ckey, p.pool_usd / p.msh AS reward_ps,
             CASE WHEN q < .05 THEN 'a<.05' WHEN q < .15 THEN 'b.05-.15' WHEN q < .35 THEN 'c.15-.35'
                  WHEN q < .65 THEN 'd.35-.65' WHEN q < .85 THEN 'e.65-.85' WHEN q < .95 THEN 'f.85-.95'
                  ELSE 'g>.95' END AS pbin,
             CASE WHEN ttr_h IS NULL THEN 'unknown' WHEN ttr_h < 1 THEN 'a<1h' WHEN ttr_h < 6 THEN 'b1-6h'
                  WHEN ttr_h < 24 THEN 'c6-24h' WHEN ttr_h < 168 THEN 'd1-7d' WHEN ttr_h < 720 THEN 'e7-30d'
                  ELSE 'f>30d' END AS tbin
      FROM 'cache/legs.parquet' l JOIN pool p USING (cond)""")
    # taker side per category key: assign each condition's taker flow to its category (sports split by time
    # is approximated by the maker-share split of that condition)
    tak = con.execute(f"""
      WITH ms AS (SELECT cond, ckey, sum(q*size) u FROM legs GROUP BY 1, 2),
           mt AS (SELECT cond, sum(u) ut FROM ms GROUP BY 1)
      SELECT ms.ckey AS ckey, sum(tk.taker_usd * ms.u / mt.ut) AS taker_usd, sum(tk.taker_fee * ms.u / mt.ut) AS taker_fee,
             sum(tk.n_taker * ms.u / mt.ut) AS n_taker
      FROM ms JOIN mt USING (cond) JOIN tk USING (cond) GROUP BY 1""").df()
    pools = con.execute(f"""
      SELECT ckey, sum(reward_ps*size) AS reward_pool_usd, count(DISTINCT cond) FILTER (WHERE reward_ps > 0) AS n_mkt_rewarded
      FROM legs GROUP BY 1""").df()

    cat = finish(con.execute(sums("ckey")).df(), a.days)
    conc = concentration(con, "ckey").rename(columns={"k": "ckey"})
    mkt = con.execute(sums("ckey, cond")).df()
    ev = pd.read_parquet(HERE / "cache" / "markets.parquet", columns=["condition_id", "ev_slug"]).drop_duplicates(
        "condition_id")
    mkt = mkt.merge(ev.rename(columns={"condition_id": "cond", "ev_slug": "event"}), on="cond", how="left")
    mkt["event"] = mkt.event.fillna(mkt.cond)
    mkt.to_parquet(HERE / "cache" / "market_level.parquet", index=False)   # 36 MB: git-ignored
    b = boot(mkt, "ckey", a.boot)
    cat = cat.merge(tak, on="ckey", how="left").merge(pools, on="ckey", how="left").merge(conc, on="ckey").merge(b, on="ckey")
    cat["taker_usd_day"] = cat.taker_usd / a.days
    cat["taker_fee_day"] = cat.taker_fee / a.days
    cat["fills_day"] = cat.n_fills / a.days
    cat["maker_pnl_end_day"] = (cat.pnl_end_d + cat.rebate_d + cat.reward_d) / a.days
    cat["capacity_2h_usd_day"] = cat.edge_2h_bps / 1e4 * cat.usd_day
    cat["capacity_res_usd_day"] = cat.edge_res_bps / 1e4 * cat.usd_day
    cat["res_cov"] = cat.us_res / cat.usd
    cat["cov_1d"] = cat.us_1d / cat.usd
    for h in ["2h", "1d", "res"]:
        cat[f"xr_{h}_c"] = cat[f"edge_{h}_c"] - cat.reward_c
    cat = cat.sort_values("edge_2h_bps", ascending=False)
    keep = ["ckey", "n_markets", "n_mkt_res", "res_cov", "fills_day", "usd_day", "taker_usd_day", "taker_fee_day",
            "n_makers", "hhi", "top5_share", "top20_share", "makers_for_90pct", "avg_px",
            "spread_c", "mk_1m_c", "mk_5m_c", "mk_30m_c", "mk_2h_c", "mk_1d_c", "mk_res_c", "rebate_c", "reward_c",
            "edge_30m_c", "edge_2h_c", "edge_1d_c", "edge_res_c", "xr_2h_c", "xr_1d_c", "xr_res_c", "cov_1d", "spread_bps", "rebrew_bps",
            "edge_30m_bps", "edge_30m_bps_lo", "edge_30m_bps_hi", "edge_2h_bps", "edge_2h_bps_lo", "edge_2h_bps_hi",
            "edge_1d_bps", "edge_1d_bps_lo", "edge_1d_bps_hi",
            "edge_res_bps", "edge_res_bps_lo", "edge_res_bps_hi",
            "edge_xr_30m_bps_boot", "edge_xr_30m_bps_lo", "edge_xr_30m_bps_hi",
            "edge_xr_2h_bps_boot", "edge_xr_2h_bps_lo", "edge_xr_2h_bps_hi",
            "edge_xr_1d_bps_boot", "edge_xr_1d_bps_lo", "edge_xr_1d_bps_hi",
            "edge_xr_res_bps_boot", "edge_xr_res_bps_lo", "edge_xr_res_bps_hi", "n_events", "capacity_2h_usd_day", "capacity_res_usd_day",
            "maker_pnl_end_day", "reward_pool_usd", "n_mkt_rewarded"]
    cat[keep].to_csv(OUT / "category_summary.csv", index=False, float_format="%.4g")

    sub = finish(con.execute(sums("ckey, subcat")).df(), a.days)
    sub = sub[sub.usd_day > 1000]
    cols = ["ckey", "subcat", "n_markets", "usd_day", "n_makers", "spread_c", "mk_5m_c", "mk_2h_c", "mk_res_c",
            "rebate_c", "reward_c", "edge_2h_c", "edge_2h_bps", "edge_res_bps"]
    sub.sort_values(["ckey", "usd_day"], ascending=[True, False])[cols].to_csv(
        OUT / "subcat_summary.csv", index=False, float_format="%.4g")
    for g, name in [("pbin", "by_price"), ("tbin", "by_ttr")]:
        x = finish(con.execute(sums(f"ckey, {g}")).df(), a.days)
        x.sort_values(["ckey", g])[["ckey", g, "n_markets", "usd_day", "spread_c", "mk_5m_c", "mk_2h_c",
                                     "mk_1d_c", "mk_res_c", "rebate_c", "reward_c", "edge_2h_c", "edge_1d_c", "edge_2h_bps",
                                     "edge_1d_bps", "edge_res_bps"]].to_csv(OUT / f"{name}.csv", index=False, float_format="%.4g")
    x = finish(con.execute(sums("ckey, pbin, tbin")).df(), a.days)
    x[["ckey", "pbin", "tbin", "n_markets", "usd_day", "spread_c", "mk_2h_c", "mk_res_c", "rebate_c",
       "reward_c", "edge_2h_c"]].to_parquet(OUT / "by_price_ttr.parquet", index=False)

    # validation: the crypto up/down study's finding inside the census
    v = finish(con.execute(sums("ckey, subcat")).df(), a.days)
    v = v[v.ckey == "crypto_updown_short"]
    v[["subcat", "n_markets", "usd_day", "n_makers", "spread_c", "mk_1m_c", "mk_5m_c", "mk_30m_c", "mk_res_c",
       "rebate_c", "reward_c", "edge_5m_c", "edge_res_c", "edge_res_bps"]].to_csv(
        OUT / "crypto_validation.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(cat[["ckey", "n_markets", "usd_day", "n_makers", "top5_share", "spread_c", "mk_5m_c", "mk_2h_c",
               "mk_res_c", "rebate_c", "reward_c", "edge_2h_bps", "edge_2h_bps_lo", "edge_2h_bps_hi",
               "edge_res_bps", "capacity_2h_usd_day"]].round(3).to_string())


if __name__ == "__main__":
    main()
