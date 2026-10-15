"""News-jump and resolution risk for makers in the politics universe.

1. Jumpiness of CLOB prices-history (outcome-0 frame, forward-filled onto a 10-min grid, Sep 15 -> Oct 15):
   share of 10-min moves above 2c / 5c / 10c, jumps per market-day, share of the total absolute variation
   that comes from the largest 1% of 10-min moves.
2. Maker P&L concentration: maker fills whose next 2 h contain a >= 5c 10-min move (a "jump window"); their
   share of fills and of the 2 h markout P&L (size-weighted, $).
3. Resolution: for markets that resolved inside the window, maker markouts of fills in the last 24 h / 1 h
   before the end.
  -> out/risk_jumps_by_category.csv, out/risk_maker_jump.csv, out/risk_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
U = {r["slug"]: r for r in json.load(open(OUT / "universe.json"))}
ph = pd.read_parquet(HERE / "data/prices_history.parquet")
tok0 = {r["tokens"][0]: r["slug"] for r in U.values()}
ph = ph[ph.token_id.isin(tok0)].assign(slug=lambda d: d.token_id.map(tok0))
t_lo = pd.Timestamp("2026-09-15", tz="UTC").timestamp()

rows, jumps = [], {}
for slug, g in ph.groupby("slug"):
    g = g[g.t >= t_lo - 86400].sort_values("t")
    if len(g) < 10:
        continue
    s = pd.Series(g.p.values, index=pd.to_datetime(g.t, unit="s", utc=True))
    s = s[~s.index.duplicated(keep="last")].resample("10min").last().ffill()
    s = s[s.index >= pd.Timestamp(t_lo, unit="s", tz="UTC")]
    d = s.diff().dropna().abs() * 100
    if len(d) < 10:
        continue
    q = d.quantile(0.99)
    days = len(d) / 144
    rows.append(dict(slug=slug, category=U[slug]["category"], days=days, mean_abs_10m_c=d.mean(),
                     p99_abs_10m_c=q, share_gt2c=(d > 2).mean(), share_gt5c=(d > 5).mean(), share_gt10c=(d > 10).mean(),
                     jumps5_per_day=(d > 5).sum() / days, jumps10_per_day=(d > 10).sum() / days,
                     top1pct_share_of_variation=d[d >= q].sum() / d.sum() if d.sum() > 0 else np.nan,
                     max_abs_10m_c=d.max()))
    jumps[slug] = np.array([ts.timestamp() for ts in d[d >= 5].index])
J = pd.DataFrame(rows).set_index("slug")
J.to_csv(OUT / "risk_jumps_market.csv")
JC = J.groupby("category").agg(markets=("days", "size"), market_days=("days", "sum"), mean_abs_10m_c=("mean_abs_10m_c", "median"),
                               p99_abs_10m_c=("p99_abs_10m_c", "median"), share_gt5c=("share_gt5c", "mean"),
                               jumps5_per_day=("jumps5_per_day", "mean"), jumps10_per_day=("jumps10_per_day", "mean"),
                               top1pct_share_of_variation=("top1pct_share_of_variation", "median"), max_abs_10m_c=("max_abs_10m_c", "max"))
JC.to_csv(OUT / "risk_jumps_by_category.csv")

M = pd.read_parquet(HERE / "data/maker_fills_politics.parquet")
M["jump2h"] = False
for slug, idx in M.groupby("slug").indices.items():
    jt = jumps.get(slug, np.array([]))
    if not len(jt):
        continue
    t = M.block_time.values[idx].astype(float)
    # a 10-min bar stamped T covers (T-10min, T]: jump inside [t, t+2h]
    i = np.searchsorted(jt, t, side="left")
    nxt = np.where(i < len(jt), jt[np.clip(i, 0, len(jt) - 1)], np.inf)
    M.loc[M.index[idx], "jump2h"] = nxt <= t + 7200 + 600
M["mo2h_usd"] = M.mo_2h_c * M["size"] / 100
M["edge_usd"] = M.edge_c * M["size"] / 100
mj = M.groupby(["category", "jump2h"]).agg(fills=("size", "size"), shares=("size", "sum"), edge_usd=("edge_usd", "sum"),
                                           mo2h_usd=("mo2h_usd", "sum")).reset_index()
mj["mo2h_c_per_share"] = 100 * mj.mo2h_usd / mj.shares
mj["edge_c_per_share"] = 100 * mj.edge_usd / mj.shares
mj.to_csv(OUT / "risk_maker_jump.csv", index=False)

# resolution: fills close to the end of markets that resolved
R = M[np.isfinite(M.pay0)].copy()
R["h_to_end"] = (R.end - R.block_time) / 3600
R["bucket"] = pd.cut(R.h_to_end, [-1e9, 0, 1, 24, 24 * 7, 1e9], labels=["after end", "<1h", "1-24h", "1-7d", ">7d"])
R["mo_res_usd"] = R.mo_res_c * R["size"] / 100
rr = R.groupby("bucket", observed=True).agg(fills=("size", "size"), shares=("size", "sum"), edge_usd=("edge_usd", "sum"),
                                            mo2h_usd=("mo2h_usd", "sum"), mo_res_usd=("mo_res_usd", "sum"))
rr["mo_res_c_per_share"] = 100 * rr.mo_res_usd / rr.shares
rr["edge_c_per_share"] = 100 * rr.edge_usd / rr.shares
rr.to_csv(OUT / "risk_resolution.csv")
with pd.option_context("display.width", 250, "display.float_format", "{:.3g}".format):
    print(JC); print(mj); print(rr)
json.dump(dict(jumps_by_category=JC.to_dict(orient="index"), maker_jump=mj.to_dict(orient="records"),
               resolution=rr.reset_index().astype({"bucket": str}).to_dict(orient="records")),
          open(OUT / "risk_summary.json", "w"), indent=1, default=float)
