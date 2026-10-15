#!/usr/bin/env python3
"""Markdown tables for REPORT.md §6 (replication): Oct 8-14 (out/) vs Sep 17-Oct 7 (out_oos/) vs pooled
(out_pooled/), and the sports league x time-to-kickoff table. Writes out_pooled/replication.md."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P = {"Oct 8–14": "out", "Sep 17–Oct 7 (OOS)": "out_oos", "pooled Sep 17–Oct 14": "out_pooled"}


def ci(d, k):
    return d[f"{k}_boot" if f"{k}_boot" in d else k].round(0).astype(int).astype(str) + " [" + \
        d[f"{k}_lo".replace("_bps_lo", "_bps_lo")].round(0).astype(int).astype(str) + ", " + \
        d[f"{k}_hi"].round(0).astype(int).astype(str) + "]"


def f(d, base):
    return d[base].round(0).astype(int).astype(str) + " [" + d[base + "_lo"].round(0).astype(int).astype(str) + \
        ", " + d[base + "_hi"].round(0).astype(int).astype(str) + "]"


out = []
cats = None
rows = []
for name, o in P.items():
    d = pd.read_csv(HERE / o / "category_summary.csv").set_index("ckey").drop(index="other", errors="ignore")
    d = d.rename(columns={f"edge_xr_{h}_bps_boot": f"xr_{h}" for h in ["30m", "2h", "1d", "res"]})
    for h in ["30m", "2h", "1d", "res"]:
        d[f"xr_{h}_lo"], d[f"xr_{h}_hi"] = d[f"edge_xr_{h}_bps_lo"], d[f"edge_xr_{h}_bps_hi"]
    t = pd.DataFrame({"period": name, "maker $/day": (d.usd_day / 1e6).round(2).astype(str) + "M",
                      "spread ¢": d.spread_c.round(2), "AS 30m ¢": (-d.mk_30m_c).round(2),
                      "ex-rew +30m": f(d, "xr_30m"), "ex-rew +2h": f(d, "xr_2h"), "ex-rew +1d": f(d, "xr_1d"),
                      "ex-rew res": f(d, "xr_res"), "with rew +2h": f(d, "edge_2h_bps"),
                      "with rew +1d": f(d, "edge_1d_bps"), "rew bps": (d.reward_c / d.avg_px * 100).round(0)},
                     index=d.index)
    rows.append(t)
r = pd.concat(rows).reset_index().rename(columns={"ckey": "category"})
order = ["sports_pregame", "elections", "sports_futures", "geopolitics", "finance_prices", "crypto_updown_short",
         "crypto_other", "politics", "sports_inplay", "weather", "economics", "tech_ai", "culture"]
r["o"] = r.category.map({c: i for i, c in enumerate(order)})
r = r.sort_values(["o", "period"], key=lambda s: s if s.name == "o" else s.map(
    {k: i for i, k in enumerate(P)})).drop(columns="o")
out.append("### Edge per $ of maker notional (bps), event-bootstrap 95% CI, by period\n")
out.append(r.to_markdown(index=False))

for name, o in P.items():
    dd = pd.read_csv(HERE / o / "daily_edge.csv")
    pos = dd.assign(p=dd.xr_2h_bps > 0).groupby("ckey").p.agg(["sum", "count"])
    out.append(f"\n{name}: days with ex-reward +2h edge > 0: " +
               ", ".join(f"{k} {int(v['sum'])}/{int(v['count'])}" for k, v in pos.iterrows() if k != "other"))

for o, nm in [("out_pooled", "pooled"), ("out_oos", "OOS")]:
    lk = pd.read_csv(HERE / o / "sports_league_kickoff.csv")
    lk = lk[(lk.ckey == "sports_pregame") & (lk.usd_day > 20000)]
    lk = lk[["subcat", "to_game", "n_markets", "usd_day", "n_makers", "spread_c", "mk_5m_c", "mk_2h_c",
             "rebate_c", "xr_2h_bps", "xr_res_bps"]].round(2)
    lk["usd_day"] = (lk.usd_day / 1e3).round(0).astype(int).astype(str) + "k"
    out.append(f"\n### Pre-game sports by league × time to kickoff ({nm}; ¢/share, bps ex rewards)\n")
    out.append(lk.sort_values(["subcat", "to_game"]).to_markdown(index=False))
    lp = pd.read_csv(HERE / o / "sports_league_kickoff.csv")
    lp = lp[(lp.ckey == "sports_inplay") & (lp.usd_day > 20000)][["subcat", "n_markets", "usd_day", "spread_c",
                                                                    "mk_2h_c", "xr_2h_bps", "xr_res_bps"]].round(2)
    out.append(f"\n### In-play sports by league ({nm})\n")
    out.append(lp.sort_values("usd_day", ascending=False).to_markdown(index=False))

for name, o in P.items():
    tv = pd.read_csv(HERE / o / "top_vs_rest.csv")
    tv = tv[tv.ckey.isin(["sports_pregame", "elections", "crypto_updown_short", "sports_inplay"])]
    out.append(f"\n### top-20 vs rest ({name})\n")
    out.append(tv[["ckey", "grp", "usd_day", "spread_c", "mk_5m_c", "mk_2h_c", "mk_res_c", "rebate_c", "xr_2h_bps",
                   "xr_res_bps"]].round(2).to_markdown(index=False))
    tm = pd.read_csv(HERE / o / "top_makers_by_cat.csv")
    out.append(f"\ntop-20 P&L bps ({name}): " + ", ".join(
        f"{a} {b:.0f} (rew {c:.0f})" for a, b, c in zip(tm.ckey, tm.pnl_bps, tm.reward_bps) if a != "other"))
    rc = pd.read_csv(HERE / o / "rewards_check.csv")
    out.append(f"\nrewards check ({name}): paid {rc.paid_total[0]:.0f} vs proxy {rc.model_total[0]:.0f} "
               f"(ratio {rc.paid_total[0] / rc.model_total[0]:.2f}, corr {rc['corr'][0]:.2f}, {rc.wallets[0]} wallets)")
(HERE / "out_pooled" / "replication.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
