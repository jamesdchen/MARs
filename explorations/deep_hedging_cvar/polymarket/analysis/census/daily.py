#!/usr/bin/env python3
"""Day-by-day stability of the category edge (ex rewards, per $ maker notional, bps) -> out/daily_edge.csv."""
import duckdb
import pandas as pd

from period import LEGS, OUT

KEY = """CASE WHEN category = 'sports_game' THEN (CASE WHEN inplay THEN 'sports_inplay' ELSE 'sports_pregame' END)
              ELSE category END"""
f = lambda h: f"WHERE mid_{h} IS NOT NULL AND mid_pre IS NOT NULL"
d = duckdb.sql(f"""
  SELECT {KEY} ckey, strftime(to_timestamp(t), '%m-%d') AS day, sum(q*size) usd,
    1e4*(sum(s*(mid_30m-q)*size) FILTER ({f('30m')}) + sum(rebate_ps*size) FILTER ({f('30m')}))
       / sum(q*size) FILTER ({f('30m')}) xr_30m_bps,
    1e4*(sum(s*(mid_2h-q)*size) FILTER ({f('2h')}) + sum(rebate_ps*size) FILTER ({f('2h')}))
       / sum(q*size) FILTER ({f('2h')}) xr_2h_bps,
    1e4*(sum(s*(final-q)*size) FILTER (WHERE final IS NOT NULL) + sum(rebate_ps*size) FILTER (WHERE final IS NOT NULL))
       / sum(q*size) FILTER (WHERE final IS NOT NULL) xr_res_bps
  FROM {LEGS} GROUP BY 1, 2 ORDER BY 1, 2""").df()
d.to_csv(OUT / "daily_edge.csv", index=False, float_format="%.4g")
p = d.pivot(index="ckey", columns="day", values="xr_2h_bps").round(0)
p["pos_days"] = (p > 0).sum(axis=1)
pd.set_option("display.width", 250)
print("ex-reward edge at +2h, bps of maker notional, by day\n", p.to_string())
q = d.pivot(index="ckey", columns="day", values="xr_res_bps").round(0)
q["pos_days"] = (q > 0).sum(axis=1)
print("ex-reward edge to resolution (resolved legs)\n", q.to_string())
