#!/usr/bin/env python3
"""Geopolitics: is the pooled +47 bps (ex rewards, +30m) a regime or noise? Per-week edges with event-bootstrap
CIs, event concentration, leave-top-k-events-out, and price-band split. -> out_pooled/geo_note.csv (+ print)."""
import duckdb
import numpy as np
import pandas as pd

from period import HERE

f = lambda h: f"mid_{h} IS NOT NULL AND mid_pre IS NOT NULL"
ev = pd.read_parquet(HERE / "cache" / "markets.parquet", columns=["condition_id", "ev_slug"]).drop_duplicates(
    "condition_id").rename(columns={"condition_id": "cond", "ev_slug": "event"})
rows, rng = [], np.random.default_rng(0)
allw = []
for w in ["w1", "w2", "w3", "w4"]:
    d = duckdb.sql(f"""
      SELECT cond, CASE WHEN q < .15 THEN 'p<.15' WHEN q < .85 THEN 'p.15-.85' ELSE 'p>.85' END pb,
        sum(q*size) usd, sum(size) sh,
        sum(s*(mid_pre-q)*size) FILTER (WHERE mid_pre IS NOT NULL) spread_d,
        sum(s*(mid_30m-q)*size + rebate_ps*size) FILTER (WHERE {f('30m')}) num30,
        sum(q*size) FILTER (WHERE {f('30m')}) den30,
        sum(s*(mid_2h-q)*size + rebate_ps*size) FILTER (WHERE {f('2h')}) num2h,
        sum(q*size) FILTER (WHERE {f('2h')}) den2h
      FROM read_parquet('{HERE}/cache/legs_{w}.parquet') WHERE category = 'geopolitics' GROUP BY 1, 2""").df()
    d = d.merge(ev, on="cond", how="left")
    d["event"] = d.event.fillna(d.cond)
    d["week"] = w
    allw.append(d)
d = pd.concat(allw)


def est(g, nb=1000):
    e = g.groupby("event")[["num30", "den30", "num2h", "den2h"]].sum().fillna(0)
    out = {"events": len(e), "usd_day": g.usd.sum()}
    for h in ["30", "2h"]:
        num, den = e[f"num{h}"].to_numpy(), e[f"den{h}"].to_numpy()
        w = rng.poisson(1, (nb, len(e)))
        b = 1e4 * (w @ num) / np.maximum(w @ den, 1e-9)
        out[f"xr_{h}"] = 1e4 * num.sum() / den.sum()
        out[f"xr_{h}_lo"], out[f"xr_{h}_hi"] = np.percentile(b, [2.5, 97.5])
    # concentration: share of the edge $ from the top-5 events, and the edge without the top 5 / 10 events
    contrib = e.num30.sort_values(ascending=False)
    out["top5_events_share_of_edge$"] = contrib.head(5).sum() / contrib.sum() if contrib.sum() else np.nan
    for k in (5, 10):
        rest = e.drop(contrib.head(k).index)
        out[f"xr_30_ex_top{k}"] = 1e4 * rest.num30.sum() / rest.den30.sum()
    out["top_event"] = contrib.index[0]
    return out


for w, g in d.groupby("week"):
    rows.append({"cut": "week", "key": w, **est(g)})
for pb, g in d.groupby("pb"):
    rows.append({"cut": "price (pooled)", "key": pb, **est(g)})
for (w, pb), g in d.groupby(["week", "pb"]):
    rows.append({"cut": "week x price", "key": f"{w} {pb}", **est(g, 300)})
rows.append({"cut": "pooled", "key": "all", **est(d)})
r = pd.DataFrame(rows)
r["usd_day"] = r.usd_day / np.where(r.cut.isin(["week", "week x price"]), 7, 28)
r.to_csv(HERE / "out_pooled" / "geo_note.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(r.round(1).to_string())
