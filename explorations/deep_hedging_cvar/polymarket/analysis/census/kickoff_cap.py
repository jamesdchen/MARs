#!/usr/bin/env python3
"""Kickoff-capped marks for pre-game sports legs (w2 RECONCILE.md, coordinator 22:25).

The census marks a pre-game leg at mid(t+h) even when t+h is past kickoff, so legs near kickoff carry in-game
moves. Here mark_cap_h = mid(min(t+h, kickoff−)), where mid(kickoff−) is the micro-price after the last taker
print strictly before gameStartTime (same micro-price as build_legs.py). Also "to kickoff" = mid(kickoff−).

  python kickoff_cap.py        # -> cache/kick_w*.parquet, out*/sports_kickoff_capped.csv
"""
import duckdb
import numpy as np
import pandas as pd

from period import HERE

F = HERE.parent.parent / "onchain" / "data" / "all" / "fills"
WEEKS = {"w1": ("2026-09-17", "2026-09-23"), "w2": ("2026-09-24", "2026-09-30"),
         "w3": ("2026-10-01", "2026-10-07"), "w4": ("2026-10-08", "2026-10-14")}
KO = """CASE WHEN to_game_h < 0.5 THEN 'a<30m' WHEN to_game_h < 1 THEN 'b30-60m' WHEN to_game_h < 6 THEN 'c1-6h'
             WHEN to_game_h < 24 THEN 'd6-24h' ELSE 'e>24h' END"""
PERIODS = {"out": ["w4"], "out_oos": ["w1", "w2", "w3"], "out_pooled": ["w1", "w2", "w3", "w4"]}


def build(con, w, lo, hi):
    t_lo = int(pd.Timestamp(lo, tz="UTC").timestamp())
    t_hi = int((pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=2)).timestamp())   # games up to 1 d after the week
    con.execute(f"""
      CREATE OR REPLACE TABLE tp AS
      SELECT k.condition_id cond, f.block, f.log_index li, f.block_time t,
             CASE WHEN k.outcome_idx = 0 THEN f.price ELSE 1 - f.price END p0,
             CASE WHEN (f.side = 'BUY') = (k.outcome_idx = 0) THEN 1 ELSE -1 END dir0
      FROM read_parquet('{F}/*/*.parquet', hive_partitioning=1) f JOIN mk k USING (token_id)
      WHERE f.role = 'taker' AND f.block_time >= {t_lo} AND f.block_time < {t_hi} AND f.size > 0
        AND k.category = 'sports_game'""")
    con.execute("""
      CREATE OR REPLACE TABLE kick AS
      WITH x AS (SELECT cond, block, li, t, p0,
          last_value(CASE WHEN dir0 = 1 THEN p0 END IGNORE NULLS) OVER w lb,
          last_value(CASE WHEN dir0 = -1 THEN p0 END IGNORE NULLS) OVER w ls
        FROM tp WINDOW w AS (PARTITION BY cond ORDER BY block, li ROWS UNBOUNDED PRECEDING)),
      m AS (SELECT cond, block, li, t,
              CASE WHEN lb IS NOT NULL AND ls IS NOT NULL AND abs(lb - ls) <= 0.10 THEN (lb + ls) / 2 ELSE p0 END mid
            FROM x),
      g AS (SELECT DISTINCT condition_id cond, game_ts FROM mk WHERE game_ts IS NOT NULL)
      SELECT g.cond, g.game_ts, arg_max(m.mid, m.block * 10000 + m.li) mid_kick0, max(m.t) last_t
      FROM m JOIN g ON m.cond = g.cond AND m.t < g.game_ts GROUP BY 1, 2""")
    con.execute(f"COPY kick TO '{HERE}/cache/kick_{w}.parquet' (FORMAT parquet)")


def cells(con, tags):
    files = ", ".join(f"'{HERE}/cache/legs_{t}.parquet'" for t in tags)
    kicks = ", ".join(f"'{HERE}/cache/kick_{t}.parquet'" for t in tags)
    own = lambda c: f"CASE WHEN l.oi = 0 THEN {c} ELSE 1 - {c} END"
    con.execute(f"""
      CREATE OR REPLACE TABLE pl AS
      SELECT l.cond, l.subcat, l.q, l.size, l.s, l.rebate_ps, l.mid_pre, l.mid_30m, l.mid_2h, l.to_game_h,
             {own('k.mid_kick0')} AS mid_kick,
             CASE WHEN l.to_game_h * 3600 > 1800 THEN l.mid_30m ELSE {own('k.mid_kick0')} END AS cap_30m,
             CASE WHEN l.to_game_h * 3600 > 7200 THEN l.mid_2h ELSE {own('k.mid_kick0')} END AS cap_2h
      FROM read_parquet([{files}]) l
      JOIN (SELECT DISTINCT ON (cond) * FROM read_parquet([{kicks}])) k ON l.cond = k.cond
      WHERE l.category = 'sports_game' AND NOT l.inplay AND l.mid_pre IS NOT NULL""")
    g = lambda c: f"{c} IS NOT NULL"
    cols = ",\n".join(f"sum(s*({c}-q)*size + rebate_ps*size) FILTER (WHERE {g(c)}) num_{c}, "
                      f"sum(q*size) FILTER (WHERE {g(c)}) den_{c}" for c in ["mid_30m", "cap_30m", "cap_2h", "mid_kick"])
    out = []
    for name, cell in [("all", "'pregame'"), ("kickoff", KO), ("league", "subcat"),
                       ("league x kickoff", f"subcat || ' ' || {KO}")]:
        d = con.execute(f"SELECT {cell} cell, cond, sum(q*size) usd, {cols} FROM pl GROUP BY ALL").df()
        d = d.merge(EV, on="cond", how="left")
        d["event"] = d.event.fillna(d.cond)
        for c, gr in d.groupby("cell"):
            e = gr.groupby("event").sum(numeric_only=True)
            w = RNG.poisson(1.0, (1000, len(e)))
            r = {"cut": name, "cell": c, "n_events": len(e), "usd": e.usd.sum()}
            for m in ["mid_30m", "cap_30m", "cap_2h", "mid_kick"]:
                num, den = e[f"num_{m}"].fillna(0).to_numpy(), e[f"den_{m}"].fillna(0).to_numpy()
                b = 1e4 * (w @ num) / np.maximum(w @ den, 1e-9)
                r[m] = 1e4 * num.sum() / max(den.sum(), 1e-9)
                r[m + "_lo"], r[m + "_hi"] = np.nanpercentile(b, [2.5, 97.5])
            out.append(r)
    return pd.DataFrame(out)


con = duckdb.connect()
con.execute("SET threads=4; SET memory_limit='10GB'")
con.execute(f"CREATE TABLE mk AS SELECT * FROM '{HERE}/cache/markets_cat.parquet'")
EV = pd.read_parquet(HERE / "cache" / "markets.parquet", columns=["condition_id", "ev_slug"]).drop_duplicates(
    "condition_id").rename(columns={"condition_id": "cond", "ev_slug": "event"})
RNG = np.random.default_rng(3)
for w, (lo, hi) in WEEKS.items():
    if not (HERE / "cache" / f"kick_{w}.parquet").exists():
        build(con, w, lo, hi)
        print("kick", w, flush=True)
rows = []
for o, tags in PERIODS.items():
    r = cells(con, tags)
    r["usd_day"] = r.usd / (7 * len(tags))
    r.to_csv(HERE / o / "sports_kickoff_capped.csv", index=False, float_format="%.4g")
    rows.append(r.assign(period=o))
for t in WEEKS:
    rows.append(cells(con, [t]).query("cut in ['all', 'kickoff']").assign(period=t))
x = pd.concat(rows)
x.to_csv(HERE / "out_pooled" / "sports_kickoff_capped_all_periods.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
print(x[x.cut.isin(["all", "kickoff"])][["period", "cell", "n_events", "usd_day", "mid_30m", "mid_30m_lo", "mid_30m_hi",
                                        "cap_30m", "cap_30m_lo", "cap_30m_hi", "cap_2h", "cap_2h_lo", "cap_2h_hi",
                                        "mid_kick", "mid_kick_lo", "mid_kick_hi"]].round(0).to_string(index=False))
