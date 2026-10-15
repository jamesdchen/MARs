#!/usr/bin/env python3
"""Pre-game sports taker events from on-chain fills, for book-free quoting-rule backtests (rules.py).

  python build_events.py --fills ../../onchain/data/all/fills --start 2026-09-24 --end 2026-10-07

Inputs: fills (onchain/extract.py --all --events OrderFilled --slim), cache/markets.parquet (gamma_tags.py,
a copy of analysis/census/gamma_tags.py writing to this folder's cache). Only markets with a gameStartTime
(categories.py 'sports_game') and prints strictly before the game start are kept.

Frame: outcome 0 of each condition (p0 = price of outcome 0; a maker leg on outcome 1 at q is a frame-0
leg at 1 - q with the opposite side), as in analysis/census/build_legs.py. The micro-price is the census
one: mean of the latest taker-buy and taker-sell prints (latest print if they differ by > 10c), taken at the
last block strictly before the event's block.

One row per taker event = (tx, condition, taker direction in frame 0), from its maker legs:
  t, block, cond, ev (gamma event slug), league, game_ts, ttk_s (seconds to kickoff), dir (+1 taker buys
  outcome 0, i.e. lifts asks), m (micro-price before), pre_age, vol (shares), ext (worst frame-0 price
  reached: max leg price for buys, min for sells), vol_ge_<k> / vol_gt_<k> (shares traded at or beyond /
  strictly beyond the price m + dir * k cents rounded outward to the 1c grid), mid_60/300/1800 (micro-price
  at t + h, capped at kickoff: the last mid before game_ts), mid_kick (last mid before game_ts),
  fee_rate, rebate_rate.
Horizons whose time passes the window end are NULL (no look past --end).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

from categories import market_category

HERE = Path(__file__).resolve().parent
DIST_C = (0, 0.5, 1, 1.5, 2, 3, 5)          # quote distance from the micro-price, cents
HORIZONS = (60, 300, 1800)


def key(k):
    return str(k).replace('.', 'p')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", required=True)
    ap.add_argument("--start", default="2026-09-24")
    ap.add_argument("--end", default="2026-10-07", help="inclusive")
    ap.add_argument("--out", default=str(HERE / "cache" / "events.parquet"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t_lo = int(pd.Timestamp(a.start, tz="UTC").timestamp())
    t_hi = int((pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    t0 = time.time()

    m = pd.read_parquet(HERE / "cache" / "markets.parquet")
    m = m[m.slug.notna()].copy()
    gs = pd.to_datetime(m.game_start, utc=True, errors="coerce")
    es = pd.to_datetime(m.event_start_time, utc=True, errors="coerce")
    m["game_ts"] = (gs.fillna(es) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    cats = [market_category(s, t, f, g == g) for s, t, f, g in zip(m.slug, m.tags, m.fee_type, m.game_ts)]
    m["category"] = [c[0] for c in cats]
    m["league"] = [c[1] for c in cats]
    m = m[(m.category == "sports_game") & m.game_ts.notna()]
    m["fee_rate"] = m.fee_rate.fillna(0.0).where(m.fees_enabled.fillna(False).astype(bool), 0.0)
    m["rebate_rate"] = m.rebate_rate.fillna(0.0)
    mk = m[["token_id", "condition_id", "outcome_idx", "league", "ev_slug", "slug", "sports_type", "game_ts",
            "fee_rate", "rebate_rate"]].copy()
    print("sports_game tokens", len(mk), "conditions", mk.condition_id.nunique(), flush=True)

    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='12GB'; SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{HERE / 'cache' / 'duck_tmp'}'")
    con.register("mk_df", mk)
    con.execute("CREATE TABLE mk AS SELECT * FROM mk_df")
    con.execute(f"""
      CREATE TABLE f AS
      SELECT f.block, f.block_time AS t, f.log_index AS li, f.tx_hash AS tx, f.role, f.side, f.size,
             CASE WHEN mk.outcome_idx = 0 THEN f.price ELSE 1 - f.price END AS p0,
             CASE WHEN (f.side = 'BUY') = (mk.outcome_idx = 0) THEN 1 ELSE -1 END AS s0,
             mk.condition_id AS cond
      FROM read_parquet('{a.fills}/*/*.parquet', hive_partitioning=1) f JOIN mk USING (token_id)
      WHERE f.block_time >= {t_lo} AND f.block_time < {t_hi} AND f.size > 0""")
    print("fills", con.execute("select count(*) from f").fetchone(), f"{time.time()-t0:.0f}s", flush=True)
    # micro-price series from taker prints (dir0 = s0 of the taker row)
    con.execute("""
      CREATE TABLE mid AS
      WITH x AS (
        SELECT cond, block, li, t, p0,
          last_value(CASE WHEN s0 = 1 THEN p0 END IGNORE NULLS) OVER w AS lb,
          last_value(CASE WHEN s0 = -1 THEN p0 END IGNORE NULLS) OVER w AS ls
        FROM f WHERE role = 'taker'
        WINDOW w AS (PARTITION BY cond ORDER BY block, li ROWS UNBOUNDED PRECEDING))
      SELECT cond, block, li, t,
             CASE WHEN lb IS NOT NULL AND ls IS NOT NULL AND abs(lb - ls) <= 0.10 THEN (lb + ls) / 2 ELSE p0 END AS mid
      FROM x""")
    con.execute("CREATE TABLE midb AS SELECT cond, block, t, arg_max(mid, li) AS mid FROM mid GROUP BY cond, block, t")
    con.execute("DROP TABLE mid")
    # taker events from maker legs; taker direction in frame 0 = -(maker s0)
    con.execute("""
      CREATE TABLE ev AS
      SELECT tx, cond, -s0 AS dir, min(block) AS block, min(t) AS t, sum(size) AS vol,
             CASE WHEN -s0 = 1 THEN max(p0) ELSE min(p0) END AS ext,
             list(p0) AS ps, list(size) AS ss
      FROM f WHERE role = 'maker' GROUP BY tx, cond, -s0""")
    print("events", con.execute("select count(*) from ev").fetchone(), f"{time.time()-t0:.0f}s", flush=True)
    con.execute("""
      CREATE TABLE ev2 AS
      SELECT e.*, b.mid AS m, e.t - b.t AS pre_age
      FROM ev e ASOF LEFT JOIN midb b ON e.cond = b.cond AND e.block > b.block""")
    con.execute("DROP TABLE ev")
    con.execute("""
      CREATE TABLE kick AS
      SELECT k.cond, k.game_ts, b.mid AS mid_kick, b.t AS kick_mid_t
      FROM (SELECT DISTINCT condition_id AS cond, game_ts FROM mk) k
      ASOF LEFT JOIN midb b ON k.cond = b.cond AND k.game_ts > b.t""")
    cur = "ev2"
    for h in HORIZONS:
        con.execute(f"""
          CREATE TABLE ev_{h} AS
          SELECT x.*, b.mid AS mid_{h}r
          FROM {cur} x ASOF LEFT JOIN midb b ON x.cond = b.cond AND x.t + {h} >= b.t""")
        con.execute(f"DROP TABLE {cur}")
        cur = f"ev_{h}"
    vols = []
    for k in DIST_C:
        lvl_buy = f"ceil((x.m + {k / 100}) * 100 - 1e-9) / 100"    # our ask for a taker buy
        lvl_sell = f"floor((x.m - {k / 100}) * 100 + 1e-9) / 100"  # our bid for a taker sell
        lvl = f"(CASE WHEN x.dir = 1 THEN {lvl_buy} ELSE {lvl_sell} END)"
        vols.append(f"{lvl} AS lvl_{key(k)}")
        vols.append(f"list_sum(list_transform(list_zip(x.ps, x.ss), z -> CASE WHEN x.dir * (z[1] - {lvl}) >= -1e-9 "
                    f"THEN z[2] ELSE 0 END)) AS vol_ge_{key(k)}")
        vols.append(f"list_sum(list_transform(list_zip(x.ps, x.ss), z -> CASE WHEN x.dir * (z[1] - {lvl}) > 1e-9 "
                    f"THEN z[2] ELSE 0 END)) AS vol_gt_{key(k)}")
    hz = ",\n".join(f"CASE WHEN x.t + {h} >= k.game_ts THEN k.mid_kick WHEN x.t + {h} < {t_hi} THEN x.mid_{h}r END "
                    f"AS mid_{h}" for h in HORIZONS)
    con.execute(f"""
      COPY (
        SELECT x.t, x.block, x.tx, x.cond, x.dir, x.vol, x.ext, x.m, x.pre_age,
               {', '.join(vols)},
               {hz},
               CASE WHEN k.game_ts < {t_hi} THEN k.mid_kick END AS mid_kick,
               k.game_ts, k.game_ts - x.t AS ttk_s,
               mm.league, mm.ev_slug AS ev, mm.slug, mm.sports_type, mm.fee_rate, mm.rebate_rate
        FROM {cur} x JOIN kick k ON x.cond = k.cond
        JOIN (SELECT DISTINCT ON (condition_id) * FROM mk) mm ON x.cond = mm.condition_id
        WHERE x.t < k.game_ts AND x.m IS NOT NULL
      ) TO '{a.out}' (FORMAT parquet)""")
    n = con.execute(f"select count(*), count(distinct cond), count(distinct ev) from read_parquet('{a.out}')").fetchone()
    print("pre-game events written", n, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
