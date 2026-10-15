#!/usr/bin/env python3
"""Census-definition maker legs for pre-game sports (analysis/census/build_legs.py restricted to sports games),
with the extra columns needed to switch the census definitions to ours one at a time (RECONCILE.md).

  python reconcile_legs.py --fills ../../onchain/data/all/fills --start 2026-10-08 --end 2026-10-14 --out cache/legs_oct8.parquet

Census conventions kept: own-token frame (q = maker price, s = +1 maker buy), the micro-price of the condition
(both tokens folded into outcome 0) at the last block strictly before the fill, mid at t + h by ASOF, and the
settlement price once the market closed by t + h. Pre-game = fill time < game_ts (gameStartTime, else
eventStartTime). Extra: mid_30cap (t + 30 min capped at kickoff), mid_kick (last mid before kickoff), touch
level of the leg's event (ceil/floor of the micro-price on the 1c grid, frame 0), event id, league, type.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

from categories import market_category

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="inclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    t_lo = int(pd.Timestamp(a.start, tz="UTC").timestamp())
    t_hi = int((pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    t0 = time.time()
    m = pd.read_parquet(HERE / "cache" / "markets.parquet")
    m = m[m.slug.notna()].copy()
    sec = lambda x: (x - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()  # noqa: E731
    m["game_ts"] = sec(pd.to_datetime(m.game_start, utc=True, errors="coerce").fillna(
        pd.to_datetime(m.event_start_time, utc=True, errors="coerce")))
    m["close_ts"] = sec(pd.to_datetime(m.closed_time, utc=True, errors="coerce"))
    cats = [market_category(s, t, f, g == g) for s, t, f, g in zip(m.slug, m.tags, m.fee_type, m.game_ts)]
    m["category"] = [c[0] for c in cats]
    m["league"] = [c[1] for c in cats]
    m = m[m.category == "sports_game"]
    m["fee_rate"] = m.fee_rate.fillna(0.0).where(m.fees_enabled.fillna(False).astype(bool), 0.0)
    m["rebate_rate"] = m.rebate_rate.fillna(0.0)
    m["resolved"] = m.closed & m.final_price.isin([0.0, 1.0])
    m["final0"] = m.final_price.where(m.outcome_idx == 0, 1 - m.final_price)
    mk = m[["token_id", "condition_id", "outcome_idx", "league", "ev_slug", "sports_type", "game_ts", "close_ts",
            "resolved", "final0", "fee_rate", "rebate_rate"]].copy()
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='10GB'; SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{HERE / 'cache' / 'duck_tmp'}'")
    con.register("mk_df", mk)
    con.execute("CREATE TABLE mk AS SELECT * FROM mk_df")
    con.execute(f"""
      CREATE TABLE f AS
      SELECT f.block, f.block_time AS t, f.log_index AS li, f.tx_hash AS tx, f.role, f.side, f.size, f.price,
             mk.condition_id AS cond, mk.outcome_idx AS oi,
             CASE WHEN mk.outcome_idx = 0 THEN f.price ELSE 1 - f.price END AS p0,
             CASE WHEN (f.side = 'BUY') = (mk.outcome_idx = 0) THEN 1 ELSE -1 END AS s0
      FROM read_parquet('{a.fills}/*/*.parquet', hive_partitioning=1) f JOIN mk USING (token_id)
      WHERE f.block_time >= {t_lo} AND f.block_time < {t_hi} AND f.size > 0""")
    con.execute("""
      CREATE TABLE midb AS
      WITH x AS (
        SELECT cond, block, li, t, p0,
          last_value(CASE WHEN s0 = 1 THEN p0 END IGNORE NULLS) OVER w AS lb,
          last_value(CASE WHEN s0 = -1 THEN p0 END IGNORE NULLS) OVER w AS ls
        FROM f WHERE role = 'taker' WINDOW w AS (PARTITION BY cond ORDER BY block, li ROWS UNBOUNDED PRECEDING)),
      y AS (SELECT cond, block, li, t,
             CASE WHEN lb IS NOT NULL AND ls IS NOT NULL AND abs(lb - ls) <= 0.10 THEN (lb + ls) / 2 ELSE p0 END AS mid
            FROM x)
      SELECT cond, block, t, arg_max(mid, li) AS mid FROM y GROUP BY cond, block, t""")
    con.execute("""
      CREATE TABLE ml AS
      SELECT f.block, f.t, f.tx, f.cond, f.oi, f.price AS q, f.size, CASE WHEN f.side = 'BUY' THEN 1 ELSE -1 END AS s,
             f.s0, mk.game_ts, mk.close_ts, mk.resolved, mk.final0, mk.league, mk.ev_slug AS ev, mk.sports_type,
             mk.rebate_rate * mk.fee_rate * f.price * (1 - f.price) AS rebate_ps
      FROM f JOIN (SELECT DISTINCT ON (condition_id) * FROM mk) mk ON f.cond = mk.condition_id
      WHERE f.role = 'maker' AND f.t < mk.game_ts""")
    con.execute("CREATE TABLE ml1 AS SELECT x.*, b.mid AS pre0, x.t - b.t AS pre_age FROM ml x "
                "ASOF LEFT JOIN midb b ON x.cond = b.cond AND x.block > b.block")
    con.execute("CREATE TABLE ml2 AS SELECT x.*, b.mid AS m30_0 FROM ml1 x "
                "ASOF LEFT JOIN midb b ON x.cond = b.cond AND x.t + 1800 >= b.t")
    con.execute("CREATE TABLE ml3 AS SELECT x.*, b.mid AS m30cap_0 FROM ml2 x "
                "ASOF LEFT JOIN midb b ON x.cond = b.cond AND least(x.t + 1800, x.game_ts - 1) >= b.t")
    con.execute("CREATE TABLE ml4 AS SELECT x.*, b.mid AS kick_0 FROM ml3 x "
                "ASOF LEFT JOIN midb b ON x.cond = b.cond AND x.game_ts - 1 >= b.t")
    own = lambda c: f"CASE WHEN oi = 0 THEN {c} ELSE 1 - {c} END"  # noqa: E731
    con.execute(f"""
      COPY (
        SELECT t, block, tx, cond, oi, q, size, s, s0, pre_age, rebate_ps, league, ev, sports_type, game_ts,
               game_ts - t AS ttk_s, pre0 AS m0,
               {own('pre0')} AS mid_pre,
               CASE WHEN resolved AND close_ts <= t + 1800 THEN {own('final0')}
                    WHEN t + 1800 < {t_hi} THEN {own('m30_0')} END AS mid_30,
               CASE WHEN least(t + 1800, game_ts) < {t_hi} THEN {own('m30cap_0')} END AS mid_30cap,
               CASE WHEN game_ts < {t_hi} THEN {own('kick_0')} END AS mid_kick
        FROM ml4) TO '{a.out}' (FORMAT parquet)""")
    n = con.execute(f"select count(*), count(distinct cond), count(distinct ev) from read_parquet('{a.out}')").fetchone()
    print("pre-game maker legs", n, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
