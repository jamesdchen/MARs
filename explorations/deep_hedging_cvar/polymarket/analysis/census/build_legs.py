#!/usr/bin/env python3
"""Per-maker-leg economics for every on-chain fill of the census window.

  python build_legs.py --fills ../../onchain/data/all/fills --start 2026-10-08 --end 2026-10-14

Inputs: fills (extract.py --all --events OrderFilled --slim), cache/markets.parquet (gamma_tags.py).
Outputs (git-ignored cache/):
  legs.parquet   one row per maker leg: wallet, condition, category, price q, size, side sign s,
                 mid_pre (micro-price before the fill's block), mid at +1m/+5m/+30m/+2h, final price,
                 mark at window end, rebate per share, time to resolution, in-play flag.
  taker.parquet  per condition: taker notional, taker fees, n taker fills.

Frame: every price is put into the outcome-0 frame p0 (p0 = price for outcome 0, 1 - price for outcome 1),
so both tokens of a condition feed one price series. The micro-price is the mean of the latest taker-buy and
latest taker-sell prices (outcome-0 frame) at or before a time, or the latest taker price if only one side
has traded. The maker's own-token value is p0 (outcome 0) or 1 - p0 (outcome 1).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

from categories import market_category

HERE = Path(__file__).resolve().parent
HORIZONS = {"1m": 60, "5m": 300, "30m": 1800, "2h": 7200, "1d": 86400}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", required=True)
    ap.add_argument("--start", default="2026-10-08")
    ap.add_argument("--end", default="2026-10-14", help="inclusive")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="", help="writes cache/legs_<tag>.parquet, cache/taker_<tag>.parquet")
    ap.add_argument("--pad-days", type=float, default=1.0,
                    help="later fills loaded for the marks (legs themselves stay inside [start, end])")
    a = ap.parse_args()
    t_lo = int(pd.Timestamp(a.start, tz="UTC").timestamp())
    t_hi = int((pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    t_pad = int(t_hi + a.pad_days * 86400)
    sfx = f"_{a.tag}" if a.tag else ""
    t0 = time.time()

    m = pd.read_parquet(HERE / "cache" / "markets.parquet")
    gs = pd.to_datetime(m.game_start, utc=True, errors="coerce")
    es = pd.to_datetime(m.event_start_time, utc=True, errors="coerce")
    sec = lambda x: (x - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    m["game_ts"] = sec(gs.fillna(es))
    ct = pd.to_datetime(m.closed_time, utc=True, errors="coerce")
    ed = pd.to_datetime(m.end_date, utc=True, errors="coerce")
    m["res_ts"] = sec(ct.fillna(ed))
    m["close_ts"] = sec(ct)
    cats = [market_category(s, t, f, g is not None and g == g)
            for s, t, f, g in zip(m.slug, m.tags, m.fee_type, m.game_ts)]
    m["category"] = [c[0] for c in cats]
    m["subcat"] = [c[1] for c in cats]
    m["fee_rate"] = m.fee_rate.fillna(0.0).where(m.fees_enabled.fillna(False).astype(bool), 0.0)
    m["rebate_rate"] = m.rebate_rate.fillna(0.0)
    m["resolved"] = m.closed & m.final_price.isin([0.0, 1.0])
    mk = m[["token_id", "condition_id", "slug", "outcome_idx", "category", "subcat", "fee_rate", "rebate_rate",
            "game_ts", "res_ts", "close_ts", "final_price", "resolved", "neg_risk"]].copy()
    mk.to_parquet(HERE / "cache" / "markets_cat.parquet", index=False)

    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='9GB'; SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{HERE / 'cache' / 'duck_tmp'}'")
    con.register("mk_df", mk)
    con.execute("CREATE TABLE mk AS SELECT * FROM mk_df")
    con.execute(f"""
      CREATE TABLE f AS
      SELECT f.block, f.block_time AS t, f.log_index AS li, f.role, f.maker AS wallet, f.side, f.price, f.size,
             f.fee / 1e6 AS fee, mk.condition_id AS cond, mk.outcome_idx AS oi
      FROM read_parquet('{a.fills}/*/*.parquet', hive_partitioning=1) f JOIN mk USING (token_id)
      WHERE f.block_time >= {t_lo} AND f.block_time < {t_pad} AND f.size > 0""")
    print("fills", con.execute("select count(*) from f").fetchone(), f"{time.time()-t0:.0f}s", flush=True)
    # marks only where later data exist: cap at the last loaded fill (the pad may run past the extracted days)
    t_mark = min(t_pad, int(con.execute("select max(t) from f").fetchone()[0]) + 1)

    # taker prints in the outcome-0 frame; dir0 = +1 when the taker buys outcome 0 (lifts the ask)
    con.execute("""
      CREATE TABLE tp AS
      SELECT cond, block, li, t, CASE WHEN oi = 0 THEN price ELSE 1 - price END AS p0,
             CASE WHEN (side = 'BUY') = (oi = 0) THEN 1 ELSE -1 END AS dir0, size, price * size AS usd, fee
      FROM f WHERE role = 'taker'""")
    con.execute(f"""
      CREATE TABLE taker AS
      SELECT cond, count(*) n_taker, sum(usd) taker_usd, sum(fee) taker_fee, sum(size) taker_sh,
             count(DISTINCT CAST(t // 86400 AS INT)) n_days FROM tp WHERE t < {t_hi} GROUP BY 1""")
    con.execute(f"COPY taker TO '{HERE / 'cache' / f'taker{sfx}.parquet'}' (FORMAT parquet)")
    con.execute("""
      CREATE TABLE mid AS
      WITH x AS (
        SELECT cond, block, li, t, p0,
          last_value(CASE WHEN dir0 = 1 THEN p0 END IGNORE NULLS) OVER w AS lb,
          last_value(CASE WHEN dir0 = -1 THEN p0 END IGNORE NULLS) OVER w AS ls
        FROM tp WINDOW w AS (PARTITION BY cond ORDER BY block, li ROWS UNBOUNDED PRECEDING))
      SELECT cond, block, li, t,
             CASE WHEN lb IS NOT NULL AND ls IS NOT NULL AND abs(lb - ls) <= 0.10 THEN (lb + ls) / 2 ELSE p0 END AS mid
      FROM x""")
    # one row per (cond, block): the mid after the block's last print
    con.execute("""
      CREATE TABLE midb AS SELECT cond, block, t, arg_max(mid, li) AS mid FROM mid GROUP BY cond, block, t""")
    con.execute("DROP TABLE mid; DROP TABLE tp")
    print("mid series", con.execute("select count(*) from midb").fetchone(), f"{time.time()-t0:.0f}s", flush=True)

    con.execute(f"""
      CREATE TABLE ml AS
      SELECT f.block, f.t, f.li, f.wallet, f.cond, f.oi, f.price AS q, f.size,
             CASE WHEN f.side = 'BUY' THEN 1 ELSE -1 END AS s, f.fee AS maker_fee
      FROM f WHERE role = 'maker' AND t < {t_hi}""")
    con.execute("DROP TABLE f")
    # pre-fill micro-price: last block strictly before the fill's block
    con.execute("""
      CREATE TABLE ml2 AS
      SELECT ml.*, b.mid AS mid_pre0, ml.t - b.t AS pre_age
      FROM ml ASOF LEFT JOIN midb b ON ml.cond = b.cond AND ml.block > b.block""")
    con.execute("DROP TABLE ml")
    cur = "ml2"
    for k, h in HORIZONS.items():
        con.execute(f"""
          CREATE TABLE ml_{k} AS
          SELECT x.*, b.mid AS mid_{k}0
          FROM {cur} x ASOF LEFT JOIN midb b ON x.cond = b.cond AND x.t + {h} >= b.t""")
        con.execute(f"DROP TABLE {cur}")
        cur = f"ml_{k}"
        print("  horizon", k, f"{time.time()-t0:.0f}s", flush=True)
    con.execute(f"""
      CREATE TABLE lastmid AS SELECT cond, arg_max(mid, block) AS mid_end0 FROM midb WHERE t < {t_hi} GROUP BY 1""")
    own = lambda c: f"CASE WHEN x.oi = 0 THEN {c} ELSE 1 - {c} END"
    # after the market closed (closedTime <= t+h) its value is the settlement price
    hz = ",\n".join(f"CASE WHEN mk.resolved AND mk.close_ts <= x.t + {h} THEN {own('mk.final0')} "
                     f"WHEN x.t + {h} < {t_mark} THEN {own('x.mid_' + k + '0')} END AS mid_{k}"
                    for k, h in HORIZONS.items())
    con.execute(f"""
      COPY (
        SELECT x.block, x.t, x.wallet, x.cond, x.oi, x.q, x.size, x.s, x.maker_fee, x.pre_age,
               {own('x.mid_pre0')} AS mid_pre, mk.close_ts, {hz},
               CASE WHEN mk.resolved THEN {own('mk.final0')} END AS final,
               {own("CASE WHEN mk.resolved THEN mk.final0 ELSE lm.mid_end0 END")} AS mark_end,
               mk.category, mk.subcat, mk.slug,
               mk.rebate_rate * mk.fee_rate * x.q * (1 - x.q) AS rebate_ps,
               mk.fee_rate,
               (mk.res_ts - x.t) / 3600.0 AS ttr_h,
               CASE WHEN mk.category = 'sports_game' THEN (x.t >= mk.game_ts) END AS inplay,
               (mk.game_ts - x.t) / 3600.0 AS to_game_h
        FROM {cur} x JOIN (SELECT DISTINCT ON (condition_id) *,
                     CASE WHEN outcome_idx = 0 THEN final_price ELSE 1 - final_price END AS final0
                     FROM mk) mk ON x.cond = mk.condition_id
        LEFT JOIN lastmid lm ON x.cond = lm.cond
      ) TO '{HERE / 'cache' / f'legs{sfx}.parquet'}' (FORMAT parquet)""")
    print("legs written", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
