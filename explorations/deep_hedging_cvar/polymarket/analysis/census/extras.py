#!/usr/bin/env python3
"""Extra census cuts (needs cache/legs.parquet, out/top_makers.csv).

  out/top_vs_rest.csv     edge per share / per $ for each category's top-20 makers (by notional) vs everyone else
  out/sports_detail.csv   sports pre-game by time to game start and by league; in-play by league
  out/rewards_check.csv   pool-model rewards vs data-api REWARD payouts for the top-20 makers
  out/holding.csv         maker notional in markets with holding rewards enabled
"""
from pathlib import Path

from period import DAYS, LEGS, OUT, PAY, taker

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = """CASE WHEN category = 'sports_game' THEN (CASE WHEN inplay THEN 'sports_inplay' ELSE 'sports_pregame' END)
              ELSE category END"""


def agg(by: str) -> str:
    f = lambda h: f"WHERE mid_{h} IS NOT NULL AND mid_pre IS NOT NULL"
    return f"""
      SELECT {by}, count(DISTINCT cond) n_markets, sum(q*size)/{DAYS} usd_day, count(DISTINCT wallet) n_makers,
        100*sum(s*(mid_pre-q)*size) FILTER (WHERE mid_pre IS NOT NULL)/sum(size) FILTER (WHERE mid_pre IS NOT NULL) spread_c,
        100*sum(s*(mid_5m-mid_pre)*size) FILTER ({f('5m')})/sum(size) FILTER ({f('5m')}) mk_5m_c,
        100*sum(s*(mid_2h-mid_pre)*size) FILTER ({f('2h')})/sum(size) FILTER ({f('2h')}) mk_2h_c,
        100*sum(s*(final-mid_pre)*size) FILTER (WHERE final IS NOT NULL AND mid_pre IS NOT NULL)
           /sum(size) FILTER (WHERE final IS NOT NULL AND mid_pre IS NOT NULL) mk_res_c,
        100*sum(rebate_ps*size)/sum(size) rebate_c,
        1e4*(sum(s*(mid_2h-q)*size) FILTER ({f('2h')}) + sum(rebate_ps*q*size) FILTER ({f('2h')}) * 0
             + sum(rebate_ps*size) FILTER ({f('2h')}))/sum(q*size) FILTER ({f('2h')}) xr_2h_bps,
        1e4*(sum(s*(final-q)*size) FILTER (WHERE final IS NOT NULL) + sum(rebate_ps*size) FILTER (WHERE final IS NOT NULL))
           /sum(q*size) FILTER (WHERE final IS NOT NULL) xr_res_bps
      FROM legs GROUP BY ALL ORDER BY ALL"""


def main():
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='9GB'")
    con.execute(f"CREATE TABLE legs AS SELECT *, {KEY} AS ckey FROM {LEGS}")
    con.execute("""CREATE TABLE rk AS SELECT ckey, wallet,
                   row_number() OVER (PARTITION BY ckey ORDER BY sum(q*size) DESC) r FROM legs GROUP BY 1, 2""")
    tv = con.execute(agg("l.ckey, CASE WHEN rk.r <= 20 THEN 'top20' ELSE 'rest' END AS grp").replace(
        "FROM legs", "FROM legs l JOIN rk USING (ckey, wallet)").replace("SELECT l.ckey", "SELECT l.ckey")).df()
    tv.to_csv(OUT / "top_vs_rest.csv", index=False, float_format="%.4g")

    sp = con.execute(agg("""ckey, CASE WHEN ckey = 'sports_inplay' THEN 'inplay'
                           WHEN to_game_h < 1 THEN 'a<1h' WHEN to_game_h < 6 THEN 'b1-6h'
                           WHEN to_game_h < 24 THEN 'c6-24h' ELSE 'd>24h' END AS to_game""").replace(
        "FROM legs", "FROM legs WHERE ckey IN ('sports_pregame', 'sports_inplay')")).df()
    lg = con.execute(agg("ckey, subcat").replace(
        "FROM legs", "FROM legs WHERE ckey IN ('sports_pregame', 'sports_inplay', 'sports_futures')")).df()
    lt = con.execute(agg("""ckey, subcat, CASE WHEN ckey = 'sports_inplay' THEN 'inplay'
                           WHEN to_game_h < 1 THEN 'a<1h' WHEN to_game_h < 6 THEN 'b1-6h'
                           WHEN to_game_h < 24 THEN 'c6-24h' ELSE 'd>24h' END AS to_game""").replace(
        "FROM legs", "FROM legs WHERE ckey IN ('sports_pregame', 'sports_inplay')")).df()
    lt.to_csv(OUT / "sports_league_kickoff.csv", index=False, float_format="%.4g")
    pd.concat([sp.assign(cut="to_game"), lg.assign(cut="league")]).to_csv(
        OUT / "sports_detail.csv", index=False, float_format="%.4g")

    # rewards: pool model (today's rate_per_day x active days, pro rata to maker shares) vs paid REWARD rows
    rw = pd.read_parquet(HERE / "cache" / "rewards_current.parquet")
    tk = taker()
    con.register("rw", rw)
    con.register("tk", tk)
    con.execute(f"""CREATE TABLE pool AS SELECT l.cond, coalesce(rw.rate_per_day, 0) * least(coalesce(tk.n_days, 1), {DAYS})
                   / sum(l.size) AS rps FROM legs l LEFT JOIN rw ON l.cond = rw.condition_id
                   LEFT JOIN tk ON l.cond = tk.cond GROUP BY l.cond, rw.rate_per_day, tk.n_days""")
    model = con.execute("""SELECT wallet, sum(rps*size) model_reward, sum(q*size) usd_all
                           FROM legs JOIN pool USING (cond) GROUP BY 1""").df()
    top = pd.read_csv(OUT / "top_makers.csv")
    import sqlite3
    paid = pd.read_sql(f"select wallet, usd paid from rw_{PAY[0]}_{PAY[1]}", sqlite3.connect(HERE / "cache" / "rewards_wallet.sqlite"))
    paid["wallet_h"] = paid.wallet.str[2:12]
    x = paid.merge(model, on="wallet")
    x = x.merge(top.groupby("wallet_h").ckey.agg(lambda s: "|".join(sorted(set(s)))), on="wallet_h")
    tot = pd.DataFrame([{"wallets": len(x), "paid_total": x.paid.sum(), "model_total": x.model_reward.sum(),
                         "corr": x[["paid", "model_reward"]].corr().iloc[0, 1]}])
    tot.to_csv(OUT / "rewards_check.csv", index=False, float_format="%.4g")

    h = pd.read_parquet(HERE / "cache" / "markets.parquet", columns=["condition_id", "holding_rewards"]).drop_duplicates(
        "condition_id")
    con.register("h", h)
    con.execute(f"""COPY (SELECT ckey, sum(q*size) FILTER (WHERE h.holding_rewards)/{DAYS} usd_day_holding,
                   sum(q*size)/{DAYS} usd_day, count(DISTINCT cond) FILTER (WHERE h.holding_rewards) n_mkts_holding
                   FROM legs JOIN h ON legs.cond = h.condition_id GROUP BY 1 ORDER BY 1)
                   TO '{OUT / "holding.csv"}' (HEADER)""")
    pd.set_option("display.width", 250)
    print(tv.round(2).to_string())
    print(pd.read_csv(OUT / "sports_detail.csv").round(2).to_string())
    print(tot)
    print(pd.read_csv(OUT / "holding.csv"))


if __name__ == "__main__":
    main()
