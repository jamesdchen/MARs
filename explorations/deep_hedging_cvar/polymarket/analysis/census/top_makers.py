#!/usr/bin/env python3
"""Realised P&L of each category's top-20 maker wallets (by maker notional) over the census window.

  python top_makers.py [--n 20]

For every in-window maker fill: spread = s·(mid_pre − q)·size, adverse selection = s·(mark_end − mid_pre)·size,
where mark_end is the settlement price if the market resolved (by 2026-10-15) and else the last micro-price
of the window. Rebates use the market's feeSchedule. Liquidity rewards: data-api /activity type=REWARD paid
for epochs inside the window (payout timestamps Oct 9 00:00 – Oct 15 06:00 UTC). REWARD rows carry no market,
so a wallet's rewards are split across categories in proportion to its maker notional in each (an
approximation; flagged in the report). Positions carried into the window are not included.
Writes out/top_makers.csv (wallets hashed to 10 hex chars) and out/top_makers_by_cat.csv.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from period import DAYS, LEGS, OUT, PAY

import duckdb
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
KEY = """CASE WHEN category = 'sports_game' THEN (CASE WHEN inplay THEN 'sports_inplay' ELSE 'sports_pregame' END)
              ELSE category END"""
P_START, P_END = PAY                             # default 2026-10-09 00:00 .. 2026-10-15 06:00 UTC
RW = f"rw_{P_START}_{P_END}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/126.0"}


def rewards(w: str, db_path: Path) -> tuple[str, float, int]:
    db = sqlite3.connect(db_path)
    r = db.execute(f"select usd, n from {RW} where wallet=?", (w,)).fetchone()
    if r:
        return w, r[0], r[1]
    s = requests.Session()
    s.headers.update(UA)
    tot, n, off = 0.0, 0, 0
    while True:
        for k in range(6):
            try:
                resp = s.get("https://data-api.polymarket.com/activity",
                             params={"user": w, "type": "REWARD", "start": P_START, "end": P_END,
                                     "limit": 500, "offset": off}, timeout=60)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** k)
        rows = resp.json() if resp.status_code == 200 else []
        tot += sum(float(x.get("usdcSize") or 0) for x in rows)
        n += len(rows)
        if len(rows) < 500 or off >= 3000:
            break
        off += 500
    db.execute(f"insert or replace into {RW} values(?,?,?)", (w, tot, n))
    db.commit()
    return w, tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--days", type=float, default=DAYS)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='9GB'")
    con.execute(f"CREATE VIEW legs AS SELECT *, {KEY} AS ckey FROM {LEGS}")
    w = con.execute(f"""
      SELECT ckey, wallet, count(*) fills, sum(q*size) usd, count(DISTINCT cond) n_mkts,
             sum(s*(mid_pre-q)*size) FILTER (WHERE mid_pre IS NOT NULL) spread,
             sum(s*(mark_end-mid_pre)*size) FILTER (WHERE mid_pre IS NOT NULL) adverse,
             sum(s*(mark_end-q)*size) FILTER (WHERE mid_pre IS NULL) unattrib,
             sum(rebate_ps*size) rebate, sum(maker_fee) maker_fee,
             sum(s*size*q) net_buy_usd,
             sum(q*size) FILTER (WHERE final IS NOT NULL) / sum(q*size) res_share
      FROM legs GROUP BY 1, 2""").df()
    w["rank"] = w.groupby("ckey").usd.rank(ascending=False, method="first")
    top = w[w["rank"] <= a.n].copy()
    wall = con.execute("SELECT wallet, sum(q*size) usd_all FROM legs GROUP BY 1").df()
    top = top.merge(wall, on="wallet")
    dbp = HERE / "cache" / "rewards_wallet.sqlite"
    sqlite3.connect(dbp).execute(f"create table if not exists {RW}(wallet text primary key, usd real, n int)")
    with ThreadPoolExecutor(4) as ex:
        rw = pd.DataFrame(list(ex.map(lambda x: rewards(x, dbp), top.wallet.unique())),
                          columns=["wallet", "reward_wallet", "n_reward_rows"])
    top = top.merge(rw, on="wallet")
    top["reward"] = top.reward_wallet * top.usd / top.usd_all
    top["pnl"] = top.spread + top.adverse + top.unattrib.fillna(0) + top.rebate + top.reward - top.maker_fee
    top["pnl_bps"] = 1e4 * top.pnl / top.usd
    top["wallet_h"] = top.wallet.str[2:12]
    cols = ["ckey", "rank", "wallet_h", "fills", "usd", "n_mkts", "res_share", "spread", "adverse", "unattrib",
            "rebate", "reward", "maker_fee", "pnl", "pnl_bps", "net_buy_usd", "usd_all"]
    top.sort_values(["ckey", "rank"])[cols].to_csv(OUT / "top_makers.csv", index=False, float_format="%.5g")
    g = top.groupby("ckey")[["usd", "spread", "adverse", "unattrib", "rebate", "reward", "maker_fee", "pnl"]].sum()
    g["n_profitable"] = top.assign(p=top.pnl > 0).groupby("ckey").p.sum()
    g["pnl_bps"] = 1e4 * g.pnl / g.usd
    for c in ["spread", "adverse", "rebate", "reward"]:
        g[c + "_bps"] = 1e4 * g[c] / g.usd
    g["pnl_day"] = g.pnl / a.days
    tot = w.groupby("ckey").usd.sum()
    g["top_share_of_maker_usd"] = g.usd / tot
    g = g.sort_values("pnl_bps", ascending=False)
    g.to_csv(OUT / "top_makers_by_cat.csv", float_format="%.5g")
    pd.set_option("display.width", 250)
    print(g.round(1).to_string())


if __name__ == "__main__":
    main()
