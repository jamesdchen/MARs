"""Actual REWARD and MAKER_REBATE payouts (data-api /activity) of the top maker wallets over the fills window.

Payout rows carry no market, so they cover each wallet's whole Polymarket activity, not only this universe.
The ratio actual_rebates / modelled_rebates_in_universe tells how much of the wallet's fee-paying maker flow is
outside the universe; universe-attributed rewards = rewards x min(1, modelled / actual rebates) (crude, assumes the
reward/rebate mix is the same in and outside the universe; it is not for fee-free geopolitics markets).

  python wallet_payouts.py politics [N=25]   -> out/politics_wallet_payouts.csv
"""
import json, sys, time, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SET = sys.argv[1] if len(sys.argv) > 1 else "politics"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25


def get(url):
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** i)
    return []


S = json.load(open(OUT / f"{SET}_summary.json"))
days = S["days"]
a = int(pd.Timestamp(days[0], tz="UTC").timestamp()); b = int((pd.Timestamp(days[-1], tz="UTC") + pd.Timedelta(days=1)).timestamp())
# payouts for epoch D arrive shortly after 00:00 UTC of D+1: shift the window by one day
a += 86400; b += 86400
W = pd.read_csv(OUT / f"{SET}_wallets.csv", index_col=0).head(N)
rows = []
for w in W.index:
    tot = {"REWARD": 0.0, "MAKER_REBATE": 0.0}
    off = 0
    while off < 5000:
        page = get(f"https://data-api.polymarket.com/activity?user={w}&type=REWARD,MAKER_REBATE&start={a}&end={b}&limit=500&offset={off}")
        for x in page:
            tot[x["type"]] = tot.get(x["type"], 0.0) + float(x["usdcSize"])
        if len(page) < 500:
            break
        off += 500
    rows.append(dict(maker=w, reward_paid=tot["REWARD"], rebate_paid=tot["MAKER_REBATE"]))
P = pd.DataFrame(rows).set_index("maker").join(W[["maker_shares", "rebate_usd", "spread_usd", "maker_pnl_mark", "taker_pnl_mark",
                                                  "fees_paid", "direction_usd", "total_pnl", "n_markets"]])
P["universe_frac_est"] = (P.rebate_usd / P.rebate_paid).clip(upper=1.0)
P["reward_attrib_est"] = P.reward_paid * P.universe_frac_est.fillna(0)
P.to_csv(OUT / f"{SET}_wallet_payouts.csv")
with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
    print(P)
    print(P.sum(numeric_only=True))
