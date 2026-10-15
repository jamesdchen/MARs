#!/usr/bin/env python3
"""Revenue decomposition for the actual top makers of the TRAIN markets.

For the top-N makers by fee-equivalent in train markets:
  spread  = sum over their in-window maker fills of size * edge vs the pre-fill mid
            (instantaneous edge, BEFORE adverse selection: placeholder column)
  rebate  = what CLOB /rebates/current says they were paid in those train markets
            (dates 2026-08-20..22, rows filtered to train condition ids immediately)
  reward  = wallet-level REWARD payout for epochs Aug 21+22 (data-api /activity, from
            out/payouts_top_makers.csv), scaled by the train share of the wallet's
            wallet-level MAKER_REBATE (crude attribution: rewards are not per-market public)
Needs out/fills_window.csv.gz (empirics.py) and out/payouts_top_makers.csv (rules_snapshot.py).
Output: out/maker_decomp.csv
"""
from __future__ import annotations

import collections

import pandas as pd

from common import OUT, http_json, load_market_cache, train_slugs

TOP_N = 25
DATES = ["2026-08-20", "2026-08-21", "2026-08-22"]


def main():
    ts = train_slugs()
    cond, fe, sh = {}, collections.Counter(), collections.Counter()
    for cl, sl in ts.items():
        for s in sl:
            d = load_market_cache(s)
            cond[d["gamma"]["conditionId"].lower()] = (s, cl)
            tk = collections.Counter(tuple(r[:7]) for r in d["taker"])
            for r in d["all"]:
                k = tuple(r[:7])
                if tk[k] > 0:
                    tk[k] -= 1
                    continue
                fe[r[2].lower()] += r[6] * 0.07 * r[5] * (1 - r[5])
                sh[r[2].lower()] += r[6]
    fl = pd.read_csv(OUT / "fills_window.csv.gz")
    fl["maker"] = fl.maker.str.lower()
    fl["edge_usd"] = fl.edge * fl["size"]
    pay = pd.read_csv(OUT / "payouts_top_makers.csv")
    pay = pay[pay.epoch.isin(["2026-08-21", "2026-08-22"])]
    rows = []
    for w, v in fe.most_common(TOP_N):
        reb_train = reb_wallet = 0.0
        reb_train_d = 0.0
        for date in DATES:
            rb = http_json(f"https://clob.polymarket.com/rebates/current?date={date}&maker_address={w}")
            tot = sum(float(x["rebated_fees_usdc"]) for x in rb)
            tr = sum(float(x["rebated_fees_usdc"]) for x in rb if x["condition_id"].lower() in cond)
            reb_train += tr
            if date != "2026-08-20":
                reb_wallet += tot
                reb_train_d += tr
        f = fl[fl.maker == w]
        pw = pay[pay.maker == w]
        rew_wallet = pw[pw.type == "REWARD"].usd.sum()
        mr_wallet = pw[pw.type == "MAKER_REBATE"].usd.sum()
        rows.append(dict(
            maker=w, train_shares=sh[w], train_feeeq=v, train_rebate=reb_train,
            eff_rebate_rate=reb_train / v if v else float("nan"),
            window_shares=f["size"].sum(), spread_edge_usd_window=f.edge_usd.sum(),
            edge_per_share=f.edge_usd.sum() / f["size"].sum() if len(f) else float("nan"),
            rebate_per_share=reb_train / sh[w],
            adverse_selection_usd=float("nan"),
            wallet_reward_aug21_22=rew_wallet, wallet_rebate_aug21_22=mr_wallet,
            reward_attrib_train=rew_wallet * reb_train_d / reb_wallet if reb_wallet else 0.0,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "maker_decomp.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    t = df[["spread_edge_usd_window", "train_rebate", "reward_attrib_train"]].sum()
    print(t, "\nshares:", (t / t.sum()).round(3).to_dict())


if __name__ == "__main__":
    main()
