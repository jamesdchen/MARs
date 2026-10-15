#!/usr/bin/env python3
"""Empirically recover the maker-rebate rate actually paid, per date.

For a sample of BTC/ETH up/down markets on given dates (never Aug 23/24 2026),
pull every fill from data-api (takerOnly=false minus takerOnly=true = maker legs),
compute each maker's fee-equivalent  C*0.07*p*(1-p)  on that UTC date, and
compare with what CLOB /rebates/current reports as rebated to that maker in
that market on that date.  ratio = rebated / fee_equivalent  is the effective
rebate rate (should equal feeSchedule.rebateRate if the docs are complete).
For the TRAIN days (Aug 21/22) we use the data agent's cached fills.
Output: out/rebate_rate_by_date.csv
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np
import pandas as pd

from common import CACHE, OUT, http_json, is_train, train_slugs, load_market_cache

FORBIDDEN = (pd.Timestamp("2026-08-23", tz="UTC"), pd.Timestamp("2026-08-25", tz="UTC"))
DATES = ["2026-08-05", "2026-08-12", "2026-08-21", "2026-08-22", "2026-09-03",
         "2026-09-10", "2026-09-24", "2026-10-01", "2026-10-08", "2026-10-13", "2026-10-14"]
N_MAKERS = 6


def fills(slug: str):
    p = CACHE / "fills" / f"{slug}.json"
    if p.exists():
        return json.loads(p.read_text())
    if is_train(slug):
        d = load_market_cache(slug)
        out = {"cond": d["gamma"]["conditionId"], "taker": d["taker"], "all": d["all"]}
    else:
        g = http_json(f"https://gamma-api.polymarket.com/markets?slug={slug}&closed=true")
        if not g:
            return None
        cond = g[0]["conditionId"]
        def page(tk):
            rows, off = [], 0
            while True:
                r = http_json(f"https://data-api.polymarket.com/trades?market={cond}&limit=10000&offset={off}&takerOnly={tk}")
                rows += [[x["transactionHash"], int(x["timestamp"]), x["proxyWallet"], x["side"],
                          int(x["outcomeIndex"]), float(x["price"]), float(x["size"])] for x in r]
                if len(r) < 10000 or off >= 10000:
                    return rows
                off += 10000
        out = {"cond": cond, "taker": page("true"), "all": page("false")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def maker_legs(d):
    tk = collections.Counter(tuple(r[:7]) for r in d["taker"])
    for r in d["all"]:
        k = tuple(r[:7])
        if tk[k] > 0:
            tk[k] -= 1
            continue
        yield r


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    ts = train_slugs()
    for date in DATES:
        day0 = pd.Timestamp(date, tz="UTC")
        assert not (FORBIDDEN[0] <= day0 < FORBIDDEN[1])
        if date in ("2026-08-21", "2026-08-22"):
            slugs = [s for s in ts["5m"] + ts["15m"] if s.startswith("btc-")][::6][:6]
        else:
            base = int((day0 + pd.Timedelta("12h")).timestamp())
            slugs = [f"btc-updown-5m-{base}", f"btc-updown-15m-{base}", f"eth-updown-5m-{base + 3600}",
                     f"btc-updown-5m-{base + 7200}"]
        for slug in slugs:
            d = fills(slug)
            if d is None:
                continue
            t0, t1 = day0.timestamp(), (day0 + pd.Timedelta("1D")).timestamp()
            fe = collections.Counter()
            for r in maker_legs(d):
                if t0 <= r[1] < t1:
                    fe[r[2].lower()] += r[6] * 0.07 * r[5] * (1 - r[5])
            taker_fees = sum(r[6] * 0.07 * r[5] * (1 - r[5]) for r in d["taker"] if t0 <= r[1] < t1)
            for w, f in fe.most_common(N_MAKERS):
                rb = http_json(f"https://clob.polymarket.com/rebates/current?date={date}&maker_address={w}")
                got = sum(float(x["rebated_fees_usdc"]) for x in rb
                          if x["condition_id"].lower() == d["cond"].lower())
                rows.append(dict(date=date, slug=slug, maker=w, fee_eq=f, rebated=got,
                                 ratio=got / f if f else np.nan, taker_fees_mkt_day=taker_fees,
                                 maker_feeeq_mkt_day=sum(fe.values())))
            print(date, slug, "done", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rebate_rate_by_date.csv", index=False)
    g = df[df.fee_eq > 0.5].groupby("date").apply(
        lambda x: pd.Series(dict(n=len(x), ratio_median=x.ratio.median(),
                                 ratio_pooled=x.rebated.sum() / x.fee_eq.sum(),
                                 frac_zero=(x.rebated == 0).mean())))
    print(g.to_string())
    g.to_csv(OUT / "rebate_rate_summary.csv")


if __name__ == "__main__":
    main()
