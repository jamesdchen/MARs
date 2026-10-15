#!/usr/bin/env python3
"""Snapshot the reward / fee configuration the APIs expose, and the realised
REWARD / MAKER_REBATE payouts of the makers active in TRAIN markets.

1. Live crypto up/down markets (now): gamma fields + CLOB /clob-markets/{cid}
   + CLOB /rewards/markets/{cid}; membership in /rewards/markets/current and
   /sampling-markets (those two lists are cached by earlier ad-hoc pulls in
   cache/api/ and re-pulled here if missing).
2. TRAIN markets (Aug 21-22): same endpoints, to show what is (not) retained.
3. Payout survey: for the top-N makers by fee-equivalent in TRAIN markets, the
   data-api /activity REWARD and MAKER_REBATE rows paid between 2026-08-01 and
   2026-08-23 12:00 UTC (epochs up to Aug 22) and between 2026-08-25 12:00 and
   2026-09-15 (program end).  Epochs Aug 23 and Aug 24 (paid Aug 24/25) are
   deliberately skipped: they contain test-period markets.
Outputs: out/api_config.json, out/payouts_top_makers.csv, out/payouts_daily.csv
"""
from __future__ import annotations

import collections
import json
import time

import pandas as pd

from common import CACHE, OUT, http_json, load_market_cache, train_slugs

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
KEYS = ["slug", "conditionId", "acceptingOrdersTimestamp", "orderMinSize", "orderPriceMinTickSize",
        "rewardsMinSize", "rewardsMaxSpread", "clobRewards", "holdingRewardsEnabled", "feesEnabled",
        "feeType", "feeSchedule", "makerBaseFee", "takerBaseFee", "makerRebatesFeeShareBps"]
TOP_N = 150
T = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())  # noqa: E731
WINDOWS = [("2026-08-01", "2026-08-23 12:00"), ("2026-08-25 12:00", "2026-09-15")]


def market_config(slug, closed):
    g = http_json(f"{GAMMA}/markets?slug={slug}" + ("&closed=true" if closed else ""))
    if not g:
        return None
    g = g[0]
    out = {k: g.get(k) for k in KEYS}
    cid = g["conditionId"]
    out["clob_markets"] = http_json(f"{CLOB}/clob-markets/{cid}")
    out["clob_rewards_market"] = http_json(f"{CLOB}/rewards/markets/{cid}")
    return out


def paged_list(path, first_cursor=""):
    rows, cur = [], first_cursor
    for _ in range(400):
        d = http_json(f"{CLOB}{path}" + (f"?next_cursor={cur}" if cur else ""))
        rows += d["data"]
        cur = d.get("next_cursor")
        if not cur or cur == "LTE=":
            break
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    api = {"fetched_at_utc": pd.Timestamp.utcnow().isoformat(), "live": {}, "train": {}}
    now = int(time.time())
    for cl, step in (("5m", 300), ("15m", 900), ("4h", 14400)):
        for coin in ("btc", "eth", "zec"):
            b = now // step * step
            for cand in (b, b + step, b - step):
                cfg = market_config(f"{coin}-updown-{cl}-{cand}", closed=False)
                if cfg:
                    api["live"][cfg["slug"]] = cfg
                    break
    ts = train_slugs()
    for cl in ts:
        for slug in ts[cl][:2]:
            api["train"][slug] = market_config(slug, closed=True)
    cur_p = CACHE / "api" / "clob_rewards_markets_current.json"
    samp_p = CACHE / "api" / "clob_sampling_markets.json"
    cur = json.loads(cur_p.read_text()) if cur_p.exists() else paged_list("/rewards/markets/current")
    samp = json.loads(samp_p.read_text()) if samp_p.exists() else paged_list("/sampling-markets", "MA==")
    cur_ids = {r["condition_id"].lower() for r in cur}
    samp_ids = {r["condition_id"].lower() for r in samp}
    for grp in ("live", "train"):
        for slug, cfg in api[grp].items():
            if cfg:
                cfg["in_rewards_markets_current"] = cfg["conditionId"].lower() in cur_ids
                cfg["in_sampling_markets"] = cfg["conditionId"].lower() in samp_ids
    api["n_rewards_markets_current"] = len(cur)
    api["n_sampling_markets"] = len(samp)
    (OUT / "api_config.json").write_text(json.dumps(api, indent=1, default=str))

    # ---------------- payout survey ----------------
    fe = collections.Counter()
    for cl, sl in ts.items():
        for s in sl:
            d = load_market_cache(s)
            tk = collections.Counter(tuple(r[:7]) for r in d["taker"])
            for r in d["all"]:
                k = tuple(r[:7])
                if tk[k] > 0:
                    tk[k] -= 1
                    continue
                fe[r[2].lower()] += r[6] * 0.07 * r[5] * (1 - r[5])
    tot = sum(fe.values())
    rows = []
    for rank, (w, v) in enumerate(fe.most_common(TOP_N)):
        for a, b in WINDOWS:
            off = 0
            while True:
                page = http_json(f"{DATA}/activity?user={w}&type=REWARD,MAKER_REBATE&start={T(a)}&end={T(b)}"
                                 f"&limit=500&offset={off}")
                rows += [dict(maker=w, rank=rank, train_feeeq=v, train_feeeq_share=v / tot, type=r["type"],
                              ts=pd.Timestamp(r["timestamp"], unit="s", tz="UTC"), usd=float(r["usdcSize"]))
                         for r in page]
                if len(page) < 500:
                    break
                off += 500
    p = pd.DataFrame(rows)
    # payout at ~00:00-01:00 UTC on day D+1 settles epoch D
    p["epoch"] = (p.ts - pd.Timedelta("3h")).dt.floor("D").dt.date
    p.to_csv(OUT / "payouts_top_makers.csv", index=False)
    daily = p.pivot_table(index="epoch", columns="type", values="usd", aggfunc="sum").fillna(0)
    daily["n_makers_reward"] = p[p.type == "REWARD"].groupby("epoch").maker.nunique()
    daily.to_csv(OUT / "payouts_daily.csv")
    print(daily.to_string())


if __name__ == "__main__":
    main()
