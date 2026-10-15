#!/usr/bin/env python3
"""Reconcile on-chain extracts against the data-api and against data/raw/trades.parquet.

  python validate.py market --slug bnb-updown-5m-1787277900 --out data/val_bnb
      fills of one market vs data-api /trades (takerOnly=false and =true): exact key match
      (wallet, side, token, size, tx), price and timestamp agreement, roles, fees.
  python validate.py day --date 2026-08-21 --trades data/crypto-updown/trades_2026-08-21.parquet
      to_trades.py output vs data/raw/trades.parquet for the markets both have, restricted to fills whose
      block time falls on that UTC day: per-market leg/taker counts, share volume and notional, and an exact
      multiset match on (slug, tx, outcome, side, is_taker, size, price).
Results are printed and written to validation/<name>.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import common as C
import extract
import markets as M

DATA_API = "https://data-api.polymarket.com"
VDIR = C.HERE / "validation"


def data_api_trades(cond: str, taker_only: bool) -> list[dict]:
    s = requests.Session()
    s.headers.update({k: v for k, v in C.HTTP_HEADERS.items() if k != "Content-Type"})
    rows, off = [], 0
    while True:
        for k in range(6):
            r = s.get(f"{DATA_API}/trades", params=dict(market=cond, limit=10000, offset=off,
                                                       takerOnly="true" if taker_only else "false"), timeout=60)
            if r.status_code == 200:
                break
            time.sleep(2 ** k)
        page = r.json()
        rows += page
        if len(page) < 10000 or off >= 10000:
            return rows
        off += 10000


def save(name: str, res: dict):
    VDIR.mkdir(exist_ok=True)
    (VDIR / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def market(args):
    g = M.Gamma()
    tok = g.markets([args.slug])
    cond = tok.condition_id.iloc[0]
    out_map = dict(zip(tok.token_id, tok.outcome))
    f = extract.load(args.out, "fills")
    f = f[f.token_id.isin(out_map)].copy()
    api = data_api_trades(cond, False)
    apiT = data_api_trades(cond, True)

    def k_chain(d):
        return (d.maker.lower(), d.side, d.token_id, round(d["size"], 4), d.tx_hash.lower())

    def k_api(a):
        return (a["proxyWallet"].lower(), a["side"], a["asset"], round(float(a["size"]), 4), a["transactionHash"].lower())
    def kp_chain(d):
        return k_chain(d) + (round(d.price, 4),)

    def kp_api(a):
        return k_api(a) + (round(float(a["price"]), 4),)
    ocp = collections.Counter(kp_chain(r) for _, r in f.iterrows())
    akp = collections.Counter(kp_api(a) for a in api)
    oc = collections.Counter(k_chain(r) for _, r in f.iterrows())
    ak = collections.Counter(k_api(a) for a in api)
    ocT = collections.Counter(k_chain(r) for _, r in f[f.role == "taker"].iterrows())
    akT = collections.Counter(k_api(a) for a in apiT)
    am = {kp_api(a): a for a in api}
    dp, dt = [], collections.Counter()
    for _, r in f.iterrows():
        a = am.get(kp_chain(r))
        if a is not None:
            dp.append(abs(float(a["price"]) - r.price))
            dt[int(a["timestamp"]) - int(r.block_time)] += 1
    tk = f[f.role == "taker"]
    notional = float((tk["size"] * tk.price).sum())
    fee_formula = float((tk["size"] * 0.07 * tk.price * (1 - tk.price)).sum())
    res = dict(
        slug=args.slug, condition_id=cond,
        onchain_fills=len(f), onchain_taker=int((f.role == "taker").sum()), onchain_maker=int((f.role == "maker").sum()),
        api_rows=len(api), api_taker_rows=len(apiT),
        exact_key_matches=int(sum((oc & ak).values())), api_only=len(list((ak - oc).elements())),
        onchain_only=len(list((oc - ak).elements())),
        taker_matches=int(sum((ocT & akT).values())),
        exact_key_and_price_matches=int(sum((ocp & akp).values())),
        max_abs_price_diff=max(dp) if dp else None,
        n_price_diff_gt_1e6=int(sum(x > 1e-6 for x in dp)),
        ts_api_minus_block=dict(dt),
        taker_fee_usdc=round(float(tk.fee.sum() / 1e6), 4), maker_fee_usdc=round(float(f[f.role == "maker"].fee.sum() / 1e6), 4),
        taker_fee_over_formula_median=round(float(np.median(tk.fee / 1e6 / (tk["size"] * 0.07 * tk.price * (1 - tk.price)))), 4),
        fee_formula="fee_usdc = size * 0.07 * p * (1 - p) per taker fill",
        taker_notional=round(notional, 2),
        distinct_makers=int(f[f.role == "maker"].maker.nunique()), txs=int(f.tx_hash.nunique()),
    )
    od = C.HERE / args.out / "orders" if not Path(args.out).is_absolute() else Path(args.out) / "orders"
    if od.exists():       # calldata summary (if calldata.py was run on this extract)
        o = pd.concat([pd.read_parquet(p) for p in od.glob("date=*/*.parquet")])
        o = o[o.token_id.isin(out_map)]
        mk = o[o.role == "maker"]
        res["calldata"] = dict(
            orders=len(o), maker_partial=f"{int(mk.partial.sum())}/{len(mk)}",
            maker_age_median_s=round(float(mk.age_s.median()), 2), taker_age_median_s=round(float(o[o.role == "taker"].age_s.median()), 2),
            orders_without_timestamp=int(o.created_s.isna().sum()))
    save(f"market_{args.slug}", res)


def day(args):
    d0 = C.utc_ts(args.date)
    d1 = d0 + 86400
    oc = pd.read_parquet(args.trades)
    raw = pd.read_parquet(C.REPO_DATA / "trades.parquet")
    for t in (oc, raw):
        t["slug"] = t.slug.astype(str)
    raw = raw[(raw.timestamp >= d0) & (raw.timestamp < d1)]
    oc = oc[(oc.timestamp >= d0) & (oc.timestamp < d1)]
    common = sorted(set(raw.slug) & set(oc.slug))
    raw_only_slugs = sorted(set(raw.slug) - set(oc.slug))
    r = raw[raw.slug.isin(common)].copy()
    o = oc[oc.slug.isin(common)].copy()

    def agg(t):
        t = t.assign(notional=t["size"].astype(np.float64) * t.price.astype(np.float64))
        g = t.groupby("slug")
        return pd.DataFrame(dict(legs=g.size(), taker=g.is_taker.sum(), vol=g["size"].sum(), notional=g.notional.sum(),
                                 txs=g.tx.nunique()))
    ar, ao = agg(r), agg(o)
    j = ar.join(ao, lsuffix="_raw", rsuffix="_chain")
    j["legs_eq"] = j.legs_raw == j.legs_chain
    j["taker_eq"] = j.taker_raw == j.taker_chain
    j["vol_rel"] = (j.vol_chain / j.vol_raw - 1).abs()
    j["notional_rel"] = (j.notional_chain / j.notional_raw - 1).abs()

    def keys(t):
        return collections.Counter(zip(t.slug, t.tx, t.outcome_idx, t.side, t.is_taker,
                                       np.round(t["size"].astype(np.float64), 3), np.round(t.price.astype(np.float64), 3)))
    kr, ko = keys(r), keys(o)
    m = sum((kr & ko).values())
    # same but ignoring price (isolates price-convention differences)
    def keys_np(t):
        return collections.Counter(zip(t.slug, t.tx, t.outcome_idx, t.side, t.is_taker, np.round(t["size"].astype(np.float64), 3)))
    knr, kno = keys_np(r), keys_np(o)
    tsr = collections.Counter(zip(r.tx, r.timestamp))
    tso = collections.Counter(zip(o.tx, o.timestamp))
    res = dict(
        date=args.date, markets_raw_that_day=int(raw.slug.nunique()), markets_chain_that_day=int(oc.slug.nunique()),
        markets_compared=len(common), raw_markets_missing_onchain=raw_only_slugs[:20],
        legs_raw=int(len(r)), legs_chain=int(len(o)), taker_raw=int(r.is_taker.sum()), taker_chain=int(o.is_taker.sum()),
        exact_leg_matches=int(m), exact_leg_match_rate_raw=round(m / max(1, len(r)), 6),
        exact_leg_match_rate_chain=round(m / max(1, len(o)), 6),
        leg_matches_ignoring_price=int(sum((knr & kno).values())),
        tx_timestamp_matches=int(sum((tsr & tso).values())),
        markets_legs_equal=int(j.legs_eq.sum()), markets_taker_equal=int(j.taker_eq.sum()),
        markets_vol_rel_err_lt_1e4=int((j.vol_rel < 1e-4).sum()), markets_notional_rel_err_lt_1e4=int((j.notional_rel < 1e-4).sum()),
        vol_raw=round(float(r["size"].sum()), 2), vol_chain=round(float(o["size"].sum()), 2),
        notional_raw=round(float((r["size"] * r.price).sum()), 2),
        notional_chain=round(float((o["size"] * o.price).sum()), 2),
        aggressor_up_agree=None,
        worst_markets=j.sort_values("vol_rel", ascending=False).head(5)[["legs_raw", "legs_chain", "vol_raw", "vol_chain"]]
        .reset_index().to_dict("records"),
        chain_markets_not_in_raw=int(len(set(oc.slug) - set(raw.slug))),
        chain_legs_not_in_raw_markets=int((~oc.slug.isin(set(raw.slug))).sum()),
    )
    ag_r = r.drop_duplicates("tx").set_index("tx").aggressor_up
    ag_o = o.drop_duplicates("tx").set_index("tx").aggressor_up
    both = ag_r.index.intersection(ag_o.index)
    res["aggressor_up_agree"] = f"{int((ag_r.loc[both] == ag_o.loc[both]).sum())}/{len(both)}"
    save(f"day_{args.date}", res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("market")
    a.add_argument("--slug", required=True)
    a.add_argument("--out", required=True, help="extract.py output dir holding fills/")
    b = sub.add_parser("day")
    b.add_argument("--date", required=True)
    b.add_argument("--trades", required=True)
    args = ap.parse_args()
    if args.cmd == "day" and args.date >= "2026-08-23":
        sys.exit("refusing: 2026-08-23/24 are the val/test days")
    {"market": market, "day": day}[args.cmd](args)


if __name__ == "__main__":
    main()
