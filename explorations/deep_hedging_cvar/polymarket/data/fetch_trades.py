#!/usr/bin/env python3
"""Fetch Polymarket market metadata (gamma) and all fills (data-api) for every
market in data/raw/updown_*.parquet.

Outputs
  data/raw/markets.parquet   one row per market (gamma metadata)
  data/raw/trades.parquet    one row per fill leg (taker leg + maker legs)
  data/raw/cache/<slug>.json.gz   per-market cache (not committed); reruns only
                                  fetch markets missing from the cache.

Empirical API notes (checked 2026-10-15, see data/README.md for the full write-up)
  * /trades?market=<conditionId>: newest first. `limit` is silently capped at
    10000; `offset` > 10000 is rejected with HTTP 400 ("max historical trades
    offset of 10000 exceeded"), so a single query reaches at most 20000 rows.
    If that cap is hit we split the query by `side=BUY` / `side=SELL`.
  * takerOnly defaults to true. takerOnly=true returns exactly one row per
    matched transaction: the TAKER's fill (side = aggressor side on the token it
    traded, price = taker's average fill price, size = taker's total size).
    takerOnly=false additionally returns one row per MAKER order filled in
    that tx (side/outcome/price from the maker's own perspective, price = the
    maker's resting limit price).
  * Complementary matching: Up and Down share one book. A taker BUY Down @ p
    can be filled by maker BUY Up @ 1-p (mint), a taker SELL Up by a maker
    SELL Down @ 1-p (merge), or same-token matches (BUY vs SELL). So in the Up
    frame: Down BUY @ p == Up SELL @ 1-p; Down SELL @ p == Up BUY @ 1-p.
  * gamma-api/data-api reject the default python urllib User-Agent (403);
    we send a browser-like UA.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CACHE = RAW / "cache"
GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data pipeline; polite)",
           "Accept": "application/json"}
PAGE = 10000          # server-side cap on limit
MAX_OFFSET = 10000    # server-side cap on offset
CACHE_VERSION = 1


def http_json(url: str, tries: int = 7):
    delay = 1.0
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            if e.code == 400 and b"offset" in body:
                raise
            if e.code in (429, 500, 502, 503, 504, 403) and k < tries - 1:
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {e.code} for {url}: {body!r}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            if k < tries - 1:
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"failed {url}: {e}")


def trim(rows):
    """Keep only the fields we use (drops names, bios, icons, titles)."""
    return [[r["transactionHash"], int(r["timestamp"]), r["proxyWallet"],
             r["side"], int(r["outcomeIndex"]), float(r["price"]), float(r["size"]),
             r["asset"]] for r in rows]


def fetch_paged(cond: str, taker_only: bool, side: str | None = None):
    rows, offset, truncated = [], 0, False
    while True:
        q = f"{DATA}/trades?market={cond}&limit={PAGE}&offset={offset}&takerOnly={'true' if taker_only else 'false'}"
        if side:
            q += f"&side={side}"
        page = http_json(q)
        rows.extend(page)
        if len(page) < PAGE:
            break
        if offset + PAGE > MAX_OFFSET:
            truncated = True
            break
        offset += PAGE
    return rows, truncated


def fetch_all(cond: str, taker_only: bool):
    rows, trunc = fetch_paged(cond, taker_only)
    if not trunc:
        return rows, False
    rows = []
    trunc_any = False
    for side in ("BUY", "SELL"):
        r, t = fetch_paged(cond, taker_only, side)
        rows.extend(r)
        trunc_any |= t
    return rows, trunc_any


GAMMA_KEYS = ["id", "question", "conditionId", "questionID", "slug", "resolutionSource",
              "startDate", "endDate", "eventStartTime", "closedTime", "outcomes",
              "outcomePrices", "clobTokenIds", "orderPriceMinTickSize", "orderMinSize",
              "makerBaseFee", "takerBaseFee", "feesEnabled", "feeType", "feeSchedule",
              "makerRebatesFeeShareBps", "rewardsMinSize", "rewardsMaxSpread",
              "umaResolutionStatus", "negRisk", "acceptingOrdersTimestamp",
              "cryptoMarketConfig", "volumeNum", "volumeClob"]


def fetch_market(slug: str) -> dict:
    path = CACHE / f"{slug}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        if d.get("v") == CACHE_VERSION:
            return d
    g = http_json(f"{GAMMA}/markets?slug={slug}&closed=true")
    if not g:
        g = http_json(f"{GAMMA}/markets?slug={slug}")
    if not g:
        raise RuntimeError(f"gamma: no market for {slug}")
    g = g[0]
    meta = {k: g.get(k) for k in GAMMA_KEYS}
    ev = (g.get("events") or [{}])[0]
    meta["eventMetadata"] = ev.get("eventMetadata")
    meta["eventStartTimeEv"] = ev.get("startTime")
    cond = g["conditionId"]
    taker, t1 = fetch_all(cond, True)
    allf, t2 = fetch_all(cond, False)
    d = {"v": CACHE_VERSION, "slug": slug, "gamma": meta,
         "taker": trim(taker), "all": trim(allf),
         "truncated_taker": t1, "truncated_all": t2,
         "fetched_at": int(time.time())}
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(d, f, separators=(",", ":"))
    tmp.rename(path)
    return d


def jl(x):
    return json.loads(x) if isinstance(x, str) else x


def build_outputs(slugs_df: pd.DataFrame, results: dict):
    mrows, trows = [], []
    for slug, info in slugs_df.iterrows():
        d = results[slug]
        g = d["gamma"]
        outcomes = jl(g["outcomes"])
        toks = jl(g["clobTokenIds"])
        tok = dict(zip(outcomes, toks))
        fs = g.get("feeSchedule") or {}
        em = g.get("eventMetadata") or {}
        cmc = g.get("cryptoMarketConfig") or {}
        prices = [float(p) for p in jl(g["outcomePrices"])]
        mrows.append(dict(
            slug=slug, coin=info.coin, contract_length=info.contract_length,
            gamma_id=int(g["id"]), condition_id=g["conditionId"], question_id=g["questionID"],
            question=g["question"],
            token_up=tok.get("Up"), token_down=tok.get("Down"),
            created_start=pd.Timestamp(g["startDate"]),
            event_start=pd.Timestamp(g.get("eventStartTime") or g.get("eventStartTimeEv")),
            end=pd.Timestamp(g["endDate"]),
            closed_time=pd.Timestamp(g["closedTime"]) if g.get("closedTime") else pd.NaT,
            tick_size=float(g["orderPriceMinTickSize"]), min_size=float(g["orderMinSize"]),
            maker_base_fee=float(g.get("makerBaseFee") or 0), taker_base_fee=float(g.get("takerBaseFee") or 0),
            fees_enabled=bool(g.get("feesEnabled")), fee_type=g.get("feeType"),
            fee_rate=float(fs.get("rate", np.nan)), fee_exponent=float(fs.get("exponent", np.nan)),
            fee_taker_only=bool(fs.get("takerOnly", False)), fee_rebate_rate=float(fs.get("rebateRate", np.nan)),
            maker_rebates_fee_share_bps=float(g.get("makerRebatesFeeShareBps") or 0),
            rewards_min_size=float(g.get("rewardsMinSize") or 0),
            rewards_max_spread=float(g.get("rewardsMaxSpread") or 0),
            resolution_source=g.get("resolutionSource"),
            twap_lookback_s=float(cmc.get("twapLookbackSeconds", np.nan)),
            outcome_price_up=prices[outcomes.index("Up")], winner_up=int(prices[outcomes.index("Up")] > 0.5),
            price_to_beat=float(em.get("priceToBeat", np.nan)), final_price=float(em.get("finalPrice", np.nan)),
            uma_status=g.get("umaResolutionStatus"),
            n_taker=len(d["taker"]), n_fills=len(d["all"]),
            truncated=bool(d["truncated_taker"] or d["truncated_all"]),
        ))
        taker_keys = {}
        for r in d["taker"]:
            k = (r[0], r[2], r[3], r[4], r[5], r[6])
            taker_keys[k] = taker_keys.get(k, 0) + 1
        for r in d["all"]:
            k = (r[0], r[2], r[3], r[4], r[5], r[6])
            is_t = taker_keys.get(k, 0) > 0
            if is_t:
                taker_keys[k] -= 1
            trows.append((slug, r[0], r[1], r[4], 1 if r[3] == "BUY" else -1, r[5], r[6], is_t))
        # taker rows that did not appear in the takerOnly=false listing (should be none)
        for k, n in taker_keys.items():
            for _ in range(n):
                trows.append((slug, k[0], int([r for r in d["taker"] if r[0] == k[0]][0][1]), k[3],
                              1 if k[2] == "BUY" else -1, k[4], k[5], True))
    markets = pd.DataFrame(mrows)
    t = pd.DataFrame(trows, columns=["slug", "tx", "timestamp", "outcome_idx", "side", "price", "size", "is_taker"])
    # per-tx aggressor side expressed in the Up frame (+1 = aggressor bought Up)
    tk = t[t.is_taker]
    up_side = np.where(tk.outcome_idx == 0, tk.side, -tk.side)
    agg = pd.Series(up_side, index=tk.tx.values)
    agg = agg[~agg.index.duplicated()]
    t["aggressor_up"] = t.tx.map(agg).fillna(0).astype(np.int8)
    t = t.sort_values(["slug", "timestamp", "tx", "is_taker"], ascending=[True, True, True, False], kind="stable")
    return markets, t.reset_index(drop=True)


def write_trades(t: pd.DataFrame, path: Path):
    slugs = pd.Categorical(t.slug)
    txbuf = b"".join(bytes.fromhex(x[2:]) for x in t.tx)
    tbl = pa.table({
        "slug": pa.DictionaryArray.from_arrays(pa.array(slugs.codes.astype(np.int32)), pa.array(list(slugs.categories))),
        "timestamp": pa.array(t.timestamp.values.astype(np.int64)),
        "outcome_idx": pa.array(t.outcome_idx.values.astype(np.int8)),
        "side": pa.array(t.side.values.astype(np.int8)),
        "price": pa.array(t.price.values.astype(np.float32)),
        "size": pa.array(t["size"].values.astype(np.float32)),
        "is_taker": pa.array(t.is_taker.values),
        "aggressor_up": pa.array(t.aggressor_up.values.astype(np.int8)),
        "tx": pa.FixedSizeBinaryArray.from_buffers(pa.binary(32), len(t), [None, pa.py_buffer(txbuf)]),
    })
    pq.write_table(tbl, path, compression="zstd", compression_level=19,
                   use_dictionary=["slug"], row_group_size=1 << 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only first N markets (debug)")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(RAW / f"updown_{cl}.parquet", columns=["market_slug", "coin", "contract_length", "token_id", "outcome"])
              for cl in ("5m", "15m", "4h")]
    raw = pd.concat(frames)
    slugs = raw.drop_duplicates("market_slug").set_index("market_slug")[["coin", "contract_length"]].sort_index()
    if args.limit:
        slugs = slugs.iloc[: args.limit]
    results, errors = {}, {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_market, s): s for s in slugs.index}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                results[s] = f.result()
            except Exception as e:  # noqa: BLE001
                errors[s] = str(e)
            if i % 50 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {time.time()-t0:.0f}s errors={len(errors)}", flush=True)
    if errors:
        for s, e in list(errors.items())[:20]:
            print("ERROR", s, e, file=sys.stderr)
        slugs = slugs.drop(list(errors))
    markets, trades = build_outputs(slugs, results)
    # sanity: gamma token ids match the order-book file token ids
    tokmap = raw.drop_duplicates(["market_slug", "outcome"]).pivot(index="market_slug", columns="outcome", values="token_id")
    mm = markets.set_index("slug")
    bad = (tokmap.loc[mm.index, "Up"] != mm.token_up) | (tokmap.loc[mm.index, "Down"] != mm.token_down)
    print("token id mismatches vs order-book file:", int(bad.sum()))
    markets.to_parquet(RAW / "markets.parquet", index=False, compression="zstd")
    write_trades(trades, RAW / "trades.parquet")
    print(f"markets: {len(markets)}  fills: {len(trades)}  taker fills: {int(trades.is_taker.sum())}  "
          f"truncated: {int(markets.truncated.sum())}  errors: {len(errors)}")
    for p in ("markets.parquet", "trades.parquet"):
        print(p, f"{(RAW / p).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
