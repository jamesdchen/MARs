#!/usr/bin/env python3
"""Liquidity-reward capture for a hypothetical new maker on the top-N midterm markets, from live CLOB books.

  python election_rewards.py snap  [--n 300] [--k 6] [--every 600]   # K book snapshots per market
  python election_rewards.py score [--boot 2000]                       # -> out_pooled/election_rewards*.csv

Market set: gamma tag `midterms`, open, in CLOB /rewards/markets/current with rate_per_day > 0; top N by rate.
Our quotes: bid and ask on the YES book at distance d = largest tick multiple < max_spread/2 from the
size-cutoff-adjusted mid, size = rewards_min_size each. Score per the CLOB formula
(analysis/rewards/reward_terms.py): S = ((v - d)/v)^2 * size, Q_min two-sided rule in [0.10, 0.90].
Competitors = the live book (YES token; the NO book is its mirror), price levels treated as orders, bracketed
'hi' (two-sided, conservative) and 'lo' (one-sided). Reward = rate_per_day * our share, averaged over snapshots.
Capital = size * (bid + 1 - ask) (YES bid collateral + NO bid collateral for the ask).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rewards"))
import reward_terms as RT  # noqa: E402

SNAP = HERE / "cache" / "election_books.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/126.0", "Accept": "application/json"}


def market_set(n: int) -> pd.DataFrame:
    m = pd.read_parquet(HERE / "cache" / "markets.parquet")
    m = m[m.tags.fillna("").str.contains(r"(^|\|)midterms(\||$)") & ~m.closed.astype(bool) & (m.outcome_idx == 0)]
    rw = pd.read_parquet(HERE / "cache" / "rewards_current.parquet")
    x = m.merge(rw, left_on="condition_id", right_on="condition_id")
    x = x[x.rate_per_day > 0].drop_duplicates("condition_id")
    x = x.sort_values(["rate_per_day", "volume_num"], ascending=False).head(n)
    return x[["token_id", "condition_id", "slug", "question", "rate_per_day", "min_size", "max_spread", "end_date",
              "volume_num"]].reset_index(drop=True)


def fetch(tok: str):
    s = requests.Session()
    s.headers.update(UA)
    for k in range(5):
        try:
            r = s.get("https://clob.polymarket.com/book", params={"token_id": tok}, timeout=30)
            if r.status_code == 200:
                b = r.json()
                return {"token_id": tok, "bids": [[float(o["price"]), float(o["size"])] for o in b.get("bids", [])],
                        "asks": [[float(o["price"]), float(o["size"])] for o in b.get("asks", [])],
                        "tick": float(b.get("tick_size") or 0.01)}
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1 + 2 ** k * 0.5)
    return None


def snap(a):
    ms = market_set(a.n)
    ms.to_parquet(HERE / "cache" / "election_markets.parquet", index=False)
    print(f"{len(ms)} markets, total rate ${ms.rate_per_day.sum():.0f}/day", flush=True)
    for k in range(a.k):
        t = time.time()
        with ThreadPoolExecutor(8) as ex:
            books = list(ex.map(fetch, ms.token_id))
        with open(SNAP, "a") as f:
            for b in books:
                if b:
                    b["ts"] = int(t)
                    f.write(json.dumps(b) + "\n")
        print(f"snapshot {k + 1}/{a.k}: {sum(b is not None for b in books)} books in {time.time() - t:.0f}s",
              flush=True)
        if k + 1 < a.k:
            time.sleep(max(0, a.every - (time.time() - t)))


def score_one(b, row, bound):
    cfg = replace(RT.RewardConfig(name="live"), reward_enabled=True, reward_min_size=float(row.min_size),
                  reward_max_spread_c=float(row.max_spread), competitor_bound=bound, tick=b["tick"])
    bids = sorted(b["bids"], key=lambda x: -x[0])
    asks = sorted(b["asks"], key=lambda x: x[0])
    if not bids or not asks:
        return None
    bp, bs = np.array([x[0] for x in bids]), np.array([x[1] for x in bids])
    ap, as_ = np.array([x[0] for x in asks]), np.array([x[1] for x in asks])
    mid = RT.adjusted_mid(bp, bs, ap, as_, cfg.reward_min_size)
    if not np.isfinite(mid):
        return None
    v = cfg.reward_max_spread_c
    tick_c = b["tick"] * 100
    d = np.floor((v / 2) / tick_c - 1e-9) * tick_c          # largest tick multiple < v/2 (in cents)
    if d <= 0:
        d = tick_c
    # place on the tick grid: bid = floor-ish below mid by >= d, ask above by >= d
    our_b = np.floor((mid - d / 100) / b["tick"] + 1e-9) * b["tick"]
    our_a = np.ceil((mid + d / 100) / b["tick"] - 1e-9) * b["tick"]
    if our_b <= 0 or our_a >= 1:
        return None
    size = cfg.reward_min_size
    o1, o2 = RT.side_scores([our_b], [size], [our_a], [size], mid, cfg)
    mine = RT.q_min(o1, o2, mid, cfg)
    c1, c2 = RT.side_scores(bp, bs, ap, as_, mid, cfg)
    comp = RT.competitor_q(c1, c2, mid, cfg)
    share = mine / (mine + comp) if mine > 0 else 0.0
    cap = size * (our_b + 1 - our_a)
    return dict(mid=mid, d_bid_c=(mid - our_b) * 100, d_ask_c=(our_a - mid) * 100, q_ours=mine, q_comp=comp,
                share=share, reward_day=row.rate_per_day * share, capital=cap,
                best_bid=bp[0], best_ask=ap[0], depth_in_v=float(((np.abs(mid - bp) * 100 < v) * bs).sum()
                                                                  + ((np.abs(ap - mid) * 100 < v) * as_).sum()))


BUDGETS = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000]


def greedy(mk, budget):
    """mk: list of (rate_per_day, q_per_lot[snaps], q_comp[snaps], capital_per_lot)."""
    import heapq
    k = np.zeros(len(mk))
    rew = lambda i, n: mk[i][0] * np.mean(n * mk[i][1] / np.maximum(n * mk[i][1] + mk[i][2], 1e-12))
    h = [(-(rew(i, 1) - 0) / mk[i][3], i) for i in range(len(mk)) if mk[i][3] > 0]
    heapq.heapify(h)
    spent = tot = 0.0
    while h:
        g, i = heapq.heappop(h)
        if spent + mk[i][3] > budget:
            continue
        spent += mk[i][3]
        tot += -g * mk[i][3]
        k[i] += 1
        heapq.heappush(h, (-(rew(i, k[i] + 1) - rew(i, k[i])) / mk[i][3], i))
    return dict(capital_used=spent, reward_usd_day=tot, yield_per_1k_day=1000 * tot / max(spent, 1e-9),
                markets_used=int((k > 0).sum()), max_lots=int(k.max()) if len(k) else 0)


def score(a):
    ms = pd.read_parquet(HERE / "cache" / "election_markets.parquet").set_index("token_id")
    rows = []
    for line in open(SNAP):
        b = json.loads(line)
        if b["token_id"] not in ms.index:
            continue
        row = ms.loc[b["token_id"]]
        for bound in ("hi", "lo"):
            r = score_one(b, row, bound)
            if r:
                rows.append(dict(token_id=b["token_id"], ts=b["ts"], bound=bound, rate_per_day=row.rate_per_day,
                                 min_size=row.min_size, max_spread=row.max_spread, **r))
    s = pd.DataFrame(rows)
    per = s.groupby(["bound", "token_id"]).agg(rate_per_day=("rate_per_day", "first"), min_size=("min_size", "first"),
                                               max_spread=("max_spread", "first"), mid=("mid", "mean"),
                                               share=("share", "mean"), reward_day=("reward_day", "mean"),
                                               capital=("capital", "mean"), depth_in_v=("depth_in_v", "mean"),
                                               n_snaps=("ts", "nunique")).reset_index()
    per["yield_per_1k_day"] = 1000 * per.reward_day / per.capital
    per = per.merge(ms[["slug"]], left_on="token_id", right_index=True)
    per.drop(columns="token_id").to_csv(HERE / "out_pooled" / "election_rewards_markets.csv", index=False,
                                        float_format="%.4g")
    rng = np.random.default_rng(0)
    out = []
    for bound, g in per.groupby("bound"):
        g = g.sort_values("yield_per_1k_day", ascending=False)
        for name, sub in [("all (equal size per market)", g), ("top 50 by yield", g.head(50)),
                          ("top 100 by yield", g.head(100))]:
            w = rng.poisson(1.0, (a.boot, len(sub)))
            y = 1000 * (w @ sub.reward_day.to_numpy()) / np.maximum(w @ sub.capital.to_numpy(), 1e-9)
            out.append(dict(bound=bound, portfolio=name, markets=len(sub), capital_usd=sub.capital.sum(),
                            reward_usd_day=sub.reward_day.sum(), yield_per_1k_day=1000 * sub.reward_day.sum()
                            / sub.capital.sum(), lo=np.percentile(y, 2.5), hi=np.percentile(y, 97.5),
                            median_share=sub.share.median(), median_mkt_yield=sub.yield_per_1k_day.median()))
    o = pd.DataFrame(out)
    o.to_csv(HERE / "out_pooled" / "election_rewards.csv", index=False, float_format="%.4g")
    # capital curve: allocate min-size lots greedily by marginal reward per $ (same quote distance; k lots in one
    # market score k*q_ours, so share = k q / (k q + comp), averaged over snapshots); bootstrap over markets
    curve = []
    # 'contested' drops markets whose scoring band held < min_size of competing depth on average (an empty
    # band pays the whole pool to whoever quotes first, and invites competitors / informed takers)
    contested = set(per[(per.bound == "hi") & (per.depth_in_v >= per.min_size)].token_id)
    for subset in ("all", "contested"):
        for bound in ("hi", "lo"):
            sb = s[(s.bound == bound) & ((subset == "all") | s.token_id.isin(contested))]
            mk = [(g.rate_per_day.iloc[0], g.q_ours.to_numpy(), g.q_comp.to_numpy(), g.capital.mean())
                  for _, g in sb.groupby("token_id")]
            for rep in range(a.curve_boot + 1):
                idx = np.arange(len(mk)) if rep == 0 else rng.integers(0, len(mk), len(mk))
                sub = [mk[i] for i in idx]
                for B in BUDGETS:
                    curve.append(dict(subset=subset, bound=bound, rep=rep, budget=B, **greedy(sub, B)))
    c = pd.DataFrame(curve)
    keys = ["subset", "bound", "budget"]
    pt = c[c.rep == 0].set_index(keys)
    ci = c[c.rep > 0].groupby(keys).yield_per_1k_day.quantile([0.025, 0.975]).unstack()
    pt = pt.join(ci.rename(columns={0.025: "lo", 0.975: "hi"})).reset_index().drop(columns="rep")
    pt.to_csv(HERE / "out_pooled" / "election_rewards_curve.csv", index=False, float_format="%.4g")
    print(pt.round(3).to_string())
    pd.set_option("display.width", 250)
    print(o.round(3).to_string())
    print(per[per.bound == "hi"].describe().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["snap", "score"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--curve-boot", type=int, default=200)
    a = ap.parse_args()
    snap(a) if a.mode == "snap" else score(a)
