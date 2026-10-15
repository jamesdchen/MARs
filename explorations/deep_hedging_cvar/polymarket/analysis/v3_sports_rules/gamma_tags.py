#!/usr/bin/env python3
"""token_id -> market metadata (tags, fee schedule, rewards config, game start, resolution) from gamma.

  python gamma_tags.py --fills ../../onchain/data/all/fills --out cache/markets.parquet

Batched `/markets?clob_token_ids=..&closed=true|false&include_tag=true`, cached in cache/gamma_tags.sqlite
(resumable). Writes one row per token.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "application/json"}
KEEP = ["id", "slug", "question", "conditionId", "clobTokenIds", "outcomes", "outcomePrices", "closed", "closedTime",
        "startDate", "endDate", "createdAt", "gameStartTime", "sportsMarketType", "feeSchedule", "feeType",
        "feesEnabled", "rewardsMinSize", "rewardsMaxSpread", "holdingRewardsEnabled", "negRisk",
        "umaResolutionStatus", "volumeNum", "eventStartTime", "category"]
BATCH = 40
HERE = Path(__file__).resolve().parent
_tl = threading.local()


def trim(m: dict) -> dict:
    r = {k: m.get(k) for k in KEEP}
    r["tags"] = [t.get("slug") for t in (m.get("tags") or [])]
    ev = (m.get("events") or [{}])[0]
    r["ev_slug"] = ev.get("slug")
    r["ev_title"] = ev.get("title")
    r["ev_start"] = ev.get("startTime")
    r["ev_series"] = ev.get("seriesSlug")
    r["ev_end"] = ev.get("endDate")
    return r


def get(params):
    s = getattr(_tl, "s", None)
    if s is None:
        s = _tl.s = requests.Session()
        s.headers.update(UA)
    d = 1.0
    for _ in range(5):  # w2 copy: shorter retries so one bad batch cannot stall the run
        try:
            r = s.get(GAMMA, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(d + random.random())
        d = min(d * 2, 8)
    raise RuntimeError("gamma failed")


def fetch(batch):
    out = {}
    for closed in ("true", "false"):
        rem = [t for t in batch if t not in out]
        if not rem:
            break
        q = [("clob_token_ids", t) for t in rem] + [("closed", closed), ("include_tag", "true"),
                                                    ("limit", str(2 * len(rem) + 10))]
        try:
            res = get(q)
        except RuntimeError:
            return batch, None
        for m in res:
            tm = trim(m)
            for t in json.loads(m.get("clobTokenIds") or "[]"):
                out[t] = tm
    return batch, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", required=True)
    ap.add_argument("--out", default=str(HERE / "cache" / "markets.parquet"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (HERE / "cache").mkdir(exist_ok=True)
    db = sqlite3.connect(HERE / "cache" / "gamma_tags.sqlite", check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS tok(token_id TEXT PRIMARY KEY, raw TEXT)")
    toks = [r[0] for r in duckdb.sql(
        f"select distinct token_id from read_parquet('{a.fills}/*/*.parquet', hive_partitioning=1)").fetchall()]
    have = {r[0] for r in db.execute("select token_id from tok")}
    todo = [t for t in toks if t not in have]
    print(f"{len(toks)} tokens, {len(todo)} to fetch", flush=True)
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(fetch, b) for b in batches]):
            batch, out = fut.result()
            if out is None:
                print("  batch failed; rerun to retry", flush=True)
                continue
            rows = [(t, json.dumps(out.get(t))) for t in batch]   # 'null' marks not found
            rows += [(t, json.dumps(m)) for t, m in out.items() if t not in set(batch)]
            db.executemany("INSERT OR REPLACE INTO tok VALUES(?,?)", rows)
            db.commit()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(batches)} batches {time.time() - t0:.0f}s", flush=True)
    rows = []
    want = set(toks)
    for t, raw in db.execute("select token_id, raw from tok"):
        if t not in want:
            continue
        m = json.loads(raw)
        if m is None:
            rows.append({"token_id": t})
            continue
        ids = json.loads(m.get("clobTokenIds") or "[]")
        outs = json.loads(m.get("outcomes") or "[]")
        px = json.loads(m.get("outcomePrices") or "[]")
        i = ids.index(t) if t in ids else -1
        fs = m.get("feeSchedule") or {}
        rows.append({
            "token_id": t, "slug": m["slug"], "condition_id": m.get("conditionId"), "question": m.get("question"),
            "outcome": outs[i] if 0 <= i < len(outs) else None, "outcome_idx": i,
            "final_price": float(px[i]) if 0 <= i < len(px) and m.get("closed") else None,
            "closed": bool(m.get("closed")), "closed_time": m.get("closedTime"), "end_date": m.get("endDate"),
            "created_at": m.get("createdAt"), "start_date": m.get("startDate"),
            "game_start": m.get("gameStartTime"), "event_start_time": m.get("eventStartTime"),
            "ev_start": m.get("ev_start"), "sports_type": m.get("sportsMarketType"),
            "fee_type": m.get("feeType"), "fee_rate": fs.get("rate"), "fee_exp": fs.get("exponent"),
            "rebate_rate": fs.get("rebateRate"), "fees_enabled": m.get("feesEnabled"),
            "rewards_min_size": m.get("rewardsMinSize"), "rewards_max_spread": m.get("rewardsMaxSpread"),
            "holding_rewards": m.get("holdingRewardsEnabled"), "neg_risk": m.get("negRisk"),
            "uma_status": m.get("umaResolutionStatus"), "volume_num": m.get("volumeNum"),
            "tags": "|".join(x for x in m.get("tags") or [] if x), "ev_slug": m.get("ev_slug"),
            "ev_title": m.get("ev_title"), "ev_series": m.get("ev_series"), "category_raw": m.get("category"),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} tokens, {df['slug'].notna().sum()} resolved", flush=True)


if __name__ == "__main__":
    main()
