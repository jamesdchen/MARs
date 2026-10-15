#!/usr/bin/env python3
"""Pick the US midterm election markets with the largest liquidity-reward pools (DECISIONS.md D11 track E)
and write a polyrec config that records exactly them.

    python select_markets.py [--top 120] [--min-rate 1] [--out sel]   # -> sel/selection.json, sel/polyrec.toml

1. CLOB GET /rewards/markets/current, all pages: condition_id -> total_daily_rate, min size, max spread.
2. gamma GET /events/keyset for every election tag (census/categories.py ELEC) with closed=false: open
   binary markets with order books; a market is "US midterm" if its event carries a midterm tag
   (midterms, senate-midterms, house-elections, governor-midterms, midtermmov, nov-4-elections) or its
   end date is 2026-11-03..2026-12-31 under an election tag.
3. Join, rank by total_daily_rate, keep the top N with rate >= --min-rate.
4. polyrec config: one gamma source per tag query with window=end_date and a long horizon, and a
   slug_regex that is the exact alternation of the selected slugs (the recorder then subscribes to
   exactly these markets, and keeps them while open).
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
ELEC_TAGS = ["midterms", "senate-midterms", "house-elections", "governor-midterms", "midtermmov",
             "nov-4-elections", "elections"]
MIDTERM_TAGS = {"midterms", "senate-midterms", "house-elections", "governor-midterms", "midtermmov",
                "nov-4-elections"}
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) Chrome/126.0"


def get(url, params=None):
    for k in range(7):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"GET {url} {params} failed")


def rewards_current():
    rows, cur = {}, ""
    while True:
        d = get(f"{CLOB}/rewards/markets/current", {"next_cursor": cur} if cur else None)
        for m in d.get("data", []):
            rows[m["condition_id"]] = m
        cur = d.get("next_cursor") or ""
        if not cur or cur == "LTE=" or not d.get("data"):
            return rows


def jl(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def election_markets():
    out = {}
    for tag in ELEC_TAGS:
        cur, pages = None, 0
        while pages < 60:
            q = {"tag_slug": tag, "closed": "false", "limit": 100}
            if cur:
                q["after_cursor"] = cur
            d = get(f"{GAMMA}/events/keyset", q)
            pages += 1
            evs = d.get("events", [])
            for ev in evs:
                etags = {t.get("slug") for t in ev.get("tags") or [] if isinstance(t, dict)} | {tag}
                for m in ev.get("markets") or []:
                    if m.get("closed") or m.get("enableOrderBook") is False or not m.get("conditionId"):
                        continue
                    toks, outs = jl(m.get("clobTokenIds")), jl(m.get("outcomes"))
                    if len(toks) != 2:
                        continue
                    cid = m["conditionId"]
                    rec = out.setdefault(cid, {
                        "condition_id": cid, "slug": m.get("slug"), "question": m.get("question"),
                        "event_slug": ev.get("slug"), "event_title": ev.get("title"), "end_date": m.get("endDate"),
                        "tokens": toks, "outcomes": outs, "tags": set(),
                        "volume24hr": m.get("volume24hr"), "liquidity": m.get("liquidity"),
                        "rewards_min_size": m.get("rewardsMinSize"), "rewards_max_spread": m.get("rewardsMaxSpread"),
                        "neg_risk": m.get("negRisk")})
                    rec["tags"] |= etags
            cur = d.get("next_cursor")
            if not cur or not evs:
                break
    for r in out.values():
        r["tags"] = sorted(t for t in r["tags"] if t)
        ed = r["end_date"] or ""
        r["midterm"] = bool(set(r["tags"]) & MIDTERM_TAGS) or ("2026-11-03" <= ed[:10] <= "2026-12-31")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=120)
    p.add_argument("--min-rate", type=float, default=1.0)
    p.add_argument("--out", default=str(HERE / "sel"))
    p.add_argument("--markets-per-conn", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rw = rewards_current()
    em = election_markets()
    rows = []
    for cid, m in em.items():
        r = rw.get(cid)
        if not m["midterm"] or r is None:
            continue
        rows.append({**m, "rate_per_day": float(r.get("total_daily_rate") or 0),
                     "native_rate": float(r.get("native_daily_rate") or 0),
                     "clob_min_size": r.get("rewards_min_size"), "clob_max_spread": r.get("rewards_max_spread"),
                     "rewards_config": r.get("rewards_config")})
    rows.sort(key=lambda r: -r["rate_per_day"])
    pool_all = sum(r["rate_per_day"] for r in rows)
    sel = [r for r in rows if r["rate_per_day"] >= a.min_rate][:a.top]
    snap = {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_reward_markets": len(rw),
            "n_election_open": len(em), "n_midterm_with_rewards": len(rows),
            "midterm_pool_per_day": pool_all, "selected_pool_per_day": sum(r["rate_per_day"] for r in sel),
            "selected": sel, "all_midterm_with_rewards": [{k: r[k] for k in ("condition_id", "slug", "rate_per_day")}
                                                           for r in rows]}
    (out / "selection.json").write_text(json.dumps(snap, indent=1))
    slugs = "|".join(re.escape(r["slug"]) for r in sel)
    toml = [f'# generated by select_markets.py at {snap["taken_at"]}: top {len(sel)} midterm markets by reward rate',
            'discovery_interval_s = 300', '',
            '[ws]', f'markets_per_conn = {a.markets_per_conn}', '',
            '[output]', 'raw_enabled = true', 'raw_retention_days = 0', '']
    tags = sorted({t for r in sel for t in r["tags"] if t in ELEC_TAGS})
    for t in tags:
        toml += ['[[source]]', f'name = "elec-{t}"', f'params = {{ tag_slug = "{t}" }}',
                 f"slug_regex = '^(?:{slugs})$'", 'window = "end_date"', 'horizon_s = 40000000',
                 'pre_open_s = 0', 'linger_s = 600', 'max_pages = 60', '']
    (out / "polyrec.toml").write_text("\n".join(toml))
    print(f"reward markets {len(rw)}, open election markets {len(em)}, midterm with rewards {len(rows)}, "
          f"pool ${pool_all:,.0f}/day; selected {len(sel)} = ${snap['selected_pool_per_day']:,.0f}/day; tags {tags}")
    for r in sel[:15]:
        print(f"  {r['rate_per_day']:8.0f}  {r['slug'][:90]}")


if __name__ == "__main__":
    main()
