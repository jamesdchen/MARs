"""PolyOrderBooks backfill under a small daily quota (Starter: 60 req/min, ~1k req/day, 3-day history).

Thin wrapper around ../v3_sports/backfill_pob.py (w1's script, imported, not edited). It adds three
things that the probe of 2026-10-16 showed are needed:

1. **Scan-window pagination fix.** The books endpoint scans `limit` buckets from `start_ts` and returns
   only the filled ones; `next_cursor` is set only when the page is full. A range that starts before the
   book exists (or spans a feed gap) therefore comes back short with no cursor and the rest of the range
   is silently lost (e.g. 1m over 02:00-12:00 returned only 04:05-05:19). `token_books` here continues
   from `start + limit * res` until `end_ts` whenever a page is short and has no cursor.
2. **Data-start detection.** One 1h request over the market window (clipped to the plan's history
   floor) finds the first hour with a book, and one 1m request finds the minute. The books window then
   starts there, so no requests are spent on empty pre-listing chunks. A market with no data at all
   (atp/wta/mlb are not in the PolyOrderBooks archive) is marked MISSING after 1 request.
3. **Request budget per run** (`--budget`), spent in priority order: oldest kickoff day first (it
   leaves the rolling history window first), highest gamma volume first within a day. Markets are
   admitted only while their estimated cost fits the remaining budget.

Every request window is clipped to the plan's history floor, so nothing 403s on age.
The API key is read only from POLYORDERBOOKS_API_KEY and is never printed or written.

    python backfill_starter.py --budget 900 --fine-hours 1          # resumable; re-run daily
    python backfill_starter.py --budget 0 --dry-run                  # plan only (gamma + no POB requests)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3_sports"))
import backfill_pob as bp  # noqa: E402

log = bp.log
FLOOR = {"t": None}


def token_books(self, token_id, a, b, res, limit=bp.BOOK_PAGE):
    """All points of /v1/tokens/{id}/books over [a, b), robust to short pages without a cursor."""
    res_s = bp.res_seconds(res)
    if FLOOR["t"] is not None:
        a = max(a, FLOOR["t"])
    pts, pages, meta, cursor, start = [], 0, {}, None, a
    pos = a                       # where the server's scan for the current page began (cursor pages: after last t)
    last = None                   # last bucket time seen
    while True:
        d = self.get(f"/v1/tokens/{token_id}/books",
                     {"start_ts": int(start), "end_ts": int(math.ceil(b)), "resolution": res, "limit": limit,
                      "cursor": cursor})
        pages += 1
        got = d.get("data") or []
        pts += got
        if got:
            last = bp.parse_ts(got[-1]["t"])
        meta = {k: d.get(k) for k in ("market_id", "token_id", "token_label", "resolution")}
        cursor = (d.get("metadata") or {}).get("next_cursor")
        if cursor:
            pos = last + res_s if last is not None else pos + limit * res_s
            continue
        # no cursor: the server scanned [pos, pos + limit*res); resume after the scanned span
        nxt = pos + limit * res_s
        if last is not None:
            nxt = max(nxt, last + res_s)
        if nxt >= b:
            break
        start = pos = int(math.ceil(nxt))
    seen, uniq = set(), []
    for p in pts:                                     # defensive: one point per bucket
        if p["t"] not in seen:
            seen.add(p["t"])
            uniq.append(p)
    return uniq, pages, meta


bp.POB.token_books = token_books


def stop_on_429(sess):
    """We run below the per-minute limit, so a 429 means the daily quota: stop at once (the base client
    would back off for up to ~2 h first). Logs the message and rate-limit headers (never the key)."""
    raw_get = sess.get

    def get(*a, **k):
        r = raw_get(*a, **k)
        if r.status_code == 429:
            hdr = {h: v for h, v in r.headers.items() if "rate" in h.lower() or "retry" in h.lower() or "reset" in h.lower()}
            log.error("429 from PolyOrderBooks: %s headers=%s", r.text[:300], hdr)
            raise bp.BudgetExhausted("429 (daily quota)")
        return r
    sess.get = get


def est_cost(ws, ko, fine_h):
    """Requests for one market (mirror, no seed): 1m pages (200 min each) + 1s pages (19 per hour)."""
    fine_from = max(ws, ko - fine_h * 3600)
    coarse = max(0, fine_from - ws)
    nfine = math.ceil(max(0, ko - fine_from) / 3600)                    # 1s chunks are hour-aligned
    return math.ceil(coarse / (3 * 3600)) + 1 + 19 * nfine


def detect_start(pob, m, out: Path):
    """First minute with a book in [max(ws, floor), kickoff). Cached in _state/start/<slug>.json."""
    sp = out / "_state" / "start" / f"{m['slug']}.json"
    if sp.exists():
        return json.loads(sp.read_text()), 0
    n0 = pob.n
    lo = max(m["ws"], FLOOR["t"] or 0)
    p, _, _ = pob.token_books(m["tokens"][0], lo, m["kickoff"], "1h")
    st = {"lo": lo, "first": None, "at": int(time.time())}
    if p:
        h = bp.parse_ts(p[0]["t"])
        q, _, _ = pob.token_books(m["tokens"][0], max(lo, h - 60), min(m["kickoff"], h + 3600), "1m")
        st["first"] = int(bp.parse_ts(q[0]["t"])) if q else int(h)
    # data present before the floor => the book is older than our window: mark it
    st["clipped_by_floor"] = bool(p) and st["first"] <= lo + 60 and lo > m["ws"]
    bp.atomic_json(sp, st)
    return st, pob.n - n0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "raw_pob"))
    ap.add_argument("--start", help="first kickoff day (default: the plan's history floor day)")
    ap.add_argument("--end", help="last kickoff day (default: today, UTC)")
    ap.add_argument("--budget", type=int, default=900, help="max PolyOrderBooks requests this run")
    ap.add_argument("--fine-hours", type=float, default=1.0)
    ap.add_argument("--min-volume", type=float, default=50000)
    ap.add_argument("--leagues", default=",".join(bp.LEAGUES))
    ap.add_argument("--skip-leagues", default="atp,wta,mlb", help="not in the PolyOrderBooks archive (probe)")
    ap.add_argument("--rpm", type=float, default=45)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = os.environ.get(bp.ENV_KEY)
    out = Path(a.out)
    bp.setup_logging(out, "INFO", key)
    if not key:
        sys.exit(f"{bp.ENV_KEY} is not set")
    pob = bp.POB(key, rpm=a.rpm, max_requests=a.budget or 0, category="sports")
    stop_on_429(pob.s)
    try:
        u = pob.usage()
    except bp.BudgetExhausted:
        sys.exit("daily quota used up (429 on /v1/usage); try later")
    lim = u.get("limits") or {}
    hd = lim.get("max_history_days") or 3
    FLOOR["t"] = int(time.time() - hd * 86400) + 900                    # 15 min margin for the run's length
    today = time.strftime("%Y-%m-%d", time.gmtime())
    start = a.start or time.strftime("%Y-%m-%d", time.gmtime(FLOOR["t"]))
    end = a.end or today
    log.info("plan=%s rpm=%s history_days=%s floor=%s window %s..%s budget=%d fine_hours=%s", u.get("plan"),
             lim.get("requests_per_minute"), hd, bp.iso(FLOOR["t"]), start, end, a.budget, a.fine_hours)
    skip = set(x for x in a.skip_leagues.split(",") if x)
    sargs = argparse.Namespace(start=start, end=end, out=str(out), market_types=["moneyline"],
                               leagues=[x for x in a.leagues.split(",") if x not in skip], hours_before=168.0,
                               post_min=0.0, min_volume=a.min_volume, order="time", max_markets=0)
    ms = bp.select_markets(sargs)
    ms = [m for m in ms if m["kickoff"] - 600 > FLOOR["t"]]
    # one market per game first (soccer has 3 Yes/No legs per game), then the other legs
    ev_rank, ms = {}, sorted(ms, key=lambda m: -m["volume"])
    for m in ms:
        ev = m["g"].get("eventSlug") or m["slug"]
        m["ev_rank"] = ev_rank[ev] = ev_rank.get(ev, -1) + 1
    ms.sort(key=lambda m: (m["kickoff"] // 86400, m["ev_rank"] > 0, -m["volume"]))
    log.info("selected %d markets (vol >= %.0f, kickoff inside history)", len(ms), a.min_volume)
    bargs = argparse.Namespace(out=str(out), mirror=True, no_seed=True, no_shift=False, keep_unchanged=False,
                               seed_hours=6, resolution="1s", fine_hours=a.fine_hours, coarse_resolution="1m",
                               retry_missing=False, retry_forbidden=False, workers=a.workers,
                               _history_floor=FLOOR["t"], trades_full=False, meta_workers=3)
    plan, left = [], a.budget - pob.n
    for m in ms:
        sd = bp.book_state_dir(out, m["slug"])
        if (sd / "MISSING.json").exists():
            continue
        if a.dry_run:
            plan.append(m)
            continue
        if left < 25:
            break
        try:
            st, used = detect_start(pob, m, out)
        except bp.BudgetExhausted:
            break
        except bp.ApiError as e:
            log.warning("start detection %s: %s", m["slug"], e)
            if e.status in (401, 429):
                break
            continue
        left -= used
        if st["first"] is None:
            bp.atomic_json(sd / "MISSING.json", {"error": "no book in window (1h scan)", "at": int(time.time())})
            continue
        m["ws_gamma"] = m["ws"]
        m["ws"] = max(m["ws"], -(-max(st["first"], FLOOR["t"]) // 60) * 60)   # ceil: never below the floor
        todo = [c for c in bp.chunks_for(m, bargs) if not (sd / f"{c[0]}_{c[2]}.json").exists()]
        if not todo:
            continue
        cost = est_cost(todo[0][0], m["kickoff"], a.fine_hours)
        if cost > left:
            continue                                                     # a later, cheaper market may fit
        left -= cost
        plan.append(m)
    log.info("plan: %d markets, est. %d book requests left after plan; leagues %s", len(plan), left,
             {lg: sum(1 for m in plan if m["league"] == lg) for lg in sorted({m["league"] for m in plan})})
    if a.dry_run:
        for m in plan[:50]:
            print(m["slug"], bp.iso(m["kickoff"]), round(m["volume"]))
        return
    meta_ms = [dict(m, ws=m.get("ws_gamma", m["ws"])) for m in plan]      # trades over the full gamma window
    summ = {"meta": bp.run_meta(bargs, meta_ms)}
    summ["books"] = bp.run_books(bargs, plan, pob, bp.BOOK_SCHEMA_RAW)
    bp.compact(out)
    summ["requests"] = pob.n
    rec = {"at": bp.iso(time.time()), "plan": u.get("plan"), "floor": bp.iso(FLOOR["t"]), "start": start,
           "end": end, "markets": [m["slug"] for m in plan], "summary": summ}
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    log.info("run done: %s", summ)


if __name__ == "__main__":
    main()
