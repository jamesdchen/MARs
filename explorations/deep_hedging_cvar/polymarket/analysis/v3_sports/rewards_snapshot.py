#!/usr/bin/env python3
"""Daily snapshot of the CLOB liquidity-reward programs of the RECORDED sports markets.

    python rewards_snapshot.py                      # -> rewards/rewards_<UTC day>.json (git-tracked, small)
    python rewards_snapshot.py --all                # keep every market in a program (large; not for git)

Fetches https://clob.polymarket.com/rewards/markets/current (every page; the code of
../census/rewards_current.py) and keeps only the markets whose condition_id belongs to a recorded
market. /current carries no slug and no tags, so "recorded" is decided by condition_id, collected from:
  - cache/gamma/<slug>.json (build_sports.py's gamma cache: conditionId),
  - markets/ tables of rec_out / rec_daily / raw_pob (condition_id),
  - the slugs of rec_out/coverage, rec_daily/*/coverage_*.jsonl and kickoff/meta.json, resolved to a
    condition_id through gamma /markets?slug=..&slug=.. (20 per request; cached in cache/gamma_cond.json).
Since the recorder only records sports (recorder_pregame.toml), this set is the "sports-tagged" one.

A second run on the same UTC day merges into that day's file (markets seen earlier are kept, those
seen again are updated), so running it a few times a day catches games listed later.

Output (one file per UTC day; build_sports.py --rewards-dir reads them):
  {"day", "taken_at": [iso...], "source", "n_markets_total", "usd_day_total",
   "n_kept", "usd_day_kept", "filter",
   "markets": {condition_id: {"slug", "rate_per_day", "max_spread", "min_size", "start_date", "end_date"}}}
rate_per_day = total_daily_rate (USD/day, all assets), max_spread in cents, min_size in shares.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "https://clob.polymarket.com/rewards/markets/current"
GAMMA = "https://gamma-api.polymarket.com/markets"
UA = "Mozilla/5.0 (X11; Linux x86_64) Chrome/126.0"


def get(s: requests.Session, url: str, params=None):
    for k in range(6):
        try:
            r = s.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"GET {url} {params} failed")


def fetch_current(s: requests.Session) -> list[dict]:
    rows, cur = [], ""
    while True:
        d = get(s, URL, {"next_cursor": cur} if cur else None)
        for m in d.get("data", []):
            rc = m.get("rewards_config") or [{}]
            rows.append({"condition_id": m["condition_id"],
                         "rate_per_day": float(m.get("total_daily_rate") or 0),
                         "max_spread": m.get("rewards_max_spread"), "min_size": m.get("rewards_min_size"),
                         "start_date": min((c.get("start_date") or "") for c in rc),
                         "end_date": max((c.get("end_date") or "") for c in rc)})
        cur = d.get("next_cursor") or ""
        if not cur or cur == "LTE=" or not d.get("data"):
            break
        time.sleep(0.2)
    return list({r["condition_id"]: r for r in rows}.values())


def recorded(recs: list[Path], s: requests.Session, offline: bool) -> dict[str, str]:
    """{condition_id: slug} of the recorded markets."""
    cond2slug, slugs = {}, set()
    for f in glob.glob(str(HERE / "cache" / "gamma" / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if d.get("conditionId"):
            cond2slug[d["conditionId"]] = d.get("slug") or Path(f).stem
    try:
        import pyarrow.parquet as pq
        for r in recs:
            for f in list(Path(r).glob("markets/**/*.parquet")) + list(Path(r).glob("*/markets_*.parquet")):
                try:
                    t = pq.read_table(f, columns=["slug", "condition_id"]).to_pylist()
                except Exception:  # noqa: BLE001  (torn part of a live recorder)
                    continue
                for row in t:
                    if row["condition_id"]:
                        cond2slug[row["condition_id"]] = row["slug"]
    except ImportError:
        pass
    for r in recs:
        for f in list(Path(r).glob("coverage/coverage_*.jsonl")) + list(Path(r).glob("*/coverage_*.jsonl")):
            for ln in open(f, "rb"):
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                if e.get("slug"):
                    slugs.add(e["slug"])
    km = HERE / "kickoff" / "meta.json"
    if km.exists():
        slugs |= set(json.loads(km.read_text()))
    cache_p = HERE / "cache" / "gamma_cond.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    known = set(cond2slug.values())
    need = sorted(x for x in slugs - known if x not in cache)
    if need and not offline:
        for i in range(0, len(need), 20):
            b = need[i:i + 20]
            try:
                res = get(s, GAMMA, [("slug", x) for x in b])
            except RuntimeError:
                continue
            for g in res or []:
                if g.get("slug") and g.get("conditionId"):
                    cache[g["slug"]] = g["conditionId"]
            for x in b:
                cache.setdefault(x, None)
            time.sleep(0.2)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(cache, sort_keys=True))
    for sl in slugs:
        if cache.get(sl):
            cond2slug.setdefault(cache[sl], sl)
    return cond2slug


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", action="append", help="recorder dirs whose markets count as recorded "
                    "(default: rec_out, rec_daily, raw_pob)")
    ap.add_argument("--out", default=str(HERE / "rewards"))
    ap.add_argument("--all", action="store_true", help="keep every market in a program, not only recorded ones")
    ap.add_argument("--offline", action="store_true", help="no gamma lookups for unresolved slugs")
    a = ap.parse_args(argv)
    recs = [Path(x) for x in (a.rec or [HERE / "rec_out", HERE / "rec_daily", HERE / "raw_pob"])]
    s = requests.Session()
    s.headers["User-Agent"] = UA
    now = time.time()
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    rows = fetch_current(s)
    total = sum(r["rate_per_day"] for r in rows)
    keep = None if a.all else recorded(recs, s, a.offline)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rewards_{day}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    mk = dict(old.get("markets", {}))
    for r in rows:
        if keep is not None and r["condition_id"] not in keep:
            continue
        mk[r["condition_id"]] = {"slug": (keep or {}).get(r["condition_id"]), "rate_per_day": r["rate_per_day"],
                                 "max_spread": r["max_spread"], "min_size": r["min_size"],
                                 "start_date": r["start_date"], "end_date": r["end_date"]}
    doc = {"day": day,
           "taken_at": old.get("taken_at", []) + [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))],
           "source": URL, "n_markets_total": len(rows), "usd_day_total": round(total, 2),
           "n_kept": len(mk), "usd_day_kept": round(sum(v["rate_per_day"] for v in mk.values()), 2),
           "filter": "all markets" if a.all else f"recorded markets ({len(keep)} condition_ids known)",
           "markets": dict(sorted(mk.items()))}
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=0))
    os.replace(tmp, path)
    print(f"{path}: {len(rows)} markets in programs (${total:,.0f}/day); kept {len(mk)} "
          f"(${doc['usd_day_kept']:,.0f}/day); {doc['filter']}")
    return doc


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
