#!/usr/bin/env python3
"""COVERAGE.md: day x league table of pre-game sports moneylines in the Joseph3222 archive with >= 1 h of book
before kickoff. Built WITHOUT reading any price data, so the v3 sports split can be pre-registered from it.

planned (default): gamma selection (convert.py select cache: moneyline, closed, two tokens, volume >= $10k)
    x archive availability: the archive day exists and every archive hour overlapping [kickoff - 1 h, kickoff)
    is present (hour presence from the day parquet's footer statistics only; no rows are read).
    v1-era days (2026-02-22..04-12) hold only ~half of all markets, so their counts are upper bounds.
measured (--measured): after the ingest, a market counts when its verified (anchored) book covers >= 1 h of
    [kickoff - 1 h... window] per raw_hf/_state/qc (live seconds inside the book window, summed over days).

    python coverage.py --start 2026-02-22 --end 2026-08-10            # footers cached in cache/archive_hours.json
    python coverage.py --measured                                      # adds the measured table
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert as cv  # noqa: E402
import hfpq  # noqa: E402

V2_FIRST = "2026-04-13"


def archive_hours(a) -> dict:
    p = a.cache / "archive_hours.json"
    have = json.loads(p.read_text()) if p.exists() else {}
    for day in cv.days(cv.ARCHIVE_FIRST, cv.ARCHIVE_LAST):
        if day in have:
            continue
        if day in cv.ARCHIVE_MISSING:
            have[day] = []
            continue
        loc = a.dl / f"footer_{day}.parquet"
        try:
            sp = hfpq.SparseParquet(f"orderbook/date={day}/data_0.parquet", str(loc))
        except Exception as e:  # noqa: BLE001
            print(f"{day}: no file ({e})")
            have[day] = []
            continue
        lo = cv.d0(day)
        hs = sorted({h for h, _, _ in cv.rg_hour_stats(sp) if h is not None and lo <= h < lo + 86400})
        have[day] = [(h - lo) // 3600 for h in hs]
        sp.close()
        print(day, len(have[day]), "hours", flush=True)
        p.write_text(json.dumps(have))
    return have


def planned(a, have) -> tuple[dict, Counter]:
    present = {cv.d0(d) + 3600 * h for d, hs in have.items() for h in hs}
    cells, lg = defaultdict(Counter), Counter()
    for kd in cv.days(a.start, a.end):
        for m in cv.selection(a, kd):
            ko = m["kickoff"]
            hours = range((ko - 3600) // 3600 * 3600, (ko - 1) // 3600 * 3600 + 1, 3600)
            if all(h in present for h in hours):
                cells[kd][m["league"]] += 1
                lg[m["league"]] += 1
    return cells, lg


def measured(a) -> tuple[dict, Counter]:
    live = defaultdict(float)
    for f in sorted((a.raw / "_state" / "qc").glob("*.json.gz")):
        with gzip.open(f, "rt") as fh:
            for slug, q in json.load(fh).items():
                live[slug] += q.get("live_s", 0)
    cells, lg = defaultdict(Counter), Counter()
    for kd in cv.days(a.start, a.end):
        for m in cv.selection(a, kd):
            if live.get(m["slug"], 0) >= 3600:
                cells[kd][m["league"]] += 1
                lg[m["league"]] += 1
    return cells, lg


def table(cells, lg, have, ncol=26) -> list[str]:
    cols = [l for l, _ in lg.most_common(ncol)]
    out = ["| day | era | archive h | " + " | ".join(cols) + " | other | total |",
           "|" + "---|" * (len(cols) + 5)]
    tot = Counter()
    for kd in sorted(set(cells) | set(have)):
        c = cells.get(kd, Counter())
        era = "absent" if kd in cv.ARCHIVE_MISSING or not have.get(kd) else ("v1~half" if kd < V2_FIRST else "v2")
        other = sum(v for k, v in c.items() if k not in cols)
        row = [kd, era, str(len(have.get(kd, [])))] + [str(c.get(l, 0) or "") for l in cols] + \
              [str(other or ""), str(sum(c.values()))]
        tot.update(c)
        out.append("| " + " | ".join(row) + " |")
    other = sum(v for k, v in tot.items() if k not in cols)
    out.append("| **total** | | | " + " | ".join(str(tot.get(l, 0)) for l in cols) +
               f" | {other} | {sum(tot.values())} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cv.ARCHIVE_FIRST)
    ap.add_argument("--end", default=cv.ARCHIVE_LAST)
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--raw", type=Path, default=HERE / "raw_hf")
    ap.add_argument("--dl", type=Path, default=HERE / "dl")
    ap.add_argument("--out", type=Path, default=HERE / "COVERAGE.md")
    ap.add_argument("--measured", action="store_true")
    ap.add_argument("--min-volume", type=float, default=10000)
    a = ap.parse_args()
    # selection args convert.selection expects
    a.markets, a.leagues, a.market_types, a.hours_before, a.post_min = None, "all", "moneyline", 48, 5
    a.dl.mkdir(parents=True, exist_ok=True)
    have = archive_hours(a)
    cells, lg = planned(a, have)
    lines = ["# Joseph3222 archive: pre-game sports moneyline coverage (Track S, w10)", "",
             f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC by `coverage.py`. **No price data was "
             "read to build the planned table** (gamma metadata + archive hour presence from parquet footers), so "
             "the v3 sports split can be pre-registered from it.", "",
             "Market = gamma sports \"Games\" moneyline (`sportsMarketType` = moneyline), closed, two tokens, "
             f"gamma volume >= ${a.min_volume:,.0f}, all leagues (league = slug prefix), counted on its kickoff "
             "(`gameStartTime`) UTC day. Soccer moneylines are three Yes/No markets per game (home / draw / away).", "",
             "**>= 1 h of book (planned)** = the archive day exists and every archive hour overlapping "
             "[kickoff - 1 h, kickoff) is present. Eras: `v2` (2026-04-13 onward) has every market; `v1~half` "
             "(2026-02-22..04-12) came from pmxt's v1 archive, which subscribed to only ~half of all markets, so its "
             "counts are upper bounds; `absent` = no archive file (2026-06-12..17). `archive h` = hours present "
             "in that UTC day's file (24 = complete). 2026-08-10 is partial.", "",
             "The planned count does not yet require the replayed book to be anchored (see README: books are "
             "replayed from listing or re-anchored by a full `book` snapshot); the measured table below does.", "",
             "Archive hours: the parquet footers show 24 h for every day except 02-24 (22), 04-04 (22), 05-22 (23), "
             "05-25 (9), 06-05 (9), 06-11 (4), 06-23 (13) and 08-10 (1, partial day); the dataset card's 5-15 % "
             "missing hours are mostly not visible as whole empty hours here (an hour can still be thin). The ingest "
             "logs per-day `hours_absent` and marks books unverified across them.", "",
             "## Planned (pre-registration basis)", ""]
    lines += table(cells, lg, have)
    if a.measured:
        mc, ml = measured(a)
        lines += ["", "## Measured (after ingest): verified book >= 1 h inside the 48 h window", ""]
        lines += table(mc, ml, have)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {sum(sum(c.values()) for c in cells.values())} markets planned")


if __name__ == "__main__":
    main()
