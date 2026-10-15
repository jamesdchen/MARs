"""Coverage of each market from the recorder's coverage JSONL logs (see watchdog.CoverageLog).

Per run, a market is covered from each "live" event to the next "down" event for it, or, if the run
ended without one (crash), to that run's last heartbeat / stop. Intervals of all runs are unioned.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import orjson


def read_events(out: Path):
    for f in sorted((Path(out) / "coverage").glob("coverage_*.jsonl")):
        with open(f, "rb") as fh:
            for ln in fh:
                try:
                    yield orjson.loads(ln)
                except orjson.JSONDecodeError:
                    continue     # torn last line after a hard crash


def live_intervals(events) -> dict:
    """slug -> sorted, merged [(t0, t1), ...]."""
    last_seen = {}
    open_ = {}
    iv = defaultdict(list)
    evs = sorted(events, key=lambda e: e["t"])
    for e in evs:
        run, t = e["run"], e["t"]
        last_seen[run] = t
        ev = e["ev"]
        if ev == "live":
            open_.setdefault((run, e["slug"]), t)
        elif ev == "down":
            t0 = open_.pop((run, e["slug"]), None)
            if t0 is not None:
                iv[e["slug"]].append((t0, t))
        elif ev == "stop":
            for (r, s), t0 in list(open_.items()):
                if r == run:
                    iv[s].append((t0, t))
                    del open_[(r, s)]
    for (r, s), t0 in open_.items():      # crashed runs: bounded by their last heartbeat
        iv[s].append((t0, last_seen.get(r, t0)))
    return {s: merge(v) for s, v in iv.items()}


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def gaps(iv, lo: float, hi: float) -> list:
    """Uncovered sub-intervals of [lo, hi]."""
    g, cur = [], lo
    for a, b in iv:
        if b <= cur:
            continue
        if a > cur:
            g.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi:
        g.append((cur, hi))
    return [x for x in g if x[1] > x[0]]


def market_coverage(iv, open_s: float, end_s: float, max_gap_s: float = 5.0, max_total_gap_s: float = 15.0):
    """(ok, total_gap_s, max_gap_s) for the market life [open, end]."""
    g = gaps(iv, open_s, end_s)
    tot = sum(b - a for a, b in g)
    mx = max((b - a for a, b in g), default=0.0)
    return (mx <= max_gap_s and tot <= max_total_gap_s), tot, mx
