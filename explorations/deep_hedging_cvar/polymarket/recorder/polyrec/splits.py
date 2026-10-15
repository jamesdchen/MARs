"""Date-based train/val/test assignment for merged datasets.

A day's split is a pure function of its date and the config, so recording more days never moves an
earlier day; a lock file additionally pins every assignment ever used by merge.py, and frozen days
(the original 2026-08-20..24 dataset) can never change. In particular the original test day stays
test, and any day that has been test once can never become train or val.

Config (JSON, see split_config.json):
    {"frozen":  {"2026-08-24": "test", ...},                 # must equal data/build.py SPLIT_DAYS
     "days":    {"2026-10-20": "val", "2026-10-21..2026-10-25": "train", "2026-10-26": "exclude"},
     "default": "auto",                                      # unlisted recorded days: auto|train|val|test|exclude
     "auto":    {"anchor": "2026-10-15", "cycle": ["train", "train", "train", "train", "train", "train", "val", "test"]},
     "lock_file": "split_lock.json"}
The default 8-day cycle (75/12.5/12.5) rotates val/test through the weekdays (8 is coprime with 7).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
NAMES = {v: k for k, v in SPLIT_IDS.items()}
VALID = set(SPLIT_IDS) | {"exclude"}


class SplitError(Exception):
    pass


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def expand_days(spec: dict) -> dict:
    """{"a..b": x, "c": y} -> {day: x}."""
    out = {}
    for k, v in (spec or {}).items():
        if ".." in k:
            a, b = (_date(x.strip()) for x in k.split(".."))
            if b < a:
                raise SplitError(f"bad range {k}")
            d = a
            while d <= b:
                out[d.isoformat()] = v
                d += dt.timedelta(days=1)
        else:
            out[_date(k).isoformat()] = v
    return out


def assign(day: str, cfg: dict) -> str:
    frozen = cfg.get("frozen", {})
    if day in frozen:
        return frozen[day]
    explicit = expand_days(cfg.get("days", {}))
    v = explicit.get(day, cfg.get("default", "auto"))
    if v == "auto":
        a = cfg.get("auto") or {}
        cyc = a.get("cycle") or ["train"] * 6 + ["val", "test"]
        anchor = _date(a.get("anchor", "2026-10-15"))
        v = cyc[(_date(day) - anchor).days % len(cyc)]
    if v not in VALID:
        raise SplitError(f"{day}: unknown split {v!r}")
    return v


def resolve(days: list[str], cfg: dict, build_split_days: dict, lock: dict | None,
            allow_reassign: bool = False) -> dict:
    """day -> 'train'|'val'|'test'|'exclude' for `days` (recorded) plus every frozen day.

    build_split_days: data/build.py SPLIT_DAYS ({day: 0|1|2}); `frozen` must reproduce it exactly.
    lock: previous assignments ({day: name}); a change is an error unless allow_reassign, and a day
    that was ever 'test' (or is frozen) can never change at all."""
    frozen = cfg.get("frozen", {})
    want = {d: NAMES[s] for d, s in build_split_days.items()}
    if frozen != want:
        raise SplitError(f"frozen days {frozen} differ from data/build.py SPLIT_DAYS {want}")
    out = dict(frozen)
    for d in sorted(set(days) - set(frozen)):
        out[d] = assign(d, cfg)
    for d, prev in (lock or {}).items():
        if d not in out:
            continue
        if out[d] != prev:
            if d in frozen or prev == "test" or not allow_reassign:
                raise SplitError(f"{d} was {prev!r}, config now says {out[d]!r} (locked; "
                                 f"{'never reassignable' if d in frozen or prev == 'test' else 'use --reassign'})")
    return out


def load_lock(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_lock(path: Path, lock: dict, used: dict):
    merged = dict(lock)
    merged.update({d: s for d, s in used.items() if s != "exclude"})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(merged.items())), indent=1))
    tmp.replace(path)


def to_build_split_days(assignment: dict) -> dict:
    """For data/build.py: {day: split id}; excluded days are left out (their markets must be dropped)."""
    return {d: SPLIT_IDS[s] for d, s in sorted(assignment.items()) if s != "exclude"}
