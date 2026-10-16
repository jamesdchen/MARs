"""Make a builder's output loadable by mm3: write markets3.bin (and x3.bin) next to its files.

The two v3 builders already write the five v2 files (markets.bin, feat.bin, book.bin, fills.bin,
fill_index.bin) in the frozen layouts. What mm3 needs on top is per-market settlement / event /
reward-program metadata and, for per-row reward programs, a small row file. Nothing else is
rewritten, and the builders' folders are never touched (only --out is written).

    python adapt3.py BUILD_DIR                         # kind from meta.json (sports / elections / crypto)
    python adapt3.py BUILD_DIR --rewards rates.json    # sports-1 builds only: reward pools (USD/day per slug)
    python adapt3.py BUILD_DIR --check                 # print what would be written, write nothing

Schema mapping (ENV3.md section 11 has the full tables):

  sports-2 (analysis/v3_sports/build_sports.py, build_version "sports-2": meta.json carries `segments`,
  one record per markets.bin row, and the build writes sports_x3.bin)
    settle_mode 1, settle_price = segments[i].settle_price (the episode's last-row mark), event_ts =
    segments[i].kickoff_ts (the kickoff, for every episode: bucket pieces (--bucket-segments) and
    episodes cut before kickoff (--cut-before-kickoff-s) too, so time to event = time to kickoff),
    row_s = segments[i].row_s, category 1, rw_v_c / rw_min_size = markets.rewards_max_spread /
    rewards_min_size (the last-row program), rw_rate_day = segments[i].rw.rate_day_last.
    x3.bin = a copy of sports_x3.bin (full recorded ladder; never recomputed, and --rewards is
    ignored). Builds made with --no-legacy-settle (no flag 16, winner_up -1) load the same; with
    the legacy copy present it must equal segments[i].settle_price. TRADES_TRUNCATED is flag 1024
    (bit 32 is unused in sports-2 and is not read).
  sports-1 (build_version "sports-1", legacy; also a sports-2 build without `segments`)
    settle_mode 1, settle_price = markets.log_strike_vs_ul_open (flag SETTLE_PX 16 required: the
    last pre-kickoff mark), event_ts = open_ts + sports_feat.seconds_to_kickoff[first row] (the
    kickoff, also for segments that end before it), row_s = meta.sports.dt_s, category 1,
    rw_v_c / rw_min_size = markets.rewards_max_spread / rewards_min_size (gamma), rw_rate_day from
    --rewards {slug: usd_per_day} (else 0). With --rewards (the fallback for builds without
    sports_x3.bin), x3.bin is computed from book.bin (top 5 levels, the elections builder's scorer:
    size-cutoff adjusted mid, competitors' 'hi' Q_min, one seeded sample second per UTC minute,
    liq_pi = rate / 1440). TRADES_TRUNCATED is flag 32.
  Fees (both): markets.bin (gamma feeSchedule).
  elections (analysis/v3_elections/build_elections.py, split feat layout)
    settle_mode 2 (mark at the last row: markets are unresolved, winner_up = -1), settle_price =
    that mark (informational), event_ts = end_ts (window end), row_s 1, category 2,
    rw_* = feat_x at the first row (rw_max_spread_c, rw_min_size, rw_daily_rate); x3.bin = per-row
    feat_x rw_max_spread_c, rw_min_size, rw_daily_rate, rw_in_program and feat.bin liq_mid,
    liq_comp, liq_pi.
  crypto (data/build): nothing is needed (mm3 synthesises the legacy records); --kind crypto
    writes them explicitly.

D16 recording gaps (ENV3.md section 11a). The gap links go into markets3.bin itself, in the two int32
words that were padding: `gap_next` / `gap_prev` = index + 1 of the linked piece (0 = none), plus
flags3 bits GAP_CUT 2, AFTER_GAP 4, EVAL_EXCL 8 (sports: last-hour gap, excluded from val/test
scoring) and DAY_INELIGIBLE 16 (a UTC day the piece spans has more than max_day_gap_s of gaps:
train only). Builds without gap fields write zeros there, as before, so mm3 stays v2 bit for bit.
The per-day gap summary, the eval-excluded buckets and the chains go into meta3.json `gaps`.
  sports-3 (build_sports.py, meta.json segments[i].gap.prev_sid / next_sid, markets.bin flags
    GAP_CUT 16384 / AFTER_GAP 32768 / EVAL_EXCL_LAST_HOUR 65536, segments[i].eval_excluded_buckets,
    meta.gaps.days {day: {n_gaps, total_gap_s, eligible}}): links by sid; DAY_INELIGIBLE when any
    UTC day of [open_ts, end_ts] is not eligible.
  elections (build_elections.py): links from meta.json episodes[i].prev_segment_id / next_segment_id
    when present, else markets_meta.parquet (prev_segment_id / next_segment_id), else the builder's
    own rule (consecutive episodes of a market, end_reason 'gap' -> start_reason 'gap');
    DAY_INELIGIBLE = not episodes[i].day_eligible, or the recorder-wide day of gaps_per_day.csv is
    not eligible; per-day totals from gaps_per_day.csv (scope recorder). A linked piece's event_ts is
    the end of its chain's last piece (the window episode the gaps cut), so time to event (obs 50,
    pull_s) never announces a gap.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from reference3 import (BOOK_DTYPE, CAT_CRYPTO, CAT_ELECTIONS, CAT_SPORTS, EXT_DTYPE, FLAG3_AFTER_GAP,
                        FLAG3_DAY_INELIGIBLE, FLAG3_EVAL_EXCL, FLAG3_GAP_CUT, MARKET_DTYPE, SETTLE_LAST_MARK,
                        SETTLE_PRICE, X_COLS, legacy_ext)

SPORTS_FLAG_SETTLE_PX = 16      # legacy settle marker (sports-1 always; sports-2 unless --no-legacy-settle)
SPORTS1_FLAG_TRUNC = 32         # sports-1 TRADES_TRUNCATED (32 is UNRESOLVED in elections; unused in sports-2)
SPORTS2_FLAG_TRUNC = 1024       # sports-2 TRADES_TRUNCATED
SPORTS3_FLAG_GAP_CUT = 16384    # sports-3 (D16): the segment ends at a recording gap
SPORTS3_FLAG_AFTER_GAP = 32768  # sports-3 (D16): the segment starts after a recording gap
SPORTS3_FLAG_EVAL_EXCL = 65536  # sports-3 (D16): last-hour gap in the game, excluded from val/test
SPORTS_COLS = ["seconds_to_kickoff", "seconds_to_seg_end", "microprice", "hours_since_first_book"]
CHUNK = 1 << 20
REWARD_SEED = 20261016


def write_atomic(path: Path, arr: np.ndarray):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    arr.tofile(tmp)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def detect(build: Path) -> str:
    meta = json.loads((build / "meta.json").read_text())
    if str(meta.get("build_version", "")).startswith("sports"):
        return "sports"
    if "v3_elections" in str(meta.get("builder", "")):
        return "elections"
    return "crypto"


def score(d, v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((d < v) & (v > 0), ((v - d) / np.where(v > 0, v, 1.0)) ** 2, 0.0)


def book_rewards(book, v, min_size):
    """Per row of book.bin (top 5 levels, mils): the size-cutoff adjusted mid (-1 if a side is empty)
    and competitors' Q per side, as analysis/rewards/reward_terms.py (adjusted_mid, side_scores) and
    the elections builder (reward_stats) compute them on the full ladder."""
    bp = book["bid_px"].astype(np.float64) / 1000.0
    ap = book["ask_px"].astype(np.float64) / 1000.0
    bs = book["bid_sz"].astype(np.float64)
    as_ = book["ask_sz"].astype(np.float64)
    bok = book["bid_px"] > 0
    aok = book["ask_px"] < 1000
    n = len(book)
    nb, na = bok.sum(1), aok.sum(1)

    def cut(p, s, ok, k_n):
        cs = np.cumsum(np.where(ok, s, 0.0), 1)
        k = (cs < min_size[:, None] - 1e-9).sum(1)
        k = np.minimum(k, np.maximum(k_n - 1, 0))
        return p[np.arange(n), k]

    valid = (book["flags"] & 1) > 0
    two = valid & (nb > 0) & (na > 0)
    mid = np.where(two, 0.5 * (cut(bp, bs, bok, nb) + cut(ap, as_, aok, na)), -1.0)
    m0 = np.where(two, mid, 0.0)
    qb = (score(np.abs(m0[:, None] - bp) * 100, v[:, None]) * bs * bok * (bs >= min_size[:, None] - 1e-9)).sum(1)
    qa = (score(np.abs(ap - m0[:, None]) * 100, v[:, None]) * as_ * aok * (as_ >= min_size[:, None] - 1e-9)).sum(1)
    return mid, np.where(two, qb, 0.0), np.where(two, qa, 0.0)


def sports_x3_top5(book, mk, ext, dt_s):
    """x3 [N, 8] from book.bin's top 5 levels for the per-market program in ext (rw_v_c, rw_min_size,
    rw_rate_day): the fallback for builds without the builder's full-ladder sports_x3.bin."""
    N = len(book)
    x3 = np.zeros((N, len(X_COLS)), np.float32)
    for i, m in enumerate(mk):
        r0, n = int(m["row_offset"]), int(m["n_rows"])
        v = np.full(n, float(ext["rw_v_c"][i]))
        ms = np.full(n, float(ext["rw_min_size"][i]))
        rate = float(ext["rw_rate_day"][i])
        prog = rate > 0 and v[0] > 0
        mid, qb, qa = book_rewards(np.asarray(book[r0:r0 + n]), v, ms)
        inside = (mid >= 0.10) & (mid <= 0.90)
        comp = np.minimum(qb, qa) + np.where(inside, np.abs(qb - qa) / 3.0, 0.0)
        ts = int(m["open_ts"]) + (np.arange(n) * dt_s).astype(np.int64)
        minute = ts // 60
        first = np.r_[True, minute[1:] != minute[:-1]]
        rng = np.random.default_rng([REWARD_SEED, i])
        offs = rng.integers(0, 60, first.sum())
        starts = np.flatnonzero(first)
        sample = np.zeros(n, bool)
        for j, s in enumerate(starts):                    # the first row at or after the drawn second
            e = starts[j + 1] if j + 1 < len(starts) else n
            hit = np.flatnonzero(ts[s:e] % 60 >= offs[j])
            if len(hit):
                sample[s + hit[0]] = True
        x = x3[r0:r0 + n]
        x[:, 0], x[:, 1], x[:, 2], x[:, 3] = v, ms, rate, float(prog)
        x[:, 4], x[:, 5] = mid, comp
        x[:, 6] = np.where(sample & (mid >= 0) & prog, rate / 1440.0, 0.0)
    return x3


# ----------------------------------------------------------------------------- D16 recording gaps
def utc_days(t0, t1):
    """UTC days ('YYYY-MM-DD') touched by [t0, t1] (unix s)."""
    d = _dt.datetime.fromtimestamp(int(t0), _dt.timezone.utc).date()
    end = _dt.datetime.fromtimestamp(int(t1), _dt.timezone.utc).date()
    out = []
    while d <= end:
        out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out


def _link(ext, i, j):
    ext["gap_next"][i], ext["gap_prev"][j] = j + 1, i + 1


def check_links(ext, mk, key=None):
    """Links are mutual, go forward in time, join a GAP_CUT piece to an AFTER_GAP piece of the same
    market (key) and split; returns the chains (lists of market indices, head first, >= 2 pieces)."""
    n = len(ext)
    nx, pv = ext["gap_next"].astype(np.int64) - 1, ext["gap_prev"].astype(np.int64) - 1
    for i in range(n):
        j = nx[i]
        if j < 0:
            continue
        if pv[j] != i:
            raise SystemExit(f"gap link {i} -> {j} is not mutual")
        if not (ext["flags3"][i] & FLAG3_GAP_CUT and ext["flags3"][j] & FLAG3_AFTER_GAP):
            raise SystemExit(f"gap link {i} -> {j}: not a GAP_CUT piece followed by an AFTER_GAP piece")
        if not int(mk["end_ts"][i]) < int(mk["open_ts"][j]):
            raise SystemExit(f"gap link {i} -> {j}: the next piece does not start after the gap")
        if int(mk["split"][i]) != int(mk["split"][j]):
            raise SystemExit(f"gap link {i} -> {j} crosses splits ({mk['split'][i]} -> {mk['split'][j]})")
        if key is not None and key[i] != key[j]:
            raise SystemExit(f"gap link {i} -> {j} joins different markets ({key[i]} / {key[j]})")
    chains = []
    for i in range(n):
        if pv[i] < 0 and nx[i] >= 0:
            c = [i]
            while nx[c[-1]] >= 0:
                c.append(int(nx[c[-1]]))
                if len(c) > n:
                    raise SystemExit("gap links form a cycle")
            chains.append(c)
    return chains


def gaps_info(ext, mk, chains, days, schema, extra=None, excluded_buckets=None):
    """meta3.json `gaps`: the per-day summary, the chains and the per-market D16 flags."""
    f3 = ext["flags3"]
    pick = lambda b: np.flatnonzero(f3 & b).tolist()  # noqa: E731
    days = {d: {k: v for k, v in x.items()} for d, x in sorted((days or {}).items())}
    return {"schema": schema, "days": days,
            "days_ineligible": sorted(d for d, x in days.items() if not x.get("eligible", True)),
            "n_gaps_total": int(sum(int(x.get("n_gaps", 0)) for x in days.values())),
            "total_gap_s": round(float(sum(float(x.get("total_gap_s", 0.0)) for x in days.values())), 3),
            "n_links": int((ext["gap_next"] > 0).sum()), "chains": chains,
            "gap_cut": pick(FLAG3_GAP_CUT), "after_gap": pick(FLAG3_AFTER_GAP),
            "eval_excl_last_hour": pick(FLAG3_EVAL_EXCL), "day_ineligible": pick(FLAG3_DAY_INELIGIBLE),
            "eval_excluded_buckets": {str(i): b for i, b in sorted((excluded_buckets or {}).items()) if b},
            **(extra or {})}


def sports_gaps(meta, mk, segs, ext):
    """sports-3 (D16): gap links by sid, flags, eval exclusions and day eligibility (from meta.gaps)."""
    gm = meta.get("gaps") or {}
    days = gm.get("days") or {}           # the builder's eligibility (total <= max_day_gap_s) is authoritative
    idx = {x["sid"]: i for i, x in enumerate(segs)}
    excl = {}
    for i, x in enumerate(segs):
        fl = int(mk["flags"][i])
        g = x.get("gap") or {}
        f3 = 0
        if fl & SPORTS3_FLAG_GAP_CUT or x.get("cut_reason") == "gap":
            f3 |= FLAG3_GAP_CUT
        if fl & SPORTS3_FLAG_AFTER_GAP or x.get("start_reason") == "gap":
            f3 |= FLAG3_AFTER_GAP
        if fl & SPORTS3_FLAG_EVAL_EXCL or x.get("excluded_from_eval_last_hour"):
            f3 |= FLAG3_EVAL_EXCL
        if any(not (days.get(d) or {}).get("eligible", True) for d in utc_days(mk["open_ts"][i], mk["end_ts"][i])):
            f3 |= FLAG3_DAY_INELIGIBLE
        ext["flags3"][i] |= f3
        excl[i] = list(x.get("eval_excluded_buckets") or [])
        if f3 & FLAG3_EVAL_EXCL and not excl[i]:
            excl[i] = list(gm.get("last_hour_buckets") or ["1h-30m", "30-5m", "5-0m"])
        if g.get("next_sid"):
            if g["next_sid"] not in idx:
                raise SystemExit(f"segments[{i}].gap.next_sid {g['next_sid']!r} is not a segment of this build")
            _link(ext, i, idx[g["next_sid"]])
    for i, x in enumerate(segs):               # prev_sid must agree with the next_sid links
        ps = (x.get("gap") or {}).get("prev_sid")
        if (ps is not None and (ps not in idx or int(ext["gap_prev"][i]) != idx[ps] + 1)) or \
                (ps is None and int(ext["gap_prev"][i]) != 0):
            raise SystemExit(f"segments[{i}].gap.prev_sid {ps!r} disagrees with the next_sid links")
    chains = check_links(ext, mk, [x.get("slug") for x in segs])
    return gaps_info(ext, mk, chains, days, "sports-3" if gm else None,
                     {**{k: gm.get(k) for k in ("gap_s", "max_day_gap_s", "last_hour_s", "games_last_hour_gap",
                                                 "val_test_days_ineligible") if k in gm},
                      "days_not_all_logged": sorted(d for d, x in days.items() if not x.get("all_logged", True))},
                     excl)


def read_gaps_per_day(build: Path):
    """build_elections gaps_per_day.csv, scope 'recorder': {day: {n_gaps, total_gap_s, eligible, ...}},
    and per day the count of ineligible market-days (scope 'market')."""
    p = build / "gaps_per_day.csv"
    if not p.exists():
        return {}
    out, mkt = {}, {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            elig = str(r.get("eligible", "True")).strip().lower() in ("true", "1")
            if r.get("scope") == "recorder":
                out[r["day"]] = {"n_gaps": int(float(r.get("n_gaps") or 0)),
                                 "total_gap_s": float(r.get("total_gap_s") or 0.0), "eligible": elig,
                                 "n_gaps_gt_max": int(float(r.get("n_gaps_gt_max") or 0)),
                                 "total_gap_gt_max_s": float(r.get("total_gap_gt_max_s") or 0.0),
                                 "longest_gap_s": float(r.get("longest_gap_s") or 0.0)}
            elif r.get("scope") == "market":
                m = mkt.setdefault(r["day"], [0, 0])
                m[0] += 1
                m[1] += int(not elig)
    for d, (n, bad) in mkt.items():
        out.setdefault(d, {"n_gaps": 0, "total_gap_s": 0.0, "eligible": True})
        out[d]["market_days"], out[d]["market_days_ineligible"] = n, bad
    return out


def election_links(build: Path, eps):
    """(prev, next) episode ids per episode: meta.json episodes fields, else markets_meta.parquet, else the
    builder's rule (consecutive episodes of a market, end_reason gap -> start_reason gap)."""
    for a, b in (("prev_segment_id", "next_segment_id"), ("prev_id", "next_id")):
        if any(b in x or a in x for x in eps):
            return [(x.get(a) or None, x.get(b) or None) for x in eps], f"meta.json episodes {a} / {b}"
    pq = build / "markets_meta.parquet"
    if pq.exists():
        try:
            import pandas as pd
            t = pd.read_parquet(pq, columns=["episode_id", "prev_segment_id", "next_segment_id"])
            m = {r.episode_id: (r.prev_segment_id or None, r.next_segment_id or None) for r in t.itertuples()}
            if all(x["id"] in m for x in eps):
                return [m[x["id"]] for x in eps], "markets_meta.parquet"
        except Exception as e:                       # noqa: BLE001  (no pyarrow: fall back to the rule)
            print(f"adapt3: markets_meta.parquet unreadable ({e}); links from the builder's rule", file=sys.stderr)
    out = [[None, None] for _ in eps]
    by = {}
    for i, x in enumerate(eps):
        by.setdefault(x["slug"], []).append(i)
    for lst in by.values():
        lst.sort(key=lambda i: int(eps[i]["open_ts"]))
        for p, q in zip(lst[:-1], lst[1:]):
            if eps[p].get("end_reason") == "gap" and eps[q].get("start_reason") == "gap":
                out[q][0], out[p][1] = eps[p]["id"], eps[q]["id"]
    return [tuple(x) for x in out], "builder rule (episodes start/end reasons)"


def election_gaps(build: Path, meta, mk, ext):
    eps = meta.get("episodes")
    days = read_gaps_per_day(build)
    if not eps or len(eps) != len(mk):
        return gaps_info(ext, mk, [], days, "elections" if days else None, {"links_source": None})
    ids = [x["id"] for x in eps]
    if meta.get("slugs") is not None and list(meta["slugs"]) != ids:
        raise SystemExit("meta.json episodes[*].id differs from meta.slugs (episodes not in market order?)")
    links, src = election_links(build, eps)
    idx = {x: i for i, x in enumerate(ids)}
    for i, x in enumerate(eps):
        f3 = 0
        if x.get("end_reason") == "gap":
            f3 |= FLAG3_GAP_CUT
        if x.get("start_reason") == "gap":
            f3 |= FLAG3_AFTER_GAP
        if not x.get("day_eligible", True) or any(not (days.get(d) or {}).get("eligible", True)
                                                 for d in utc_days(mk["open_ts"][i], mk["end_ts"][i])):
            f3 |= FLAG3_DAY_INELIGIBLE
        ext["flags3"][i] |= f3
    for i, (pv, nx) in enumerate(links):
        if nx:
            if nx not in idx:
                raise SystemExit(f"episode {ids[i]}: next segment {nx!r} is not in this build")
            _link(ext, i, idx[nx])
    for i, (pv, nx) in enumerate(links):
        if (pv or None) != (ids[int(ext["gap_prev"][i]) - 1] if ext["gap_prev"][i] else None):
            raise SystemExit(f"episode {ids[i]}: prev segment {pv!r} disagrees with the next links")
    chains = check_links(ext, mk, [x["slug"] for x in eps])
    for c in chains:                       # time to event = to the end of the window episode the gaps cut
        ext["event_ts"][c] = int(mk["end_ts"][c[-1]])
    return gaps_info(ext, mk, chains, days, "elections",
                     {"links_source": src, "max_gap_s": (meta.get("gaps") or {}).get("max_gap_s"),
                      "day_max_gap_s": (meta.get("gaps") or {}).get("day_max_gap_s"),
                      "ineligible_day": (meta.get("gaps") or {}).get("ineligible_day")})


def sports(build: Path, args):
    meta = json.loads((build / "meta.json").read_text())
    if meta.get("segments") is not None:
        return sports2(build, args, meta)
    return sports1(build, args, meta)


def _sports_common(mk, ext):
    ext["settle_mode"] = SETTLE_PRICE
    ext["category"] = CAT_SPORTS
    ext["rw_v_c"] = mk["rewards_max_spread"]
    ext["rw_min_size"] = mk["rewards_min_size"]
    ext["fee_rate"] = ext["fee_exp"] = ext["rebate_rate"] = -1.0


def sports2(build: Path, args, meta):
    """build_version sports-2: settlement, kickoff and reward pool from meta.json `segments`, x3.bin =
    the builder's full-ladder sports_x3.bin."""
    mk = np.fromfile(build / "markets.bin", MARKET_DTYPE)
    N = int(meta["N"])
    segs = meta["segments"]
    if len(segs) != len(mk):
        raise SystemExit(f"meta.json has {len(segs)} segments for {len(mk)} markets")
    for k in ("row_offset", "n_rows", "open_ts", "end_ts"):
        got = np.array([int(x[k]) for x in segs], np.int64)
        if not np.array_equal(got, mk[k].astype(np.int64)):
            raise SystemExit(f"meta.json segments[*].{k} differs from markets.bin (not the same build?)")
    if meta.get("slugs") is not None and [x["sid"] for x in segs] != list(meta["slugs"]):
        raise SystemExit("meta.json segments[*].sid differs from meta.slugs")
    ext = np.zeros(len(mk), EXT_DTYPE)
    _sports_common(mk, ext)
    settle = np.array([float(x["settle_price"]) for x in segs], np.float64)
    if not ((settle >= 0) & (settle <= 1)).all():
        raise SystemExit("segments[*].settle_price outside [0, 1]")
    legacy = (mk["flags"] & SPORTS_FLAG_SETTLE_PX) > 0
    if legacy.any() and not np.array_equal(mk["log_strike_vs_ul_open"][legacy], settle[legacy].astype(np.float32)):
        raise SystemExit("legacy settle copy (log_strike_vs_ul_open, flag 16) != segments[*].settle_price")
    ext["settle_price"] = settle
    ext["event_ts"] = [int(x["kickoff_ts"]) for x in segs]          # time to event = time to kickoff, always
    ext["row_s"] = [float(x.get("row_s", meta["sports"]["dt_s"])) for x in segs]
    ext["rw_rate_day"] = [float((x.get("rw") or {}).get("rate_day_last", 0.0)) for x in segs]
    if (ext["event_ts"] < mk["end_ts"]).any():
        raise SystemExit("segments[*].kickoff_ts before the episode end (an episode past kickoff?)")
    trunc = (mk["flags"] & SPORTS2_FLAG_TRUNC) > 0
    gaps = sports_gaps(meta, mk, segs, ext)
    src = build / "sports_x3.bin"
    x3, how = None, "none"
    if src.exists():
        if src.stat().st_size != N * len(X_COLS) * 4:
            raise SystemExit(f"{src}: {src.stat().st_size} bytes, expected N={N} x {len(X_COLS)} float32")
        how = "sports_x3.bin (full ladder, copied)"
        if args.rewards:
            print(f"adapt3: --rewards {args.rewards} ignored: {src.name} holds the full-ladder reward columns "
                  "and meta.json the pools (--rewards is the fallback for builds without them)", file=sys.stderr)
        if not args.check:
            out = Path(args.out or build)
            out.mkdir(parents=True, exist_ok=True)
            x3 = out / f".x3.bin.tmp{os.getpid()}"
            shutil.copyfile(src, x3)
    elif args.rewards:
        rates = json.loads(Path(args.rewards).read_text())
        slugs = [x["slug"] for x in segs]
        ext["rw_rate_day"] = [float(rates.get(s, 0.0)) for s in slugs]
        book = np.memmap(build / "book.bin", BOOK_DTYPE, "r", shape=(N,))
        x3 = sports_x3_top5(book, mk, ext, float(meta["sports"]["dt_s"]))
        how = "book.bin top 5 (--rewards fallback: no sports_x3.bin)"
    sp = meta.get("sports") or {}
    return ext, x3, {"build_version": meta.get("build_version"), "schema": "segments", "x3_source": how,
                     "legacy_settle": bool(legacy.all()), "no_legacy_settle": bool(not legacy.any()),
                     "bucket_segments": bool((sp.get("buckets") or {}).get("bucket_segments", False)),
                     "cut_before_kickoff_s": sp.get("cut_before_kickoff_s"),
                     "trades_truncated": int(trunc.sum()), "trades_truncated_markets": np.flatnonzero(trunc).tolist(),
                     "trades_truncated_flag": SPORTS2_FLAG_TRUNC, "gaps": gaps}


def sports1(build: Path, args, meta):
    """build_version sports-1 (legacy): settlement from the markets.bin copy (flag 16), kickoff from
    sports_feat.bin row 0, pools from --rewards (x3.bin from book.bin's top 5 levels)."""
    mk = np.fromfile(build / "markets.bin", MARKET_DTYPE)
    N = int(meta["N"])
    dt_s = float(meta["sports"]["dt_s"])
    sf = np.memmap(build / "sports_feat.bin", np.float32, "r", shape=(N, len(SPORTS_COLS)))
    slugs = [s.split("#")[0] for s in meta["slugs"]]
    rates = json.loads(Path(args.rewards).read_text()) if args.rewards else {}
    ext = np.zeros(len(mk), EXT_DTYPE)
    if not (mk["flags"] & SPORTS_FLAG_SETTLE_PX).all():
        raise SystemExit("sports-1 build without flag SETTLE_PX on every market: not a build_sports.py output? "
                         "(sports-2 builds carry meta.json segments; rebuild with the current build_sports.py)")
    _sports_common(mk, ext)
    ext["settle_price"] = mk["log_strike_vs_ul_open"].astype(np.float64)
    ext["event_ts"] = mk["open_ts"] + np.rint(sf[mk["row_offset"], 0]).astype(np.int64)
    ext["row_s"] = dt_s
    ext["rw_rate_day"] = [float(rates.get(s, 0.0)) for s in slugs]
    x3 = None
    if args.rewards:
        x3 = sports_x3_top5(np.memmap(build / "book.bin", BOOK_DTYPE, "r", shape=(N,)), mk, ext, dt_s)
    trunc = (mk["flags"] & SPORTS1_FLAG_TRUNC) > 0
    return ext, x3, {"build_version": meta.get("build_version"), "schema": "legacy", "dt_s": dt_s,
                     "rates_given": len(rates), "x3_source": "book.bin top 5 (--rewards)" if args.rewards else "none",
                     "trades_truncated": int(trunc.sum()), "trades_truncated_markets": np.flatnonzero(trunc).tolist(),
                     "trades_truncated_flag": SPORTS1_FLAG_TRUNC}


def elections(build: Path, args):
    fmeta = json.loads((build / "feat_meta.json").read_text())
    ex = fmeta.get("extra") or {}
    if ex.get("file") != "feat_x.bin":
        raise SystemExit("elections build without feat_x.bin (wide layout?): rebuild with --feat-layout split")
    xi, X, N = ex["column_index"], int(ex["X"]), int(fmeta["N"])
    mk = np.fromfile(build / "markets.bin", MARKET_DTYPE)
    fx = np.memmap(build / "feat_x.bin", np.float32, "r", shape=(N, X))
    feat = np.memmap(build / "feat.bin", np.float32, "r", shape=(N, 44))
    ci = fmeta["column_index"]
    last = mk["row_offset"] + mk["n_rows"] - 1
    ext = np.zeros(len(mk), EXT_DTYPE)
    ext["settle_mode"] = SETTLE_LAST_MARK
    ext["settle_price"] = feat[last, ci["mark"]].astype(np.float64)
    ext["event_ts"] = mk["end_ts"]
    ext["category"] = CAT_ELECTIONS
    ext["row_s"] = 1.0
    first = mk["row_offset"]
    ext["rw_v_c"] = fx[first, xi["rw_max_spread_c"]]
    ext["rw_min_size"] = fx[first, xi["rw_min_size"]]
    ext["rw_rate_day"] = fx[first, xi["rw_daily_rate"]]
    ext["fee_rate"] = ext["fee_exp"] = ext["rebate_rate"] = -1.0
    gaps = election_gaps(build, json.loads((build / "meta.json").read_text()), mk, ext)
    if args.check:
        return ext, None, {"x3_rows": N, "gaps": gaps}
    out = Path(args.out or build)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".x3.bin.tmp{os.getpid()}"              # streamed: an election build can hold 10^7+ rows
    x3 = np.memmap(tmp, np.float32, "w+", shape=(N, len(X_COLS)))
    src = [("rw_max_spread_c", fx), ("rw_min_size", fx), ("rw_daily_rate", fx), ("rw_in_program", fx),
           ("liq_mid", feat), ("liq_comp", feat), ("liq_pi", feat)]
    for lo in range(0, N, CHUNK):
        hi = min(N, lo + CHUNK)
        for j, (name, arr) in enumerate(src):
            x3[lo:hi, j] = arr[lo:hi, (xi if arr is fx else ci)[name]]
    x3.flush()
    del x3
    return ext, tmp, {"x3_rows": N, "gaps": gaps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build")
    ap.add_argument("--kind", choices=("auto", "sports", "elections", "crypto"), default="auto")
    ap.add_argument("--out", help="directory for markets3.bin / x3.bin / meta3.json (default: the build dir)")
    ap.add_argument("--rewards", help="sports-1 builds (no sports_x3.bin): JSON {slug: reward pool USD/day}; "
                    "computes x3.bin from book.bin's top 5 levels. Ignored when sports_x3.bin exists")
    ap.add_argument("--check", action="store_true", help="print the mapping, write nothing")
    args = ap.parse_args(argv)
    build = Path(args.build)
    out = Path(args.out or build)
    kind = detect(build) if args.kind == "auto" else args.kind
    if kind == "sports":
        ext, x3, info = sports(build, args)
    elif kind == "elections":
        ext, x3, info = elections(build, args)
    else:
        ext, x3, info = legacy_ext(np.fromfile(build / "markets.bin", MARKET_DTYPE)), None, {}
    per_market = {k: info.pop(k) for k in ("trades_truncated_markets", "gaps") if k in info}
    g = per_market.get("gaps") or {}
    info.update({"gap_links": int(g.get("n_links", 0)), "gap_days_ineligible": len(g.get("days_ineligible", [])),
                 "gap_eval_excl_last_hour": len(g.get("eval_excl_last_hour", [])),
                 "gap_day_ineligible_markets": len(g.get("day_ineligible", []))})
    summary = {"kind": kind, "markets": int(len(ext)), "x3": x3 is not None or info.get("x3_source", "none") != "none",
               "settle_mode": np.unique(ext["settle_mode"]).tolist(), "row_s": np.unique(ext["row_s"]).tolist(),
               "rw_programs": int((ext["rw_rate_day"] > 0).sum()), **info}
    print(json.dumps(summary))
    if args.check:
        if isinstance(x3, Path):
            x3.unlink()
        return summary
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "markets3.bin", ext)
    if isinstance(x3, Path):
        os.replace(x3, out / "x3.bin")
    elif x3 is not None:
        write_atomic(out / "x3.bin", x3)
    elif (out / "x3.bin").exists():
        (out / "x3.bin").unlink()
    meta3 = {"adapter": "polymarket/v3env/adapt3.py", "kind": kind, "summary": summary,
             "markets3": {"itemsize": EXT_DTYPE.itemsize, "fields": list(EXT_DTYPE.names)},
             "x3": {"present": x3 is not None, "columns": X_COLS, "dtype": "float32", "order": "row-major [N, 8]"},
             "source_markets_sha256": sha256(build / "markets.bin"), **per_market}
    (out / "meta3.json").write_text(json.dumps(meta3, indent=1))
    return summary


if __name__ == "__main__":
    main()
