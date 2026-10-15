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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from reference3 import (BOOK_DTYPE, CAT_CRYPTO, CAT_ELECTIONS, CAT_SPORTS, EXT_DTYPE, MARKET_DTYPE, SETTLE_LAST_MARK,
                        SETTLE_PRICE, X_COLS, legacy_ext)

SPORTS_FLAG_SETTLE_PX = 16      # legacy settle marker (sports-1 always; sports-2 unless --no-legacy-settle)
SPORTS1_FLAG_TRUNC = 32         # sports-1 TRADES_TRUNCATED (32 is UNRESOLVED in elections; unused in sports-2)
SPORTS2_FLAG_TRUNC = 1024       # sports-2 TRADES_TRUNCATED
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
                     "trades_truncated_flag": SPORTS2_FLAG_TRUNC}


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
    if args.check:
        return ext, None, {"x3_rows": N}
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
    return ext, tmp, {"x3_rows": N}


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
    per_market = {k: info.pop(k) for k in ("trades_truncated_markets",) if k in info}
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
