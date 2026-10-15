#!/usr/bin/env python3
"""Pre-game sports books -> the C market-making env's binaries (Track S, DECISIONS.md D11).

Writes the same files as data/build (ticks.bin, feat.bin, book.bin, fills.bin, fill_index.bin,
markets.bin, meta.json, feat_meta.json, splits.json) with the same record layouts and the same
feat.bin column layout, so ../../env.py / ../../mm.h / ../../train/envs.py (data_dir=OUT) load it
unchanged. Plus one side file, sports_feat.bin. See README_build.md for every schema decision.

    source ../../../.venv/bin/activate          # numpy, pandas, pyarrow, orjson (polyrec)
    python build_sports.py --rec rec_out [--rec raw_pob] --out build            # all finished games
    python build_sports.py --rec rec_out --dt 6 --max-hours 24                  # 1 episode / game
    python build_sports.py --selftest                                           # synthetic, end to end

Episode = one segment of exactly `seg` rows (+1) on a `dt`-second grid. Segments tile the window
[kickoff - max_hours, kickoff] backwards from kickoff; the last segment of a game ends AT kickoff.
Book rows captured at/after kickoff and trades with block time >= kickoff are dropped. Each
segment settles at its last row's mark (fractional; stored in markets.bin `log_strike_vs_ul_open`,
flag SETTLE_PX). The stock mm.h only settles at the int `winner_up` (0/1): use the patched copy in
env_sports/ for any P&L number (README_build.md, "Settlement").

Nothing here reads the game result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PM = HERE.parents[1]                              # polymarket/
sys.path.insert(0, str(PM / "data"))
sys.path.insert(0, str(PM / "recorder"))
import build as B  # noqa: E402  data/build.py (read-only use: COLS, MARKET_DTYPE, ladder_topk, merged_snapshots)

BUILD_VERSION = "sports-1"
NLEV = 5
SEG_LENS = (300, 900, 14400)                      # mm.h mm_len_bit: other lengths are never sampled
SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "purged"}
SPLIT_IDS = {v: k for k, v in SPLIT_NAMES.items()}
GAMMA = "https://gamma-api.polymarket.com"
CACHE = HERE / "cache"

# markets.bin flags (1..8 as in data/build.py)
FLAG_HAS_TRADES = 1
FLAG_SETTLE_PX = 16          # settlement = log_strike_vs_ul_open (float), not winner_up (env_sports/mm.h)
FLAG_TRADES_TRUNC = 32       # data-api history capped: the oldest pre-game legs may be missing
FLAG_HEAD_PADDED = 64        # segment starts before the first book row (padded with no-book rows)
FLAG_AT_KICKOFF = 128        # segment ends at kickoff (the last segment of the game)
FLAG_KICKOFF_FALLBACK = 256  # no gamma gameStartTime: kickoff from eventStartTime / endDate
FLAG_COVERAGE = 512          # recorder coverage log available (stale flag = recorder not live)

# ---- frozen layouts (data/features.py); asserted against data/build/feat_meta.json when present
OBS_COLS = ["p_logit", "mark_logit", "gap_c", "dgap5_c", "z", "sdv_bp", "sigma_1m_bp", "tau_frac", "log_tau",
            "in_twap", "delta_usd", "spread_ticks", "bid_present", "ask_present", "imb_l1", "bid_sz0", "ask_sz0",
            "flow5", "flow30", "flow120", "mret5_c", "mret30_c", "mret120_c", "ulz5", "ulz30", "ulz120",
            "len_5m", "len_15m", "len_4h", "tick_fine", "book_age", "crossed_raw", "mark_is_fv"]
AUX_COLS = ["p_model", "mark", "delta_usd_raw", "sigma", "sqrt_v", "liq_mid", "liq_comp", "liq_pi",
            "bid_minus_p_c", "ask_minus_p_c"]
FCOLS = OBS_COLS + AUX_COLS + ["pad_0"]
G = len(FCOLS)
CI = {c: i for i, c in enumerate(FCOLS)}
N_OBS = len(OBS_COLS)
assert G == 44 and N_OBS == 33
SPORTS_COLS = ["seconds_to_kickoff", "seconds_to_seg_end", "microprice", "hours_since_first_book"]

BOOK_DTYPE = np.dtype([("bid_px", "<i2", 5), ("ask_px", "<i2", 5), ("bid_sz", "<f4", 5),
                       ("ask_sz", "<f4", 5), ("flags", "u1"), ("tick", "u1"), ("tick_alt", "u1"),
                       ("same_run", "u1")])
FILL_DTYPE = np.dtype([("row", "<i4"), ("side", "i1"), ("first", "u1"), ("px", "<i2"),
                       ("price", "<f4"), ("size", "<f4"), ("tx_size", "<f4")])
assert BOOK_DTYPE.itemsize == 64 and FILL_DTYPE.itemsize == 20
BOOK_VALID, BOOK_CROSSED, BOOK_STALE, BOOK_FROM_DOWN, BOOK_SAME = 1, 2, 4, 8, 16
TCOLS, TC, F = B.COLS, B.C, B.F
assert F == 40
MARKET_DTYPE = B.MARKET_DTYPE
CLIP = 5.0
BYTES_PER_ROW = F * 4 + G * 4 + 64 + 4 + len(SPORTS_COLS) * 4


def clip(x, lo=-CLIP, hi=CLIP):
    return np.clip(x, lo, hi)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def lag(x, k):
    i = np.maximum(np.arange(len(x)) - k, 0)
    return x[i]


def window_sum(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - k, 0)]


def ffill(v, ok, fill):
    """v forward-filled from the rows where ok, `fill` before the first such row (causal)."""
    idx = np.where(ok, np.arange(len(v)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, v[np.maximum(idx, 0)], fill)


def num(x, default):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def to_ts(x) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        t = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


def write_atomic(path: Path, data):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    if isinstance(data, np.ndarray):
        data.tofile(tmp)
    else:
        tmp.write_text(data)
    os.replace(tmp, path)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- inputs
def _files(root: Path, name: str) -> list[Path]:
    """Recorder dataset files (final hourly files + unabsorbed parts), else any *.parquet below."""
    root = Path(root)
    if root.is_file():
        return [root]
    if (root / name).is_dir():
        try:
            from polyrec.sinks import dataset_files
            return dataset_files(root, name)
        except ImportError:
            return sorted((root / name).rglob("*.parquet"))
    return []


def load_books(recs: list[Path], slug_rx: re.Pattern | None) -> pa.Table | None:
    cols = ["market_slug", "token_id", "outcome", "market_end_at", "captured_at", "bid_prices", "bid_sizes",
            "ask_prices", "ask_sizes", "crossed"]
    parts = []
    for r in recs:
        files = _files(r, "books")
        if not files and Path(r).is_dir() and not (Path(r) / "books").exists():
            files = [f for f in sorted(Path(r).rglob("*.parquet")) if "bid_prices" in pq.read_schema(f).names]
        for f in files:
            try:
                t = pq.read_table(f, columns=cols)
            except Exception as e:  # noqa: BLE001  (torn part file of a live recorder)
                print(f"  skip unreadable {f}: {e}")
                continue
            if slug_rx is not None:
                keep = np.array([bool(slug_rx.search(s)) for s in t.column("market_slug").to_pylist()])
                t = t.filter(pa.array(keep))
            if t.num_rows:
                parts.append(t.cast(pa.schema([t.schema.field(c) for c in cols])))
    if not parts:
        return None
    return pa.concat_tables(parts, promote_options="permissive")


def load_rec_tables(recs: list[Path], name: str) -> pd.DataFrame:
    out = []
    for r in recs:
        for f in _files(r, name):
            try:
                df = pq.read_table(f).to_pandas()
            except Exception as e:  # noqa: BLE001
                print(f"  skip unreadable {f}: {e}")
                continue
            df["_src"] = str(r)
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_coverage(recs: list[Path]) -> dict:
    try:
        from polyrec import coverage as cov
    except ImportError:
        return {}
    ev = []
    for r in recs:
        ev.extend(cov.read_events(Path(r)))
    return cov.live_intervals(ev) if ev else {}


def gamma_market(slug: str, offline: bool) -> dict | None:
    path = CACHE / "gamma" / f"{slug}.json"
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("closed") or offline:
            return d
    if offline:
        return None
    import fetch_trades as ft            # data/fetch_trades.py: http_json (browser UA, retries)
    g = ft.http_json(f"{GAMMA}/markets?slug={slug}&closed=true") or ft.http_json(f"{GAMMA}/markets?slug={slug}")
    if not g:
        return None
    g = g[0]
    keep = ["slug", "conditionId", "outcomes", "clobTokenIds", "gameStartTime", "eventStartTime", "endDate",
            "startDate", "acceptingOrdersTimestamp", "closed", "orderPriceMinTickSize", "orderMinSize",
            "feesEnabled", "feeSchedule", "feeType", "rewardsMinSize", "rewardsMaxSpread", "sportsMarketType",
            "secondsDelay", "clearBookOnStart", "negRisk"]
    d = {k: g.get(k) for k in keep}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(d))
    return d


def api_trades(slug: str, cond: str, offline: bool, final: bool = True) -> tuple[pd.DataFrame | None, bool]:
    """data-api /trades legs of one market in the data/raw/trades.parquet schema (cached once the
    market is past kickoff). Returns (legs, truncated)."""
    path = CACHE / "trades" / f"{slug}.parquet"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        return pd.read_parquet(path), bool(json.loads(meta.read_text())["truncated"])
    if offline or not cond:
        return None, False
    import fetch_trades as ft
    taker, t1 = ft.fetch_all(cond, True)
    allf, t2 = ft.fetch_all(cond, False)
    taker, allf = ft.trim(taker), ft.trim(allf)
    keys = {}
    for r in taker:
        k = (r[0], r[2], r[3], r[4], r[5], r[6])
        keys[k] = keys.get(k, 0) + 1
    rows = []
    for r in allf:                    # same taker / maker labelling as data/fetch_trades.build_outputs
        k = (r[0], r[2], r[3], r[4], r[5], r[6])
        is_t = keys.get(k, 0) > 0
        if is_t:
            keys[k] -= 1
        rows.append((slug, r[0], r[1], r[4], 1 if r[3] == "BUY" else -1, r[5], r[6], is_t))
    t = pd.DataFrame(rows, columns=["slug", "tx", "timestamp", "outcome_idx", "side", "price", "size", "is_taker"])
    tk = t[t.is_taker]
    agg = pd.Series(np.where(tk.outcome_idx == 0, tk.side, -tk.side), index=tk.tx.values)
    agg = agg[~agg.index.duplicated()]
    t["aggressor_up"] = t.tx.map(agg).fillna(0).astype(np.int8)
    t["tx"] = [bytes.fromhex(x[2:]) for x in t.tx]
    if not final:                     # game not over: do not cache a partial history
        return t, bool(t1 or t2)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path, index=False)
    write_atomic(meta, json.dumps({"truncated": bool(t1 or t2), "fetched_at": int(time.time())}))
    return t, bool(t1 or t2)


# ----------------------------------------------------------------------------- splits
def game_day(slug: str, kickoff: int) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", slug)
    return m.group(1) if m else pd.Timestamp(kickoff, unit="s", tz="UTC").strftime("%Y-%m-%d")


def assign_splits(days: list[str], cfg: dict | None) -> tuple[dict, str]:
    """day -> split id. cfg (--splits JSON): {"days": {"2026-10-16": "train", "2026-10-20..2026-10-24":
    "val", ...}, "default": "exclude" | "auto"}, or {"train": [...], "val": [...], "test": [...]} with
    days / "a..b" ranges. Without cfg: D11 auto rule, oldest ~60% of game days train, next ~20% val,
    newest ~20% test (exclude -> 3 purged)."""
    days = sorted(set(days))
    out = {}
    if cfg:
        spec = dict(cfg.get("days", {}))
        for name in ("train", "val", "test", "exclude", "purged"):
            for d in cfg.get(name, []):
                spec[d] = name
        for key, name in spec.items():
            lo, _, hi = key.partition("..")
            hi = hi or lo
            for d in days:
                if lo <= d <= hi:
                    out[d] = SPLIT_IDS.get(name, 3)
        rest = [d for d in days if d not in out]
        if cfg.get("default", "exclude") != "auto":
            out.update({d: 3 for d in rest})
            return out, "explicit --splits " + json.dumps(cfg, sort_keys=True)
        days = rest
    n = len(days)
    n_tr = max(1, int(round(0.6 * n))) if n else 0
    n_va = int(round(0.2 * n)) if n >= 3 else (1 if n == 2 else 0)
    for i, d in enumerate(days):
        out[d] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    return out, "auto (D11): oldest ~60% of game days train, next ~20% val, newest ~20% test"


# ----------------------------------------------------------------------------- per market
def market_grid(snap: pd.DataFrame, kickoff: int, n_rows_total: int, dt: int, cov_iv, stale_s: float):
    """Book state on the grid t_j = kickoff - (n_rows_total - 1 - j) * dt (last row = kickoff)."""
    t = kickoff - (n_rows_total - 1 - np.arange(n_rows_total, dtype=np.int64)) * dt
    n = len(t)
    X = np.zeros((n, F))
    X[:, TC["const_one"]] = 1.0
    sts = snap.index.values.astype(np.int64)
    j_raw = np.searchsorted(sts, t, side="right") - 1
    X[:, TC["crossed_raw"]] = np.where(j_raw >= 0, snap.raw_crossed.values[np.maximum(j_raw, 0)], 0)
    v = snap[snap.valid.values]
    vts = v.index.values.astype(np.int64)
    j = np.searchsorted(vts, t, side="right") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)
    X[:, TC["book_valid"]] = ok
    age = np.where(ok, t - vts[jj], 0)
    X[:, TC["book_age"]] = age
    X[:, TC["book_from_down"]] = np.where(ok, v.from_down.values[jj], 0)
    for k in range(NLEV):
        bp = np.nan_to_num(v[f"bp{k}"].values[jj], nan=0.0)
        ap = np.nan_to_num(v[f"ap{k}"].values[jj], nan=1.0)
        bs = np.where(bp > 0, v[f"bs{k}"].values[jj], 0.0)
        az = np.where(ap < 1, v[f"as{k}"].values[jj], 0.0)
        X[:, TC[f"bid_px_{k}"]] = np.where(ok, bp, 0.0)
        X[:, TC[f"ask_px_{k}"]] = np.where(ok, ap, 1.0)
        X[:, TC[f"bid_sz_{k}"]] = np.where(ok, bs, 0.0)
        X[:, TC[f"ask_sz_{k}"]] = np.where(ok, az, 0.0)
    X[:, TC["bid_depth_total"]] = np.where(ok, v.bdep.values[jj], 0)
    X[:, TC["ask_depth_total"]] = np.where(ok, v.adep.values[jj], 0)
    if cov_iv is not None:                       # recorder coverage: stale = recorder not live at t
        live = np.zeros(n, bool)
        for a, b in cov_iv:
            live |= (t >= a) & (t <= b)
        stale = ~live
    else:
        stale = age > stale_s
    return t, X, stale, (int(vts[0]) if len(vts) else None)


def features(t, X, legs_row_sub, stale, p_mode, seg, dt, kickoff, first_book):
    """feat.bin rows for the whole window (causal: row j uses grid rows <= j only)."""
    n = len(t)
    bp, bs = X[:, TC["bid_px_0"]:TC["bid_px_0"] + 5], X[:, TC["bid_sz_0"]:TC["bid_sz_0"] + 5]
    ap, az = X[:, TC["ask_px_0"]:TC["ask_px_0"] + 5], X[:, TC["ask_sz_0"]:TC["ask_sz_0"] + 5]
    valid = X[:, TC["book_valid"]] > 0
    crossed = X[:, TC["crossed_raw"]] > 0
    age = X[:, TC["book_age"]]
    bm, am = np.rint(bp * 1000).astype(np.int64), np.rint(ap * 1000).astype(np.int64)
    sub_book = (((bm % 10) != 0) & (bm > 0)).any(1) | (((am % 10) != 0) & (am < 1000)).any(1)
    fine = np.maximum.accumulate(sub_book | legs_row_sub)
    tick = np.where(fine, 0.001, 0.01)
    has_b = valid & (bp[:, 0] > 0)
    has_a = valid & (ap[:, 0] < 1)
    two = has_b & has_a
    mid = 0.5 * (bp[:, 0] + ap[:, 0])
    s1 = bs[:, 0] + az[:, 0]
    micro = np.where(two & (s1 > 0), (bp[:, 0] * az[:, 0] + ap[:, 0] * bs[:, 0]) / np.where(s1 > 0, s1, 1), mid)
    fair = micro if p_mode == "micro" else mid
    p = ffill(fair, two, 0.5)                    # last two-sided value; 0.5 before the first one
    fine_alt = np.maximum.accumulate(fine | (two & ((mid < 0.04) | (mid > 0.96))))
    mark = np.where(two, mid, p).astype(np.float32).astype(np.float64)
    gap = p - mark
    seen = np.maximum.accumulate(two)            # a two-sided book has been seen at or before the row

    def dlag(x, k):                              # x - x[t-k], 0 while x[t-k] was still the 0.5 placeholder
        return np.where(lag(seen, k), x - lag(x, k), 0.0)
    bvol, svol = X[:, TC["buy_vol"]], X[:, TC["sell_vol"]]
    out = np.zeros((n, G))
    out[:, CI["p_logit"]] = logit(p) / 3
    out[:, CI["mark_logit"]] = logit(mark) / 3
    out[:, CI["gap_c"]] = clip(gap * 100 / 5)
    out[:, CI["dgap5_c"]] = clip(dlag(gap, 5) * 100 / 2)
    # z, sdv_bp, sigma_1m_bp, in_twap, delta_usd, ulz*: no underlying -> 0
    to_ko = (kickoff - t).astype(np.float64)
    out[:, CI["log_tau"]] = np.log1p(to_ko) / np.log1p(14400)
    spread_t = np.where(two, (ap[:, 0] - bp[:, 0]) / tick, 1.0)
    out[:, CI["spread_ticks"]] = np.where(two, clip(np.log(np.maximum(spread_t, 1.0)) / 2), 0.0)
    out[:, CI["bid_present"]] = has_b
    out[:, CI["ask_present"]] = has_a
    out[:, CI["imb_l1"]] = np.where(s1 > 0, (bs[:, 0] - az[:, 0]) / np.where(s1 > 0, s1, 1), 0)
    out[:, CI["bid_sz0"]] = np.log1p(bs[:, 0]) / 5
    out[:, CI["ask_sz0"]] = np.log1p(az[:, 0]) / 5
    for k in (5, 30, 120):
        b, s_ = window_sum(bvol, k), window_sum(svol, k)
        out[:, CI[f"flow{k}"]] = (b - s_) / (b + s_ + 5)
    out[:, CI["mret5_c"]] = clip(dlag(mark, 5) * 100 / 2)
    out[:, CI["mret30_c"]] = clip(dlag(mark, 30) * 100 / 5)
    out[:, CI["mret120_c"]] = clip(dlag(mark, 120) * 100 / 5)
    out[:, CI[{300: "len_5m", 900: "len_15m", 14400: "len_4h"}[seg]]] = 1.0
    out[:, CI["tick_fine"]] = fine
    out[:, CI["book_age"]] = np.log1p(age) / 4
    out[:, CI["crossed_raw"]] = crossed
    out[:, CI["mark_is_fv"]] = ~two
    out[:, CI["p_model"]] = p
    out[:, CI["mark"]] = mark
    out[:, CI["liq_mid"]] = -1.0                 # liquidity-reward hook inert (no sports pools modelled)
    out[:, CI["bid_minus_p_c"]] = np.where(has_b, clip((bp[:, 0] - p) * 100 / 5), 0.0)
    out[:, CI["ask_minus_p_c"]] = np.where(has_a, clip((ap[:, 0] - p) * 100 / 5), 0.0)
    sf = np.zeros((n, len(SPORTS_COLS)))
    sf[:, 0] = to_ko
    sf[:, 2] = np.where(two, micro, -1.0)
    sf[:, 3] = np.maximum((t - first_book) / 3600.0, 0.0) if first_book is not None else 0.0
    book = np.zeros(n, BOOK_DTYPE)
    book["bid_px"] = bm
    book["ask_px"] = am
    book["bid_sz"] = bs
    book["ask_sz"] = az
    book["flags"] = (valid * BOOK_VALID | crossed * BOOK_CROSSED | stale * BOOK_STALE
                     | (X[:, TC["book_from_down"]] > 0) * BOOK_FROM_DOWN)
    book["tick"] = np.where(fine, 1, 10)
    book["tick_alt"] = np.where(fine_alt, 1, 10)
    return out, sf, book


def seg_book_same(book: np.ndarray) -> None:
    """SAME flag (ladder identical to the previous row of the same episode) and same_run, in place."""
    lad = np.concatenate([book["bid_px"].astype(np.float64), book["ask_px"], book["bid_sz"], book["ask_sz"]], 1)
    same = np.r_[False, (lad[1:] == lad[:-1]).all(1)]
    n = len(same)
    nxt = np.where(~same, np.arange(n), n)                        # next row >= r with same False
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    book["same_run"] = np.minimum(nxt - np.arange(n), 255)
    book["flags"] = (book["flags"] & np.uint8(0xFF ^ BOOK_SAME)) | (same * BOOK_SAME).astype(np.uint8)


# ----------------------------------------------------------------------------- main build
def build(args) -> dict:
    t0 = time.time()
    recs = [Path(r) for r in (args.rec or [])]
    out = Path(args.out)
    if "jun15" in str(out.resolve()).lower():
        raise PermissionError("never write into data/build_jun15")
    dt, seg = int(args.dt), int(args.seg)
    if seg not in SEG_LENS:
        raise SystemExit(f"--seg must be one of {SEG_LENS} (mm.h samples only these lengths)")
    seg_s = seg * dt
    nseg = int(args.max_hours * 3600 // seg_s)
    if nseg < 1:
        raise SystemExit(f"--max-hours {args.max_hours} is shorter than one segment ({seg} rows x {dt} s = "
                         f"{seg_s / 3600:.2f} h)")
    n_tot = nseg * seg + 1
    slug_rx = re.compile(args.slugs) if args.slugs else None
    now = int(args.now or time.time())

    books = load_books(recs, slug_rx)
    if books is None:
        raise SystemExit("no book rows found under " + ", ".join(map(str, recs)))
    print(f"books: {books.num_rows} rows, {len(set(books.column('market_slug').to_pylist()))} markets "
          f"({time.time() - t0:.1f}s)")
    bpx, bsz, bdep, bn = B.ladder_topk(books, "bid_prices", "bid_sizes")
    apx, asz, adep, an = B.ladder_topk(books, "ask_prices", "ask_sizes")
    bk = pd.DataFrame({
        "slug": books.column("market_slug").to_numpy(zero_copy_only=False),
        "token": books.column("token_id").to_numpy(zero_copy_only=False),
        "outcome": books.column("outcome").to_numpy(zero_copy_only=False),
        "end_at": books.column("market_end_at").cast(pa.timestamp("us", tz="UTC")).cast("int64").to_numpy() // 1_000_000,
        "ts": books.column("captured_at").cast(pa.timestamp("us", tz="UTC")).cast("int64").to_numpy() // 1_000_000,
        "crossed": books.column("crossed").to_numpy(zero_copy_only=False),
        "bdep": bdep, "adep": adep, "bn": bn, "an": an})
    for k in range(NLEV):
        bk[f"bp{k}"], bk[f"bs{k}"], bk[f"ap{k}"], bk[f"as{k}"] = bpx[:, k], bsz[:, k], apx[:, k], asz[:, k]
    del books
    bk = bk.drop_duplicates(["token", "ts"], keep="last")
    rec_mk = load_rec_tables(recs, "markets")
    rec_mk = rec_mk.drop_duplicates("slug", keep="last").set_index("slug") if len(rec_mk) else None
    rec_tr = load_rec_tables(recs, "trades")
    if len(rec_tr):
        rec_tr["slug"] = rec_tr.slug.astype(str)
    cov_all = load_coverage(recs)

    # ---- metadata, kickoff, eligibility
    rows, skipped = [], []
    for slug, g in bk.groupby("slug", sort=True):
        gm = gamma_market(slug, args.offline) if not args.no_gamma else None
        rm = rec_mk.loc[slug] if rec_mk is not None and slug in rec_mk.index else None
        toks = json.loads(gm["clobTokenIds"]) if gm and gm.get("clobTokenIds") else None
        if toks is None and rm is not None and rm.get("token_up"):
            toks = [rm["token_up"], rm["token_down"]]
        if toks is None:                          # last resort: data-raw convention, Up/Yes first
            outs = g.drop_duplicates("token")[["token", "outcome"]]
            if len(outs) != 2:
                skipped.append((slug, "cannot identify outcome-0 token"))
                continue
            o = outs.sort_values("outcome", key=lambda s: ~s.isin(["Up", "Yes"])).token.tolist()
            toks = o
        # kickoff: gamma gameStartTime, else the backfill's `kickoff` column (also gameStartTime), else
        # eventStartTime / endDate (endDate is kickoff + 6 h .. 7 d for some leagues: flagged)
        g_ = gm or {}
        r_ = (lambda k: rm.get(k) if rm is not None and k in rm.index else None)
        ko = to_ts(g_.get("gameStartTime")) or to_ts(r_("kickoff"))
        fallback = ko is None
        if ko is None:
            cands = [g_.get("eventStartTime"), r_("event_start"), g_.get("endDate"), r_("end")]
            ko = next((to_ts(c) for c in cands if to_ts(c) is not None), None) or int(g.end_at.max())
        ko_true = ko
        if args.allow_open:                       # TESTING ONLY: cut open markets at the last book row
            ko = min(ko, int(g.ts.max()) + 1)
        elif ko > now - args.settle_wait:
            skipped.append((slug, f"kickoff {pd.Timestamp(ko, unit='s', tz='UTC')} not yet past"))
            continue
        rg = (lambda k, d: num(rm.get(k), d) if rm is not None else d)
        tick = num((gm or {}).get("orderPriceMinTickSize"), rg("tick_size", 0.01))
        if abs(tick * 1000 - round(tick * 1000)) > 1e-6:
            skipped.append((slug, f"tick {tick} not on the 0.001 grid (int16 mils)"))
            continue
        fs = (gm or {}).get("feeSchedule") or {}
        fees_on = (gm or {}).get("feesEnabled")
        if fees_on is None:
            fees_on = bool(rm.get("fees_enabled")) if rm is not None and rm.get("fees_enabled") is not None else False
        fee_rate = num(fs.get("rate"), rg("fee_rate", 0.0))
        rows.append(dict(
            slug=slug, kickoff=ko, kickoff_true=ko_true, kickoff_fallback=fallback, tok0=str(toks[0]), tok1=str(toks[1]),
            cond=(gm or {}).get("conditionId") or (rm.get("condition_id") if rm is not None else None),
            tick=tick, min_size=num((gm or {}).get("orderMinSize"), rg("min_size", 5.0)),
            fee_rate=fee_rate if fees_on else 0.0,
            fee_exp=num(fs.get("exponent"), rg("fee_exponent", 1.0)),
            rebate=num(fs.get("rebateRate"), rg("fee_rebate_rate", 0.0)),
            rew_min=num((gm or {}).get("rewardsMinSize"), rg("rewards_min_size", 0.0)),
            rew_spread=num((gm or {}).get("rewardsMaxSpread"), rg("rewards_max_spread", 0.0)),
            league=slug.split("-")[0], day=game_day(slug, ko)))
    mk = pd.DataFrame(rows)
    if args.max_markets and len(mk):
        mk = mk.iloc[: args.max_markets]
    print(f"markets eligible {len(mk)}, skipped {len(skipped)} ({time.time() - t0:.1f}s)")
    if not len(mk):
        raise SystemExit("no eligible market; skipped: " + "; ".join(f"{s}: {r}" for s, r in skipped[:10]))
    cfg = json.loads(Path(args.splits).read_text()) if args.splits else None
    day_split, split_rule = assign_splits(mk.day.tolist(), cfg)
    mk["split"] = mk.day.map(day_split).astype(int)
    leagues = sorted(mk.league.unique())
    league_id = {lg: i for i, lg in enumerate(leagues)}

    # ---- pass 1: per market grid + features; keep segments in memory per market only as small
    # metadata, the big arrays are spooled to a temp dir and concatenated in final order.
    out.mkdir(parents=True, exist_ok=True)
    spool = Path(tempfile.mkdtemp(prefix=".spool_", dir=out))
    segs, stats = [], {"markets": 0, "segments_kept": 0, "segments_dropped_cover": 0, "segments_dropped_no_two_sided": 0,
                       "segments_dropped_trades_truncated": 0,
                       "txs_dropped_inconsistent": 0, "legs_after_kickoff_dropped": 0, "legs_before_window": 0,
                       "book_rows_after_kickoff_dropped": 0, "markets_trades_truncated": 0,
                       "markets_without_trades": 0, "trade_source": {}}
    bgroups = {s: g for s, g in bk.groupby("slug", sort=False)}
    rec_tr_by = {}
    if len(rec_tr):
        for (s, src), g in rec_tr.groupby(["slug", "_src"], sort=False):
            if s not in rec_tr_by or len(g) > len(rec_tr_by[s]):
                rec_tr_by[s] = g
    for _, m in mk.iterrows():
        slug, ko = m.slug, int(m.kickoff)
        g = bgroups[slug]
        stats["book_rows_after_kickoff_dropped"] += int((g.ts >= ko).sum())
        g = g[g.ts < ko].copy()
        g = g[g.token.isin([m.tok0, m.tok1])]
        g["is_up"] = g.token == m.tok0
        if not len(g):
            continue
        snap = B.merged_snapshots(g)
        if not snap.valid.any():
            skipped.append((slug, "no uncrossed book before kickoff"))
            continue
        cov_iv = cov_all.get(slug)
        t, X, stale, first_book = market_grid(snap, ko, n_tot, dt, cov_iv, args.stale_s)
        # ---- trades
        tr, trunc, src = rec_tr_by.get(slug), False, "recorder"
        if tr is None and not args.no_fetch_trades:
            tr, trunc = api_trades(slug, m.cond, args.offline, final=m.kickoff_true < now - args.settle_wait)
            src = "data-api"
        if tr is None or not len(tr):
            src = "none"
            stats["markets_without_trades"] += 1
        stats["trade_source"][src] = stats["trade_source"].get(src, 0) + 1
        stats["markets_trades_truncated"] += int(trunc)
        legs_tbl, earliest = None, None
        sub_row = np.zeros(n_tot, bool)
        if tr is not None and len(tr):
            tr = tr.copy()
            tr["px_up"] = np.where(tr.outcome_idx == 0, tr.price.astype(np.float64), 1.0 - tr.price.astype(np.float64))
            earliest = int(tr.timestamp.min()) if trunc else None    # legs before it may be missing
            stats["legs_after_kickoff_dropped"] += int((tr.timestamp >= ko).sum())
            tr = tr[tr.timestamp < ko]
            taker, maker = tr[tr.is_taker], tr[~tr.is_taker].copy()
            tx_size = taker.groupby("tx")["size"].sum().astype(np.float64)
            tx_side = taker.groupby("tx")["aggressor_up"].first()
            msum = maker.groupby("tx")["size"].sum()
            ok_tx = msum.index[(msum - tx_size.reindex(msum.index)).abs().fillna(np.inf) < 0.01]
            side_ok = maker.aggressor_up.values == tx_side.reindex(maker.tx).values
            bad = set(maker.tx[~side_ok]) | (set(msum.index) - set(ok_tx))
            stats["txs_dropped_inconsistent"] += len(bad | (set(taker.tx) - set(maker.tx)))
            maker = maker[maker.tx.isin(ok_tx) & ~maker.tx.isin(bad)]
            lo = t[0] - dt + 1
            stats["legs_before_window"] += int((maker.timestamp < lo).sum())
            maker = maker[maker.timestamp >= lo]
            if len(maker):
                maker["row"] = np.ceil((maker.timestamp.values - t[0]) / dt).astype(np.int64).clip(0)
                maker["tx_size"] = maker.tx.map(tx_size).astype(np.float64)
                maker["px_mil"] = np.rint(maker.px_up * 1000).astype(np.int64)
                maker["ord"] = np.arange(len(maker))
                fo = maker.groupby("tx")["ord"].transform("min")
                maker = maker.assign(first_ord=fo, key=np.where(maker.aggressor_up > 0, maker.px_mil, -maker.px_mil))
                maker = maker.sort_values(["row", "first_ord", "key", "ord"], kind="stable")
                legs_tbl = maker
                r = maker.row.values
                sz = maker["size"].values.astype(np.float64)
                px = maker.px_up.values
                ag = maker.aggressor_up.values
                for sgn, vcol, wcol, xcol, fn in ((1, "buy_vol", "buy_vwap", "buy_max_px", np.maximum),
                                                  (-1, "sell_vol", "sell_vwap", "sell_min_px", np.minimum)):
                    msk = ag == sgn
                    vol = np.bincount(r[msk], weights=sz[msk], minlength=n_tot)
                    pv = np.bincount(r[msk], weights=(px * sz)[msk], minlength=n_tot)
                    ext = np.full(n_tot, -np.inf if sgn == 1 else np.inf)
                    fn.at(ext, r[msk], px[msk])
                    has = vol > 0
                    X[:, TC[vcol]] = vol
                    X[:, TC[wcol]] = np.where(has, pv / np.where(has, vol, 1), 0)
                    X[:, TC[xcol]] = np.where(np.isfinite(ext) & has, ext, 0)
                txr = pd.DataFrame({"r": r, "tx": maker.tx.values}).drop_duplicates("tx")
                X[:, TC["trade_count"]] = np.bincount(txr.r.values, minlength=n_tot)
                sub = (maker.px_mil.values % 10) != 0
                sub_row[r[sub]] = True
        feat, sf, book = features(t, X, sub_row, stale, args.p_model, seg, dt, ko, first_book)
        stats["markets"] += 1
        # ---- segments
        for k in range(nseg):
            a, b = k * seg, (k + 1) * seg                     # rows [a, b] inclusive
            sl = slice(a, b + 1)
            covered = X[sl, TC["book_valid"]] > 0
            two = feat[sl, CI["mark_is_fv"]] < 0.5
            cover = float(covered.mean())
            if cover < args.min_cover:
                stats["segments_dropped_cover"] += 1
                continue
            if not two.any():
                stats["segments_dropped_no_two_sided"] += 1
                continue
            if earliest is not None and t[a] < earliest and not args.keep_truncated:
                stats["segments_dropped_trades_truncated"] += 1
                continue
            Xs = X[sl].copy()
            Xs[:, TC["seconds_to_close"]] = np.arange(seg, -1, -1)       # rows to the episode end
            fs = feat[sl].copy()
            fs[:, CI["tau_frac"]] = np.arange(seg, -1, -1) / seg
            sfs = sf[sl].copy()
            sfs[:, 1] = np.arange(seg, -1, -1) * dt
            bks = book[sl].copy()
            seg_book_same(bks)
            assert np.isfinite(Xs).all() and np.isfinite(fs).all(), slug
            assert np.abs(fs[:, :N_OBS]).max() <= CLIP + 1e-9, slug
            settle = float(np.float32(fs[-1, CI["mark"]]))
            lg = None
            if legs_tbl is not None:
                lg = legs_tbl[(legs_tbl.row >= a) & (legs_tbl.row <= b)]
                if len(lg):
                    first = lg.tx.values != np.roll(lg.tx.values, 1)
                    first[0] = True
                    L = np.zeros(len(lg), FILL_DTYPE)
                    L["row"] = lg.row.values - a                      # local; globalised at write
                    L["side"] = lg.aggressor_up.values
                    L["first"] = first
                    L["px"] = lg.px_mil.values
                    L["price"] = lg.px_up.values
                    L["size"] = lg["size"].values
                    L["tx_size"] = lg.tx_size.values
                    lg = L
                else:
                    lg = None
            sid = f"{slug}#k{nseg - 1 - k}"           # k0 = the segment ending at kickoff
            f = spool / f"{len(segs)}.npz"
            np.savez(f, X=Xs.astype(np.float32), feat=fs.astype(np.float32), sf=sfs.astype(np.float32), book=bks,
                     legs=lg if lg is not None else np.zeros(0, FILL_DTYPE))
            flags = (FLAG_SETTLE_PX | (FLAG_HAS_TRADES if lg is not None else 0)
                     | (FLAG_TRADES_TRUNC if trunc else 0) | (FLAG_HEAD_PADDED if cover < 1 else 0)
                     | (FLAG_AT_KICKOFF if k == nseg - 1 else 0) | (FLAG_KICKOFF_FALLBACK if m.kickoff_fallback else 0)
                     | (FLAG_COVERAGE if cov_iv is not None else 0))
            segs.append(dict(sid=sid, slug=slug, file=f, open_ts=int(t[a]), end_ts=int(t[b]), split=int(m.split),
                             league=league_id[m.league], settle=settle, flags=flags, m=m, cover=cover,
                             n_legs=0 if lg is None else len(lg), two_frac=float(two.mean()), kickoff=ko))
            stats["segments_kept"] += 1
    if not segs:
        shutil.rmtree(spool, ignore_errors=True)
        raise SystemExit(f"no segment kept: {json.dumps(stats)}")

    # ---- pass 2: final order (split, end_ts, sid), contiguous splits; stream the files
    segs.sort(key=lambda s: (s["split"], s["end_ts"], s["sid"]))
    n_rows = seg + 1
    N = n_rows * len(segs)
    recs_arr = np.zeros(len(segs), MARKET_DTYPE)
    fh = {name: open(out / f".{name}.tmp{os.getpid()}", "wb") for name in
          ("ticks.bin", "feat.bin", "book.bin", "fills.bin", "sports_feat.bin")}
    fill_counts = np.zeros(N, np.int64)
    n_fills = 0
    for i, s in enumerate(segs):
        z = np.load(s["file"])
        r0 = i * n_rows
        z["X"].tofile(fh["ticks.bin"])
        z["feat"].tofile(fh["feat.bin"])
        z["book"].tofile(fh["book.bin"])
        z["sf"].tofile(fh["sports_feat.bin"])
        legs = z["legs"].copy()
        if len(legs):
            fill_counts[r0:r0 + n_rows] += np.bincount(legs["row"], minlength=n_rows)
            legs["row"] += r0
            legs.tofile(fh["fills.bin"])
            n_fills += len(legs)
        m = s["m"]
        recs_arr[i] = (r0, s["open_ts"], s["end_ts"], n_rows, seg, s["league"], int(s["settle"] >= 0.5),
                       s["split"], s["flags"], 0.01, m.fee_rate, m.fee_exp, m.min_size, m.rew_min, m.rew_spread,
                       s["settle"], m.rebate, i, m.tick)
    for f in fh.values():
        f.close()
    for name in fh:
        os.replace(out / f".{name}.tmp{os.getpid()}", out / name)
    shutil.rmtree(spool, ignore_errors=True)
    index = np.concatenate([[0], np.cumsum(fill_counts)]).astype(np.int32)
    assert index[-1] == n_fills
    write_atomic(out / "fill_index.bin", index)
    write_atomic(out / "markets.bin", recs_arr)

    # ---- meta.json, splits.json, feat_meta.json
    ticks = np.memmap(out / "ticks.bin", np.float32, "r", shape=(N, F))
    te = recs_arr[recs_arr["split"] == 2]
    if len(te):
        lo, hi = int(te["row_offset"].min()), int((te["row_offset"] + te["n_rows"]).max())
        assert hi - lo == int(te["n_rows"].sum())
        test_sha = hashlib.sha256(np.ascontiguousarray(ticks[lo:hi]).tobytes()).hexdigest()
    else:
        lo = hi = 0
        test_sha = None
    del ticks
    slugs = [s["sid"] for s in segs]
    write_atomic(out / "splits.json", json.dumps({s["sid"]: s["split"] for s in segs}, indent=0, sort_keys=True))
    layout = [{"name": n_, "offset": MARKET_DTYPE.fields[n_][1], "type": B.C_TYPES[MARKET_DTYPE.fields[n_][0].str],
               "size": MARKET_DTYPE.fields[n_][0].itemsize} for n_ in MARKET_DTYPE.names]
    seg_name = {300: "5m", 900: "15m", 14400: "4h"}[seg]
    counts = {SPLIT_NAMES[k]: {"markets": int((recs_arr["split"] == k).sum()),
                               "rows": int(recs_arr["n_rows"][recs_arr["split"] == k].sum()),
                               "games": len({s["slug"] for s in segs if s["split"] == k}),
                               "days": sorted({d for d, v in day_split.items() if v == k}),
                               **{cl: int(((recs_arr["split"] == k) & (recs_arr["contract_len_s"] == L)).sum())
                                  for cl, L in B.CLEN.items()}} for k in SPLIT_NAMES}
    sizes = {f: (out / f).stat().st_size for f in ("ticks.bin", "feat.bin", "book.bin", "fills.bin",
                                                   "fill_index.bin", "markets.bin", "sports_feat.bin")}
    meta = {
        "build_version": BUILD_VERSION, "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "F": F, "N": int(N), "n_markets": len(segs), "dtype": "float32", "order": "row-major [N, F]",
        "columns": TCOLS, "column_index": TC,
        "market_record": {"itemsize": MARKET_DTYPE.itemsize, "fields": layout,
                          "flags": {"HAS_TRADES": FLAG_HAS_TRADES, "SETTLE_PX": FLAG_SETTLE_PX,
                                    "TRADES_TRUNCATED": FLAG_TRADES_TRUNC, "HEAD_PADDED": FLAG_HEAD_PADDED,
                                    "AT_KICKOFF": FLAG_AT_KICKOFF, "KICKOFF_FALLBACK": FLAG_KICKOFF_FALLBACK,
                                    "COVERAGE": FLAG_COVERAGE}},
        "coin_ids": league_id,
        "contract_len_s": {seg_name: seg},
        "splits": {"ids": SPLIT_NAMES, "rule": split_rule + "; by game day (slug date, else kickoff UTC date); "
                   "every segment of a game takes the game's split", "day_split": day_split, "counts": counts},
        "market_order": "sorted by (split, end_ts, segment id); each split's rows are contiguous",
        "slugs": slugs,
        "test_rows": {"row_lo": lo, "row_hi": hi, "sha256": test_sha},
        "normalisation_train_only": {},
        "sentinels": {"empty_bid_px": 0.0, "empty_ask_px": 1.0, "empty_size": 0.0,
                      "no_trade_vwap_maxmin": 0.0, "no_book": "book_valid=0, book_age=0, empty ladders"},
        "sports": {
            "dt_s": dt, "seg_rows": seg, "seg_hours": seg_s / 3600, "segments_per_game": nseg,
            "max_hours_before_kickoff": args.max_hours, "window_rows_per_game": n_tot,
            "min_cover": args.min_cover, "p_model": args.p_model, "stale_s": args.stale_s,
            "allow_open_TESTING_ONLY": bool(args.allow_open),
            "row_time": "row j of a segment is unix time open_ts + j * dt_s; the last row (end_ts) is kickoff "
                        "for AT_KICKOFF segments. Every env quantity counted in rows (latency, match delay, dt "
                        "params, markout_h, flow/mret windows, minute boundaries) is in units of dt_s seconds.",
            "kickoff": "gamma gameStartTime; fallback eventStartTime, endDate, recorder markets, books market_end_at",
            "cut": "book rows with captured_at >= kickoff and trade legs with block time >= kickoff are dropped",
            "settlement": "log_strike_vs_ul_open = settle price of Up (outcome 0) = feat mark at the segment's last "
                          "row (float32). winner_up = int(settle >= 0.5) is a FALLBACK for the stock mm.h only, NOT "
                          "the game result; the stock env's terminal P&L is wrong by inv * (winner_up - settle). "
                          "Use env_sports/ (flag SETTLE_PX) for P&L.",
            "seconds_to_close_column": "ticks.bin seconds_to_close is in ROWS to the segment end (seg..0), so "
                                       "data/load.py verify() holds for dt_s = 1",
            "frame": "outcome 0 (first clobTokenId; a team or Yes) plays Up",
            "tick_size": "0.01 base; markets.tick_size_final = gamma orderPriceMinTickSize",
            "trades_from": "recorder trades/ when present (largest per slug across --rec dirs), else data-api /trades",
            "file_sizes_bytes": sizes, "bytes_per_row": BYTES_PER_ROW,
            "stats": stats, "skipped": skipped[:200], "n_skipped": len(skipped),
            "games": sorted({s["slug"] for s in segs}),
        },
        "conventions": {
            "grid": "dt_s grid; see sports.row_time", "frame": "outcome-0 token. Outcome-1 snapshots mapped as "
            "Up bid = 1 - ask, Up ask = 1 - bid (data/build.py merged_snapshots)",
            "book": "latest snapshot with captured_at <= t and < kickoff (data/build.py rules)",
            "trades": "maker legs with block time in (t - dt_s, t]; txs whose maker legs do not sum to the taker "
                      "size (abs dev >= 0.01) or disagree on the aggressor are dropped",
            "underlying": "none: ul_* columns are 0",
            "leakage": "the game result is never read",
        },
        "row_counts": {"ticks": int(N), "markets": len(segs)},
    }
    write_atomic(out / "meta.json", json.dumps(meta, indent=1, default=str))
    ref = PM / "data" / "build" / "feat_meta.json"
    if ref.exists():
        rc = json.loads(ref.read_text())
        assert rc["columns"] == FCOLS and rc["column_index"] == CI and rc["n_obs_features"] == N_OBS, \
            "frozen feat.bin layout differs from data/build/feat_meta.json"
    docs = {c: "" for c in FCOLS}
    docs.update({
        "p_logit": "logit(p_model) / 3", "mark_logit": "logit(mark) / 3; mark = book mid if two-sided, else p_model",
        "gap_c": "(p_model - mark) cents / 5 (0 with p_model=mid)", "dgap5_c": "5-row change of gap, cents / 2",
        "z": "0 (no underlying)", "sdv_bp": "0", "sigma_1m_bp": "0",
        "tau_frac": "rows to the segment (episode) end / seg", "log_tau": "log1p(seconds to KICKOFF) / log1p(14400)",
        "in_twap": "0", "delta_usd": "0", "ulz5": "0", "ulz30": "0", "ulz120": "0",
        "len_5m": "segment-length one-hot (seg rows)", "len_15m": "segment-length one-hot",
        "len_4h": "segment-length one-hot (seg = 14400 rows)",
        "p_model": f"fair value = book {args.p_model} of the last two-sided valid book (forward-filled; 0.5 before)",
        "mark": "two-sided mid, else p_model", "delta_usd_raw": "0", "sigma": "0", "sqrt_v": "0",
        "liq_mid": "-1 (hook inert)", "liq_comp": "0", "liq_pi": "0"})
    for c in FCOLS:
        docs[c] = docs[c] or "as data/build (data/features.py), windows in rows"
    fmeta = {
        "features_version": "sports-1", "built_at_utc": meta["built_at_utc"], "N": int(N), "G": G,
        "n_obs_features": N_OBS, "dtype": "float32", "order": "row-major [N, G]",
        "columns": FCOLS, "column_index": CI, "column_docs": docs,
        "obs_block": "identical to data/build: columns 0..32 -> obs 0..32; bid_minus_p_c / ask_minus_p_c -> obs 47/48",
        "causality": "row t uses grid rows <= t of the same game (lags reach into the previous segment) and maker "
                     "legs with block time <= t; no outcome information",
        "book_record": {"itemsize": 64, "flags": "bit0 valid, bit1 crossed_raw, bit2 STALE (recorder not live at t "
                        f"when a coverage log exists, else book_age > {args.stale_s:g} s), bit3 from outcome 1, "
                        "bit4 ladder same as previous row of the segment", "layout": "as data/build/feat_meta.json"},
        "fill_record": {"itemsize": 20, "layout": "as data/build/feat_meta.json; row = global row of the segment"},
        "fill_index": "int32 [N + 1]: legs of row r are fills[fill_index[r]:fill_index[r+1]]",
        "sports_feat": {"file": "sports_feat.bin", "dtype": "float32", "shape": [int(N), len(SPORTS_COLS)],
                        "columns": SPORTS_COLS,
                        "docs": {"seconds_to_kickoff": "kickoff - t (s)", "seconds_to_seg_end": "end_ts - t (s)",
                                 "microprice": "top-of-book size-weighted price if two-sided, else -1",
                                 "hours_since_first_book": "(t - first book row) / 3600, 0 before"},
                        "note": "not read by mm.h; the frozen 44-column feat.bin has no slot for it"},
        "stats": stats,
    }
    write_atomic(out / "feat_meta.json", json.dumps(fmeta, indent=1, default=str))
    total = sum(sizes.values())
    print(f"wrote {out}: {len(segs)} segments ({len({s['slug'] for s in segs})} games) x {n_rows} rows = {N} rows, "
          f"{n_fills} legs, {total / 1e6:.1f} MB ({BYTES_PER_ROW} B/row; {n_tot * BYTES_PER_ROW / 1e6:.1f} MB per "
          f"game window); {time.time() - t0:.1f}s")
    print("splits:", json.dumps({k: {kk: v[kk] for kk in ('markets', 'games', 'days')} for k, v in counts.items()}))
    print("stats:", json.dumps(stats))
    return meta


def verify(out: Path) -> None:
    """Sports-build checks (data/load.py verify() also holds when dt_s = 1)."""
    import load as L
    out = Path(out)
    ticks, mk, meta = L.load(out)
    sp = meta["sports"]
    feat = np.memmap(out / "feat.bin", np.float32, "r", shape=(meta["N"], G))
    book = np.fromfile(out / "book.bin", BOOK_DTYPE)
    fills = np.fromfile(out / "fills.bin", FILL_DTYPE)
    idx = np.fromfile(out / "fill_index.bin", np.int32)
    sf = np.fromfile(out / "sports_feat.bin", np.float32).reshape(meta["N"], len(SPORTS_COLS))
    assert len(book) == meta["N"] and len(idx) == meta["N"] + 1 and idx[-1] == len(fills)
    assert (np.diff(fills["row"]) >= 0).all() if len(fills) else True
    assert (mk["n_rows"] == mk["contract_len_s"] + 1).all() and set(mk["contract_len_s"]) <= set(SEG_LENS)
    assert (mk["end_ts"] - mk["open_ts"] == mk["contract_len_s"] * sp["dt_s"]).all()
    assert np.isfinite(feat).all() and np.abs(feat[:, :N_OBS]).max() <= CLIP
    last = mk["row_offset"] + mk["n_rows"] - 1
    assert np.array_equal(mk["log_strike_vs_ul_open"], feat[last, CI["mark"]]), "settle != last-row mark"
    ko = mk["flags"] & FLAG_AT_KICKOFF > 0
    assert (sf[last[ko], 0] == 0).all(), "AT_KICKOFF segment does not end at kickoff"
    assert (sf[:, 0] >= 0).all(), "row after kickoff"
    for m in mk:
        r0, n = int(m["row_offset"]), int(m["n_rows"])
        assert not (book["flags"][r0] & BOOK_SAME)
        if idx[r0 + n] > idx[r0]:
            assert fills["row"][idx[r0]] >= r0 and fills["row"][idx[r0 + n] - 1] < r0 + n
    vol = np.bincount(fills["row"], weights=fills["size"].astype(np.float64), minlength=meta["N"])
    tv = np.asarray(ticks[:, TC["buy_vol"]], np.float64) + np.asarray(ticks[:, TC["sell_vol"]], np.float64)
    assert np.abs(vol - tv).max() < 1e-2, "leg volume != ticks volume"
    if sp["dt_s"] == 1:
        L.verify(out)
    print(f"sports verify OK: {len(mk)} segments, N={meta['N']}, legs={len(fills)}")


# ----------------------------------------------------------------------------- selftest
def synth(root: Path, n_games: int = 5, seed: int = 0, lead_s: int = 1000):
    """Synthetic recorder output (books / trades / markets in the recorder schemas): n_games binary
    games on n_games consecutive days; books every ~3 s from lead_s before kickoff to 300 s after;
    trades with maker legs on both outcome tokens, some after kickoff. Game 1 is listed late."""
    from polyrec.schema import BOOK_SCHEMA, TRADES_SCHEMA, MARKETS_SCHEMA
    rng = np.random.default_rng(seed)
    base = int(pd.Timestamp("2026-10-01T18:00:00Z").timestamp())
    for gi in range(n_games):
        slug = f"nhl-aaa-bb{gi}-{pd.Timestamp(base + gi * 86400, unit='s').strftime('%Y-%m-%d')}"
        ko = base + gi * 86400
        day = pd.Timestamp(ko, unit="s").strftime("%Y-%m-%d")
        toks = [str(10 ** 20 + 2 * gi), str(10 ** 20 + 2 * gi + 1)]
        start = ko - (500 if gi == 1 else lead_s)
        ts = np.arange(start, ko + 300, 3)
        mid = np.clip(0.45 + np.cumsum(rng.normal(0, 0.004, len(ts))), 0.05, 0.95)
        brows, trows = [], []
        for i, (t_, m_) in enumerate(zip(ts, mid)):
            bb = np.floor(m_ * 100) / 100
            ba = bb + 0.01 * (1 + (i % 3 == 0))
            bids = [round(bb - 0.01 * k, 2) for k in range(8)]
            asks = [round(ba + 0.01 * k, 2) for k in range(8)]
            bsz = list(rng.integers(10, 500, 8).astype(float))
            asz = list(rng.integers(10, 500, 8).astype(float))
            kb = [k for k in range(8) if 0 < bids[k] < 1]           # ladders stay inside (0, 1)
            ka = [k for k in range(8) if 0 < asks[k] < 1]
            bids, bsz = [bids[k] for k in kb], [bsz[k] for k in kb]
            asks, asz = [asks[k] for k in ka], [asz[k] for k in ka]
            crossed = (i % 97 == 5)
            if crossed:
                bids = [asks[0]] + bids[1:]
            for o, (tok, name) in enumerate(zip(toks, ["Bruins", "Sabres"])):
                if o == 0:
                    bp, bs_, ap, as_ = bids, bsz, asks, asz
                else:
                    bp, bs_ = [round(1 - a, 2) for a in asks], asz
                    ap, as_ = [round(1 - b, 2) for b in bids], bsz
                brows.append(dict(market_slug=slug, coin=None, contract_length=None, token_id=tok, outcome=name,
                                  winner=None, is_winning_outcome=None,
                                  market_end_at=pd.Timestamp(ko, unit="s", tz="UTC"),
                                  captured_at=pd.Timestamp(int(t_), unit="s", tz="UTC"),
                                  seconds_to_close=float(ko - t_), bid_prices=bp, bid_sizes=bs_, ask_prices=ap,
                                  ask_sizes=as_, best_bid=bp[0], best_ask=ap[0], crossed=bool(bp[0] >= ap[0]),
                                  reconciled=False))
            if rng.random() < 0.5:                                   # a taker tx
                txh = rng.bytes(32)
                buy = rng.random() < 0.5
                agg = 1 if buy else -1
                lv = [asks[0], asks[1]] if buy else [bids[0], bids[1]]
                sizes = [float(rng.integers(1, 40)), float(rng.integers(0, 20))]
                lv, sizes = [p for p, s in zip(lv, sizes) if s > 0], [s for s in sizes if s > 0]
                tt = int(t_) + int(rng.integers(0, 3))
                if buy:   # taker buys outcome 0 at the touch
                    trows.append((slug, tt, 0, 1, float(np.mean(lv)), float(sum(sizes)), True, agg, txh))
                else:     # taker buys outcome 1 (= sells Up)
                    trows.append((slug, tt, 1, 1, float(1 - np.mean(lv)), float(sum(sizes)), True, agg, txh))
                for p, s in zip(lv, sizes):
                    if rng.random() < 0.5:   # maker on outcome 0
                        trows.append((slug, tt, 0, -1 if buy else 1, p, s, False, agg, txh))
                    else:                    # mirrored maker on outcome 1
                        trows.append((slug, tt, 1, 1 if buy else -1, round(1 - p, 2), s, False, agg, txh))
        bt = pa.Table.from_pylist(brows, schema=BOOK_SCHEMA)
        (root / "books" / day).mkdir(parents=True, exist_ok=True)
        pq.write_table(bt, root / "books" / day / f"books_{day}T18.parquet")
        td = pd.DataFrame(trows, columns=TRADES_SCHEMA.names)
        tt_ = pa.table({"slug": pa.array(td.slug).dictionary_encode().cast(pa.dictionary(pa.int32(), pa.string())),
                        "timestamp": pa.array(td.timestamp, pa.int64()), "outcome_idx": pa.array(td.outcome_idx, pa.int8()),
                        "side": pa.array(td.side, pa.int8()), "price": pa.array(td.price, pa.float32()),
                        "size": pa.array(td["size"], pa.float32()), "is_taker": pa.array(td.is_taker),
                        "aggressor_up": pa.array(td.aggressor_up, pa.int8()),
                        "tx": pa.array(td.tx, pa.binary(32))})
        (root / "trades" / day).mkdir(parents=True, exist_ok=True)
        pq.write_table(tt_, root / "trades" / day / f"trades_{day}T18.parquet")
        mrow = {n_: None for n_ in MARKETS_SCHEMA.names}
        mrow.update(slug=slug, condition_id="0x" + "0" * 64, token_up=toks[0], token_down=toks[1],
                    created_start=pd.Timestamp(ko - 30 * 86400, unit="s", tz="UTC"),
                    end=pd.Timestamp(ko, unit="s", tz="UTC"), tick_size=0.01, min_size=5.0, fees_enabled=True,
                    fee_rate=0.05, fee_exponent=1.0, fee_rebate_rate=0.15, winner_up=1)
        (root / "markets" / day).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([mrow], schema=MARKETS_SCHEMA), root / "markets" / day / f"markets_{day}T18.parquet")


ENV_PROBE = r"""
import json, os, sys, numpy as np
sys.path.insert(0, %(train)r)
import envs
out, env_dir = %(out)r, %(env_dir)r
ad = envs.get_adapter('real')
if env_dir:                                   # the patched copy (fractional settlement)
    import importlib.util
    sys.path.insert(0, env_dir)
    spec = importlib.util.spec_from_file_location('pm_env_sports', os.path.join(env_dir, 'env.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ad.mod = mod
res = {}
env = ad.make_train(num_envs=16, seed=1, data_dir=out, dt=%(dtp)d, match_delay=%(md)s)
obs, _ = env.reset()
assert obs.shape == (16, ad.obs_size), obs.shape
rng = np.random.default_rng(0)
terms = 0
for i in range(400):
    a = np.stack([rng.integers(0, 7, 16), rng.integers(0, 7, 16), rng.integers(0, 3, 16)], 1).astype(np.int32)
    obs, r, term, _, info = env.step(a)
    assert np.isfinite(obs).all() and np.isfinite(r).all()
    terms += int(term.sum())
res['train_terminals'] = terms
env.close()
ev, n = ad.make_eval(split=0, seed=0, data_dir=out, dt=%(dtp)d, match_delay=%(md)s)
obs, _ = ev.reset()
while ev.active.any():
    a = np.full((n, 3), 3, np.int32); a[:, 2] = 2           # join both sides, 5x size
    ev.step(a)
recs = ad.eval_records(ev)
res['eval'] = sorted([(r['market'], round(r['pnl'], 6), round(r['fills'], 1), round(r['final_abs_inv'], 3))
                      for r in recs])
print('PROBE' + json.dumps(res))
"""


def env_probe(out: Path, env_dir: Path | None, dtp: int = 1, md: float = -1) -> dict:
    py = sys.executable
    code = ENV_PROBE % {"train": str(PM / "train"), "out": str(out), "env_dir": str(env_dir) if env_dir else "",
                        "dtp": dtp, "md": md}
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, cwd=str(PM), timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"env probe failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    line = [x for x in r.stdout.splitlines() if x.startswith("PROBE")][-1]
    return json.loads(line[5:])


def build_env_sports() -> Path | None:
    d = HERE / "env_sports"
    if not (d / "mm.h").exists():
        return None
    so = list(d.glob("binding*.so"))
    src_t = max((d / f).stat().st_mtime for f in ("mm.h", "binding.c"))
    if not so or so[0].stat().st_mtime < src_t:
        r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=d, capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError("env_sports build failed:\n" + r.stderr[-3000:])
    return d


def selftest(args) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sports_selftest_", dir=args.tmp or None))
    try:
        rec = tmp / "rec"
        synth(rec)
        outs = {}
        for wu in ("a", "b"):
            o = tmp / f"build_{wu}"
            a = argparse.Namespace(**{**vars(args), "rec": [str(rec)], "out": str(o), "dt": 1, "seg": 300,
                                      "max_hours": 900 / 3600, "min_cover": 0.6, "offline": True, "no_gamma": True,
                                      "no_fetch_trades": True, "splits": None, "slugs": None, "allow_open": False,
                                      "now": None, "max_markets": 0, "keep_truncated": False})
            meta = build(a)
            if wu == "b":    # same build with the 0/1 fallback flipped: the patched env must not care
                mk = np.fromfile(o / "markets.bin", MARKET_DTYPE)
                mk["winner_up"] = 1 - mk["winner_up"]
                write_atomic(o / "markets.bin", mk)
            verify(o)
            outs[wu] = o
        sp = meta["sports"]["stats"]
        assert sp["legs_after_kickoff_dropped"] > 0 and sp["book_rows_after_kickoff_dropped"] > 0
        assert meta["splits"]["counts"]["train"]["games"] == 3 and meta["splits"]["counts"]["test"]["games"] == 1
        print("selftest build OK:", json.dumps({k: v["markets"] for k, v in meta["splits"]["counts"].items()}))
        # dt>1 build (no load.verify, own verify only)
        o6 = tmp / "build_dt6"
        build(argparse.Namespace(**{**vars(args), "rec": [str(rec)], "out": str(o6), "dt": 6, "seg": 300,
                                    "max_hours": 1800 / 3600, "min_cover": 0.3, "offline": True, "no_gamma": True,
                                    "no_fetch_trades": True, "splits": None, "slugs": None, "allow_open": False,
                                    "now": None, "max_markets": 0, "keep_truncated": False}))
        verify(o6)
        if args.no_env:
            return
        pa_ = env_probe(outs["a"], None)
        pb = env_probe(outs["b"], None)
        print("stock env.py (data_dir=build): train terminals", pa_["train_terminals"], "eval", pa_["eval"][:3])
        assert pa_["train_terminals"] > 0 and len(pa_["eval"]) > 0
        diff_stock = sum(abs(x[1] - y[1]) for x, y in zip(pa_["eval"], pb["eval"]))
        print(f"stock env: |pnl(winner_up fallback) - pnl(flipped fallback)| summed = {diff_stock:.4f} "
              "(nonzero = the stock env settles at the 0/1 field)")
        env_probe(o6, None, dtp=1, md=0)
        print("stock env loads the dt=6 build too (dt=1 row, match_delay=0)")
        d = build_env_sports()
        if d is None:
            print("env_sports/ not present: settlement test skipped")
            return
        qa, qb = env_probe(outs["a"], d), env_probe(outs["b"], d)
        diff = sum(abs(x[1] - y[1]) for x, y in zip(qa["eval"], qb["eval"]))
        assert diff < 1e-4, f"patched env still depends on winner_up: {diff}"
        same_fills = all(x[2] == y[2] for x, y in zip(pa_["eval"], qa["eval"]))
        assert same_fills, "patched env changed the fills"
        print(f"env_sports: pnl independent of winner_up (diff {diff:.2e}); fills identical to the stock env")
        print("SELFTEST OK")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("kept", tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", action="append", help="recorder out dir (books/ trades/ markets/ coverage/) or a dir/file "
                    "of book parquet in the data/raw schema (e.g. the PolyOrderBooks backfill); repeatable")
    ap.add_argument("--out", default=str(HERE / "build"))
    ap.add_argument("--dt", type=int, default=1, help="grid step in seconds (1 = the crypto build's grid)")
    ap.add_argument("--seg", type=int, default=14400, help="rows per episode: 300 / 900 / 14400 (mm.h buckets)")
    ap.add_argument("--max-hours", type=float, default=24.0, help="window before kickoff (floored to whole segments)")
    ap.add_argument("--min-cover", type=float, default=0.8,
                    help="keep a segment if this fraction of its rows has a book (head padded with no-book rows)")
    ap.add_argument("--p-model", choices=("mid", "micro"), default="mid")
    ap.add_argument("--stale-s", type=float, default=600.0,
                    help="STALE flag when book_age > this (only for markets without a recorder coverage log)")
    ap.add_argument("--splits", help="JSON split config (see assign_splits); default: D11 auto 60/20/20 by game day")
    ap.add_argument("--slugs", help="regex filter on market slugs")
    ap.add_argument("--max-markets", type=int, default=0)
    ap.add_argument("--settle-wait", type=int, default=1800, help="only games whose kickoff is this many s past")
    ap.add_argument("--now", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--offline", action="store_true", help="no network (gamma / data-api); cache only")
    ap.add_argument("--no-gamma", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-fetch-trades", action="store_true", help="recorder trades only")
    ap.add_argument("--keep-truncated", action="store_true",
                    help="keep segments older than the earliest leg of a data-api-truncated market (fills missing)")
    ap.add_argument("--allow-open", action="store_true",
                    help="TESTING ONLY: include games not yet started, cut at their last book row")
    ap.add_argument("--verify", action="store_true", help="verify --out after building")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-env", action="store_true", help="selftest: skip the env probes")
    ap.add_argument("--keep", action="store_true", help="selftest: keep the temp dir")
    ap.add_argument("--tmp", help="selftest: parent of the temp dir")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args)
    if not args.rec:
        ap.error("--rec is required")
    build(args)
    if args.verify:
        verify(Path(args.out))


if __name__ == "__main__":
    main()
