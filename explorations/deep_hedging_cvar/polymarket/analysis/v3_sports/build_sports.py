#!/usr/bin/env python3
"""Pre-game sports books -> the C market-making env's binaries (Track S, DECISIONS.md D11).

Writes the same files as data/build (ticks.bin, feat.bin, book.bin, fills.bin, fill_index.bin,
markets.bin, meta.json, feat_meta.json, splits.json) with the same record layouts and the same
feat.bin column layout, so ../../env.py / ../../mm.h / ../../train/envs.py (data_dir=OUT) load it
unchanged. Plus two side files, sports_feat.bin and sports_x3.bin (the per-row reward program in
v3env's x3.bin layout). See README_build.md for every schema decision.

    source ../../../.venv/bin/activate          # numpy, pandas, pyarrow, orjson (polyrec)
    python build_sports.py --rec rec_out [--rec raw_pob] --out build            # all finished games
    python build_sports.py --rec rec_out --seg 0 --out build_w                  # env3: 1 episode / game window
    python ../../v3env/adapt3.py build_w                                        # -> markets3.bin (env3)
    python build_sports.py --selftest                                           # synthetic, end to end

Cut: episodes END at E = T - cut (`--cut-before-kickoff-s`, default 0 = at kickoff; DECISIONS.md D15
withdrew D14's T-1h cut, the pull time is a policy hyperparameter, not a build cut), T = the
scheduled kickoff (gamma gameStartTime, the v3 data rule), and settle at the mid at E. Every
segment carries its time-to-kickoff bucket in meta.json (>24h, 6-24h, 1-6h, 1h-30m, 30-5m, 5-0m).
Episode: `--seg 300|900|14400` = segments of exactly `seg` rows (+1) on a `dt`-second grid, tiling
[E - max_hours, E] backwards from E (the v2 env samples only these lengths); `--seg 0` = ONE
episode per game, from the first valid book row (or E - max_hours) to E, any length (v3env/mm3
only); `--seg 0 --bucket-segments` splits it at the bucket boundaries (24 h, 6 h, 1 h, 30 m, 5 m). Build at dt 1 (default) and use mm3's dt_far for a slower far-from-kickoff
cadence: latency, match delay, markouts and the T5 flags count rows, so --dt > 1 is discouraged.
Book rows captured at/after kickoff and trades with block time >= kickoff are dropped. Each
episode settles at its last row's mark (fractional): meta.json `segments[i].settle_price`
(settle_mode 1) and, while v3env/adapt3.py and env_sports/ still read it, the legacy copy in
markets.bin `log_strike_vs_ul_open` + flag SETTLE_PX 16 (`--no-legacy-settle` drops it). The
stock mm.h only settles at the int `winner_up` (0/1): use env_sports/ or v3env for any P&L number.
Reward columns (feat liq_mid/liq_comp/liq_pi and sports_x3.bin) come from the FULL recorded ladder
with Polymarket's scorer and the daily CLOB program snapshots of `--rewards-dir`.

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

BUILD_VERSION = "sports-2"     # adapt3.py detects "sports*"
NLEV = 5
SEG_LENS = (300, 900, 14400)                      # mm.h mm_len_bit: other lengths are never sampled
SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "purged"}
SPLIT_IDS = {v: k for k, v in SPLIT_NAMES.items()}
GAMMA = "https://gamma-api.polymarket.com"
CACHE = HERE / "cache"
REWARDS_DIR = HERE / "rewards"                    # rewards_snapshot.py output (git-tracked)
KICKOFF_CSV = HERE / "kickoff" / "kickoffs.csv"   # kickoff_watch.py: observed actual starts (labels only)

# markets.bin flags (1..8 as in data/build.py). 16 and 32 mean ELECTION / UNRESOLVED in the elections
# builder: sports keeps 16 only as the legacy settle marker, and TRADES_TRUNCATED moved from 32 to 1024.
FLAG_HAS_TRADES = 1
FLAG_SETTLE_PX = 16          # LEGACY: settle = log_strike_vs_ul_open (env_sports/mm.h, adapt3.py "sports-1")
FLAG_HEAD_PADDED = 64        # segment starts before the first book row (padded with no-book rows)
FLAG_AT_CUT = 128            # segment ends at the cut T - cut_s (= AT_KICKOFF with the default cut 0)
FLAG_AT_KICKOFF = FLAG_AT_CUT  # old name
FLAG_KICKOFF_FALLBACK = 256  # no gamma gameStartTime: kickoff from eventStartTime / endDate
FLAG_COVERAGE = 512          # recorder coverage log available (stale flag = recorder not live)
FLAG_TRADES_TRUNC = 1024     # data-api history capped: the oldest pre-game legs may be missing (was 32)
FLAG_IN_PROGRAM = 2048       # in a liquidity-reward program at some row of the episode
FLAG_RW_LATER_SNAPSHOT = 4096  # some row's program comes from a snapshot taken after that row's day
FLAG_EARLY_START = 8192      # kickoff_watch saw the game live before the scheduled kickoff (in-play rows)

# D14/D15 time-to-kickoff buckets: (name, lower bound s, upper bound s); a row belongs by its seconds to kickoff
BUCKETS = [(">24h", 86400, None), ("6-24h", 21600, 86400), ("1-6h", 3600, 21600), ("1h-30m", 1800, 3600),
           ("30-5m", 300, 1800), ("5-0m", 0, 300)]
BUCKET_BOUNDS = [b[1] for b in BUCKETS[:-1]]          # 86400, 21600, 3600, 1800, 300 (descending)


def bucket_of(sec) -> np.ndarray:
    sec = np.asarray(sec)
    return np.searchsorted(-np.array(BUCKET_BOUNDS), -sec, side="right")   # 0 = >24h ... 5 = 5-0m


def bucket_info(to_ko_rows: np.ndarray) -> dict:
    """Bucket of an episode = the bucket holding most of its decision rows (all rows but the settlement row);
    pure = all of them in one bucket (always with --bucket-segments)."""
    r = to_ko_rows[:-1] if len(to_ko_rows) > 1 else to_ko_rows
    cnt = np.bincount(bucket_of(r), minlength=len(BUCKETS))
    return {"bucket": BUCKETS[int(cnt.argmax())][0], "bucket_pure": bool((cnt > 0).sum() == 1),
            "bucket_rows": {BUCKETS[k][0]: int(c) for k, c in enumerate(cnt) if c}}

# Liquidity rewards (analysis/rewards/reward_terms.py, the elections builder, ENV3.md section 5)
REWARD_BAND = (0.10, 0.90)   # single-sided scoring allowed inside, c = 3
REWARD_C = 3.0
LIQ_SEED = 20261016          # sample-second draw: one uniform second per UTC minute per market
X3_COLS = ["rw_v_c", "rw_min_size", "rw_rate_day", "rw_in_program", "liq_mid", "liq_comp", "liq_pi", "pad"]
EXTRA = ["amid", "qb", "qa"]

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
BYTES_PER_ROW = F * 4 + G * 4 + 64 + 4 + len(SPORTS_COLS) * 4 + len(X3_COLS) * 4


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
    # explicit dtypes: an empty history would give object columns, and t[empty object Series] selects
    # COLUMNS (-> "'DataFrame' object has no attribute 'outcome_idx'", the --allow-open crash)
    t = pd.DataFrame(rows, columns=["slug", "tx", "timestamp", "outcome_idx", "side", "price", "size", "is_taker"])
    t = t.astype({"slug": object, "tx": object, "timestamp": np.int64, "outcome_idx": np.int8, "side": np.int8,
                  "price": np.float64, "size": np.float64, "is_taker": bool})
    tk = t[t.is_taker.values]
    agg = pd.Series(np.where(tk.outcome_idx == 0, tk.side, -tk.side), index=tk.tx.values)
    agg = agg[~agg.index.duplicated()]
    t["aggressor_up"] = t.tx.map(agg).fillna(0).astype(np.int8)
    t["tx"] = pd.Series([bytes.fromhex(x[2:]) for x in t.tx], index=t.index, dtype=object)
    if not final:                     # game not over: do not cache a partial history
        return t, bool(t1 or t2)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path, index=False)
    write_atomic(meta, json.dumps({"truncated": bool(t1 or t2), "fetched_at": int(time.time())}))
    return t, bool(t1 or t2)


def load_actual_kickoffs(path: Path | None) -> dict:
    """kickoff_watch.py's kickoffs.csv -> {slug: {...}} (labels only: the cut stays the schedule)."""
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:  # noqa: BLE001  (being rewritten by the watcher)
        print(f"  kickoffs.csv unreadable ({e}); no actual-start labels")
        return {}
    out = {}
    for r in df.to_dict("records"):
        if r.get("actual_unix"):
            out[r["slug"]] = {"actual_start_ts": int(float(r["actual_unix"])), "source": "kickoff_watch_" + r["source"],
                              "confidence": r["confidence"], "delta_s": num(r.get("delta_s"), 0.0)}
    return out


# ----------------------------------------------------------------------------- liquidity rewards
def load_rewards_dir(d) -> dict:
    """rewards_snapshot.py files -> {day: {condition_id: program}} (empty without a dir)."""
    if not d or not Path(d).is_dir():
        return {}
    out = {}
    for f in sorted(Path(d).glob("rewards_*.json")):
        doc = json.loads(f.read_text())
        out[doc.get("day") or f.stem.split("_")[-1]] = doc.get("markets", {})
    return out


def program_for(snaps: dict, cond: str | None, day: str, gv: float, gm: float) -> dict:
    """The reward program of one market on one UTC day: the latest snapshot taken on or before that day,
    else the earliest one (flagged `later`; then the program counts from its rewards_config start_date).
    Out of program: rate 0, scorer v / min size from gamma (rewardsMaxSpread / rewardsMinSize)."""
    o = {"in": 0.0, "rate": 0.0, "v": gv, "m": gm, "later": False, "snap": None}
    if not snaps or not isinstance(cond, str) or not cond:
        return o
    days = sorted(snaps)
    i = int(np.searchsorted(days, day, side="right")) - 1
    if i < 0:
        i, o["later"] = 0, True
    o["snap"] = days[i]
    rec = snaps[days[i]].get(cond)
    if rec is None:
        return o
    o["v"] = num(rec.get("max_spread"), gv)
    o["m"] = num(rec.get("min_size"), gm)
    if (rec.get("start_date") or "") <= day and (not rec.get("end_date") or day <= rec["end_date"]):
        o["in"], o["rate"] = 1.0, num(rec.get("rate_per_day"), 0.0)
    return o


def utc_day(ts) -> np.ndarray:
    return pd.to_datetime(np.asarray(ts, np.int64), unit="s", utc=True).strftime("%Y-%m-%d").values


def S(d, v):
    """Polymarket's order score ((v - d) / v)^2 for d < v, else 0 (v = 0 -> 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((d < v) & (v > 0), ((v - d) / np.where(v > 0, v, 1.0)) ** 2, 0.0)


def side_ladder(tbl, pcol, scol, m):
    """Flattened FULL ladder of one side and the price where cumulative depth reaches m (row-wise).
    Same as analysis/v3_elections/build_elections.py side_ladder (ladders best-first, as recorded)."""
    pc_ = tbl.column(pcol).combine_chunks()
    sc_ = tbl.column(scol).combine_chunks()
    off = pc_.offsets.to_numpy().astype(np.int64)
    assert (sc_.offsets.to_numpy() == off).all(), "price/size list offsets differ"
    lo, hi = int(off[0]), int(off[-1])
    p = pc_.values.to_numpy(zero_copy_only=False)[lo:hi].astype(np.float64)
    s = sc_.values.to_numpy(zero_copy_only=False)[lo:hi].astype(np.float64)
    n = len(off) - 1
    ln = np.diff(off)
    start = off[:-1] - lo
    row = np.repeat(np.arange(n), ln)
    cs = np.cumsum(s)
    cs0 = np.concatenate([[0.0], cs])
    cum = cs - cs0[start][row]
    below = cum < m[row] - 1e-9
    k = np.minimum(np.bincount(row[below], minlength=n), np.maximum(ln - 1, 0))
    cut = np.where(ln > 0, p[np.minimum(start + k, max(len(p) - 1, 0))] if len(p) else np.nan, np.nan)
    return row, p, s, cut


def reward_stats(tbl, m, v):
    """Per raw snapshot, in the token's own frame: the size-cutoff adjusted mid (NaN if a side is empty)
    and competitors' Q of the bid and the ask side over the FULL ladder (reward_terms.adjusted_mid /
    side_scores; the elections builder's reward_stats)."""
    rb, pb, sb, cb = side_ladder(tbl, "bid_prices", "bid_sizes", m)
    ra, pa_, sa, ca = side_ladder(tbl, "ask_prices", "ask_sizes", m)
    mid = 0.5 * (cb + ca)
    n = len(m)
    okm = np.isfinite(mid)
    mid0 = np.where(okm, mid, 0.0)
    qb = np.bincount(rb, weights=S(np.abs(mid0[rb] - pb) * 100, v[rb]) * sb * (sb >= m[rb] - 1e-9), minlength=n)
    qa = np.bincount(ra, weights=S(np.abs(pa_ - mid0[ra]) * 100, v[ra]) * sa * (sa >= m[ra] - 1e-9), minlength=n)
    return mid, np.where(okm, qb, 0.0), np.where(okm, qa, 0.0)


def sample_rows(t: np.ndarray, seed) -> np.ndarray:
    """One reward sample per UTC minute: a seeded uniform second, i.e. the first grid row of the minute at
    or after it (exactly that second at dt 1)."""
    n = len(t)
    minute = t // 60
    first = np.r_[True, minute[1:] != minute[:-1]] if n else np.zeros(0, bool)
    starts = np.flatnonzero(first)
    offs = np.random.default_rng(seed).integers(0, 60, len(starts))
    off_row = np.repeat(offs, np.diff(np.r_[starts, n]))
    cand = np.flatnonzero(t % 60 >= off_row)
    out = np.zeros(n, bool)
    if len(cand):
        _, i = np.unique(minute[cand], return_index=True)
        out[cand[i]] = True
    return out


def market_seed(cond: str | None, slug: str) -> list:
    try:
        return [LIQ_SEED, int(str(cond)[2:14], 16)]
    except (TypeError, ValueError):
        return [LIQ_SEED, int(hashlib.sha1(slug.encode()).hexdigest()[:12], 16)]


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
    ex = (np.where(ok[:, None], v[EXTRA].values[jj], np.nan) if all(c in v for c in EXTRA)
          else np.full((n, 3), np.nan))                       # full-ladder reward stats of the same snapshot
    if cov_iv is not None:                       # recorder coverage: stale = recorder not live at t
        live = np.zeros(n, bool)
        for a, b in cov_iv:
            live |= (t >= a) & (t <= b)
        stale = ~live
    else:
        stale = age > stale_s
    return t, X, stale, (int(vts[0]) if len(vts) else None), ex


def features(t, X, legs_row_sub, stale, p_mode, seg, dt, kickoff, first_book, ex=None, prog=None, seed=None,
             cov_known=False):
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
    if seg:                                      # --seg 0: no v2 length bucket (mm3 length bit 8)
        out[:, CI[{300: "len_5m", 900: "len_15m", 14400: "len_4h"}[seg]]] = 1.0
    out[:, CI["tick_fine"]] = fine
    out[:, CI["book_age"]] = np.log1p(age) / 4
    out[:, CI["crossed_raw"]] = crossed
    out[:, CI["mark_is_fv"]] = ~two
    out[:, CI["p_model"]] = p
    out[:, CI["mark"]] = mark
    # ---- liquidity rewards: full-ladder stats of the snapshot in use, program of the row's UTC day
    x3 = np.zeros((n, len(X3_COLS)))
    if ex is None:
        ex = np.full((n, 3), np.nan)
    if prog is None:
        prog = {k: np.zeros(n) for k in ("v", "m", "rate", "in")}
    amid, qb, qa = ex[:, 0], ex[:, 1], ex[:, 2]
    scorable = valid & np.isfinite(amid) & ~(stale if cov_known else np.zeros(n, bool))
    amid = np.where(scorable, amid, -1.0)
    qb, qa = np.where(scorable, qb, 0.0), np.where(scorable, qa, 0.0)
    inside = (amid >= REWARD_BAND[0]) & (amid <= REWARD_BAND[1])
    comp = np.minimum(qb, qa) + np.where(inside, np.abs(qb - qa) / REWARD_C, 0.0)
    sample = sample_rows(t, seed) if seed is not None else np.zeros(n, bool)
    pi = np.where(sample & scorable & (prog["in"] > 0), prog["rate"] / 1440.0, 0.0)
    out[:, CI["liq_mid"]] = amid
    out[:, CI["liq_comp"]] = comp
    out[:, CI["liq_pi"]] = pi
    x3[:, 0], x3[:, 1], x3[:, 2], x3[:, 3] = prog["v"], prog["m"], prog["rate"], prog["in"]
    x3[:, 4], x3[:, 5], x3[:, 6] = amid, comp, pi
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
    return out, sf, book, x3


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
    if seg not in SEG_LENS + (0,):
        raise SystemExit(f"--seg must be 0 (one episode per game window, env3 only) or one of {SEG_LENS} "
                         "(the v2 mm.h samples only these lengths)")
    if dt > 1:
        print(f"WARNING: --dt {dt}: latency, match delay, markouts, part_cap and the T5 flags count ROWS in the "
              "env, so they become dt-second quantities. For v3env build at dt 1 and use dt_far for cadence.")
    max_hours = args.max_hours if args.max_hours is not None else (0.0 if seg == 0 else 24.0)
    if seg:
        seg_s = seg * dt
        nseg = int(max_hours * 3600 // seg_s)
        if nseg < 1:
            raise SystemExit(f"--max-hours {max_hours} is shorter than one segment ({seg} rows x {dt} s = "
                             f"{seg_s / 3600:.2f} h)")
        n_tot = nseg * seg + 1
    else:
        seg_s, nseg, n_tot = 0, 1, None                # per game: first valid book row (or the cap) .. kickoff
        cap_rows = int(max_hours * 3600 // dt) + 1 if max_hours > 0 else None
    legacy_settle = not args.no_legacy_settle
    cut_s = int(args.cut_before_kickoff_s)           # D14: training episodes end at T - cut_s
    if cut_s < 0:
        raise SystemExit("--cut-before-kickoff-s must be >= 0")
    if args.bucket_segments and seg:
        raise SystemExit("--bucket-segments needs --seg 0 (fixed-length segments cannot follow bucket boundaries)")
    slug_rx = re.compile(args.slugs) if args.slugs else None
    now = int(args.now or time.time())
    actual = load_actual_kickoffs(args.kickoff_csv)
    snaps = load_rewards_dir(args.rewards_dir)
    print(f"reward snapshots: {len(snaps)} day(s) from {args.rewards_dir or '-'}; actual-start labels: {len(actual)}")

    books = load_books(recs, slug_rx)
    if books is None:
        raise SystemExit("no book rows found under " + ", ".join(map(str, recs)))
    print(f"books: {books.num_rows} rows, {len(set(books.column('market_slug').to_pylist()))} markets "
          f"({time.time() - t0:.1f}s)")
    lv = np.r_[np.diff(books.column("bid_prices").combine_chunks().offsets.to_numpy()),
               np.diff(books.column("ask_prices").combine_chunks().offsets.to_numpy())]
    ladder = {"levels_max": int(lv.max()) if len(lv) else 0, "levels_p50": float(np.median(lv)) if len(lv) else 0,
              "levels_p99": float(np.percentile(lv, 99)) if len(lv) else 0}
    ladder["source"] = ("full recorded ladder (recorder / backfill books rows carry every level)"
                        if ladder["levels_max"] > 10 else
                        "AT MOST 10 LEVELS per side: a compact export (rec_daily keeps 10) or thin books; reward "
                        "stats then miss deeper levels")
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
        act = actual.get(slug)                    # labels only (v3 data rule: the cut is the schedule)
        early = bool(act and act["actual_start_ts"] < ko_true and act["confidence"] in ("high", "medium"))
        if early and args.drop_early_starts:
            skipped.append((slug, f"early start: live at {act['actual_start_ts']} < scheduled {ko_true}"))
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
            league=slug.split("-")[0], day=game_day(slug, ko),
            kickoff_source="gameStartTime" if not fallback else "fallback (eventStartTime / endDate / books end)",
            actual=act, early_start=early))
    mk = pd.DataFrame(rows)
    if args.max_markets and len(mk):
        mk = mk.iloc[: args.max_markets]
    print(f"markets eligible {len(mk)}, skipped {len(skipped)} ({time.time() - t0:.1f}s)")
    if not len(mk):
        raise SystemExit("no eligible market; skipped: " + "; ".join(f"{s}: {r}" for s, r in skipped[:10]))

    # ---- full-ladder reward stats per raw book row (own token frame), with the scorer's v / min size of the
    # market's program on the row's UTC day; mapped to the Up frame per market in pass 1
    for c in EXTRA:
        bk[c] = np.nan
    info = mk.set_index("slug")
    sel = np.flatnonzero(bk.slug.isin(info.index).values)
    if len(sel):
        sub = bk.iloc[sel]
        days = utc_day(sub.ts.values)
        m_arr, v_arr = np.zeros(len(sel)), np.zeros(len(sel))
        for (s_, d_), ix in pd.DataFrame({"s": sub.slug.values, "d": days}).groupby(["s", "d"]).indices.items():
            r_ = info.loc[s_]
            p_ = program_for(snaps, r_.cond, d_, r_.rew_spread, r_.rew_min)
            m_arr[ix], v_arr[ix] = p_["m"], p_["v"]
        amid, qb, qa = reward_stats(books.take(pa.array(sel)), m_arr, v_arr)
        bk.iloc[sel, [bk.columns.get_loc(c) for c in EXTRA]] = np.c_[amid, qb, qa]
    del books
    bk = bk.drop_duplicates(["token", "ts"], keep="last")
    print(f"reward stats on {len(sel)} book rows ({time.time() - t0:.1f}s)")
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
                       "segments_dropped_trades_truncated": 0, "segments_dropped_short": 0,
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
        # full-ladder reward stats of the snapshot merged_snapshots picked, in the Up frame
        # (an outcome-1 bid at q is an Up ask at 1 - q: amid -> 1 - amid, bid / ask Q swap)
        up = g[g.is_up.values].set_index("ts")[EXTRA].reindex(snap.index)
        dn = g[~g.is_up.values].set_index("ts")[EXTRA].reindex(snap.index)
        dn_up = np.c_[1.0 - dn.amid.values, dn.qa.values, dn.qb.values]
        snap[EXTRA] = np.where(snap.from_down.values[:, None], dn_up, up.values)
        cov_iv = cov_all.get(slug)
        # cut: episodes end at E = kickoff - cut_s (default 0; with --allow-open: the last book row if earlier),
        # on a grid anchored at E
        cut = min(ko, int(m.kickoff_true) - cut_s)
        grid_end = cut
        if seg:
            n_m = n_tot
        else:                                          # first valid book row .. cut, on the cut grid
            fv = int(snap.index.values[snap.valid.values][0])
            n_tr = (cut - fv) // dt + 1                # training rows (<= 0: book starts after the cut)
            if cap_rows:
                n_tr = min(n_tr, cap_rows)
            n_m = n_tr
            if n_m < 2:
                stats["segments_dropped_short"] += 1
                continue
        e_idx = n_m - 1                                # grid row of the cut
        stats["book_rows_after_cut_dropped"] = stats.get("book_rows_after_cut_dropped", 0) + int(
            (g.ts > grid_end).sum())
        t, X, stale, first_book, ex = market_grid(snap, grid_end, n_m, dt, cov_iv, args.stale_s)
        # reward program per grid row (UTC day of the row)
        cond = m.cond if isinstance(m.cond, str) else None
        rdays = utc_day(t)
        prog = {k: np.zeros(n_m) for k in ("v", "m", "rate", "in", "later")}
        snap_days = set()
        for d_ in np.unique(rdays):
            p_ = program_for(snaps, cond, d_, m.rew_spread, m.rew_min)
            k_ = rdays == d_
            for key in ("v", "m", "rate", "in", "later"):
                prog[key][k_] = float(p_[key])
            if p_["snap"]:
                snap_days.add(p_["snap"])
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
        sub_row = np.zeros(n_m, bool)
        if tr is not None and len(tr):
            tr = tr.copy()
            tr["is_taker"] = tr.is_taker.astype(bool)
            tr["px_up"] = np.where(tr.outcome_idx == 0, tr.price.astype(np.float64), 1.0 - tr.price.astype(np.float64))
            earliest = int(tr.timestamp.min()) if trunc else None    # legs before it may be missing
            stats["legs_after_kickoff_dropped"] += int((tr.timestamp >= ko).sum())
            stats["legs_after_cut_dropped"] = stats.get("legs_after_cut_dropped", 0) + int(
                ((tr.timestamp < ko) & (tr.timestamp > grid_end)).sum())
            tr = tr[(tr.timestamp < ko) & (tr.timestamp <= grid_end)]
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
                    vol = np.bincount(r[msk], weights=sz[msk], minlength=n_m)
                    pv = np.bincount(r[msk], weights=(px * sz)[msk], minlength=n_m)
                    ext = np.full(n_m, -np.inf if sgn == 1 else np.inf)
                    fn.at(ext, r[msk], px[msk])
                    has = vol > 0
                    X[:, TC[vcol]] = vol
                    X[:, TC[wcol]] = np.where(has, pv / np.where(has, vol, 1), 0)
                    X[:, TC[xcol]] = np.where(np.isfinite(ext) & has, ext, 0)
                txr = pd.DataFrame({"r": r, "tx": maker.tx.values}).drop_duplicates("tx")
                X[:, TC["trade_count"]] = np.bincount(txr.r.values, minlength=n_m)
                sub = (maker.px_mil.values % 10) != 0
                sub_row[r[sub]] = True
        feat, sf, book, x3 = features(t, X, sub_row, stale, args.p_model, seg, dt, ko, first_book, ex=ex, prog=prog,
                                      seed=market_seed(cond, slug), cov_known=cov_iv is not None)
        stats["markets"] += 1
        # ---- segments: (first row, last row, segments before the cut, eval_only)
        to_ko = (ko - t).astype(np.int64)
        if seg:
            spans = [(k * seg, (k + 1) * seg, nseg - 1 - k, False) for k in range(nseg)]
        else:
            a0 = 0
            if earliest is not None and not args.keep_truncated and t[0] < earliest:
                a0 = int(np.searchsorted(t, earliest))         # start where the (truncated) trade history starts
                stats["segments_trimmed_trades_truncated"] = stats.get("segments_trimmed_trades_truncated", 0) + 1
            cuts = [a0]
            if args.bucket_segments:                           # split at the bucket boundaries
                for bnd in BUCKET_BOUNDS:
                    p_ = int(np.searchsorted(-to_ko, -bnd))    # first row with to_ko <= bnd
                    if cuts[-1] < p_ < e_idx:
                        cuts.append(p_)
            cuts.append(e_idx)
            spans = [(cuts[j], cuts[j + 1], len(cuts) - 2 - j, False) for j in range(len(cuts) - 1)
                     if cuts[j + 1] > cuts[j]]
            if (e_idx - a0) * dt < args.min_window_s:          # the whole game window (before any bucket split)
                stats["segments_dropped_short"] += 1
                spans = []
        for a, b, kk, _ in spans:
            Ls = b - a                                         # episode rows - 1 (= contract_len_s)
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
            Xs[:, TC["seconds_to_close"]] = np.arange(Ls, -1, -1)        # rows to the episode end
            fs = feat[sl].copy()
            fs[:, CI["tau_frac"]] = np.arange(Ls, -1, -1) / Ls
            for L_, c_ in ((300, "len_5m"), (900, "len_15m"), (14400, "len_4h")):   # one-hot of THIS segment
                fs[:, CI[c_]] = float(Ls == L_)
            sfs = sf[sl].copy()
            sfs[:, 1] = np.arange(Ls, -1, -1) * dt
            xs = x3[sl]
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
            sid = f"{slug}#k{kk}"                     # k0 = the segment ending at the cut (kickoff by default)
            bk_ = bucket_info(to_ko[a:b + 1])
            f = spool / f"{len(segs)}.npz"
            np.savez(f, X=Xs.astype(np.float32), feat=fs.astype(np.float32), sf=sfs.astype(np.float32), book=bks,
                     x3=xs.astype(np.float32), legs=lg if lg is not None else np.zeros(0, FILL_DTYPE))
            in_prog = bool((xs[:, 3] > 0).any())
            later = bool((prog["later"][sl] > 0).any())
            flags = ((FLAG_SETTLE_PX if legacy_settle else 0) | (FLAG_HAS_TRADES if lg is not None else 0)
                     | (FLAG_TRADES_TRUNC if trunc else 0) | (FLAG_HEAD_PADDED if cover < 1 else 0)
                     | (FLAG_AT_CUT if kk == 0 else 0) | (FLAG_KICKOFF_FALLBACK if m.kickoff_fallback else 0)
                     | (FLAG_COVERAGE if cov_iv is not None else 0) | (FLAG_IN_PROGRAM if in_prog else 0)
                     | (FLAG_RW_LATER_SNAPSHOT if later else 0) | (FLAG_EARLY_START if m.early_start else 0))
            seg_meta = {
                "sid": sid, "slug": slug, "condition_id": cond, "n_rows": Ls + 1,
                "open_ts": int(t[a]), "end_ts": int(t[b]), "row_s": dt,
                "kickoff_ts": int(ko), "cut_ts": int(t[e_idx]), "event_ts": int(t[b]),
                "kickoff_scheduled_ts": int(m.kickoff_true), "kickoff_source": m.kickoff_source,
                "kickoff_cut_open_TESTING": bool(args.allow_open and ko != m.kickoff_true),
                "at_cut": bool(kk == 0), "segments_before_cut": int(kk),
                "to_kickoff_start_s": int(to_ko[a]), "to_kickoff_end_s": int(to_ko[b]), **bk_,
                "settle_mode": 1, "settle_price": settle,
                "actual_start": m.actual if isinstance(m.actual, dict) else None, "early_start": bool(m.early_start),
                "rw": {"in_program_any": in_prog, "in_program_last": bool(xs[-1, 3] > 0),
                       "rate_day_last": float(xs[-1, 2]), "v_c_last": float(xs[-1, 0]),
                       "min_size_last": float(xs[-1, 1]), "usd_paid_pool": float(xs[:, 6].sum()),
                       "sample_rows_scorable": int((xs[:, 6] > 0).sum()),
                       "snapshot_days": sorted(snap_days), "later_snapshot": later}}
            segs.append(dict(sid=sid, slug=slug, file=f, open_ts=int(t[a]), end_ts=int(t[b]),
                             split=int(m.split),
                             league=league_id[m.league], settle=settle, flags=flags, m=m, cover=cover, L=Ls,
                             rw_min=float(xs[-1, 1]), rw_spread=float(xs[-1, 0]), meta=seg_meta,
                             n_legs=0 if lg is None else len(lg), two_frac=float(two.mean()), kickoff=ko))
            stats["segments_kept"] += 1
    if not segs:
        shutil.rmtree(spool, ignore_errors=True)
        raise SystemExit(f"no segment kept: {json.dumps(stats)}")

    # ---- pass 2: final order (split, end_ts, sid), contiguous splits; stream the files
    segs.sort(key=lambda s: (s["split"], s["end_ts"], s["sid"]))
    seg_rows = np.array([s["L"] + 1 for s in segs], np.int64)
    offsets = np.r_[0, np.cumsum(seg_rows)[:-1]]
    N = int(seg_rows.sum())
    recs_arr = np.zeros(len(segs), MARKET_DTYPE)
    side_files = ("ticks.bin", "feat.bin", "book.bin", "fills.bin", "sports_feat.bin", "sports_x3.bin")
    fh = {name: open(out / f".{name}.tmp{os.getpid()}", "wb") for name in side_files}
    fill_counts = np.zeros(N, np.int64)
    n_fills = 0
    seg_meta = []
    for i, s in enumerate(segs):
        z = np.load(s["file"])
        r0, n_rows = int(offsets[i]), int(seg_rows[i])
        z["X"].tofile(fh["ticks.bin"])
        z["feat"].tofile(fh["feat.bin"])
        z["book"].tofile(fh["book.bin"])
        z["sf"].tofile(fh["sports_feat.bin"])
        z["x3"].tofile(fh["sports_x3.bin"])
        legs = z["legs"].copy()
        if len(legs):
            fill_counts[r0:r0 + n_rows] += np.bincount(legs["row"], minlength=n_rows)
            legs["row"] += r0
            legs.tofile(fh["fills.bin"])
            n_fills += len(legs)
        m = s["m"]
        # winner_up: 0/1 fallback of the stock mm.h (NOT the result). Without the legacy settle copy it is -1,
        # which mm3 settles at the last-row mark (= settle_price), and log_strike_vs_ul_open is 0.
        wu = int(s["settle"] >= 0.5) if legacy_settle else -1
        recs_arr[i] = (r0, s["open_ts"], s["end_ts"], n_rows, s["L"], s["league"], wu,
                       s["split"], s["flags"], 0.01, m.fee_rate, m.fee_exp, m.min_size, s["rw_min"], s["rw_spread"],
                       s["settle"] if legacy_settle else 0.0, m.rebate, i, m.tick)
        seg_meta.append({"market_idx": i, "row_offset": r0, **s["meta"], "flags": int(s["flags"])})
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
    seg_name = {300: "5m", 900: "15m", 14400: "4h", 0: "window"}[seg]
    counts ={SPLIT_NAMES[k]: {"markets": int((recs_arr["split"] == k).sum()),
                               "rows": int(recs_arr["n_rows"][recs_arr["split"] == k].sum()),
                               "games": len({s["slug"] for s in segs if s["split"] == k}),
                               "days": sorted({d for d, v in day_split.items() if v == k}),
                               **{cl: int(((recs_arr["split"] == k) & (recs_arr["contract_len_s"] == L)).sum())
                                  for cl, L in B.CLEN.items()}} for k in SPLIT_NAMES}
    sizes = {f: (out / f).stat().st_size for f in side_files + ("fill_index.bin", "markets.bin")}
    meta = {
        "build_version": BUILD_VERSION, "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "F": F, "N": int(N), "n_markets": len(segs), "dtype": "float32", "order": "row-major [N, F]",
        "columns": TCOLS, "column_index": TC,
        "market_record": {"itemsize": MARKET_DTYPE.itemsize, "fields": layout,
                          "flags": {"HAS_TRADES": FLAG_HAS_TRADES, "SETTLE_PX": FLAG_SETTLE_PX,
                                    "HEAD_PADDED": FLAG_HEAD_PADDED, "AT_CUT": FLAG_AT_CUT,
                                    "KICKOFF_FALLBACK": FLAG_KICKOFF_FALLBACK, "COVERAGE": FLAG_COVERAGE,
                                    "TRADES_TRUNCATED": FLAG_TRADES_TRUNC, "IN_PROGRAM": FLAG_IN_PROGRAM,
                                    "RW_LATER_SNAPSHOT": FLAG_RW_LATER_SNAPSHOT, "EARLY_START": FLAG_EARLY_START},
                          "flags_note": "bit 32 is no longer used (TRADES_TRUNCATED moved to 1024, sports-2); bit 16 "
                                        "SETTLE_PX only while legacy settle is on (see settle_source)"},
        "coin_ids": league_id,
        "contract_len_s": {seg_name: seg} if seg else {"window": "per market (n_rows - 1); --seg 0"},
        "settle_source": {
            "primary": "segments[i].settle_price with settle_mode 1 (meta.json, market order) = feat mark at the "
                       "episode's last row (the last pre-kickoff mid, float32)",
            "legacy_markets_bin": bool(legacy_settle),
            "legacy": ("markets.bin log_strike_vs_ul_open (= settle_price) + flags bit 16 SETTLE_PX on every market; "
                       "winner_up = int(settle >= 0.5) is a stock-mm.h fallback, NOT the result. Read by "
                       "v3env/adapt3.py (sports branch) and env_sports/mm.h.") if legacy_settle else
                      ("dropped (--no-legacy-settle): log_strike_vs_ul_open = 0, no bit 16, winner_up = -1 (mm3: "
                       "mode-0 fallback to the last-row mark = settle_price). env_sports/ and the current "
                       "adapt3.py cannot read this build."),
            "kickoff_ts": "segments[i].kickoff_ts: the scheduled kickoff T (gameStartTime; --allow-open: the last "
                          "book row if earlier); equals open_ts + sports_feat.seconds_to_kickoff[row_offset]",
            "event_ts": "segments[i].event_ts = end_ts: where the episode settles (AT_CUT episodes: cut_ts = "
                        "T - cut_before_kickoff_s = T by default)"},
        "segments": seg_meta,
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
            "max_hours_before_cut": max_hours if max_hours > 0 else None, "window_rows_per_game": n_tot,
            "episode": ("one episode per game: first valid book row (or kickoff - max_hours) .. kickoff, any length "
                        "(n_rows varies; v3env/mm3 only: the v2 mm.h samples only 300/900/14400 rows and "
                        "train/envs.py MarketSetEnv asserts the len_* one-hot, which is all 0 here)") if not seg else
                       f"segments of {seg} rows tiling [kickoff - max_hours, kickoff] backwards",
            "min_window_s": args.min_window_s if not seg else None,
            "cut_before_kickoff_s": cut_s,
            "cut_note": "episodes end at the cut T - cut_before_kickoff_s and settle at the mid there; D15: default 0 "
                        "(at kickoff); the quote pull before kickoff is a policy / env (pull_s) hyperparameter",
            "buckets": {"def": {b[0]: [b[1], b[2]] for b in BUCKETS},
                        "rule": "segments[i].bucket = the bucket of most decision rows (seconds to kickoff); "
                                "bucket_pure = all in one; --bucket-segments (seg 0) splits windows at every boundary",
                        "bucket_segments": bool(args.bucket_segments),
                        "counts": {b[0]: sum(1 for x in seg_meta if x["bucket"] == b[0]) for b in BUCKETS}},
            "env3": {"recommended": "dt 1 build; mm3 len_mask 15 (bit 8 dt_other), dt_far / far_s for the far-from-"
                                    "kickoff cadence (e.g. dt_other=2, dt_far=30, far_s=7200), pull_s, flat_c",
                     "dt_gt_1": "discouraged: latency / match delay / markouts / T5 flags count rows (ENV3.md issue 3)"},
            "min_cover": args.min_cover, "p_model": args.p_model, "stale_s": args.stale_s,
            "allow_open_TESTING_ONLY": bool(args.allow_open),
            "row_time": "row j of a segment is unix time open_ts + j * dt_s; the last row (end_ts) is the cut "
                        "T - cut_before_kickoff_s for AT_CUT segments. Every env quantity counted in rows (latency, match delay, dt "
                        "params, markout_h, flow/mret windows, minute boundaries) is in units of dt_s seconds.",
            "kickoff": "gamma gameStartTime; fallback eventStartTime, endDate, recorder markets, books market_end_at",
            "cut": "book rows with captured_at >= kickoff and trade legs with block time >= kickoff are dropped",
            "settlement": "see settle_source: segments[i].settle_price = feat mark at the segment's last row "
                          "(float32); legacy copy in log_strike_vs_ul_open + flag SETTLE_PX. winner_up is a FALLBACK "
                          "for the stock mm.h only, NOT the game result; the stock env's terminal P&L is wrong by "
                          "inv * (winner_up - settle). Use v3env (markets3.bin via adapt3.py) or env_sports/.",
            "kickoff_actual": f"labels from {args.kickoff_csv} (kickoff_watch.py; high/medium confidence); "
                              "EARLY_START = live before the schedule. Never moves the cut"
                              + (" (--drop-early-starts: such games skipped)" if args.drop_early_starts else ""),
            "ladder": ladder,
            "rewards": {
                "dir": str(args.rewards_dir) if args.rewards_dir else None, "snapshot_days": sorted(snaps),
                "x3_file": {"file": "sports_x3.bin", "dtype": "float32", "shape": [int(N), len(X3_COLS)],
                            "columns": X3_COLS, "order": "row-major, same rows as feat.bin",
                            "layout": "identical to v3env x3.bin (ENV3.md section 2): adapt3 can copy it as x3.bin"},
                "program": "per UTC day of the row: the latest snapshot taken on or before that day, else the earliest "
                           "(flag RW_LATER_SNAPSHOT; then in program only from its rewards_config start_date). In "
                           "program: rw_rate_day = total_daily_rate, v = rewards_max_spread (c), min size = "
                           "rewards_min_size. Out of program: rate 0, rw_in_program 0, v / min size from gamma "
                           "rewardsMaxSpread / rewardsMinSize",
                "scorer": "Polymarket liquidity rewards (analysis/rewards/reward_terms.py, the elections builder): on "
                          "the FULL recorded ladder of the book snapshot in use (each token's own ladder, mapped to "
                          "the Up frame), adjusted mid = midpoint of the prices where cumulative depth reaches the min "
                          "size; competitors' Q per side = sum ((v - d)/v)^2 * size over orders >= min size within v; "
                          "liq_comp = 'hi' Q_min (min + |qb - qa|/3 inside [0.10, 0.90], else min)",
                "liq_pi": "rate / 1440 on one seeded uniform sample second per UTC minute (first grid row at or after "
                          f"it; seed [{LIQ_SEED}, condition_id[2:14]]), when in program and scorable",
                "scorable": "valid book, both sides non-empty, and (with a recorder coverage log) recorder live",
                "markets_bin": "rewards_min_size / rewards_max_spread = the program (else gamma) value at the "
                               "episode's last row; per-row values in sports_x3.bin",
                "caveats": "competitors use the 'hi' bound; our orders move neither the mid nor Q; in-game "
                           "multiplier b = 1 (pre-game only)"},
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
        "tau_frac": "rows to the segment (episode) end / (episode rows - 1)", "log_tau": "log1p(seconds to KICKOFF) / log1p(14400)",
        "in_twap": "0", "delta_usd": "0", "ulz5": "0", "ulz30": "0", "ulz120": "0",
        "len_5m": "segment-length one-hot (seg rows)", "len_15m": "segment-length one-hot",
        "len_4h": "segment-length one-hot (seg = 14400 rows)",
        "p_model": f"fair value = book {args.p_model} of the last two-sided valid book (forward-filled; 0.5 before)",
        "mark": "two-sided mid, else p_model", "delta_usd_raw": "0", "sigma": "0", "sqrt_v": "0",
        "liq_mid": "size-cutoff adjusted mid of the FULL recorded ladder (cutoff = reward min size of the row's "
                   "program, else gamma's); -1 unscorable", "liq_comp": "competitors' Q_min ('hi' bound) with the "
                   "row's v / min size", "liq_pi": "rate / 1440 on the minute's sample row when in program and "
                   "scorable, else 0 (meta.sports.rewards)"})
    for c in FCOLS:
        docs[c] = docs[c] or "as data/build (data/features.py), windows in rows"
    fmeta = {
        "features_version": BUILD_VERSION, "built_at_utc": meta["built_at_utc"], "N": int(N), "G": G,
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
        "sports_x3": {"file": "sports_x3.bin", "dtype": "float32", "shape": [int(N), len(X3_COLS)],
                      "columns": X3_COLS, "note": "v3env x3.bin layout; see meta.sports.rewards"},
        "stats": stats,
    }
    write_atomic(out / "feat_meta.json", json.dumps(fmeta, indent=1, default=str))
    total = sum(sizes.values())
    print(f"wrote {out}: {len(segs)} segments ({len({s['slug'] for s in segs})} games), rows/segment "
          f"{int(seg_rows.min())}..{int(seg_rows.max())}, N = {N} rows, {n_fills} legs, {total / 1e6:.1f} MB "
          f"({BYTES_PER_ROW} B/row); in reward program: {sum(1 for s in segs if s['flags'] & FLAG_IN_PROGRAM)} "
          f"segments; {time.time() - t0:.1f}s")
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
    assert (mk["n_rows"] == mk["contract_len_s"] + 1).all()
    if sp["seg_rows"]:
        assert set(mk["contract_len_s"]) <= set(SEG_LENS)
    else:
        games = [s.split("#")[0] for s in meta["slugs"]]
        at = [g_ for g_, f_ in zip(games, mk["flags"]) if f_ & FLAG_AT_CUT]
        assert len(at) == len(set(at)), "more than one AT_CUT episode per game"
        if not sp.get("buckets", {}).get("bucket_segments"):     # (a bucket piece can be dropped, e.g. no book)
            assert len(at) == len(set(games)), "one AT_CUT episode per game"
        if not sp.get("buckets", {}).get("bucket_segments"):
            assert len(set(games)) == len(games)
    assert (mk["end_ts"] - mk["open_ts"] == mk["contract_len_s"] * sp["dt_s"]).all()
    assert np.isfinite(feat).all() and np.abs(feat[:, :N_OBS]).max() <= CLIP
    last = mk["row_offset"] + mk["n_rows"] - 1
    sm = meta["segments"]
    assert [x["sid"] for x in sm] == meta["slugs"] and [x["row_offset"] for x in sm] == mk["row_offset"].tolist()
    settle = np.array([x["settle_price"] for x in sm], np.float32)
    assert np.array_equal(settle, feat[last, CI["mark"]]), "settle_price != last-row mark"
    if meta["settle_source"]["legacy_markets_bin"]:
        assert np.array_equal(mk["log_strike_vs_ul_open"], settle) and (mk["flags"] & FLAG_SETTLE_PX > 0).all()
    else:
        assert not (mk["flags"] & FLAG_SETTLE_PX).any() and (mk["winner_up"] == -1).all()
    kts = np.array([x["kickoff_ts"] for x in sm], np.int64)
    assert np.array_equal(kts, mk["open_ts"] + np.rint(sf[mk["row_offset"], 0]).astype(np.int64)), \
        "meta kickoff_ts != open_ts + sports_feat.seconds_to_kickoff[row 0] (adapt3's event_ts)"
    cut = np.array([x["cut_ts"] for x in sm], np.int64)
    ko = mk["flags"] & FLAG_AT_CUT > 0
    assert (mk["end_ts"][ko] == cut[ko]).all() and (np.rint(sf[last[ko], 0]) == (kts - cut)[ko]).all(), \
        "AT_CUT segment does not end at the cut"
    assert (mk["end_ts"] <= cut).all()
    assert all(x["bucket_pure"] for x in sm) if sp.get("buckets", {}).get("bucket_segments") else True
    assert all(x["event_ts"] == x["end_ts"] for x in sm)
    assert (sf[:, 0] >= 0).all(), "row after kickoff"
    x3 = np.fromfile(out / "sports_x3.bin", np.float32).reshape(meta["N"], len(X3_COLS))
    for j, c in ((4, "liq_mid"), (5, "liq_comp"), (6, "liq_pi")):
        assert np.array_equal(x3[:, j], feat[:, CI[c]]), f"sports_x3 {c} != feat {c}"
    assert ((x3[:, 6] == 0) | ((x3[:, 3] > 0) & (x3[:, 4] >= 0))).all(), "paid sample outside program / unscorable"
    assert np.array_equal(mk["rewards_min_size"], x3[last, 1]) and np.array_equal(mk["rewards_max_spread"], x3[last, 0])
    assert ((mk["flags"] & FLAG_IN_PROGRAM > 0) == np.array([x["rw"]["in_program_any"] for x in sm])).all()
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
        mrow.update(slug=slug, condition_id="0x" + f"{gi + 1:064x}", token_up=toks[0], token_down=toks[1],
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


def synth_rewards(d: Path) -> dict:
    """Reward snapshots for the synthetic games (game gi: condition 0x..gi+1, kickoff day 2026-10-0{gi+1}):
    g0 in program from a LATER snapshot (start_date before its day, v 9 c so levels 6-8 count), g1 listed
    but its program starts after its day, g3 in program, g2 / g4 not listed."""
    c = {gi: "0x" + f"{gi + 1:064x}" for gi in range(5)}
    snaps = {"2026-10-02": {c[0]: dict(slug="g0", rate_per_day=14.4, max_spread=9.0, min_size=100, start_date="2026-09-30",
                                       end_date="2500-12-31"),
                            c[1]: dict(slug="g1", rate_per_day=7.2, max_spread=4.5, min_size=50, start_date="2026-10-03",
                                       end_date="2500-12-31")},
             "2026-10-04": {c[3]: dict(slug="g3", rate_per_day=28.8, max_spread=3.0, min_size=100,
                                       start_date="2026-10-04", end_date="2500-12-31")}}
    d.mkdir(parents=True, exist_ok=True)
    for day, mk in snaps.items():
        (d / f"rewards_{day}.json").write_text(json.dumps({"day": day, "markets": mk}))
    return c


def check_reward_row(rec: Path, out: Path, sid: str, v: float, m: float) -> dict:
    """Independent check of one paid sample row: rebuild the competitors' Q from the raw outcome-0 ladder
    with plain loops (all levels), and compare with the top-5 approximation of book.bin."""
    meta = json.loads((out / "meta.json").read_text())
    sm = next(x for x in meta["segments"] if x["sid"] == sid)
    N = meta["N"]
    x3 = np.fromfile(out / "sports_x3.bin", np.float32).reshape(N, len(X3_COLS))
    book = np.fromfile(out / "book.bin", BOOK_DTYPE)
    r0, n = sm["row_offset"], sm["n_rows"]
    rows = [r for r in range(r0, r0 + n) if x3[r, 6] > 0 and not (book["flags"][r] & BOOK_FROM_DOWN)]
    assert rows, "no paid sample row from an outcome-0 book"
    r = rows[len(rows) // 2]
    t = sm["open_ts"] + (r - r0) * meta["sports"]["dt_s"]
    raw = pq.read_table(rec / "books").to_pandas()
    raw = raw[(raw.market_slug == sid.split("#")[0]) & (raw.outcome == "Bruins")]
    ts_ = np.array([int(x.timestamp()) for x in raw.captured_at])       # synth captures whole seconds
    raw = raw[(ts_ <= t) & ~raw.crossed.values.astype(bool)].iloc[-1]   # the latest valid snapshot in use

    def cut(px, sz):
        c = 0.0
        for p_, s_ in zip(px, sz):
            c += s_
            if c >= m - 1e-9:
                return p_
        return px[-1]
    mid = 0.5 * (cut(raw.bid_prices, raw.bid_sizes) + cut(raw.ask_prices, raw.ask_sizes))
    qb = sum(((v - abs(mid - p_) * 100) / v) ** 2 * s_ for p_, s_ in zip(raw.bid_prices, raw.bid_sizes)
             if abs(mid - p_) * 100 < v and s_ >= m)
    qa = sum(((v - abs(p_ - mid) * 100) / v) ** 2 * s_ for p_, s_ in zip(raw.ask_prices, raw.ask_sizes)
             if abs(p_ - mid) * 100 < v and s_ >= m)
    comp = min(qb, qa) + (abs(qb - qa) / 3 if 0.1 <= mid <= 0.9 else 0.0)
    assert abs(mid - x3[r, 4]) < 1e-6 and abs(comp - x3[r, 5]) < 1e-3 * max(1.0, comp), (mid, comp, x3[r, 4:6])
    # top-5 levels only (what adapt3 --rewards computes from book.bin)
    b = book[r]
    bp, ap_ = b["bid_px"] / 1000.0, b["ask_px"] / 1000.0
    qb5 = sum(((v - abs(mid - p_) * 100) / v) ** 2 * s_ for p_, s_ in zip(bp, b["bid_sz"])
              if p_ > 0 and abs(mid - p_) * 100 < v and s_ >= m)
    qa5 = sum(((v - abs(p_ - mid) * 100) / v) ** 2 * s_ for p_, s_ in zip(ap_, b["ask_sz"])
              if p_ < 1 and abs(p_ - mid) * 100 < v and s_ >= m)
    comp5 = min(qb5, qa5) + (abs(qb5 - qa5) / 3 if 0.1 <= mid <= 0.9 else 0.0)
    return {"row": int(r), "levels": [len(raw.bid_prices), len(raw.ask_prices)], "liq_mid": round(float(mid), 4),
            "comp_full": round(float(comp), 2), "comp_top5": round(float(comp5), 2)}


ENV3_PROBE = r"""
import json, sys, numpy as np
sys.path.insert(0, %(v3env)r)
from env3 import MmEnv3
res = []
for data in [%(data)r]:              # mm3 maps one data dir per process
    meta = json.load(open(data + '/meta.json'))
    out = []
    for i, sm in enumerate(meta['segments']):
        env = MmEnv3(num_envs=1, seed=1, data_dir=data, split=1, fixed_market=i, fixed_start=0, obs_size=80,
                     liq_rewards=1, size_mode=1, Q=1000, dt_other=1, dt_far=%(dt_far)d, far_s=%(far_s)d, pull_s=60,
                     log_interval=10 ** 9)
        env.reset(1)
        a = np.array([[3, 3, 2]], np.int32)                 # join both touches, size head 2
        steps, act = 0, 0.0
        while True:
            o, r, d, _, _ = env.step(a)
            assert np.isfinite(o).all() and np.isfinite(r).all()
            act = max(act, float(o[0, 55]))
            steps += 1
            if d[0]:
                break
        log = env._binding.vec_log(env.c_envs)
        env.close()
        out.append(dict(sid=sm['sid'], steps=steps, rows=log['episode_rows'], settle=log['settle_px'],
                        liq=log['liq_rewards'], pnl=log['pnl'], fills=log['fills'], rw_active=act))
    res.append(out)
print('PROBE3' + json.dumps(res))
"""


def env3_probe(v3env: Path, dirs: list, dt_far: int = 0, far_s: int = 7200) -> list:
    res = []
    for d in dirs:
        code = ENV3_PROBE % {"v3env": str(v3env), "data": str(d), "dt_far": dt_far, "far_s": far_s}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(v3env), timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"env3 probe failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        res += json.loads([x for x in r.stdout.splitlines() if x.startswith("PROBE3")][-1][6:])
    return res


def x3_variant(build: Path, dst: Path) -> Path:
    """What adapt3.py should do for sports-2 (proposal): x3.bin = sports_x3.bin. Built as a symlinked copy."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in build.iterdir():
        if f.is_file() and f.name != "x3.bin":
            (dst / f.name).unlink(missing_ok=True)
            (dst / f.name).symlink_to(f.resolve())
    (dst / "x3.bin").unlink(missing_ok=True)
    (dst / "x3.bin").symlink_to((build / "sports_x3.bin").resolve())
    return dst


def selftest(args) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sports_selftest_", dir=args.tmp or None))
    try:
        rec = tmp / "rec"
        synth(rec)
        conds = synth_rewards(tmp / "rewards")
        base = {**vars(args), "rec": [str(rec)], "dt": 1, "min_cover": 0.6, "offline": True, "no_gamma": True,
                "no_fetch_trades": True, "splits": None, "slugs": None, "allow_open": False, "now": None,
                "max_markets": 0, "keep_truncated": False, "rewards_dir": str(tmp / "rewards"), "kickoff_csv": None,
                "drop_early_starts": False, "no_legacy_settle": False, "min_window_s": 600.0,
                "cut_before_kickoff_s": 0, "bucket_segments": False}
        outs = {}
        for wu in ("a", "b"):
            o = tmp / f"build_{wu}"
            meta = build(argparse.Namespace(**{**base, "out": str(o), "seg": 300, "max_hours": 900 / 3600}))
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
        build(argparse.Namespace(**{**base, "out": str(o6), "dt": 6, "seg": 300, "max_hours": 1800 / 3600,
                                    "min_cover": 0.3}))
        verify(o6)
        # --seg 0: one episode per game window (first valid book row .. kickoff), all games in split 1
        sv = tmp / "splits_val.json"
        sv.write_text(json.dumps({"val": ["2000-01-01..2100-12-31"]}))
        ow = tmp / "build_w"
        mw = build(argparse.Namespace(**{**base, "out": str(ow), "seg": 0, "max_hours": None, "splits": str(sv)}))
        verify(ow)
        segs = {x["slug"]: x for x in mw["segments"]}
        g = sorted(segs)
        assert len(g) == 4 and mw["sports"]["stats"]["segments_dropped_short"] == 1, "game 1 (500 s) must be dropped"
        for x in mw["segments"]:
            assert x["at_cut"] and x["end_ts"] == x["kickoff_ts"] == x["cut_ts"] and x["n_rows"] > 900
            assert x["bucket"] == "30-5m" and not x["bucket_pure"] and x["to_kickoff_end_s"] == 0
        # reward programs: g0 in program via a later snapshot, g1 not started yet, g3 in program, g2 / g4 out
        prog = {s_: (segs[s_]["rw"]["in_program_any"], segs[s_]["rw"]["later_snapshot"]) for s_ in g}
        assert prog[g[0]] == (True, True) and prog[g[1]] == (False, False) and prog[g[2]] == (True, False) \
            and prog[g[3]] == (False, False), prog
        x3 = np.fromfile(ow / "sports_x3.bin", np.float32).reshape(mw["N"], len(X3_COLS))
        for x in mw["segments"]:
            r0, n = x["row_offset"], x["n_rows"]
            paid = int((x3[r0:r0 + n, 6] > 0).sum())
            minutes = len(np.unique((x["open_ts"] + np.arange(n)) // 60))
            if x["rw"]["in_program_any"]:
                assert minutes - 2 <= paid <= minutes and abs(x3[r0:r0 + n, 6].max() - x["rw"]["rate_day_last"] / 1440) < 1e-6
            else:
                assert paid == 0
        chk = check_reward_row(rec, ow, segs[g[0]]["sid"], 9.0, 100.0)
        assert chk["comp_full"] > chk["comp_top5"], chk
        print("seg 0 build OK: 4 games, one episode each; full-ladder reward stats == independent recomputation:",
              json.dumps(chk))
        ow2 = tmp / "build_w_nolegacy"
        build(argparse.Namespace(**{**base, "out": str(ow2), "seg": 0, "max_hours": None, "splits": str(sv),
                                    "no_legacy_settle": True}))
        verify(ow2)
        # cut 300 s before kickoff, and --bucket-segments (windows split at 30 m / 5 m here)
        oc = tmp / "build_cut300"
        mc = build(argparse.Namespace(**{**base, "out": str(oc), "seg": 0, "max_hours": None, "splits": str(sv),
                                          "cut_before_kickoff_s": 300}))
        verify(oc)
        for x, y in zip(sorted(mc["segments"], key=lambda z: z["sid"]), sorted(mw["segments"], key=lambda z: z["sid"])):
            assert x["end_ts"] == x["cut_ts"] == x["kickoff_ts"] - 300 and x["n_rows"] == y["n_rows"] - 300
            assert x["to_kickoff_end_s"] == 300 and x["bucket"] == "30-5m" and x["bucket_pure"]
        ob = tmp / "build_buckets"
        mb_ = build(argparse.Namespace(**{**base, "out": str(ob), "seg": 0, "max_hours": None, "splits": str(sv),
                                          "bucket_segments": True}))
        verify(ob)
        bks = sorted((x["slug"], x["bucket"], x["n_rows"], x["at_cut"]) for x in mb_["segments"])
        assert len(bks) == 8 and all(x["bucket_pure"] for x in mb_["segments"]), bks
        assert {b_[1] for b_ in bks} == {"30-5m", "5-0m"} and all(b_[2] == 301 for b_ in bks if b_[1] == "5-0m")
        print("cut 300 s and --bucket-segments builds OK:", bks[:2])
        if args.no_env:
            return
        # ---- v3env: adapt3.py (today's sports branch) + env3, and the proposed x3 = sports_x3.bin variant
        v3 = Path(args.v3env).resolve()
        if (v3 / "adapt3.py").exists():
            r = subprocess.run([sys.executable, str(v3 / "adapt3.py"), str(ow)], capture_output=True, text=True,
                               cwd=str(v3), timeout=600)
            assert r.returncode == 0, r.stderr[-3000:]
            print("adapt3.py on the seg 0 build:", r.stdout.strip().splitlines()[-1])
            ext = np.fromfile(ow / "markets3.bin", np.dtype([("settle_price", "<f8"), ("event_ts", "<i8"),
                                                              ("rest", "V48")]))
            assert np.array_equal(ext["settle_price"].astype(np.float32),
                                  np.array([x["settle_price"] for x in mw["segments"]], np.float32))
            assert np.array_equal(ext["event_ts"], [x["kickoff_ts"] for x in mw["segments"]])
            if sorted(v3.glob("binding3*.so")):
                ox = x3_variant(ow, tmp / "build_w_x3")
                p0, p1 = env3_probe(v3, [ow, ox])
                for x, y, sm in zip(p0, p1, mw["segments"]):
                    assert x["rows"] == sm["n_rows"] - 1 and abs(x["settle"] - sm["settle_price"]) < 1e-6, (x, sm)
                    assert (x["liq"] > 0) == sm["rw"]["in_program_any"], (x, sm["rw"])
                    assert abs(x["liq"] - y["liq"]) < 1e-6 and x["fills"] == y["fills"]
                    assert x["rw_active"] == y["rw_active"] == float(sm["rw"]["in_program_any"])  # adapt3 copies sports_x3.bin
                print("env3 (adapt3 sports-2): every episode runs to kickoff, settles at settle_price, liq rewards "
                      "paid only in program:", json.dumps([{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in x.items()} for x in p0]))
                print("env3 with x3.bin = sports_x3.bin (proposed adapt3): same rewards and fills, rw_active "
                      "observation on in-program games")
                pf = env3_probe(v3, [ow], dt_far=30, far_s=300)[0]
                assert all(a_["steps"] < b_["steps"] for a_, b_ in zip(pf, p0))
                print("env3 dt_far 30 / far_s 300: steps", [x["steps"] for x in pf], "vs", [x["steps"] for x in p0])
            else:
                print(f"no binding3*.so in {v3}: env3 probe skipped (build it, or pass --v3env <built copy>)")
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
    ap.add_argument("--dt", type=int, default=1, help="grid step in seconds (1 = the crypto build's grid; > 1 is "
                    "discouraged for v3env: latency, markouts and flags count rows; use mm3 dt_far instead)")
    ap.add_argument("--seg", type=int, default=14400, help="rows per episode: 300 / 900 / 14400 (mm.h buckets), or 0 = "
                    "one episode per game window, first valid book row .. kickoff, any length (v3env only)")
    ap.add_argument("--max-hours", type=float, default=None, help="window before the CUT (floored to whole "
                    "segments); default 24 for --seg > 0, no cap (first book row) for --seg 0")
    ap.add_argument("--min-window-s", type=float, default=600.0, help="--seg 0: drop games with a shorter window")
    ap.add_argument("--cut-before-kickoff-s", type=int, default=0,
                    help="episodes end (and settle) this many s before the scheduled kickoff (default 0 = at kickoff, "
                         "D15; D14's 3600 was withdrawn)")
    ap.add_argument("--bucket-segments", action="store_true",
                    help="--seg 0: split each game window at the bucket boundaries 24h/6h/1h/30m/5m before kickoff")
    ap.add_argument("--rewards-dir", default=str(REWARDS_DIR) if REWARDS_DIR.is_dir() else None,
                    help="rewards_snapshot.py daily files (rewards_<day>.json) for the per-day reward programs; "
                         "'' = none (rate 0 everywhere)")
    ap.add_argument("--kickoff-csv", default=str(KICKOFF_CSV), help="kickoff_watch.py kickoffs.csv (actual-start "
                    "labels; '' = none)")
    ap.add_argument("--drop-early-starts", action="store_true",
                    help="skip games kickoff_watch saw live before the scheduled kickoff (in-play rows in the window)")
    ap.add_argument("--no-legacy-settle", action="store_true",
                    help="do not write the legacy settle copy (markets.bin log_strike_vs_ul_open + flag 16); "
                         "needs an adapt3.py that reads meta.json segments (env_sports/ cannot read such a build)")
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
    ap.add_argument("--v3env", default=str(PM / "v3env"), help="selftest: v3env dir (adapt3.py; env3 needs a built "
                    "binding3*.so there, else the env3 probe is skipped)")
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
