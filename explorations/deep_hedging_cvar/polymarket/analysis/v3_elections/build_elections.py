#!/usr/bin/env python3
"""Build env binaries for track E (election reward farming, DECISIONS.md D11) from the recorder's
election-market books, the hourly reward-config snapshots and the markets' fills.

    source ../../../.venv/bin/activate          # needs pandas, pyarrow, scipy, orjson
    python build_elections.py --split-days days.json --trades-source data-api
    python build_elections.py --window-s 300 --split-days '{"2026-10-15": "train"}' \
        --trades-source data-api --out build_smoke/w300            # smoke test (5-minute episodes)

Output (--out, default ./build/): the schema of data/build/ so data/load.py and the C env (mm.h,
env.MmEnv(data_dir=...)) read it unchanged:
  ticks.bin, markets.bin, meta.json, splits.json           (data/build.py layout, F = 52)
  feat.bin, book.bin, fills.bin, fill_index.bin, feat_meta.json   (data/features.py layout, G = 44)
  feat_x.bin   float32 [N, X]: election / liquidity-reward columns (names in feat_meta.json["extra"]).
               With --feat-layout wide they are appended to feat.bin instead ([N, 44 + X]); the
               current env then refuses feat.bin (MM_G = 44), a v3 env would read them.
BUILD.md (next to this file) documents every decision; this docstring is the short version.

Episode = one recorded market x one window of --window-s seconds aligned to UTC midnight (default 4 h,
D16), cut at coverage gaps > --max-gap (15 s), the market end and the cutoff (pieces < --min-episode-s
dropped); days whose gaps exceed --day-max-gap-s are purged; GAPS.md / gaps_per_day.csv are written;
older text below describes the per-window rules, rows t = open..open + window inclusive (n_rows = window + 1, as data/build.py), Yes token
(outcome 0, selection.json tokens[0]) in place of Up. Kept only if the recorder covered the whole
window (recorder/merge.py's rule: coverage log, max single gap --max-gap, total --max-total-gap, and
the coverage segment holding the open started at or before it), the market's end date and the
--cutoff (election night) are after the window end, its day is not excluded by --split-days, and its
fills are complete over the window. --allow-partial keeps partially covered windows (flag PARTIAL);
rows outside coverage then have book_valid = 0.

Causality: as data/build.py (book = latest uncrossed snapshot with captured second <= t, trades with
block time == t) plus: a snapshot is only used inside the coverage segment it was recorded in, and
the reward program at row t is the latest hourly snapshot taken at or before t (selection.json is the
first snapshot). Nothing reads outcomes; markets are unresolved at build time (winner_up = -1).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent                 # analysis/v3_elections
PM = HERE.parents[1]                                   # polymarket/
DATA = PM / "data"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(PM / "recorder"))
import build as B  # noqa: E402  data/build.py (import only; main() is guarded)
import features as FT  # noqa: E402  data/features.py (layouts, helpers; main() is guarded)
import load as LD  # noqa: E402
from polyrec import coverage as cov  # noqa: E402
from polyrec.sinks import dataset_files  # noqa: E402

pa.set_cpu_count(1)
pa.set_io_thread_count(2)

BUILD_VERSION = 1
DEFAULT_OUT = HERE / "build"
SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "purged": 3}
# markets.bin flags: data/build.py's 1/2/4/8 plus
FLAG_ELECTION = 16          # no underlying, Yes/No frame (outcome 0 = Up slot); ul_* columns are 0
FLAG_UNRESOLVED = 32        # winner_up = -1: unresolved at build time, terminal settlement must use a mark
FLAG_PARTIAL = 64           # --allow-partial window: some rows outside recorder coverage (book_valid = 0)
FLAG_IN_PROGRAM = 128       # the market was in the liquidity-reward program at some row of the window
FLAG_TRADES_INCOMPLETE = 256  # fills source does not cover the whole window (only with --allow-partial / --no-trades)
NO_COIN = -1                # coin_id: no underlying
REWARD_BAND = (0.10, 0.90)  # single-sided scoring allowed inside, c = 3 (reward_terms.py)
REWARD_C = 3.0
LIQ_SEED = 20261015         # sample-second draws: one uniform second per UTC minute per (condition, window)
MM_OFFS = (0, 2, 1, 0, -1, -2, -4)   # mm.h MM_OFFS: price head a -> ticks better than the touch
MM_SIZES = (1.0, 2.0, 5.0)           # mm.h MM_SIZES: size head k -> x min_order_size
COV_TOL_S = 1.0             # a snapshot may precede its coverage segment start by this much

# ----------------------------------------------------------------------------- feat_x columns
X_DOCS = [
    ("rw_in_program", "1 if the market is in the liquidity-reward program (latest reward snapshot <= t)"),
    ("rw_daily_rate", "reward pool of the market, USD per day (total_daily_rate; 0 if not in program)"),
    ("rw_min_size", "rewards_min_size (shares per order) in force at t"),
    ("rw_max_spread_c", "rewards_max_spread v (cents from the adjusted mid) in force at t"),
    ("rw_cfg_age_s", "t - taken_at of the reward snapshot in force (s)"),
    ("rw_adj_mid", "size-cutoff-adjusted mid of the recorded FULL ladder (Yes frame, without us), "
                   "cutoff = rw_min_size; -1 if one side is empty (unscorable). = feat liq_mid"),
    ("rw_comp_q_bid", "competitors' Q on the Yes bid side: sum over levels within v of ((v - d)/v)^2 * size, "
                      "levels with size >= min size (whole level treated as one order)"),
    ("rw_comp_q_ask", "same on the Yes ask side (= No bids)"),
    ("rw_comp_q", "competitors' Q_min, 'hi' bound of reward_terms.competitor_q: min(qb, qa) + |qb - qa| / 3 inside "
                  "[0.10, 0.90], else min(qb, qa). = feat liq_comp"),
    ("rw_two_sided_only", "1 if rw_adj_mid is outside [0.10, 0.90]: single-sided quotes score 0"),
    ("rw_sample", "1 if row t is this UTC minute's reward sample second (seeded draw)"),
    ("rw_usd_per_sample", "rw_daily_rate / 1440: USD shared by all makers at one sample (feat liq_pi = this "
                          "on sample rows with a scorable book)"),
]
X_DOCS += [(f"rw_bid_px_a{a}", f"env bid price (USD) for bid head {a} (mm.h mm_targets, tick_rule 0, ask off); "
                               "0 = no quote (below one tick)") for a in range(1, 7)]
X_DOCS += [(f"rw_ask_px_a{a}", f"env ask price (USD) for ask head {a} (bid off); 0 = no quote") for a in range(1, 7)]
X_DOCS += [(f"rw_bid_score_a{a}", f"((v - d)/v)^2 for the bid of head {a}, d = |px - rw_adj_mid| in cents, 0 if "
                                  "d >= v, unscorable or no quote: reward score per share") for a in range(1, 7)]
X_DOCS += [(f"rw_ask_score_a{a}", f"same for the ask of head {a}") for a in range(1, 7)]
X_DOCS += [(f"rw_size_ok_{k}", f"1 if size head {k} (min_order_size x {MM_SIZES[k]:g}, size_mult 1) >= rw_min_size "
                               "and the market is in program") for k in range(3)]
X_DOCS += [
    ("rw_min_size_mult", "rw_min_size / min_order_size: size multiple a quote needs to be eligible"),
    ("x_covered", "1 if row t is inside recorder coverage (gaps <= --max-gap bridged)"),
    ("x_log_days_to_end", "log1p(days from t to the market's end_date)"),
]
X_COLS = [c for c, _ in X_DOCS]
X_COLS += [f"xpad_{i}" for i in range((-len(X_COLS)) % 4)]
X = len(X_COLS)
XI = {c: i for i, c in enumerate(X_COLS)}

# what the 44 env feature columns mean for elections (features.py docs otherwise apply)
FEAT_ELECTION_DOCS = {
    "p_logit": "logit(p_model) / 3 with p_model = mid proxy (see p_model)",
    "gap_c": "(p_model - mark) cents / 5: 0 while the book is two-sided",
    "dgap5_c": "change of gap over 5 s",
    "z": "0: no underlying / pricer", "sdv_bp": "0: no pricer", "sigma_1m_bp": "0: no underlying",
    "tau_frac": "seconds to the WINDOW end / window length (not to resolution: see feat_x x_log_days_to_end)",
    "log_tau": "log1p(seconds to window end) / log1p(14400) (exceeds 1 for windows > 4 h)",
    "in_twap": "0: no closing TWAP", "delta_usd": "0: no underlying",
    "ulz5": "0: no underlying", "ulz30": "0", "ulz120": "0",
    "len_5m": "1 if the window is 300 s", "len_15m": "1 if the window is 900 s",
    "len_4h": "1 if the window is 14400 s (all three 0 for a 1-day window)",
    "p_model": "NO pricer: two-sided book mid at t, else the last two-sided mid of the episode, else 0.5",
    "mark": "book mid if two-sided and valid, else p_model (= p_model always)",
    "delta_usd_raw": "0", "sigma": "0", "sqrt_v": "0",
    "liq_mid": "size-cutoff adjusted mid of the full recorded ladder with the market's own rewards_min_size; -1 unscorable",
    "liq_comp": "competitors' Q_min ('hi' bound) with the market's own v and min size (feat_x rw_comp_q)",
    "liq_pi": "rw_daily_rate / 1440 on the minute's sample second when in program and scorable, else 0",
    "bid_minus_p_c": "(best bid - p_model) cents / 5", "ask_minus_p_c": "(best ask - p_model) cents / 5",
}


def utc_ts(s) -> int:
    t = pd.Timestamp(s)
    return int((t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")).timestamp())


def day_of(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- inputs
def load_selection(path: Path, args) -> tuple[pd.DataFrame, str]:
    sel = json.loads(path.read_text())
    rows = []
    for m in sel["selected"]:
        assert len(m["tokens"]) == 2, m["slug"]
        rows.append(dict(slug=m["slug"], condition_id=m["condition_id"], tok0=str(m["tokens"][0]),
                         tok1=str(m["tokens"][1]), outcomes=list(m["outcomes"]), end_ts=utc_ts(m["end_date"]),
                         rate_per_day=float(m.get("rate_per_day") or 0.0),
                         rewards_min_size=float(m.get("rewards_min_size") or 0.0),
                         rewards_max_spread=float(m.get("rewards_max_spread") or 0.0),
                         neg_risk=bool(m.get("neg_risk")), question=m.get("question", "")))
    df = pd.DataFrame(rows)
    full = df.copy()
    if args.markets:
        p = Path(args.markets)
        want = set((p.read_text().split() if p.exists() else args.markets.split(",")))
        df = df[df.slug.isin(want) | df.condition_id.isin(want)]
    if args.min_rate:
        df = df[df.rate_per_day >= args.min_rate]
    if args.top_n:
        df = df.sort_values(["rate_per_day", "slug"], ascending=[False, True]).head(args.top_n)
    return df.sort_values("slug").reset_index(drop=True), full, sel.get("taken_at")


def load_reward_cfg(rec: Path, sel_full: pd.DataFrame, sel_taken: str):
    """{condition_id: dict of arrays ts, in_program, rate, min_size, max_spread} (step functions, sorted)."""
    cols = ["taken_at", "condition_id", "in_program", "total_daily_rate", "rewards_min_size", "rewards_max_spread"]
    files = sorted((rec / "rewards").glob("*/rewards_*.parquet"))
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    base = pd.DataFrame({"taken_at": sel_taken, "condition_id": sel_full.condition_id, "in_program": True,
                         "total_daily_rate": sel_full.rate_per_day, "rewards_min_size": sel_full.rewards_min_size,
                         "rewards_max_spread": sel_full.rewards_max_spread})
    r = pd.concat([base] + frames, ignore_index=True)
    r = r[r.condition_id.isin(set(sel_full.condition_id))].copy()
    r["ts"] = ((pd.to_datetime(r.taken_at, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype(np.int64)
    for c in ("rewards_min_size", "rewards_max_spread", "total_daily_rate"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.sort_values(["condition_id", "ts"], kind="stable").drop_duplicates(["condition_id", "ts"], keep="last")
    # scorer params persist across a snapshot that lacks them (market out of program): ffill per market
    r[["rewards_min_size", "rewards_max_spread"]] = r.groupby("condition_id")[
        ["rewards_min_size", "rewards_max_spread"]].ffill().fillna(0.0)
    r["in_program"] = r.in_program.astype(bool)
    r["rate"] = np.where(r.in_program, r.total_daily_rate.fillna(0.0), 0.0)
    out = {}
    for c, g in r.groupby("condition_id", sort=False):
        out[c] = {"ts": g.ts.values.astype(np.int64), "in_program": g.in_program.values.astype(np.float64),
                  "rate": g.rate.values.astype(np.float64), "min_size": g.rewards_min_size.values.astype(np.float64),
                  "max_spread": g.rewards_max_spread.values.astype(np.float64)}
    return out, [str(f.relative_to(rec)) for f in files]


def cfg_at(cfg: dict, t: np.ndarray) -> dict:
    """Reward program in force at times t (latest snapshot taken at or before t)."""
    i = np.searchsorted(cfg["ts"], t, side="right") - 1
    ok = i >= 0
    ii = np.maximum(i, 0)
    o = {k: np.where(ok, cfg[k][ii], 0.0) for k in ("in_program", "rate", "min_size", "max_spread")}
    o["age"] = np.where(ok, t - cfg["ts"][ii], 0.0)
    return o


def coverage_segments(rec: Path, max_gap: float) -> tuple[dict, dict]:
    """slug -> merged coverage intervals (recorder coverage log), and the same with gaps <= max_gap bridged."""
    iv = cov.live_intervals(cov.read_events(rec))
    seg = {}
    for s, v in iv.items():
        out = []
        for a, b in v:
            if out and a - out[-1][1] <= max_gap:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        seg[s] = (np.array([a for a, _ in out]), np.array([b for _, b in out]))
    return iv, seg


def recorded_days(rec: Path) -> list[str]:
    base = rec / "books"
    return sorted(d.name for d in base.iterdir() if d.is_dir()) if base.exists() else []


def parse_split_days(s: str | None) -> dict:
    if not s:
        return {}
    p = Path(s)
    d = json.loads(p.read_text() if p.exists() else s)
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        pd.Timestamp(k)   # validates the date
        if isinstance(v, int):
            v = {0: "train", 1: "val", 2: "test", 3: "purged"}[v]
        v = str(v).lower()
        if v not in ("train", "val", "test", "purged", "exclude"):
            raise SystemExit(f"--split-days: bad value {v!r} for {k} (train/val/test/purged/exclude)")
        out[k] = v
    return out


# ----------------------------------------------------------------------------- books
BOOK_COLS = ["market_slug", "token_id", "captured_at", "bid_prices", "bid_sizes", "ask_prices", "ask_sizes", "crossed"]
EXTRA = ["amid", "qb", "qa"]


def S(d, v):
    """Polymarket's order score ((v - d) / v)^2 for d < v, else 0 (v = 0 -> 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((d < v) & (v > 0), ((v - d) / np.where(v > 0, v, 1.0)) ** 2, 0.0)


def side_ladder(tbl, pcol, scol, m):
    """Flattened ladder of one side and the price where cumulative depth reaches m (row-wise)."""
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
    """Per raw snapshot, in the token's own frame: adjusted mid (NaN if a side is empty), competitors' Q of
    the bid and the ask side (reward_terms.adjusted_mid / side_scores on the full ladder)."""
    rb, pb, sb, cb = side_ladder(tbl, "bid_prices", "bid_sizes", m)
    ra, pa_, sa, ca = side_ladder(tbl, "ask_prices", "ask_sizes", m)
    mid = 0.5 * (cb + ca)
    n = len(m)
    okm = np.isfinite(mid)
    mid0 = np.where(okm, mid, 0.0)
    qb = np.bincount(rb, weights=S(np.abs(mid0[rb] - pb) * 100, v[rb]) * sb * (sb >= m[rb] - 1e-9), minlength=n)
    qa = np.bincount(ra, weights=S(np.abs(pa_ - mid0[ra]) * 100, v[ra]) * sa * (sa >= m[ra] - 1e-9), minlength=n)
    return mid, np.where(okm, qb, 0.0), np.where(okm, qa, 0.0)


def compact_file(f: Path, tokmap: dict, cfgs: dict) -> pd.DataFrame:
    """One recorder book file -> one row per (token, captured second) with build.py's top-5 columns, plus the
    reward stats mapped to the Yes (Up) frame. Unknown tokens are dropped."""
    t = pq.read_table(f, columns=BOOK_COLS)
    tok = t.column("token_id").to_numpy(zero_copy_only=False).astype(str)
    us = t.column("captured_at").cast("int64").to_numpy()
    small = pd.DataFrame({"tok": tok, "us": us})
    small = small[small.tok.isin(list(tokmap))]
    small["ts"] = small.us // 1_000_000
    small = small.sort_values(["tok", "us"], kind="stable").drop_duplicates(["tok", "ts"], keep="last")
    if not len(small):
        return pd.DataFrame()
    t = t.take(pa.array(small.index.values))
    info = [tokmap[x] for x in small.tok.values]
    slug = np.array([i[0] for i in info], dtype=object)
    is_up = np.array([i[1] for i in info], dtype=bool)
    cond = np.array([i[2] for i in info], dtype=object)
    ts = small.ts.values.astype(np.int64)
    m = np.zeros(len(ts))
    v = np.zeros(len(ts))
    for c in np.unique(cond):
        k = cond == c
        cf = cfg_at(cfgs[c], ts[k]) if c in cfgs else {"min_size": 0.0, "max_spread": 0.0}
        m[k], v[k] = cf["min_size"], cf["max_spread"]
    bpx, bsz, bdep, bn = B.ladder_topk(t, "bid_prices", "bid_sizes")
    apx, asz, adep, an = B.ladder_topk(t, "ask_prices", "ask_sizes")
    amid, qb, qa = reward_stats(t, m, v)
    df = pd.DataFrame({"slug": slug, "is_up": is_up, "ts": ts,
                       "crossed": t.column("crossed").to_numpy(zero_copy_only=False).astype(bool),
                       "bdep": bdep, "adep": adep, "bn": bn, "an": an})
    for k in range(B.NLEV):
        df[f"bp{k}"], df[f"bs{k}"], df[f"ap{k}"], df[f"as{k}"] = bpx[:, k], bsz[:, k], apx[:, k], asz[:, k]
    # reward stats into the Yes frame (No bid at q == Yes ask at 1 - q)
    df["amid"] = np.where(is_up, amid, 1.0 - amid)
    df["qb"] = np.where(is_up, qb, qa)
    df["qa"] = np.where(is_up, qa, qb)
    return df


class Compactor:
    """compact_file with a cache for finished hourly files (keyed by file identity, the reward snapshots that
    can affect it, the selection and this script's version). In-progress parts are never cached."""

    def __init__(self, cache: Path, tokmap, cfgs, reward_files, rec: Path, sel_sha: str):
        self.cache, self.tokmap, self.cfgs = cache, tokmap, cfgs
        self.reward_files = [(f, (rec / f).stat().st_size) for f in reward_files]
        self.sel_sha = sel_sha
        self.hits = self.misses = 0

    def __call__(self, f: Path) -> pd.DataFrame:
        if "_parts" in f.parts:
            return compact_file(f, self.tokmap, self.cfgs)
        st = f.stat()
        hour = f.stem.split("_")[-1]                       # <day>T<HH>
        rw = [x for x in self.reward_files if Path(x[0]).stem.split("_")[-1] <= hour]
        key = hashlib.sha1(json.dumps([f.name, st.st_size, st.st_mtime_ns, rw, self.sel_sha, BUILD_VERSION]).encode()
                           ).hexdigest()[:16]
        p = self.cache / f"{f.stem}.{key}.parquet"
        if p.exists():
            self.hits += 1
            return pd.read_parquet(p)
        self.misses += 1
        df = compact_file(f, self.tokmap, self.cfgs)
        self.cache.mkdir(parents=True, exist_ok=True)
        for old in self.cache.glob(f"{f.stem}.*.parquet"):
            old.unlink()
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        return df


def read_day(compactor, rec: Path, day: str, hours: set | None = None) -> pd.DataFrame:
    for attempt in range(3):          # the recorder may absorb parts into an hourly file while we list
        try:
            files = dataset_files(rec, "books", [day])
            if hours is not None:
                files = [f for f in files if any(f"T{h}" in f.name for h in hours)]
            frames = [compactor(f) for f in files]
            frames = [x for x in frames if len(x)]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(2.0)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return df
    return df.sort_values(["slug", "is_up", "ts"], kind="stable").drop_duplicates(["slug", "is_up", "ts"], keep="last")


def carry_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Last row and last uncrossed row per (market, token): all a later day needs to forward-fill."""
    if not len(df):
        return df
    d = df.sort_values(["slug", "is_up", "ts"], kind="stable")
    a = d.groupby(["slug", "is_up"], sort=False).tail(1)
    b = d[~d.crossed].groupby(["slug", "is_up"], sort=False).tail(1)
    return dedup(pd.concat([a, b]))


# ----------------------------------------------------------------------------- trades
TRADE_COLS = ["slug", "tx", "timestamp", "outcome_idx", "side", "price", "size", "is_taker", "aggressor_up"]


def classify_data_api(slug: str, d: dict) -> pd.DataFrame:
    """data-api rows ([tx, ts, wallet, side, outcomeIndex, price, size, asset]) -> data/raw trades rows; taker
    legs are the takerOnly=true rows (as data/fetch_trades.build_outputs)."""
    taker_keys = {}
    for r in d["taker"]:
        k = (r[0], r[2], r[3], r[4], r[5], r[6])
        taker_keys[k] = taker_keys.get(k, 0) + 1
    rows = []
    for r in d["all"]:
        k = (r[0], r[2], r[3], r[4], r[5], r[6])
        is_t = taker_keys.get(k, 0) > 0
        if is_t:
            taker_keys[k] -= 1
        rows.append((slug, r[0], int(r[1]), int(r[4]), 1 if r[3] == "BUY" else -1, float(r[5]), float(r[6]), is_t))
    by_tx = {}
    for r in d["taker"]:
        by_tx.setdefault(r[0], r)
    for k, n in taker_keys.items():        # taker rows missing from the takerOnly=false listing
        for _ in range(n):
            r = by_tx[k[0]]
            rows.append((slug, k[0], int(r[1]), int(k[3]), 1 if k[2] == "BUY" else -1, float(k[4]), float(k[5]), True))
    t = pd.DataFrame(rows, columns=TRADE_COLS[:-1])
    return t


def finish_trades(t: pd.DataFrame) -> pd.DataFrame:
    if not len(t):
        return pd.DataFrame(columns=TRADE_COLS)
    t = t.astype({"slug": str, "tx": str, "timestamp": np.int64, "outcome_idx": np.int8, "side": np.int8,
                  "price": np.float64, "size": np.float64, "is_taker": bool})
    tk = t[t.is_taker]
    up_side = np.where(tk.outcome_idx == 0, tk.side, -tk.side)
    agg = pd.Series(up_side, index=tk.tx.values)
    agg = agg[~agg.index.duplicated()]
    t = t.copy()
    t["aggressor_up"] = t.tx.map(agg).fillna(0).astype(np.int8)
    return t.sort_values(["slug", "timestamp", "tx", "is_taker"], ascending=[True, True, True, False],
                         kind="stable").reset_index(drop=True)


def trades_data_api(mk: pd.DataFrame, cache: Path, need_to: dict, workers: int, refresh: bool):
    """Fallback source: data-api /trades?market=<condition id> (newest first, <= 20000 rows per query): fetch
    takerOnly=false (all legs) and takerOnly=true (taker legs). Returns trades and per-slug [from, to] seconds
    over which the listing is complete."""
    import fetch_trades as FTR   # data/fetch_trades.py: http_json / fetch_all (import only)
    cache.mkdir(parents=True, exist_ok=True)

    def one(row):
        p = cache / f"{row.condition_id}.json.gz"
        if p.exists() and not refresh:
            d = json.loads(gzip.decompress(p.read_bytes()))
            if d["fetched_at"] >= need_to.get(row.slug, 0) + 600:   # data-api lags block time a little
                return row.slug, d
        fetched_at = time.time()
        al, ta = FTR.fetch_all(row.condition_id, False)
        tk, tt = FTR.fetch_all(row.condition_id, True)
        d = {"fetched_at": fetched_at, "condition_id": row.condition_id, "all": FTR.trim(al), "taker": FTR.trim(tk),
             "truncated_all": bool(ta), "truncated_taker": bool(tt)}
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(d).encode()))
        tmp.replace(p)
        return row.slug, d

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, mk.itertuples()))
    frames, span = [], {}
    for slug, d in res.items():
        t = classify_data_api(slug, d)
        frames.append(t)
        lo = -np.inf
        for key in ("all", "taker"):
            if d[f"truncated_{key}"] and d[key]:
                lo = max(lo, min(int(r[1]) for r in d[key]) + 1)   # the oldest second may be cut
        span[slug] = (lo, d["fetched_at"] - 120.0)                # data-api indexes trades with a short delay
    t = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRADE_COLS[:-1])
    return finish_trades(t), span


def trades_onchain(mk: pd.DataFrame, src: Path):
    """Census source: onchain/extract.py output (fills/date=D/*.parquet, e.g. extract.py --tokens <map from
    --emit-tokens>), converted with onchain/to_trades.convert. Complete for every whole UTC day present."""
    sys.path.insert(0, str(PM / "onchain"))
    import to_trades as TT
    tokmap = pd.concat([pd.DataFrame({"token_id": mk.tok0, "slug": mk.slug, "outcome_idx": 0}),
                        pd.DataFrame({"token_id": mk.tok1, "slug": mk.slug, "outcome_idx": 1})], ignore_index=True)
    parts, days = [], []
    for d in sorted((src / "fills").glob("date=*")):
        files = sorted(d.glob("*.parquet"))
        day_file = [f for f in files if f.name == "day.parquet"]
        files = day_file or files
        parts += [pd.read_parquet(f) for f in files]
        days.append(d.name.split("=", 1)[1])
    fills = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    fills["token_id"] = fills.token_id.astype(str)
    t = TT.convert(fills, tokmap) if len(fills) else pd.DataFrame(columns=TRADE_COLS)
    covered = [(utc_ts(x), utc_ts(x) + 86400) for x in days]
    span = {s: covered for s in mk.slug}
    return t[TRADE_COLS].reset_index(drop=True), span


def trades_complete(span, o: int, e: int) -> bool:
    if span is None:
        return False
    if isinstance(span, list):     # on-chain: whole days
        return all(any(a <= x < b for a, b in span) for x in range(o - o % 86400, e + 1, 86400))
    lo, hi = span
    return lo <= o and e <= hi


# ----------------------------------------------------------------------------- gamma
def gamma_meta(mk: pd.DataFrame, cache: Path, offline: bool) -> dict:
    """condition id -> tick, min order size, fees from gamma (cached). Defaults if offline/missing."""
    p = cache / "gamma.json"
    got = json.loads(p.read_text()) if p.exists() else {}
    need = [c for c in mk.condition_id if c not in got]
    if need and not offline:
        import fetch_trades as FTR
        for closed in ("", "&closed=true"):
            need = [c for c in mk.condition_id if c not in got]
            for i in range(0, len(need), 20):
                q = "&".join(f"condition_ids={c}" for c in need[i:i + 20])
                for g in FTR.http_json(f"https://gamma-api.polymarket.com/markets?{q}&limit=100{closed}"):
                    fs = g.get("feeSchedule") or {}
                    got[g["conditionId"]] = {
                        "tick": float(g.get("orderPriceMinTickSize") or 0.01), "min_size": float(g.get("orderMinSize") or 5),
                        "fees_enabled": bool(g.get("feesEnabled")), "fee_rate": float(fs.get("rate") or 0.0),
                        "fee_exponent": float(fs.get("exponent") or 1.0), "rebate_rate": float(fs.get("rebateRate") or 0.0),
                        "fee_type": g.get("feeType"), "fetched_at": time.time()}
        cache.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(got, indent=0))
    default = {"tick": 0.01, "min_size": 5.0, "fees_enabled": False, "fee_rate": 0.0, "fee_exponent": 1.0,
               "rebate_rate": 0.0, "fee_type": None, "default": True}
    return {c: got.get(c, default) for c in mk.condition_id}


# ----------------------------------------------------------------------------- one episode
def episode_rows(ep, mb: pd.DataFrame, seg, trades_g, cfg, gm, L: int):
    """ticks [n, F], feat [n, 44], feat_x [n, X], book records, maker legs (rows relative), stats."""
    o, e = ep["open_ts"], ep["end_ts"]
    n = L + 1
    C, CI = B.C, FT.CI
    tgrid = np.arange(o, e + 1, dtype=np.int64)
    Xt = np.zeros((n, B.F))
    Xt[:, C["seconds_to_close"]] = e - tgrid
    Xt[:, C["const_one"]] = 1.0
    # ---- coverage segments: row covered, and the start of its segment
    if seg is not None and len(seg[0]):
        si = np.searchsorted(seg[0], tgrid, side="right") - 1
        covered = (si >= 0) & (tgrid <= seg[1][np.maximum(si, 0)])
        seg_lo = np.where(covered, seg[0][np.maximum(si, 0)], np.inf)
    else:
        covered = np.zeros(n, bool)
        seg_lo = np.full(n, np.inf)
    # ---- book (data/build.py: merged_snapshots, latest uncrossed snapshot <= t)
    ex = np.full((n, 3), np.nan)
    ok = np.zeros(n, bool)
    if mb is not None and len(mb):
        snap = B.merged_snapshots(mb)
        up = mb[mb.is_up].set_index("ts")[EXTRA].reindex(snap.index)
        dn = mb[~mb.is_up].set_index("ts")[EXTRA].reindex(snap.index)
        snap[EXTRA] = np.where(snap.from_down.values[:, None], dn.values, up.values)
        sts = snap.index.values.astype(np.int64)
        j_raw = np.searchsorted(sts, tgrid, side="right") - 1
        raw_ok = (j_raw >= 0) & covered & (sts[np.maximum(j_raw, 0)] >= seg_lo - COV_TOL_S)
        Xt[:, C["crossed_raw"]] = np.where(raw_ok, snap.raw_crossed.values[np.maximum(j_raw, 0)], 0)
        vsnap = snap[snap.valid.values]
        vts = vsnap.index.values.astype(np.int64)
        j = np.searchsorted(vts, tgrid, side="right") - 1
        jj = np.maximum(j, 0)
        ok = (j >= 0) & covered & (vts[jj] >= seg_lo - COV_TOL_S if len(vts) else False)
        if len(vts):
            Xt[:, C["book_age"]] = np.where(ok, tgrid - vts[jj], 0)
            Xt[:, C["book_from_down"]] = np.where(ok, vsnap.from_down.values[jj], 0)
            for k in range(B.NLEV):
                bp = np.nan_to_num(vsnap[f"bp{k}"].values[jj], nan=0.0)
                ap = np.nan_to_num(vsnap[f"ap{k}"].values[jj], nan=1.0)
                bsz = np.where(bp > 0, vsnap[f"bs{k}"].values[jj], 0.0)
                asz = np.where(ap < 1, vsnap[f"as{k}"].values[jj], 0.0)
                Xt[:, C[f"bid_px_{k}"]] = np.where(ok, bp, 0.0)
                Xt[:, C[f"ask_px_{k}"]] = np.where(ok, ap, 1.0)
                Xt[:, C[f"bid_sz_{k}"]] = np.where(ok, bsz, 0.0)
                Xt[:, C[f"ask_sz_{k}"]] = np.where(ok, asz, 0.0)
            Xt[:, C["bid_depth_total"]] = np.where(ok, vsnap.bdep.values[jj], 0)
            Xt[:, C["ask_depth_total"]] = np.where(ok, vsnap.adep.values[jj], 0)
            ex = np.where(ok[:, None], vsnap[EXTRA].values[jj], np.nan)
    if not ok.any():
        Xt[:, [C[f"ask_px_{k}"] for k in range(B.NLEV)]] = 1.0
    Xt[:, C["book_valid"]] = ok
    # ---- trades (data/build.py: maker legs' prices, taker fallback; aggressor from the taker leg)
    n_tx = 0
    legs = None
    sub_trade_row = np.zeros(n, bool)
    lg_stats = {"txs_without_maker_legs": 0}
    if trades_g is not None and len(trades_g):
        tg = trades_g[(trades_g.timestamp >= o) & (trades_g.timestamp <= e)]
        if len(tg):
            use = tg[tg.use]
            r = (use.timestamp.values - o).astype(np.int64)
            sz = use["size"].values.astype(np.float64)
            px = use.px_up.values
            ag = use.aggressor_up.values
            for sgn, vcol, wcol, xcol, fn in ((1, "buy_vol", "buy_vwap", "buy_max_px", np.maximum),
                                               (-1, "sell_vol", "sell_vwap", "sell_min_px", np.minimum)):
                msk = ag == sgn
                vol = np.bincount(r[msk], weights=sz[msk], minlength=n)
                pv = np.bincount(r[msk], weights=(px * sz)[msk], minlength=n)
                ext = np.full(n, -np.inf if sgn == 1 else np.inf)
                fn.at(ext, r[msk], px[msk])
                has = vol > 0
                Xt[:, C[vcol]] = vol
                Xt[:, C[wcol]] = np.where(has, pv / np.where(has, vol, 1), 0)
                Xt[:, C[xcol]] = np.where(np.isfinite(ext) & has, ext, 0)
            txr = pd.DataFrame({"r": r, "tx": use.tx.values}).drop_duplicates("tx")
            Xt[:, C["trade_count"]] = np.bincount(txr.r.values, minlength=n)
            n_tx = len(txr)
            # ---- maker legs -> fills (data/features.py ordering)
            g = tg[~tg.is_taker]
            lg_stats["txs_without_maker_legs"] = int(len(set(tg[tg.is_taker].tx) - set(g.tx)))
            if len(g):
                g = g.assign(row=(g.timestamp.values - o).astype(np.int64))
                first_ord = g.groupby("tx")["ord"].transform("min")
                g = g.assign(first_ord=first_ord, key=np.where(g.aggressor_up > 0, g.px_mil, -g.px_mil))
                g = g.sort_values(["row", "first_ord", "key", "ord"], kind="stable")
                first = g.tx.values != np.roll(g.tx.values, 1)
                first[0] = True
                legs = np.zeros(len(g), FT.FILL_DTYPE)
                legs["row"] = g.row.values
                legs["side"] = g.aggressor_up.values
                legs["first"] = first
                legs["px"] = g.px_mil.values
                legs["price"] = g.px_up.values
                legs["size"] = g["size"].values
                legs["tx_size"] = g.tx_size.values
                sub_trade_row[g.row.values[(g.px_mil.values % 10) != 0]] = True
    assert np.isfinite(Xt).all(), ep["id"]

    # ---- book record + 44 env features (data/features.py, election semantics)
    c = C
    bp, bs = Xt[:, c["bid_px_0"]:c["bid_px_0"] + 5], Xt[:, c["bid_sz_0"]:c["bid_sz_0"] + 5]
    ap, asz = Xt[:, c["ask_px_0"]:c["ask_px_0"] + 5], Xt[:, c["ask_sz_0"]:c["ask_sz_0"] + 5]
    valid = ok
    crossed = Xt[:, c["crossed_raw"]] > 0
    age = Xt[:, c["book_age"]]
    tau = Xt[:, c["seconds_to_close"]]
    bm, am = np.rint(bp * 1000).astype(np.int64), np.rint(ap * 1000).astype(np.int64)
    sub_book = (((bm % 10) != 0) & (bm > 0)).any(1) | (((am % 10) != 0) & (am < 1000)).any(1)
    fine = np.maximum.accumulate(sub_book | sub_trade_row)
    tick = np.where(fine, 0.001, 0.01)
    has_b = valid & (bp[:, 0] > 0)
    has_a = valid & (ap[:, 0] < 1)
    two = has_b & has_a
    mid = 0.5 * (bp[:, 0] + ap[:, 0])
    two_fine_alt = two & ((mid < 0.04) | (mid > 0.96))
    fine_alt = np.maximum.accumulate(fine | two_fine_alt)
    rec = np.zeros(n, FT.BOOK_DTYPE)
    rec["bid_px"], rec["ask_px"], rec["bid_sz"], rec["ask_sz"] = bm, am, bs, asz
    lad = np.concatenate([bm, am, bs, asz], 1)
    same = np.r_[False, (lad[1:] == lad[:-1]).all(1)]
    rec["flags"] = (valid * FT.BOOK_VALID | crossed * FT.BOOK_CROSSED | (age > FT.STALE_S) * FT.BOOK_STALE
                    | (Xt[:, c["book_from_down"]] > 0) * FT.BOOK_FROM_DOWN | same * FT.BOOK_SAME)
    # same_run: consecutive rows from r with SAME set, capped at 255 (vectorised features.py loop)
    run = np.zeros(n, np.int64)
    idx = np.arange(n)
    nxt_false = np.minimum.accumulate(np.where(~same, idx, n)[::-1])[::-1]   # first index >= r with same False
    run = np.minimum(np.where(same, nxt_false - idx, 0), 255)
    rec["same_run"] = run
    rec["tick"] = np.where(fine, 1, 10)
    rec["tick_alt"] = np.where(fine_alt, 1, 10)

    # p_model: mid proxy, causal forward fill within the episode
    last_mid = pd.Series(np.where(two, mid, np.nan)).ffill().values
    p = np.where(two, mid, np.where(np.isfinite(last_mid), last_mid, 0.5))
    p = np.clip(p, 1e-4, 1 - 1e-4)
    mark = np.where(two, mid, p).astype(np.float32).astype(np.float64)
    gap = p - mark
    bvol, svol = Xt[:, c["buy_vol"]], Xt[:, c["sell_vol"]]
    out = np.zeros((n, FT.G))
    out[:, CI["p_logit"]] = FT.logit(p) / 3
    out[:, CI["mark_logit"]] = FT.logit(mark) / 3
    out[:, CI["gap_c"]] = FT.clip(gap * 100 / 5)
    out[:, CI["dgap5_c"]] = FT.clip((gap - FT.lag(gap, 5)) * 100 / 2)
    out[:, CI["tau_frac"]] = tau / L
    out[:, CI["log_tau"]] = FT.clip(np.log1p(tau) / np.log1p(14400))
    spread_t = np.where(two, (ap[:, 0] - bp[:, 0]) / tick, 1.0)
    out[:, CI["spread_ticks"]] = np.where(two, FT.clip(np.log(np.maximum(spread_t, 1.0)) / 2), 0.0)
    out[:, CI["bid_present"]] = has_b
    out[:, CI["ask_present"]] = has_a
    s1 = bs[:, 0] + asz[:, 0]
    out[:, CI["imb_l1"]] = np.where(s1 > 0, (bs[:, 0] - asz[:, 0]) / np.where(s1 > 0, s1, 1), 0)
    out[:, CI["bid_sz0"]] = np.log1p(bs[:, 0]) / 5
    out[:, CI["ask_sz0"]] = np.log1p(asz[:, 0]) / 5
    for k in (5, 30, 120):
        b_, s_ = FT.window_sum(bvol, k), FT.window_sum(svol, k)
        out[:, CI[f"flow{k}"]] = (b_ - s_) / (b_ + s_ + 5)
    out[:, CI["mret5_c"]] = FT.clip((mark - FT.lag(mark, 5)) * 100 / 2)
    out[:, CI["mret30_c"]] = FT.clip((mark - FT.lag(mark, 30)) * 100 / 5)
    out[:, CI["mret120_c"]] = FT.clip((mark - FT.lag(mark, 120)) * 100 / 5)
    for L_, name in zip(FT.CLEN_ALL, ("len_5m", "len_15m", "len_4h")):
        out[:, CI[name]] = float(L == L_)
    out[:, CI["tick_fine"]] = fine
    out[:, CI["book_age"]] = np.log1p(age) / 4
    out[:, CI["crossed_raw"]] = crossed
    out[:, CI["mark_is_fv"]] = ~two
    out[:, CI["p_model"]] = p
    out[:, CI["mark"]] = mark
    out[:, CI["bid_minus_p_c"]] = np.where(has_b, FT.clip((bp[:, 0] - p) * 100 / 5), 0.0)
    out[:, CI["ask_minus_p_c"]] = np.where(has_a, FT.clip((ap[:, 0] - p) * 100 / 5), 0.0)

    # ---- liquidity rewards (program at t; book stats of the snapshot in use)
    cf = cfg_at(cfg, tgrid) if cfg is not None else {k: np.zeros(n) for k in
                                                     ("in_program", "rate", "min_size", "max_spread", "age")}
    amid, qb, qa = ex[:, 0], ex[:, 1], ex[:, 2]
    scorable = valid & np.isfinite(amid)
    amid = np.where(scorable, amid, -1.0)
    qb = np.where(scorable, qb, 0.0)
    qa = np.where(scorable, qa, 0.0)
    inside = (amid >= REWARD_BAND[0]) & (amid <= REWARD_BAND[1])
    comp = np.minimum(qb, qa) + np.where(inside, np.abs(qb - qa) / REWARD_C, 0.0)
    rng = np.random.default_rng([LIQ_SEED, int(ep["condition_id"][2:14], 16), o])
    minute = tgrid // 60
    first = np.r_[True, minute[1:] != minute[:-1]]
    offs = rng.integers(0, 60, first.sum())
    sample = (tgrid % 60) == np.repeat(offs, np.diff(np.r_[np.flatnonzero(first), n]))
    usd_ps = cf["rate"] / 1440.0
    out[:, CI["liq_mid"]] = amid
    out[:, CI["liq_comp"]] = comp
    out[:, CI["liq_pi"]] = np.where(sample & scorable & (cf["in_program"] > 0), usd_ps, 0.0)

    xf = np.zeros((n, X))
    xf[:, XI["rw_in_program"]] = cf["in_program"]
    xf[:, XI["rw_daily_rate"]] = cf["rate"]
    xf[:, XI["rw_min_size"]] = cf["min_size"]
    xf[:, XI["rw_max_spread_c"]] = cf["max_spread"]
    xf[:, XI["rw_cfg_age_s"]] = cf["age"]
    xf[:, XI["rw_adj_mid"]] = amid
    xf[:, XI["rw_comp_q_bid"]] = qb
    xf[:, XI["rw_comp_q_ask"]] = qa
    xf[:, XI["rw_comp_q"]] = comp
    xf[:, XI["rw_two_sided_only"]] = scorable & ~inside
    xf[:, XI["rw_sample"]] = sample
    xf[:, XI["rw_usd_per_sample"]] = usd_ps
    # env quote prices per price head (mm.h mm_targets with the other side off), and their reward score
    tk = rec["tick"].astype(np.int64)
    bb, ba = bm[:, 0], am[:, 0]
    fvm = p.astype(np.float32).astype(np.float64) * 1000.0
    v = cf["max_spread"]
    for a in range(1, 7):
        base = np.where(bb > 0, bb, np.floor(fvm / tk).astype(np.int64) * tk)
        b_ = base + MM_OFFS[a] * tk
        b_ = np.where((ba < 1000) & (b_ >= ba), ba - tk, b_)
        b_ = np.where(b_ < tk, 0, b_)
        base = np.where(ba < 1000, ba, np.ceil(fvm / tk).astype(np.int64) * tk)
        a_ = base - MM_OFFS[a] * tk
        a_ = np.where((bb > 0) & (a_ <= bb), bb + tk, a_)
        a_ = np.where(a_ > 1000 - tk, 0, a_)
        xf[:, XI[f"rw_bid_px_a{a}"]] = b_ / 1000.0
        xf[:, XI[f"rw_ask_px_a{a}"]] = a_ / 1000.0
        xf[:, XI[f"rw_bid_score_a{a}"]] = np.where((b_ > 0) & scorable, S(np.abs(b_ / 1000.0 - amid) * 100, v), 0.0)
        xf[:, XI[f"rw_ask_score_a{a}"]] = np.where((a_ > 0) & scorable, S(np.abs(a_ / 1000.0 - amid) * 100, v), 0.0)
    for k in range(3):
        xf[:, XI[f"rw_size_ok_{k}"]] = (gm["min_size"] * MM_SIZES[k] >= cf["min_size"] - 1e-9) & (cf["in_program"] > 0)
    xf[:, XI["rw_min_size_mult"]] = cf["min_size"] / gm["min_size"]
    xf[:, XI["x_covered"]] = covered
    xf[:, XI["x_log_days_to_end"]] = np.log1p(np.maximum(ep["market_end_ts"] - tgrid, 0) / 86400.0)
    assert np.isfinite(out).all() and np.isfinite(xf).all(), ep["id"]
    assert np.abs(out[:, :FT.N_OBS]).max() <= FT.CLIP + 1e-9, ep["id"]
    st = {"rows_valid": int(valid.sum()), "rows_covered": int(covered.sum()), "rows_two_sided": int(two.sum()),
          "rows_in_program": int((cf["in_program"] > 0).sum()), "rows_scorable": int(scorable.sum()),
          "n_tx": n_tx, "n_legs": 0 if legs is None else int(len(legs)), **lg_stats}
    # per-minute reward series (rewards_ts.parquet): the row of each UTC minute's sample second
    mi = np.flatnonzero(sample)
    mts = pd.DataFrame({
        "minute_ts": (tgrid[mi] // 60) * 60, "sample_ts": tgrid[mi],
        "in_program": cf["in_program"][mi] > 0, "daily_rate_usd": cf["rate"][mi], "min_size": cf["min_size"][mi],
        "max_spread_c": cf["max_spread"][mi], "snapshot_age_s": cf["age"][mi],
        "mid": np.where(two, mid, np.nan)[mi], "adjusted_mid": np.where(scorable, amid, np.nan)[mi],
        "comp_q_bid": qb[mi], "comp_q_ask": qa[mi], "comp_q_min": comp[mi],
        "two_sided_only": (scorable & ~inside)[mi], "usd_per_sample": usd_ps[mi],
        "liq_pi": out[mi, CI["liq_pi"]], "book_valid": valid[mi], "covered": covered[mi], "scorable": scorable[mi]})
    return Xt, out, xf, rec, legs, st, mts


# ----------------------------------------------------------------------------- gaps (D16)
def recorder_up_intervals(events) -> list:
    """Recorder-wide up time: per run from its 'start' to its 'stop' (or its last event if it died), unioned."""
    runs = {}
    for e in events:
        r = runs.setdefault(e["run"], [e["t"], e["t"], None])
        r[0], r[1] = min(r[0], e["t"]), max(r[1], e["t"])
        if e["ev"] == "stop":
            r[2] = e["t"]
    return cov.merge([(a_, c_ if c_ is not None else b_) for a_, b_, c_ in runs.values()])


def gap_stats(g: list, max_gap: float) -> dict:
    big = [(x, y) for x, y in g if y - x > max_gap]
    lg = max(g, key=lambda z: z[1] - z[0]) if g else None
    return {"total_gap_s": round(sum(y - x for x, y in g), 1), "n_gaps": len(g), "n_gaps_gt_max": len(big),
            "total_gap_gt_max_s": round(sum(y - x for x, y in big), 1),
            "longest_gap_s": round(lg[1] - lg[0], 1) if lg else 0.0,
            "longest_gap_start_utc": iso(lg[0]) if lg else ""}


def gap_tables(rec: Path, sel_full: pd.DataFrame, max_gap: float, day_max: float) -> dict:
    """Per UTC day: recorder-wide gaps (whole recorder down) and per-market gaps (market not live), measured from
    the first recorder start (market: its first live) to the last coverage event; 'eligible' = total <= day_max."""
    events = list(cov.read_events(rec))
    up = recorder_up_intervals(events)
    iv = cov.live_intervals(events)
    if not up:
        return {"days": pd.DataFrame(), "recorder_gaps": [], "up": up}
    t_lo, t_hi = up[0][0], up[-1][1]
    rows = []
    days = pd.date_range(day_of(t_lo), day_of(t_hi), freq="D").strftime("%Y-%m-%d")
    rec_gaps = cov.gaps(up, t_lo, t_hi)
    for d in days:
        d0 = utc_ts(d)
        lo, hi = max(d0, t_lo), min(d0 + 86400, t_hi)
        st = gap_stats(cov.gaps(up, lo, hi), max_gap)
        rows.append({"scope": "recorder", "day": d, "slug": "", "condition_id": "", "span_start_utc": iso(lo),
                     "span_end_utc": iso(hi), "span_s": round(hi - lo, 1), **st,
                     "eligible": st["total_gap_s"] <= day_max, "day_complete": hi >= d0 + 86400})
        for m in sel_full.itertuples():
            v = iv.get(m.slug)
            if not v:
                continue
            mlo, mhi = max(d0, v[0][0]), min(d0 + 86400, v[-1][1], m.end_ts)
            if mhi <= mlo:
                continue
            st = gap_stats(cov.gaps(v, mlo, mhi), max_gap)
            rows.append({"scope": "market", "day": d, "slug": m.slug, "condition_id": m.condition_id,
                         "span_start_utc": iso(mlo), "span_end_utc": iso(mhi), "span_s": round(mhi - mlo, 1), **st,
                         "eligible": st["total_gap_s"] <= day_max, "day_complete": mhi >= d0 + 86400})
    return {"days": pd.DataFrame(rows), "recorder_gaps": rec_gaps, "up": up}


def write_gaps(dirp: Path, gp: dict, a) -> None:
    dirp.mkdir(parents=True, exist_ok=True)
    df = gp["days"]
    tmp = dirp / ".gaps_per_day.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, dirp / "gaps_per_day.csv")
    L = [f"# Recording gaps (D16), track E elections",
         "",
         f"Generated by `build_elections.py{' --gaps-only' if a.gaps_only else ''}` at "
         f"{pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%SZ')} from `rec/coverage/` (read only).",
         f"A gap longer than {a.max_gap:g} s splits an episode. A day is eligible for train/val/test if its total gap "
         f"time is at most {a.day_max_gap_s:g} s (ineligible days: {a.ineligible_day}).",
         "",
         "**Recorder-wide gaps** are the times when no recorder run was up: each run counts from its `start` to its "
         "`stop`, or to its last event if it died. **Market gaps** are the times a market was not live; they "
         "include recorder-wide gaps and per-market reconnects. A market is measured from its first `live` event; "
         "time before that is not recorded, and not counted as a gap. The current day is measured up to the "
         "last coverage event, so `day_complete = False`.",
         "",
         "## Per day, recorder-wide", "",
         "| day | span | total gap s | gaps | gaps > max | longest s | longest starts | eligible | complete |",
         "|---|---|---|---|---|---|---|---|---|"]
    if len(df):
        for r in df[df.scope == "recorder"].itertuples():
            L.append(f"| {r.day} | {r.span_start_utc[11:19]}-{r.span_end_utc[11:19]} | {r.total_gap_s:g} | {r.n_gaps} | "
                     f"{r.n_gaps_gt_max} | {r.longest_gap_s:g} | {r.longest_gap_start_utc} | {r.eligible} | {r.day_complete} |")
        L += ["", "## Per day, across the selected markets that were live", "",
              "| day | markets | median total gap s | max total gap s | markets with a gap > max | "
              "longest market gap s | ineligible market-days |", "|---|---|---|---|---|---|---|"]
        mm_ = df[df.scope == "market"]
        for d, g in mm_.groupby("day"):
            L.append(f"| {d} | {len(g)} | {g.total_gap_s.median():g} | {g.total_gap_s.max():g} | "
                     f"{int((g.n_gaps_gt_max > 0).sum())} | {g.longest_gap_s.max():g} | {int((~g.eligible).sum())} |")
    L += ["", f"## Recorder-wide gaps (all, by time)", "", "| start UTC | end UTC | seconds |",
          "|---|---|---|"]
    for x, y in gp["recorder_gaps"]:
        L.append(f"| {iso(x)} | {iso(y)} | {y - x:.1f} |")
    L += ["", "Per-market rows are in `gaps_per_day.csv` (`scope = market`)."]
    tmp = dirp / ".GAPS.md.tmp"
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, dirp / "GAPS.md")
    print(f"wrote {dirp / 'GAPS.md'} and gaps_per_day.csv ({len(df)} rows)", flush=True)


# ----------------------------------------------------------------------------- side tables
FLAG_NAMES = {"has_trades": B.FLAG_HAS_TRADES, "has_underlying": B.FLAG_HAS_UNDERLYING, "election": FLAG_ELECTION,
              "unresolved": FLAG_UNRESOLVED, "partial": FLAG_PARTIAL, "in_program": FLAG_IN_PROGRAM,
              "trades_incomplete": FLAG_TRADES_INCOMPLETE}


def write_side_tables(out: Path, eps, ep_meta, minute_parts, mk):
    """rewards_ts.parquet (per episode and UTC minute) and markets_meta.parquet (per episode)."""
    if minute_parts:
        ts = pd.concat(minute_parts, ignore_index=True).sort_values(["market_idx", "minute_ts"], kind="stable")
    else:
        ts = pd.DataFrame()
    tmp = out / ".rewards_ts.parquet.tmp"
    ts.to_parquet(tmp, index=False)
    os.replace(tmp, out / "rewards_ts.parquet")
    q = dict(zip(mk.slug, mk.question))
    em = {d["market_idx"]: d for d in ep_meta}
    rows = []
    for x in eps:
        d = em[x["i"]]
        r = {"market_idx": np.int32(x["i"]), "condition_id": x["condition_id"], "slug": x["slug"],
             "question": q.get(x["slug"], ""), "episode_id": x["id"], "day": x["day"],
             "window_start_ts": np.int64(x["window_start_ts"]), "window_end_ts": np.int64(x["window_end_ts"]),
             "window_start_utc": pd.Timestamp(x["window_start_ts"], unit="s", tz="UTC"),
             "window_end_utc": pd.Timestamp(x["window_end_ts"], unit="s", tz="UTC"),
             "market_end_ts": np.int64(x["market_end_ts"]),
             "split": np.int32(x["split"]), "split_name": {0: "train", 1: "val", 2: "test", 3: "purged"}[x["split"]],
             "row_offset": np.int64(x["row_offset"]), "n_rows": np.int32(x["n_rows"]),
             "episode_start_ts": np.int64(x["open_ts"]), "episode_end_ts": np.int64(x["end_ts"]),
             "episode_start_utc": pd.Timestamp(x["open_ts"], unit="s", tz="UTC"),
             "episode_end_utc": pd.Timestamp(x["end_ts"], unit="s", tz="UTC"),
             "start_reason": x["start_reason"], "end_reason": x["end_reason"], "split_reason": x["split_reason"],
             "prev_segment_id": x.get("prev_id", ""), "next_segment_id": x.get("next_id", ""),
             "gap_before_s": float(x.get("gap_before_s", 0.0)),
             "split_assigned": x["split_assigned"], "day_eligible": bool(x["day_eligible"]),
             "rewards_in_program_at_open": d["rw_in"], "rewards_daily_rate_at_open": d["rw_rate"],
             "rewards_min_size": d["rw_min"], "rewards_max_spread_c": d["rw_spread"],
             "tick_size": d["gm"]["tick"], "min_order_size": d["gm"]["min_size"], "taker_fee_rate": d["fee"],
             "fee_exponent": d["gm"]["fee_exponent"], "maker_rebate_rate": d["gm"]["rebate_rate"],
             "flags": np.int32(d["flags"]),
             **{f"flag_{k}": bool(d["flags"] & v) for k, v in FLAG_NAMES.items()},
             "cov_total_gap_s": x["cov_total_gap"], "cov_max_gap_s": x["cov_max_gap"],
             "rows_valid": d["rows_valid"], "rows_covered": d["rows_covered"], "rows_two_sided": d["rows_two_sided"],
             "rows_in_program": d["rows_in_program"], "rows_scorable": d["rows_scorable"],
             "n_taker_txs": d["n_tx"], "n_fill_legs": d["n_legs"]}
        rows.append(r)
    tmp = out / ".markets_meta.parquet.tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, out / "markets_meta.parquet")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=str(HERE / "rec"), help="recorder out dir (read only)")
    ap.add_argument("--sel", default=str(HERE / "sel" / "selection.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--window-s", type=int, default=14400, help="window length; must divide 86400. Windows are "
                    "aligned to UTC midnight (default 4 h: 00/04/08/12/16/20 UTC, D16)")
    ap.add_argument("--min-episode-s", type=int, default=600, help="drop (sub-)episodes shorter than this")
    ap.add_argument("--day-max-gap-s", type=float, default=1800.0, help="a day whose recorder-wide (or the market's) "
                    "total gap exceeds this is not eligible for train/val/test (D16)")
    ap.add_argument("--ineligible-day", choices=["purged", "train-only"], default="train-only",
                    help="what an ineligible day becomes: train-only (default, D16: keep train, purge val/test) or purged (3)")
    ap.add_argument("--gaps-only", action="store_true", help="only write GAPS.md and gaps_per_day.csv "
                    "(to --gaps-dir, default analysis/v3_elections/) from rec/coverage, build nothing")
    ap.add_argument("--gaps-dir", default=str(HERE))
    ap.add_argument("--split-days", default=None, help="JSON file or inline JSON {day: train|val|test|purged|exclude}; "
                    "days not listed -> purged (3). No default ever assigns test.")
    ap.add_argument("--cutoff", default="2026-11-03T00:00:00Z", help="windows must end at or before this "
                    "(election night out of scope, D11)")
    ap.add_argument("--max-gap", type=float, default=15.0, help="coverage gaps longer than this split an episode; "
                    "shorter ones are forward-filled (D16: 15 s)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="keep episodes whose fills coverage is incomplete (smoke tests); flag TRADES_INCOMPLETE")
    ap.add_argument("--markets", default=None, help="comma list or file of slugs / condition ids to build")
    ap.add_argument("--top-n", type=int, default=20, help="only the N markets with the highest reward rate "
                    "(selection.json rate_per_day); 0 = every market")
    ap.add_argument("--all", action="store_true", help="every selected market (= --top-n 0)")
    ap.add_argument("--min-rate", type=float, default=0.0, help="only markets with rate_per_day >= this")
    ap.add_argument("--trades-source", choices=["onchain", "data-api", "file", "none"], default=None,
                    help="fills: onchain = extract.py output dir (--trades-path), the census source; data-api = "
                         "/trades by condition id (fallback); file = a data/raw-schema trades.parquet "
                         "(--trades-path); none = no fills (all windows flagged TRADES_INCOMPLETE)")
    ap.add_argument("--trades-path", default=None)
    ap.add_argument("--refresh-trades", action="store_true", help="re-fetch data-api even if cached")
    ap.add_argument("--workers", type=int, default=3, help="concurrent HTTP requests (I/O only)")
    ap.add_argument("--no-gamma", action="store_true", help="do not fetch tick / min size / fees from gamma")
    ap.add_argument("--feat-layout", choices=["split", "wide"], default="split",
                    help="split: feat.bin [N, 44] (current env) + feat_x.bin [N, X]; wide: feat.bin [N, 44 + X]")
    ap.add_argument("--max-gb", type=float, default=20.0, help="refuse builds larger than this")
    ap.add_argument("--emit-tokens", default=None, help="write the token map (token_id, slug, outcome_idx, "
                    "condition_id) for onchain/extract.py --tokens, then exit")
    ap.add_argument("--dry-run", action="store_true", help="select windows and print the plan only")
    a = ap.parse_args(argv)
    if a.all:
        a.top_n = 0
    t0 = time.time()
    rec, out = Path(a.rec).resolve(), Path(a.out).resolve()
    for bad in (DATA.resolve(), (rec).resolve()):
        if out == bad or bad in out.parents:
            raise SystemExit(f"--out {out} is inside {bad}: refused")
    if 86400 % a.window_s:
        raise SystemExit("--window-s must divide 86400")
    L = a.window_s
    mk, sel_full, sel_taken = load_selection(Path(a.sel), a)
    sel_sha = hashlib.sha1(Path(a.sel).read_bytes()).hexdigest()[:12]
    if a.emit_tokens:
        tm = pd.concat([pd.DataFrame({"token_id": mk.tok0, "slug": mk.slug, "outcome_idx": 0, "condition_id": mk.condition_id}),
                        pd.DataFrame({"token_id": mk.tok1, "slug": mk.slug, "outcome_idx": 1, "condition_id": mk.condition_id})])
        tm.to_parquet(a.emit_tokens, index=False)
        print(f"wrote {len(tm)} tokens to {a.emit_tokens}")
        return 0
    if a.gaps_only:
        gp = gap_tables(rec, sel_full, a.max_gap, a.day_max_gap_s)
        write_gaps(Path(a.gaps_dir), gp, a)
        return 0
    if a.trades_source is None:
        raise SystemExit("--trades-source is required (onchain / data-api / file / none)")
    split_days = parse_split_days(a.split_days)
    cutoff = utc_ts(a.cutoff)
    cfgs, reward_files = load_reward_cfg(rec, sel_full, sel_taken)
    iv, seg = coverage_segments(rec, a.max_gap)
    days = recorded_days(rec)
    print(f"selection {len(mk)} markets, recorded days {days}, reward snapshots {len(reward_files)}, "
          f"coverage for {len(iv)} markets", flush=True)

    # ---- day eligibility (D16) and gap report
    gp = gap_tables(rec, sel_full, a.max_gap, a.day_max_gap_s)
    rec_elig = {r.day: bool(r.eligible) for r in gp["days"][gp["days"].scope == "recorder"].itertuples()}
    mkt_elig = {(r.slug, r.day): bool(r.eligible) for r in gp["days"][gp["days"].scope == "market"].itertuples()}

    # ---- candidate episodes: windows cut at coverage gaps > --max-gap, the market end and the cutoff
    eps, reasons = [], {}

    def why(r, k=1):
        reasons[r] = reasons.get(r, 0) + k

    last_cov = {s_: float(v[-1][1]) for s_, v in iv.items() if v}
    first_cov = {s_: float(v[0][0]) for s_, v in iv.items() if v}
    warned = set()
    for m in mk.itertuples():
        sg = seg.get(m.slug)
        for day in days:
            d0 = utc_ts(day)
            sname = split_days.get(day, "purged")
            if sname == "exclude":
                why("day excluded", 86400 // L)
                continue
            for w in range(86400 // L):
                o, e = d0 + w * L, d0 + (w + 1) * L
                hi = min(e, m.end_ts, cutoff)
                if hi - o < a.min_episode_s:
                    why("window after market end / cutoff")
                    continue
                if sg is None or not ((sg[0] < hi) & (sg[1] > o)).any():
                    why("not recorded")
                    continue
                for a_, b_ in zip(sg[0], sg[1]):
                    if b_ <= o or a_ >= hi:
                        continue
                    s0, s1 = max(o, int(np.ceil(a_))), min(hi, int(np.floor(b_)))
                    if s1 - s0 < a.min_episode_s:
                        why("sub-episode shorter than --min-episode-s")
                        continue
                    start_reason = ("window_boundary" if s0 == o else
                                    "coverage_start" if a_ <= first_cov.get(m.slug, a_) else "gap")
                    if s1 == hi:
                        end_reason = "window_boundary" if hi == e else ("market_end" if hi == m.end_ts else "cutoff")
                    else:
                        end_reason = "coverage_end" if b_ >= last_cov.get(m.slug, b_) else "gap"
                    split_name = sname
                    elig = rec_elig.get(day, False) and mkt_elig.get((m.slug, day), False)
                    if not elig and split_name in ("train", "val", "test"):
                        if a.ineligible_day == "purged" or split_name != "train":
                            if (day, m.slug) not in warned:
                                print(f"WARNING: {day} {m.slug[:60]}: gaps exceed --day-max-gap-s "
                                      f"{a.day_max_gap_s:g} s: {split_name} -> purged", flush=True)
                                warned.add((day, m.slug))
                            split_name = "purged"
                    gaps_in = cov.gaps(iv.get(m.slug, []), s0, s1)
                    eps.append({"slug": m.slug, "condition_id": m.condition_id, "day": day, "open_ts": s0, "end_ts": s1,
                                "window_start_ts": o, "window_end_ts": e, "start_reason": start_reason,
                                "end_reason": end_reason, "split_reason": end_reason,
                                "split": SPLIT_IDS[split_name], "split_assigned": sname, "day_eligible": elig,
                                "partial": False, "cov_total_gap": round(sum(y - x for x, y in gaps_in), 1),
                                "cov_max_gap": round(max((y - x for x, y in gaps_in), default=0.0), 1),
                                "market_end_ts": m.end_ts, "id": f"{m.slug}@{iso(s0)}"})
    # link consecutive segments of one market separated by a gap (D16: inventory carried across the gap)
    by_m = {}
    for x in eps:
        by_m.setdefault(x["slug"], []).append(x)
    for lst in by_m.values():
        lst.sort(key=lambda x: x["open_ts"])
        for p_, q_ in zip(lst[:-1], lst[1:]):
            if q_["start_reason"] == "gap" and p_["end_reason"] == "gap":
                q_["prev_id"], p_["next_id"] = p_["id"], q_["id"]
                q_["gap_before_s"] = q_["open_ts"] - p_["end_ts"]
    print(f"episodes: {len(eps)} kept; skipped {reasons}; end reasons "
          f"{pd.Series([x['end_reason'] for x in eps]).value_counts().to_dict() if eps else {}}", flush=True)
    if not eps:
        print("nothing to build")
        return 1
    used = mk[mk.slug.isin({x["slug"] for x in eps})].reset_index(drop=True)

    # ---- fills
    need_to = {}
    for x in eps:
        need_to[x["slug"]] = max(need_to.get(x["slug"], 0), x["end_ts"])
    if a.trades_source == "data-api":
        trades, span = trades_data_api(used, out / "cache" / "data_api", need_to, a.workers, a.refresh_trades)
    elif a.trades_source == "onchain":
        trades, span = trades_onchain(used, Path(a.trades_path))
    elif a.trades_source == "file":
        trades = pd.read_parquet(a.trades_path)
        trades["slug"] = trades.slug.astype(str)
        trades = trades[trades.slug.isin(set(used.slug))][TRADE_COLS].reset_index(drop=True)
        span = {s: (-np.inf, np.inf) for s in used.slug}
    else:
        trades, span = pd.DataFrame(columns=TRADE_COLS), {}
    kept = []
    for x in eps:
        x["trades_complete"] = a.trades_source != "none" and trades_complete(span.get(x["slug"]), x["open_ts"], x["end_ts"])
        if x["trades_complete"] or a.allow_partial or a.trades_source == "none":
            kept.append(x)
        else:
            why("fills incomplete over the window")
    eps = kept
    print(f"fills: {len(trades)} legs from {a.trades_source}; windows after the fills check: {len(eps)}", flush=True)
    if not eps:
        print("nothing to build")
        return 1
    if a.dry_run:
        print(pd.DataFrame(eps)[["id", "open_ts", "end_ts", "start_reason", "end_reason", "split"]].to_string(max_rows=60))
        return 0
    trades = trades.copy()
    trades["px_up"] = np.where(trades.outcome_idx == 0, trades.price.astype(np.float64), 1.0 - trades.price.astype(np.float64))
    has_maker = trades[~trades.is_taker].groupby("tx").size()
    trades["use"] = ~trades.is_taker | ~trades.tx.isin(has_maker.index)
    tx_size = trades[trades.is_taker].drop_duplicates("tx").set_index("tx")["size"].astype(np.float64)
    maker_sum = trades[~trades.is_taker].groupby("tx")["size"].sum().astype(np.float64)
    trades["tx_size"] = trades.tx.map(tx_size).fillna(trades.tx.map(maker_sum)).astype(np.float64)
    trades["px_mil"] = np.rint(trades.px_up * 1000).astype(np.int64)
    trades["ord"] = np.arange(len(trades))
    dev = (maker_sum - tx_size.reindex(maker_sum.index)).abs()
    trade_sanity = {"legs": int(len(trades)), "taker_txs": int(len(tx_size)), "maker_legs": int((~trades.is_taker).sum()),
                    "txs_without_maker_legs": int(len(set(tx_size.index) - set(maker_sum.index))),
                    "max_abs_maker_sum_minus_taker_size": float(dev.max()) if len(dev) else 0.0,
                    "txs_abs_dev_gt_1e-3": int((dev > 1e-3).sum())}
    tr_by = {s: g for s, g in trades.groupby("slug", sort=False)}
    gmeta = gamma_meta(used, out / "cache", a.no_gamma)

    # ---- order, offsets, allocation (data/build.py: sorted by split, open, id; contiguous splits)
    eps.sort(key=lambda x: (x["split"], x["open_ts"], x["id"]))
    for x in eps:
        x["n_rows"] = x["end_ts"] - x["open_ts"] + 1
    N = int(sum(x["n_rows"] for x in eps))
    G_file = FT.G + (X if a.feat_layout == "wide" else 0)
    bpr = B.F * 4 + G_file * 4 + (0 if a.feat_layout == "wide" else X * 4) + FT.BOOK_DTYPE.itemsize + 4
    gb = N * bpr / 1e9
    print(f"episodes {len(eps)}, {N} rows, ~{gb:.2f} GB ({bpr} B/row)", flush=True)
    if gb > a.max_gb:
        raise SystemExit(f"build would be {gb:.1f} GB > --max-gb {a.max_gb}: use --top-n / --markets / fewer days")
    if N >= 2**31 - 1:
        raise SystemExit("too many rows for int32 fill rows")
    out.mkdir(parents=True, exist_ok=True)
    gi = out / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n")

    def mm(name, dtype, shape):
        return np.memmap(out / f".{name}.tmp", dtype=dtype, mode="w+", shape=shape)

    ticks = mm("ticks.bin", np.float32, (N, B.F))
    feat = mm("feat.bin", np.float32, (N, G_file))
    featx = None if a.feat_layout == "wide" else mm("feat_x.bin", np.float32, (N, X))
    book = mm("book.bin", FT.BOOK_DTYPE, (N,))
    recs = np.zeros(len(eps), dtype=B.MARKET_DTYPE)
    off = 0
    for i, x in enumerate(eps):
        x["i"], x["row_offset"] = i, off
        off += x["n_rows"]
    by_day = {}
    for x in eps:
        by_day.setdefault(x["day"], []).append(x)

    tokmap = {}
    for m in sel_full.itertuples():
        tokmap[m.tok0] = (m.slug, True, m.condition_id)
        tokmap[m.tok1] = (m.slug, False, m.condition_id)
    comp = Compactor(out / "cache" / "compact", tokmap, cfgs, reward_files, rec, sel_sha)
    carry = pd.DataFrame()
    fill_parts, per_ep, mirror, minute_parts, ep_meta = [], [], {}, [], []
    last_day = max(by_day)
    for day in days:
        if day > last_day:
            break
        D = read_day(comp, rec, day)
        D = dedup(pd.concat([carry, D], ignore_index=True)) if len(carry) else dedup(D)
        if day in by_day:
            nxt = day_of(utc_ts(day) + 86400)
            if nxt in days:     # rows at e = next midnight live in the next day's first hour
                E = read_day(comp, rec, nxt, hours={"00"})
                if len(E):
                    E = E[E.ts <= max(x["end_ts"] for x in by_day[day])]
                    DD = dedup(pd.concat([D, E], ignore_index=True))
                else:
                    DD = D
            else:
                DD = D
            today = DD[DD.ts >= utc_ts(day)]
            try:
                mirror[day] = B.mirror_stats(today) if len(today) else {}
            except Exception as ex_:  # noqa: BLE001
                mirror[day] = {"error": str(ex_)}
            groups = {s: g for s, g in DD.groupby("slug", sort=False)}
            for x in by_day[day]:
                gm = gmeta[x["condition_id"]]
                Xt, f44, xf, brec, legs, st, mts = episode_rows(x, groups.get(x["slug"]), seg.get(x["slug"]),
                                                           tr_by.get(x["slug"]), cfgs.get(x["condition_id"]), gm,
                                                           x["n_rows"] - 1)
                r0, n_rows = x["row_offset"], x["n_rows"]
                ticks[r0:r0 + n_rows] = Xt.astype(np.float32)
                feat[r0:r0 + n_rows, :FT.G] = f44.astype(np.float32)
                if featx is None:
                    feat[r0:r0 + n_rows, FT.G:] = xf.astype(np.float32)
                else:
                    featx[r0:r0 + n_rows] = xf.astype(np.float32)
                book[r0:r0 + n_rows] = brec
                if legs is not None:
                    legs["row"] += r0
                    fill_parts.append(legs)
                cf0 = cfg_at(cfgs[x["condition_id"]], np.array([x["open_ts"]])) if x["condition_id"] in cfgs else None
                flags = ((B.FLAG_HAS_TRADES if st["n_tx"] > 0 else 0) | FLAG_ELECTION | FLAG_UNRESOLVED
                         | (FLAG_PARTIAL if st["rows_covered"] < n_rows else 0) | (FLAG_IN_PROGRAM if st["rows_in_program"] else 0)
                         | (0 if x["trades_complete"] else FLAG_TRADES_INCOMPLETE))
                fee = gm["fee_rate"] if gm["fees_enabled"] else 0.0
                recs[x["i"]] = (r0, x["open_ts"], x["end_ts"], n_rows, n_rows - 1, NO_COIN, -1, x["split"], flags,
                                B.BASE_TICK, fee, gm["fee_exponent"], gm["min_size"],
                                float(cf0["min_size"][0]) if cf0 else 0.0, float(cf0["max_spread"][0]) if cf0 else 0.0,
                                0.0, gm["rebate_rate"], x["i"], gm["tick"])
                per_ep.append({"id": x["id"], **st})
                mts.insert(0, "slug", x["slug"])
                mts.insert(0, "condition_id", x["condition_id"])
                mts.insert(0, "market_idx", np.int32(x["i"]))
                minute_parts.append(mts)
                ep_meta.append({"market_idx": x["i"], "flags": flags, "fee": fee, "gm": gm,
                                "rw_min": float(cf0["min_size"][0]) if cf0 else 0.0,
                                "rw_spread": float(cf0["max_spread"][0]) if cf0 else 0.0,
                                "rw_rate": float(cf0["rate"][0]) if cf0 else 0.0,
                                "rw_in": bool(cf0["in_program"][0] > 0) if cf0 else False, **st})
            print(f"  {day}: {len(by_day[day])} episodes  {time.time() - t0:.0f}s  (compact cache hits {comp.hits}, "
                  f"misses {comp.misses})", flush=True)
        carry = carry_rows(D)
        del D

    fills = np.concatenate(fill_parts) if fill_parts else np.zeros(0, FT.FILL_DTYPE)
    if len(fills):
        fills = fills[np.argsort(fills["row"], kind="stable")]
    dummy_fill = False
    if not len(fills):
        # mm.h refuses an empty fills.bin; one zero-size leg on the very last row changes nothing
        fills = np.zeros(1, FT.FILL_DTYPE)
        fills["row"], fills["side"], fills["first"], fills["px"] = N - 1, 1, 1, 500
        fills["price"], fills["size"], fills["tx_size"] = 0.5, 0.0, 0.0
        dummy_fill = True
    index = np.searchsorted(fills["row"], np.arange(N + 1), side="left").astype(np.int32)
    vol = np.bincount(fills["row"], weights=fills["size"].astype(np.float64), minlength=N)
    tv = np.asarray(ticks[:, B.C["buy_vol"]], np.float64) + np.asarray(ticks[:, B.C["sell_vol"]], np.float64)
    trade_sanity["max_abs_row_volume_fills_vs_ticks"] = float(np.abs(vol - tv).max()) if N else 0.0
    for x_ in (ticks, feat, book) + ((featx,) if featx is not None else ()):
        x_.flush()
    assert np.isfinite(np.asarray(ticks)).all()

    # ---- test-split hash, splits, meta
    te = recs[recs["split"] == 2]
    if len(te):
        lo, hi = int(te["row_offset"].min()), int((te["row_offset"] + te["n_rows"]).max())
        test_sha = hashlib.sha256(np.asarray(ticks[lo:hi]).tobytes()).hexdigest()
    else:
        lo = hi = 0
        test_sha = None
    del ticks, feat, book, featx
    for name in ("ticks.bin", "feat.bin", "book.bin") + (("feat_x.bin",) if a.feat_layout == "split" else ()):
        os.replace(out / f".{name}.tmp", out / name)
    if a.feat_layout == "wide" and (out / "feat_x.bin").exists():
        (out / "feat_x.bin").unlink()
    recs.tofile(out / ".markets.bin.tmp")
    os.replace(out / ".markets.bin.tmp", out / "markets.bin")
    FT.write_atomic(out / "fills.bin", fills)
    FT.write_atomic(out / "fill_index.bin", index)

    ids = [x["id"] for x in eps]
    FT.write_atomic(out / "splits.json", json.dumps({i: int(x["split"]) for i, x in zip(ids, eps)}, indent=0, sort_keys=True))
    layout = [{"name": nm, "offset": B.MARKET_DTYPE.fields[nm][1], "type": B.C_TYPES[B.MARKET_DTYPE.fields[nm][0].str],
               "size": B.MARKET_DTYPE.fields[nm][0].itemsize} for nm in B.MARKET_DTYPE.names]
    names = {v: k for k, v in SPLIT_IDS.items()}
    split_counts = {names[k]: {"markets": int((recs["split"] == k).sum()), "rows": int(recs["n_rows"][recs["split"] == k].sum())}
                    for k in names}
    day_split = {d: split_days.get(d, "purged (unassigned)") for d in sorted({x["day"] for x in eps})}
    meta = {
        "build_version": BUILD_VERSION, "builder": "analysis/v3_elections/build_elections.py",
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(), "args": vars(a),
        "F": B.F, "N": N, "n_markets": int(len(recs)), "dtype": "float32", "order": "row-major [N, F]",
        "columns": B.COLS, "column_index": B.C,
        "market_record": {"itemsize": B.MARKET_DTYPE.itemsize, "fields": layout,
                          "flags": {"HAS_TRADES": B.FLAG_HAS_TRADES, "HAS_UNDERLYING": B.FLAG_HAS_UNDERLYING,
                                    "UL_PERP_PROXY": B.FLAG_UL_PERP_PROXY, "HAS_STRIKE": B.FLAG_HAS_STRIKE,
                                    "ELECTION": FLAG_ELECTION, "UNRESOLVED": FLAG_UNRESOLVED, "PARTIAL": FLAG_PARTIAL,
                                    "IN_PROGRAM": FLAG_IN_PROGRAM, "TRADES_INCOMPLETE": FLAG_TRADES_INCOMPLETE},
                          "election_fields": {
                              "open_ts/end_ts": "window [open, open + window_s], aligned to UTC midnight",
                              "contract_len_s": "window_s (the env's length one-hot / len_mask only know 300/900/14400)",
                              "coin_id": "-1 (no underlying)", "winner_up": "-1 (unresolved at build time; never a label)",
                              "tick_size": "0.01 (data/build.py BASE_TICK); book.bin tick follows the data rule",
                              "taker_fee_rate/fee_exponent/maker_rebate_rate": "gamma feeSchedule at build time (0 if fees off)",
                              "min_order_size": "gamma orderMinSize", "tick_size_final": "gamma orderPriceMinTickSize at build time",
                              "rewards_min_size/rewards_max_spread": "reward program in force at the window open (cents); "
                                                                     "per-row values in feat_x",
                              "log_strike_vs_ul_open": "0", "market_idx": "index of the episode (meta slugs / episodes)"}},
        "coin_ids": {}, "contract_len_s": {"window": L, "note": "per episode: end - start (gap / market-end / cutoff cuts)"},
        "splits": {"ids": names, "rule": "by UTC day of the window open from --split-days; days not listed -> purged "
                   "(3); 'exclude' days are not built. No default assigns test.", "day_split": day_split,
                   "counts": split_counts},
        "market_order": "sorted by (split, open_ts, id); each split's rows are contiguous",
        "slugs": ids,
        "episodes": [{k: x[k] for k in ("id", "slug", "condition_id", "day", "open_ts", "end_ts", "start_reason",
                                         "end_reason", "split", "split_assigned", "day_eligible",
                                         "cov_total_gap", "cov_max_gap", "trades_complete")} for x in eps],
        "gaps": {"max_gap_s": a.max_gap, "day_max_gap_s": a.day_max_gap_s, "ineligible_day": a.ineligible_day,
                 "files": ["GAPS.md", "gaps_per_day.csv"]},
        "test_rows": {"row_lo": lo, "row_hi": hi, "sha256": test_sha},
        "normalisation_train_only": {},
        "sentinels": {"empty_bid_px": 0.0, "empty_ask_px": 1.0, "empty_size": 0.0, "no_trade_vwap_maxmin": 0.0,
                      "no_book": "book_valid=0, book_age=0, empty ladders (also outside recorder coverage)"},
        "conventions": {
            "grid": "row t for every unix second t in [open, end] inclusive (window_s + 1 rows); seconds_to_close = end - t "
                    "(time to the WINDOW end, not to resolution)",
            "frame": "Yes token (outcome 0 = selection.json tokens[0]) in the Up slot; No snapshots mapped as "
                     "Yes bid = 1 - No ask (data/build.py to_up_frame)",
            "book": "data/build.py merged_snapshots on the recorder's books (one row per token and captured second; "
                    "last per second kept), latest uncrossed snapshot <= t, and only within the recorder coverage "
                    f"segment holding t (coverage log, gaps <= {a.max_gap:g} s bridged); outside coverage book_valid=0",
            "trades": "data/build.py rule: block time == t, maker legs' prices (taker fallback), aggressor from the taker leg",
            "underlying": "none: ul_logp_rel_open, ul_logret_1s, ul_logp_vs_strike are 0, HAS_UNDERLYING and "
                          "HAS_STRIKE are 0, coin_id = -1",
            "leakage": "no outcome anywhere (markets unresolved; winner_up = -1)"},
        "sources": {"rec": str(rec), "selection": str(Path(a.sel).resolve()), "selection_sha1": sel_sha,
                    "reward_snapshots": reward_files, "trades_source": a.trades_source, "trades_path": a.trades_path},
        "mirror_stats": mirror, "skipped_windows": reasons,
        "row_counts": {"ticks": N, "markets": int(len(recs))},
    }
    FT.write_atomic(out / "meta.json", json.dumps(meta, indent=1, default=str))

    fdocs = dict(FT.OBS_COLS + FT.AUX_COLS)
    fdocs.update(FEAT_ELECTION_DOCS)
    fmeta = {
        "features_version": FT.FEATURES_VERSION, "builder": "analysis/v3_elections/build_elections.py",
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "N": int(N), "G": G_file, "n_obs_features": FT.N_OBS, "dtype": "float32", "order": "row-major [N, G]",
        "columns": FT.COLS + (X_COLS if a.feat_layout == "wide" else []),
        "column_index": {**FT.CI, **({c: FT.G + i for c, i in XI.items()} if a.feat_layout == "wide" else {})},
        "column_docs": fdocs,
        "obs_block": f"columns 0..{FT.N_OBS - 1} are observed (data/features.py layout, unchanged indices)",
        "extra": {"file": "feat.bin (columns 44..)" if a.feat_layout == "wide" else "feat_x.bin", "X": X,
                  "dtype": "float32", "order": "row-major [N, X]", "columns": X_COLS, "column_index": XI,
                  "column_docs": dict(X_DOCS),
                  "eligibility": "a quote of price head a and size head k (size_mult 1) earns at a sample iff "
                                 "rw_{bid|ask}_score_a{a} > 0 and rw_size_ok_{k} = 1 and it is old enough; with "
                                 "rw_two_sided_only = 1 only the min of both sides counts, else single-sided "
                                 "Q = max(min(Q1, Q2), max(Q1, Q2) / 3). Prices assume the other side is off "
                                 "(mm_targets also keeps bid < ask)."},
        "causality": "row t uses book snapshots with captured second <= t, fills with block time <= t (tick flag) and "
                     "the reward snapshot taken <= t. No outcome information.",
        "pricer": {"source": "none (elections): p_model = mid proxy"},
        "liquidity_rewards": {"source": "recorder reward snapshots (rewards_logger.py) + selection.json; scorer of "
                                        "analysis/rewards/reward_terms.py (S, adjusted_mid, competitor_q 'hi')",
                              "v_cents": "per market and hour (rw_max_spread_c)", "min_size": "per market and hour",
                              "c": REWARD_C, "band": list(REWARD_BAND), "sample_seed": LIQ_SEED,
                              "approximations": "full recorded ladder, a price level counts as one order for the "
                                                "min-size rule; book stats use the program at the snapshot time"},
        "tick_rule": "as data/features.py: 0.01 until the top-5 book or a maker fill shows a sub-cent price, then 0.001 (sticky)",
        "book_record": {"itemsize": 64, "doc": "identical to data/features.py (feat_meta of data/build)"},
        "fill_record": {"itemsize": 20, "doc": "identical to data/features.py; row = global row = block time",
                        "dummy_zero_leg": dummy_fill},
        "fill_index": "int32 [N + 1]: legs of row r are fills[fill_index[r]:fill_index[r+1]]",
        "sanity": trade_sanity,
        "stats": {"episodes": per_ep[:2000], "n_fills": int(len(fills))},
    }
    FT.write_atomic(out / "feat_meta.json", json.dumps(fmeta, indent=1, default=str))
    write_side_tables(out, eps, ep_meta, minute_parts, mk)
    write_gaps(out, gp, a)
    LD.verify(out)
    tot = pd.DataFrame(per_ep).drop(columns="id").sum().to_dict() if per_ep else {}
    print(f"built {len(eps)} episodes, N={N}, fills {len(fills)}{' (dummy)' if dummy_fill else ''} into {out} "
          f"in {time.time() - t0:.0f}s; totals {tot}; trade sanity {trade_sanity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
