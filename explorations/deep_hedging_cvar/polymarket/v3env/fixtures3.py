"""Synthetic builder outputs for test_env3.py: small sports and election builds in the exact schemas
of analysis/v3_sports/build_sports.py and analysis/v3_elections/build_elections.py (the five v2
files, meta.json, feat_meta.json, plus sports_feat.bin / sports_x3.bin / feat_x.bin), and a hand-made
reward-hook market with a static book. Each is then made loadable with adapt3.py, as a real build
would be. The sports builds: `sports` (sports-2 schema, legacy settle copy on), `sports1` (the
sports-1 layout, rewards through adapt3 --rewards), `sports2_nolegacy` (--no-legacy-settle) and
`sports2_buckets` (--bucket-segments); the first three hold the same six games and adapt to the same
markets3.bin / x3.bin. D16 recording gaps: `sports3_gaps` (build_sports sports-3: gap-cut pieces
linked by segments[i].gap.prev_sid / next_sid, GAP_CUT / AFTER_GAP / EVAL_EXCL_LAST_HOUR flags,
eval_excluded_buckets, meta.gaps.days with an ineligible day), `sports3_nogap` (game 0 of it as one
continuous market: its two pieces back to back, for the chained-P&L test) and `elections_gaps`
(build_elections: a window cut at a gap into linked episodes, markets_meta.parquet links,
gaps_per_day.csv with an ineligible day).

    python fixtures3.py OUT_DIR          # writes OUT_DIR/{sports,...,sports3_gaps,sports3_nogap,elections_gaps,unit}
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from reference3 import (BOOK_DTYPE, CAT_ELECTIONS, CAT_SPORTS, EXT_DTYPE, FILL_DTYPE, MARKET_DTYPE, SETTLE_LAST_MARK,
                        SETTLE_PRICE, X_COLS)

FCOLS = ["p_logit", "mark_logit", "gap_c", "dgap5_c", "z", "sdv_bp", "sigma_1m_bp", "tau_frac", "log_tau",
         "in_twap", "delta_usd", "spread_ticks", "bid_present", "ask_present", "imb_l1", "bid_sz0", "ask_sz0",
         "flow5", "flow30", "flow120", "mret5_c", "mret30_c", "mret120_c", "ulz5", "ulz30", "ulz120",
         "len_5m", "len_15m", "len_4h", "tick_fine", "book_age", "crossed_raw", "mark_is_fv",
         "p_model", "mark", "delta_usd_raw", "sigma", "sqrt_v", "liq_mid", "liq_comp", "liq_pi",
         "bid_minus_p_c", "ask_minus_p_c", "pad_0"]
CI = {c: i for i, c in enumerate(FCOLS)}
# feat_x.bin columns of build_elections.py (X_DOCS order, padded to a multiple of 4)
EX_COLS = (["rw_in_program", "rw_daily_rate", "rw_min_size", "rw_max_spread_c", "rw_cfg_age_s", "rw_adj_mid",
            "rw_comp_q_bid", "rw_comp_q_ask", "rw_comp_q", "rw_two_sided_only", "rw_sample", "rw_usd_per_sample"]
           + [f"rw_bid_px_a{a}" for a in range(1, 7)] + [f"rw_ask_px_a{a}" for a in range(1, 7)]
           + [f"rw_bid_score_a{a}" for a in range(1, 7)] + [f"rw_ask_score_a{a}" for a in range(1, 7)]
           + [f"rw_size_ok_{k}" for k in range(3)] + ["rw_min_size_mult", "x_covered", "x_log_days_to_end"])
EX_COLS += [f"xpad_{i}" for i in range((-len(EX_COLS)) % 4)]
EXI = {c: i for i, c in enumerate(EX_COLS)}


def same_run(book):
    lad = np.concatenate([book["bid_px"].astype(np.float64), book["ask_px"], book["bid_sz"], book["ask_sz"]], 1)
    same = np.r_[False, (lad[1:] == lad[:-1]).all(1)]
    n = len(same)
    nxt = np.where(~same, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    book["same_run"] = np.minimum(nxt - np.arange(n), 255)
    book["flags"] = (book["flags"] & np.uint8(0xEF)) | (same * 16).astype(np.uint8)


def episode(rng, n, mid0, deep, update_p, fill_p, crossed_p=0.004):
    """Book, legs (row relative) and the mid of one synthetic market of n rows on the 1c grid."""
    book = np.zeros(n, BOOK_DTYPE)
    mid = np.zeros(n)
    m = mid0
    bb = ba = 0
    for r in range(n):
        if r == 0 or rng.random() < update_p:
            m = float(np.clip(m + rng.normal(0, 0.004), 0.06, 0.94))
            bb = int(np.floor(m * 100)) * 10
            ba = bb + 10 * (1 + (rng.random() < 0.3))
            book["bid_sz"][r] = rng.integers(5, deep, 5)
            book["ask_sz"][r] = rng.integers(5, deep, 5)
        else:
            book["bid_sz"][r] = book["bid_sz"][r - 1]
            book["ask_sz"][r] = book["ask_sz"][r - 1]
        book["bid_px"][r] = [max(bb - 10 * k, 0) for k in range(5)]
        book["ask_px"][r] = [min(ba + 10 * k, 1000) for k in range(5)]
        book["bid_sz"][r][book["bid_px"][r] == 0] = 0
        book["ask_sz"][r][book["ask_px"][r] == 1000] = 0
        fl = 1
        if rng.random() < crossed_p:
            fl |= 2
        if r % 997 == 400:                       # a short one-sided stretch
            book["ask_px"][r] = 1000
            book["ask_sz"][r] = 0
        book["flags"][r] = fl
        book["tick"][r] = book["tick_alt"][r] = 10
        mid[r] = (bb + ba) / 2000.0
    same_run(book)
    legs = []
    for r in range(1, n):
        if rng.random() >= fill_p:
            continue
        side = 1 if rng.random() < 0.5 else -1          # +1 taker buys (hits asks)
        pr = book[r - 1]
        lv = (pr["ask_px"][:3] if side > 0 else pr["bid_px"][:3])
        k = int(rng.integers(1, 4))
        sizes = rng.integers(1, 60, k).astype(np.float32)
        for j in range(k):
            px = int(lv[min(j, 2)])
            if px <= 0 or px >= 1000:
                continue
            legs.append((r, side, int(j == 0), px, px / 1000.0, float(sizes[j]), float(sizes.sum())))
    fills = np.array(legs, FILL_DTYPE) if legs else np.zeros(0, FILL_DTYPE)
    return book, fills, mid


def features(book, mid, n, tick=0.01):
    """The feat.bin columns the env reads or observes, filled as the v3 builders do (p_model = mid)."""
    f = np.zeros((n, 44), np.float32)
    bb, ba = book["bid_px"][:, 0], book["ask_px"][:, 0]
    two = (bb > 0) & (ba < 1000)
    m = mid.copy()
    for r in range(n):                          # one-sided: the last two-sided mid
        if not two[r]:
            m[r] = m[r - 1] if r else 0.5
    lg = np.log(np.clip(m, 1e-4, 1 - 1e-4) / (1 - np.clip(m, 1e-4, 1 - 1e-4)))
    tau = np.arange(n)[::-1]
    f[:, CI["p_logit"]] = lg / 3
    f[:, CI["mark_logit"]] = lg / 3
    f[:, CI["tau_frac"]] = tau / (n - 1)
    f[:, CI["log_tau"]] = np.clip(np.log1p(tau) / np.log1p(14400), 0, 5)
    f[:, CI["spread_ticks"]] = np.where(two, np.log(np.maximum((ba - bb) / 10.0, 1)) / 2, 0)
    f[:, CI["bid_present"]] = bb > 0
    f[:, CI["ask_present"]] = ba < 1000
    f[:, CI["bid_sz0"]] = np.log1p(book["bid_sz"][:, 0]) / 5
    f[:, CI["ask_sz0"]] = np.log1p(book["ask_sz"][:, 0]) / 5
    f[:, CI["book_age"]] = 0.1
    f[:, CI["crossed_raw"]] = (book["flags"] & 2) > 0
    f[:, CI["mark_is_fv"]] = ~two
    f[:, CI["p_model"]] = m
    f[:, CI["mark"]] = m
    f[:, CI["liq_mid"]] = -1.0
    f[:, CI["bid_minus_p_c"]] = np.where(bb > 0, np.clip((bb / 1000 - m) * 100 / 5, -5, 5), 0)
    f[:, CI["ask_minus_p_c"]] = np.where(ba < 1000, np.clip((ba / 1000 - m) * 100 / 5, -5, 5), 0)
    return f


def write_build(out: Path, recs, feats, books, legs, meta_extra, fmeta_extra):
    out.mkdir(parents=True, exist_ok=True)
    mk = np.zeros(len(recs), MARKET_DTYPE)
    r0 = 0
    all_legs = []
    counts = []
    for i, (rec, b, lg) in enumerate(zip(recs, books, legs)):
        n = len(b)
        for k, v in rec.items():
            mk[i][k] = v
        mk[i]["row_offset"] = r0
        mk[i]["n_rows"] = n
        mk[i]["market_idx"] = i
        lg = lg.copy()
        counts.append(np.bincount(lg["row"], minlength=n) if len(lg) else np.zeros(n, np.int64))
        lg["row"] += r0
        all_legs.append(lg)
        r0 += n
    N = r0
    np.concatenate(feats).astype(np.float32).tofile(out / "feat.bin")
    np.concatenate(books).tofile(out / "book.bin")
    fills = np.concatenate(all_legs)
    if len(fills) == 0:
        fills = np.zeros(1, FILL_DTYPE)                 # mmap needs a non-empty file (as build_elections)
    fills.tofile(out / "fills.bin")
    idx = np.r_[0, np.cumsum(np.concatenate(counts))].astype(np.int32)
    idx[-1] = len(fills) if idx[-1] != len(fills) else idx[-1]
    idx.tofile(out / "fill_index.bin")
    mk.tofile(out / "markets.bin")
    layout = [{"name": nm, "offset": MARKET_DTYPE.fields[nm][1],
               "type": {"<i8": "int64_t", "<i4": "int32_t", "<f4": "float"}[MARKET_DTYPE.fields[nm][0].str],
               "size": MARKET_DTYPE.fields[nm][0].itemsize} for nm in MARKET_DTYPE.names]
    meta = {"N": N, "n_markets": len(recs), "market_record": {"itemsize": 88, "fields": layout},
            "slugs": [r.get("_slug", f"m{i}") for i, r in enumerate(recs)], **meta_extra}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    fmeta = {"N": N, "G": 44, "n_obs_features": 33, "columns": FCOLS, "column_index": CI, **fmeta_extra}
    (out / "feat_meta.json").write_text(json.dumps(fmeta, indent=1))
    return mk


SPORTS_RATES = [50.0, 20.0, 0.0, 100.0, 5.0, 30.0]       # reward pool USD/day per game (game 2: no program)
SPORTS_BUCKETS = [(">24h", 86400), ("6-24h", 21600), ("1-6h", 3600), ("1h-30m", 1800), ("30-5m", 300), ("5-0m", 0)]
FLAG_SETTLE_PX, FLAG_AT_CUT, FLAG_HAS_TRADES, FLAG_IN_PROGRAM = 16, 128, 1, 2048
FLAG_TRUNC_S1, FLAG_TRUNC_S2 = 32, 1024


def bucket_of(sec):
    """build_sports.bucket_of: index into SPORTS_BUCKETS (upper bounds inclusive: 1800 s is 30-5m)."""
    return int(np.searchsorted(-np.array([b for _, b in SPORTS_BUCKETS[:-1]]), -np.asarray(sec), side="right"))


def make_sports(out: Path, seed=1, schema="sports-2", legacy_settle=True, bucket_segments=False):
    """Pre-game sports build in the build_sports.py schema: six games, each one episode of arbitrary
    length ending at kickoff (--seg 0), settling at the last pre-kickoff mark.

    schema "sports-2" (default; build_sports.py sports-2): meta.json `segments` (one record per market:
    kickoff_ts, settle_price, bucket fields, rw.rate_day_last, flags), sports_x3.bin with the reward
    columns (computed here with adapt3's book scorer, so the sports-1 fallback gives the same x3), feat
    liq_* = its columns, TRADES_TRUNCATED = flag 1024 (game 3); bit 32 is set on game 5 to check that it
    is not read. `legacy_settle=False` is --no-legacy-settle (no flag 16, winner_up -1, log_strike 0).
    `bucket_segments=True` is --bucket-segments: each game is split at the 30 m / 5 m boundaries before
    kickoff, every piece settling at its own last-row mark (ids #k0 = the piece ending at kickoff).
    schema "sports-1": the legacy layout (flag 16 + log_strike_vs_ul_open, kickoff from sports_feat.bin,
    no pools: adapt3.py --rewards rates.json), TRADES_TRUNCATED = flag 32 (game 3)."""
    from adapt3 import sports_x3_top5
    assert schema in ("sports-1", "sports-2") and (schema == "sports-2" or (legacy_settle and not bucket_segments))
    rng = np.random.default_rng(seed)
    lens = [1800, 2401, 3000, 1201, 2000, 2700]           # rows - 1 is the game window length
    splits = [0, 0, 1, 1, 0, 1]
    rw_min = [100, 50, 0, 200, 20, 100]
    rw_v = [3.5, 4.5, 0, 3.0, 2.0, 3.5]
    base = 1_790_000_000
    eps = []                                               # (game, a, b) row ranges of each episode (inclusive)
    games = []
    for i, (L, sp) in enumerate(zip(lens, splits)):
        n = L + 1
        b, lg, mid = episode(rng, n, 0.3 + 0.1 * i, deep=400, update_p=0.25, fill_p=0.35)
        f = features(b, mid, n)
        ko = base + i * 86400
        games.append(dict(book=b, legs=lg, feat=f, ko=ko, open_ts=ko - L, L=L, split=sp,
                          slug=f"nhl-aaa-bb{i}-2026-10-0{i + 1}"))
        cuts = [0, L]
        if bucket_segments:
            cuts = sorted({0, L} | {L - bnd for _, bnd in SPORTS_BUCKETS[:-1] if 0 < L - bnd < L})
        for k, (a, e) in enumerate(zip(cuts[:-1], cuts[1:])):
            eps.append((i, a, e, len(cuts) - 2 - k))
    recs, feats, books, legs, sfs, seg_meta = [], [], [], [], [], []
    r0 = 0
    for gi, a, e, kk in eps:
        g = games[gi]
        n = e - a + 1
        f = g["feat"][a:e + 1].copy()
        bk = g["book"][a:e + 1].copy()
        lg = g["legs"][(g["legs"]["row"] > a) & (g["legs"]["row"] <= e)].copy()   # legs in (a, e]; row a is the start
        lg["row"] -= a
        tau = np.arange(n)[::-1]                                                   # the episode's own clock
        f[:, CI["tau_frac"]] = tau / (n - 1)
        open_ts = g["open_ts"] + a
        end_ts = g["open_ts"] + e
        settle = float(f[-1, CI["mark"]])
        to_ko = g["ko"] - (open_ts + np.arange(n))
        prog = SPORTS_RATES[gi] > 0 and rw_v[gi] > 0
        flags = FLAG_HAS_TRADES | (FLAG_AT_CUT if kk == 0 else 0)
        if legacy_settle:
            flags |= FLAG_SETTLE_PX
        if gi == 3:
            flags |= FLAG_TRUNC_S1 if schema == "sports-1" else FLAG_TRUNC_S2
        if schema == "sports-2":
            flags |= FLAG_IN_PROGRAM if prog else 0
            if gi == 5:
                flags |= 32                               # unused in sports-2: must not read as TRADES_TRUNCATED
        recs.append(dict(open_ts=open_ts, end_ts=end_ts, contract_len_s=n - 1, coin_id=gi % 3,
                         winner_up=int(settle >= 0.5) if legacy_settle else -1, split=g["split"], flags=flags,
                         tick_size=0.01, taker_fee_rate=0.05, fee_exponent=1.0, min_order_size=5.0,
                         rewards_min_size=rw_min[gi], rewards_max_spread=rw_v[gi],
                         log_strike_vs_ul_open=settle if legacy_settle else 0.0, maker_rebate_rate=0.15,
                         tick_size_final=0.01))
        feats.append(f)
        books.append(bk)
        legs.append(lg)
        sf = np.zeros((n, 4), np.float32)
        sf[:, 0] = to_ko
        sf[:, 1] = tau
        sf[:, 2] = -1
        sfs.append(sf)
        cnt = np.bincount([bucket_of(x) for x in to_ko[:-1]], minlength=len(SPORTS_BUCKETS))
        seg_meta.append({
            "market_idx": len(seg_meta), "row_offset": r0, "sid": f"{g['slug']}#k{kk}", "slug": g["slug"],
            "condition_id": f"0x{gi:064x}", "n_rows": n, "open_ts": int(open_ts), "end_ts": int(end_ts), "row_s": 1,
            "kickoff_ts": int(g["ko"]), "cut_ts": int(g["ko"]), "event_ts": int(end_ts),
            "kickoff_scheduled_ts": int(g["ko"]), "kickoff_source": "gamma", "at_cut": kk == 0,
            "segments_before_cut": kk, "to_kickoff_start_s": int(to_ko[0]), "to_kickoff_end_s": int(to_ko[-1]),
            "bucket": SPORTS_BUCKETS[int(cnt.argmax())][0], "bucket_pure": bool((cnt > 0).sum() == 1),
            "bucket_rows": {SPORTS_BUCKETS[k][0]: int(c) for k, c in enumerate(cnt) if c},
            "settle_mode": 1, "settle_price": settle, "actual_start": None, "early_start": False,
            "rw": {"in_program_any": prog, "in_program_last": prog, "rate_day_last": SPORTS_RATES[gi],
                   "v_c_last": rw_v[gi], "min_size_last": float(rw_min[gi])}, "flags": flags})
        r0 += n
    slugs = [x["sid"] for x in seg_meta]
    if schema == "sports-1":
        meta = {"build_version": "sports-1", "slugs": slugs, "sports": {"dt_s": 1}}
    else:
        meta = {"build_version": "sports-2", "slugs": slugs, "segments": seg_meta,
                "settle_source": {"legacy_markets_bin": legacy_settle},
                "sports": {"dt_s": 1, "seg_rows": 0, "cut_before_kickoff_s": 0,
                           "buckets": {"bucket_segments": bucket_segments,
                                       "counts": {b: sum(x["bucket"] == b for x in seg_meta) for b, _ in SPORTS_BUCKETS}}}}
    mk = write_build(out, recs, feats, books, legs, meta, {})
    np.concatenate(sfs).tofile(out / "sports_feat.bin")
    rates = {}
    for x in seg_meta:
        rates[x["slug"]] = x["rw"]["rate_day_last"]
    (out / "rates.json").write_text(json.dumps(rates))
    if schema == "sports-2":                              # the builder's reward columns (sports_x3.bin, feat liq_*)
        prog = {"rw_v_c": mk["rewards_max_spread"], "rw_min_size": mk["rewards_min_size"],
                "rw_rate_day": np.array([x["rw"]["rate_day_last"] for x in seg_meta], np.float32)}
        x3 = sports_x3_top5(np.fromfile(out / "book.bin", BOOK_DTYPE), mk, prog, 1.0)
        x3.tofile(out / "sports_x3.bin")
        feat = np.fromfile(out / "feat.bin", np.float32).reshape(-1, 44)
        for j, c in ((4, "liq_mid"), (5, "liq_comp"), (6, "liq_pi")):
            feat[:, CI[c]] = x3[:, j]
        feat.tofile(out / "feat.bin")
    return mk


def make_elections(out: Path, seed=2):
    """Election reward-farming build (build_elections.py schema, split feat layout): 1 h windows,
    winner_up = -1, the program changes mid-window (min size, rate), one sample second per minute."""
    rng = np.random.default_rng(seed)
    L = 3600
    recs, feats, books, legs, fxs = [], [], [], [], []
    for i, sp in enumerate([0, 0, 1, 1, 0]):
        n = L + 1
        b, lg, mid = episode(rng, n, [0.5, 0.07, 0.35, 0.62, 0.93][i], deep=300, update_p=0.1, fill_p=0.15)
        f = features(b, mid, n)
        v = np.full(n, [3.5, 4.5, 3.0, 3.5, 2.5][i])
        ms = np.full(n, [50.0, 20.0, 100.0, 50.0, 200.0][i])
        rate = np.full(n, [120.0, 40.0, 300.0, 0.0, 60.0][i])
        prog = (rate > 0).astype(np.float32)
        ms[n // 2:] *= 2                                  # the hourly snapshot changed the program
        rate[n // 2:] *= 0.5
        # competitors from the top-5 book with the market's program (adapt3's scorer)
        from adapt3 import book_rewards
        amid, qb, qa = book_rewards(b, v, ms)
        inside = (amid >= 0.1) & (amid <= 0.9)
        comp = np.minimum(qb, qa) + np.where(inside, np.abs(qb - qa) / 3, 0)
        sample = np.zeros(n, bool)
        for m0 in range(0, n, 60):
            sample[min(m0 + int(rng.integers(0, 60)), n - 1)] = True
        f[:, CI["liq_mid"]] = amid
        f[:, CI["liq_comp"]] = comp
        f[:, CI["liq_pi"]] = np.where(sample & (amid >= 0) & (prog > 0), rate / 1440, 0)
        fx = np.zeros((n, len(EX_COLS)), np.float32)
        fx[:, EXI["rw_in_program"]] = prog
        fx[:, EXI["rw_daily_rate"]] = rate
        fx[:, EXI["rw_min_size"]] = ms
        fx[:, EXI["rw_max_spread_c"]] = v
        fx[:, EXI["rw_adj_mid"]] = amid
        fx[:, EXI["rw_comp_q"]] = comp
        fx[:, EXI["rw_sample"]] = sample
        fx[:, EXI["rw_usd_per_sample"]] = rate / 1440
        open_ts = 1_790_000_000 + i * 86400
        recs.append(dict(open_ts=open_ts, end_ts=open_ts + L, contract_len_s=L, coin_id=-1, winner_up=-1, split=sp,
                         flags=16 | 32 | 128 | 1, tick_size=0.01, taker_fee_rate=0.04, fee_exponent=1.0,
                         min_order_size=5.0, rewards_min_size=ms[0], rewards_max_spread=v[0],
                         log_strike_vs_ul_open=0.0, maker_rebate_rate=0.25, tick_size_final=0.01))
        feats.append(f)
        books.append(b)
        legs.append(lg)
        fxs.append(fx)
    mk = write_build(out, recs, feats, books, legs, {"builder": "analysis/v3_elections/build_elections.py"},
                     {"extra": {"file": "feat_x.bin", "X": len(EX_COLS), "columns": EX_COLS, "column_index": EXI}})
    np.concatenate(fxs).tofile(out / "feat_x.bin")
    return mk


# ----------------------------------------------------------------------------- D16 gap fixtures
S3_GAP_CUT, S3_AFTER_GAP, S3_EVAL_EXCL = 16384, 32768, 65536          # build_sports sports-3 markets.bin flags
LAST_HOUR_BUCKETS = ("1h-30m", "30-5m", "5-0m")
GAP_DAY0 = int(_dt.datetime(2026, 10, 1, 18, tzinfo=_dt.timezone.utc).timestamp())   # game 0 kickoff
# (window rows L, removed rows [g0, g1) of the timeline or None, keep the post-gap piece, split, day index)
# game 0: a 120 s gap 4000 s before kickoff (a chain, scored); game 1: a 60 s gap in the last hour (both
# pieces excluded from eval for the last-hour buckets); game 2: a gap 100 s before kickoff whose post-gap
# piece was dropped as too short (GAP_CUT, no continuation: flattened at its settle); game 3: no gap, on a
# day with 40 min of recorder gaps (ineligible: train only); games 4 / 5 train, game 4 with a chain.
SPORTS3_GAMES = [(6000, (2000, 2120), True, 1, 0), (5400, (4800, 4860), True, 1, 1), (2000, (1900, 1960), False, 1, 2),
                 (1800, None, True, 1, 5), (5000, (900, 980), True, 0, 3), (1500, None, True, 0, 4)]
SPORTS3_DAYS = {"2026-10-01": (1, 120.0), "2026-10-02": (1, 60.0), "2026-10-03": (1, 60.0), "2026-10-04": (1, 80.0),
                "2026-10-05": (0, 0.0), "2026-10-06": (2, 2400.0)}


def make_sports3_gaps(out: Path, seed=7):
    """A build_sports sports-3 build with D16 gap cuts (SPORTS3_GAMES): pieces end at the last row before a
    removed stretch of the timeline (GAP_CUT, cut_reason 'gap', settle = their last-row mark) and restart
    after it (AFTER_GAP, start_reason 'gap'), linked by segments[i].gap.prev_sid / next_sid; market order
    (split, end_ts, sid) as the builder's. Returns the per-game piece table."""
    from adapt3 import sports_x3_top5
    rng = np.random.default_rng(seed)
    pieces = []
    for gi, (L, gap, keep_post, sp, day) in enumerate(SPORTS3_GAMES):
        n = L + 1
        b, lg, mid = episode(rng, n, 0.35 + 0.08 * gi, deep=400, update_p=0.25, fill_p=0.35)
        f = features(b, mid, n)
        ko = GAP_DAY0 + day * 86400
        slug = f"nhl-ccc-dd{gi}-{_dt.datetime.fromtimestamp(ko, _dt.timezone.utc):%Y-%m-%d}"
        g = dict(book=b, legs=lg, feat=f, ko=ko, open_ts=ko - L, L=L, split=sp, slug=slug, gi=gi)
        spans = [(0, L, "listing", "cut")] if gap is None else [(0, gap[0] - 1, "listing", "gap")] + (
            [(gap[1], L, "gap", "cut")] if keep_post else [])
        lh = gap is not None and (L - gap[1]) < 3600            # gap within the last hour before kickoff
        for k, (a, e, sr, er) in enumerate(spans):
            pieces.append(dict(g=g, a=a, e=e, sr=sr, er=er, kk=len(spans) - 1 - k, gap=gap, lh=lh))
    recs, feats, books, legs, segs = [], [], [], [], []
    for pc in pieces:
        g, a, e = pc["g"], pc["a"], pc["e"]
        n = e - a + 1
        f = g["feat"][a:e + 1].copy()
        bk = g["book"][a:e + 1].copy()
        same_run(bk)
        lg = g["legs"][(g["legs"]["row"] > a) & (g["legs"]["row"] <= e)].copy()
        lg["row"] -= a
        f[:, CI["tau_frac"]] = np.arange(n)[::-1] / (n - 1)
        open_ts, end_ts = g["open_ts"] + a, g["open_ts"] + e
        to_ko = g["ko"] - (open_ts + np.arange(n))
        cnt = np.bincount([bucket_of(x) for x in to_ko[:-1]], minlength=len(SPORTS_BUCKETS))
        rows = {SPORTS_BUCKETS[k][0]: int(c) for k, c in enumerate(cnt) if c}
        excl = [x for x in LAST_HOUR_BUCKETS if x in rows] if pc["lh"] else []
        flags = (FLAG_HAS_TRADES | FLAG_IN_PROGRAM | (FLAG_AT_CUT if e == g["L"] else 0)
                 | (S3_GAP_CUT if pc["er"] == "gap" else 0) | (S3_AFTER_GAP if pc["sr"] == "gap" else 0)
                 | (S3_EVAL_EXCL if excl else 0))
        settle = float(f[-1, CI["mark"]])
        gp = pc["gap"]
        giv = None if gp is None else {"t0": float(g["open_ts"] + gp[0] - 1), "t1": float(g["open_ts"] + gp[1]),
                                       "len_s": float(gp[1] - gp[0] + 1)}
        recs.append(dict(open_ts=open_ts, end_ts=end_ts, contract_len_s=n - 1, coin_id=0, winner_up=-1,
                         split=g["split"], flags=flags, tick_size=0.01, taker_fee_rate=0.05, fee_exponent=1.0,
                         min_order_size=5.0, rewards_min_size=50.0, rewards_max_spread=3.5, log_strike_vs_ul_open=0.0,
                         maker_rebate_rate=0.15, tick_size_final=0.01))
        feats.append(f)
        books.append(bk)
        legs.append(lg)
        segs.append({"sid": f"{g['slug']}#k{pc['kk']}", "slug": g["slug"], "condition_id": f"0x{g['gi']:064x}",
                     "n_rows": n, "open_ts": int(open_ts), "end_ts": int(end_ts), "row_s": 1,
                     "kickoff_ts": int(g["ko"]), "cut_ts": int(g["ko"]), "event_ts": int(end_ts),
                     "at_cut": e == g["L"], "segments_before_cut": pc["kk"], "cut_reason": pc["er"],
                     "start_reason": pc["sr"],
                     "gap": {"gap_before": giv if pc["sr"] == "gap" else None,
                             "gap_after": giv if pc["er"] == "gap" else None, "prev_sid": None, "next_sid": None,
                             "n_gaps_game": int(gp is not None), "total_gap_s_game": giv["len_s"] if giv else 0.0},
                     "last_hour_gap": bool(pc["lh"]), "excluded_from_eval_last_hour": bool(excl),
                     "eval_excluded_buckets": excl, "to_kickoff_start_s": int(to_ko[0]),
                     "to_kickoff_end_s": int(to_ko[-1]), "bucket": SPORTS_BUCKETS[int(cnt.argmax())][0],
                     "bucket_pure": bool((cnt > 0).sum() == 1), "bucket_rows": rows, "settle_mode": 1,
                     "settle_price": settle, "rw": {"in_program_any": True, "in_program_last": True,
                                                    "rate_day_last": 40.0, "v_c_last": 3.5, "min_size_last": 50.0},
                     "flags": flags, "_split": g["split"]})
    for p_, q_ in zip(segs[:-1], segs[1:]):          # link the pieces of one game across its gap
        if p_["slug"] == q_["slug"] and p_["cut_reason"] == "gap" and q_["start_reason"] == "gap":
            p_["gap"]["next_sid"], q_["gap"]["prev_sid"] = q_["sid"], p_["sid"]
    order = sorted(range(len(segs)), key=lambda i: (segs[i]["_split"], segs[i]["end_ts"], segs[i]["sid"]))
    recs, feats, books, legs, segs = ([x[i] for i in order] for x in (recs, feats, books, legs, segs))
    r0 = 0
    for i, x in enumerate(segs):
        x["market_idx"], x["row_offset"] = i, r0
        x.pop("_split")
        r0 += x["n_rows"]
    days = {d: {"n_gaps": k, "total_gap_s": t, "all_logged": True, "eligible": t <= 1800}
            for d, (k, t) in SPORTS3_DAYS.items()}
    meta = {"build_version": "sports-3", "slugs": [x["sid"] for x in segs], "segments": segs,
            "settle_source": {"legacy_markets_bin": False},
            "sports": {"dt_s": 1, "seg_rows": 0, "cut_before_kickoff_s": 0, "buckets": {"bucket_segments": False}},
            "gaps": {"gap_s": 15.0, "max_day_gap_s": 1800.0, "last_hour_s": 3600, "days": days,
                     "last_hour_buckets": list(LAST_HOUR_BUCKETS),
                     "val_test_days_ineligible": sorted(d for d, x in days.items() if not x["eligible"]),
                     "games_last_hour_gap": sorted({x["slug"] for x in segs if x["last_hour_gap"]})}}
    mk = write_build(out, recs, feats, books, legs, meta, {})
    np.zeros((len(np.fromfile(out / "book.bin", BOOK_DTYPE)), 4), np.float32).tofile(out / "sports_feat.bin")
    prog = {"rw_v_c": mk["rewards_max_spread"], "rw_min_size": mk["rewards_min_size"],
            "rw_rate_day": np.full(len(mk), 40.0, np.float32)}
    x3 = sports_x3_top5(np.fromfile(out / "book.bin", BOOK_DTYPE), mk, prog, 1.0)
    x3.tofile(out / "sports_x3.bin")
    feat = np.fromfile(out / "feat.bin", np.float32).reshape(-1, 44)
    for j, c in ((4, "liq_mid"), (5, "liq_comp"), (6, "liq_pi")):
        feat[:, CI[c]] = x3[:, j]
    feat.tofile(out / "feat.bin")
    return mk


def make_sports3_nogap(out: Path, gap_dir: Path, game=0):
    """Game `game` of the sports3_gaps build as ONE market: its pre-gap and post-gap pieces back to back
    (the gap's rows are absent, as in the build), with the same feat / book / x3 / fills rows, settling at
    the post-gap piece's settle (markets3.bin written directly). Returns (gap_row, gap_s): the first
    post-gap row of the market and the seconds the timeline skips there."""
    meta = json.loads((gap_dir / "meta.json").read_text())
    mk = np.fromfile(gap_dir / "markets.bin", MARKET_DTYPE)
    ext = np.fromfile(gap_dir / "markets3.bin", EXT_DTYPE)
    N = len(np.fromfile(gap_dir / "book.bin", BOOK_DTYPE))
    feat = np.fromfile(gap_dir / "feat.bin", np.float32).reshape(N, 44)
    book = np.fromfile(gap_dir / "book.bin", BOOK_DTYPE)
    x3 = np.fromfile(gap_dir / "x3.bin", np.float32).reshape(N, len(X_COLS))
    fills = np.fromfile(gap_dir / "fills.bin", FILL_DTYPE)
    fidx = np.fromfile(gap_dir / "fill_index.bin", np.int32)
    slug = f"nhl-ccc-dd{game}-"
    ids = sorted((i for i, x in enumerate(meta["segments"]) if x["slug"].startswith(slug)),
                 key=lambda i: int(mk["open_ts"][i]))
    assert len(ids) == 2 and ext["gap_next"][ids[0]] == ids[1] + 1, "game is not a two-piece chain"
    fs, bs, xs, ls = [], [], [], []
    off = 0
    for i in ids:
        r0, n = int(mk["row_offset"][i]), int(mk["n_rows"][i])
        fs.append(feat[r0:r0 + n])
        bs.append(book[r0:r0 + n])
        xs.append(x3[r0:r0 + n])
        lg = fills[fidx[r0]:fidx[r0 + n]].copy()
        lg["row"] -= r0
        ls.append(lg)
        off += n
    a, b = ids
    rec = {k: mk[b][k] for k in MARKET_DTYPE.names if k not in ("row_offset", "n_rows", "market_idx")}
    n_all = int(mk["n_rows"][a] + mk["n_rows"][b])
    rec.update(open_ts=int(mk["open_ts"][a]), end_ts=int(mk["open_ts"][a]) + n_all - 1, contract_len_s=n_all - 1,
               flags=int(mk["flags"][b]) & ~(S3_GAP_CUT | S3_AFTER_GAP))
    post = ls[1][ls[1]["row"] > 0].copy()                 # the piece's own row 0 holds no legs
    post["row"] += int(mk["n_rows"][a])
    write_build(out, [rec], [np.concatenate(fs)], [np.concatenate(bs)], [np.concatenate([ls[0], post])], {}, {})
    e = ext[b:b + 1].copy()
    e["gap_next"] = e["gap_prev"] = 0
    e["flags3"] = 0
    e.tofile(out / "markets3.bin")
    np.concatenate(xs).astype(np.float32).tofile(out / "x3.bin")
    return int(mk["n_rows"][a]), int(mk["open_ts"][b] - mk["end_ts"][a] - 1)


def make_elections_gaps(out: Path, seed=3, parquet=True):
    """build_elections schema with D16 gaps: one 1 h window per market and day, market 0's window cut at a
    gap into two linked episodes (start / end reason 'gap'; ids '<slug>@<open ISO>'), market 1's window cut
    at a gap whose post-gap episode is the chain's end, meta.json episodes (market order, day_eligible),
    gaps_per_day.csv (scope recorder / market; day 2 has 2400 s of gaps: ineligible) and, with `parquet`,
    markets_meta.parquet carrying prev_segment_id / next_segment_id as the builder writes them."""
    rng = np.random.default_rng(seed)
    L = 3600
    day0 = int(_dt.datetime(2026, 10, 16, tzinfo=_dt.timezone.utc).timestamp())
    iso = lambda t: _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    plan = [("pres-a", 0, (1500, 1600), 1), ("senate-b", 0, (2500, 2540), 1), ("house-c", 1, None, 0),
            ("pres-a", 2, None, 0)]
    eps, feats, books, legs, fxs = [], [], [], [], []
    for k, (slug, day, gap, sp) in enumerate(plan):
        n = L + 1
        b, lg, mid = episode(rng, n, [0.45, 0.2, 0.7, 0.5][k], deep=300, update_p=0.1, fill_p=0.15)
        f = features(b, mid, n)
        from adapt3 import book_rewards                  # the reward columns, as make_elections
        amid, qb, qa = book_rewards(b, np.full(n, 3.5), np.full(n, 50.0))
        sample = np.zeros(n, bool)
        sample[np.minimum(np.arange(0, n, 60) + rng.integers(0, 60, len(range(0, n, 60))), n - 1)] = True
        f[:, CI["liq_mid"]] = amid
        f[:, CI["liq_comp"]] = np.minimum(qb, qa) + np.where((amid >= 0.1) & (amid <= 0.9), np.abs(qb - qa) / 3, 0)
        f[:, CI["liq_pi"]] = np.where(sample & (amid >= 0), 50.0 / 1440, 0)
        o = day0 + day * 86400 + 4 * 3600
        spans = [(0, L, "window_boundary", "window_boundary")] if gap is None else [
            (0, gap[0] - 1, "window_boundary", "gap"), (gap[1], L, "gap", "window_boundary")]
        for a, e, sr, er in spans:
            nn = e - a + 1
            ff = f[a:e + 1].copy()
            bk = b[a:e + 1].copy()
            same_run(bk)
            ll = lg[(lg["row"] > a) & (lg["row"] <= e)].copy()
            ll["row"] -= a
            fx = np.zeros((nn, len(EX_COLS)), np.float32)
            fx[:, EXI["rw_in_program"]] = 1
            fx[:, EXI["rw_daily_rate"]] = 50.0
            fx[:, EXI["rw_min_size"]] = 50.0
            fx[:, EXI["rw_max_spread_c"]] = 3.5
            eps.append(dict(slug=slug, id=f"{slug}@{iso(o + a)}", day=_dt.datetime.fromtimestamp(o, _dt.timezone.utc)
                            .strftime("%Y-%m-%d"), open_ts=o + a, end_ts=o + e, start_reason=sr, end_reason=er,
                            split=sp, n=nn, day_eligible=day != 2))
            feats.append(ff)
            books.append(bk)
            legs.append(ll)
            fxs.append(fx)
    order = sorted(range(len(eps)), key=lambda i: (eps[i]["split"], eps[i]["open_ts"], eps[i]["id"]))
    eps, feats, books, legs, fxs = ([x[i] for i in order] for x in (eps, feats, books, legs, fxs))
    recs = [dict(open_ts=x["open_ts"], end_ts=x["end_ts"], contract_len_s=x["n"] - 1, coin_id=-1, winner_up=-1,
                 split=x["split"], flags=16 | 32 | 1, tick_size=0.01, taker_fee_rate=0.04, fee_exponent=1.0,
                 min_order_size=5.0, rewards_min_size=50.0, rewards_max_spread=3.5, log_strike_vs_ul_open=0.0,
                 maker_rebate_rate=0.25, tick_size_final=0.01) for x in eps]
    by = {}
    for i, x in enumerate(eps):
        by.setdefault(x["slug"], []).append(i)
    prev, nxt = [""] * len(eps), [""] * len(eps)          # build_elections' link rule
    for lst in by.values():
        lst.sort(key=lambda i: eps[i]["open_ts"])
        for p_, q_ in zip(lst[:-1], lst[1:]):
            if eps[p_]["end_reason"] == "gap" and eps[q_]["start_reason"] == "gap":
                prev[q_], nxt[p_] = eps[p_]["id"], eps[q_]["id"]
    ep_meta = [{k: x[k] for k in ("id", "slug", "day", "open_ts", "end_ts", "start_reason", "end_reason", "split",
                                  "day_eligible")} for x in eps]
    write_build(out, recs, feats, books, legs, {"builder": "analysis/v3_elections/build_elections.py",
                                                "slugs": [x["id"] for x in eps], "episodes": ep_meta, "gaps": {"max_gap_s": 15.0, "day_max_gap_s": 1800.0,
                                                                              "ineligible_day": "train-only"}},
                {"extra": {"file": "feat_x.bin", "X": len(EX_COLS), "columns": EX_COLS, "column_index": EXI}})
    np.concatenate(fxs).tofile(out / "feat_x.bin")
    rows = ["scope,day,slug,condition_id,span_start_utc,span_end_utc,span_s,total_gap_s,n_gaps,n_gaps_gt_max,"
            "total_gap_gt_max_s,longest_gap_s,longest_gap_start_utc,eligible,day_complete"]
    for d, (tot, ng) in {"2026-10-16": (140.0, 2), "2026-10-17": (0.0, 0), "2026-10-18": (2400.0, 3)}.items():
        rows.append(f"recorder,{d},,,,,86400,{tot},{ng},{ng},{tot},{tot / max(ng, 1)},,{tot <= 1800},True")
        rows.append(f"market,{d},pres-a,0x0,,,86400,{tot},{ng},{ng},{tot},{tot / max(ng, 1)},,{tot <= 1800},True")
    (out / "gaps_per_day.csv").write_text("\n".join(rows) + "\n")
    if parquet:
        import pandas as pd
        pd.DataFrame({"market_idx": np.arange(len(eps), dtype=np.int32), "episode_id": [x["id"] for x in eps],
                      "prev_segment_id": prev, "next_segment_id": nxt,
                      "gap_before_s": [0.0] * len(eps)}).to_parquet(out / "markets_meta.parquet", index=False)
    return eps


UNIT = dict(n=301, bid=490, ask=510, sample_row=120, v=3.0, min_size=100.0, rate=14.4, comp=100.0, mid=0.50)


def make_unit(out: Path, mid=0.50, comp=100.0):
    """Reward-hook unit market: static book 0.49 / 0.51 (deep), no trades, one sample row, markets3 and
    x3 written directly (v = 3c, min size 100, 14.4 USD/day -> 0.01 USD per sample)."""
    u = UNIT
    n = u["n"]
    b = np.zeros(n, BOOK_DTYPE)
    b["bid_px"] = [u["bid"] - 10 * k for k in range(5)]
    b["ask_px"] = [u["ask"] + 10 * k for k in range(5)]
    b["bid_sz"] = 1000.0
    b["ask_sz"] = 1000.0
    b["flags"] = 1
    b["tick"] = b["tick_alt"] = 10
    same_run(b)
    f = features(b, np.full(n, 0.5), n)
    recs = [dict(open_ts=1_790_000_000, end_ts=1_790_000_000 + n - 1, contract_len_s=n - 1, coin_id=-1, winner_up=-1,
                 split=0, flags=1, tick_size=0.01, taker_fee_rate=0.0, fee_exponent=1.0, min_order_size=5.0,
                 rewards_min_size=u["min_size"], rewards_max_spread=u["v"], log_strike_vs_ul_open=0.0,
                 maker_rebate_rate=0.0, tick_size_final=0.01)]
    write_build(out, recs, [f], [b], [np.zeros(0, FILL_DTYPE)], {}, {})
    ext = np.zeros(1, EXT_DTYPE)
    ext["settle_mode"] = SETTLE_LAST_MARK
    ext["event_ts"] = recs[0]["end_ts"]
    ext["category"] = CAT_ELECTIONS
    ext["row_s"] = 1
    ext["rw_v_c"], ext["rw_min_size"], ext["rw_rate_day"] = u["v"], u["min_size"], u["rate"]
    ext["fee_rate"] = ext["fee_exp"] = ext["rebate_rate"] = -1
    ext.tofile(out / "markets3.bin")
    x3 = np.zeros((n, len(X_COLS)), np.float32)
    x3[:, 0], x3[:, 1], x3[:, 2], x3[:, 3] = u["v"], u["min_size"], u["rate"], 1
    x3[:, 4], x3[:, 5] = mid, comp
    x3[u["sample_row"], 6] = u["rate"] / 1440.0
    x3.tofile(out / "x3.bin")


def make_all(root: Path):
    import adapt3
    root = Path(root)
    make_sports(root / "sports")                          # sports-2 (--rewards is ignored: sports_x3.bin is used)
    adapt3.main([str(root / "sports"), "--rewards", str(root / "sports" / "rates.json")])
    for name, kw in (("sports1", dict(schema="sports-1")), ("sports2_nolegacy", dict(legacy_settle=False)),
                     ("sports2_buckets", dict(bucket_segments=True))):
        make_sports(root / name, **kw)
        adapt3.main([str(root / name)] + (["--rewards", str(root / name / "rates.json")] if name == "sports1" else []))
    make_elections(root / "elections")
    adapt3.main([str(root / "elections")])
    make_sports3_gaps(root / "sports3_gaps")              # D16
    adapt3.main([str(root / "sports3_gaps")])
    make_sports3_nogap(root / "sports3_nogap", root / "sports3_gaps")
    make_elections_gaps(root / "elections_gaps")
    adapt3.main([str(root / "elections_gaps")])
    make_unit(root / "unit")
    return root


if __name__ == "__main__":
    make_all(Path(sys.argv[1]))
