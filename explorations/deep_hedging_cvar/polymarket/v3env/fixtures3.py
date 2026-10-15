"""Synthetic builder outputs for test_env3.py: small sports and election builds in the exact schemas
of analysis/v3_sports/build_sports.py and analysis/v3_elections/build_elections.py (the five v2
files, meta.json, feat_meta.json, plus sports_feat.bin / feat_x.bin), and a hand-made reward-hook
market with a static book. Each is then made loadable with adapt3.py, as a real build would be.

    python fixtures3.py OUT_DIR          # writes OUT_DIR/{sports,elections,unit}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from reference3 import (BOOK_DTYPE, CAT_ELECTIONS, EXT_DTYPE, FILL_DTYPE, MARKET_DTYPE, SETTLE_LAST_MARK, X_COLS)

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


def make_sports(out: Path, seed=1):
    """Pre-game sports build (build_sports.py schema): episodes of arbitrary length ending at kickoff,
    settlement price = the last pre-kickoff mark (log_strike_vs_ul_open, flag SETTLE_PX 16)."""
    rng = np.random.default_rng(seed)
    lens = [1800, 2401, 3000, 1201, 2000, 2700]           # rows - 1 is the episode length
    splits = [0, 0, 1, 1, 0, 1]
    recs, feats, books, legs, sfs = [], [], [], [], []
    base = 1_790_000_000
    for i, (L, sp) in enumerate(zip(lens, splits)):
        n = L + 1
        b, lg, mid = episode(rng, n, 0.3 + 0.1 * i, deep=400, update_p=0.25, fill_p=0.35)
        f = features(b, mid, n)
        ko = base + i * 86400
        open_ts = ko - L
        settle = float(f[-1, CI["mark"]])
        recs.append(dict(open_ts=open_ts, end_ts=ko, contract_len_s=L, coin_id=i % 3, winner_up=int(settle >= 0.5),
                         split=sp, flags=16 | 128 | 1, tick_size=0.01, taker_fee_rate=0.05, fee_exponent=1.0,
                         min_order_size=5.0, rewards_min_size=[100, 50, 0, 200, 20, 100][i],
                         rewards_max_spread=[3.5, 4.5, 0, 3.0, 2.0, 3.5][i], log_strike_vs_ul_open=settle,
                         maker_rebate_rate=0.15, tick_size_final=0.01, _slug=f"nhl-aaa-bb{i}-2026-10-0{i + 1}"))
        feats.append(f)
        books.append(b)
        legs.append(lg)
        sf = np.zeros((n, 4), np.float32)
        sf[:, 0] = ko - (open_ts + np.arange(n))
        sf[:, 1] = sf[:, 0]
        sf[:, 2] = -1
        sfs.append(sf)
    recs = [{k: v for k, v in r.items()} for r in recs]
    slugs = [r.pop("_slug") + "#k0" for r in recs]
    mk = write_build(out, recs, feats, books, legs,
                     {"build_version": "sports-1", "slugs": slugs, "sports": {"dt_s": 1}}, {})
    np.concatenate(sfs).tofile(out / "sports_feat.bin")
    rates = {s.split("#")[0]: r for s, r in zip(slugs, [50.0, 20.0, 0.0, 100.0, 5.0, 30.0])}
    (out / "rates.json").write_text(json.dumps(rates))
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
    make_sports(root / "sports")
    adapt3.main([str(root / "sports"), "--rewards", str(root / "sports" / "rates.json")])
    make_elections(root / "elections")
    adapt3.main([str(root / "elections")])
    make_unit(root / "unit")
    return root


if __name__ == "__main__":
    make_all(Path(sys.argv[1]))
