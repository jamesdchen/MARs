#!/usr/bin/env python3
"""No-RL liquidity-reward farming baseline on the recorded US-midterm books (track E, DECISIONS.md D11).

    source ../../../.venv/bin/activate
    python baseline_farm.py                              # everything recorded with full coverage
    python baseline_farm.py --start 2026-10-15T21:00Z --end 2026-10-16T00:00Z

Strategy (pre-specified, coordinator's spec; every variant is reported, nothing is tuned):
  Per market, every UTC minute t_k, quote a Yes bid and a Yes ask at adjusted_mid -/+ f * max_spread,
  rounded to the 1c tick toward the mid (post-only: never at or through the opposite best), size =
  k x rewards_min_size. f in {0.25, 0.5, 0.75} of max_spread from the mid (f = 0.5 is the spec's
  "max_spread / 2"), k in {1, 2}.
  * adjusted mid / competitors' Q per side: build_elections.reward_stats / side_ladder (full recorded
    ladder, size cutoff = rewards_min_size, a price level counts as one order), program in force at t
    (build_elections.load_reward_cfg / cfg_at), books in the Yes frame (No book mirrored), latest
    uncrossed snapshot <= t and only inside its recorder-coverage segment (build_elections rules).
  * rewards: one sample per minute at build_elections' sample second (seeded uniform second). At the
    sample our orders are re-scored against the book of that second: the adjusted mid includes our
    orders (bid cutoff -> max(cutoff, our bid) when our live size >= min size; same for the ask),
    Q_side = ((v - s) / v)^2 x size (s = cents from that mid, size >= min size, order age >= --moas),
    q_ours = reward_terms.q_min (single-sided /3 inside [0.10, 0.90], two-sided only outside),
    competitors = reward_terms.competitor_q 'hi' bound on the visible book (= build_elections liq_comp),
    share = q_ours / (q_ours + comp), $ = share x daily_rate / 1440.
  * fills: data-api trade legs (build_elections.trades_data_api, maker legs' prices in the Yes frame,
    aggressor from the taker leg) strictly after the quote went live. Back of queue: a leg at exactly
    our price first consumes the displayed size that was at that price when we joined; legs through
    our price fill us (the level was exhausted). An order whose price is unchanged at the next minute and
    was not filled keeps its queue position and age; otherwise it is replaced at the back of the queue.
    Inventory is capped at +-5 x min_size: a side is only quoted while a full fill keeps |inv| <= cap.
  * P&L: spread vs top-of-book mid just before the fill; markouts at +1 m, +5 m, +1 h and to the end of
    the evaluation (vs mid; horizons past the end use the end mid). data-api times are block times, ~2 s after
    the match, so a leg fills an order only if its block time is > placement + --lag (3 s), and the pre-fill
    mid is the mid at block time - --lag; net = rewards + spread + markout_end
    (= rewards + cash + inventory marked to the end mid).
Comparators: no-quote ($0), and rewards-only (same quotes, fills switched off: an upper bound).
Sensitivity (base variant only): front-of-queue fills, and --moas 0.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "analysis" / "rewards"))
import build_elections as BE  # noqa: E402  (import only; main() is guarded)
import reward_terms as RT  # noqa: E402

pa.set_cpu_count(1)
TICK = 0.01
FRACS = (0.25, 0.5, 0.75)
MULTS = (1, 2)
BASE = (0.5, 1)
HORIZONS = {"1m": 60, "5m": 300, "1h": 3600}
INV_CAP_MULT = 5.0
RATE_BUCKETS = [20, 50, 100, 250, 500, np.inf]
EPS = 5e-4
BLOCK_S = 1800                     # bootstrap block (30 min)
RCFG = RT.RewardConfig(name="elections", reward_enabled=True)   # only band / c / competitor bound are used


# ----------------------------------------------------------------------------- books
def snap_file(f: Path, tokmap: dict, cfgs: dict) -> pd.DataFrame:
    """One recorder book file -> one Yes-frame row per (token, captured second): best bid/ask, the size-cutoff
    prices, adjusted mid, competitors' Q per side (BE.reward_stats), and for each spread fraction the quote
    prices and the displayed size at them (queue ahead when joining)."""
    t = pq.read_table(f, columns=BE.BOOK_COLS)
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
    up = np.array([i[1] for i in info], dtype=bool)
    cond = np.array([i[2] for i in info], dtype=object)
    ts = small.ts.values.astype(np.int64)
    n = len(ts)
    m = np.zeros(n)
    v = np.zeros(n)
    for c in np.unique(cond):
        k = cond == c
        cf = BE.cfg_at(cfgs[c], ts[k]) if c in cfgs else {"min_size": 0.0, "max_spread": 0.0}
        m[k], v[k] = cf["min_size"], cf["max_spread"]
    amid, qb, qa = BE.reward_stats(t, m, v)
    rb, pb, sb, cb = BE.side_ladder(t, "bid_prices", "bid_sizes", m)
    ra, pa_, sa, ca = BE.side_ladder(t, "ask_prices", "ask_sizes", m)
    nb = np.bincount(rb, minlength=n)
    na = np.bincount(ra, minlength=n)
    ob = np.r_[0, np.cumsum(nb)[:-1]]
    oa = np.r_[0, np.cumsum(na)[:-1]]
    b0 = np.where(nb > 0, pb[np.minimum(ob, max(len(pb) - 1, 0))] if len(pb) else np.nan, np.nan)
    a0 = np.where(na > 0, pa_[np.minimum(oa, max(len(pa_) - 1, 0))] if len(pa_) else np.nan, np.nan)
    crossed = t.column("crossed").to_numpy(zero_copy_only=False).astype(bool)
    Y = pd.DataFrame({"slug": slug, "is_up": up, "ts": ts, "crossed": crossed,
                      "bb": np.where(up, b0, 1 - a0), "ba": np.where(up, a0, 1 - b0),
                      "cb": np.where(up, cb, 1 - ca), "ca": np.where(up, ca, 1 - cb),
                      "amid": np.where(up, amid, 1 - amid), "qb": np.where(up, qb, qa), "qa": np.where(up, qa, qb),
                      "m": m, "v": v})
    am = Y.amid.values
    bbY = np.nan_to_num(Y.bb.values, nan=0.0)
    baY = np.nan_to_num(Y.ba.values, nan=1.0)
    for f_ in FRACS:
        d = f_ * v / 100.0
        with np.errstate(invalid="ignore"):
            pbq = np.ceil(np.round((am - d) / TICK, 6)) * TICK
            paq = np.floor(np.round((am + d) / TICK, 6)) * TICK
            pbq = np.where(pbq >= baY - EPS, baY - TICK, pbq)          # post-only
            paq = np.where(paq <= bbY + EPS, bbY + TICK, paq)
        pbq = np.round(pbq, 4)
        paq = np.round(paq, 4)
        # 'join' mode: never improve the visible touch (join the best or stay behind it)
        pbj = np.where(np.isfinite(Y.bb.values), np.minimum(pbq, np.round(bbY, 4)), pbq)
        paj = np.where(np.isfinite(Y.ba.values), np.maximum(paq, np.round(baY, 4)), paq)
        for pre, pbx, pax in (("", pbq, paq), ("J", pbj, paj)):
            okb = np.isfinite(pbx) & (pbx >= TICK - EPS)
            oka = np.isfinite(pax) & (pax <= 1 - TICK + EPS)
            # displayed size at our price: Yes bid at p is the Yes bid ladder at p, or the No ask ladder at 1 - p
            tgt_bidlad = np.where(up, pbx, 1 - pax)     # target on this token's bid ladder
            tgt_asklad = np.where(up, pax, 1 - pbx)
            sz_bl = np.bincount(rb, weights=sb * (np.abs(pb - tgt_bidlad[rb]) < EPS), minlength=n) if len(rb) else np.zeros(n)
            sz_al = np.bincount(ra, weights=sa * (np.abs(pa_ - tgt_asklad[ra]) < EPS), minlength=n) if len(ra) else np.zeros(n)
            tag = f"{pre}{int(f_ * 100)}"
            Y[f"pb{tag}"] = np.where(okb, pbx, np.nan)
            Y[f"pa{tag}"] = np.where(oka, pax, np.nan)
            Y[f"qab{tag}"] = np.where(up, sz_bl, sz_al)
            Y[f"qaa{tag}"] = np.where(up, sz_al, sz_bl)
    return Y


def load_books(rec: Path, tokmap, cfgs, days, t_lo, t_hi) -> pd.DataFrame:
    frames = []
    for day in days:
        for attempt in range(3):
            try:
                files = BE.dataset_files(rec, "books", [day])
                fr = []
                for f in files:
                    x = snap_file(f, tokmap, cfgs)
                    if len(x):
                        fr.append(x[(x.ts >= t_lo - 6 * 3600) & (x.ts <= t_hi)])
                frames += fr
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise
                time.sleep(2.0)
    D = pd.concat(frames, ignore_index=True)
    D = D.sort_values(["slug", "is_up", "ts"], kind="stable").drop_duplicates(["slug", "is_up", "ts"], keep="last")
    # merged_snapshots rule: per second the Yes book if uncrossed, else the mirrored No book if uncrossed
    D = D[~D.crossed]
    D = D.assign(pref=np.where(D.is_up, 0, 1)).sort_values(["slug", "ts", "pref"], kind="stable")
    D = D.drop_duplicates(["slug", "ts"], keep="first").drop(columns=["pref", "crossed"])
    return D.reset_index(drop=True)


# ----------------------------------------------------------------------------- one market
class Market:
    """Per-second Yes-frame snapshot arrays of one market + coverage + reward config + trades."""

    def __init__(self, row, snaps: pd.DataFrame, seg, cfg, trades: pd.DataFrame | None, lo: int, hi: int):
        self.row, self.lo, self.hi = row, lo, hi
        self.s = {c: snaps[c].values for c in snaps.columns if c != "slug"}
        self.sts = self.s["ts"].astype(np.int64)
        self.seg, self.cfg = seg, cfg
        two = np.isfinite(self.s["bb"]) & np.isfinite(self.s["ba"])
        mid = np.where(two, 0.5 * (self.s["bb"] + self.s["ba"]), np.nan)
        self.mid = pd.Series(mid).ffill().values          # last two-sided mid
        if trades is not None and len(trades):
            tr = trades[(trades.timestamp > lo) & (trades.timestamp <= hi) & trades.use]
            self.tr_ts = tr.timestamp.values.astype(np.int64)
            self.tr_px = tr.px_up.values
            self.tr_sz = tr["size"].values.astype(np.float64)
            self.tr_ag = tr.aggressor_up.values.astype(np.int64)
            self.tr_tx = tr.tx.values
            self.n_tx = int(tr.tx.nunique())
        else:
            self.tr_ts = np.zeros(0, np.int64)
            self.tr_px = self.tr_sz = np.zeros(0)
            self.tr_ag = np.zeros(0, np.int64)
            self.tr_tx = np.zeros(0, object)
            self.n_tx = 0

    def seg_lo(self, t):
        a, b = self.seg
        i = np.searchsorted(a, t, side="right") - 1
        return a[i] if (i >= 0 and t <= b[i]) else None

    def covered(self, t0, t1):
        a, b = self.seg
        i = np.searchsorted(a, t0, side="right") - 1
        return i >= 0 and b[i] >= t1

    def snap_idx(self, t):
        """Latest snapshot <= t within the coverage segment of t (BE.COV_TOL_S tolerance)."""
        lo = self.seg_lo(t)
        if lo is None:
            return -1
        j = np.searchsorted(self.sts, t, side="right") - 1
        if j < 0 or self.sts[j] < lo - BE.COV_TOL_S:
            return -1
        return j

    def mid_at(self, t):
        j = np.searchsorted(self.sts, t, side="right") - 1
        return self.mid[j] if j >= 0 else np.nan


def run_market(M: Market, sample_off: np.ndarray, frac: float, mult: float, fills_on: bool, queue: str, moas: float,
               lag: int = 3, mode: str = ""):
    """Simulate one market for one variant. Returns (summary dict, fills list)."""
    tag = f"{mode}{int(frac * 100)}"
    S = M.s
    out = dict(rewards=0.0, rewards_samples=0, samples_quoted=0, samples_scorable=0, minutes=0, minutes_quoted=0,
               mid_moved_by_us=0, capital=0.0, rewards_inside=0.0, samples_inside=0, rew_blk={}, ntx_inside={})
    fills = []
    inv = 0.0
    orders = {1: None, -1: None}           # side -> dict(px, size, rem, t_place, queue)
    ti = 0                                  # trade pointer
    ntr = len(M.tr_ts)
    t0 = M.lo - M.lo % 60 + (60 if M.lo % 60 else 0)
    for k, tk in enumerate(range(t0, M.hi - 59, 60)):
        out["minutes"] += 1
        j = M.snap_idx(tk)
        cf = BE.cfg_at(M.cfg, np.array([tk]))
        m_, rate, inprog = float(cf["min_size"][0]), float(cf["rate"][0]), cf["in_program"][0] > 0
        quotable = (j >= 0 and M.covered(tk, tk + 60) and inprog and rate > 0 and m_ > 0
                    and np.isfinite(S["amid"][j]))
        size = mult * m_
        cap = INV_CAP_MULT * m_
        if quotable:
            for side, pcol, qcol in ((1, f"pb{tag}", f"qab{tag}"), (-1, f"pa{tag}", f"qaa{tag}")):
                px = S[pcol][j]
                allowed = np.isfinite(px) and abs(inv + side * size) <= cap + 1e-9
                o = orders[side]
                if not allowed:
                    orders[side] = None
                    continue
                if o is not None and abs(o["px"] - px) < EPS and o["rem"] >= o["size"] - 1e-9 and abs(o["size"] - size) < 1e-9:
                    continue                                    # unchanged, unfilled: keep queue and age
                orders[side] = dict(px=px, size=size, rem=size, t_place=tk, queue=float(S[qcol][j]))
            for side, best in ((1, S["bb"][j]), (-1, S["ba"][j])):
                o = orders[side]
                if o is not None:
                    o["inside"] = (not np.isfinite(best)) or (o["px"] > best + EPS if side == 1 else o["px"] < best - EPS)
            out["minutes_quoted"] += 1
        else:
            orders = {1: None, -1: None}
        # capital: position + collateral of resting orders (inventory covers the opposite order)
        midk = M.mid_at(tk)
        if np.isfinite(midk):
            cap_pos = inv * midk if inv > 0 else -inv * (1 - midk)
            ob, oa = orders[1], orders[-1]
            cb_ = (ob["rem"] - min(max(-inv, 0), ob["rem"])) * ob["px"] if ob else 0.0
            ca_ = (oa["rem"] - min(max(inv, 0), oa["rem"])) * (1 - oa["px"]) if oa else 0.0
            out["capital"] = max(out["capital"], cap_pos + cb_ + ca_)
        # events of minute (tk, tk + 60]: trades and the reward sample
        ts_s = tk + int(sample_off[k]) if k < len(sample_off) else tk + 30
        while ti < ntr and M.tr_ts[ti] <= tk:
            ti += 1
        te = tk + 60
        sampled = False
        while True:
            nxt_t = M.tr_ts[ti] if ti < ntr else None
            if not sampled and (nxt_t is None or nxt_t > ts_s or nxt_t > te):
                if ts_s <= te:
                    usd = _sample(M, out, orders, ts_s, m_, rate, inprog, moas)
                    if usd:
                        b_ = ts_s // BLOCK_S
                        out["rew_blk"][b_] = out["rew_blk"].get(b_, 0.0) + usd
                sampled = True
            if nxt_t is None or nxt_t > te:
                break
            # one transaction (all its legs at this second with the same tx)
            tx = M.tr_tx[ti]
            tj = ti
            while tj < ntr and M.tr_ts[tj] == nxt_t and M.tr_tx[tj] == tx:
                tj += 1
            ag = M.tr_ag[ti]
            # diagnostic: taker txs on the side of a resting order of ours that improves the visible touch
            oo = orders.get(-ag) if ag != 0 else None
            if oo is not None and oo.get("inside"):
                hit = (M.tr_px[ti:tj] <= oo["px"] + EPS) if -ag == 1 else (M.tr_px[ti:tj] >= oo["px"] - EPS)
                key = "at_or_through" if hit.any() else "other_side_px"
                out["ntx_inside"][key] = out["ntx_inside"].get(key, 0) + 1
            if fills_on:
                side = -ag                      # sell aggressor hits bids (we buy), buy aggressor lifts asks
                o = orders.get(side)
                # block time lags matching by ~2 s: a leg can only hit an order placed > lag s before its block
                if ag != 0 and o is not None and o["rem"] > 0 and nxt_t > o["t_place"] + lag:
                    px, sz = M.tr_px[ti:tj], M.tr_sz[ti:tj]
                    at = sz[np.abs(px - o["px"]) < EPS].sum()
                    thru = sz[(px < o["px"] - EPS) if side == 1 else (px > o["px"] + EPS)].sum()
                    if queue == "back":
                        if thru > 0:
                            got = min(o["rem"], max(at - o["queue"], 0.0) + thru)
                            o["queue"] = 0.0
                        else:
                            used = min(o["queue"], at)
                            o["queue"] -= used
                            got = min(o["rem"], at - used)
                    else:                       # front of queue
                        got = min(o["rem"], at + thru)
                    if got > 1e-9:
                        o["rem"] -= got
                        inv += side * got
                        mid_f = M.mid_at(nxt_t - lag)          # mid before the match (block time - lag)
                        fills.append((M.row.slug, int(nxt_t), side, o["px"], got, mid_f))
            ti = tj
        if not quotable:
            orders = {1: None, -1: None}
    out["inv_end"] = inv
    return out, fills


def _sample(M: Market, out, orders, ts_s, m_, rate, inprog, moas):
    if not inprog or rate <= 0:
        return 0.0
    j = M.snap_idx(ts_s)
    if j < 0 or not np.isfinite(M.s["amid"][j]):
        return 0.0
    out["samples_scorable"] += 1
    ob, oa = orders[1], orders[-1]
    if ob is None and oa is None:
        return 0.0
    v = M.s["v"][j]
    cb, ca = M.s["cb"][j], M.s["ca"][j]
    live_b = ob is not None and ob["rem"] >= m_ - 1e-9
    live_a = oa is not None and oa["rem"] >= m_ - 1e-9
    cb2 = max(cb, ob["px"]) if live_b and np.isfinite(cb) else cb
    ca2 = min(ca, oa["px"]) if live_a and np.isfinite(ca) else ca
    mid = 0.5 * (cb2 + ca2)
    if abs(mid - M.s["amid"][j]) > 1e-9:
        out["mid_moved_by_us"] += 1
    qb = float(RT.S(abs(mid - ob["px"]) * 100, v) * ob["rem"]) if live_b and ts_s - ob["t_place"] >= moas else 0.0
    qa = float(RT.S(abs(oa["px"] - mid) * 100, v) * oa["rem"]) if live_a and ts_s - oa["t_place"] >= moas else 0.0
    mine = RT.q_min(qb, qa, mid, RCFG)
    comp = RT.competitor_q(M.s["qb"][j], M.s["qa"][j], mid, RCFG)
    if mine > 0:
        usd = mine / (mine + comp) * rate / 1440.0
        out["samples_quoted"] += 1
        out["rewards"] += usd
        out["rewards_samples"] += 1
        # our quote improves the visible touch: we would set the price, and the recorded trades say nothing
        # about the flow that would then trade with us
        bb, ba = M.s["bb"][j], M.s["ba"][j]
        inside = (live_b and (not np.isfinite(bb) or ob["px"] > bb + EPS)) or \
                 (live_a and (not np.isfinite(ba) or oa["px"] < ba - EPS))
        if inside:
            out["rewards_inside"] += usd
            out["samples_inside"] += 1
        return usd
    return 0.0


# ----------------------------------------------------------------------------- P&L of fills
def fill_pnl(F: pd.DataFrame, markets: dict, t_end: dict) -> pd.DataFrame:
    if not len(F):
        return F
    cols = {f"mo_{h}": [] for h in HORIZONS} | {"mo_end": [], "spread": [], "trunc_1h": []}
    for r in F.itertuples():
        M = markets[r.slug]
        e = t_end[r.slug]
        mf = r.mid_f
        cols["spread"].append(r.side * (mf - r.px) * r.size)
        for h, dt in HORIZONS.items():
            cols[f"mo_{h}"].append(r.side * (M.mid_at(min(r.ts + dt, e)) - mf) * r.size)
        cols["trunc_1h"].append(r.ts + 3600 > e)
        cols["mo_end"].append(r.side * (M.mid_at(e) - mf) * r.size)
    for c, v in cols.items():
        F[c] = v
    return F


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=str(HERE / "rec"))
    ap.add_argument("--sel", default=str(HERE / "sel" / "selection.json"))
    ap.add_argument("--out", default=str(HERE / "baseline_out"))
    ap.add_argument("--start", default=None, help="UTC start (default: first recorded second)")
    ap.add_argument("--end", default=None, help="UTC end (default: last second covered by books and fills)")
    ap.add_argument("--max-gap", type=float, default=5.0, help="coverage gaps up to this are bridged (s)")
    ap.add_argument("--moas", type=float, default=30.0, help="minimum order age for scoring (s), reward_terms LIVE")
    ap.add_argument("--lag", type=int, default=3, help="block time minus match time (s): fills need block > placement + lag; mid_pre = mid at block - lag")
    ap.add_argument("--workers", type=int, default=3, help="concurrent data-api requests (I/O only)")
    ap.add_argument("--trades-cache", default=None, help="data-api cache dir (default <out>/cache/data_api)")
    ap.add_argument("--quick", action="store_true", help="base variant only (plus its comparators)")
    a = ap.parse_args(argv)
    t0 = time.time()
    rec, out = Path(a.rec), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    class _A:
        markets = None
        min_rate = 0
        top_n = 0
    mk, sel_full, sel_taken = BE.load_selection(Path(a.sel), _A)
    cfgs, reward_files = BE.load_reward_cfg(rec, sel_full, sel_taken)
    iv, seg = BE.coverage_segments(rec, a.max_gap)
    days = BE.recorded_days(rec)
    tokmap = {}
    for m in sel_full.itertuples():
        tokmap[m.tok0] = (m.slug, True, m.condition_id)
        tokmap[m.tok1] = (m.slug, False, m.condition_id)
    lo = BE.utc_ts(a.start) if a.start else int(min(s[0][0] for s in seg.values()))
    hi_req = BE.utc_ts(a.end) if a.end else int(time.time())
    print(f"{len(mk)} markets, days {days}, reward snapshots {len(reward_files)}, coverage for {len(seg)}", flush=True)
    D = load_books(rec, tokmap, cfgs, days, lo, hi_req)
    book_hi = int(D.ts.max())
    print(f"books: {len(D)} Yes-frame snapshots, {D.slug.nunique()} markets, up to {BE.iso(book_hi)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # fills (data-api by condition id, cached; complete up to fetch time - 120 s)
    used = mk[mk.slug.isin(set(D.slug))].reset_index(drop=True)
    need_to = {s: min(hi_req, book_hi) for s in used.slug}
    cache = Path(a.trades_cache) if a.trades_cache else out / "cache" / "data_api"
    trades, span = BE.trades_data_api(used, cache, need_to, a.workers, False)
    trades = trades.copy()
    trades["px_up"] = np.where(trades.outcome_idx == 0, trades.price.astype(float), 1.0 - trades.price.astype(float))
    has_maker = trades[~trades.is_taker].groupby("tx").size()
    trades["use"] = ~trades.is_taker | ~trades.tx.isin(has_maker.index)
    tr_by = {s: g.sort_values(["timestamp", "tx"], kind="stable") for s, g in trades.groupby("slug", sort=False)}
    fills_hi = int(min(span[s][1] for s in used.slug))
    hi = int(min(hi_req, book_hi, fills_hi))
    hi -= hi % 60
    print(f"trades: {len(trades)} legs; window {BE.iso(lo)} .. {BE.iso(hi)} ({(hi - lo) / 3600:.2f} h)  "
          f"({time.time() - t0:.0f}s)", flush=True)

    markets, sample_offs, t_end, skipped = {}, {}, {}, {}
    groups = {s: g for s, g in D.groupby("slug", sort=False)}
    for r in used.itertuples():
        e = min(hi, r.end_ts, BE.utc_ts("2026-11-03T00:00:00Z"))
        sp = span.get(r.slug)
        if sp is None or not np.isfinite(sp[1]):
            skipped[r.slug] = "no fills listing"
            continue
        if sp[0] > lo:
            skipped[r.slug] = "fills listing truncated"
            continue
        if r.slug not in seg or r.condition_id not in cfgs:
            skipped[r.slug] = "no coverage / config"
            continue
        M = Market(r, groups[r.slug], seg[r.slug], cfgs[r.condition_id], tr_by.get(r.slug), lo, e)
        markets[r.slug] = M
        t_end[r.slug] = e
        # build_elections' sample-second draw (seeded by condition id and window open)
        rng = np.random.default_rng([BE.LIQ_SEED, int(r.condition_id[2:14], 16), lo - lo % 86400])
        nmin = (86400 * 40) // 60
        offs = rng.integers(0, 60, nmin)
        k0 = ((lo + (60 - lo % 60) % 60) - (lo - lo % 86400)) // 60
        sample_offs[r.slug] = offs[k0:]
    print(f"simulating {len(markets)} markets (skipped {len(skipped)})", flush=True)

    variants = [(f, k) for f in FRACS for k in MULTS] if not a.quick else [BASE]
    runs = []
    for f, k in variants:
        runs.append((f"f{f:g}_x{k}", f, k, True, "back", a.moas, ""))
        runs.append((f"f{f:g}_x{k}_nofills", f, k, False, "back", a.moas, ""))
    for f, k in variants:
        runs.append((f"f{f:g}_x{k}_join", f, k, True, "back", a.moas, "J"))
    runs.append((f"f{BASE[0]:g}_x{BASE[1]}_frontq", BASE[0], BASE[1], True, "front", a.moas, ""))
    runs.append((f"f{BASE[0]:g}_x{BASE[1]}_moas0", BASE[0], BASE[1], True, "back", 0.0, ""))

    hours = (hi - lo) / 3600.0
    per_market, summary, all_fills = [], [], []
    blk_rows = []
    for name, f, k, fon, q, moas, mode in runs:
        rows, fl = [], []
        for slug, M in markets.items():
            o, fills = run_market(M, sample_offs[slug], f, k, fon, q, moas, a.lag, mode)
            for b_, u_ in o.pop("rew_blk").items():
                blk_rows.append((name, slug, int(b_), u_, 0.0))
            nti = o.pop("ntx_inside")
            o["ntx_inside_hit"] = nti.get("at_or_through", 0)
            o["ntx_inside_other"] = nti.get("other_side_px", 0)
            me = M.mid_at(t_end[slug])
            o.update(slug=slug, n_tx_window=M.n_tx, mid_end=me,
                     inv_end_notional=o["inv_end"] * me if o["inv_end"] > 0 else -o["inv_end"] * (1 - me),
                     rate=M.row.rate_per_day, min_size=M.row.rewards_min_size,
                     max_spread=M.row.rewards_max_spread, neg_risk=M.row.neg_risk)
            rows.append(o)
            fl += fills
        P = pd.DataFrame(rows)
        F = pd.DataFrame(fl, columns=["slug", "ts", "side", "px", "size", "mid_f"])
        F = fill_pnl(F, markets, t_end)
        for r in F.itertuples():
            blk_rows.append((name, r.slug, int(r.ts // BLOCK_S), 0.0, r.spread + r.mo_end))
        agg = F.groupby("slug")[["spread", "mo_1m", "mo_5m", "mo_1h", "mo_end", "size"]].sum() if len(F) else \
            pd.DataFrame(columns=["spread", "mo_1m", "mo_5m", "mo_1h", "mo_end", "size"])
        nf = F.groupby("slug").size() if len(F) else pd.Series(dtype=float)
        P = P.set_index("slug").join(agg).join(nf.rename("n_fills")).fillna(
            {c: 0.0 for c in ["spread", "mo_1m", "mo_5m", "mo_1h", "mo_end", "size", "n_fills"]}).reset_index()
        P = P.rename(columns={"size": "fill_shares"})
        P["net"] = P.rewards + P.spread + P.mo_end
        P["cov_min"] = P.minutes_quoted
        P["variant"] = name
        per_market.append(P)
        if len(F):
            F["variant"] = name
            all_fills.append(F)
        rew, cap = P.rewards.sum(), P.capital.sum()
        s = dict(variant=name, frac=f, size_mult=k, fills=fon, queue=q, moas=moas, hours=round(hours, 2),
                 markets=int(len(P)), markets_quoted=int((P.minutes_quoted > 0).sum()),
                 markets_rewarded=int((P.rewards > 0).sum()), markets_filled=int((P.n_fills > 0).sum()),
                 n_fills=int(P.n_fills.sum()), fill_shares=float(P.fill_shares.sum()),
                 fill_notional=float((F.px * F["size"]).sum()) if len(F) else 0.0,
                 rewards=rew, rewards_inside_touch=P.rewards_inside.sum(), spread=P.spread.sum(), mo_1m=P.mo_1m.sum(), mo_5m=P.mo_5m.sum(), mo_1h=P.mo_1h.sum(),
                 mo_end=P.mo_end.sum(), net=P.net.sum(), inv_end_abs_shares=float(P.inv_end.abs().sum()),
                 inv_end_notional=float(P.inv_end_notional.sum()),
                 net_behind_touch=P.net.sum() - P.rewards_inside.sum(),
                 rewards_per_day=rew / hours * 24, net_per_day=P.net.sum() / hours * 24, capital=cap,
                 rewards_per_day_per_capital=(rew / hours * 24 / cap) if cap > 0 else np.nan,
                 net_per_day_per_capital=(P.net.sum() / hours * 24 / cap) if cap > 0 else np.nan,
                 share_samples_quoted=P.samples_quoted.sum() / max(P.samples_scorable.sum(), 1),
                 frac_samples_mid_moved_by_us=P.mid_moved_by_us.sum() / max(P.samples_scorable.sum(), 1),
                 fills_horizon_1h_truncated=int(F.trunc_1h.sum()) if len(F) else 0)
        summary.append(s)
        print(f"  {name:22s} rewards {rew:9.2f}  spread {s['spread']:8.2f}  mo_end {s['mo_end']:8.2f}  net {s['net']:9.2f}"
              f"  fills {s['n_fills']:5d}  cap {cap:10.0f}  ({time.time() - t0:.0f}s)", flush=True)

    S_ = pd.DataFrame(summary)
    # ---- bootstrap CIs of $/hour and of net per $ capital per hour: 30-min clock blocks, and markets
    BK = pd.DataFrame(blk_rows, columns=["variant", "slug", "blk", "rew", "fill_pnl"])
    blks = np.arange(lo // BLOCK_S, (hi - 1) // BLOCK_S + 1)
    dur = np.array([min(hi, (b + 1) * BLOCK_S) - max(lo, b * BLOCK_S) for b in blks]) / 3600.0
    rng = np.random.default_rng(20261016)
    NB = 4000
    it = rng.integers(0, len(blks), (NB, len(blks)))
    PMall = pd.concat(per_market, ignore_index=True)
    ci = []
    for name in S_.variant:
        P = PMall[PMall.variant == name].set_index("slug")
        cap = P.capital.sum()
        b = BK[BK.variant == name]
        row = {"variant": name, "capital": cap, "block_hours": list(np.round(dur, 3))}
        for what, col in (("rewards", b.rew), ("net", b.rew + b.fill_pnl)):
            per_blk = col.groupby(b.blk).sum().reindex(blks, fill_value=0.0).values
            rate = per_blk.sum() / dur.sum()
            bt = per_blk[it].sum(1) / dur[it].sum(1)
            pm = (P.rewards if what == "rewards" else P.net).values
            mi = rng.integers(0, len(P), (NB, len(P)))
            bm = pm[mi].sum(1) / hours / np.maximum(P.capital.values[mi].sum(1), 1e-9)
            row.update({f"{what}_per_h": rate, f"{what}_per_h_blk_lo": np.percentile(bt, 2.5),
                        f"{what}_per_h_blk_hi": np.percentile(bt, 97.5),
                        f"{what}_per_h_per_cap_pct": 100 * rate / cap if cap > 0 else np.nan,
                        f"{what}_per_h_per_cap_pct_blk_lo": 100 * np.percentile(bt, 2.5) / cap if cap > 0 else np.nan,
                        f"{what}_per_h_per_cap_pct_blk_hi": 100 * np.percentile(bt, 97.5) / cap if cap > 0 else np.nan,
                        f"{what}_per_h_per_cap_pct_mkt_lo": 100 * np.percentile(bm, 2.5),
                        f"{what}_per_h_per_cap_pct_mkt_hi": 100 * np.percentile(bm, 97.5),
                        f"{what}_blocks": list(np.round(per_blk / dur, 3))})
        ci.append(row)
    CI = pd.DataFrame(ci)
    CI.to_csv(HERE / "baseline_ci.csv", index=False, float_format="%.6g")
    CI_small = CI.drop(columns=[c for c in CI.columns if c.endswith("blocks") or c == "block_hours"])
    S_ = S_.merge(CI_small.drop(columns=["capital"]), on="variant", how="left")
    S_ = pd.concat([pd.DataFrame([dict(variant="no_quote", hours=round(hours, 2), markets=len(markets), rewards=0.0,
                                       spread=0.0, mo_1m=0.0, mo_5m=0.0, mo_1h=0.0, mo_end=0.0, net=0.0, n_fills=0,
                                       capital=0.0, rewards_per_day=0.0, net_per_day=0.0)]), S_], ignore_index=True)
    PM_ = pd.concat(per_market, ignore_index=True)
    PM_["rate_bucket"] = pd.cut(PM_.rate, RATE_BUCKETS, right=False).astype(str)
    byb = PM_.groupby(["variant", "rate_bucket"]).agg(
        markets=("slug", "size"), rate_sum=("rate", "sum"), rewards=("rewards", "sum"), spread=("spread", "sum"),
        mo_5m=("mo_5m", "sum"), mo_end=("mo_end", "sum"), net=("net", "sum"), n_fills=("n_fills", "sum"),
        capital=("capital", "sum")).reset_index()
    byb["net_per_day"] = byb.net / hours * 24
    byb["rewards_per_day_per_capital"] = byb.rewards / hours * 24 / byb.capital.replace(0, np.nan)
    S_.to_csv(HERE / "baseline_summary.csv", index=False, float_format="%.6g")
    byb.to_csv(out / "by_rate_bucket.csv", index=False, float_format="%.6g")
    PM_.to_csv(out / "per_market.csv", index=False, float_format="%.6g")
    if all_fills:
        pd.concat(all_fills, ignore_index=True).to_csv(out / "fills.csv", index=False, float_format="%.6g")
    pools = {}
    selc = set(mk.condition_id)
    for f_ in reward_files:
        r_ = pd.read_parquet(rec / f_, columns=["taken_at", "condition_id", "in_program", "total_daily_rate"])
        r_ = r_[r_.condition_id.isin(selc)]
        pools[str(r_.taken_at.iloc[0])] = dict(selected_pool_per_day=float(r_.total_daily_rate.where(r_.in_program, 0).sum()),
                                               in_program=int(r_.in_program.sum()))
    meta = dict(reward_pool_by_snapshot=pools, bootstrap=dict(block_s=BLOCK_S, n=NB, blocks=len(blks)),
                window=[BE.iso(lo), BE.iso(hi)], hours=hours, markets_simulated=len(markets), skipped=skipped,
                books_up_to=BE.iso(book_hi), fills_complete_to=BE.iso(fills_hi), reward_snapshots=reward_files,
                trades_legs=int(len(trades)),
                taker_txs_in_window=int(sum(M.n_tx for M in markets.values())),
                markets_with_trades_in_window=int(sum(M.n_tx > 0 for M in markets.values())), moas=a.moas, lag_s=a.lag, inv_cap_mult=INV_CAP_MULT, fracs=FRACS, mults=MULTS,
                built_at=pd.Timestamp.now("UTC").isoformat(), runtime_s=round(time.time() - t0, 1),
                build_elections_import="ok (BE.load_selection/load_reward_cfg/cfg_at/coverage_segments/reward_stats/"
                                       "side_ladder/trades_data_api/LIQ_SEED; reward_terms.S/q_min/competitor_q)")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (HERE / "baseline_summary.json").write_text(json.dumps(dict(meta=meta, summary=S_.to_dict("records")),
                                                           indent=1, default=float))
    print(S_.drop(columns=["queue"], errors="ignore").to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
