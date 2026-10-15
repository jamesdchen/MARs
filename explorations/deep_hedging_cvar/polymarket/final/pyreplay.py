"""Python stand-in replay env for dry runs and tests. NEVER a number of record.

It implements the *stepper interface* that adapters.CEnvBackend asks of the C env
(polymarket/env.py + mm.h), on the same data/build files, so the final-test pipeline
(run_test.py -> report.py) can be exercised end to end on val (or on the synthetic tiny
dataset of synth.py) before the C env exists. Its fill engine is a simplified version of
EVALUATION.md T1, not the C env's:

* Action MultiDiscrete([7, 7, 3]): price choice {off, improve 2, improve 1, join, 1 behind,
  2 behind, 4 behind} ticks relative to the current best quote on that side (improvement capped
  at spread - 1 tick: never locks or crosses; a missing side is off), size {1, 2, 5} x
  min_order_size x the variant's size_mult.
* Latency L: the decision made on row k is live for prints matched in seconds k+1+L ..;
  cancels take effect with the same delay (the live quote is simply the decision of row s-1-L).
* Trade-time alignment: a maker leg printed at block row r is matched at r - d with
  d = 0 / 3.5 / 5 s (optimistic / central / pessimistic): processed in second s = r - floor(d).
* optimistic: prints at or through our price fill us, full leg size, any book.
  central: through our price fully; at our price only after the displayed depth at that price
  in the snapshot (queue reset every second); no fills on a forward-filled (crossed_raw) book;
  our fills <= 25% of taker volume per market-minute.
  pessimistic: strictly through only; 50% of the central size capped at 20% of the leg; no
  fills on crossed or stale (> 10 s) books.
* Maker fee 0; rebate = maker_rebate_rate * taker_fee_rate * qty * p (1 - p) at fill time.
  Liquidity rewards are 0 here (reward_cum stays 0).
* mirror=True replays the Down token as "Up": book p -> 1 - p with sides swapped, print
  sides negated, observation features mirrored by MIRROR_RULES, settlement 1 - winner_up.

Observation: feat.bin's observation block, then inventory/50, best_bid, best_ask,
seconds_to_close/14400, contract_len_s/14400, p_model, book_valid, min_order_size/5,
const_one (obs_spec() documents it for the baselines).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from common import CODE_ROOT, FILL_MODELS, Dataset

sys.path.insert(0, str(CODE_ROOT / "baselines"))

PRICE_OFFSETS = [None, 2, 1, 0, -1, -2, -4]
SIZE_MULTS = [1, 2, 5]
TRADE_DELAY = {0: 0.0, 1: 3.5, 2: 5.0}
EXTRA = [("inventory", 50.0), ("best_bid", 1.0), ("best_ask", 1.0), ("seconds_to_close", 14400.0),
         ("contract_len_s", 14400.0), ("p_model", 1.0), ("book_valid", 1.0),
         ("min_order_size", 5.0), ("const_one", 1.0)]
# how each observed feature transforms under Up <-> Down (unknown names: mirror refused)
MIRROR_RULES = {
    "p_logit": "neg", "mark_logit": "neg", "gap_c": "neg", "dgap5_c": "neg", "z": "neg",
    "sdv_bp": "same", "sigma_1m_bp": "same", "tau_frac": "same", "log_tau": "same", "in_twap": "same",
    "delta_usd": "same", "spread_ticks": "same", "bid_present": "swap:ask_present",
    "ask_present": "swap:bid_present", "imb_l1": "neg", "imb_l5": "neg", "bid_sz0": "swap:ask_sz0",
    "ask_sz0": "swap:bid_sz0", "flow5": "neg", "flow30": "neg", "flow120": "neg", "mret5_c": "neg",
    "mret30_c": "neg", "mret120_c": "neg", "ulz5": "neg", "ulz30": "neg", "ulz120": "neg",
    "len_5m": "same", "len_15m": "same", "len_4h": "same", "tick_fine": "same", "book_age": "same",
    "crossed_raw": "same", "mark_is_fv": "same", "gross120": "same",
}


class ReplayEnv:
    """B markets in parallel, each from its row 0 (start offset 0), then idle."""

    of_record = False

    def __init__(self, ds: Dataset, markets, fill_model: int = 1, latency: int = 1,
                 size_mult: float = 1.0, mirror: bool = False, split: int | None = None,
                 allow_test: bool = False):
        self.ds = ds
        self.mk = np.asarray(markets, dtype=np.int64)
        recs = ds.markets[self.mk]
        if split is not None and np.any(recs["split"] != split):
            raise ValueError("markets outside the requested split")
        if np.any(recs["split"] == 2) and not allow_test:
            raise PermissionError("split 2 (test) needs allow_test (run_test.py with MM_FINAL=1)")
        self.B = len(self.mk)
        self.fm, self.L = int(fill_model), int(latency)
        self.D = int(math.floor(TRADE_DELAY[self.fm]))
        self.size_mult, self.mirror = float(size_mult), bool(mirror)
        self.off = recs["row_offset"].astype(np.int64)
        self.n = recs["n_rows"].astype(np.int64)
        self.T = recs["contract_len_s"].astype(np.float64)
        self.minsz = recs["min_order_size"].astype(np.float64)
        self.reb_rate = (recs["maker_rebate_rate"] * recs["taker_fee_rate"]).astype(np.float64)
        self.open_ts = recs["open_ts"].astype(np.int64)
        c = ds.ci
        self._tc = [c[f"bid_px_{i}"] for i in range(5)] + [c[f"bid_sz_{i}"] for i in range(5)] \
            + [c[f"ask_px_{i}"] for i in range(5)] + [c[f"ask_sz_{i}"] for i in range(5)] \
            + [c["seconds_to_close"], c["book_valid"], c["crossed_raw"], c["book_age"]]
        self.obs_names = ds.obs_feature_names
        self.n_feat = len(self.obs_names)
        self.obs_size = self.n_feat + len(EXTRA)
        self.j_pm, self.j_tick = ds.fcol("p_model"), ds.fcol("tick_fine")
        if self.mirror:
            bad = [n for n in self.obs_names if n not in MIRROR_RULES]
            if bad:
                raise NotImplementedError(f"no mirror rule for observed features {bad}")
            self._msign = np.array([-1.0 if MIRROR_RULES[n] == "neg" else 1.0 for n in self.obs_names])
            self._mperm = np.array([self.obs_names.index(MIRROR_RULES[n][5:])
                                    if MIRROR_RULES[n].startswith("swap:") else i
                                    for i, n in enumerate(self.obs_names)], dtype=np.int64)

    # ------------------------------------------------------------------ interface
    def obs_spec(self):
        from obs_spec import ObsSpec
        feats = {n: {"index": i} for i, n in enumerate(self.obs_names)}
        for k, (n, sc) in enumerate(EXTRA):
            feats[n] = {"index": self.n_feat + k, "scale": sc}
        return ObsSpec(feats, meta={"obs_size": self.obs_size, "min_order_size": 5.0,
                                    "actions": {"price_offsets_ticks": PRICE_OFFSETS,
                                                "size_mults": SIZE_MULTS}},
                       source="pyreplay")

    def reset(self):
        B = self.B
        self.k = np.zeros(B, np.int64)
        self.active = np.ones(B, bool)
        self.cash = np.zeros(B)
        self.inv = np.zeros(B)
        self.reb = np.zeros(B)
        self.rew = np.zeros(B)
        self.fee = np.zeros(B)
        # decision ring: [L+1, B] desired prices (NaN = off) and sizes
        self.dec_bid = np.full((self.L + 1, B), np.nan)
        self.dec_ask = np.full((self.L + 1, B), np.nan)
        self.dec_sz = np.zeros((self.L + 1, B))
        self.live_bid = np.full(B, np.nan)
        self.live_ask = np.full(B, np.nan)
        self.minute = np.full(B, -1, np.int64)
        self.min_taker = np.zeros(B)
        self.min_ours = np.zeros(B)
        self._fills = []
        self._settle = np.asarray(self.ds.markets["winner_up"][self.mk], dtype=np.float64)
        if self.mirror:
            self._settle = 1.0 - self._settle
        return self._obs()

    def settlement(self):
        return self._settle.copy()

    def trace(self) -> dict:
        row = np.where(self.active, self.off + self.k, -1)
        return dict(active=self.active.copy(), row=row, cash=self.cash.copy(), inv=self.inv.copy(),
                    reward_cum=self.rew.copy(), rebate_cum=self.reb.copy(), fee_cum=self.fee.copy(),
                    bid_live=np.isfinite(self.live_bid), ask_live=np.isfinite(self.live_ask),
                    bid_px=self.live_bid.copy(), ask_px=self.live_ask.copy())

    def drain_fills(self) -> dict:
        f = self._fills
        self._fills = []
        keys = ("env", "row", "side", "qty", "price", "fee", "rebate", "print_px")
        if not f:
            return {k: np.zeros(0) for k in keys}
        a = np.array(f, dtype=np.float64)
        out = {k: a[:, i] for i, k in enumerate(keys)}
        out["env"] = out["env"].astype(np.int64)
        out["row"] = out["row"].astype(np.int64)
        out["side"] = out["side"].astype(np.int64)
        return out

    # ------------------------------------------------------------------ internals
    def _book(self, rows):
        """Up-frame (or mirrored) book arrays for global rows (B,)."""
        t = np.asarray(self.ds.ticks[rows][:, self._tc], dtype=np.float64)
        bp, bs, ap, as_ = t[:, 0:5], t[:, 5:10], t[:, 10:15], t[:, 15:20]
        stc, valid, crossed, age = t[:, 20], t[:, 21] > 0.5, t[:, 22] > 0.5, t[:, 23]
        if self.mirror:
            bp, ap = 1.0 - ap, 1.0 - bp
            bs, as_ = as_, bs
        return dict(bp=bp, bs=bs, ap=ap, as_=as_, stc=stc, valid=valid, crossed=crossed, age=age)

    def _tick(self, rows, bk):
        if self.j_tick is not None:
            fine = np.asarray(self.ds.feat[rows, self.j_tick]) > 0.5
        else:
            fine = (bk["bp"][:, 0] > 0) & (bk["bp"][:, 0] < 0.04) | (bk["ap"][:, 0] < 1) & (bk["ap"][:, 0] > 0.96)
        return np.where(fine, 0.001, 0.01)

    def _obs(self):
        rows = self.off + np.minimum(self.k, self.n - 1)
        bk = self._book(rows)
        o = np.zeros((self.B, self.obs_size), np.float32)
        if self.n_feat:
            f = np.asarray(self.ds.feat[rows, : self.n_feat], dtype=np.float64)
            if self.mirror:
                f = f[:, self._mperm] * self._msign
            o[:, : self.n_feat] = f
        pm = np.asarray(self.ds.feat[rows, self.j_pm], np.float64) if self.j_pm is not None else \
            np.full(self.B, 0.5)
        if self.mirror and self.j_pm is not None:
            pm = 1.0 - pm
        ex = [self.inv / 50.0, bk["bp"][:, 0], bk["ap"][:, 0], bk["stc"] / 14400.0, self.T / 14400.0,
              pm, bk["valid"].astype(float), self.minsz / 5.0, np.ones(self.B)]
        o[:, self.n_feat:] = np.stack(ex, 1)
        o[~self.active] = 0.0
        self._last_book = bk
        self._last_rows = rows
        return o

    def _decide(self, actions):
        """Desired prices of the decision made on the current row (NaN = off)."""
        a = np.asarray(actions, dtype=np.int64)
        bk, rows = self._last_book, self._last_rows
        tick = self._tick(rows, bk)
        bb, ba = bk["bp"][:, 0], bk["ap"][:, 0]
        bid_ok = bk["valid"] & (bb > 1e-9)
        ask_ok = bk["valid"] & (ba < 1 - 1e-9)
        offs = np.array([np.nan if o is None else o for o in PRICE_OFFSETS], dtype=np.float64)
        ob, oa = offs[np.clip(a[:, 0], 0, 6)], offs[np.clip(a[:, 1], 0, 6)]
        spread_t = np.where(bid_ok & ask_ok, np.round((ba - bb) / tick), np.inf)
        cap = np.maximum(spread_t - 1, 0)
        ob = np.minimum(ob, cap)
        oa = np.minimum(oa, cap)
        bid = np.where(bid_ok & np.isfinite(ob), bb + ob * tick, np.nan)
        ask = np.where(ask_ok & np.isfinite(oa), ba - oa * tick, np.nan)
        bid = np.where(bid > 1e-9, np.round(bid / tick) * tick, np.nan)
        ask = np.where(ask < 1 - 1e-9, np.round(ask / tick) * tick, np.nan)
        sz = np.asarray(SIZE_MULTS, float)[np.clip(a[:, 2], 0, 2)] * self.minsz * self.size_mult
        return bid, ask, sz

    def step(self, actions):
        bid, ask, sz = self._decide(actions)
        self.dec_bid = np.roll(self.dec_bid, 1, axis=0)
        self.dec_ask = np.roll(self.dec_ask, 1, axis=0)
        self.dec_sz = np.roll(self.dec_sz, 1, axis=0)
        self.dec_bid[0], self.dec_ask[0], self.dec_sz[0] = bid, ask, sz
        was_active = self.active.copy()
        last = self.k >= self.n - 1
        self.active &= ~last
        self.k = np.where(self.active, self.k + 1, self.k)
        # live quote for second s = k: decision of row s-1-L = ring slot L
        self.live_bid = np.where(self.active, self.dec_bid[self.L], np.nan)
        self.live_ask = np.where(self.active, self.dec_ask[self.L], np.nan)
        live_sz = self.dec_sz[self.L]
        idx = np.flatnonzero(self.active)
        if idx.size and self.ds.fills is not None:
            self._match(idx, live_sz)
        _ = was_active
        return self._obs()

    def _match(self, idx, live_sz):
        ds = self.ds
        s_rows = self.off[idx] + self.k[idx]                       # match second (global row)
        r_rows = s_rows + self.D                                   # block row of the prints
        ok = r_rows < self.off[idx] + self.n[idx]
        bk = self._book(s_rows)
        minute = (self.open_ts[idx] + self.k[idx]) // 60
        new_min = minute != self.minute[idx]
        self.minute[idx] = minute
        self.min_taker[idx[new_min]] = 0.0
        self.min_ours[idx[new_min]] = 0.0
        fi = ds.fill_index
        for j, b in enumerate(idx):
            if not ok[j]:
                continue
            lo, hi = int(fi[r_rows[j]]), int(fi[r_rows[j] + 1])
            if lo == hi:
                continue
            legs = ds.fills[lo:hi]
            rem = {+1: live_sz[b] if np.isfinite(self.live_ask[b]) else 0.0,     # our ask (sell)
                   -1: live_sz[b] if np.isfinite(self.live_bid[b]) else 0.0}    # our bid (buy)
            queue_used = {+1: 0.0, -1: 0.0}
            crossed, stale = bk["crossed"][j], bk["age"][j] > 10.0
            for leg in legs:
                tside = int(leg["side"])
                p = float(leg["price"])
                if self.mirror:
                    tside, p = -tside, 1.0 - p
                size = float(leg["size"])
                self.min_taker[b] += size
                ours = self.live_ask[b] if tside == +1 else self.live_bid[b]
                if not np.isfinite(ours) or rem[tside] <= 1e-12:
                    continue
                through = (ours < p - 1e-9) if tside == +1 else (ours > p + 1e-9)
                at = abs(ours - p) < 1e-9
                if self.fm == 0:
                    q = size if (through or at) else 0.0
                else:
                    if crossed or (self.fm == 2 and stale):
                        continue
                    if through:
                        q = size
                    elif at and self.fm == 1:
                        lvl_p = bk["ap"][j] if tside == +1 else bk["bp"][j]
                        lvl_s = bk["as_"][j] if tside == +1 else bk["bs"][j]
                        hit = np.abs(lvl_p - ours) < 1e-9
                        depth = float(lvl_s[hit].sum()) if hit.any() else 0.0
                        ahead = max(depth - queue_used[tside], 0.0)
                        q = max(size - ahead, 0.0)
                        queue_used[tside] += min(size, ahead)
                    else:
                        q = 0.0
                    if q > 0:
                        cap = max(0.25 * self.min_taker[b] - self.min_ours[b], 0.0)
                        q = min(q, cap)
                    if self.fm == 2:
                        q = min(0.5 * q, 0.2 * size)
                q = min(q, rem[tside])
                if q <= 1e-12:
                    continue
                rem[tside] -= q
                self.min_ours[b] += q
                our_side = -tside                                  # taker buys -> we sell
                rebate = self.reb_rate[b] * q * ours * (1.0 - ours)
                self.cash[b] += -our_side * q * ours + rebate
                self.inv[b] += our_side * q
                self.reb[b] += rebate
                self._fills.append((b, s_rows[j], our_side, q, ours, 0.0, rebate, p))


def make_env(ds: Dataset, markets, variant: dict, split: int, allow_test: bool):
    return ReplayEnv(ds, markets, fill_model=FILL_MODELS[variant["fill_model"]],
                     latency=int(variant["latency_s"]), size_mult=float(variant["size_mult"]),
                     mirror=bool(variant.get("mirror", False)), split=split, allow_test=allow_test)
