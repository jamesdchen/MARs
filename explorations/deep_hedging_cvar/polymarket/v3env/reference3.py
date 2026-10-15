"""Readable numpy/Python reference of the v3 replay env (mm3.h). Used only by the tests.

It implements ENV3.md directly, one episode at a time and one row at a time, in float64, on the
same files the C core reads (the five v2 files plus the optional markets3.bin / x3.bin). It is a
copy of ../reference.py (the v2 reference, frozen) with the v3 extensions written out:
per-market episode length, float settlement (settle_mode), flattening cost, close-inventory
shaping, quote pull before the event, variable dt, per-market / per-row reward program, size heads
relative to the reward min size, per-category fees and the extra observation slots.

    from reference3 import Data3, Config3, Episode3
    data = Data3(build_dir)
    ep = Episode3(data, Config3(obs_size=80, liq_rewards=1), market=3, start=0)
    obs = ep.obs()
    while not done: obs, reward, done = ep.step((3, 3, 0))
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "data" / "build"

OPT, CENTRAL, PESS = 0, 1, 2
OFFSETS = [None, 2, 1, 0, -1, -2, -4]
SIZES = [1.0, 2.0, 5.0]
DELAY_S = {OPT: 0.0, CENTRAL: 3.5, PESS: 5.0}
BOOK_VALID, BOOK_CROSSED, BOOK_STALE = 1, 2, 4
LEGACY_V, LEGACY_MIN = 1.5, 50.0
BAND = (0.10, 0.90)
REWARD_C = 3.0
N_FEAT_OBS, OBS_V2, NX = 33, 50, 28
SETTLE_WINNER, SETTLE_PRICE, SETTLE_LAST_MARK = 0, 1, 2
CAT_CRYPTO, CAT_SPORTS, CAT_ELECTIONS, CAT_OTHER = 0, 1, 2, 3
FLAG3_LEGACY = 1
X_COLS = ["rw_v_c", "rw_min_size", "rw_rate_day", "rw_in_program", "liq_mid", "liq_comp", "liq_pi", "pad"]
FLAGS = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside", "inspread"]

BOOK_DTYPE = np.dtype([("bid_px", "<i2", 5), ("ask_px", "<i2", 5), ("bid_sz", "<f4", 5),
                       ("ask_sz", "<f4", 5), ("flags", "u1"), ("tick", "u1"), ("tick_alt", "u1"),
                       ("same_run", "u1")])
FILL_DTYPE = np.dtype([("row", "<i4"), ("side", "i1"), ("first", "u1"), ("px", "<i2"),
                       ("price", "<f4"), ("size", "<f4"), ("tx_size", "<f4")])
MARKET_DTYPE = np.dtype([("row_offset", "<i8"), ("open_ts", "<i8"), ("end_ts", "<i8"), ("n_rows", "<i4"),
                         ("contract_len_s", "<i4"), ("coin_id", "<i4"), ("winner_up", "<i4"), ("split", "<i4"),
                         ("flags", "<i4"), ("tick_size", "<f4"), ("taker_fee_rate", "<f4"),
                         ("fee_exponent", "<f4"), ("min_order_size", "<f4"), ("rewards_min_size", "<f4"),
                         ("rewards_max_spread", "<f4"), ("log_strike_vs_ul_open", "<f4"),
                         ("maker_rebate_rate", "<f4"), ("market_idx", "<i4"), ("tick_size_final", "<f4")])
EXT_DTYPE = np.dtype([("settle_price", "<f8"), ("event_ts", "<i8"), ("settle_mode", "<i4"), ("category", "<i4"),
                      ("row_s", "<f4"), ("rw_v_c", "<f4"), ("rw_min_size", "<f4"), ("rw_rate_day", "<f4"),
                      ("fee_rate", "<f4"), ("fee_exp", "<f4"), ("rebate_rate", "<f4"), ("flags3", "<i4"),
                      ("pad", "<i4", 2)])
assert BOOK_DTYPE.itemsize == 64 and FILL_DTYPE.itemsize == 20
assert MARKET_DTYPE.itemsize == 88 and EXT_DTYPE.itemsize == 64


@dataclass
class Config3:
    """Same names and defaults as M3_PARAM_NAMES / M3_DEFAULTS in mm3.h (test_env3.py checks)."""
    split: float = 0
    allow_test: float = 0
    fill_model: float = CENTRAL
    latency: float = 1
    match_delay: float = -1
    dt: float = 0
    dt_5m: float = 1
    dt_15m: float = 1
    dt_4h: float = 5
    Q: float = 100
    inv_penalty: float = 0.001
    reward_scale: float = 0
    size_mult: float = 1
    part_cap: float = -1
    fee_rate: float = -1
    rebate_rate: float = -1
    liq_rewards: float = 0
    liq_moas: float = 1
    mark_mode: float = 0
    reward_mode: float = 0
    tick_rule: float = 0
    len_mask: float = 15
    fixed_market: float = -1
    fixed_start: float = -1
    cycle_stride: float = 0
    start_frac: float = 0.2
    markout_h: float = 60
    # v3
    dt_other: float = 5            # decision interval for lengths other than 300 / 900 / 14400 rows
    dt_far: float = 0              # > 0: decision interval while more than far_s s remain to the event
    far_s: float = 7200
    cat_mask: float = 15           # 1 crypto, 2 sports, 4 elections, 8 other
    obs_size: float = 50           # 50 = v2 observation; up to 128 (50 + 28 extras + zero pad)
    size_mode: float = 0           # 0 v2 sizes; 1 size_rel{k} x max(reward min size, min_order_size)
    size_rel0: float = 1
    size_rel1: float = 2
    size_rel2: float = 4
    liq_v_c: float = -1            # > 0 overrides the reward max spread (cents)
    liq_min_size: float = -1       # > 0 overrides the reward min size (shares)
    settle_mode: float = -1        # >= 0 overrides markets3 settle_mode
    flat_c: float = 0              # flattening cost (cents per share) at a mode 1 / 2 settlement
    pull_s: float = 0              # > 0: quotes forced off within pull_s s of the event
    close_inv_coef: float = 0
    close_ramp: float = 0.2
    close_ramp_s: float = 0
    fee_rate_crypto: float = -1
    rebate_rate_crypto: float = -1
    fee_rate_sports: float = -1
    rebate_rate_sports: float = -1
    fee_rate_elections: float = -1
    rebate_rate_elections: float = -1

    def values(self):
        return np.array([float(getattr(self, f.name)) for f in fields(self)])

    @staticmethod
    def names():
        return [f.name for f in fields(Config3)]


class Data3:
    """The five v2 files plus markets3.bin / x3.bin when present (legacy records otherwise)."""

    def __init__(self, build=BUILD):
        build = Path(build)
        self.dir = build
        fmeta = json.loads((build / "feat_meta.json").read_text())
        self.markets = np.fromfile(build / "markets.bin", dtype=MARKET_DTYPE)
        self.N, self.G = int(fmeta["N"]), int(fmeta["G"])
        assert self.G == 44, "mm3 reads the 44-column feat.bin (split layout)"
        self.col = fmeta["column_index"]
        self.feat = np.memmap(build / "feat.bin", np.float32, "r", shape=(self.N, self.G))
        self.book = np.memmap(build / "book.bin", BOOK_DTYPE, "r", shape=(self.N,))
        self.fills = np.fromfile(build / "fills.bin", FILL_DTYPE)
        self.fill_index = np.fromfile(build / "fill_index.bin", np.int32)
        assert self.fill_index.size == self.N + 1
        if (build / "markets3.bin").exists():
            self.ext = np.fromfile(build / "markets3.bin", dtype=EXT_DTYPE)
            assert len(self.ext) == len(self.markets)
        else:
            self.ext = legacy_ext(self.markets)
        self.x3 = (np.memmap(build / "x3.bin", np.float32, "r", shape=(self.N, len(X_COLS)))
                   if (build / "x3.bin").exists() else None)


def legacy_ext(markets):
    """markets3 records mm3.h synthesises for v2 data (no markets3.bin)."""
    e = np.zeros(len(markets), EXT_DTYPE)
    res = markets["winner_up"] >= 0
    e["settle_price"] = np.where(res, markets["winner_up"], 0)
    e["event_ts"] = markets["end_ts"]
    e["settle_mode"] = np.where(res, SETTLE_WINNER, SETTLE_LAST_MARK)
    e["category"] = CAT_CRYPTO
    e["row_s"] = 1.0
    e["rw_v_c"] = LEGACY_V
    e["rw_min_size"] = LEGACY_MIN
    e["fee_rate"] = e["fee_exp"] = e["rebate_rate"] = -1.0
    e["flags3"] = FLAG3_LEGACY
    return e


def len_class(n):
    return {300: 0, 900: 1, 14400: 2}.get(int(n), 3)


def score(d, v):
    """Polymarket's per-share score ((v - d) / v)^2 for d < v, else 0."""
    return (v - d) / v * ((v - d) / v) if d < v else 0.0


@dataclass
class Order:
    live: bool = False
    px: int = 0
    rem: float = 0.0
    queue: float = 0.0
    shown: float = -1.0
    born: int = 0


def shown_size(book, side, px):
    prices = book["bid_px"] if side == 0 else book["ask_px"]
    sizes = book["bid_sz"] if side == 0 else book["ask_sz"]
    for k in range(5):
        p = int(prices[k])
        if p == px:
            return float(sizes[k])
        if (side == 0 and p < px) or (side == 1 and p > px):
            return 0.0
    return -1.0


class Episode3:
    def __init__(self, data: Data3, cfg: Config3, market: int, start: int):
        self.d, self.cfg = data, cfg
        m, e = data.markets[market], data.ext[market]
        self.market = market
        self.r0 = int(m["row_offset"])
        self.r_end = self.r0 + int(m["n_rows"]) - 1
        start = min(max(start, 0), int(m["n_rows"]) - 2)
        self.s0 = self.t = self.r0 + start
        self.len = int(m["n_rows"]) - 1                 # episode length in rows (from the data)
        self.len_s = float(self.len)
        self.open_ts = int(m["open_ts"])
        self.min_size = float(m["min_order_size"])
        self.category = int(e["category"])
        self.legacy = bool(int(e["flags3"]) & FLAG3_LEGACY)
        # fees: global override > per-category override > markets3 > markets.bin
        fr, rr = float(m["taker_fee_rate"]), float(m["maker_rebate_rate"])
        if e["fee_rate"] >= 0:
            fr = float(e["fee_rate"])
        if e["rebate_rate"] >= 0:
            rr = float(e["rebate_rate"])
        cat_name = {0: "crypto", 1: "sports", 2: "elections"}.get(self.category)
        if cat_name and getattr(cfg, f"fee_rate_{cat_name}") >= 0:
            fr = float(getattr(cfg, f"fee_rate_{cat_name}"))
        if cat_name and getattr(cfg, f"rebate_rate_{cat_name}") >= 0:
            rr = float(getattr(cfg, f"rebate_rate_{cat_name}"))
        self.fee_rate = cfg.fee_rate if cfg.fee_rate >= 0 else fr
        self.rebate_rate = cfg.rebate_rate if cfg.rebate_rate >= 0 else rr
        self.fee_exp = float(e["fee_exp"]) if e["fee_exp"] >= 0 else float(m["fee_exponent"])
        # settlement
        mode = int(cfg.settle_mode) if cfg.settle_mode >= 0 else int(e["settle_mode"])
        if mode == SETTLE_WINNER and int(m["winner_up"]) < 0:
            mode = SETTLE_LAST_MARK
        if mode == SETTLE_PRICE and self.legacy:
            mode = SETTLE_LAST_MARK
        self.settle_mode = mode
        if mode == SETTLE_WINNER:
            self.settle = float(int(m["winner_up"]))
        elif mode == SETTLE_PRICE:
            self.settle = float(e["settle_price"])
        else:
            self.settle = float(data.feat[self.r_end, data.col["mark"]])
        self.flat = 0.0 if mode == SETTLE_WINNER else cfg.flat_c / 100.0
        self.row_s = float(e["row_s"])
        ev = int(e["event_ts"]) if int(e["event_ts"]) > 0 else int(m["end_ts"])
        self.ev_off = float(ev - self.open_ts)
        self.rw_v, self.rw_min, self.rw_rate = float(e["rw_v_c"]), float(e["rw_min_size"]), float(e["rw_rate_day"])
        model = int(cfg.fill_model)
        self.model = model
        self.L = int(cfg.latency)
        delay = cfg.match_delay if cfg.match_delay >= 0 else DELAY_S[model]
        self.D = math.ceil(delay + 0.5)
        if cfg.dt > 0:
            self.dt = int(cfg.dt)
        else:
            self.dt = int([cfg.dt_5m, cfg.dt_15m, cfg.dt_4h, cfg.dt_other][len_class(self.len)])
        self.part_cap = cfg.part_cap if cfg.part_cap >= 0 else (0.0 if model == OPT else 0.25)
        self.scale = cfg.reward_scale if cfg.reward_scale > 0 else 1.0 / cfg.Q
        self.orders = [Order(), Order()]
        self.pending = []
        self.inv = self.cash = self.rebates = self.liq = self.wealth = self.penalty = 0.0
        self.close_pen = self.flat_cost = 0.0
        self.rw_samples = self.rw_scored = 0
        self.rw_share = 0.0
        self.spread = 0.0
        self.markout = {5: 0.0, 30: 0.0, 60: 0.0}
        self.minute = None
        self.minute_vol = self.minute_fill = self.tx_fill = 0.0
        self.credit = 0.0
        self.credit_total = 0.0
        self.n_fills = 0
        self.fill_vol = self.buy_vol = self.sell_vol = self.max_inv = 0.0
        self.flags = dict.fromkeys(FLAGS, 0.0)
        self.flag_pnl = dict.fromkeys(FLAGS, 0.0)
        self.rows = self.rows_two = self.rows_any = 0
        self.steps = self.clips = self.rejects = 0
        self.ret = 0.0
        self.last_fill_row = self.t - 1000000
        self.fill_log = []                   # (u, tau, side, qty, px)
        self.decisions = []                  # (t, bid, ask) after the pull rule

    # ---------------------------------------------------------------- data helpers
    def f(self, row, name):
        return float(self.d.feat[row, self.d.col[name]])

    def x(self, row, name):
        return float(self.d.x3[row, X_COLS.index(name)])

    def mark(self, row):
        mid, p = self.f(row, "mark"), self.f(row, "p_model")
        return [mid, p, 0.5 * (mid + p)][int(self.cfg.mark_mode)]

    def tick(self, book):
        return int(book["tick_alt"] if self.cfg.tick_rule else book["tick"])

    def mid_at(self, row):
        """Book mark for attribution; the settlement value past the end of the episode."""
        return self.f(row, "mark") if row <= self.r_end else self.settle

    def to_event(self, t):
        """Seconds from row t to the market's event (kickoff / window end)."""
        return self.ev_off - (t - self.r0) * self.row_s

    # reward program and liquidity inputs at a row
    def liq_mid(self, r):
        return self.x(r, "liq_mid") if self.d.x3 is not None else self.f(r, "liq_mid")

    def liq_comp(self, r):
        return self.x(r, "liq_comp") if self.d.x3 is not None else self.f(r, "liq_comp")

    def liq_pi(self, r):
        return self.x(r, "liq_pi") if self.d.x3 is not None else self.f(r, "liq_pi")

    def rw_v_at(self, r):
        if self.cfg.liq_v_c > 0:
            return float(self.cfg.liq_v_c)
        return self.x(r, "rw_v_c") if self.d.x3 is not None else self.rw_v

    def rw_min_at(self, r):
        if self.cfg.liq_min_size > 0:
            return float(self.cfg.liq_min_size)
        return self.x(r, "rw_min_size") if self.d.x3 is not None else self.rw_min

    def rw_rate_at(self, r):
        return self.x(r, "rw_rate_day") if self.d.x3 is not None else self.rw_rate

    def in_program(self, r):
        if self.d.x3 is not None:
            return self.x(r, "rw_in_program") > 0
        return self.legacy or self.rw_rate > 0

    # ---------------------------------------------------------------- quoting
    def head_px(self, book, tick, fv_mils, side, a):
        """Price (mils) of price head a on one side with the other side off; 0 = no quote."""
        bb, ba = int(book["bid_px"][0]), int(book["ask_px"][0])
        if side == 0:
            base = bb if bb > 0 else math.floor(fv_mils / tick) * tick
            b = base + OFFSETS[a] * tick
            if ba < 1000 and b >= ba:
                b = ba - tick
            return 0 if b < tick else b
        base = ba if ba < 1000 else math.ceil(fv_mils / tick) * tick
        x = base - OFFSETS[a] * tick
        if bb > 0 and x <= bb:
            x = bb + tick
        return 0 if x > 1000 - tick else x

    def targets(self, t, act):
        book = self.d.book[t]
        tick = self.tick(book)
        fv_mils = float(self.d.feat[t, self.d.col["p_model"]]) * 1000.0
        a_bid, a_ask = min(max(int(act[0]), 0), 6), min(max(int(act[1]), 0), 6)
        bid = self.head_px(book, tick, fv_mils, 0, a_bid) if a_bid else 0
        ask = self.head_px(book, tick, fv_mils, 1, a_ask) if a_ask else 0
        if bid and ask and bid >= ask:
            ask = bid + tick
            if ask > 1000 - tick:
                ask = 0
        return bid, ask

    def order_size(self, k, tau):
        if int(self.cfg.size_mode) == 0:
            return self.cfg.size_mult * self.min_size * SIZES[k]
        base = self.rw_min_at(tau)
        if not base > self.min_size:
            base = self.min_size
        return self.cfg.size_mult * base * [self.cfg.size_rel0, self.cfg.size_rel1, self.cfg.size_rel2][k]

    def activate(self, side, px, size_idx, tau):
        o = self.orders[side]
        if px == 0:
            o.live = False
            return
        if o.live and o.px == px:
            return
        book = self.d.book[tau]
        if side == 0:
            crosses = int(book["ask_px"][0]) < 1000 and px >= int(book["ask_px"][0])
        else:
            crosses = int(book["bid_px"][0]) > 0 and px <= int(book["bid_px"][0])
        if crosses:
            o.live = False
            self.rejects += 1
            return
        size = self.order_size(size_idx, tau)
        room = self.cfg.Q - self.inv if side == 0 else self.cfg.Q + self.inv
        if size > room:
            size = room
            if size < self.min_size:
                o.live = False
                return
        shown = shown_size(book, side, px)
        self.orders[side] = Order(live=True, px=px, rem=size,
                                  queue=0.0 if (self.model == OPT or shown < 0) else shown,
                                  shown=shown, born=tau)

    # ---------------------------------------------------------------- one row of matching
    def match_row(self, u):
        cfg, model = self.cfg, self.model
        tau = u - self.D
        while self.pending and self.pending[0][0] <= tau:
            _, bid, ask, size_idx = self.pending.pop(0)
            self.activate(0, bid, size_idx, tau)
            self.activate(1, ask, size_idx, tau)
        minute = (self.open_ts + (u - self.r0)) // 60
        if minute != self.minute:
            self.minute, self.minute_vol, self.minute_fill = minute, 0.0, 0.0
        bid_live, ask_live = self.orders[0].live, self.orders[1].live
        self.rows += 1
        self.rows_two += bid_live and ask_live
        self.rows_any += bid_live or ask_live
        legs = self.d.fills[self.d.fill_index[u]:self.d.fill_index[u + 1]]
        if not (bid_live or ask_live):
            for leg in legs:
                self.minute_vol += float(leg["size"])
            return
        book = self.d.book[tau]
        if cfg.liq_rewards:
            self.liquidity(tau)
        flags = int(book["flags"])
        allowed = (model == OPT or (not flags & BOOK_CROSSED and not (model == PESS and flags & BOOK_STALE)))
        traded_at = [0.0, 0.0]
        for leg in legs:
            size = float(leg["size"])
            self.minute_vol += size
            if leg["first"]:
                self.tx_fill = 0.0
            side = 0 if leg["side"] < 0 else 1
            o = self.orders[side]
            if not o.live:
                continue
            px = int(leg["px"])
            at = px == o.px
            through = px < o.px if side == 0 else px > o.px
            if not (at or through):
                continue
            if model == OPT:
                avail = size
            elif at:
                traded_at[side] += size
                eat = min(o.queue, size)
                o.queue -= eat
                avail = size - eat if model == CENTRAL else 0.0
            elif model == CENTRAL:
                avail = size
                o.queue = 0.0
            else:
                eat = min(o.queue, size)
                o.queue -= eat
                avail = size - eat
            fill = 0.5 * avail if model == PESS else avail
            if model == PESS:
                fill = min(fill, 0.2 * float(leg["tx_size"]) - self.tx_fill)
            if self.part_cap > 0:
                fill = min(fill, self.part_cap * self.minute_vol - self.minute_fill)
            fill = min(fill, o.rem)
            fill = min(fill, cfg.Q - self.inv if side == 0 else cfg.Q + self.inv)
            if not allowed:
                fill = 0.0
            if fill > 1e-12:
                self.book_fill(side, fill, tau, u, book, px)
        if model == CENTRAL:
            for side in (0, 1):
                o = self.orders[side]
                if not o.live or o.queue <= 0:
                    continue
                shown = shown_size(book, side, o.px)
                if shown >= 0 and o.shown >= 0:
                    others = o.shown - traded_at[side]
                    cancelled = others - shown
                    if cancelled > 0 and others > 1e-12:
                        o.queue -= cancelled * o.queue / others
                if shown >= 0 and o.queue > shown:
                    o.queue = shown
                o.queue = max(o.queue, 0.0)
                o.shown = shown

    def book_fill(self, side, qty, tau, u, book, print_px):
        o = self.orders[side]
        signed = qty if side == 0 else -qty
        p = o.px / 1000.0
        self.inv += signed
        self.cash -= signed * p
        self.rebates += self.rebate_rate * self.fee_rate * qty * (p * (1.0 - p)) ** self.fee_exp
        o.rem -= qty
        if o.rem <= 1e-9:
            o.live = False
        self.tx_fill += qty
        self.minute_fill += qty
        self.n_fills += 1
        self.fill_vol += qty
        if side == 0:
            self.buy_vol += qty
        else:
            self.sell_vol += qty
        self.max_inv = max(self.max_inv, abs(self.inv))
        self.last_fill_row = u
        mid0 = self.f(tau, "mark")
        self.spread += signed * (mid0 - p)
        for h in self.markout:
            self.markout[h] += signed * (self.mid_at(tau + h) - mid0)
        fill_pnl = float(np.float32(signed * (self.mid_at(tau + 60) - p)))
        if self.cfg.reward_mode == 2:
            rh = min(tau + int(self.cfg.markout_h), self.r_end)
            c = signed * (self.f(rh, "p_model") - p)
            self.credit += c
            self.credit_total += c
        tick, bb, ba = self.tick(book), int(book["bid_px"][0]), int(book["ask_px"][0])
        flags = int(book["flags"])
        inside = (bb > 0 and o.px > bb + 2 * tick) if side == 0 else (ba < 1000 and o.px < ba - 2 * tick)
        for name, hit in (("crossed", flags & BOOK_CROSSED), ("stale", flags & BOOK_STALE),
                          ("onesided", bb == 0 or ba == 1000), ("last30", self.r_end - tau < 30),
                          ("last10", self.r_end - tau < 10), ("inside2", inside),
                          ("extreme", o.px <= 20 or o.px >= 980), ("outside", print_px < bb or print_px > ba),
                          ("inspread", print_px > bb if side == 0 else print_px < ba)):
            if hit:
                self.flags[name] += qty
                self.flag_pnl[name] += fill_pnl
        self.fill_log.append((u, tau, side, qty, o.px))

    def our_qmin(self, row, mid, v, min_size):
        """Q_min of our live orders: ((v - d)/v)^2 * size per eligible order (size >= min size, age >=
        liq_moas), two-sided rule inside [0.10, 0.90] with c = 3, else min of the sides."""
        q = [0.0, 0.0]
        for side, o in enumerate(self.orders):
            if not o.live or o.rem < min_size or row - o.born < self.cfg.liq_moas:
                continue
            q[side] = score(abs(o.px / 1000.0 - mid) * 100.0, v) * o.rem
        lo, hi = min(q), max(q)
        return max(lo, hi / REWARD_C) if BAND[0] <= mid <= BAND[1] else lo

    def liquidity(self, tau):
        pi, mid = self.liq_pi(tau), self.liq_mid(tau)
        if pi <= 0 or mid < 0:
            return
        v = self.rw_v_at(tau)
        if not v > 0:
            return
        self.rw_samples += 1
        q_min = self.our_qmin(tau, mid, v, self.rw_min_at(tau))
        if q_min > 0:
            comp = self.liq_comp(tau)
            self.liq += pi * q_min / (q_min + comp)
            self.rw_scored += 1
            self.rw_share += q_min / (q_min + comp)

    # ---------------------------------------------------------------- env interface
    def step(self, act):
        """One decision; returns (obs, reward, done). obs is None when done."""
        cfg = self.cfg
        t = self.t
        bid, ask = self.targets(t, act)
        sec = self.to_event(t)
        if cfg.pull_s > 0 and sec <= cfg.pull_s:
            bid = ask = 0
        cpen = 0.0
        if cfg.close_inv_coef > 0:
            left = sec / cfg.close_ramp_s if cfg.close_ramp_s > 0 else (self.r_end - t) / self.len / cfg.close_ramp
            cpen = cfg.close_inv_coef * abs(self.inv) / cfg.Q * max(0.0, 1.0 - max(left, 0.0))
        self.decisions.append((t, bid, ask))
        self.pending.append([t + self.L, bid, ask, min(max(int(act[2]), 0), 2)])
        dt = self.dt
        if cfg.dt_far > 0 and sec > cfg.far_s:
            dt = int(cfg.dt_far)
        t_next = min(t + dt, self.r_end)
        income0 = self.rebates + self.liq
        self.credit = 0.0
        for u in range(t + 1, t_next + 1):
            self.match_row(u)
        self.t = t_next
        done = t_next == self.r_end
        mark = self.settle if done else self.mark(t_next)
        wealth = self.cash + self.rebates + self.liq + self.inv * mark
        if done and self.flat > 0:
            self.flat_cost = abs(self.inv) * self.flat
            wealth -= self.flat_cost
        pen = cfg.inv_penalty * self.inv ** 2 * mark * (1 - mark) * (t_next - t) / self.len_s
        if cfg.reward_mode == 0:
            r = (wealth - self.wealth - pen) * self.scale
        elif cfg.reward_mode == 1:
            r = 1.0 if (act[0] == 3 and act[1] == 3) else 0.0
        else:
            r = (self.credit + ((self.rebates + self.liq) - income0) - pen) * self.scale
        if cpen > 0:
            r -= cpen
            self.close_pen += cpen
        self.wealth = wealth
        self.penalty += pen
        r = float(np.float32(r))
        if abs(r) > 1:
            self.clips += 1
            r = float(np.clip(r, -1, 1))
        self.ret += float(np.float32(r))
        self.steps += 1
        return (None if done else self.obs()), r, done

    def extra_obs(self):
        """obs 50 .. 50 + NX - 1 (ENV3.md section 5)."""
        t, cfg = self.t, self.cfg
        e = np.zeros(NX, np.float64)
        sec = self.to_event(t)
        e[0] = np.clip(math.log1p(max(sec, 0.0)) / math.log1p(86400.0), 0, 5)
        e[1] = (self.r_end - t) / self.len
        e[2 + min(self.category, 2)] = 1.0 if self.category <= 2 else 0.0
        if not cfg.liq_rewards or not self.in_program(t):
            return e
        v, minsz, mid, comp = self.rw_v_at(t), self.rw_min_at(t), self.liq_mid(t), self.liq_comp(t)
        rate = self.rw_rate_at(t)
        e[5] = 1.0
        e[6] = np.clip(math.log1p(max(rate, 0.0)) / 5.0, 0, 5)
        e[7] = np.clip(v / 5.0, 0, 5)
        e[8] = np.clip(math.log1p(minsz / self.min_size if minsz > 0 else 0.0) / 3.0, 0, 5)
        scorable = mid >= 0 and v > 0
        e[9] = float(scorable)
        e[10] = float(scorable and (mid < BAND[0] or mid > BAND[1]))
        e[11] = np.clip(math.log1p(max(comp, 0.0)) / 10.0, 0, 5)
        for k in range(3):
            e[12 + k] = float(self.order_size(k, t) >= minsz - 1e-9)
        if not scorable:
            return e
        q = self.our_qmin(t, mid, v, minsz)
        e[15] = q / (q + comp) if q > 0 else 0.0
        book = self.d.book[t]
        tick = self.tick(book)
        fv_mils = float(self.d.feat[t, self.d.col["p_model"]]) * 1000.0
        size = max(minsz, self.min_size)
        for side in (0, 1):
            for a in range(1, 7):
                px = self.head_px(book, tick, fv_mils, side, a)
                sc = score(abs(px / 1000.0 - mid) * 100.0, v) * size if px else 0.0
                e[16 + 6 * side + a - 1] = sc / (sc + comp) if sc > 0 else 0.0
        return e

    def obs(self):
        t = self.t
        W = int(self.cfg.obs_size)
        out = np.zeros(W, np.float64)
        out[:N_FEAT_OBS] = self.d.feat[t, :N_FEAT_OBS]
        Q, mid = self.cfg.Q, self.f(t, "mark")
        reb = self.rebate_rate * self.fee_rate * (1.0 / (0.2 * 0.07 * 0.25))
        wealth = self.cash + self.rebates + self.liq + self.inv * self.mark(t)
        a = N_FEAT_OBS
        out[a + 0] = self.inv / Q
        out[a + 1] = np.clip(wealth / (0.05 * Q), -5, 5)
        out[a + 2] = np.clip(self.inv * self.f(t, "delta_usd_raw") / (200.0 * Q), -5, 5)
        for side, o in enumerate(self.orders):
            p = o.px / 1000.0
            out[a + 3 + side] = float(o.live)
            out[a + 5 + side] = np.clip((p - mid) * 100 / 2, -5, 5) if o.live else 0.0
            out[a + 7 + side] = np.clip(o.queue / o.shown, 0, 1) if (o.live and o.shown > 0) else 0.0
            out[a + 9 + side] = np.clip(reb * p * (1 - p), 0, 5) if o.live else 0.0
        since = min(t - self.last_fill_row, 3600)
        out[a + 11] = math.log1p(since) / math.log(3601.0)
        pend = 0.0
        if self.pending:
            last = self.pending[-1]
            pend = float(last[1] != (self.orders[0].px if self.orders[0].live else 0)
                         or last[2] != (self.orders[1].px if self.orders[1].live else 0))
        out[a + 12] = pend
        out[a + 13] = math.log1p(self.n_fills) / 5.0
        out[a + 14] = self.f(t, "bid_minus_p_c")
        out[a + 15] = self.f(t, "ask_minus_p_c")
        out[a + 16] = 1.0
        if W > OBS_V2:
            k = min(W - OBS_V2, NX)
            out[OBS_V2:OBS_V2 + k] = self.extra_obs()[:k]
        return out.astype(np.float32)

    def pnl(self):
        return self.cash + self.rebates + self.liq + self.inv * self.settle - self.flat_cost

    def log(self):
        trading = self.cash + self.inv * self.settle - self.flat_cost
        pnl = self.wealth
        return dict(score=pnl, perf=float(pnl > 0), episode_return=self.ret, episode_length=self.steps,
                    pnl=pnl, pnl_trading=trading, spread_capture=self.spread, markout_5=self.markout[5],
                    markout_30=self.markout[30], markout_60=self.markout[60],
                    inventory_pnl=trading - self.spread - self.markout[60], rebates=self.rebates,
                    liq_rewards=self.liq, inv_penalty=self.penalty, fills=self.n_fills,
                    fill_vol=self.fill_vol, buy_vol=self.buy_vol, sell_vol=self.sell_vol,
                    max_abs_inv=self.max_inv, final_abs_inv=abs(self.inv),
                    uptime_two_sided=self.rows_two / self.rows if self.rows else 0.0,
                    uptime_any=self.rows_any / self.rows if self.rows else 0.0,
                    reward_clip_frac=self.clips / self.steps if self.steps else 0.0, rejects=self.rejects,
                    **{f"vol_{k}": v for k, v in self.flags.items()},
                    **{f"pnl_{k}": v for k, v in self.flag_pnl.items()},
                    close_penalty=self.close_pen, flat_cost=self.flat_cost, settle_px=self.settle,
                    rw_samples=self.rw_samples, rw_scored=self.rw_scored, rw_share=self.rw_share,
                    episode_rows=self.r_end - self.s0, n=1.0)

    def state(self):
        """Same layout as m3_record() in harness3.c."""
        b, a = self.orders
        return np.array([self.t, self.market, self.s0, self.inv, self.cash, self.rebates, self.liq,
                         self.wealth, self.penalty, self.n_fills, self.fill_vol, b.live, b.px, b.rem,
                         b.queue, a.live, a.px, a.rem, a.queue, self.spread, self.markout[5],
                         self.markout[30], self.markout[60], self.rejects, self.settle, self.close_pen,
                         self.flat_cost, self.rw_samples, self.rw_scored, self.rw_share], np.float64)
