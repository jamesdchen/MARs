"""Readable numpy/Python reference of the market-making replay env (mm.h). Used only by tests.

It implements the model of ENV.md directly, one episode at a time and one second at a time,
in float64. test_env.py checks that the C core matches it: fills, inventory, cash, rewards and
observations, on identical markets, start offsets and action sequences.

    from reference import Data, Config, Episode
    data = Data()                                 # memmaps data/build/*
    ep = Episode(data, Config(fill_model=1, latency=1), market=12, start=30)
    obs = ep.obs()
    while not done: obs, reward, done = ep.step((3, 3, 0))
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "data" / "build"

OPT, CENTRAL, PESS = 0, 1, 2
OFFSETS = [None, 2, 1, 0, -1, -2, -4]       # action -> ticks better than the touch (None = off)
SIZES = [1.0, 2.0, 5.0]                       # action -> multiple of min_order_size
DELAY_S = {OPT: 0.0, CENTRAL: 3.5, PESS: 5.0}  # EVALUATION.md T1: print time - match time
BOOK_VALID, BOOK_CROSSED, BOOK_STALE = 1, 2, 4
LIQ_V, LIQ_MIN = 1.5, 50.0
N_FEAT_OBS, OBS = 33, 50
FLAGS = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside", "inspread"]

BOOK_DTYPE = np.dtype([("bid_px", "<i2", 5), ("ask_px", "<i2", 5), ("bid_sz", "<f4", 5),
                       ("ask_sz", "<f4", 5), ("flags", "u1"), ("tick", "u1"), ("tick_alt", "u1"),
                       ("same_run", "u1")])
FILL_DTYPE = np.dtype([("row", "<i4"), ("side", "i1"), ("first", "u1"), ("px", "<i2"),
                       ("price", "<f4"), ("size", "<f4"), ("tx_size", "<f4")])


@dataclass
class Config:
    """Same names and defaults as MM_PARAM_NAMES / MM_DEFAULTS in mm.h."""
    split: float = 0
    allow_test: float = 0
    fill_model: float = CENTRAL
    latency: float = 1
    match_delay: float = -1        # seconds; < 0: the fill model's default
    dt: float = 0                  # > 0 overrides dt_5m / dt_15m / dt_4h
    dt_5m: float = 1
    dt_15m: float = 1
    dt_4h: float = 5
    Q: float = 100
    inv_penalty: float = 0.001
    reward_scale: float = 0        # <= 0: 1 / Q
    size_mult: float = 1
    part_cap: float = -1           # < 0: 0 (off) optimistic, 0.25 central / pessimistic
    fee_rate: float = -1           # < 0: markets.bin
    rebate_rate: float = -1        # < 0: markets.bin
    liq_rewards: float = 0
    liq_moas: float = 1
    mark_mode: float = 0
    reward_mode: float = 0
    tick_rule: float = 0
    len_mask: float = 7
    fixed_market: float = -1
    fixed_start: float = -1
    cycle_stride: float = 0
    start_frac: float = 0.2
    markout_h: float = 60          # reward_mode 2: fill credit horizon (s)

    def values(self):
        return np.array([float(getattr(self, f.name)) for f in fields(self)])

    @staticmethod
    def names():
        return [f.name for f in fields(Config)]


class Data:
    def __init__(self, build=BUILD):
        build = Path(build)
        meta = json.loads((build / "meta.json").read_text())
        fmeta = json.loads((build / "feat_meta.json").read_text())
        rec = meta["market_record"]
        np_t = {"int64_t": "<i8", "int32_t": "<i4", "float": "<f4"}
        dt = np.dtype({"names": [f["name"] for f in rec["fields"]],
                       "formats": [np_t[f["type"]] for f in rec["fields"]],
                       "offsets": [f["offset"] for f in rec["fields"]], "itemsize": rec["itemsize"]})
        self.markets = np.fromfile(build / "markets.bin", dtype=dt)
        self.N, self.G = fmeta["N"], fmeta["G"]
        self.col = fmeta["column_index"]
        self.feat = np.memmap(build / "feat.bin", np.float32, "r", shape=(self.N, self.G))
        self.book = np.memmap(build / "book.bin", BOOK_DTYPE, "r", shape=(self.N,))
        self.fills = np.fromfile(build / "fills.bin", FILL_DTYPE)
        self.fill_index = np.fromfile(build / "fill_index.bin", np.int32)
        assert self.fill_index.size == self.N + 1


@dataclass
class Order:
    live: bool = False
    px: int = 0              # mils
    rem: float = 0.0         # shares left
    queue: float = 0.0       # shares ahead of us
    shown: float = -1.0      # displayed size at our level at the last match second (-1 unknown)
    born: int = 0            # match row it went live


def shown_size(book, side, px):
    """Displayed size at price px (mils) on side 0 bid / 1 ask: 0 for an empty level inside the
    5 shown levels, -1 when px lies beyond them."""
    prices = book["bid_px"] if side == 0 else book["ask_px"]
    sizes = book["bid_sz"] if side == 0 else book["ask_sz"]
    for k in range(5):
        p = int(prices[k])
        if p == px:
            return float(sizes[k])
        if (side == 0 and p < px) or (side == 1 and p > px):
            return 0.0
    return -1.0


class Episode:
    def __init__(self, data: Data, cfg: Config, market: int, start: int):
        self.d, self.cfg = data, cfg
        m = data.markets[market]
        self.market = market
        self.r0 = int(m["row_offset"])
        self.r_end = self.r0 + int(m["n_rows"]) - 1
        start = min(max(start, 0), int(m["n_rows"]) - 2)
        self.s0 = self.t = self.r0 + start
        self.len_s = int(m["contract_len_s"])
        self.open_ts = int(m["open_ts"])
        self.winner = int(m["winner_up"])
        self.min_size = float(m["min_order_size"])
        self.fee_rate = cfg.fee_rate if cfg.fee_rate >= 0 else float(m["taker_fee_rate"])
        self.rebate_rate = cfg.rebate_rate if cfg.rebate_rate >= 0 else float(m["maker_rebate_rate"])
        self.fee_exp = float(m["fee_exponent"])
        model = int(cfg.fill_model)
        self.model = model
        self.L = int(cfg.latency)
        delay = cfg.match_delay if cfg.match_delay >= 0 else DELAY_S[model]
        self.D = math.ceil(delay + 0.5)      # trade printed in row u matched at row u - D
        if cfg.dt > 0:
            self.dt = int(cfg.dt)
        else:
            self.dt = int({300: cfg.dt_5m, 900: cfg.dt_15m, 14400: cfg.dt_4h}[self.len_s])
        self.part_cap = cfg.part_cap if cfg.part_cap >= 0 else (0.0 if model == OPT else 0.25)
        self.scale = cfg.reward_scale if cfg.reward_scale > 0 else 1.0 / cfg.Q
        self.orders = [Order(), Order()]
        self.pending = []                    # [activation row, bid px, ask px, size index]
        self.inv = self.cash = self.rebates = self.liq = self.wealth = self.penalty = 0.0
        self.spread = 0.0
        self.markout = {5: 0.0, 30: 0.0, 60: 0.0}
        self.minute = None
        self.minute_vol = self.minute_fill = self.tx_fill = 0.0
        self.credit = 0.0                    # reward_mode 2: fill credits of the current step
        self.credit_total = 0.0
        self.n_fills = 0
        self.fill_vol = self.buy_vol = self.sell_vol = self.max_inv = 0.0
        self.flags = dict.fromkeys(FLAGS, 0.0)          # fill volume per exploit flag
        self.flag_pnl = dict.fromkeys(FLAGS, 0.0)       # spread capture + 60 s markout per flag
        self.rows = self.rows_two = self.rows_any = 0
        self.steps = self.clips = self.rejects = 0
        self.ret = 0.0
        self.last_fill_row = self.t - 1000000
        self.fill_log = []                   # (u, tau, side, qty, px) for inspection

    # ---------------------------------------------------------------- data helpers
    def f(self, row, name):
        return float(self.d.feat[row, self.d.col[name]])

    def mark(self, row):
        mid, p = self.f(row, "mark"), self.f(row, "p_model")
        return [mid, p, 0.5 * (mid + p)][int(self.cfg.mark_mode)]

    def tick(self, book):
        return int(book["tick_alt"] if self.cfg.tick_rule else book["tick"])

    def mid_at(self, row):
        """Book mark for attribution; settlement value past the end of the window."""
        return self.f(row, "mark") if row <= self.r_end else float(self.winner)

    # ---------------------------------------------------------------- quoting
    def targets(self, t, act):
        book = self.d.book[t]
        tick = self.tick(book)
        best_bid, best_ask = int(book["bid_px"][0]), int(book["ask_px"][0])
        fv_mils = float(self.d.feat[t, self.d.col["p_model"]]) * 1000.0
        a_bid, a_ask = min(max(int(act[0]), 0), 6), min(max(int(act[1]), 0), 6)
        bid = ask = 0
        if a_bid:
            base = best_bid if best_bid > 0 else math.floor(fv_mils / tick) * tick
            bid = base + OFFSETS[a_bid] * tick
            if best_ask < 1000 and bid >= best_ask:      # post-only: one tick behind the ask
                bid = best_ask - tick
            if bid < tick:
                bid = 0
        if a_ask:
            base = best_ask if best_ask < 1000 else math.ceil(fv_mils / tick) * tick
            ask = base - OFFSETS[a_ask] * tick
            if best_bid > 0 and ask <= best_bid:
                ask = best_bid + tick
            if ask > 1000 - tick:
                ask = 0
        if bid and ask and bid >= ask:
            ask = bid + tick
            if ask > 1000 - tick:
                ask = 0
        return bid, ask

    def activate(self, side, px, size_idx, tau):
        o = self.orders[side]
        if px == 0:
            o.live = False
            return
        if o.live and o.px == px:
            return                                      # keep the order and its queue position
        book = self.d.book[tau]
        if side == 0:
            crosses = int(book["ask_px"][0]) < 1000 and px >= int(book["ask_px"][0])
        else:
            crosses = int(book["bid_px"][0]) > 0 and px <= int(book["bid_px"][0])
        if crosses:
            o.live = False
            self.rejects += 1
            return
        size = self.cfg.size_mult * self.min_size * SIZES[size_idx]
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

    # ---------------------------------------------------------------- one second of matching
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
            for leg in legs:                                # taker volume of this minute
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
            side = 0 if leg["side"] < 0 else 1           # taker sells hit bids, buys hit asks
            o = self.orders[side]
            if not o.live:
                continue
            px = int(leg["px"])
            at = px == o.px
            through = px < o.px if side == 0 else px > o.px
            if not (at or through):
                continue
            if model == OPT:                               # front of the queue, any print
                avail = size
            elif at:                                       # queue ahead fills first
                traded_at[side] += size
                eat = min(o.queue, size)
                o.queue -= eat
                avail = size - eat if model == CENTRAL else 0.0
            elif model == CENTRAL:                         # swept through our level
                avail = size
                o.queue = 0.0
            else:                                          # pessimistic: displayed size first
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
        if self.cfg.reward_mode == 2:        # markout to fair value at tau + H (last row at most)
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

    def liquidity(self, tau):
        pi, mid = self.f(tau, "liq_pi"), self.f(tau, "liq_mid")
        if pi <= 0 or mid < 0:
            return
        q = [0.0, 0.0]
        for side, o in enumerate(self.orders):
            if not o.live or o.rem < LIQ_MIN or tau - o.born < self.cfg.liq_moas:
                continue
            dist = abs(o.px / 1000.0 - mid) * 100.0
            if dist < LIQ_V:
                q[side] = ((LIQ_V - dist) / LIQ_V) ** 2 * o.rem
        lo, hi = min(q), max(q)
        q_min = max(lo, hi / 3.0) if 0.10 <= mid <= 0.90 else lo
        if q_min > 0:
            self.liq += pi * q_min / (q_min + self.f(tau, "liq_comp"))

    # ---------------------------------------------------------------- env interface
    def step(self, act):
        """One decision; returns (obs, reward, done). obs is None when done (the env starts a
        new episode there)."""
        t = self.t
        bid, ask = self.targets(t, act)
        self.pending.append([t + self.L, bid, ask, min(max(int(act[2]), 0), 2)])
        t_next = min(t + self.dt, self.r_end)
        income0 = self.rebates + self.liq
        self.credit = 0.0
        for u in range(t + 1, t_next + 1):
            self.match_row(u)
        self.t = t_next
        done = t_next == self.r_end
        mark = float(self.winner) if done else self.mark(t_next)
        wealth = self.cash + self.rebates + self.liq + self.inv * mark
        pen = self.cfg.inv_penalty * self.inv ** 2 * mark * (1 - mark) * (t_next - t) / self.len_s
        if self.cfg.reward_mode == 0:                        # mark-to-market
            r = (wealth - self.wealth - pen) * self.scale
        elif self.cfg.reward_mode == 1:                      # probe
            r = 1.0 if (act[0] == 3 and act[1] == 3) else 0.0
        else:                                                # fill markouts + rebates + rewards
            r = (self.credit + ((self.rebates + self.liq) - income0) - pen) * self.scale
        self.wealth = wealth
        self.penalty += pen
        r = float(np.float32(r))
        if abs(r) > 1:
            self.clips += 1
            r = float(np.clip(r, -1, 1))
        self.ret += float(np.float32(r))
        self.steps += 1
        return (None if done else self.obs()), r, done

    def obs(self):
        t = self.t
        out = np.zeros(OBS, np.float64)
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
            # queue ahead / shown size at our level, both as tracked in match time
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
        return out.astype(np.float32)

    def log(self):
        """Episode Log (after the terminal step), keyed like mm.h's Log."""
        trading = self.cash + self.inv * self.winner
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
                    **{f"pnl_{k}": v for k, v in self.flag_pnl.items()}, n=1.0)

    def state(self):
        """Same layout as record() in mm_harness.c."""
        b, a = self.orders
        return np.array([self.t, self.market, self.s0, self.inv, self.cash, self.rebates, self.liq,
                         self.wealth, self.penalty, self.n_fills, self.fill_vol, b.live, b.px, b.rem,
                         b.queue, a.live, a.px, a.rem, a.queue, self.spread, self.markout[5],
                         self.markout[30], self.markout[60], self.rejects], np.float64)
