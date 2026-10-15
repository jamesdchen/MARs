"""Tests of the v3 replay env (mm3.h). python test_env3.py [--quick] [--no-builders] [--json OUT]

(a) v2 bit-exactness: cmp_harness.c steps ../mm.h (frozen) and mm3.h side by side on data/build
    val (every val market, plus a train sample) under 18 v2 configs with the v3 options at their
    defaults, and with obs_size 80 (only obs 0..49 compared). Observations, rewards, terminals,
    the double state and the Log must be bit-identical, and every episode's P&L (computed in double
    at its terminal step) must match to 1e-9 (it matches exactly).
(b) C == numpy reference (reference3.py) on synthetic sports and election builds in the builders'
    schemas (fixtures3.py, made loadable by adapt3.py), on the real output of
    analysis/v3_sports/build_sports.py --selftest (when its deps are importable), and on crypto
    markets with the v3 options on: rewards, terminals, observations (all obs_size slots), state
    and Log at every step, including the auto-reset episode; reward telescoping with the
    flattening cost and close-inventory shaping; lookahead (perturbing rows after t changes nothing
    up to t).
(c) Settlement: sports episodes end at kickoff and settle at the kickoff mid (markets3 settle_price
    = the last row's mark), independent of winner_up; the flattening cost; settle_mode overrides;
    an episode flat at kickoff (pull_s: no live quote and no fill after the pull); elections settle
    at the last mark (winner_up = -1).
(d) Reward hook: a hand-computed quadratic score (v = 3 c, our bid 1 c and ask 2 c from the
    adjusted mid, 100 shares each, two-sided rule, Q_comp = 100, 0.01 USD per sample), its
    one-sided / ineligible / age variants and the reward observations, in C and in the
    reference; reference3's scorer == analysis/rewards/reward_terms.py on 2000 random books.
(e) adapt3.py on the build_sports schemas (fixtures3): sports-2 (meta.json segments, sports_x3.bin
    copied, never recomputed, --rewards ignored, rw_rate_day from segments), --no-legacy-settle and the
    legacy sports-1 layout (--rewards fallback) adapt the same games to identical markets3.bin /
    x3.bin; TRADES_TRUNCATED at 1024 (sports-2) / 32 (sports-1); --bucket-segments pieces get
    event_ts = kickoff (obs 50 = time to kickoff), and C == reference on them with pull_s / flat_c.
(f) D15 netting / flatten Log fields (net_gross_sh, net_notional, net_end_inv, net_share, net_eps,
    net_flat_drag) against fill_vol / final_abs_inv / flat_cost, per-episode netted share, a
    never-quoting policy, and env3.netting's pooled share and drag in bps. (b) compares every one
    of them with reference3 too.
Plus: parameter tables (mm3.h == Config3, first 27 == v2), split guard, PufferLib 3.0 binding ==
core (crypto and a v3 fixture, in subprocesses; vec_log carries every Log field).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PM = HERE.parent
sys.path.insert(0, str(HERE))
from reference3 import Config3, Data3, Episode3, MARKET_DTYPE, X_COLS  # noqa: E402
import adapt3  # noqa: E402
import fixtures3  # noqa: E402

V2_DATA = PM / "data" / "build"
SRC = (HERE / "mm3.h").read_text()
NSTATE = 30
LOG_FIELDS = re.findall(r"float (\w+);", re.search(r"struct Mm3Log \{(.*?)\n\};", SRC, re.S)[1])
STATE_NAMES = ["t", "market", "s0", "inv", "cash", "rebates", "liq", "wealth", "penalty", "n_fills",
               "fill_vol", "bid_live", "bid_px", "bid_rem", "bid_q", "ask_live", "ask_px", "ask_rem",
               "ask_q", "spread", "mo5", "mo30", "mo60", "rejects", "settle", "close_pen", "flat_cost",
               "rw_samples", "rw_scored", "rw_share"]
FLOAT_STATE = {"n_fills", "fill_vol", "rejects", "rw_samples", "rw_scored"}
CFLAGS = ["-O2", "-std=c99", "-ffp-contract=off", "-D_POSIX_C_SOURCE=200809L", "-Wno-unused-variable",
          "-shared", "-fPIC"]


def cc(src, out):
    subprocess.run([os.environ.get("CC", "cc"), *CFLAGS, "-o", str(out), str(src), "-lm"], check=True)
    return ctypes.CDLL(str(out))


class Harness:
    """harness3.c through ctypes; open() switches the data directory (m3_h_release)."""

    def __init__(self, tmp):
        self.lib = cc(HERE / "harness3.c", Path(tmp) / "libmm3_test.so")
        self.dir = None
        assert self.lib.m3_h_sizeof_log() == 4 * len(LOG_FIELDS)
        assert self.lib.m3_h_num_params() == len(Config3.names())

    def open(self, d):
        d = str(d)
        if self.dir == d:
            return
        self.lib.m3_h_release()
        err = ctypes.create_string_buffer(512)
        assert self.lib.m3_h_open(d.encode(), err, 512) == 0, err.value
        self.dir = d

    def run(self, cfg: Config3, markets, starts, actions, seed=0):
        T, n = actions.shape[:2]
        W = int(cfg.obs_size)
        obs = np.zeros((T + 1, n, W), np.float32)
        rew = np.zeros((T, n), np.float32)
        term = np.zeros((T, n), np.float32)
        state = np.zeros((T + 1, n, NSTATE))
        logs = np.zeros((n, len(LOG_FIELDS)), np.float32)
        err = ctypes.create_string_buffer(512)
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        vals = cfg.values()
        mk = np.ascontiguousarray(markets, np.int32)
        st = np.ascontiguousarray(starts, np.int32)
        act = np.ascontiguousarray(actions, np.int32)
        rc = self.lib.m3_h_run(p(vals), n, T, p(mk), p(st), p(act), ctypes.c_uint64(seed), p(obs), p(rew), p(term),
                               p(state), p(logs), err, 512)
        if rc != 0:
            raise ValueError(err.value.decode())
        return obs, rew, term, state, dict(zip(LOG_FIELDS, logs.T))


class Checks:
    def __init__(self):
        self.rows = {}

    def add(self, name, value, tol):
        value = float(value)
        ok = np.isfinite(value) and value <= tol
        prev = self.rows.get(name, (0.0, tol, True))
        self.rows[name] = (max(prev[0], value), tol, prev[2] and ok)

    def report(self):
        for name, (value, tol, ok) in self.rows.items():
            print(f"{name:78s} {value:9.2e}  tol {tol:7.0e}  {'ok' if ok else 'FAIL'}")
        bad = [k for k, (_, _, ok) in self.rows.items() if not ok]
        return bad


def random_actions(rng, T, n, p_off=0.08):
    p_lvl = np.array([p_off, 0.07, 0.12, 0.35, 0.18, 0.12, 0.08])
    p_lvl = p_lvl / p_lvl.sum()
    a = np.zeros((T, n, 3), np.int32)
    cur = np.stack([rng.choice(7, n, p=p_lvl), rng.choice(7, n, p=p_lvl), rng.integers(0, 3, n)], 1)
    for t in range(T):
        new = rng.random(n) < 0.35
        cur[new] = np.stack([rng.choice(7, new.sum(), p=p_lvl), rng.choice(7, new.sum(), p=p_lvl),
                             rng.integers(0, 3, new.sum())], 1)
        a[t] = cur
    return a


def rel(a, b):
    return np.max(np.abs(np.asarray(a, np.float64) - b) / (1 + np.abs(b))) if np.size(a) else 0.0


def max_steps(data, cfg, market, start):
    n = int(data.markets["n_rows"][market]) - 1
    dt = int(cfg.dt) if cfg.dt > 0 else int([cfg.dt_5m, cfg.dt_15m, cfg.dt_4h, cfg.dt_other][
        {300: 0, 900: 1, 14400: 2}.get(n, 3)])
    if cfg.dt_far > 0:
        dt = min(dt, int(cfg.dt_far))
    start = min(start, n - 1)
    return -(-(n - start) // dt)


# ------------------------------------------------------------------ (a) v2 bit-exactness

V2_CONFIGS = ([(f"model{m}-L{L}", dict(fill_model=m, latency=L)) for m in (0, 1, 2) for L in (0, 1, 2)] + [
    ("mark1-penalty", dict(mark_mode=1, inv_penalty=0.01)),
    ("mark2-penalty", dict(mark_mode=2, inv_penalty=0.01)),
    ("probe", dict(reward_mode=1)),
    ("tick-rule1-dt3", dict(tick_rule=1, dt=3)),
    ("delay2-nocap-rebate35", dict(match_delay=2, part_cap=0, rebate_rate=0.35)),
    ("pess-delay4-L2", dict(fill_model=2, match_delay=4, latency=2)),
    ("liq-on-size10-Q300", dict(liq_rewards=1, size_mult=10, Q=300, dt_4h=2)),
    ("opt-size0.5-Q20", dict(fill_model=0, size_mult=0.5, Q=20, inv_penalty=0.05)),
    ("credit-liq-H30-mark2", dict(reward_mode=2, markout_h=30, liq_rewards=1, size_mult=10, Q=300, mark_mode=2)),
    ("central-obs80 (v3 slots appended)", dict(obs_size=80)),
])


def v2_bitexact(tmp, checks, quick, results):
    lib = cc(HERE / "cmp_harness.c", Path(tmp) / "libcmp.so")
    err = ctypes.create_string_buffer(512)
    assert lib.cmp_open(str(V2_DATA).encode(), err, 512) == 0, err.value
    mk = np.fromfile(V2_DATA / "markets.bin", MARKET_DTYPE)
    val = np.flatnonzero(mk["split"] == 1)
    rng = np.random.default_rng(2026)
    train = rng.choice(np.flatnonzero(mk["split"] == 0), 40, replace=False)
    out = {}
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    for ci, (tag, kw) in enumerate(V2_CONFIGS):
        cfg = replace(Config3(split=1), **kw)
        for split_tag, markets in (("val", val), ("train", train)):
            if quick and split_tag == "train":
                continue
            n = len(markets)
            starts = np.array([int(rng.integers(0, int(0.2 * (mk["n_rows"][i] - 1)) + 1)) for i in markets], np.int32)
            long_run = ci in (0, 4, 11, 15, 16) or split_tag == "train"
            T = 3000 if (long_run and not quick) else (1000 if not quick else 400)
            dt4 = cfg.dt if cfg.dt > 0 else cfg.dt_4h
            if T * dt4 < 14400:                   # 4h markets would not reach their terminal: drop them
                markets = markets[mk["contract_len_s"][markets] < 14400]
                starts = np.array([int(rng.integers(0, int(0.2 * (mk["n_rows"][i] - 1)) + 1)) for i in markets],
                                  np.int32)
                n = len(markets)
            vals = replace(cfg, split=split_tag == "val" and 1 or 0).values()
            acts = random_actions(np.random.default_rng(100 + ci), T, n)
            stats = np.zeros(10)
            cap = 4 * n + 1000
            p2, p3 = np.zeros(cap), np.zeros(cap)
            ne = ctypes.c_int(0)
            rc = lib.cmp_run(p(vals), n, T, p(np.ascontiguousarray(markets, np.int32)), p(starts), p(acts),
                             ctypes.c_uint64(7 + ci), p(stats), p(p2), p(p3), cap, ctypes.byref(ne), err, 512)
            assert rc == 0, err.value
            k = ne.value
            checks.add("(a) v2 == v3: observations 0..49 bit-identical (count of differing floats)", stats[1], 0)
            checks.add("(a) v2 == v3: rewards bit-identical (max |diff|)", stats[2], 0)
            checks.add("(a) v2 == v3: terminal flags (mismatches)", stats[3], 0)
            checks.add("(a) v2 == v3: Log fields bit-identical (count of differing floats)", stats[5], 0)
            checks.add("(a) v2 == v3: double state (inv, cash, rebates, rewards, wealth, orders) max |diff|",
                       stats[6], 0)
            checks.add("(a) v2 == v3: episode P&L in double, max |diff| (USD)", float(np.max(np.abs(p2[:k] - p3[:k])))
                       if k else 0.0, 1e-9)
            checks.add("(a) every run finished episodes (1 = none)", float(k == 0), 0)
            out[f"{tag}/{split_tag}"] = dict(envs=n, steps=int(stats[7]), episodes=k,
                                             max_pnl_diff=float(np.max(np.abs(p2[:k] - p3[:k]))) if k else None,
                                             sum_abs_pnl=float(np.abs(p2[:k]).sum()))
        print(f"(a) {tag:38s} " + "  ".join(f"{s}: {v['episodes']} eps / {v['steps']} steps"
                                          for s, v in out.items() if s.startswith(tag + "/")), flush=True)
    results["v2_bitexact"] = out


# ------------------------------------------------------------------ (b) C == reference

def equivalence(h, data, checks, cfg: Config3, markets, starts, rng, tag, extra=25, seed=11, p_off=0.08):
    h.open(data.dir)
    steps = [max_steps(data, cfg, m, s) for m, s in zip(markets, starts)]
    T = max(steps) + extra
    actions = random_actions(rng, T, len(markets), p_off)
    obs, rew, term, state, log = h.run(cfg, markets, starts, actions, seed=seed)
    W = int(cfg.obs_size)
    checks.add("(b) observations finite, |obs| <= 5, obs 49 == 1",
               0 if (np.isfinite(obs).all() and np.abs(obs).max() <= 5 and (obs[..., 49] == 1).all()) else 1, 0)
    scale = cfg.reward_scale if cfg.reward_scale > 0 else 1 / cfg.Q
    n_fills = 0.0
    for i, (m, s) in enumerate(zip(markets, starts)):
        ep = Episode3(data, cfg, int(m), int(s))
        checks.add(f"(b) [{tag}] initial observation", np.abs(obs[0, i] - ep.obs()).max(), 1e-6)
        k = 0
        while True:
            o, r, done = ep.step(actions[k, i])
            checks.add("(b) reward == reference", abs(rew[k, i] - r), 1e-7)
            checks.add("(b) terminal flag", abs(term[k, i] - float(done)), 0)
            if done:
                break
            checks.add(f"(b) observation == reference (all {W} slots)", np.abs(obs[k + 1, i] - o).max(), 1e-6)
            st, ref = state[k + 1, i], ep.state()
            for j, name in enumerate(STATE_NAMES):
                checks.add(f"(b) state {name} (rel)", rel(st[j], ref[j]), 1e-4 if name in FLOAT_STATE else 1e-9)
            k += 1
        ref_log = ep.log()
        for f in LOG_FIELDS:
            checks.add(f"(b) Log {f} (rel)", rel(log[f][i], ref_log[f]), 2e-4 if f == "episode_return" else 1e-4)
        if cfg.reward_mode == 0 and ref_log["reward_clip_frac"] == 0:
            checks.add("(b) telescoping: sum r/scale + inv penalty + close penalty/scale == P&L ($)",
                       abs(np.sum(np.asarray(rew[:k + 1, i], np.float64)) / scale + ep.penalty + ep.close_pen / scale
                           - ref_log["pnl"]), 2e-4)
            checks.add("(b) P&L == cash + rebates + rewards + inv * settle - |inv| * flat ($)",
                       abs(ref_log["pnl"] - ep.pnl()), 1e-9)
        if cfg.reward_mode == 2 and ref_log["reward_clip_frac"] == 0:
            checks.add("(b) reward_mode 2 identity ($)", abs(np.sum(np.asarray(rew[:k + 1, i], np.float64)) / scale
                       + ep.penalty + ep.close_pen / scale - (ep.credit_total + ep.rebates + ep.liq)), 2e-4)
        nm, ns = int(state[k + 1, i, 1]), int(state[k + 1, i, 2])
        nxt = Episode3(data, cfg, nm, ns - int(data.markets["row_offset"][nm]))
        checks.add("(b) auto-reset: new episode's first observation", np.abs(obs[k + 1, i] - nxt.obs()).max(), 1e-6)
        for kk in range(k + 1, T):
            o, r, done = nxt.step(actions[kk, i])
            checks.add("(b) auto-reset episode: reward", abs(rew[kk, i] - r), 1e-7)
            if done:
                break
            checks.add("(b) auto-reset episode: observation", np.abs(obs[kk + 1, i] - o).max(), 1e-6)
        n_fills += ref_log["fills"]
    return dict(fills=float(n_fills), liq=float(np.sum(log["liq_rewards"])), rw_scored=float(np.sum(log["rw_scored"])),
                close=float(np.sum(log["close_penalty"])), flat=float(np.sum(log["flat_cost"])))


SPORTS_CONFIGS = [
    ("sports-central", dict()),
    ("sports-obs80-liq-size1-Q1000", dict(obs_size=80, liq_rewards=1, size_mode=1, Q=1000)),
    ("sports-pull-flat-close-ramp_s", dict(pull_s=300, flat_c=0.5, close_inv_coef=0.05, close_ramp_s=900,
                                           obs_size=64)),
    ("sports-dtfar-pess-L2", dict(dt_far=10, far_s=1200, dt_other=2, fill_model=2, latency=2)),
    ("sports-opt-L0-credit-fees", dict(fill_model=0, latency=0, reward_mode=2, fee_rate_sports=0.03,
                                       rebate_rate_sports=0.25)),
    ("sports-mark1-settle2-close-frac", dict(mark_mode=1, settle_mode=2, flat_c=1.0, close_inv_coef=0.02,
                                             close_ramp=0.3, inv_penalty=0.01)),
    ("sports-liq-relsizes-moas30", dict(obs_size=80, liq_rewards=1, size_mode=1, size_rel0=1.5, size_rel1=3,
                                        size_rel2=6, Q=2000, liq_moas=30, liq_v_c=4.5)),
]
ELECTION_CONFIGS = [
    ("elections-obs80-liq-size1", dict(obs_size=80, liq_rewards=1, size_mode=1, Q=2000)),
    ("elections-v2sizes-mult10-liq", dict(liq_rewards=1, size_mult=10, Q=500, obs_size=80)),
    ("elections-credit-liq-pess", dict(reward_mode=2, liq_rewards=1, size_mode=1, Q=3000, fill_model=2,
                                       obs_size=80)),
    ("elections-fees-dt1-minsize", dict(fee_rate_elections=0.05, rebate_rate_elections=0.1, dt_other=1,
                                        liq_rewards=1, liq_min_size=10, size_mode=1, Q=500)),
    ("elections-pull-close-flat", dict(pull_s=600, close_inv_coef=0.05, flat_c=0.5, liq_rewards=1, size_mode=1,
                                       Q=2000, obs_size=80)),
]
CRYPTO_V3_CONFIGS = [
    ("crypto-obs80-liq-size1-Q300", dict(obs_size=80, liq_rewards=1, size_mode=1, Q=300)),
    ("crypto-settle2-flat-pull-dtfar", dict(settle_mode=2, flat_c=1.0, pull_s=30, dt_far=3, far_s=600,
                                            close_inv_coef=0.03, obs_size=80)),
    ("crypto-fee-crypto-override", dict(fee_rate_crypto=0.05, rebate_rate_crypto=0.5, obs_size=66)),
]


def fixture_equivalence(h, root, checks, results, builders_dir=None):
    out = {}
    for name, configs in (("sports", SPORTS_CONFIGS), ("elections", ELECTION_CONFIGS)):
        data = Data3(root / name)
        rng = np.random.default_rng(5)
        markets = np.arange(len(data.markets))
        starts = np.array([int(rng.integers(0, int(0.2 * (data.markets["n_rows"][i] - 1)) + 1)) for i in markets])
        for i, (tag, kw) in enumerate(configs):
            cfg = replace(Config3(), **kw)
            out[tag] = equivalence(h, data, checks, cfg, markets, starts, np.random.default_rng(300 + i), tag)
            print(f"(b) {tag:40s} {json.dumps(out[tag])}", flush=True)
        if name == "elections":
            checks.add("(b) election fixture: liquidity rewards were paid (1 = none)",
                       float(out["elections-obs80-liq-size1"]["liq"] == 0), 0)
        if name == "sports":
            checks.add("(b) sports fixture: rewards paid from adapt3's x3 (1 = none)",
                       float(out["sports-obs80-liq-size1-Q1000"]["liq"] == 0), 0)
            checks.add("(b) sports fixture: flattening cost charged (1 = none)",
                       float(out["sports-pull-flat-close-ramp_s"]["flat"] == 0), 0)
    data = Data3(V2_DATA)
    mk = data.markets
    rng = np.random.default_rng(9)
    has = (mk["flags"] & 1) > 0
    idx = list(rng.choice(np.flatnonzero((mk["split"] == 1) & has & (mk["contract_len_s"] < 14400)), 4, replace=False))
    idx += list(rng.choice(np.flatnonzero((mk["split"] == 0) & has & (mk["contract_len_s"] == 14400)), 1))
    starts = [int(rng.integers(0, int(0.2 * mk["contract_len_s"][i]) + 1)) for i in idx]
    for i, (tag, kw) in enumerate(CRYPTO_V3_CONFIGS):
        cfg = replace(Config3(), **kw)
        out[tag] = equivalence(h, data, checks, cfg, np.array(idx), np.array(starts), np.random.default_rng(400 + i),
                               tag)
        print(f"(b) {tag:40s} {json.dumps(out[tag])}", flush=True)
    if builders_dir is not None:
        data = Data3(builders_dir)
        ok = np.flatnonzero(data.markets["split"] <= 1)
        rng = np.random.default_rng(11)
        for i, (tag, kw) in enumerate([("build_sports-selftest-central", dict()),
                                       ("build_sports-selftest-obs80-liq-pull", dict(obs_size=80, liq_rewards=1,
                                        size_mode=1, Q=1000, pull_s=60, flat_c=0.5))]):
            cfg = replace(Config3(), **kw)
            st = np.array([int(rng.integers(0, 50)) for _ in ok])
            out[tag] = equivalence(h, data, checks, cfg, ok, st, np.random.default_rng(500 + i), tag)
            print(f"(b) {tag:40s} {json.dumps(out[tag])}", flush=True)
    results["reference_equivalence"] = out


class _Rows:
    def __init__(self, arr, base):
        self.arr, self.base = arr, base

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.arr[(key[0] - self.base,) + key[1:]]
        return self.arr[key - self.base]


def lookahead(data, checks, cfg, market, cut_frac, rng):
    m = data.markets[market]
    r0, n = int(m["row_offset"]), int(m["n_rows"])
    cut = r0 + int(cut_frac * n)

    class Local:
        pass
    loc = Local()
    loc.markets, loc.col, loc.N, loc.G, loc.ext, loc.dir = data.markets, data.col, data.N, data.G, data.ext, data.dir
    feat = np.array(data.feat[r0:r0 + n])
    book = np.array(data.book[r0:r0 + n])
    lo, hi = int(data.fill_index[r0]), int(data.fill_index[r0 + n])
    fills = np.array(data.fills[lo:hi])
    fidx = np.array(data.fill_index[r0:r0 + n + 1]) - lo
    after = np.arange(n) > cut - r0
    feat[after] = feat[after] + rng.normal(0, 0.3, feat[after].shape).astype(np.float32)
    feat[after, data.col["mark"]] = rng.uniform(0.05, 0.95, after.sum())
    book["bid_sz"][after] *= rng.uniform(0, 3, (after.sum(), 5)).astype(np.float32)
    book["ask_sz"][after] *= rng.uniform(0, 3, (after.sum(), 5)).astype(np.float32)
    fa = fills["row"] > cut
    fills["size"][fa] *= rng.uniform(0.2, 5, fa.sum()).astype(np.float32)
    fills["px"][fa] = np.clip(fills["px"][fa] + rng.integers(-20, 21, fa.sum()), 1, 999)
    loc.feat, loc.book = _Rows(feat, r0), _Rows(book, r0)
    loc.fills, loc.fill_index = fills, _Rows(fidx, r0)
    if data.x3 is not None:
        x3 = np.array(data.x3[r0:r0 + n])
        x3[after] = x3[after] * rng.uniform(0.5, 2, x3[after].shape).astype(np.float32)
        loc.x3 = _Rows(x3, r0)
    else:
        loc.x3 = None
    a = random_actions(rng, max_steps(data, cfg, market, 0), 1)[:, 0]
    e1, e2 = Episode3(data, cfg, market, 0), Episode3(loc, cfg, market, 0)
    diff = np.abs(e1.obs() - e2.obs()).max()
    causal = [j for j in range(NSTATE) if STATE_NAMES[j] not in ("mo5", "mo30", "mo60", "settle")]
    k = 0
    for k in range(len(a)):
        if e1.t + e1.dt > cut:
            break
        o1, r1, _ = e1.step(a[k])
        o2, r2, _ = e2.step(a[k])
        diff = max(diff, np.abs(o1 - o2).max(), abs(r1 - r2), np.abs(e1.state()[causal] - e2.state()[causal]).max())
    checks.add("(b) lookahead: perturbing rows after t changes nothing up to t", diff, 0)
    changed = 0.0
    for k in range(k, len(a)):
        o1, r1, d1 = e1.step(a[k])
        o2, r2, d2 = e2.step(a[k])
        if d1:
            break
        changed = max(changed, np.abs(o1 - o2).max())
    checks.add("(b) lookahead: perturbation is visible after the cut (1 = not)", float(changed == 0), 0)


# ------------------------------------------------------------------ (c) settlement

def settlement(h, root, tmp, checks, results):
    sd = root / "sports"
    data = Data3(sd)
    mk, ext = data.markets, data.ext
    last = mk["row_offset"] + mk["n_rows"] - 1
    marks = np.asarray(data.feat[last, data.col["mark"]], np.float64)
    checks.add("(c) sports data: the last row is the kickoff (open + (n_rows - 1) * row_s == event_ts == end_ts)",
               float(np.max(np.abs(mk["open_ts"] + (mk["n_rows"] - 1) * ext["row_s"] - ext["event_ts"])
                            + np.abs(ext["event_ts"] - mk["end_ts"]))), 0)
    checks.add("(c) sports data: settle_price == markets log_strike (SETTLE_PX) == mark at the last row",
               float(np.max(np.abs(ext["settle_price"] - marks) + np.abs(ext["settle_price"] - mk["log_strike_vs_ul_open"]))),
               0)
    n = len(mk)
    markets, starts = np.arange(n), np.zeros(n, np.int64)
    T = int(mk["n_rows"].max()) + 5
    rng = np.random.default_rng(21)
    acts = random_actions(rng, T, n, p_off=0.0)
    acts[:, :, 2] = 2
    base = Config3(Q=1000, dt_other=1, pull_s=120, latency=1)
    h.open(sd)
    runs = {}
    for tag, kw in (("flat0", dict()), ("flat0.5", dict(flat_c=0.5)), ("mode0", dict(settle_mode=0)),
                    ("mode2", dict(settle_mode=2))):
        runs[tag] = h.run(replace(base, **kw), markets, starts, acts, seed=3)
    lg = runs["flat0"][4]
    checks.add("(c) every sports episode ended inside the run (1 = no)", float((lg["n"] < 1).any()), 0)
    checks.add("(c) C Log settle_px == kickoff mark (float32)", float(np.max(np.abs(lg["settle_px"] - marks))), 1e-7)
    checks.add("(c) some sports episode holds inventory into kickoff (1 = none)",
               float((lg["final_abs_inv"] == 0).all()), 0)
    d = runs["flat0.5"][4]["pnl"] - lg["pnl"]
    checks.add("(c) flat_c 0.5: P&L lowered by |inv| * 0.005 exactly (USD, float32 Log)",
               float(np.max(np.abs(d + lg["final_abs_inv"] * 0.005))), 2e-5)
    checks.add("(c) flat_c 0.5: Log flat_cost == |inv| * 0.005",
               float(np.max(np.abs(runs["flat0.5"][4]["flat_cost"] - lg["final_abs_inv"] * 0.005))), 1e-6)
    checks.add("(c) flat_c does not change the trading (fills identical)",
               float(np.max(np.abs(runs["flat0.5"][4]["fill_vol"] - lg["fill_vol"]))), 0)
    d0 = runs["mode0"][4]["pnl"] - lg["pnl"]
    want = lg["final_abs_inv"] * np.abs(mk["winner_up"] - marks)
    checks.add("(c) settle_mode 0 (winner_up) differs by |inv| * |winner - kickoff mid| (USD)",
               float(np.max(np.abs(np.abs(d0) - want))), 2e-4)
    checks.add("(c) settle_mode 2 (mark at the last row) == settle_mode 1 (kickoff mid) for sports",
               float(np.max(np.abs(runs["mode2"][4]["pnl"] - lg["pnl"]))), 0)
    # winner_up (the game result) is never read
    flip = Path(tmp) / "sports_flip"
    shutil.copytree(sd, flip)
    m2 = np.fromfile(flip / "markets.bin", MARKET_DTYPE)
    m2["winner_up"] = 1 - m2["winner_up"]
    m2.tofile(flip / "markets.bin")
    h.open(flip)
    lf = h.run(base, markets, starts, acts, seed=3)[4]
    checks.add("(c) flipping winner_up changes nothing (Log max |diff|)",
               float(max(np.max(np.abs(lf[f] - lg[f])) for f in LOG_FIELDS)), 0)
    # pull_s: quotes are off from the pull on, so the book is flat (no live quote, no fill) at kickoff
    worst_tau, worst_live = 0, 0
    for i in markets:
        ep = Episode3(data, base, int(i), 0)
        k, done = 0, False
        while not done:
            _, _, done = ep.step(acts[k, i])
            k += 1
        tp = min(t for t, b, a in ep.decisions if ep.to_event(t) <= base.pull_s)
        late = [tau for (u, tau, s, q, px) in ep.fill_log if tau >= tp + ep.L]
        worst_tau = max(worst_tau, len(late))
        worst_live = max(worst_live, int(ep.orders[0].live or ep.orders[1].live))
    checks.add("(c) pull_s: no fill matched after the pull reached the exchange", worst_tau, 0)
    checks.add("(c) pull_s: no quote live at kickoff", worst_live, 0)
    # an episode that ends flat: the settlement price does not matter
    off = acts.copy()
    off[:, :, :2] = 0
    h.open(sd)
    lo = {t: h.run(replace(base, settle_mode=s), markets, starts, off, seed=3)[4] for t, s in (("1", 1), ("0", 0))}
    checks.add("(c) never quoting: P&L 0 under every settle mode", float(max(np.abs(lo["1"]["pnl"]).max(),
                                                                          np.abs(lo["0"]["pnl"]).max())), 0)
    # elections: unresolved (winner_up = -1), settled at the last mark
    ed = Data3(root / "elections")
    h.open(root / "elections")
    em = ed.markets
    elast = em["row_offset"] + em["n_rows"] - 1
    acts_e = random_actions(np.random.default_rng(4), int(em["n_rows"].max()), len(em), p_off=0.0)
    le = h.run(Config3(dt_other=1, Q=500), np.arange(len(em)), np.zeros(len(em)), acts_e, seed=1)[4]
    checks.add("(c) elections: winner_up = -1 everywhere (1 = not)", float((em["winner_up"] != -1).any()), 0)
    checks.add("(c) elections: settle_px == mark at the last row",
               float(np.max(np.abs(le["settle_px"] - ed.feat[elast, ed.col["mark"]]))), 1e-7)
    checks.add("(c) elections: P&L finite and inventory held into the window end (1 = not)",
               float(not (np.isfinite(le["pnl"]).all() and (le["final_abs_inv"] > 0).any())), 0)
    results["settlement"] = dict(
        sports_final_abs_inv=lg["final_abs_inv"].tolist(), sports_pnl=lg["pnl"].tolist(),
        sports_pnl_flat05=runs["flat0.5"][4]["pnl"].tolist(), sports_pnl_winner=runs["mode0"][4]["pnl"].tolist(),
        kickoff_marks=marks.tolist(), elections_pnl=le["pnl"].tolist())


# ------------------------------------------------------------------ (e) adapt3 on the build_sports schemas

def adapt_schemas(h, root, tmp, checks, results):
    """adapt3.py on build_sports sports-2 (segments, sports_x3.bin, --no-legacy-settle, --bucket-segments)
    and on the legacy sports-1 layout: same six games -> the same markets3.bin / x3.bin."""
    from reference3 import EXT_DTYPE
    b = lambda d, f: (root / d / f).read_bytes()  # noqa: E731
    for f in ("markets3.bin", "x3.bin"):
        checks.add(f"(e) {f}: sports-2 == sports-1 (--rewards top-5 fallback) == --no-legacy-settle (1 = differ)",
                   float(not (b("sports", f) == b("sports1", f) == b("sports2_nolegacy", f))), 0)
    checks.add("(e) sports-2: x3.bin is a byte copy of sports_x3.bin (1 = not)",
               float(b("sports", "x3.bin") != b("sports", "sports_x3.bin")), 0)
    # sports_x3.bin is never recomputed, and --rewards does not override the full-ladder columns or the pools
    pd = Path(tmp) / "sports_perturbed"
    shutil.copytree(root / "sports", pd)
    meta = json.loads((pd / "meta.json").read_text())
    x = np.fromfile(pd / "sports_x3.bin", np.float32).reshape(-1, len(X_COLS))
    x[:, 5] *= 2.0
    x[:, 6] *= 0.5
    x.tofile(pd / "sports_x3.bin")
    (pd / "rates999.json").write_text(json.dumps({s["slug"]: 999.0 for s in meta["segments"]}))
    adapt3.main([str(pd), "--rewards", str(pd / "rates999.json")])
    ext = np.fromfile(pd / "markets3.bin", EXT_DTYPE)
    checks.add("(e) sports-2: x3.bin follows sports_x3.bin, not book.bin, with --rewards given (1 = not)",
               float((pd / "x3.bin").read_bytes() != (pd / "sports_x3.bin").read_bytes()), 0)
    checks.add("(e) sports-2: rw_rate_day == segments[i].rw.rate_day_last, --rewards ignored (max |diff|)",
               float(np.max(np.abs(ext["rw_rate_day"] - [s["rw"]["rate_day_last"] for s in meta["segments"]]))), 0)
    # TRADES_TRUNCATED: 1024 in sports-2 (bit 32 set on game 5 must not count), 32 in sports-1
    tr = {d: json.loads((root / d / "meta3.json").read_text()).get("trades_truncated_markets")
          for d in ("sports", "sports1", "sports2_nolegacy")}
    checks.add("(e) TRADES_TRUNCATED read as 1024 (sports-2) / 32 (sports-1), bit 32 ignored in sports-2 (1 = not)",
               float(any(v != [3] for v in tr.values())), 0)
    # --no-legacy-settle: no flag 16, winner_up -1; settles at segments[i].settle_price == the last-row mark
    nl = Data3(root / "sports2_nolegacy")
    seg = json.loads((root / "sports2_nolegacy" / "meta.json").read_text())["segments"]
    last = nl.markets["row_offset"] + nl.markets["n_rows"] - 1
    checks.add("(e) --no-legacy-settle build: no flag 16, winner_up -1, log_strike 0 (1 = not)",
               float((nl.markets["flags"] & 16).any() or (nl.markets["winner_up"] != -1).any()
                     or (nl.markets["log_strike_vs_ul_open"] != 0).any()), 0)
    checks.add("(e) --no-legacy-settle: settle_price == segments settle_price == last-row mark, mode 1",
               float(np.max(np.abs(nl.ext["settle_price"] - [s["settle_price"] for s in seg])
                            + np.abs(nl.ext["settle_price"] - nl.feat[last, nl.col["mark"]]))
                     + np.any(nl.ext["settle_mode"] != 1)), 0)
    # a legacy copy that disagrees with meta is refused; a sports-1 build without flag 16 is refused
    for name, src, edit in (("legacy settle copy != segments settle_price", "sports", "log_strike_vs_ul_open"),
                            ("sports-1 build without flag 16", "sports1", "flags")):
        bad = Path(tmp) / f"bad_{src}"
        shutil.copytree(root / src, bad)
        mk = np.fromfile(bad / "markets.bin", MARKET_DTYPE)
        mk[edit][0] = 0.123 if edit == "log_strike_vs_ul_open" else mk["flags"][0] & ~16
        mk.tofile(bad / "markets.bin")
        try:
            adapt3.main([str(bad), "--check"])
            refused = 0.0
        except SystemExit:
            refused = 1.0
        checks.add(f"(e) adapt3 refuses a {name} (1 = not)", 1 - refused, 0)
    # --bucket-segments: event_ts = kickoff for every piece, so time to event = time to kickoff
    bd = Data3(root / "sports2_buckets")
    bseg = json.loads((root / "sports2_buckets" / "meta.json").read_text())["segments"]
    mk, ext = bd.markets, bd.ext
    last = mk["row_offset"] + mk["n_rows"] - 1
    checks.add("(e) bucket pieces: event_ts == kickoff_ts (not the piece end) (max |diff| s)",
               float(np.max(np.abs(ext["event_ts"] - [s["kickoff_ts"] for s in bseg]))), 0)
    checks.add("(e) bucket pieces: pure, end at 1800 / 300 / 0 s before kickoff, settle at their last-row mark",
               float(np.max(np.abs(ext["settle_price"] - bd.feat[last, bd.col["mark"]])))
               + float(not set((ext["event_ts"] - mk["end_ts"]).tolist()) <= {0, 300, 1800})
               + float(not all(s["bucket_pure"] for s in bseg)), 0)
    n = len(mk)
    h.open(root / "sports2_buckets")
    obs = h.run(Config3(obs_size=80), np.arange(n), np.zeros(n), np.zeros((1, n, 3), np.int32))[0]
    ev = np.expm1(obs[0, :, 50].astype(np.float64) * np.log1p(86400.0))
    checks.add("(e) bucket pieces: obs 50 at row 0 == seconds to kickoff (rel)",
               rel(ev, (ext["event_ts"] - mk["open_ts"]).astype(np.float64)), 1e-5)
    rng = np.random.default_rng(31)
    starts = np.array([int(rng.integers(0, max(1, int(0.2 * (mk["n_rows"][i] - 1))))) for i in range(n)])
    out = equivalence(h, bd, checks, replace(Config3(), obs_size=80, liq_rewards=1, size_mode=1, Q=1000,
                                                 pull_s=600, flat_c=0.5, close_inv_coef=0.05, close_ramp_s=1800),
                      np.arange(n), starts, np.random.default_rng(32), "sports-buckets-pull600-flat-liq")
    print(f"(b) {'sports-buckets-pull600-flat-liq':40s} {json.dumps(out)}", flush=True)
    results["adapt_schemas"] = dict(trades_truncated=tr, buckets=out)


# ------------------------------------------------------------------ (f) D15 netting / flatten diagnostics

def netting_checks(h, root, checks, results):
    """The Log's net_* fields against the fields they summarise, a never-quoting policy, and env3.netting."""
    import env3
    sd = root / "sports"
    data = Data3(sd)
    mk = data.markets
    n = len(mk)
    T = int(mk["n_rows"].max()) + 5
    acts = random_actions(np.random.default_rng(41), T, n, p_off=0.0)
    acts[:, :, 2] = 2
    h.open(sd)
    lg = h.run(Config3(Q=1000, dt_other=1, flat_c=0.5), np.arange(n), np.zeros(n), acts, seed=5)[4]
    g, end = lg["net_gross_sh"].astype(np.float64), lg["net_end_inv"].astype(np.float64)
    checks.add("(f) net_end_inv == final_abs_inv (|inventory| at the settlement row)",
               float(np.max(np.abs(end - lg["final_abs_inv"]))), 0)
    checks.add("(f) net_gross_sh == fill_vol == buy_vol + sell_vol (rel)",
               max(rel(g, lg["fill_vol"]), rel(g, lg["buy_vol"].astype(np.float64) + lg["sell_vol"])), 1e-5)
    checks.add("(f) net_flat_drag == flat_cost == |end inv| x flat_c / 100 (USD)",
               float(max(np.max(np.abs(lg["net_flat_drag"] - lg["flat_cost"])),
                         np.max(np.abs(lg["net_flat_drag"] - end * 0.005)))), 1e-6)
    has = g > 0
    want = np.where(has, np.maximum(1 - end / np.where(has, g, 1), 0), 0)
    checks.add("(f) net_share == 1 - net_end_inv / net_gross_sh per episode, in [0, 1]",
               float(np.max(np.abs(lg["net_share"] - want)) + np.any((lg["net_share"] < 0) | (lg["net_share"] > 1))),
               1e-6)
    checks.add("(f) net_eps == 1 exactly for episodes with fills", float(np.max(np.abs(lg["net_eps"] - has))), 0)
    checks.add("(f) net_notional / net_gross_sh is a price in (0, 1) (1 = not)",
               float(not ((lg["net_notional"][has] > 0).all() and (lg["net_notional"][has] < g[has]).all())), 0)
    tot = {k: float(np.sum(v, dtype=np.float64)) for k, v in lg.items()}
    nt = env3.netting(tot)
    checks.add("(f) env3.netting: pooled netted share == 1 - sum |end inv| / sum gross (RECONCILE.md section 6)",
               abs(nt["netted_share"] - (1 - end.sum() / g.sum())), 1e-6)
    checks.add("(f) env3.netting: flat_drag_bps == drag / notional x 1e4",
               abs(nt["flat_drag_bps"] - tot["net_flat_drag"] / tot["net_notional"] * 1e4), 1e-6)
    off = acts.copy()
    off[:, :, :2] = 0
    lo = h.run(Config3(Q=1000, dt_other=1, flat_c=0.5), np.arange(n), np.zeros(n), off, seed=5)[4]
    checks.add("(f) never quoting: gross, end inventory, netted share, episodes with fills, drag all 0",
               float(sum(np.abs(lo[k]).sum() for k in ("net_gross_sh", "net_end_inv", "net_share", "net_eps",
                                                       "net_flat_drag", "net_notional"))), 0)
    results["netting"] = dict(netted_share_pooled=nt["netted_share"], netted_share_ep=nt["netted_share_ep"],
                              flat_drag_bps=nt["flat_drag_bps"], gross=g.tolist(), end_inv=end.tolist())


# ------------------------------------------------------------------ (d) reward hook

def reward_hook(h, root, checks, results):
    u = fixtures3.UNIT
    assert abs(0.01 * 400 / 3100 - u["rate"] / 1440.0 * (400 / 27) / (400 / 27 + 100)) < 1e-15
    pi = float(np.float32(u["rate"] / 1440.0))      # x3.bin stores the per-sample pool as float32
    hand = {}
    # two-sided: bid 0.49 (d = 1 c), ask 0.52 (d = 2 c), 100 shares each, v = 3 c, mid 0.50, Q_comp 100
    q1, q2 = (2 / 3) ** 2 * 100, (1 / 3) ** 2 * 100
    qmin = max(min(q1, q2), max(q1, q2) / 3)
    hand["two_sided"] = pi * qmin / (qmin + u["comp"])      # = 0.01 * 400 / 3100 with an exact pool
    hand["one_sided_bid_at_mid"] = pi * (100 / 3) / (100 / 3 + u["comp"])      # bid 0.50 (d = 0), ask off
    hand["too_small"] = 0.0                                                   # size_mode 0: 5 shares < 100
    hand["too_young"] = 0.0                                                   # liq_moas beyond the sample
    hand["no_competitor"] = pi                                                # Q_comp 0: share 1
    cases = {"two_sided": ((3, 4, 0), dict(), u["comp"]), "one_sided_bid_at_mid": ((2, 0, 0), dict(), u["comp"]),
             "too_small": ((3, 4, 0), dict(size_mode=0), u["comp"]),
             "too_young": ((3, 4, 0), dict(liq_moas=1000), u["comp"]),
             "no_competitor": ((3, 4, 0), dict(), 0.0)}
    out = {}
    for name, (a, kw, comp) in cases.items():
        d = root / f"unit_{name}"
        fixtures3.make_unit(d, comp=comp)
        data = Data3(d)
        cfg = replace(Config3(size_mode=1, liq_rewards=1, Q=1000, dt_other=1, obs_size=80, fixed_start=0), **kw)
        T = u["n"] - 1
        acts = np.tile(np.array(a, np.int32), (T, 1, 1))
        h.open(d)
        obs, rew, term, state, lg = h.run(cfg, np.array([0]), np.array([0]), acts)
        ep = Episode3(data, cfg, 0, 0)
        for k in range(T):
            ep.step(a)
        out[name] = dict(c=float(lg["liq_rewards"][0]), ref=ep.liq, hand=hand[name], rw_scored=float(lg["rw_scored"][0]))
        checks.add(f"(d) reward hook {name}: reference == hand ($)", abs(ep.liq - hand[name]), 1e-15)
        checks.add(f"(d) reward hook {name}: C == hand (rel, float32 Log)", abs(lg["liq_rewards"][0] - hand[name])
                   / max(hand[name], 1e-12), 1e-6)
        checks.add(f"(d) reward hook {name}: C P&L includes it ($)", abs(lg["pnl"][0] - lg["liq_rewards"][0]), 1e-9)
        if name == "two_sided":
            o = obs[150, 0]                             # well after both orders are live
            checks.add("(d) obs rw_cur_share == hand share Q_min / (Q_min + Q_comp)", abs(o[50 + 15] - qmin / (qmin + 100)),
                       1e-6)
            s_bid3 = (2 / 3) ** 2 * 100                 # bid head 3 = join 0.49, eligible size 100
            checks.add("(d) obs bid share of head 3 == (S(1c) * 100) / (S(1c) * 100 + Q_comp)",
                       abs(o[50 + 16 + 2] - s_bid3 / (s_bid3 + 100)), 1e-6)
            checks.add("(d) obs ask share of head 6 (4 ticks behind the 0.51 ask: d = 5 c >= v) == 0", abs(o[50 + 22 + 5]), 0)
            checks.add("(d) obs size eligibility (size_mode 1: all heads >= min size)", abs(o[62:65] - 1).max(), 0)
            checks.add("(d) obs rw_active, rw_scorable, two_sided_only", abs(o[55] - 1) + abs(o[59] - 1) + abs(o[60]), 0)
        if name == "too_small":
            checks.add("(d) obs size eligibility (size_mode 0: 5 / 10 / 25 shares < 100)", obs[150, 0, 62:65].max(), 0)
    # reference3's scorer against analysis/rewards/reward_terms.py (side_scores + q_min), random books
    sys.path.insert(0, str(PM / "analysis" / "rewards"))
    import reward_terms as RT
    rng = np.random.default_rng(0)
    worst = 0.0
    ep = Episode3(Data3(root / "unit_two_sided"), Config3(liq_moas=0), 0, 0)
    for _ in range(2000):
        mid = float(rng.uniform(0.02, 0.98))
        v = float(rng.choice([1.5, 2.0, 3.0, 3.5, 4.5, 6.5]))
        ms = float(rng.choice([0.0, 20.0, 50.0, 100.0]))
        bpx, apx = int(rng.integers(1, 999)), int(rng.integers(1, 999))
        bs, as_ = float(rng.uniform(1, 300)), float(rng.uniform(1, 300))
        live = rng.random(2) < 0.85
        ep.orders[0].live, ep.orders[0].px, ep.orders[0].rem, ep.orders[0].born = bool(live[0]), bpx, bs, 0
        ep.orders[1].live, ep.orders[1].px, ep.orders[1].rem, ep.orders[1].born = bool(live[1]), apx, as_, 0
        ours = ep.our_qmin(10, mid, v, ms)
        cfg = replace(RT.AUG_2026, reward_max_spread_c=v, reward_min_size=ms)
        b = ([bpx / 1000], [bs]) if live[0] else ([], [])
        a_ = ([apx / 1000], [as_]) if live[1] else ([], [])
        q1, q2 = RT.side_scores(b[0], b[1], a_[0], a_[1], mid, cfg)
        worst = max(worst, abs(ours - RT.q_min(q1, q2, mid, cfg)))
    checks.add("(d) reference3 Q_min == reward_terms.side_scores + q_min (2000 random cases)", worst, 1e-9)
    results["reward_hook"] = out


# ------------------------------------------------------------------ guards, tables, binding

def tables(checks):
    names = re.findall(r'"(\w+)"', re.search(r"M3_PARAM_NAMES\[\] = \{(.*?)\};", SRC, re.S)[1])
    defs = [float(x) for x in re.search(r"M3_DEFAULTS\[\] = \{(.*?)\};", SRC, re.S)[1].split(",") if x.strip()]
    c = Config3()
    checks.add("parameters: mm3.h names == Config3 (1 = not)", float(names != Config3.names()), 0)
    checks.add("parameters: mm3.h defaults == Config3", float(np.max(np.abs(np.array(defs) - c.values()))), 0)
    v2 = (PM / "mm.h").read_text()
    n2 = re.findall(r'"(\w+)"', re.search(r"MM_PARAM_NAMES\[\] = \{(.*?)\};", v2, re.S)[1])
    d2 = [float(x) for x in re.search(r"MM_DEFAULTS\[\] = \{(.*?)\};", v2, re.S)[1].split(",") if x.strip()]
    checks.add("parameters: the first 27 are v2's names (1 = not)", float(names[:27] != n2), 0)
    diff = [k for k, a, b in zip(n2, defs, d2) if a != b]
    checks.add("parameters: v2 defaults kept except len_mask 7 -> 15 (1 = not)", float(diff != ["len_mask"]), 0)
    v2log = re.findall(r"float (\w+);", re.search(r"struct Log \{(.*?)\n\};", v2, re.S)[1])
    checks.add("Log: the first 42 fields are v2's, n last (1 = not)",
               float(LOG_FIELDS[:42] != v2log[:42] or LOG_FIELDS[-1] != "n"), 0)


def split_guard(h, checks):
    h.open(V2_DATA)
    try:
        h.run(Config3(split=2), np.array([-1]), np.array([0]), np.zeros((1, 1, 3), np.int32))
        refused = 0.0
    except ValueError as e:
        refused = float("refused" in str(e))
    checks.add("split guard: split=2 without allow_test refused (1 = not)", 1 - refused, 0)


BINDING_CHILD = r"""
import sys, json, numpy as np
sys.path.insert(0, %(here)r)
from env3 import MmEnv3
kw = json.loads(%(kw)r)
acts = np.load(%(acts)r)
import env3
env = MmEnv3(num_envs=acts.shape[1], seed=%(seed)d, data_dir=%(data)r, log_interval=10**9, **kw)
env.reset(%(seed)d)
obs = [env.observations.copy()]; rew = []; term = []
for t in range(acts.shape[0]):
    o, r, d, _, _ = env.step(acts[t])
    obs.append(o.copy()); rew.append(r.copy()); term.append(d.astype(np.float32))
log = env._binding.vec_log(env.c_envs)
log.update(env3.netting(log))
env.close()
np.savez(%(out)r, obs=np.array(obs), rew=np.array(rew), term=np.array(term), log=json.dumps(log))
"""


def binding_matches_core(h, tmp, checks, data_dir, kw, tag, n=32, T=600, seed=3):
    if not glob.glob(str(HERE / "binding3*.so")):
        print("binding3 not built; skipped")
        return
    rng = np.random.default_rng(seed)
    acts = random_actions(rng, T, n)
    ap, op = Path(tmp) / f"acts_{tag}.npy", Path(tmp) / f"bind_{tag}.npz"
    np.save(ap, acts)
    code = BINDING_CHILD % dict(here=str(HERE), kw=json.dumps(kw), acts=str(ap), seed=seed, data=str(data_dir),
                                out=str(op))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp))
    assert r.returncode == 0, r.stderr[-3000:]
    z = np.load(op)
    h.open(data_dir)
    cfg = replace(Config3(), **kw)
    obs, rew, term, _, _ = h.run(cfg, -np.ones(n, np.int32), np.zeros(n, np.int32), acts, seed=seed * n)
    err = max(np.abs(z["obs"] - obs).max(), np.abs(z["rew"] - rew).max(), np.abs(z["term"] - term).max())
    checks.add("PufferLib 3.0 binding3 == core (same seeds and actions)", err, 0)
    checks.add("binding run crossed at least one terminal (1 = no)", float(term.sum() == 0), 0)
    lg = json.loads(str(z["log"]))
    missing = [f for f in LOG_FIELDS if f not in lg]
    checks.add("binding3 vec_log carries every Log field incl. the D15 net_* fields (missing count)", len(missing), 0)
    ok = (lg.get("net_eps", 0) == 0 or 0 <= lg["netted_share"] <= 1) and lg["end_abs_inv"] == lg["final_abs_inv"]
    checks.add("binding3 / env3.netting: netted_share in [0, 1], end_abs_inv == final_abs_inv (1 = not)", float(not ok), 0)


def builders_selftest(tmp):
    """Run analysis/v3_sports/build_sports.py --selftest --no-env into tmp and adapt its build_a."""
    script = PM / "analysis" / "v3_sports" / "build_sports.py"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(script), "--selftest", "--no-env", "--keep", "--tmp", str(tmp)],
                       capture_output=True, text=True, cwd=str(script.parent), timeout=900)
    if r.returncode != 0:
        print("build_sports.py --selftest unavailable here; skipped:", (r.stderr or r.stdout)[-400:])
        return None
    d = sorted(Path(tmp).glob("sports_selftest_*/build_a"))
    if not d:
        return None
    d = d[0]
    meta = json.loads((d / "meta.json").read_text())
    slugs = sorted({s.split("#")[0] for s in meta["slugs"]})
    rates = {s: float(10 * (i + 1)) for i, s in enumerate(slugs)}
    (Path(tmp) / "rates_selftest.json").write_text(json.dumps(rates))
    mk = np.fromfile(d / "markets.bin", MARKET_DTYPE)
    mk["rewards_min_size"], mk["rewards_max_spread"] = 50.0, 3.5      # gamma reward fields (absent offline)
    mk.tofile(d / "markets.bin")
    adapt3.main([str(d), "--rewards", str(Path(tmp) / "rates_selftest.json")])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="shorter v2 comparison")
    ap.add_argument("--no-builders", action="store_true", help="skip build_sports.py --selftest")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    t0 = time.time()
    checks = Checks()
    results = {}
    tables(checks)
    with tempfile.TemporaryDirectory(prefix="mm3_test_") as tmp:
        tmp = Path(tmp)
        h = Harness(tmp)
        root = fixtures3.make_all(tmp / "fx")
        bdir = None if args.no_builders else builders_selftest(tmp / "builders")
        fixture_equivalence(h, root, checks, results, bdir)
        for j, (d, mi) in enumerate(((root / "sports", 1), (root / "elections", 0), (V2_DATA, 12))):
            data = Data3(d)
            h.open(d)
            lookahead(data, checks, Config3(obs_size=80, liq_rewards=1, size_mode=1, Q=1000, fill_model=j % 3),
                      mi, 0.5, np.random.default_rng(j))
        settlement(h, root, tmp, checks, results)
        adapt_schemas(h, root, tmp, checks, results)
        netting_checks(h, root, checks, results)
        reward_hook(h, root, checks, results)
        split_guard(h, checks)
        binding_matches_core(h, tmp, checks, V2_DATA, dict(obs_size=80, liq_rewards=1, size_mult=10, Q=300), "crypto")
        binding_matches_core(h, tmp, checks, root / "elections",
                             dict(obs_size=80, liq_rewards=1, size_mode=1, Q=2000, pull_s=300, flat_c=0.5), "elections")
        v2_bitexact(tmp, checks, args.quick, results)
    bad = checks.report()
    results["checks"] = {k: dict(value=v, tol=t, ok=ok) for k, (v, t, ok) in checks.rows.items()}
    results["seconds"] = time.time() - t0
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=float))
    assert not bad, f"failed: {bad}"
    print(f"ok ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
