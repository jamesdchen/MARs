"""Check the C core (mm.h) against the numpy reference (reference.py), and the env's contracts.

1. Equivalence: mm_harness.c is built into a shared library and run through ctypes on fixed
   markets, start offsets and action sequences (train and val markets only). Every config
   (3 fill models x 3 latencies, plus mark modes, reward probe, tick rule, liquidity rewards,
   explicit delays, dt, size and penalty settings) must reproduce the reference: rewards,
   terminal flags, fills, inventory, cash, rebates, order state and observations at every step,
   and the episode Log. The steps run past the terminal, so the auto-reset episode (new market
   and offset from the env's own RNG) is checked too.
2. Telescoping: the sum of rewards / scale plus the inventory penalty equals the final P&L
   (cash + rebates + rewards + inventory * settlement) for mark_mode 0, 1 and 2.
3. Lookahead: in the reference, perturbing every data row after a cut leaves all observations
   and rewards up to the cut unchanged.
4. Split guard: split=2 is refused unless allow_test=1 AND MM_FINAL=1.
5. The PufferLib 3.0 binding (env.py) steps exactly like the core.

    python test_env.py            (~2-4 min)
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from reference import Config, Data, Episode, OBS

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "build"
NSTATE = 24
LOG_FIELDS = ["score", "perf", "episode_return", "episode_length", "pnl", "pnl_trading",
              "spread_capture", "markout_5", "markout_30", "markout_60", "inventory_pnl", "rebates",
              "liq_rewards", "inv_penalty", "fills", "fill_vol", "buy_vol", "sell_vol", "max_abs_inv",
              "final_abs_inv", "uptime_two_sided", "uptime_any", "vol_crossed", "vol_stale",
              "vol_onesided", "vol_last30", "vol_last10", "vol_inside2", "vol_extreme", "vol_outside",
              "reward_clip_frac", "rejects", "n"]
STATE_NAMES = ["t", "market", "s0", "inv", "cash", "rebates", "liq", "wealth", "penalty", "n_fills",
               "fill_vol", "bid_live", "bid_px", "bid_rem", "bid_q", "ask_live", "ask_px", "ask_rem",
               "ask_q", "spread", "mo5", "mo30", "mo60", "rejects"]
FLOAT_STATE = {"n_fills", "fill_vol", "rejects"}   # float32 counters in C (tolerance 1e-4)


def build(tmp):
    lib = os.path.join(tmp, "libmm_test.so")
    subprocess.run([os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off",
                    "-D_POSIX_C_SOURCE=200809L", "-shared",
                    "-fPIC", "-o", lib, str(HERE / "mm_harness.c"), "-lm"], check=True)
    lib = ctypes.CDLL(lib)
    err = ctypes.create_string_buffer(512)
    assert lib.mm_h_open(str(DATA_DIR).encode(), err, 512) == 0, err.value
    assert lib.mm_h_sizeof_log() == 4 * len(LOG_FIELDS)
    return lib


def run_c(lib, cfg: Config, markets, starts, actions, seed=0):
    T, n = actions.shape[:2]
    obs = np.zeros((T + 1, n, OBS), np.float32)
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
    rc = lib.mm_h_run(p(vals), n, T, p(mk), p(st), p(act), ctypes.c_uint64(seed), p(obs), p(rew), p(term),
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
            print(f"{name:66s} {value:9.2e}  tol {tol:7.0e}  {'ok' if ok else 'FAIL'}")
        bad = [k for k, (_, _, ok) in self.rows.items() if not ok]
        assert not bad, f"failed: {bad}"


def random_actions(rng, T, n):
    """Sticky random actions biased toward the touch, so quotes rest, queue and fill."""
    p_lvl = np.array([0.08, 0.07, 0.12, 0.35, 0.18, 0.12, 0.08])
    a = np.zeros((T, n, 3), np.int32)
    cur = np.stack([rng.choice(7, n, p=p_lvl), rng.choice(7, n, p=p_lvl), rng.integers(0, 3, n)], 1)
    for t in range(T):
        new = rng.random(n) < 0.35
        cur[new] = np.stack([rng.choice(7, new.sum(), p=p_lvl), rng.choice(7, new.sum(), p=p_lvl),
                             rng.integers(0, 3, new.sum())], 1)
        a[t] = cur
    return a


def pick_markets(data, rng, n_train=14, n_val=6, n_4h=2):
    mk = data.markets
    has = (mk["flags"] & 1) > 0
    idx = []
    for split, k in ((0, n_train), (1, n_val)):
        short = np.flatnonzero((mk["split"] == split) & has & (mk["contract_len_s"] < 14400))
        idx += list(rng.choice(short, k, replace=False))
    long = np.flatnonzero((mk["split"] <= 1) & has & (mk["contract_len_s"] == 14400))
    idx += list(rng.choice(long, n_4h, replace=False))
    starts = [int(rng.integers(0, int(0.2 * mk["contract_len_s"][i]) + 1)) for i in idx]
    return np.array(idx), np.array(starts)


def episode_steps(data, cfg, market, start):
    L = int(data.markets["contract_len_s"][market])
    dt = int(cfg.dt) if cfg.dt > 0 else int({300: cfg.dt_5m, 900: cfg.dt_15m, 14400: cfg.dt_4h}[L])
    start = min(start, L - 1)
    return -(-(L - start) // dt)


def rel(a, b):
    return np.max(np.abs(np.asarray(a, np.float64) - b) / (1 + np.abs(b))) if np.size(a) else 0.0


def equivalence(lib, data, checks, cfg: Config, markets, starts, rng, tag, extra=25, seed=11):
    steps = [episode_steps(data, cfg, m, s) for m, s in zip(markets, starts)]
    T = max(steps) + extra
    actions = random_actions(rng, T, len(markets))
    obs, rew, term, state, log = run_c(lib, cfg, markets, starts, actions, seed=seed)
    checks.add("observations finite, |obs| <= 5, last == 1",
               0 if (np.isfinite(obs).all() and np.abs(obs).max() <= 5 and (obs[..., -1] == 1).all()) else 1, 0)
    scale = cfg.reward_scale if cfg.reward_scale > 0 else 1 / cfg.Q
    for i, (m, s) in enumerate(zip(markets, starts)):
        ep = Episode(data, cfg, int(m), int(s))
        k_end = steps[i] - 1
        checks.add(f"[{tag}] initial observation", np.abs(obs[0, i] - ep.obs()).max(), 1e-6)
        rs = []
        for k in range(k_end + 1):
            o, r, done = ep.step(actions[k, i])
            rs.append(r)
            checks.add("reward == reference", abs(rew[k, i] - r), 1e-7)
            checks.add("terminal flag", abs(term[k, i] - float(done)), 0)
            if not done:
                checks.add("observation == reference", np.abs(obs[k + 1, i] - o).max(), 1e-6)
                st = state[k + 1, i]
                ref = ep.state()
                for j, name in enumerate(STATE_NAMES):
                    tol = 1e-4 if name in FLOAT_STATE else 1e-9
                    checks.add(f"state {name} (rel)", rel(st[j], ref[j]), tol)
        assert done, "episode did not end where expected"
        ref_log = ep.log()
        for f in LOG_FIELDS:
            tol = 2e-4 if f in ("episode_return",) else 1e-4
            checks.add(f"Log {f} (rel)", rel(log[f][i], ref_log[f]), tol)
        if cfg.reward_mode == 0 and ref_log["reward_clip_frac"] == 0:
            pnl = ref_log["pnl"]
            checks.add(f"telescoping: sum r/scale + penalty == P&L, mark_mode {int(cfg.mark_mode)} ($)",
                       abs(np.sum(np.asarray(rew[:k_end + 1, i], np.float64)) / scale + ep.penalty - pnl), 2e-4)
            checks.add("P&L == cash + rebates + rewards + inv * settlement ($)",
                       abs(pnl - (ep.cash + ep.rebates + ep.liq + ep.inv * ep.winner)), 1e-9)
        # auto-reset: the next episode on the env's own market and offset
        nm, ns = int(state[k_end + 1, i, 1]), int(state[k_end + 1, i, 2])
        nxt = Episode(data, cfg, nm, ns - int(data.markets["row_offset"][nm]))
        checks.add("auto-reset: new episode's first observation", np.abs(obs[k_end + 1, i] - nxt.obs()).max(), 1e-6)
        for k in range(k_end + 1, T):
            o, r, done = nxt.step(actions[k, i])
            checks.add("auto-reset episode: reward", abs(rew[k, i] - r), 1e-7)
            if done:
                break
            checks.add("auto-reset episode: observation", np.abs(obs[k + 1, i] - o).max(), 1e-6)
    return float(sum(log["fills"]))


class _Rows:
    """Global-row view of a local copy (for the lookahead test)."""

    def __init__(self, arr, base):
        self.arr, self.base = arr, base

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r = key[0]
            return self.arr[(r - self.base,) + key[1:]]
        return self.arr[key - self.base]


def lookahead(data, checks, cfg, market, start, cut_frac, rng):
    """Perturb every row after the cut; observations / rewards up to the cut must not change."""
    m = data.markets[market]
    r0, n = int(m["row_offset"]), int(m["n_rows"])
    cut = r0 + int(cut_frac * n)

    class Local:
        pass
    loc = Local()
    loc.markets, loc.col, loc.N, loc.G = data.markets, data.col, data.N, data.G
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
    a = random_actions(rng, episode_steps(data, cfg, market, start), 1)[:, 0]
    e1, e2 = Episode(data, cfg, market, start), Episode(loc, cfg, market, start)
    diff = np.abs(e1.obs() - e2.obs()).max()
    for k in range(len(a)):
        if e1.t + e1.dt > cut:
            break
        o1, r1, _ = e1.step(a[k])
        o2, r2, _ = e2.step(a[k])
        causal = [j for j in range(NSTATE) if STATE_NAMES[j] not in ("mo5", "mo30", "mo60")]  # Log-only markouts
        diff = max(diff, np.abs(o1 - o2).max(), abs(r1 - r2), np.abs(e1.state()[causal] - e2.state()[causal]).max())
    checks.add("lookahead: perturbing rows after t changes nothing up to t", diff, 0)
    # and the perturbation does matter afterwards (the test has teeth)
    changed = 0.0
    for k in range(k, len(a)):
        o1, r1, d1 = e1.step(a[k])
        o2, r2, d2 = e2.step(a[k])
        if d1:
            break
        changed = max(changed, np.abs(o1 - o2).max())
    checks.add("lookahead: perturbation is visible after the cut (1 = not)", float(changed == 0), 0)


def split_guard(lib, checks):
    cfg = Config(split=2)
    try:
        run_c(lib, cfg, np.array([-1]), np.array([0]), np.zeros((1, 1, 3), np.int32))
        refused = 0.0
    except ValueError as e:
        refused = float("refused" in str(e))
    checks.add("split guard: split=2 without allow_test refused (1 = refused)", 1 - refused, 0)
    code = ("import sys; sys.path.insert(0, %r); from env import MmEnv\n"
            "e = MmEnv(num_envs=2, split=2, allow_test=%d); e.reset(); print('OPENED')")
    outcomes = {}
    for allow, final in ((1, None), (0, "1"), (1, "1")):
        env = {k: v for k, v in os.environ.items() if k != "MM_FINAL"}
        if final:
            env["MM_FINAL"] = final
        r = subprocess.run([sys.executable, "-c", code % (str(HERE), allow)], env=env, capture_output=True,
                           text=True, cwd=str(HERE))
        outcomes[(allow, final)] = ("OPENED" in r.stdout, "TEST SPLIT" in r.stderr + r.stdout)
    checks.add("split guard: allow_test=1 without MM_FINAL=1 refused", float(outcomes[(1, None)][0]), 0)
    checks.add("split guard: MM_FINAL=1 without allow_test=1 refused", float(outcomes[(0, "1")][0]), 0)
    checks.add("split guard: allow_test=1 + MM_FINAL=1 opens with a loud message (1 = not)",
               float(not all(outcomes[(1, "1")])), 0)


def binding_matches_core(lib, checks, n=64, T=600, seed=3):
    try:
        from env import MmEnv
        import binding  # noqa: F401
    except ImportError as e:
        print(f"binding not built ({e}); skipped")
        return
    cfg = Config(fill_model=1, latency=1)
    rng = np.random.default_rng(seed)
    actions = random_actions(rng, T, n)
    obs, rew, term, _, _ = run_c(lib, cfg, -np.ones(n, np.int32), np.zeros(n, np.int32), actions,
                                 seed=seed * n)       # env_binding seeds env i with i + seed * num_envs
    env = MmEnv(num_envs=n, seed=seed, data_dir=str(DATA_DIR), fill_model=1, latency=1)
    env.reset(seed)
    err = np.abs(env.observations - obs[0]).max()
    for t in range(T):
        o, r, d, _, _ = env.step(actions[t])
        err = max(err, np.abs(o - obs[t + 1]).max(), np.abs(r - rew[t]).max(),
                  np.abs(d.astype(np.float32) - term[t]).max())
    env.close()
    checks.add("PufferLib 3.0 binding == core (same seeds and actions)", err, 0)
    checks.add("binding run crossed at least one terminal (1 = no)", float(term.sum() == 0), 0)


def main():
    data = Data()
    checks = Checks()
    rng = np.random.default_rng(2026)
    markets, starts = pick_markets(data, rng)
    base = Config()
    configs = [(f"model{m}-L{L}", replace(base, fill_model=m, latency=L)) for m in (0, 1, 2) for L in (0, 1, 2)]
    configs += [
        ("mark1-penalty", replace(base, mark_mode=1, inv_penalty=0.01)),
        ("mark2-penalty", replace(base, mark_mode=2, inv_penalty=0.01)),
        ("mark0-nopenalty", replace(base, mark_mode=0, inv_penalty=0.0)),
        ("probe", replace(base, reward_mode=1)),
        ("tick-rule1-dt3", replace(base, tick_rule=1, dt=3)),
        ("delay2-nocap-rebate35", replace(base, match_delay=2, part_cap=0, rebate_rate=0.35)),
        ("pess-delay4-L2", replace(base, fill_model=2, match_delay=4, latency=2)),
        ("liq-on-size10-Q300", replace(base, liq_rewards=1, size_mult=10, Q=300, dt_4h=2)),
        ("opt-size0.5-Q20", replace(base, fill_model=0, size_mult=0.5, Q=20, inv_penalty=0.05)),
        ("fee0-reb0-dt15m=2", replace(base, fee_rate=0.0, rebate_rate=0.0, dt_15m=2)),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(tmp)
        fills = {}
        for i, (tag, cfg) in enumerate(configs):
            fills[tag] = equivalence(lib, data, checks, cfg, markets, starts, np.random.default_rng(100 + i), tag)
            print(f"{tag:24s} C fills over {len(markets)} envs: {fills[tag]:.0f}", flush=True)
        checks.add("every config produced fills (1 = some config had none)",
                   float(min(fills.values()) == 0), 0)
        liq = run_c(lib, configs[-3][1], markets, starts,
                    random_actions(np.random.default_rng(7), 400, len(markets)))[4]["liq_rewards"]
        print("liquidity rewards (liq-on config, USD summed over envs):", float(np.sum(liq)))
        for j in range(4):
            mi = int(markets[j])
            lookahead(data, checks, replace(base, fill_model=j % 3, latency=1), mi, int(starts[j]),
                      0.5, np.random.default_rng(j))
        split_guard(lib, checks)
        binding_matches_core(lib, checks)
    checks.report()
    print("ok")


if __name__ == "__main__":
    main()
