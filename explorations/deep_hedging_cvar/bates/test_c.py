"""Check the C env core (hedge.h) against the reference model (market.py).

Builds test_harness.c into a shared library, runs it through ctypes on
injected market noise and recorded actions for two episodes, and replays
every episode through market.simulate in float64. Checks observations,
terminal losses, the 32-step layout and its terminal flags, shaped rewards
(against market.potential and telescoping to the terminal reward), the
unshaped reward, the Robbins-Monro update of w and the Log sums, for
several configurations. Then runs the C generator without injected noise
and checks prices and realized variance against the Fourier model.

    python test_c.py
"""

import ctypes
import math
import os
import re
import subprocess
import tempfile
from dataclasses import replace

import numpy as np
import torch

from market import (BatesConfig, market_noise, simulate, potential, bates_call, OBS_DIM,
                    ACT_DIM)

HERE = os.path.dirname(os.path.abspath(__file__))
DATES, EPISODE = 30, 32
LOG_FIELDS = ['score', 'perf', 'ru', 'loss', 'excess', 'w', 'trades_stock', 'trades_swap',
              'episode_return', 'episode_length', 'n']
PARAM_NAMES = re.findall(r'"(\w+)"', re.search(r'HEDGE_PARAM_NAMES\[\] = \{(.*?)\};',
                                               open(os.path.join(HERE, 'hedge.h')).read(),
                                               re.S)[1])


def build(tmp):
    lib = os.path.join(tmp, 'libhedge_test.so')
    subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-shared', '-fPIC', '-o', lib,
                    os.path.join(HERE, 'test_harness.c'), '-lm'], check=True)
    return ctypes.CDLL(lib)


def c_values(cfg, w_init=0.3, w_eta=0.01, shaping=1):
    kw = {**cfg.c_kwargs(), 'w_init': w_init, 'w_eta': w_eta, 'shaping': shaping}
    return np.array([float(kw[k]) for k in PARAM_NAMES])


def run_c(lib, values, actions, noise=None, seed=0):
    """actions (T, N, 4); noise (N, episodes, 30, n_sub, 4) or None."""
    T, n = actions.shape[:2]
    obs = np.zeros((T + 1, n, OBS_DIM), np.float32)
    rewards = np.zeros((T, n), np.float32)
    terminals = np.zeros((T, n), np.float32)
    state = np.zeros((T + 1, n, 4))
    logs = np.zeros((n, len(LOG_FIELDS)), np.float32)
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    acts = np.ascontiguousarray(actions, np.float32)
    nz = None if noise is None else np.ascontiguousarray(noise, np.float64)
    rc = lib.hedge_run(ptr(values), n, T, ptr(acts), None if nz is None else ptr(nz),
                       ctypes.c_uint64(seed), ptr(obs), ptr(rewards), ptr(terminals),
                       ptr(state), ptr(logs))
    assert rc == 0
    return obs, rewards, terminals, dict(zip(['k', 'w', 'phi', 'loss'], state.transpose(2, 0, 1))), \
        dict(zip(LOG_FIELDS, logs.T))


def reference(cfg, noise, w, actions):
    """Replay one episode through market.simulate (float64). Returns the
    observations at dates 0..29, the loss and the potential at dates 0..30."""
    obs = []

    def replay(o):
        obs.append(o.clone())
        return torch.from_numpy(actions[len(obs) - 1]).double()
    loss, rec = simulate(torch.from_numpy(noise), replay, torch.from_numpy(w), cfg, record=True)
    obs = torch.stack(obs)
    phi = []
    for k in range(DATES):
        o = obs[k]
        s = cfg.strike_call * torch.exp(o[:, 1] * cfg.sigma0 * math.sqrt(cfg.maturity))
        v = (o[:, 2] * cfg.sigma0)**2
        swap, wealth = o[:, 9] * cfg.scale, o[:, 5] * cfg.scale
        cash = wealth - o[:, 3] * s - o[:, 4] * swap
        phi.append(potential(k, s, v, o[:, 3], o[:, 4], cash, torch.from_numpy(w), swap,
                             o[:, 10] * cfg.theta * cfg.maturity, cfg))
    return obs.numpy(), loss.numpy(), torch.stack(phi).numpy()


class Checks:
    def __init__(self):
        self.rows = {}

    def add(self, name, value, tol):
        value = float(value)
        ok = value <= tol and np.isfinite(value)
        prev = self.rows.get(name, (0.0, tol, True))
        self.rows[name] = (max(prev[0], value), tol, prev[2] and ok)

    def report(self):
        for name, (value, tol, ok) in self.rows.items():
            print(f'{name:58s} {value:9.2e}  tol {tol:7.0e}  {"ok" if ok else "FAIL"}')
        assert all(ok for _, _, ok in self.rows.values()), 'some checks failed'


def rel(a, b):
    return np.max(np.abs(a - b) / (1 + np.abs(b)))


def equivalence(lib, checks, cfg, shaping, w_init, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    T = 2 * EPISODE
    actions = np.stack([rng.uniform(-1.8, 1.8, (T, n)), rng.normal(0, 1, (T, n)),
                        rng.uniform(-3.5, 3.5, (T, n)), rng.normal(0, 1, (T, n))],
                       -1).astype(np.float32)
    noise = market_noise(2 * n, cfg, torch.Generator().manual_seed(seed)).double().numpy()
    noise = noise.reshape(2, n, DATES, cfg.n_sub, 4).transpose(1, 0, 2, 3, 4)
    obs, rewards, terminals, state, log = run_c(lib, c_values(cfg, w_init, 0.01, shaping),
                                                actions, noise)
    flags = np.zeros(T)
    flags[[DATES - 1, EPISODE - 1, EPISODE + DATES - 1, 2 * EPISODE - 1]] = 1
    checks.add('terminal flags on expiry and reset steps only',
               np.abs(terminals - flags[:, None]).max(), 0)
    checks.add('observation 11 == 1', np.abs(obs[..., 11] - 1).max(), 0)
    sums = {f: 0.0 for f in LOG_FIELDS}
    for e in range(2):
        t0 = e * EPISODE
        w = state['w'][t0]
        ref_obs, loss, phi = reference(cfg, noise[:, e], w, actions[t0:t0 + DATES])
        checks.add('observations at dates 0-29 (rel err)',
                   rel(obs[t0:t0 + DATES].astype(np.float64), ref_obs), 1e-6)
        checks.add('terminal loss', np.abs(state['loss'][t0 + DATES] - loss).max(), 1e-9)
        checks.add('padding step keeps the expiry observation',
                   np.abs(obs[t0 + DATES + 1] - obs[t0 + DATES]).max(), 0)
        checks.add('obs at expiry: date fraction 1', np.abs(obs[t0 + DATES, :, 0] - 1).max(), 0)
        checks.add('potential at the episode start', np.abs(state['phi'][t0] - phi[0]).max(), 1e-9)
        r = rewards[t0:t0 + DATES].astype(np.float64)
        term = -np.maximum(loss - w, 0) / cfg.scale
        checks.add('reward 0 on the two steps after expiry',
                   np.abs(rewards[t0 + DATES:t0 + EPISODE]).max(), 0)
        if shaping:
            target = np.diff(np.concatenate([phi, term[None]]), axis=0)
            checks.add('shaped reward == Phi(k+1) - Phi(k) (rel err)', rel(r, target), 1e-6)
            checks.add('shaped rewards + Phi(0) == -(L - w)^+ / scale',
                       np.abs(r.sum(0) + phi[0] - term).max(), 1e-5)
        else:
            checks.add('unshaped reward at expiry (rel err)', rel(r[-1], term), 1e-6)
            checks.add('unshaped reward 0 before expiry', np.abs(r[:-1]).max(), 0)
        excess = np.maximum(loss - w, 0)
        ru = w + excess / (1 - cfg.alpha)
        trades = (actions[t0:t0 + DATES, :, [1, 3]] > 0).sum(0)
        for f, v in [('score', -ru), ('perf', -ru / cfg.scale), ('ru', ru), ('loss', loss),
                     ('excess', excess), ('w', w), ('trades_stock', trades[:, 0]),
                     ('trades_swap', trades[:, 1]), ('episode_length', EPISODE), ('n', 1)]:
            sums[f] = sums[f] + v
        sums['episode_return'] = sums['episode_return'] + r.sum(0)
        if e == 0:
            w_next = w + 0.01 * cfg.scale * ((loss > w) / (1 - cfg.alpha) - 1)
            checks.add('w of episode 2: Robbins-Monro step',
                       np.abs(state['w'][EPISODE] - w_next).max(), 1e-12)
    for f in LOG_FIELDS:
        checks.add('Log sums after two episodes (rel err)', rel(log[f], sums[f]), 1e-5)


def generator(lib, checks, cfg, n=20000, episodes=8):
    """Own generator, no trades: S_T and realized variance against the model."""
    T = episodes * EPISODE
    actions = np.zeros((T, n, ACT_DIM), np.float32)
    actions[..., 1] = actions[..., 3] = -1
    obs, _, _, _, _ = run_c(lib, c_values(cfg), actions, None, seed=12)
    ends = obs[np.arange(episodes) * EPISODE + DATES].astype(np.float64)
    s = cfg.strike_call * np.exp(ends[..., 1] * cfg.sigma0 * math.sqrt(cfg.maturity)).ravel()
    rv = ends[..., 10].ravel() * cfg.theta  # realized / T
    m = s.size
    call = np.maximum(s - cfg.strike_call, 0)
    put = np.maximum(cfg.strike_put - s, 0)
    fc = float(bates_call(cfg.s0, cfg.strike_call, cfg.maturity, cfg.v0, cfg))
    fp = float(bates_call(cfg.s0, cfg.strike_put, cfg.maturity, cfg.v0, cfg)) - cfg.s0 + cfg.strike_put
    checks.add('generator: E[S_T] - S0 (in s.e.)', abs(s.mean() - cfg.s0) / (s.std() / m**0.5), 4)
    checks.add('generator: call price vs Fourier (in s.e.)', abs(call.mean() - fc) / (call.std() / m**0.5), 4)
    checks.add('generator: put price vs Fourier (in s.e.)', abs(put.mean() - fp) / (put.std() / m**0.5), 4)
    checks.add('generator: E[realized]/T vs K_var (in s.e.)',
               abs(rv.mean() - cfg.k_var) / (rv.std() / m**0.5), 4)
    print(f'generator: {m} episodes, E[S_T] {s.mean():.3f}, call {call.mean():.4f} '
          f'(Fourier {fc:.4f}), put {put.mean():.4f} ({fp:.4f}), E[RV]/T {rv.mean():.5f} '
          f'(K_var {cfg.k_var:.5f})')


def binding_matches_core(lib, checks, cfg, n=512, T=3 * EPISODE):
    """The PufferLib 3.0 env (binding.c) steps exactly like the core."""
    try:
        from env import BatesEnv
    except ImportError as e:
        print(f'binding not built ({e}); skipped')
        return
    rng = np.random.default_rng(5)
    actions = np.stack([rng.uniform(-1.8, 1.8, (T, n)), rng.normal(0, 1, (T, n)),
                        rng.uniform(-3.5, 3.5, (T, n)), rng.normal(0, 1, (T, n))],
                       -1).astype(np.float32)
    obs, rewards, terminals, _, log = run_c(lib, c_values(cfg), actions, None, seed=0)
    env = BatesEnv(cfg, num_agents=n, seed=0)
    env.reset(0)
    err = np.abs(env.observations - obs[0]).max()
    for t in range(T):
        o, r, d, _, _ = env.step(actions[t])
        err = max(err, np.abs(o - obs[t + 1]).max(), np.abs(r - rewards[t]).max(),
                  np.abs(d.astype(np.float32) - terminals[t]).max())
    env.close()
    checks.add('PufferLib 3.0 binding == core (same seeds and actions)', err, 0)


def main():
    torch.set_default_dtype(torch.float64)
    checks = Checks()
    base = BatesConfig()
    configs = [(base, 1, 0.3), (base, 0, 0.3),
               (replace(base, kappa=0.0, fixed_cost=0.0, vs_cost=0.0, lam=0.0), 1, -0.2),
               (replace(base, kappa=2.0, fixed_cost=0.1, half_life=3.0, n_sub=2, lam=8.0), 1, 0.5)]
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(tmp)
        for i, (cfg, shaping, w_init) in enumerate(configs):
            equivalence(lib, checks, cfg, shaping, w_init, seed=i)
        generator(lib, checks, base)
        binding_matches_core(lib, checks, base)
    checks.report()
    print('ok')


if __name__ == '__main__':
    main()
