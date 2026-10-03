"""Check the PufferLib 5.0 env deep_hedging.h against the reference model.

Compiles c_env_harness.c with the header and runs 2000 envs for two
episodes (64 env steps) on recorded actions (random targets, some outside
the action bounds, and random trade signals, also on the two steps after
expiry where they must be ignored) with the noise injected. Each episode is
replayed through market.simulate in float64 at the w the env used, and the
test compares observations at every date and after expiry, terminal
losses, the terminal flags of the 32-step layout, shaped and unshaped
rewards, the Robbins-Monro step of w and the Log sums. Then it runs the
env's own generator with no trades, recovers the noise from the
log-moneyness and checks that it is standard normal and independent across
envs and seeds. SPEC.md is the contract.

    python test_c_env.py

PUFFERLIB_DIR is a checkout of PufferLib's 5.0 branch (commit 6ffa5b1,
default /tmp/claude-0/pl5) where ./build.sh has downloaded raylib; CC picks
the C compiler (default cc).
"""

import configparser
import math
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from market import ImpactConfig, simulate, make_obs, close_out_value  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
PUFFERLIB_DIR = os.environ.get('PUFFERLIB_DIR', '/tmp/claude-0/pl5')
N_STEPS, EPISODE = 30, 32
LOG_FIELDS = ['score', 'perf', 'ru', 'loss', 'excess', 'w', 'trades',
              'episode_return', 'episode_length', 'n']


def bs_price(s, tau, cfg):
    """Black-Scholes call price in float64 (hedging.bs_call rounds to float32)."""
    sd = cfg.sigma * math.sqrt(tau)
    d1 = (torch.log(s / cfg.strike) + 0.5 * sd**2) / sd
    return s * torch.special.ndtr(d1) - cfg.strike * torch.special.ndtr(d1 - sd)


class Cfg64(ImpactConfig):
    """ImpactConfig with the premium in float64, as the C env computes it;
    the float32 premium would shift every loss by about 1e-7."""

    @property
    def premium(self):
        return bs_price(torch.tensor(float(self.s0)), self.maturity, self).item()


def potential(cfg, k, s, delta, wealth, w):
    """CyHedging.potential before expiry: -(option - close_out - w)^+ / scale
    with the Black-Scholes value of the option."""
    option = bs_price(s, cfg.maturity * (1 - k / cfg.n_steps), cfg)
    close = close_out_value(wealth - delta * s, delta, s, cfg)
    return -(option - close - w).clamp_min(0) / cfg.scale


def env_kwargs(cfg, w_init=0.3, w_eta=0.01, shaping=1, seed=0):
    """The [env] kwargs: every ImpactConfig field but n_steps, which the
    episode layout fixes at 30, and the env's own settings."""
    assert cfg.n_steps == N_STEPS
    kwargs = {k: v for k, v in cfg.to_dict().items() if k != 'n_steps'}
    kwargs.update(w_init=w_init, w_eta=w_eta, shaping=shaping, seed=seed)
    return {k: float(v) for k, v in kwargs.items()}


def read_log_fields():
    src = open(os.path.join(HERE, 'deep_hedging.h')).read()
    body = re.search(r'struct Log \{(.*?)\};', src, re.S).group(1)
    fields = re.findall(r'^\s*float (\w+);', body, re.M)
    assert len(fields) == body.count(';'), 'struct Log must hold floats only'
    return fields


def build_harness(tmp):
    src = os.path.join(PUFFERLIB_DIR, 'src')
    raylib = os.path.join(PUFFERLIB_DIR, 'raylib-5.5_linux_amd64', 'include')
    for path in (os.path.join(src, 'pufferenv.h'), os.path.join(raylib, 'raylib.h')):
        if not os.path.exists(path):
            sys.exit(f'{path} not found: set PUFFERLIB_DIR to a PufferLib 5.0 checkout '
                     'where ./build.sh has downloaded raylib')
    exe = os.path.join(tmp, 'c_env_harness')
    subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-I' + src, '-I' + raylib,
                    os.path.join(HERE, 'c_env_harness.c'), '-o', exe, '-lm'], check=True)
    return exe


def run_env(exe, tmp, kwargs, actions, z=None):
    """Run the harness on actions (T, N, 2), injecting z (N, episodes, 30)
    unless it is None. Returns obs (T + 1, N, 8), the state columns k, w,
    phi, loss, premium, each (T + 1, N), rewards and terminals (T, N) and
    the Log of each env as a dict of (N,) arrays."""
    T, n = actions.shape[:2]
    actions.astype(np.float32).tofile(os.path.join(tmp, 'actions.bin'))
    if z is not None:
        z.astype(np.float64).tofile(os.path.join(tmp, 'z.bin'))
    subprocess.run([exe, tmp, str(n), str(T), str(int(z is not None))]
                   + [f'{k}={v!r}' for k, v in kwargs.items()], check=True)

    def read(name, dtype, shape):
        return np.fromfile(os.path.join(tmp, name), dtype).reshape(shape)
    obs = read('obs.bin', np.float32, (T + 1, n, 8))
    state = read('state.bin', np.float64, (T + 1, n, 5))
    rewards = read('rewards.bin', np.float32, (T, n))
    terminals = read('terminals.bin', np.float32, (T, n))
    log = read('log.bin', np.float32, (n, len(LOG_FIELDS)))
    return (obs, dict(zip(['k', 'w', 'phi', 'loss', 'premium'], state.transpose(2, 0, 1))),
            rewards, terminals, dict(zip(LOG_FIELDS, log.T)))


def reference(cfg, z, w, actions):
    """Replay one episode of actions (n_steps, N, 2) through market.simulate.
    Returns the observations at dates 0..n (n + 1, N, 7), the loss, the
    potential at dates 0..n (n + 1, N) and the number of trades."""
    obs = []

    def replay(o):
        obs.append(o)
        return torch.from_numpy(actions[len(obs) - 1]).double()
    w = torch.from_numpy(w)
    loss, rec = simulate(torch.from_numpy(z), replay, w, cfg, record=True)
    n = cfg.n_steps
    # Expiry observation, from the recorded positions and the loss.
    s, delta = rec['s_T'], rec['delta'][:, -1]
    q = torch.diff(rec['delta'], dim=1, prepend=torch.zeros_like(s)[:, None])
    impact = cfg.kappa * (q * cfg.decay ** torch.arange(n, 0, -1)).sum(1)
    cash = (s - cfg.strike).clamp_min(0) - loss - close_out_value(torch.zeros_like(s), delta, s, cfg)
    obs.append(make_obs(n, s, delta, cash + delta * s, w, impact, cfg))
    phi = [potential(cfg, k, rec['s'][:, k], o[:, 2], o[:, 3] * cfg.scale, w)
           for k, o in enumerate(obs[:n])]
    phi.append(-(loss - w).clamp_min(0) / cfg.scale)
    return dict(obs=torch.stack(obs).numpy(), loss=loss.numpy(), phi=torch.stack(phi).numpy(),
                trades=rec['trade'].sum(1).double().numpy())


def rel_err(a, b):
    """max |a - b| / max(1, |b|), for float32 output against float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((np.abs(a - b) / np.maximum(1, np.abs(b))).max())


class Checks:
    """Largest value of each named check over all runs, and where it was."""

    def __init__(self):
        self.rows = {}

    def add(self, name, value, tol, where=''):
        value = float(value) if np.isfinite(value) else math.inf
        if name not in self.rows or value > self.rows[name][0]:
            self.rows[name] = (value, tol, where)

    def report(self):
        for name, (value, tol, where) in self.rows.items():
            status = 'ok' if value <= tol else f'FAIL ({where})'
            tol = 'exact' if tol == 0 else f'tol {tol:.3g}'
            print(f'{name:<58} {value:9.2e}  {tol:<10} {status}')
        failed = [name for name, (value, tol, _) in self.rows.items() if value > tol]
        assert not failed, f'failed: {failed}'


def check_ini(cfg_kwargs):
    ini = configparser.ConfigParser()
    ini.read(os.path.join(HERE, 'deep_hedging.ini'))
    env = {k: float(v) for k, v in ini['env'].items()}
    assert env.keys() == cfg_kwargs.keys(), set(env) ^ set(cfg_kwargs)
    bad = {k: (env[k], cfg_kwargs[k]) for k in env if env[k] != cfg_kwargs[k]}
    assert not bad, f'deep_hedging.ini [env] differs from ImpactConfig(): {bad}'
    assert int(ini['train']['horizon']) == EPISODE
    print(f'deep_hedging.ini: [env] has the {len(env)} kwargs with the defaults of '
          f'ImpactConfig(), horizon = {EPISODE}')


def check_episodes(exe, tmp, base, w_init, shaping, checks, n=2000, seed=0):
    where = f'kappa={base.kappa} fixed_cost={base.fixed_cost} shaping={shaping}'
    kwargs = env_kwargs(base, w_init=w_init, shaping=shaping)
    cfg = Cfg64(**base.to_dict())
    rng = np.random.default_rng(seed)
    T = 2 * EPISODE
    actions = np.stack([rng.uniform(-0.7, 1.7, (T, n)),
                        rng.normal(0, 1, (T, n))], axis=-1).astype(np.float32)
    z = rng.standard_normal((n, 2, N_STEPS))
    obs, state, rewards, terminals, log = run_env(exe, tmp, kwargs, actions, z)
    w, phase = state['w'], np.arange(T + 1) % EPISODE

    def add(name, value, tol):
        checks.add(name, value, tol, where)
    add('step counter k follows the 32-step layout', (state['k'] != phase[:, None]).sum(), 0)
    add('premium against float64 Black-Scholes', np.abs(state['premium'] - cfg.premium).max(), 1e-12)
    add('premium against float32 ImpactConfig.premium',
        np.abs(state['premium'] - base.premium).max(), 1e-5)
    flags = (phase[:T] == N_STEPS - 1) | (phase[:T] == EPISODE - 1)
    add('terminal = 1 exactly on expiry and reset steps',
        (terminals != flags[:, None]).sum(), 0)
    add('observation 7 == 1', (obs[..., 7] != 1).sum(), 0)
    add('initial w == w_init * scale', np.abs(w[0] - w_init * cfg.scale).max(), 1e-12)

    episodes = {name: [] for name in LOG_FIELDS}
    for e in range(2):
        t0 = e * EPISODE
        we = w[t0]
        ref = reference(cfg, z[:, e], we, actions[t0:t0 + N_STEPS])
        add('w constant within an episode', np.abs(w[t0:t0 + EPISODE] - we).max(), 0)
        add('observations 0-6 at dates 0-30 (rel err)',
            rel_err(obs[t0:t0 + N_STEPS + 1, :, :7], ref['obs']), 1e-6)
        add('obs after the step at k=30 == expiry obs',
            (obs[t0 + N_STEPS + 1] != obs[t0 + N_STEPS]).sum(), 0)
        add('terminal loss', np.abs(state['loss'][t0 + N_STEPS] - ref['loss']).max(), 1e-9)
        hit = ref['loss'] > we
        w_next = we + kwargs['w_eta'] * cfg.scale * (hit / (1 - cfg.alpha) - 1)
        add('w of the next episode: Robbins-Monro step', np.abs(w[t0 + EPISODE] - w_next).max(), 1e-12)
        start = make_obs(0, torch.full((n,), cfg.s0), torch.zeros(n), torch.full((n,), cfg.premium),
                         torch.from_numpy(w_next), torch.zeros(n), cfg)
        add('obs after the reset step: date 0 at the new w (rel err)',
            rel_err(obs[t0 + EPISODE, :, :7], start.numpy()), 1e-6)
        add('potential at the episode start', np.abs(state['phi'][t0] - ref['phi'][0]).max(), 1e-12)

        r = rewards[t0:t0 + EPISODE]
        excess = np.maximum(ref['loss'] - we, 0)
        terminal = -excess / cfg.scale
        add('reward 0 on the two steps after expiry', (r[N_STEPS:] != 0).sum(), 0)
        if shaping:
            add('shaped reward == Phi(s_k+1) - Phi(s_k) (rel err)',
                rel_err(r[:N_STEPS], np.diff(ref['phi'], axis=0)), 1e-6)
            add('shaped rewards + Phi(s_0) == -(L - w)^+ / scale',
                np.abs(r[:N_STEPS].astype(np.float64).sum(0) + ref['phi'][0] - terminal).max(), 1e-5)
            ret = terminal - ref['phi'][0]
        else:
            add('unshaped reward at expiry == -(L - w)^+ / scale (rel err)',
                rel_err(r[N_STEPS - 1], terminal), 1e-6)
            add('unshaped reward 0 before expiry', (r[:N_STEPS - 1] != 0).sum(), 0)
            ret = terminal
        ru = we + excess / (1 - cfg.alpha)
        for name, value in [('score', -ru), ('perf', -ru / cfg.scale), ('ru', ru),
                            ('loss', ref['loss']), ('excess', excess), ('w', we),
                            ('trades', ref['trades']), ('episode_return', ret),
                            ('episode_length', EPISODE), ('n', 1)]:
            episodes[name].append(np.broadcast_to(value, (n,)))

    # The Log holds float32 sums of two episodes: compare with the size of the terms.
    errs = {name: np.nan_to_num((np.abs(log[name] - np.sum(v, 0))
                                 / np.maximum(1, np.abs(v).sum(0))).max(), nan=np.inf)
            for name, v in episodes.items()}
    worst = max(errs, key=errs.get)
    checks.add('Log sums after two episodes (rel err, 10 fields)', errs[worst], 1e-6,
               f'{where}, field {worst}')


def check_generator(exe, tmp, checks, n=256, episodes=24):
    """No trades (signal -1), so S = F and the log-moneyness increments give
    back drift + vol z. Seeds 0 and 1 give 2n streams of episodes * 30 draws."""
    cfg = ImpactConfig()
    T = episodes * EPISODE
    rng = np.random.default_rng(10)
    actions = np.stack([rng.uniform(-0.7, 1.7, (T, n)), -np.ones((T, n))], axis=-1).astype(np.float32)
    dates = (np.arange(episodes)[:, None] * EPISODE + np.arange(N_STEPS + 1)).ravel()
    drift = (cfg.mu - 0.5 * cfg.sigma**2) * cfg.dt
    vol = cfg.sigma * math.sqrt(cfg.dt)
    streams, trades = [], 0
    for seed in (0, 1):
        obs, _, _, _, log = run_env(exe, tmp, env_kwargs(cfg, seed=seed), actions)
        trades += np.abs(obs[..., [2, 6]]).sum() + log['trades'].sum()
        m = obs[dates, :, 1].astype(np.float64).reshape(episodes, N_STEPS + 1, n)
        z = (np.diff(m, axis=1) * cfg.sigma * math.sqrt(cfg.maturity) - drift) / vol
        streams.append(z.transpose(2, 0, 1).reshape(n, -1))
    z = np.concatenate(streams)
    x, size = z.ravel(), z.size
    print(f'generator: {size} draws from {len(z)} streams ({n} envs x seeds 0, 1) '
          f'of {z.shape[1]} draws, mean {x.mean():+.4f}, std {x.std():.4f}')
    checks.add('no trades: positions, impact and Log trades all 0', trades, 0)

    c = x - x.mean()
    skew = (c**3).mean() / x.std()**3
    kurt = (c**4).mean() / x.var()**2 - 3
    cdf = torch.special.ndtr(torch.from_numpy(np.sort(x))).numpy()
    i = np.arange(1, size + 1)
    ks = max((i / size - cdf).max(), (cdf - (i - 1) / size).max())
    lag1 = np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1]
    corr = np.corrcoef(z)
    neighbours = np.abs(np.diagonal(corr, 1)[np.arange(len(z) - 1) % n != n - 1]).max()
    np.fill_diagonal(corr, 0)
    # In standard errors under N(0, 1) draws, independent across streams.
    se = 1 / math.sqrt(z.shape[1])
    checks.add('noise mean / s.e.', abs(x.mean()) * math.sqrt(size), 4)
    checks.add('noise |std - 1| / s.e.', abs(x.std() - 1) * math.sqrt(2 * size), 4)
    checks.add('noise skewness / s.e.', abs(skew) / math.sqrt(6 / size), 4)
    checks.add('noise excess kurtosis / s.e.', abs(kurt) / math.sqrt(24 / size), 4)
    checks.add('noise KS distance to N(0, 1) * sqrt(n) (p = 0.001)', ks * math.sqrt(size), 1.95)
    checks.add('noise lag-1 autocorrelation / s.e.', abs(lag1) * math.sqrt(size - len(z)), 4)
    checks.add('noise max |corr| of neighbouring envs / s.e.', neighbours / se, 6)
    checks.add(f'noise max |corr| over all {len(z) * (len(z) - 1) // 2} pairs of streams / s.e.',
               np.abs(corr).max() / se, 6)


if __name__ == '__main__':
    fields = read_log_fields()
    assert fields == LOG_FIELDS, f'struct Log is {fields}, expected {LOG_FIELDS}'
    print(f'struct Log: {", ".join(fields)}')
    check_ini(env_kwargs(ImpactConfig()))
    checks = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        runs = [(ImpactConfig(), 0.3), (ImpactConfig(kappa=0.0, fixed_cost=0.0), 0.3),
                (ImpactConfig(kappa=2.0, fixed_cost=0.1, half_life=3.0), -0.2)]
        print(f'injected noise: {len(runs)} configs x shaping 1, 0, 2000 envs, 64 env steps')
        for i, (cfg, w_init) in enumerate(runs):
            for shaping in (1, 0):
                check_episodes(exe, tmp, cfg, w_init, shaping, checks, seed=2 * i + shaping)
        check_generator(exe, tmp, checks)
    checks.report()
    print('ok')
