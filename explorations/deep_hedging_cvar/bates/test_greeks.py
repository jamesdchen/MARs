"""Checks for greeks.py (CPU, about two minutes).

1. The split characteristic function equals market._bates_cf, and the
   quadrature reproduces market.bates_call where the latter is accurate; the
   error of bates_call's fixed 256-node rule at short tau and small v is
   reported (greeks.py uses its own composite rule).
2. Interpolation: grid values of the book's value, delta, gamma, dB/dv,
   d2B/dSdv and d2B/dv2 against direct quadrature with the same smoothing,
   at uniform random points of the grid domain and at points visited by
   simulated paths.
3. Smoothing: delta with eps = 1e-3 against eps = 0 (a long quadrature) at
   the path points with v >= 0.002, where the unsmoothed integral converges.
4. BS limit: with xi -> 0 and lam = 0 the variance is deterministic, so the
   book's delta, gamma and dB/dv are Black-Scholes at total variance
   expected_variance(v, tau) + eps^2.
5. Monte Carlo: the t = 0 delta and value against market.evolve paths, by
   the pathwise estimator (with S_T / S_0 as control variate) and by
   central bump-and-revalue with common random numbers; the grid value
   against cfg.premium (unsmoothed Fourier).

    python test_greeks.py
"""

import math
import tempfile
from dataclasses import replace

import numpy as np
import torch

from market import (BatesConfig, _bates_cf, bates_call, evolve, expected_variance,
                    market_noise, payoff)
from greeks import BookGreeks, GridSpec, book_direct, call_fields, cf_parts

FIELDS = ('value', 'delta', 'gamma', 'vega_v', 'vanna', 'volga')


def path_states(cfg, n=400, seed=5):
    """(k, S, v) visited by n Bates paths at every hedging date (no impact)."""
    noise = market_noise(n, cfg, torch.Generator().manual_seed(seed)).double()
    x = torch.full((n,), math.log(cfg.s0), dtype=torch.float64)
    v = torch.full((n,), cfg.v0, dtype=torch.float64)
    out = []
    for k in range(cfg.n_steps):
        out.append((k, torch.exp(x).numpy(), v.clamp_min(0).numpy()))
        x, v, _ = evolve(x, v, noise[:, k], cfg)
    return out


def test_quadrature():
    cfg = BatesConfig()
    z = np.linspace(0, 400, 41) - 0.5j
    for tau in [cfg.maturity, cfg.dt]:
        C, D, J = cf_parts(z, tau, cfg)
        assert np.abs(np.exp(C + D * 0.03 + J) - _bates_cf(z, tau, 0.03, cfg)).max() < 1e-14
    exact = GridSpec(eps=0.0, u_max=4000.0)
    x = np.linspace(-0.5, 0.35, 35)
    print('market.bates_call (u_max 200, 256 nodes) vs composite rule, '
          'max abs error in price units (K = 100):')
    for tau in [cfg.maturity, cfg.maturity / 2, 3 * cfg.dt, cfg.dt]:
        row = []
        for v in [0.01, 0.04, 0.2]:
            mine = call_fields(x, tau, v, cfg, exact)['c']
            finer = call_fields(x, tau, v, cfg, replace(exact, u_max=8000.0, nodes=32))['c']
            assert np.abs(mine - finer).max() < 1e-10
            err = np.abs(bates_call(np.exp(x), 1.0, tau, np.full_like(x, v), cfg) - mine).max()
            row.append(f'v={v}: {100 * err:.1e}')
            if tau == cfg.maturity and v >= 0.04:
                assert err < 1e-8      # where the premium is computed
        print(f'  tau = {tau * 365:4.1f} days  ' + '  '.join(row))


def test_interpolation():
    cfg = BatesConfig()
    g = BookGreeks(cfg)
    rng = np.random.default_rng(0)
    for name, points in [('uniform', None), ('paths', path_states(cfg))]:
        err = {f: 0.0 for f in FIELDS}
        for k in range(cfg.n_steps):
            if points is None:
                s = cfg.strike_call * np.exp(rng.uniform(-0.5, 0.35, 300))
                v = rng.uniform(0, 0.4, 300)
            else:
                s, v = points[k][1], points[k][2]
            ref = book_direct(cfg.maturity * (1 - k / cfg.n_steps), s, v, cfg, g.spec)
            out = g(k, torch.tensor(s), torch.tensor(v))
            for f in FIELDS:
                err[f] = max(err[f], np.abs(out[f].numpy() - ref[f]).max())
        print(f'interpolation, {name} points, max abs error: '
              + ', '.join(f'{f} {e:.1e}' for f, e in err.items()))
        assert err['delta'] < (1e-3 if points is None else 1e-5)
        assert err['gamma'] < 1e-3 and err['vega_v'] < 1e-2


def test_smoothing():
    cfg = BatesConfig()
    spec = GridSpec()
    exact = replace(spec, eps=0.0, u_max=25000.0)
    rows = []
    for k, s, v in path_states(cfg, n=300):
        if k not in (0, 15, 25, 28, 29):
            continue
        keep = v >= 0.002
        tau = cfg.maturity * (1 - k / cfg.n_steps)
        a = book_direct(tau, s[keep], v[keep], cfg, spec)['delta']
        b = book_direct(tau, s[keep], v[keep], cfg, exact)['delta']
        rows.append(f'k={k}: max {np.abs(a - b).max():.1e} mean {np.abs(a - b).mean():.1e}')
        assert np.abs(a - b).max() < 1e-2
    print('smoothing eps = 1e-3, |delta - unsmoothed delta| at path points: ' + ', '.join(rows))


def bs_book(tau, s, v, cfg, eps):
    w = expected_variance(torch.tensor(v), tau, cfg).numpy() + eps**2
    sd = np.sqrt(w)
    pdf = lambda d: np.exp(-0.5 * d**2) / math.sqrt(2 * math.pi)
    cdf = lambda d: 0.5 * (1 + np.vectorize(math.erf)(d / math.sqrt(2)))
    d1 = [np.log(s / kk) / sd + 0.5 * sd for kk in (cfg.strike_call, cfg.strike_put)]
    dw_dv = -math.expm1(-cfg.kappa_v * tau) / cfg.kappa_v
    return {'delta': cdf(d1[0]) + cdf(d1[1]) - 1,
            'gamma': (pdf(d1[0]) + pdf(d1[1])) / (s * sd),
            'vega_v': s * (pdf(d1[0]) + pdf(d1[1])) / (2 * sd) * dw_dv}


def test_bs_limit():
    cfg = BatesConfig(xi=1e-5, lam=0.0)   # the gap to BS is O(rho xi)
    with tempfile.TemporaryDirectory() as tmp:
        g = BookGreeks(cfg, cache_dir=tmp)
    rng = np.random.default_rng(1)
    err = {f: 0.0 for f in ('delta', 'gamma', 'vega_v')}
    for k in range(cfg.n_steps):
        tau = cfg.maturity * (1 - k / cfg.n_steps)
        s = cfg.strike_call * np.exp(rng.uniform(-0.3, 0.2, 200))
        v = rng.uniform(0.005, 0.2, 200)
        ref = bs_book(tau, s, v, cfg, g.spec.eps)
        out = g(k, torch.tensor(s), torch.tensor(v))
        for f in err:
            err[f] = max(err[f], np.abs(out[f].numpy() - ref[f]).max())
    print('BS limit (xi = 1e-5, lam = 0), max abs difference: '
          + ', '.join(f'{f} {e:.1e}' for f, e in err.items()))
    assert err['delta'] < 1e-4 and err['gamma'] < 1e-4 and err['vega_v'] < 1e-3


def test_mc_delta(n=400_000, chunk=100_000, h=1.0):
    cfg = BatesConfig()
    g = BookGreeks(cfg)
    gen = torch.Generator().manual_seed(7)
    est = {'pathwise': [], 'bump': [], 'value': [], 'cv': []}
    for _ in range(n // chunk):
        noise = market_noise(chunk, cfg, gen).double()
        xs = {d: torch.full((chunk,), math.log(cfg.s0 + d), dtype=torch.float64)
              for d in (-h, 0.0, h)}
        v = torch.full((chunk,), cfg.v0, dtype=torch.float64)
        for k in range(cfg.n_steps):
            # identical noise and v for the three starting prices
            vk = v
            for d in xs:
                xs[d], v, _ = evolve(xs[d], vk, noise[:, k], cfg)
        st = torch.exp(xs[0.0])
        slope = (st > cfg.strike_call).double() - (st < cfg.strike_put).double()
        est['pathwise'].append(slope * st / cfg.s0)
        est['cv'].append(st / cfg.s0 - 1)
        up, down = payoff(torch.exp(xs[h]), cfg), payoff(torch.exp(xs[-h]), cfg)
        est['bump'].append((up - down) / (2 * h))
        est['value'].append(payoff(st, cfg))
    e = {k: torch.cat(val) for k, val in est.items()}
    cv = e['cv']
    beta = ((e['pathwise'] - e['pathwise'].mean()) * cv).mean() / cv.var()
    pw = e['pathwise'] - beta * cv
    stats = {name: (x.mean().item(), x.std().item() / math.sqrt(x.numel()))
             for name, x in [('pathwise (control variate)', pw), ('bump, CRN', e['bump']),
                             ('value', e['value'])]}
    grid = g(0, torch.tensor([cfg.s0], dtype=torch.float64),
             torch.tensor([cfg.v0], dtype=torch.float64))
    d0, b0 = grid['delta'].item(), grid['value'].item()
    print(f'MC at t = 0 ({n} paths of market.evolve): grid delta {d0:.5f}, value {b0:.5f}')
    for name, (m, se) in stats.items():
        ref = b0 if name == 'value' else d0
        print(f'  {name:26s} {m:.5f} +- {se:.5f}  (grid - MC = {ref - m:+.5f})')
        assert abs(ref - m) < 4 * se + 2e-3
    # the smoothing adds eps^2 = 1e-6 to the total variance (about 3e-4 here)
    print(f'  grid value - cfg.premium (Fourier, no smoothing) = {b0 - cfg.premium:+.1e}')
    assert abs(b0 - cfg.premium) < 1e-3


if __name__ == '__main__':
    torch.set_num_threads(2)
    for t in [test_quadrature, test_interpolation, test_smoothing, test_bs_limit, test_mc_delta]:
        t()
    print('all greeks tests passed')
