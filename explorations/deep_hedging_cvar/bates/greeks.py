"""Bates value and greeks of the short book on a grid, interpolated in torch.

The classical baselines (baselines.py) need the book's delta, gamma and
variance sensitivities at every hedging date on 100k paths. Pricing each
path by Fourier inversion would be far too slow, so we tabulate once and
interpolate.

Homogeneity. With zero rates a call is homogeneous of degree one in
(S, K): C(S, K) = K c(x), x = log(S / K), with c the call on a unit strike.
One table of c and its derivatives therefore serves both strikes, and the
book (short call K_c plus short put K_p, the put by parity) is

    B = K_c c(x_c) + K_p c(x_p) - S + K_p,
    dB/dS = Delta(x_c) + Delta(x_p) - 1,        Delta(x) = c'(x) e^-x,
    d2B/dS2 = G(x_c) / K_c + G(x_p) / K_p,      G(x) = (c'' - c') e^-2x,

and dB/dv, d2B/dSdv, d2B/dv2 follow in the same way.

Quadrature. c and its derivatives come from the Lewis (2001) formula that
market.bates_call uses, c = e^x - e^(x/2) / pi int_0^inf Re[e^(iux)
phi(u - i/2)] / (u^2 + 1/4) du, differentiated under the integral: d/dx
multiplies the integrand by iu and d/dv by D(u - i/2), where phi =
exp(C + D v + J) is the Bates characteristic function in the Albrecher et
al. (2007) form (the same formulas as market._bates_cf). In particular
Delta = 1 - e^(-x/2) / pi int Re[e^(iux) phi / (1/2 - iu)] du and
G = e^(-3x/2) / pi int Re[e^(iux) phi] du (the density). The integrals use
composite Gauss-Legendre panels on [0, u_max] and, for all grid nodes at
once, two real matrix products.

Smoothing. At one day to expiry and small v the log-return density is
nearly a point mass (plus rare jumps), so the call delta is nearly a step
and gamma nearly a Dirac mass: no grid resolves that, and the Fourier
integrand does not decay. We therefore price with the log-return
convolved with an independent N(-eps^2/2, eps^2) factor, a martingale, so
the result is still an arbitrage-free price; its characteristic function
multiplies phi(u - i/2) by exp(-eps^2 (u^2 + 1/4) / 2), which also makes
the integrand decay. With eps = 1e-3 the added variance is 1e-6, i.e. 0.9%
of the variance left one day before expiry at v = 0.04 and negligible
earlier; test_greeks.py measures the effect on delta.

Grid. Dates k = 0..n_steps-1 (tau_k = T (1 - k/n)); x = a sinh(s) on a
uniform s-grid, dense around the strike; sqrt(v) = b sinh(r) on a uniform
r-grid, dense near v = 0. Interpolation is the tensor-product cubic
B-spline interpolant in (s, r) (de Boor 1978), with natural end conditions
except the even reflection f(-sqrt(v)) = f(sqrt(v)) at v = 0; a query
reads 16 coefficients for each strike. Tables are cached in an .npz named
by a hash of the pricing parameters and the grid, in $BATES_GREEKS_CACHE
or ~/.cache/deep_hedging_bates (about 50 MB, 20 s to build on 2 threads).
"""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

FIELDS = ('c', 'delta', 'gamma', 'vega', 'vanna', 'volga')
PRICING_KEYS = ('maturity', 'n_steps', 'kappa_v', 'theta', 'xi', 'rho', 'lam', 'mu_j',
                'sig_j')
VERSION = 1


@dataclass(frozen=True)
class GridSpec:
    x_lo: float = -0.6          # log(S/K) range of the unit-strike call table
    x_hi: float = 0.5
    x_a: float = 0.01           # x = x_a sinh(s): spacing x_a ds at the strike
    n_x: int = 385
    v_max: float = 0.4
    v_b: float = 0.02           # sqrt(v) = v_b sinh(r)
    n_v: int = 89
    eps: float = 1e-3           # smoothing std of the log-return (see module doc)
    u_max: float = 0.0          # 0: 6.5 / eps
    panel: float = 20.0         # Gauss-Legendre panel width in u and nodes on each
    nodes: int = 24

    def x_grid(self):
        s = np.linspace(np.arcsinh(self.x_lo / self.x_a), np.arcsinh(self.x_hi / self.x_a),
                        self.n_x)
        return self.x_a * np.sinh(s)

    def v_grid(self):
        r = np.linspace(0, np.arcsinh(math.sqrt(self.v_max) / self.v_b), self.n_v)
        return (self.v_b * np.sinh(r))**2


def cf_parts(z, tau, cfg):
    """C, D, J with phi(z) = E[exp(i z log(S_tau / S_0))] = exp(C + D v + J);
    the formulas of market._bates_cf, split so that d/dv is a product."""
    k, th, xi, rho = cfg.kappa_v, cfg.theta, cfg.xi, cfg.rho
    iz = 1j * z
    b = k - rho * xi * iz
    d = np.sqrt(b**2 + xi**2 * (iz + z**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    C = k * th / xi**2 * ((b - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    J = cfg.lam * tau * (np.exp(iz * cfg.mu_j - 0.5 * cfg.sig_j**2 * z**2) - 1
                         - iz * cfg.jump_mean)
    return C, D, J


def u_nodes(spec):
    """Composite Gauss-Legendre nodes and weights on [0, u_max]: short panels
    near 0 (the 1/(u^2 + 1/4) factor), then panels of width spec.panel."""
    u_max = spec.u_max or 6.5 / spec.eps
    edges = [0.0, 0.5, 1.5, 3.5, 7.5]
    while edges[-1] < u_max:
        edges.append(edges[-1] + min(spec.panel, 2 * (edges[-1] - edges[-2])))
    gu, gw = np.polynomial.legendre.leggauss(spec.nodes)
    a, b = np.array(edges[:-1])[:, None], np.array(edges[1:])[:, None]
    return (0.5 * (b - a) * (gu + 1) + a).ravel(), (0.5 * (b - a) * gw).ravel()


def _multipliers(u, D):
    """Integrand factors of the six fields, before e^(iux) phi."""
    lew = 1 / (u**2 + 0.25)
    dlt = 1 / (0.5 - 1j * u)
    return [lew + 0j, dlt, np.ones_like(dlt), D * lew, D * dlt, D**2 * lew]


def _assemble(x, ints):
    """Fields from the six integrals I_0..I_5 (arrays broadcastable with x)."""
    ex = np.exp(x / 2)
    return {'c': np.exp(x) - ex * ints[0] / math.pi,
            'delta': 1 - ints[1] / (ex * math.pi),
            'gamma': ints[2] / (ex**3 * math.pi),
            'vega': -ex * ints[3] / math.pi,
            'vanna': -ints[4] / (ex * math.pi),
            'volga': -ex * ints[5] / math.pi}


def call_fields(x, tau, v, cfg, spec=GridSpec()):
    """Direct quadrature (no grid) of the unit-strike call fields at paired
    points x, v (1-D numpy arrays, tau scalar). Used to build and test."""
    x, v = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(v, float))
    x, v = np.broadcast_arrays(x, v)
    u, w = u_nodes(spec)
    C, D, J = cf_parts(u - 0.5j, tau, cfg)
    damp = np.exp(-0.5 * spec.eps**2 * (u**2 + 0.25))
    out = {f: np.empty(x.shape) for f in FIELDS}
    for i in range(0, x.size, 256):
        xs, vs = x[i:i + 256], np.maximum(v[i:i + 256], 0)
        base = np.exp(C + J + D * vs[:, None] + 1j * u * xs[:, None]) * (w * damp)
        ints = [(base * m).real.sum(-1) for m in _multipliers(u, D)]
        for f, val in _assemble(xs, ints).items():
            out[f][i:i + 256] = val
    return out


def build_tables(cfg, spec=GridSpec(), verbose=True):
    """Fields on the (date, x, v) grid: dict of (n_steps, n_x, n_v) arrays."""
    t0 = time.time()
    x, vg = spec.x_grid(), spec.v_grid()
    u, w = u_nodes(spec)
    damp = np.exp(-0.5 * spec.eps**2 * (u**2 + 0.25)) * w
    cos, sin = np.cos(np.outer(x, u)), np.sin(np.outer(x, u))
    out = {f: np.empty((cfg.n_steps, len(x), len(vg))) for f in FIELDS}
    for k in range(cfg.n_steps):
        tau = cfg.maturity * (1 - k / cfg.n_steps)
        C, D, J = cf_parts(u - 0.5j, tau, cfg)
        phi = np.exp((C + J)[:, None] + D[:, None] * vg) * damp[:, None]
        g = np.concatenate([phi * m[:, None] for m in _multipliers(u, D)], axis=1)
        # contiguous copies: numpy's matmul skips BLAS on strided views
        re, im = np.ascontiguousarray(g.real), np.ascontiguousarray(g.imag)
        ints = (cos @ re - sin @ im).reshape(len(x), len(FIELDS), len(vg))
        for f, val in _assemble(x[:, None], ints.transpose(1, 0, 2)).items():
            out[f][k] = val
    if verbose:
        print(f'greeks: built {cfg.n_steps}x{len(x)}x{len(vg)} grid with {len(u)} '
              f'u-nodes in {time.time() - t0:.1f}s', flush=True)
    return out


def cache_key(cfg, spec):
    d = {'version': VERSION, 'spec': asdict(spec),
         'cfg': {k: getattr(cfg, k) for k in PRICING_KEYS}}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def default_cache_dir():
    return os.environ.get('BATES_GREEKS_CACHE',
                          os.path.join(os.path.expanduser('~'), '.cache', 'deep_hedging_bates'))


def load_tables(cfg, spec=GridSpec(), cache_dir=None, verbose=True):
    """Tables from the cache, building and saving them on a miss."""
    cache_dir = cache_dir or default_cache_dir()
    path = os.path.join(cache_dir, f'greeks_{cache_key(cfg, spec)}.npz')
    if os.path.exists(path):
        with np.load(path) as z:
            return {f: z[f] for f in FIELDS}, path
    tab = build_tables(cfg, spec, verbose)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, **tab)
    os.replace(tmp, path)
    return tab, path


def prefilter(n, even_left=False):
    """P (n + 2, n): cubic B-spline coefficients, one ghost at each end, from
    values on n uniform nodes. Natural end conditions (zero second
    derivative), or on the left the even reflection c[-1] = c[1]."""
    a = np.diag(np.full(n, 4 / 6)) + np.diag(np.full(n - 1, 1 / 6), 1) \
        + np.diag(np.full(n - 1, 1 / 6), -1)
    ghost_l = np.zeros(n)
    ghost_l[:2] = [0, 1] if even_left else [2, -1]
    ghost_r = np.zeros(n)
    ghost_r[-2:] = [-1, 2]
    a[0] += ghost_l / 6
    a[-1] += ghost_r / 6
    g = np.vstack([ghost_l, np.eye(n), ghost_r])
    return g @ np.linalg.inv(a)


def _bspline(t):
    t2, t3 = t * t, t * t * t
    return ((1 - t)**3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6,
            t3 / 6)


class BookGreeks:
    """greeks(k, s, v) -> dict(value, delta, gamma, vega_v, vanna, volga) of
    the short book (call strike_call + put strike_put), for batched tensors
    s, v and an int or tensor date index k; on the device and dtype of s.
    vega_v = dB/dv, vanna = d2B/dSdv, volga = d2B/dv2. Interpolation is
    tensor-product cubic B-spline (de Boor 1978) in the uniform grid
    coordinates."""

    def __init__(self, cfg, spec=GridSpec(), cache_dir=None, verbose=True):
        self.cfg, self.spec = cfg, spec
        tab, self.path = load_tables(cfg, spec, cache_dir, verbose)
        val = np.stack([tab[f] for f in FIELDS], -1)          # (n, n_x, n_v, 6)
        px, pv = prefilter(spec.n_x), prefilter(spec.n_v, even_left=True)
        coef = np.einsum('ai,kijf->kajf', px, val)
        coef = np.einsum('bj,kajf->kabf', pv, coef)
        self.shape = coef.shape[:3]
        self._np = coef.reshape(-1, len(FIELDS))
        self._dev = {}
        self.s_lo = math.asinh(spec.x_lo / spec.x_a)
        self.ds = (math.asinh(spec.x_hi / spec.x_a) - self.s_lo) / (spec.n_x - 1)
        self.dr = math.asinh(math.sqrt(spec.v_max) / spec.v_b) / (spec.n_v - 1)

    def table(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._dev:
            self._dev[key] = torch.as_tensor(self._np, dtype=dtype, device=device)
        return self._dev[key]

    def call(self, k, x, v):
        """Unit-strike call fields at log-moneyness x: (N, 6) in FIELDS order."""
        sp, (_, nx, nv) = self.spec, self.shape
        tab = self.table(x.device, x.dtype)
        xc = x.clamp(sp.x_lo, sp.x_hi)
        fs = (torch.asinh(xc / sp.x_a) - self.s_lo) / self.ds
        i = fs.floor().clamp(0, sp.n_x - 2)
        wx = _bspline(fs - i)
        fr = torch.asinh(v.clamp(0, sp.v_max).sqrt() / sp.v_b) / self.dr
        j = fr.floor().clamp(0, sp.n_v - 2)
        wv = _bspline(fr - j)
        k = torch.as_tensor(k, device=x.device).long().expand_as(i)
        corner = (k * nx + i.long()) * nv + j.long()   # coefficient of node (i - 1, j - 1)
        out = 0
        for a in range(4):
            for b in range(4):
                out = out + (wx[a] * wv[b])[:, None] * tab[corner + a * nv + b]
        # beyond the x range: constant greeks, value continued by intrinsic value
        intr = lambda z: torch.expm1(z).clamp_min(0)
        out[:, 0] = out[:, 0] + intr(x) - intr(xc)
        return out

    def __call__(self, k, s, v):
        c = self.cfg
        kc, kp = c.strike_call, c.strike_put
        x = torch.cat([torch.log(s / kc), torch.log(s / kp)])
        kk = torch.cat([k, k]) if torch.is_tensor(k) and k.dim() > 0 else k
        f = self.call(kk, x, torch.cat([v, v]))
        n = s.shape[0]
        g = dict(zip(FIELDS, zip(f[:n].unbind(-1), f[n:].unbind(-1))))
        return {'value': kc * g['c'][0] + kp * g['c'][1] - s + kp,
                'delta': g['delta'][0] + g['delta'][1] - 1,
                'gamma': g['gamma'][0] / kc + g['gamma'][1] / kp,
                'vega_v': kc * g['vega'][0] + kp * g['vega'][1],
                'vanna': g['vanna'][0] + g['vanna'][1],
                'volga': kc * g['volga'][0] + kp * g['volga'][1]}


def book_direct(tau, s, v, cfg, spec=GridSpec()):
    """Book fields by direct quadrature (numpy), for tests."""
    s, v = np.asarray(s, float), np.asarray(v, float)
    kc, kp = cfg.strike_call, cfg.strike_put
    fc = call_fields(np.log(s / kc), tau, v, cfg, spec)
    fp = call_fields(np.log(s / kp), tau, v, cfg, spec)
    return {'value': kc * fc['c'] + kp * fp['c'] - s + kp,
            'delta': fc['delta'] + fp['delta'] - 1,
            'gamma': fc['gamma'] / kc + fp['gamma'] / kp,
            'vega_v': kc * fc['vega'] + kp * fp['vega'],
            'vanna': fc['vanna'] + fp['vanna'],
            'volga': kc * fc['volga'] + kp * fp['volga']}
