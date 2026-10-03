"""Reference model for part 3: a short option book under Bates dynamics.

This torch implementation defines the model. The C env (hedge.h, used by
PufferLib 3.0 through binding.c and by PufferLib 5.0 through
puffer5/deep_hedging.h) implements the same dynamics; test_c.py checks that
both give the same losses on the same noise and actions. Pathwise training,
the baselines and all scoring run on this file.

Market (simulation measure = pricing measure, zero rates). The fundamental
log-price x and variance v follow Bates (1996): Heston (1993) stochastic
variance with leverage correlation rho, plus lognormal jumps at rate lam,

    dx = (-lam m - v/2) dt + sqrt(v) dW1 + log(J) dN,   m = E[J] - 1,
    dv = kappa_v (theta - v) dt + xi sqrt(v) dW2,       d<W1, W2> = rho dt,

discretized with full-truncation Euler (Lord, Koekkoek & van Dijk 2010) on
n_sub substeps between hedging dates, with at most one jump in a substep.

Instruments. The stock trades at the quoted price S = F + impact; our trades
leave a transient displacement (Obizhaeva & Wang 2013) and pay a
proportional cost and a fixed fee, as in part 2. A variance swap on the
daily log-returns of F, maturing with the book, trades at its model value
V_k = N_vs ((A_k + R_k) / T - K_var), where A_k is the realized sum of
squared daily log-returns so far, R_k the expected remainder given v_k
(continuous-time Bates, so V is close to but not exactly a martingale of
the discrete scheme), K_var = R_0 / T and N_vs the BS variance vega of the
call at inception, so one contract roughly offsets the call's variance
exposure. Swap trades pay vs_cost per contract and the fixed fee.

Book. Short one call (strike_call) and one put (strike_put), cash settled on
S_n, premium = Bates value at inception (Lewis 2001 Fourier formula). At
expiry the stock is liquidated with costs and the swap settles at V_n.
The loss is L = payoff - final cash; the objective is CVaR_alpha(L).

Crowd. Other traders' orders hit the same transient impact state as ours
(jointly aggregated transient price impact, as in Neuman & Voss, "Trading
with the crowd", Math. Finance 2023). Two kinds of order flow, both
reacting to the quoted price, so the crowd's trades depend on ours through
thresholds and counts. On a path, the loss is then a discontinuous function
of our actions; its expectation over paths stays smooth, and
crowd_gradient.py measures how much of its gradient autograd misses:
- dealers, short crowd_books copies of the same book in total, split evenly
  over `dealers` dealers; each keeps its stock hedge within a no-trade band
  around the book's BS delta (at the expected average remaining variance)
  and, when the quoted price after our trade pushes it outside, trades back
  to that delta. Band half-widths are log-spaced over
  [dealer_band_low, dealer_band_high] shares for each book.
- random orders: a Poisson count with mean arrival_base + arrival_move |r|,
  where r is the last quoted log-return in standard deviations, of
  arrival_size shares each, each following the direction of r with
  probability arrival_follow (at most ARRIVAL_MAX orders on a date).
On each hedging date: we trade, the dealers react to the new quoted price,
the random orders arrive, then the market moves and impact decays.

Policy. At each date it outputs (stock target, stock signal, swap target,
swap signal); it trades an instrument to its clamped target iff the signal
is positive.
"""

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class BatesConfig:
    s0: float = 100.0
    strike_call: float = 100.0
    strike_put: float = 95.0
    maturity: float = 30 / 365
    n_steps: int = 30
    n_sub: int = 4
    v0: float = 0.04
    kappa_v: float = 2.0
    theta: float = 0.04
    xi: float = 0.6
    rho: float = -0.7
    lam: float = 2.0
    mu_j: float = -0.06
    sig_j: float = 0.08
    cost: float = 0.002         # stock: proportional cost
    fixed_cost: float = 0.02    # fee for every trade, either instrument
    kappa: float = 0.5          # stock: price impact of one share
    half_life: float = 1.0      # stock: impact half-life in hedging dates
    vs_cost: float = 0.03       # swap: cost for each contract traded
    alpha: float = 0.95
    delta_low: float = -1.5
    delta_high: float = 1.5
    vs_low: float = -3.0
    vs_high: float = 3.0
    crowd_books: float = 3.0       # dealers' total short book, in units of ours
    dealers: int = 8
    dealer_band_low: float = 0.02  # dealer band half-widths, shares for each book
    dealer_band_high: float = 0.3
    arrival_base: float = 0.5      # random orders: mean count on a date with no move
    arrival_move: float = 1.0      # extra mean count for each standard deviation of |r|
    arrival_size: float = 0.3      # shares in each order
    arrival_follow: float = 0.7    # probability an order follows the direction of r

    @property
    def dt(self):
        return self.maturity / self.n_steps

    @property
    def dt_sub(self):
        return self.dt / self.n_sub

    @property
    def sigma0(self):
        return math.sqrt(self.v0)

    @property
    def scale(self):
        return self.sigma0 * math.sqrt(self.maturity) * self.s0

    @property
    def decay(self):
        return 0.5 ** (1 / self.half_life)

    @property
    def jump_mean(self):
        """m = E[J] - 1."""
        return math.exp(self.mu_j + 0.5 * self.sig_j**2) - 1

    @property
    def jump_var(self):
        """E[log(J)^2]."""
        return self.mu_j**2 + self.sig_j**2

    @property
    def n_vs(self):
        """BS sensitivity of the call to variance at inception."""
        sd = self.sigma0 * math.sqrt(self.maturity)
        d1 = math.log(self.s0 / self.strike_call) / sd + 0.5 * sd
        return self.s0 * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) \
            * math.sqrt(self.maturity) / (2 * self.sigma0)

    @property
    def dealer_bands(self):
        n = self.dealers
        return [self.dealer_band_low * (self.dealer_band_high / self.dealer_band_low)
                ** (i / max(n - 1, 1)) for i in range(n)]

    @property
    def k_var(self):
        return expected_variance(self.v0, self.maturity, self) / self.maturity

    @property
    def premium(self):
        c = bates_call(self.s0, self.strike_call, self.maturity, self.v0, self)
        p = bates_call(self.s0, self.strike_put, self.maturity, self.v0, self) \
            - self.s0 + self.strike_put
        return float(c + p)

    def to_dict(self):
        return asdict(self)

    def c_kwargs(self):
        """Every number the C env needs, including the derived ones."""
        return {**self.to_dict(), 'premium': self.premium, 'n_vs': self.n_vs,
                'k_var': self.k_var}


OBS_DIM = 13
ACT_DIM = 4
ARRIVAL_MAX = 8
CROWD_ROWS = 3   # rows of 4 uniforms for the crowd on each date: count + up to 8 signs


def expected_variance(v, tau, cfg):
    """E[integrated variance over the next tau | v], diffusion plus jumps."""
    k = cfg.kappa_v
    v = torch.as_tensor(v).clamp_min(0) if torch.is_tensor(v) else max(v, 0.0)
    if torch.is_tensor(v):
        tau = torch.as_tensor(tau, dtype=v.dtype, device=v.device)
        return cfg.theta * tau + (v - cfg.theta) * -torch.expm1(-k * tau) / k \
            + cfg.lam * cfg.jump_var * tau
    return cfg.theta * tau + (v - cfg.theta) * -math.expm1(-k * tau) / k \
        + cfg.lam * cfg.jump_var * tau


# Bates characteristic function and the Lewis (2001) call formula.

_GL_U, _GL_W = np.polynomial.legendre.leggauss(256)


def _bates_cf(u, tau, v, cfg):
    """E[exp(i u log(S_tau / S_0))] for complex u (numpy), Albrecher et al.
    (2007) form of the Heston part."""
    k, th, xi, rho = cfg.kappa_v, cfg.theta, cfg.xi, cfg.rho
    iu = 1j * u
    b = k - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    C = k * th / xi**2 * ((b - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    jump = cfg.lam * tau * (np.exp(iu * cfg.mu_j - 0.5 * cfg.sig_j**2 * u**2) - 1
                            - iu * cfg.jump_mean)
    return np.exp(C + D * v + jump)


def bates_call(s, strike, tau, v, cfg, u_max=200.0):
    """Bates call price, Lewis (2001): C = S - sqrt(S K) / pi *
    int_0^inf Re[exp(i u x) phi(u - i/2)] / (u^2 + 1/4) du, x = log(S/K).
    Works on numpy arrays of s, v (tau scalar)."""
    s, v = np.asarray(s, dtype=np.float64), np.asarray(v, dtype=np.float64)
    u = 0.5 * u_max * (_GL_U + 1)
    w = 0.5 * u_max * _GL_W
    x = np.log(s / strike)[..., None]
    phi = _bates_cf(u - 0.5j, tau, np.maximum(v, 0)[..., None], cfg)
    integrand = (np.exp(1j * u * x) * phi).real / (u**2 + 0.25)
    return s - np.sqrt(s * strike) / math.pi * (integrand * w).sum(-1)


def _norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def bs_book(s, var_avg, tau, cfg):
    """Short book value with BS at average variance var_avg over tau: the
    approximation the shaping potential uses (cheap, and any potential keeps
    the optimal policy)."""
    sd = torch.sqrt(var_avg.clamp_min(1e-12) * tau)
    out = torch.zeros_like(s)
    for strike, put in [(cfg.strike_call, False), (cfg.strike_put, True)]:
        d1 = torch.log(s / strike) / sd + 0.5 * sd
        call = s * _norm_cdf(d1) - strike * _norm_cdf(d1 - sd)
        out = out + (call - s + strike if put else call)
    return out


def bs_book_delta(s, var_avg, tau, cfg):
    """Delta of the book at BS with average variance var_avg over tau."""
    sd = torch.sqrt(var_avg.clamp_min(1e-12) * tau)
    d_call = _norm_cdf(torch.log(s / cfg.strike_call) / sd + 0.5 * sd)
    d_put = _norm_cdf(torch.log(s / cfg.strike_put) / sd + 0.5 * sd)
    return d_call + d_put - 1


def payoff(s, cfg):
    return (s - cfg.strike_call).clamp_min(0) + (cfg.strike_put - s).clamp_min(0)


def market_noise(n_paths, cfg, generator=None, device='cpu'):
    """(B, n_steps, n_sub + CROWD_ROWS, 4). Rows 0..n_sub-1 drive the market:
    z1, z2 standard normal, u uniform on [0, 1), zj standard normal. The
    last CROWD_ROWS rows are uniforms on [0, 1) for the crowd's random
    orders, read in order: the count, then one sign for each order. Drawn on
    the CPU so a seed gives the same paths on every device."""
    z = torch.randn(n_paths, cfg.n_steps, cfg.n_sub, 4, generator=generator)
    z[..., 2] = torch.rand(n_paths, cfg.n_steps, cfg.n_sub, generator=generator)
    crowd = torch.rand(n_paths, cfg.n_steps, CROWD_ROWS, 4, generator=generator)
    return torch.cat([z, crowd], dim=2).to(device)


def poisson_count(u, mean):
    """Poisson(mean) by inversion of the uniform u, capped at ARRIVAL_MAX."""
    p = torch.exp(-mean)
    cdf = p
    count = torch.zeros_like(mean)
    for j in range(1, ARRIVAL_MAX + 1):
        count = count + (u > cdf).to(mean.dtype)
        p = p * mean / j
        cdf = cdf + p
    return count


def crowd_flow(k, s_after, s, s_prev, v, dealer_pos, crowd_u, cfg):
    """Shares the crowd trades on date k after our trade: the dealers whose
    band the quoted price s_after leaves, then the random orders. Returns the
    flow and the dealers' new positions (B, dealers)."""
    tau = cfg.maturity * (1 - k / cfg.n_steps)
    target = bs_book_delta(s_after, expected_variance(v, tau, cfg) / tau, tau, cfg)[:, None]
    bands = torch.tensor(cfg.dealer_bands, dtype=s.dtype, device=s.device)
    out = ((dealer_pos - target).abs() > bands).to(s.dtype)
    each = cfg.crowd_books / cfg.dealers
    flow = each * (out * (target - dealer_pos)).sum(-1)
    dealer_pos = dealer_pos + out * (target - dealer_pos)
    r = observed_return(s, s_prev, cfg)
    count = poisson_count(crowd_u[:, 0], cfg.arrival_base + cfg.arrival_move * r.abs())
    up = torch.where(r != 0, torch.sign(r), torch.ones_like(r))
    signs = crowd_u[:, 1:1 + ARRIVAL_MAX]
    j = torch.arange(1, ARRIVAL_MAX + 1, device=s.device, dtype=s.dtype)
    follow = torch.where(r[:, None] != 0, signs < cfg.arrival_follow, signs < 0.5).to(s.dtype)
    orders = (j <= count[:, None]).to(s.dtype) * (2 * follow - 1)
    flow = flow + cfg.arrival_size * up * orders.sum(-1)
    return flow, dealer_pos


def observed_return(s, s_prev, cfg):
    """Last quoted log-return in standard deviations of a hedging interval."""
    return torch.log(s / s_prev) / (cfg.sigma0 * math.sqrt(cfg.dt))


def evolve(x, v, noise, cfg):
    """One hedging interval of n_sub substeps; returns x, v and the
    interval's log-return. noise: (B, n_sub, 4)."""
    dt = cfg.dt_sub
    x0 = x
    for j in range(cfg.n_sub):
        z1, z2, u, zj = noise[:, j].unbind(-1)
        vp = v.clamp_min(0)
        sq = torch.sqrt(vp * dt)
        jump = (u < cfg.lam * dt).to(x.dtype) * (cfg.mu_j + cfg.sig_j * zj)
        x = x + (-cfg.lam * cfg.jump_mean - 0.5 * vp) * dt + sq * z1 + jump
        v = v + cfg.kappa_v * (cfg.theta - vp) * dt \
            + cfg.xi * sq * (cfg.rho * z1 + math.sqrt(1 - cfg.rho**2) * z2)
    return x, v, x - x0


def swap_value(realized, v, k, cfg):
    """V_k = N_vs ((A_k + R_k) / T - K_var); R_n = 0 at expiry."""
    tau = cfg.maturity * (1 - k / cfg.n_steps)
    rest = expected_variance(v, tau, cfg) if k < cfg.n_steps else 0.0
    return cfg.n_vs * ((realized + rest) / cfg.maturity - cfg.k_var)


def make_obs(k, s, v, delta, y, wealth, w, impact, swap, realized, ret, cfg):
    """[time fraction, log(S/K_call) / (sigma0 sqrt T), sqrt(v+) / sigma0,
    stock position, swap position, wealth / scale, w / scale,
    (w + wealth) / scale, impact / kappa, swap value / scale,
    realized / (theta T), last quoted return in standard deviations, 1];
    wealth = cash + delta S + y V; impact includes the crowd's. The constant
    feature lets bias-free networks (PufferLib 5.0's) represent offsets."""
    c = cfg
    tau = torch.full_like(s, k / c.n_steps)
    m = torch.log(s / c.strike_call) / (c.sigma0 * math.sqrt(c.maturity))
    imp = impact / c.kappa if c.kappa > 0 else torch.zeros_like(impact)
    return torch.stack([tau, m, torch.sqrt(v.clamp_min(0)) / c.sigma0, delta, y,
                        wealth / c.scale, w / c.scale, (w + wealth) / c.scale, imp,
                        swap / c.scale, realized / (c.theta * c.maturity), ret,
                        torch.ones_like(s)], dim=-1)


def close_out_value(cash, delta, s, cfg):
    """Cash after selling the stock position at quoted price s."""
    fee = cfg.fixed_cost * (delta != 0).to(delta.dtype)
    return cash + delta * s - 0.5 * cfg.kappa * delta**2 - cfg.cost * delta.abs() * s - fee


def gate(signal, mode, temp):
    """Trade indicator 1{signal > 0} and the gradient it lets through
    (hard: none; ste: sigmoid gradient; sigmoid: relaxed forward). The
    hybrid estimator of train_pathwise.py samples the gate instead."""
    hard = (signal > 0).to(signal.dtype)
    if mode == 'hard':
        return hard
    soft = torch.sigmoid(signal / temp)
    if mode == 'sigmoid':
        return soft
    return hard + soft - soft.detach()


def simulate(noise, policy_fn, w, cfg, gate_mode='hard', temp=0.1, gates=None,
             record=False):
    """Run a hedging policy on market noise (B, n_steps, n_sub + CROWD_ROWS, 4).

    policy_fn maps observations (B, OBS_DIM) to actions (B, 4). gates, if
    given, is a function (actions, k) -> (stock gate, swap gate) that
    overrides gate_mode (used by the hybrid estimator). Returns the loss
    L (B,), and with record=True a dict of positions, trades and prices.
    """
    n = noise.shape[0]
    dt_type, dev = noise.dtype, noise.device
    w = torch.as_tensor(w, dtype=dt_type, device=dev).expand(n)
    x = noise.new_full((n,), math.log(cfg.s0))
    v = noise.new_full((n,), cfg.v0)
    impact = noise.new_zeros(n)
    cash = noise.new_full((n,), cfg.premium)
    delta = noise.new_zeros(n)
    y = noise.new_zeros(n)
    realized = noise.new_zeros(n)
    swap = noise.new_zeros(n)
    s_prev = torch.exp(x) + impact   # so the first observed return is exactly 0
    v_t = noise.new_full((n,), cfg.v0)
    dealer_pos = bs_book_delta(s_prev, expected_variance(v_t, cfg.maturity, cfg) / cfg.maturity,
                               cfg.maturity, cfg)[:, None].expand(n, cfg.dealers)
    rec = {'delta': [], 'y': [], 'trade_s': [], 'trade_y': [], 's': [], 'v': [], 'crowd': []}
    for k in range(cfg.n_steps):
        s = torch.exp(x) + impact
        obs = make_obs(k, s, v, delta, y, cash + delta * s + y * swap, w, impact, swap,
                       realized, observed_return(s, s_prev, cfg), cfg)
        act = policy_fn(obs)
        if gates is not None:
            gs, gy = gates(act, k)
        else:
            gs, gy = gate(act[:, 1], gate_mode, temp), gate(act[:, 3], gate_mode, temp)
        q = gs * (act[:, 0].clamp(cfg.delta_low, cfg.delta_high) - delta)
        p = gy * (act[:, 2].clamp(cfg.vs_low, cfg.vs_high) - y)
        cash = cash - q * s - 0.5 * cfg.kappa * q**2 - cfg.cost * q.abs() * s \
            - cfg.fixed_cost * gs - p * swap - cfg.vs_cost * p.abs() - cfg.fixed_cost * gy
        impact = impact + cfg.kappa * q
        delta = delta + q
        y = y + p
        crowd_u = noise[:, k, cfg.n_sub:].reshape(n, -1)
        flow, dealer_pos = crowd_flow(k, torch.exp(x) + impact, s, s_prev, v, dealer_pos,
                                      crowd_u, cfg)
        impact = impact + cfg.kappa * flow
        s_prev = s
        if record:
            rec['crowd'].append(flow.detach())
            rec['delta'].append(delta.detach())
            rec['y'].append(y.detach())
            rec['trade_s'].append((act[:, 1] > 0).detach())
            rec['trade_y'].append((act[:, 3] > 0).detach())
            rec['s'].append(s.detach())
            rec['v'].append(v.detach())
        x, v, r = evolve(x, v, noise[:, k, :cfg.n_sub], cfg)
        realized = realized + r**2
        impact = impact * cfg.decay
        swap = swap_value(realized, v, k + 1, cfg)
    s = torch.exp(x) + impact
    loss = payoff(s, cfg) - close_out_value(cash + y * swap, delta, s, cfg)
    if record:
        rec = {k: torch.stack(val, dim=1) for k, val in rec.items()}
        rec['s_T'] = s.detach()
        return loss, rec
    return loss


def potential(k, s, v, delta, y, cash, w, swap, realized, cfg):
    """Shaping potential -(Lhat - w)^+ / scale, Lhat = book value (BS at the
    expected average remaining variance; payoff at expiry) minus the cash
    after closing out. Used by the C env; here for the tests."""
    if k < cfg.n_steps:
        tau = cfg.maturity * (1 - k / cfg.n_steps)
        book = bs_book(s, expected_variance(v, tau, cfg) / tau, tau, cfg)
    else:
        book = payoff(s, cfg)
    lhat = book - close_out_value(cash + y * swap, delta, s, cfg)
    return -(lhat - w).clamp_min(0) / cfg.scale


# Risk measures (same definitions as ../hedging.py).

def var(loss, alpha):
    return torch.quantile(loss, alpha).item()


def cvar(loss, alpha):
    k = max(1, int(math.ceil((1 - alpha) * loss.numel())))
    return torch.topk(loss, k).values.mean().item()


def ru_objective(loss, w, alpha):
    return w + (loss - w).clamp_min(0).mean() / (1 - alpha)


def summarize(loss, alpha):
    return {'mean': loss.mean().item(), 'std': loss.std().item(),
            f'VaR{alpha:g}': var(loss, alpha), f'CVaR{alpha:g}': cvar(loss, alpha)}
