"""Classical hedgers for the short Bates book (market.py).

Every factory returns a deterministic, vectorized policy obs (B, 12) ->
actions (B, 4) = (stock target, stock signal, swap target, swap signal),
the contract of market.simulate. Policies read the date k, the quoted
price S, the variance v and the current positions (delta, y) from the
observation (inverting the scalings of market.make_obs) and get the
book's Bates greeks from greeks.BookGreeks.

We are short the book B (call strike_call plus put strike_put), so a hedge
holds +dB/dS shares and +y* variance swaps, where y* = (dB/dv) / (dV/dv)
offsets the book's sensitivity to v; the swap value V_k = N_vs ((A_k +
R_k) / T - K_var) has dV/dv = N_vs (1 - exp(-kappa_v tau)) / (kappa_v T)
(market.expected_variance). The swap also accrues the squared daily
returns, and since dB/dv is close to S^2 Gamma tau / 2 and dV/dv to
N_vs tau / T, y* also roughly matches the book's dollar gamma.

- no_hedge: never trades.
- bs_delta: Black-Scholes delta of the book at sigma0 = sqrt(v0), every date
  (Black & Scholes 1973).
- bates_delta: Bates delta dB/dS, every date. With min_var=True it uses the
  minimum-variance delta dB/dS + rho xi (dB/dv) / S, which also hedges the
  part of the variance risk that is correlated with S (Bakshi, Cao & Chen
  1997; Hull & White 2017).
- bates_delta_vega: dB/dS shares and y* swaps, both every date.
- band: no-trade bands around the delta-vega targets, the strong classical
  baseline; see band().

Targets are clamped to the position limits of market.simulate before the
band test, since a trade signal pays the fixed fee even when the clamped
trade is zero.
"""

import json
import math

import torch

from market import _norm_cdf


def read_obs(obs, cfg):
    """Date index k, quoted price S, variance v+, stock and swap positions."""
    k = (obs[:, 0] * cfg.n_steps).round().long().clamp(0, cfg.n_steps - 1)
    s = cfg.strike_call * torch.exp(obs[:, 1] * cfg.sigma0 * math.sqrt(cfg.maturity))
    v = (obs[:, 2] * cfg.sigma0)**2
    return k, s, v, obs[:, 3], obs[:, 4]


def time_left(k, cfg, like):
    return cfg.maturity * (1 - k.to(like.dtype) / cfg.n_steps)


def swap_vega(tau, cfg):
    """dV/dv of one variance swap contract."""
    return cfg.n_vs * -torch.expm1(-cfg.kappa_v * tau) / (cfg.kappa_v * cfg.maturity)


def _actions(stock, trade_s, swap, trade_y):
    sig = lambda t: torch.where(t, 1.0, -1.0).to(stock.dtype)
    return torch.stack([stock, sig(trade_s), swap, sig(trade_y)], dim=-1)


def no_hedge(cfg=None):
    def fn(obs):
        z = obs.new_zeros(obs.shape[0])
        return _actions(z, z > 1, z, z > 1)
    return fn


def bs_delta(cfg):
    """Black-Scholes delta of the book at sigma0, stock only, every date."""
    def fn(obs):
        k, s, v, _, _ = read_obs(obs, cfg)
        sd = cfg.sigma0 * torch.sqrt(time_left(k, cfg, s))
        d = sum(_norm_cdf(torch.log(s / kk) / sd + 0.5 * sd)
                for kk in (cfg.strike_call, cfg.strike_put)) - 1
        on = torch.ones_like(s, dtype=torch.bool)
        return _actions(d.clamp(cfg.delta_low, cfg.delta_high), on, torch.zeros_like(s), ~on)
    return fn


def bates_delta(cfg, greeks, min_var=False):
    """Bates delta of the book (or the minimum-variance delta), stock only,
    every date."""
    def fn(obs):
        k, s, v, _, _ = read_obs(obs, cfg)
        g = greeks(k, s, v)
        d = g['delta'] + (cfg.rho * cfg.xi * g['vega_v'] / s if min_var else 0)
        on = torch.ones_like(s, dtype=torch.bool)
        return _actions(d.clamp(cfg.delta_low, cfg.delta_high), on, torch.zeros_like(s), ~on)
    return fn


def bates_delta_vega(cfg, greeks):
    """Bates delta in the stock and y* = (dB/dv) / (dV/dv) swaps, every date."""
    def fn(obs):
        k, s, v, _, _ = read_obs(obs, cfg)
        g = greeks(k, s, v)
        y = g['vega_v'] / swap_vega(time_left(k, cfg, s), cfg)
        on = torch.ones_like(s, dtype=torch.bool)
        return _actions(g['delta'].clamp(cfg.delta_low, cfg.delta_high), on,
                        y.clamp(cfg.vs_low, cfg.vs_high), on)
    return fn


# The theory point: gamma = 1 / scale, no minimum widths, full adjustment.
BAND_DEFAULTS = dict(a_p=1.0, a_f=1.0, h_min=0.0, eta_s=1.0, a_y=1.0, y_min=0.0, eta_y=1.0,
                     m_delta=1.0, m_vega=1.0)


def band_rule(pos, target, h_in, h_out, eta):
    """Trade iff |pos - target| > h_out; then move a fraction eta of the way
    to the target, but end between the inner and the outer band edge."""
    gap = pos - target
    trade = gap.abs() > h_out
    dev = torch.minimum(torch.maximum((1 - eta) * gap.abs(), h_in), h_out)
    return torch.where(trade, target + torch.sign(gap) * dev, pos), trade


def band(cfg, greeks, params=None, min_var=False):
    """No-trade bands around the delta-vega targets, with widths from the
    small-cost asymptotics and partial adjustment for the impact cost.

    Targets: y* = m_vega (dB/dv) / (dV/dv) swaps and delta* = m_delta dB/dS
    shares. With min_var=True the stock target adds rho xi (dB/dv - y' dV/dv)
    / S, the minimum-variance correction for the variance exposure left by
    the swap position y' after this date's swap trade (zero when y' = y*);
    it made no measurable difference once m_delta and m_vega are tuned, so
    it is off by default.

    Widths. For a frictionless target phi and an instrument X, the small-cost
    expansions give no-trade half-widths that depend on the state only
    through the ratio of local quadratic variations q = d<phi>/d<X>
    (Kallsen & Muhle-Karbe 2015). With a cost lam for each unit traded,
    (3 lam q / (2 gamma))^(1/3) (Whalley & Wilmott 1997 for BS, lam = c S,
    q = Gamma^2); with a fixed fee f, (12 f q / gamma)^(1/4) (Altarovici,
    Muhle-Karbe & Soner 2015); gamma is absolute risk aversion. Both follow
    from balancing the cost rate against gamma/2 times the variance of the
    hedge error when the deviation is a reflected (proportional) or
    restarted (fixed) Brownian motion. We take gamma = 1 / scale and tune
    multipliers instead (a = (scale gamma)^(-1/3) or ^(-1/4) in theory):

      stock  q_s = d<delta*>/d<S> = m_delta^2 (Gamma^2 + 2 rho xi Gamma Va / S
                                    + xi^2 Va^2 / S^2),   Va = d2B/dSdv,
             inner h_in = a_p (1.5 c S q_s scale)^(1/3),
             outer h_out = h_in + a_f (12 f q_s scale)^(1/4) + h_min,
      swap   q_y = d<y*>/d<V> = m_vega^2 (Va^2 S^2 + 2 rho xi S Va Vo
                                    + xi^2 Vo^2) / (xi^2 (dV/dv)^4),  Vo = d2B/dv2,
             inner a_y a_p (1.5 vs_cost q_y scale)^(1/3),
             outer a_y (inner part + a_f (12 f q_y scale)^(1/4)) + y_min,

    with diffusive quadratic variations (dS = S sqrt(v) dW1,
    dv = xi sqrt(v) dW2), so v cancels. Positions are in shares and
    contracts, so h_min and y_min are too.

    Trading. Inside the outer band nothing trades. Outside, the position
    moves a fraction eta of the way to the target (Garleanu & Pedersen 2013
    partial adjustment, for the quadratic impact cost), but ends between the
    inner and the outer edge: eta = 1 returns to the inner edge (the
    proportional-cost band, as in the two-band policy with both fixed and
    proportional costs; Zakamouline 2006), eta = 0 stops at the outer edge.
    The swap rule uses its own eta_y.

    params: dict overriding BAND_DEFAULTS (a_p, a_f, h_min, eta_s, a_y,
    y_min, eta_y, m_delta, m_vega). tune_baselines.py tunes all but eta_s,
    whose optimum in a pilot was 1 with no measurable effect (the trade
    sizes are set by the band edges, and the impact cost of a 0.1-share
    trade is small next to the fixed fee).
    """
    p = {**BAND_DEFAULTS, **(params or {})}
    c = cfg
    rx = c.rho * c.xi

    def fn(obs):
        k, s, v, pos_s, pos_y = read_obs(obs, c)
        g = greeks(k, s, v)
        vv = swap_vega(time_left(k, c, s), c)
        va, vo = g['vanna'], g['volga']

        y_tgt = (p['m_vega'] * g['vega_v'] / vv).clamp(c.vs_low, c.vs_high)
        q_y = p['m_vega']**2 * ((va * s + rx * vo)**2 + (c.xi**2 - rx**2) * vo**2) \
            / (c.xi**2 * vv**4)
        in_y = p['a_p'] * (1.5 * c.vs_cost * q_y * c.scale)**(1 / 3)
        out_y = p['a_y'] * (in_y + p['a_f'] * (12 * c.fixed_cost * q_y * c.scale)**0.25) \
            + p['y_min']
        y_new, trade_y = band_rule(pos_y, y_tgt, p['a_y'] * in_y, out_y, p['eta_y'])

        d_tgt = p['m_delta'] * g['delta']
        if min_var:
            d_tgt = d_tgt + rx * (g['vega_v'] - y_new * vv) / s
        d_tgt = d_tgt.clamp(c.delta_low, c.delta_high)
        q_s = p['m_delta']**2 * ((g['gamma'] + rx * va / s)**2 + (c.xi**2 - rx**2) * (va / s)**2)
        in_s = p['a_p'] * (1.5 * c.cost * s * q_s * c.scale)**(1 / 3)
        out_s = in_s + p['a_f'] * (12 * c.fixed_cost * q_s * c.scale)**0.25 + p['h_min']
        d_new, trade_s = band_rule(pos_s, d_tgt, in_s, out_s, p['eta_s'])
        return _actions(d_new, trade_s, y_new, trade_y)
    return fn


def untuned(cfg, greeks):
    """The fixed classical hedgers, by display name."""
    return {'no hedge': no_hedge(cfg),
            'BS delta': bs_delta(cfg),
            'Bates delta': bates_delta(cfg, greeks),
            'Bates min-variance delta': bates_delta(cfg, greeks, min_var=True),
            'Bates delta-vega': bates_delta_vega(cfg, greeks),
            'band, theory point': band(cfg, greeks)}


def tuned_band(cfg, greeks, path='results/baselines.json'):
    """The band with the best parameters found by tune_baselines.py."""
    with open(path) as f:
        return band(cfg, greeks, json.load(f)['best']['params'])
