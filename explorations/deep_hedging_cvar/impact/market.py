"""Reference model: CVaR hedging with transient price impact and fixed costs.

This torch implementation defines the model. The Cython env used for PPO
(cy_hedging.pyx) implements the same dynamics, and test_env.py checks that
both give the same losses on the same noise and actions. Pathwise training,
the baselines and all evaluation run on this file.

Market. The fundamental price F follows geometric Brownian motion. Our own
trades leave a transient displacement I on the quoted price S = F + I, as in
Obizhaeva & Wang (2013): buying q shares at date k costs
q S_k + kappa q^2 / 2 (walking up a book of constant depth 1 / kappa), moves
the quoted price by kappa q, and the displacement decays geometrically with
a given half-life. On top of that every trade pays the proportional cost
c |q| S_k and a fixed fee f.

Hedger. A short European call (cash settled on the quoted price S_n), the
premium p0 in cash, and a stock position delta. At each date the policy
outputs a target position and a trade signal; it trades to the target when
the signal is positive and does nothing otherwise. The position is
liquidated at expiry, after settlement, with the same costs.
The loss is L = payoff - final cash, and the objective is CVaR_alpha(L).
"""

import math
import os
import sys
from dataclasses import dataclass, asdict

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hedging import bs_call, cvar, var, ru_objective, summarize  # noqa: E402,F401


@dataclass
class ImpactConfig:
    s0: float = 100.0
    strike: float = 100.0
    sigma: float = 0.2
    mu: float = 0.0
    maturity: float = 30 / 365
    n_steps: int = 30
    cost: float = 0.002         # proportional cost
    fixed_cost: float = 0.02    # fee on every trade
    kappa: float = 0.5          # price impact of one share, in price units
    half_life: float = 1.0      # impact half-life, in hedging dates
    alpha: float = 0.95
    target_low: float = -0.5
    target_high: float = 1.5

    @property
    def dt(self):
        return self.maturity / self.n_steps

    @property
    def scale(self):
        return self.sigma * math.sqrt(self.maturity) * self.s0

    @property
    def decay(self):
        return 0.5 ** (1 / self.half_life)

    @property
    def premium(self):
        return float(bs_call(torch.tensor(self.s0), torch.tensor(self.maturity), self)[0])

    def to_dict(self):
        return asdict(self)


OBS_DIM = 7
ACT_DIM = 2


def fundamental_noise(n_paths, cfg, generator=None, device='cpu'):
    """Standard normals for the fundamental price, drawn on the CPU so a seed
    gives the same paths on every device."""
    return torch.randn(n_paths, cfg.n_steps, generator=generator).to(device)


def make_obs(k, s, delta, wealth, w, impact, cfg):
    """[time fraction, log-moneyness / (sigma sqrt T), position,
    wealth / scale, w / scale, (w + wealth) / scale, impact / kappa].

    wealth = cash + delta * S_k is the mark-to-market value before costs of
    closing out; impact / kappa is the decayed sum of our past trades.
    """
    tau = torch.full_like(s, k / cfg.n_steps)
    m = torch.log(s / cfg.strike) / (cfg.sigma * math.sqrt(cfg.maturity))
    imp = impact / cfg.kappa if cfg.kappa > 0 else torch.zeros_like(impact)
    return torch.stack([tau, m, delta, wealth / cfg.scale, w / cfg.scale,
                        (w + wealth) / cfg.scale, imp], dim=-1)


def close_out_value(cash, delta, s, cfg):
    """Cash after selling the whole position at quoted price s."""
    fee = cfg.fixed_cost * (delta != 0).to(delta.dtype)
    return cash + delta * s - 0.5 * cfg.kappa * delta**2 - cfg.cost * delta.abs() * s - fee


def gate(signal, mode, temp):
    """Trade indicator 1{signal > 0} and the gradient it lets through.

    hard:    exact indicator, zero gradient almost everywhere.
    ste:     exact indicator forward, sigmoid gradient backward
             (straight-through estimator, Bengio et al. 2013).
    sigmoid: sigmoid(signal / temp) forward and backward, a relaxation
             that trades a fraction of the way to the target.
    """
    hard = (signal > 0).to(signal.dtype)
    if mode == 'hard':
        return hard
    soft = torch.sigmoid(signal / temp)
    if mode == 'sigmoid':
        return soft
    return hard + soft - soft.detach()


def simulate(z, policy_fn, w, cfg, gate_mode='hard', temp=0.1, record=False):
    """Run a hedging policy on fundamental noise z (B, n_steps).

    Runs on z's device and dtype.

    policy_fn maps observations (B, OBS_DIM) to actions (B, 2) =
    (target position, trade signal). Returns the loss L (B,) and, with
    record=True, a dict of positions, trade flags and quoted prices.
    """
    n = z.shape[0]
    w = torch.as_tensor(w, dtype=z.dtype, device=z.device).expand(n)
    drift = (cfg.mu - 0.5 * cfg.sigma**2) * cfg.dt
    vol = cfg.sigma * math.sqrt(cfg.dt)
    f = z.new_full((n,), cfg.s0)
    impact = z.new_zeros(n)
    cash = z.new_full((n,), cfg.premium)
    delta = z.new_zeros(n)
    rec = {'delta': [], 'trade': [], 's': []}
    for k in range(cfg.n_steps):
        s = f + impact
        obs = make_obs(k, s, delta, cash + delta * s, w, impact, cfg)
        act = policy_fn(obs)
        target = act[:, 0].clamp(cfg.target_low, cfg.target_high)
        g = gate(act[:, 1], gate_mode, temp)
        q = g * (target - delta)
        cash = cash - q * s - 0.5 * cfg.kappa * q**2 - cfg.cost * q.abs() * s - cfg.fixed_cost * g
        impact = impact + cfg.kappa * q
        delta = delta + q
        if record:
            rec['delta'].append(delta.detach())
            rec['trade'].append((act[:, 1] > 0).detach())
            rec['s'].append(s.detach())
        f = f * torch.exp(drift + vol * z[:, k])
        impact = impact * cfg.decay
    s = f + impact
    payoff = (s - cfg.strike).clamp_min(0)
    loss = payoff - close_out_value(cash, delta, s, cfg)
    if record:
        rec = {k: torch.stack(v, dim=1) for k, v in rec.items()}
        rec['s_T'] = s.detach()
        return loss, rec
    return loss


def bs_delta_band(cfg, width=0.0, to_edge=0.0):
    """No-transaction band around the Black-Scholes delta (on quoted price).

    Trade only when the position is more than `width` away from delta, then
    move to delta + to_edge * width on the side of the current position
    (to_edge = 0: back to delta; 1: to the band edge). width = 0 is daily
    delta hedging. In the spirit of Whalley & Wilmott (1997) and, for fixed
    costs, Zakamouline (2006); here the band is tuned by grid search.
    """
    def fn(obs):
        tau_left = cfg.maturity * (1 - obs[:, 0])
        s = cfg.strike * torch.exp(obs[:, 1] * cfg.sigma * math.sqrt(cfg.maturity))
        d = bs_call(s, tau_left, cfg)[1]
        pos = obs[:, 2]
        gap = pos - d
        trade = gap.abs() > width
        target = d + to_edge * width * torch.sign(gap)
        return torch.stack([target, torch.where(trade, 1.0, -1.0)], dim=-1)
    return fn


def no_hedge(obs):
    return torch.stack([obs.new_zeros(obs.shape[0]), -obs.new_ones(obs.shape[0])], dim=-1)
