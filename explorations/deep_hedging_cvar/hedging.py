"""Market model, hedging P&L and risk measures shared by every strategy.

Setup follows Buehler, Gonon, Teichmann, Wood (2019), "Deep hedging":
a trader sells one European call, receives the premium p0, and trades the
underlying at n dates with proportional transaction costs. With hedge
positions delta_0..delta_{n-1} (and delta_{-1} = delta_n = 0) the terminal
P&L is

    PnL = p0 - (S_n - K)^+ + sum_k delta_k (S_{k+1} - S_k)
          - c * sum_{k=0}^{n} |delta_k - delta_{k-1}| S_k

and the loss is L = -PnL. The risk measure is CVaR at level alpha, written
in the Rockafellar-Uryasev form

    CVaR_alpha(L) = min_w  w + E[(L - w)^+] / (1 - alpha).

Everything here is torch so the pathwise trainer can backpropagate through
it and the PufferLib env can reuse the exact same observation and P&L code.
"""

import math
from dataclasses import dataclass, asdict

import torch


@dataclass
class MarketConfig:
    s0: float = 100.0
    strike: float = 100.0
    sigma: float = 0.2
    mu: float = 0.0           # drift under the simulation measure
    maturity: float = 30 / 365
    n_steps: int = 30
    cost: float = 0.002       # proportional transaction cost
    alpha: float = 0.95       # CVaR level
    # Range of the Rockafellar-Uryasev threshold w the RL policy is trained on,
    # in units of `scale` (sigma * sqrt(T) * s0, roughly one std of S_T).
    w_low: float = 0.0
    w_high: float = 0.5

    @property
    def dt(self):
        return self.maturity / self.n_steps

    @property
    def scale(self):
        return self.sigma * math.sqrt(self.maturity) * self.s0

    def to_dict(self):
        return asdict(self)


OBS_DIM = 6


def simulate_gbm(n_paths, cfg, generator=None):
    """Geometric Brownian motion paths, shape (n_paths, n_steps + 1)."""
    z = torch.randn(n_paths, cfg.n_steps, generator=generator)
    inc = (cfg.mu - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * math.sqrt(cfg.dt) * z
    log_s = torch.cumsum(inc, dim=1)
    log_s = torch.cat([torch.zeros(n_paths, 1), log_s], dim=1)
    return cfg.s0 * torch.exp(log_s)


def _norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def bs_call(s, tau, cfg):
    """Black-Scholes call price and delta with zero rates."""
    s = torch.as_tensor(s, dtype=torch.float32)
    tau = torch.as_tensor(tau, dtype=torch.float32).clamp_min(1e-10)
    sd = cfg.sigma * torch.sqrt(tau)
    d1 = (torch.log(s / cfg.strike) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    price = s * _norm_cdf(d1) - cfg.strike * _norm_cdf(d2)
    return price, _norm_cdf(d1)


def premium(cfg):
    return float(bs_call(torch.tensor(cfg.s0), torch.tensor(cfg.maturity), cfg)[0])


def make_obs(k, s, delta_prev, wealth, w, cfg):
    """Observation at date k (all tensors of shape (B,)).

    [time fraction, log-moneyness / (sigma sqrt T), previous hedge,
     wealth / scale, w / scale, (w + wealth) / scale]

    `wealth` is the self-financing account so far (premium, trading gains,
    costs). The last entry is the loss budget left before the RU threshold
    is breached if the option expired worthless now; it is redundant but
    makes the w-conditioned value function easier to fit.
    """
    tau = torch.full_like(s, k / cfg.n_steps)
    m = torch.log(s / cfg.strike) / (cfg.sigma * math.sqrt(cfg.maturity))
    return torch.stack(
        [tau, m, delta_prev, wealth / cfg.scale, w / cfg.scale, (w + wealth) / cfg.scale],
        dim=-1,
    )


def step_wealth(wealth, s_now, s_next, delta_prev, delta, cfg):
    """Rebalance from delta_prev to delta at s_now, then hold to s_next."""
    return wealth - cfg.cost * (delta - delta_prev).abs() * s_now + delta * (s_next - s_now)


def terminal_loss(wealth, s_T, delta_last, cfg):
    """Liquidate the hedge (delta_n = 0), settle the call, return the loss."""
    wealth = wealth - cfg.cost * delta_last.abs() * s_T
    return -(wealth - (s_T - cfg.strike).clamp_min(0))


def hedge_loss(paths, policy_fn, w, cfg, return_deltas=False):
    """Run a hedging strategy over given paths and return the loss L.

    policy_fn maps an observation batch (B, OBS_DIM) to hedge ratios (B,).
    Differentiable whenever policy_fn is. With return_deltas, also returns
    the hedge ratios, shape (B, n_steps).
    """
    n = paths.shape[0]
    w = torch.as_tensor(w, dtype=torch.float32).expand(n)
    wealth = torch.full((n,), premium(cfg))
    delta = torch.zeros(n)
    deltas = []
    for k in range(cfg.n_steps):
        obs = make_obs(k, paths[:, k], delta, wealth, w, cfg)
        new_delta = policy_fn(obs)
        wealth = step_wealth(wealth, paths[:, k], paths[:, k + 1], delta, new_delta, cfg)
        delta = new_delta
        deltas.append(delta)
    loss = terminal_loss(wealth, paths[:, -1], delta, cfg)
    return (loss, torch.stack(deltas, dim=1)) if return_deltas else loss


def bs_delta_policy(cfg):
    def fn(obs):
        tau_left = cfg.maturity * (1 - obs[:, 0])
        s = cfg.strike * torch.exp(obs[:, 1] * cfg.sigma * math.sqrt(cfg.maturity))
        return bs_call(s, tau_left, cfg)[1]
    return fn


def no_hedge_policy(obs):
    return torch.zeros(obs.shape[0])


def var(loss, alpha):
    return torch.quantile(loss, alpha).item()


def cvar(loss, alpha):
    """Empirical CVaR: mean of the worst (1 - alpha) fraction of losses."""
    k = max(1, int(math.ceil((1 - alpha) * loss.numel())))
    return torch.topk(loss, k).values.mean().item()


def ru_objective(loss, w, alpha):
    """Rockafellar-Uryasev objective w + E[(L - w)^+] / (1 - alpha)."""
    return w + (loss - w).clamp_min(0).mean() / (1 - alpha)


def summarize(loss, alpha):
    return {
        'mean': loss.mean().item(),
        'std': loss.std().item(),
        f'VaR{alpha:g}': var(loss, alpha),
        f'CVaR{alpha:g}': cvar(loss, alpha),
    }
