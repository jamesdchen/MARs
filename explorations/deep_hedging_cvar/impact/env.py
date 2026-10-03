"""PufferLib env around the Cython step (cy_hedging.pyx).

The MDP is the one of ../env.py (augmented state for CVaR, synchronized
episodes of n_steps + 1 env steps so each PufferLib segment holds one
episode, optional potential-based shaping) with the market of market.py.
Actions are (target position, trade signal); the agent trades when the
signal is positive.

The threshold w. Instead of sampling w over a fixed range, the env tracks
the outer Rockafellar-Uryasev minimization online. For a fixed policy the
minimizing w is VaR_alpha(L), so after every episode the center of w moves
toward the empirical alpha-quantile of the losses just observed (a
stochastic approximation of VaR, cf. Bardou, Frikha & Pages 2009). Each
agent gets w = center + jitter, so the policy also learns how to act near
the center, which evaluate.py uses for a final grid search on w.
"""

import math

import numpy as np
import gymnasium
import torch

import pufferlib

from market import ImpactConfig, OBS_DIM
from hedging import cvar
from cy_hedging import CyHedging


class TorchStep:
    """Vectorized torch twin of CyHedging, used only to benchmark it.

    Works in place on the same numpy arrays through torch.from_numpy views.
    """

    def __init__(self, cfg, obs, rewards, z, w, loss, f, impact, cash, delta, phi,
                 trades, shaping):
        self.c, self.shaping, self.k = cfg, shaping, 0
        self.obs, self.rewards = torch.from_numpy(obs), torch.from_numpy(rewards)
        self.z, self.w, self.loss = torch.from_numpy(z), torch.from_numpy(w), torch.from_numpy(loss)
        self.f, self.impact, self.cash = map(torch.from_numpy, (f, impact, cash))
        self.delta, self.phi, self.trades = map(torch.from_numpy, (delta, phi, trades))
        c = cfg
        self.drift = (c.mu - 0.5 * c.sigma**2) * c.dt
        self.vol = c.sigma * math.sqrt(c.dt)

    def potential(self):
        c, s = self.c, self.f + self.impact
        if self.k < c.n_steps:
            sd = c.sigma * math.sqrt(c.maturity * (1 - self.k / c.n_steps))
            d1 = (torch.log(s / c.strike) + 0.5 * sd**2) / sd
            cdf = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
            option = s * cdf(d1) - c.strike * cdf(d1 - sd)
        else:
            option = (s - c.strike).clamp_min(0)
        d = self.delta
        close = (self.cash + d * s - 0.5 * c.kappa * d**2 - c.cost * d.abs() * s
                 - c.fixed_cost * (d != 0).double())
        return -(option - close - self.w).clamp_min(0) / c.scale

    def write_obs(self):
        c, s = self.c, self.f + self.impact
        wealth = self.cash + self.delta * s
        self.obs[:, 0] = self.k / c.n_steps
        self.obs[:, 1] = torch.log(s / c.strike) / (c.sigma * math.sqrt(c.maturity))
        self.obs[:, 2] = self.delta
        self.obs[:, 3] = wealth / c.scale
        self.obs[:, 4] = self.w / c.scale
        self.obs[:, 5] = (self.w + wealth) / c.scale
        self.obs[:, 6] = self.impact / c.kappa if c.kappa > 0 else 0.0

    def reset_episode(self):
        self.k = 0
        self.f.fill_(self.c.s0)
        for t in (self.impact, self.delta, self.trades, self.rewards):
            t.zero_()
        self.cash.fill_(self.c.premium)
        self.phi.copy_(self.potential())
        self.write_obs()

    def step(self, actions):
        c, a = self.c, torch.from_numpy(actions).double()
        s = self.f + self.impact
        trade = a[:, 1] > 0
        q = torch.where(trade, a[:, 0].clamp(c.target_low, c.target_high) - self.delta, 0.0)
        self.cash -= torch.where(trade, q * s + 0.5 * c.kappa * q**2
                                 + c.cost * q.abs() * s + c.fixed_cost, 0.0)
        self.impact += c.kappa * q
        self.delta += q
        self.trades += trade.double()
        self.f.mul_(torch.exp(self.drift + self.vol * self.z[:, self.k]))
        self.impact.mul_(c.decay)
        self.k += 1
        last = self.k == c.n_steps
        if last:
            s = self.f + self.impact
            d = self.delta
            close = (self.cash + d * s - 0.5 * c.kappa * d**2 - c.cost * d.abs() * s
                     - c.fixed_cost * (d != 0).double())
            self.loss.copy_((s - c.strike).clamp_min(0) - close)
        if self.shaping:
            phi = self.potential()
            self.rewards.copy_(phi - self.phi)
            self.phi.copy_(phi)
        elif last:
            self.rewards.copy_(-(self.loss - self.w).clamp_min(0) / c.scale)
        else:
            self.rewards.zero_()
        self.write_obs()


class ImpactHedgingEnv(pufferlib.PufferEnv):
    def __init__(self, cfg=None, num_agents=4096, shaping=True, w_init=0.2,
                 w_jitter=0.1, w_rate=0.05, adapt_w=True, backend='cython', seed=0,
                 buf=None):
        self.cfg = cfg or ImpactConfig()
        c = self.cfg
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([c.target_low, -1], dtype=np.float32),
            high=np.array([c.target_high, 1], dtype=np.float32), dtype=np.float32)
        self.num_agents = num_agents
        super().__init__(buf)
        self.rng = np.random.default_rng(seed)
        n = num_agents
        self.z = np.zeros((n, c.n_steps))
        self.w = np.zeros(n)
        self.loss = np.zeros(n)
        self.state = {k: np.zeros(n) for k in ['f', 'impact', 'cash', 'delta', 'phi', 'trades']}
        arrays = (self.observations, self.rewards, self.z, self.w, self.loss)
        if backend == 'cython':
            self.cy = CyHedging({**c.to_dict(), 'premium': c.premium}, *arrays,
                                **self.state, shaping=shaping)
        else:
            self.cy = TorchStep(c, *arrays, **self.state, shaping=shaping)
        self.w_center = w_init * c.scale
        self.w_jitter, self.w_rate, self.adapt_w = w_jitter * c.scale, w_rate, adapt_w

    def _new_episode(self):
        self.rng.standard_normal(out=self.z)
        self.w[:] = self.w_center + self.w_jitter * self.rng.uniform(-1, 1, self.num_agents)
        self.cy.reset_episode()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._new_episode()
        self.terminals[:] = False
        return self.observations, []

    def step(self, actions):
        c = self.cfg
        self.terminals[:] = False
        if self.cy.k == c.n_steps:
            # Dummy step after expiry: start the next episode.
            self._new_episode()
            self.terminals[:] = True
            return self.observations, self.rewards, self.terminals, self.truncations, []

        self.cy.step(np.ascontiguousarray(actions, dtype=np.float32))
        if self.cy.k < c.n_steps:
            return self.observations, self.rewards, self.terminals, self.truncations, []

        loss = self.loss
        q = float(np.quantile(loss, c.alpha))
        if self.adapt_w:
            self.w_center += self.w_rate * (q - self.w_center)
        info = [{
            'loss_mean': float(loss.mean()),
            f'loss_cvar{c.alpha:g}': cvar(torch.from_numpy(loss), c.alpha),
            f'loss_var{c.alpha:g}': q,
            'w_center': self.w_center,
            'ru_excess': float(np.maximum(loss - self.w, 0).mean() / c.scale),
            'trades': float(self.state['trades'].mean()),
        }]
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        pass
