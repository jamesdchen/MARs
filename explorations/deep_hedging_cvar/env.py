"""PufferLib native env: CVaR hedging as an MDP on an augmented state.

CVaR is not an expected sum of rewards, so PPO cannot optimize it directly.
We use the state augmentation of Bauerle & Ott (2011), "Markov decision
processes with average-value-at-risk criteria": for a fixed threshold w,

    CVaR_alpha(L) = min_w  w + E[(L - w)^+] / (1 - alpha),

and the inner problem  min_pi E[(L - w)^+]  is an ordinary expected
terminal cost once the accumulated wealth is part of the state. We draw w
at random for every episode and put it in the observation, so one policy
pi(a | s, w) covers all thresholds; the outer minimization over w is a
one-dimensional search done after training (see evaluate.py).

Episode layout. Every agent runs a synchronized episode of n_steps + 1
env steps: n_steps hedging decisions, then one extra step whose action is
ignored and which resets the episode. The terminal reward arrives with the
observation at k = n_steps (time fraction 1), and the policy masks its
value to 0 there. With bptt_horizon = n_steps + 1 every PufferLib segment
holds exactly one episode, so the terminal reward never falls across a
segment boundary (PufferLib's advantage kernel does not look across them).

Reward shaping (optional, on by default). With shaping=False the only
reward is -(L - w)^+ / scale at expiry. With shaping=True every step pays
Phi(s_{k+1}) - Phi(s_k) with the potential

    Phi(s_k) = -(Lhat_k - w)^+ / scale,
    Lhat_k   = -(wealth_k - C_BS(S_k, T - t_k) - c |delta_k| S_k),

the hinge applied to the Black-Scholes mark-to-market loss of closing out
now. At expiry C_BS is the payoff, so Phi(s_n) is the true terminal reward
and the rewards telescope to it minus Phi(s_0), a constant within an
episode. This is potential-based shaping (Ng, Harada, Russell 1999); with
gamma = 1 it leaves the optimal policy unchanged and only changes how
credit is spread over the steps.
"""

import numpy as np
import torch
import gymnasium

import pufferlib

from hedging import (MarketConfig, OBS_DIM, make_obs, premium, step_wealth, terminal_loss,
                     cvar, bs_call)


class HedgingEnv(pufferlib.PufferEnv):
    def __init__(self, cfg=None, num_agents=4096, delta_low=-0.5, delta_high=1.5,
                 shaping=True, seed=0, buf=None):
        self.cfg = cfg or MarketConfig()
        self.shaping = shaping
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=delta_low, high=delta_high, shape=(1,), dtype=np.float32)
        self.num_agents = num_agents
        super().__init__(buf)
        self.gen = torch.Generator().manual_seed(seed)
        self.p0 = premium(self.cfg)
        self.drift = (self.cfg.mu - 0.5 * self.cfg.sigma**2) * self.cfg.dt
        self.vol = self.cfg.sigma * np.sqrt(self.cfg.dt)

    def _new_episode(self):
        n, c = self.num_agents, self.cfg
        self.k = 0
        self.s = torch.full((n,), c.s0)
        self.delta = torch.zeros(n)
        self.wealth = torch.full((n,), self.p0)
        u = torch.rand(n, generator=self.gen)
        self.w = (c.w_low + (c.w_high - c.w_low) * u) * c.scale
        self.phi = self._potential()

    def _potential(self):
        c = self.cfg
        tau = c.maturity * (1 - self.k / c.n_steps)
        option = bs_call(self.s, torch.tensor(tau), c)[0] if self.k < c.n_steps \
            else (self.s - c.strike).clamp_min(0)
        mtm_loss = -(self.wealth - option - c.cost * self.delta.abs() * self.s)
        return -(mtm_loss - self.w).clamp_min(0) / c.scale

    def _write_obs(self):
        obs = make_obs(self.k, self.s, self.delta, self.wealth, self.w, self.cfg)
        self.observations[:] = obs.numpy()

    def reset(self, seed=None):
        if seed is not None:
            self.gen.manual_seed(seed)
        self._new_episode()
        self._write_obs()
        self.rewards[:] = 0
        self.terminals[:] = False
        return self.observations, []

    def step(self, actions):
        c = self.cfg
        info = []
        self.rewards[:] = 0
        self.terminals[:] = False

        if self.k == c.n_steps:
            # Dummy step after expiry: start the next episode.
            self._new_episode()
            self.terminals[:] = True
            self._write_obs()
            return self.observations, self.rewards, self.terminals, self.truncations, info

        new_delta = torch.from_numpy(np.asarray(actions, dtype=np.float32).reshape(-1)).clone()
        z = torch.randn(self.num_agents, generator=self.gen)
        s_next = self.s * torch.exp(self.drift + self.vol * z)
        self.wealth = step_wealth(self.wealth, self.s, s_next, self.delta, new_delta, c)
        self.s, self.delta = s_next, new_delta
        self.k += 1

        if self.shaping:
            phi = self._potential()
            self.rewards[:] = (phi - self.phi).numpy()
            self.phi = phi

        if self.k == c.n_steps:
            loss = terminal_loss(self.wealth, self.s, self.delta, c)
            excess = (loss - self.w).clamp_min(0) / c.scale
            if not self.shaping:
                self.rewards[:] = (-excess).numpy()
            info.append({
                'loss_mean': loss.mean().item(),
                'loss_std': loss.std().item(),
                f'loss_cvar{c.alpha:g}': cvar(loss, c.alpha),
                'ru_excess': excess.mean().item(),
                'reward_clipped_frac': (excess > 1).float().mean().item(),
            })

        self._write_obs()
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        pass
