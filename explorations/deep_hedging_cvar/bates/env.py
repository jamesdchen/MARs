"""PufferLib 3.0 env: hedge.h stepped in C through PufferLib's native binding.

Build first with `python setup.py build_ext --inplace`. The C core owns the
whole MDP (see hedge.h): 32-step episodes, shaping, the Robbins-Monro
threshold w for each agent, and the Log, which PufferLib aggregates into
the info dict once an episode has finished.
"""

import gymnasium
import numpy as np

import pufferlib

import binding
from market import BatesConfig, OBS_DIM, ACT_DIM

EPISODE = 32


class BatesEnv(pufferlib.PufferEnv):
    def __init__(self, cfg=None, num_agents=4096, shaping=True, w_init=0.3, w_eta=0.01,
                 reward_scale=0.1, seed=0, buf=None):
        self.cfg = cfg or BatesConfig()
        c = self.cfg
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([c.delta_low, -1, c.vs_low, -1], dtype=np.float32),
            high=np.array([c.delta_high, 1, c.vs_high, 1], dtype=np.float32),
            shape=(ACT_DIM,), dtype=np.float32)
        self.num_agents = num_agents
        super().__init__(buf)
        kwargs = {**c.c_kwargs(), 'w_init': w_init, 'w_eta': w_eta, 'shaping': float(shaping),
                  'reward_scale': reward_scale}
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
                                       self.terminals, self.truncations, num_agents, seed,
                                       **{k: float(v) for k, v in kwargs.items()})
        self.tick = 0

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % EPISODE == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        binding.vec_close(self.c_envs)
