"""PufferLib 3.0 env: the market-making replay core (mm.h) stepped in C through PufferLib's
native binding (binding.c). Build first with `python setup.py build_ext --inplace`; the data
comes from data/build.py and data/features.py.

    env = MmEnv(num_envs=4096, split=0, fill_model=1, latency=1, seed=0)
    obs, _ = env.reset()
    obs, rew, term, trunc, info = env.step(actions)   # actions int32 [num_envs, 3]

Every keyword in PARAM_NAMES (mm.h MM_PARAM_NAMES; defaults MM_DEFAULTS) can be passed; ENV.md
documents them. The core owns the whole MDP (auto-reset, the Log); the info list carries the
Log averaged over the episodes finished since the last report, plus volume shares of the
exploit flags.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import gymnasium
import numpy as np

import pufferlib

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "build"
_SRC = (HERE / "mm.h").read_text()
PARAM_NAMES = re.findall(r'"(\w+)"', re.search(r"MM_PARAM_NAMES\[\] = \{(.*?)\};", _SRC, re.S)[1])
DEFAULTS = dict(zip(PARAM_NAMES, [float(x) for x in re.search(
    r"MM_DEFAULTS\[\] = \{(.*?)\};", _SRC, re.S)[1].split(",") if x.strip()]))
assert len(DEFAULTS) == len(PARAM_NAMES)
OBS_SIZE = int(re.search(r"#define MM_OBS (\d+)", _SRC)[1])
ACT_SIZES = [int(x) for x in re.search(r"#define MM_ACT_SIZES \{([\d, ]+)\}", _SRC)[1].split(",")]
FLAG_KEYS = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside"]


def params(**kw) -> dict:
    """Full parameter dict (floats) from defaults plus overrides; rejects unknown names."""
    bad = set(kw) - set(DEFAULTS)
    if bad:
        raise TypeError(f"unknown env parameters {sorted(bad)}; known: {PARAM_NAMES}")
    return {k: float(kw.get(k, v)) for k, v in DEFAULTS.items()}


class MmEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=4096, data_dir=None, seed=0, log_interval=128, buf=None, **kw):
        import binding  # built by setup.py
        self.single_observation_space = gymnasium.spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete(ACT_SIZES)
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.params = params(**kw)
        if self.params["split"] == 2:
            print("\n!!! MmEnv: TEST SPLIT requested (needs allow_test=1 and MM_FINAL=1) !!!\n", flush=True)
        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards, self.terminals, self.truncations,
            num_envs, seed, data_dir=str(data_dir or DATA_DIR), **self.params)
        self._binding = binding
        self.tick = 0

    def reset(self, seed=0):
        self._binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self._binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            log = self._binding.vec_log(self.c_envs)
            if log:
                vol = log.get("fill_vol", 0.0)
                for k in FLAG_KEYS:
                    log[f"share_{k}"] = log[f"vol_{k}"] / vol if vol > 0 else 0.0
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        self._binding.vec_close(self.c_envs)


def make(num_envs=4096, **kw):
    return MmEnv(num_envs=num_envs, **kw)


if __name__ == "__main__":
    env = MmEnv(num_envs=8, seed=int(os.environ.get("SEED", 0)))
    obs, _ = env.reset()
    print(obs.shape, obs[0])
