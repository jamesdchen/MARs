"""PufferLib 3.0 env for the v3 replay core (mm3.h), stepped in C through binding3.c. Build first
with `python setup.py build_ext --inplace` (in this directory). Imports side by side with the v2 env
(../env.py, extension `binding`): this one loads the extension `binding3` from this directory.

    from env3 import MmEnv3
    env = MmEnv3(num_envs=4096, data_dir=".../build", obs_size=80, liq_rewards=1, size_mode=1, Q=1000)
    obs, _ = env.reset()
    obs, rew, term, trunc, info = env.step(actions)   # actions int32 [num_envs, 3]

Every keyword in PARAM_NAMES (mm3.h M3_PARAM_NAMES; defaults M3_DEFAULTS) can be passed; ENV3.md
documents them. With the defaults and a v2 data directory the env is v2's (obs 50).
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import re
import sys
from pathlib import Path

import gymnasium
import numpy as np

import pufferlib

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "build"
_SRC = (HERE / "mm3.h").read_text()
PARAM_NAMES = re.findall(r'"(\w+)"', re.search(r"M3_PARAM_NAMES\[\] = \{(.*?)\};", _SRC, re.S)[1])
DEFAULTS = dict(zip(PARAM_NAMES, [float(x) for x in re.search(
    r"M3_DEFAULTS\[\] = \{(.*?)\};", _SRC, re.S)[1].split(",") if x.strip()]))
assert len(DEFAULTS) == len(PARAM_NAMES)
OBS_V2 = int(re.search(r"#define M3_OBS_V2 (\d+)", _SRC)[1])
OBS_FULL = int(re.search(r"#define M3_OBS_FULL (\d+)", _SRC)[1])
ACT_SIZES = [int(x) for x in re.search(r"#define M3_ACT_SIZES \{([\d, ]+)\}", _SRC)[1].split(",")]
LOG_FIELDS = re.findall(r"float (\w+);", re.search(r"struct Mm3Log \{(.*?)\n\};", _SRC, re.S)[1])
FLAG_KEYS = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside", "inspread"]
EXTRA_OBS = ["ev_log", "ep_left", "cat_crypto", "cat_sports", "cat_elections", "rw_active", "rw_rate", "rw_v",
             "rw_min_rel", "rw_scorable", "rw_two_sided_only", "rw_comp", "rw_size_ok0", "rw_size_ok1",
             "rw_size_ok2", "rw_cur_share"] + [f"rw_bid_share_a{a}" for a in range(1, 7)] \
    + [f"rw_ask_share_a{a}" for a in range(1, 7)]


def params(**kw) -> dict:
    """Full parameter dict (floats) from defaults plus overrides; rejects unknown names."""
    bad = set(kw) - set(DEFAULTS)
    if bad:
        raise TypeError(f"unknown env parameters {sorted(bad)}; known: {PARAM_NAMES}")
    return {k: float(kw.get(k, v)) for k, v in DEFAULTS.items()}


def load_binding():
    """binding3 from this directory, whatever is on sys.path (it cannot collide with v2's binding)."""
    if "binding3" in sys.modules:
        return sys.modules["binding3"]
    so = sorted(HERE.glob("binding3*.so"))
    if not so:
        raise ImportError(f"binding3 is not built: cd {HERE} && python setup.py build_ext --inplace")
    loader = importlib.machinery.ExtensionFileLoader("binding3", str(so[0]))
    spec = importlib.util.spec_from_file_location("binding3", str(so[0]), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules["binding3"] = mod
    return mod


class MmEnv3(pufferlib.PufferEnv):
    def __init__(self, num_envs=4096, data_dir=None, seed=0, log_interval=128, buf=None, **kw):
        binding = load_binding()
        self.params = params(**kw)
        self.obs_size = int(self.params["obs_size"])
        self.single_observation_space = gymnasium.spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_size,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete(ACT_SIZES)
        self.num_agents = num_envs
        self.log_interval = log_interval
        if self.params["split"] == 2:
            print("\n!!! MmEnv3: TEST SPLIT requested (needs allow_test=1 and MM_FINAL=1) !!!\n", flush=True)
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
                fill_pnl = log.get("spread_capture", 0.0) + log.get("markout_60", 0.0)
                for k in FLAG_KEYS:
                    log[f"share_{k}"] = log[f"vol_{k}"] / vol if vol > 0 else 0.0
                    log[f"pnl_share_{k}"] = log[f"pnl_{k}"] / fill_pnl if fill_pnl > 0 else float("nan")
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        self._binding.vec_close(self.c_envs)


def make(num_envs=4096, **kw):
    return MmEnv3(num_envs=num_envs, **kw)


if __name__ == "__main__":
    env = MmEnv3(num_envs=8, seed=int(os.environ.get("SEED", 0)), obs_size=OBS_FULL)
    obs, _ = env.reset()
    print(obs.shape, obs[0])
