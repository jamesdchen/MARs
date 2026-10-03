"""Throughput of the Cython env vs its vectorized torch twin.

1. env only: steps of all agents with fixed random actions;
2. end to end: PuffeRL collection + PPO update, the setting of train_ppo.py.

    python bench.py   (writes results/bench.json)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import pufferlib.pufferl as pufferl

from market import ImpactConfig
from env import ImpactHedgingEnv
from policy import ImpactHedgePolicy
from train_ppo import ppo_config


def env_only(backend, n, reps=310):
    cfg = ImpactConfig()
    env = ImpactHedgingEnv(cfg, num_agents=n, backend=backend)
    env.reset()
    rng = np.random.default_rng(0)
    acts = [np.stack([rng.uniform(0, 1, n), rng.normal(0, 1, n)], -1).astype(np.float32)
            for _ in range(cfg.n_steps)]
    t0 = time.perf_counter()
    for i in range(reps):
        env.step(acts[i % cfg.n_steps])
    return reps * n / (time.perf_counter() - t0)


def end_to_end(backend, n, epochs=8):
    cfg = ImpactConfig()
    args = argparse.Namespace(seed=0, lr=3e-4, timesteps=10**12, minibatches=8,
                              update_epochs=4, gae_lambda=0.95, ckpt_dir='/tmp/bench_ckpt')
    env = ImpactHedgingEnv(cfg, num_agents=n, backend=backend)
    pufferl.PuffeRL.print_dashboard = lambda self, *a, **k: None
    tr = pufferl.PuffeRL(ppo_config(args, cfg.n_steps + 1, n), env, ImpactHedgePolicy())
    for _ in range(2):
        tr.evaluate()
        tr.train()
    t0, s0 = time.perf_counter(), tr.global_step
    for _ in range(epochs):
        tr.evaluate()
        tr.train()
    total = time.perf_counter() - t0
    tr.utilization.stop()
    return (tr.global_step - s0) / total


def main():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    out = {'threads': torch.get_num_threads(), 'env_only': {}, 'end_to_end': {}}
    for backend in ['cython', 'torch']:
        for n in [4096, 65536]:
            sps = env_only(backend, n)
            out['env_only'][f'{backend}_{n}'] = sps
            print(f'env only  {backend:6s} agents={n:6d}: {sps / 1e6:7.1f}M steps/s', flush=True)
    for backend in ['cython', 'torch']:
        sps = end_to_end(backend, 4096)
        out['end_to_end'][f'{backend}_4096'] = sps
        print(f'end to end {backend:6s} agents=  4096: {sps / 1e3:7.1f}k steps/s', flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/bench.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
