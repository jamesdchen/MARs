"""Throughput (agent steps/s) of the CPU training stack over a grid of settings.

    python bench.py --env real --threads 1 2 4 --num-envs 4096 8192 16384 --horizon 64 128
    python bench.py --env real --compile 0 1 --minibatch 16384 65536

For every combination: build the train env and policy, run puffer3.Trainer
for --warmup + --epochs epochs (rollout + PPO update each) and report the
median over the timed epochs of total, rollout and update steps/s. Also
reports raw env steps/s (random actions, no policy) per num_envs. Appends one
JSON line per setting to runs/bench.jsonl, with the machine's load average,
since the 4 cores are shared and numbers move with the load.
"""

import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')  # shared CPUs: no spin-waiting threads (bench.py)

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pufferlib.pufferl  # noqa: E402,F401
warnings.filterwarnings('default', category=RuntimeWarning)

from envs import get_adapter  # noqa: E402
from policy import make_policy  # noqa: E402
from puffer3 import Trainer, ppo_config  # noqa: E402


def env_sps(adapter, n, steps=200):
    env = adapter.make_train(n, seed=0)
    env.reset(0)
    acts = np.random.randint(0, adapter.nvec, (8, n, 3)).astype(np.int32)
    t = time.perf_counter()
    for i in range(steps):
        env.step(acts[i % 8])
    dt = time.perf_counter() - t
    env.close()
    return n * steps / dt


def bench_one(adapter, s, warmup, epochs):
    torch.set_num_threads(s['threads'])
    torch.manual_seed(0)
    env = adapter.make_train(s['num_envs'], seed=0)
    pk = {'hidden': s['hidden']}
    if s['policy'] == 'mlp':
        pk.update(layers=s['layers'], shared=s['shared'])
    else:
        pk.update(layers=s['layers'])
    pol = make_policy(s['policy'], env=env, **pk)
    batch = s['num_envs'] * s['horizon']
    cfg = ppo_config(total_timesteps=batch * (warmup + epochs + 1) * 10, bptt_horizon=s['horizon'],
                     minibatch_size=min(s['minibatch'], batch), update_epochs=s['update_epochs'],
                     use_rnn=pol.recurrent, compile=bool(s['compile']), compile_eval=bool(s['compile']),
                     fast_sample=bool(s['fast_sample']))
    tr = Trainer(cfg, env, pol)
    rows = []
    for e in range(warmup + epochs):
        tr.evaluate()
        logs = tr.train()
        if e >= warmup:
            rows.append(logs)
    env.close()
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    return {'sps': med('sps'), 'sps_rollout': med('sps_rollout'), 'sps_update': med('sps_update'),
            'minibatches_per_epoch': tr.total_minibatches}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='auto', choices=['auto', 'standin', 'real'])
    p.add_argument('--threads', type=int, nargs='*', default=[4])
    p.add_argument('--num-envs', type=int, nargs='*', default=[4096])
    p.add_argument('--horizon', type=int, nargs='*', default=[64])
    p.add_argument('--minibatch', type=int, nargs='*', default=[16384])
    p.add_argument('--update-epochs', type=int, nargs='*', default=[2])
    p.add_argument('--policy', nargs='*', default=['mlp'])
    p.add_argument('--hidden', type=int, nargs='*', default=[128])
    p.add_argument('--layers', type=int, nargs='*', default=[2])
    p.add_argument('--shared', type=int, nargs='*', default=[0])
    p.add_argument('--compile', type=int, nargs='*', default=[0])
    p.add_argument('--fast-sample', type=int, nargs='*', default=[1])
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--no-env-sps', action='store_true')
    p.add_argument('--out', default=os.path.join(HERE, 'runs', 'bench.jsonl'))
    a = p.parse_args()
    adapter = get_adapter(a.env)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if not a.no_env_sps:
        for n in a.num_envs:
            row = {'kind': 'env_only', 'env': adapter.name, 'num_envs': n,
                   'env_sps': env_sps(adapter, n), 'load': os.getloadavg()[0]}
            print(json.dumps(row), flush=True)
            open(a.out, 'a').write(json.dumps(row) + '\n')
    keys = ['threads', 'num_envs', 'horizon', 'minibatch', 'update_epochs', 'policy', 'hidden', 'layers',
            'shared', 'compile', 'fast_sample']
    grid = [a.threads, a.num_envs, a.horizon, a.minibatch, a.update_epochs, a.policy, a.hidden, a.layers,
            a.shared, a.compile, a.fast_sample]
    for combo in itertools.product(*grid):
        s = dict(zip(keys, combo))
        if s['policy'] == 'puffernet' and s['shared']:
            continue
        load0 = os.getloadavg()[0]
        try:
            r = bench_one(adapter, s, a.warmup, a.epochs)
        except Exception as e:  # noqa: BLE001
            r = {'error': f'{type(e).__name__}: {e}'}
        row = {'kind': 'trainer', 'env': adapter.name, **s, **r, 'load': load0,
               'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
        print(json.dumps({k: (round(v) if isinstance(v, float) and v > 100 else v) for k, v in row.items()}),
              flush=True)
        open(a.out, 'a').write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
