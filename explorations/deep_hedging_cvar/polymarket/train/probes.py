"""Learnability probes: tasks a correct PPO stack must learn quickly, on the
env's own observations and dynamics with the reward replaced.

    python probes.py                       # all probes, auto env (real if present)
    python probes.py --probe join --env standin --steps 2e6

- join:    reward 1 when both price heads choose 'join best', else 0
           (random policy: 1/49 of the steps).
- const:   imitate a fixed action (bid 'improve 1', ask '1 behind', size 2x):
           reward = fraction of the three heads that match.
- teacher: imitate an observation-dependent teacher (the adapter's teacher:
           the fair-value baseline on the real env, the hand-made quoter on the
           stand-in): reward = fraction of heads matching teacher(obs).
           Tests that the policy reads its observations.

The reward is computed from the action and the observation it was chosen on,
by `RewardOverride`, a thin vecenv wrapper that steps the env and then
overwrites its reward buffer, unless the env implements the probe itself
(adapter.native_probes: the real env's `reward_mode=1` is the 'join' probe,
computed in C). Terminals and resets are the env's own. Pass:
greedy success (mean probe reward of the argmax policy on fresh rollouts) >=
0.9 within the step budget; prints one JSON line per probe.
"""

import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')  # shared CPUs: no spin-waiting threads (bench.py)

import argparse
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
from policy import greedy_actions, make_policy  # noqa: E402
from puffer3 import Trainer, ppo_config  # noqa: E402

PROBES = ('join', 'const', 'teacher')


class RewardOverride:
    """Vecenv wrapper: reward = fn(actions, obs the actions were chosen on)."""

    def __init__(self, env, fn):
        self.env, self.fn = env, fn

    def __getattr__(self, k):
        return getattr(self.env, k)

    def async_reset(self, seed=None):
        self.env.async_reset(seed)

    def send(self, actions):
        obs = self.env.observations.copy()
        self.env.send(actions)
        self.env.rewards[:] = self.fn(np.asarray(actions), obs)

    def recv(self):
        return self.env.recv()

    def close(self):
        self.env.close()


def probe_fn(name, adapter):
    J = adapter.join
    if name == 'join':
        return lambda a, o: ((a[:, 0] == J) & (a[:, 1] == J)).astype(np.float32)
    if name == 'const':
        target = np.array([J - 1, J + 1, 1])
        return lambda a, o: (a == target).mean(1).astype(np.float32)
    if name == 'teacher':
        teacher = adapter.teacher()
        return lambda a, o: (a == teacher(o)).mean(1).astype(np.float32)
    raise ValueError(name)


@torch.no_grad()
def greedy_success(policy, adapter, fn, num_envs=256, steps=256, seed=123):
    env = RewardOverride(adapter.make_train(num_envs, seed=seed), fn)
    env.async_reset(seed)
    rec = policy.recurrent
    state = {'lstm_h': None} if rec else None
    tot = []
    for _ in range(steps):
        o, _, d, _, _, _, _ = env.recv()
        if rec:
            state['done'] = torch.as_tensor(d).float()
        logits, _ = policy.forward_eval(torch.as_tensor(o), state)
        a = greedy_actions(logits).numpy()
        tot.append(fn(a, o).mean())
        env.send(a)
    env.close()
    return float(np.mean(tot))


def run_probe(name, backend='auto', steps=2_000_000, num_envs=1024, threads=4, policy='mlp',
              hidden=128, layers=None, quiet=False, seed=0, **ppo):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    adapter = get_adapter(backend)
    fn = probe_fn(name, adapter)
    native = getattr(adapter, 'native_probes', {}).get(name)
    if native is not None:
        env = adapter.make_train(num_envs, seed=seed, **native)
    else:
        env = RewardOverride(adapter.make_train(num_envs, seed=seed), fn)
    kw = {'hidden': hidden}
    if layers is not None:
        kw['layers'] = layers
    pol = make_policy(policy, env=env, **kw)
    horizon = ppo.pop('bptt_horizon', 64)
    cfg = ppo_config(seed=seed, total_timesteps=int(steps), use_rnn=pol.recurrent, bptt_horizon=horizon,
                     minibatch_size=min(16384, num_envs * horizon), **{
                         'learning_rate': 1e-3, 'gamma': 0.9, 'ent_coef': 0.003, **ppo})
    tr = Trainer(cfg, env, pol)
    t0 = time.time()
    last = None
    while tr.global_step < cfg['total_timesteps']:
        tr.evaluate()
        logs = tr.train()
        last = float(tr.rewards.mean())
        if not quiet and tr.epoch % 10 == 0:
            print(json.dumps({'probe': name, 'step': tr.global_step, 'mean_reward': round(last, 4),
                              'entropy': round(logs['losses/entropy'], 3)}), flush=True)
    env.close()
    gs = greedy_success(pol, adapter, fn)
    res = {'probe': name, 'backend': adapter.name, 'policy': policy, 'native': native is not None,
           'steps': tr.global_step,
           'wall_s': round(time.time() - t0, 1), 'train_reward_last_rollout': last,
           'greedy_success': gs, 'pass': gs >= 0.9}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--probe', nargs='*', default=list(PROBES), choices=PROBES)
    p.add_argument('--env', default='auto', choices=['auto', 'standin', 'real'])
    p.add_argument('--steps', type=float, default=2e6)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--policy', default='mlp', choices=['mlp', 'puffernet'])
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=None)
    p.add_argument('--out', default=os.path.join(HERE, 'runs', 'probes.jsonl'))
    a = p.parse_args()
    for name in a.probe:
        res = run_probe(name, a.env, int(a.steps), a.num_envs, a.threads, a.policy, a.hidden,
                        a.layers, quiet=True)
        print(json.dumps(res), flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps({**res, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}) + '\n')


if __name__ == '__main__':
    main()
