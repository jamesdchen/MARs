"""Behaviour-cloning warm start: imitate a teacher quoter on TRAIN markets.

    python bc.py --teacher fair_value --out runs/bc/fv_mlp.pt
    python train_ppo.py --init runs/bc/fv_mlp.pt --kl-coef 0.1 --kl-decay 0.5 ...

1. Collect: run the teacher in the training env (split 0, random markets and
   start offsets, so the states are those of training) for --steps steps on
   --num-envs envs, recording observation, teacher action, reward and
   terminal flag. Teachers: the adapter's (`fair_value` = baselines/baselines.py
   on the real env, through baselines/obs_spec.py; a simple obs-based quoter if
   the spec does not resolve; the hand-made quoter on the stand-in).
   --teacher-params takes a JSON file of baseline parameters (e.g. tuned).
2. Fit: cross-entropy of the three heads on the teacher's actions, and
   optionally (--value) the value head on discounted Monte Carlo returns of
   the (clamped) rewards under the teacher, cut at terminals, with the last
   3 / (1 - gamma) steps of each env (whose returns are truncated) left out.
   Recurrent policies train on horizon-long segments with the terminal resets.
   The last 10% of the envs are held out for accuracy.
3. Save a checkpoint in train_ppo.py's format (policy, policy_kwargs, obs_size,
   nvec) plus a 'bc' block with the teacher and fit metrics. Nothing here
   touches validation or test data.
"""

import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')  # shared CPUs: no spin-waiting threads (bench.py)

import argparse
import json
import math
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
from evaluator import evaluate  # noqa: E402
from policy import make_policy  # noqa: E402


def collect(adapter, teacher, num_envs, steps, seed=0, env_kw=None):
    env = adapter.make_train(num_envs, seed=seed, **(env_kw or {}))
    obs, _ = env.reset(seed)
    D = obs.shape[1]
    O = np.empty((steps, num_envs, D), np.float32)
    A = np.empty((steps, num_envs, 3), np.int64)
    R = np.empty((steps, num_envs), np.float32)
    Tn = np.empty((steps, num_envs), np.float32)
    for t in range(steps):
        O[t] = env.observations
        a = np.asarray(teacher(O[t]), dtype=np.int32)
        A[t] = a
        env.step(a)
        R[t] = np.clip(env.rewards, -1, 1)
        Tn[t] = env.terminals
    env.close()
    done = np.zeros_like(Tn)
    done[1:] = Tn[:-1]  # terminal flag that arrives with obs t (reset the state before t)
    return dict(obs=O, act=A, rew=R, term_next=Tn, done=done)


def mc_returns(rew, term_next, gamma):
    G = np.zeros_like(rew)
    acc = np.zeros(rew.shape[1], np.float32)
    for t in range(rew.shape[0] - 1, -1, -1):
        acc = rew[t] + gamma * (1.0 - term_next[t]) * acc
        G[t] = acc
    return G


def to_segments(x, H):
    """(T, N, ...) -> (N * (T // H), H, ...) env-major segments."""
    T, N = x.shape[:2]
    T = (T // H) * H
    x = np.swapaxes(x[:T], 0, 1)  # (N, T, ...)
    return x.reshape(N * (T // H), H, *x.shape[2:])


def fit(policy, data, epochs=8, lr=1e-3, horizon=64, mb_segments=32, value=False, gamma=0.995,
        vf_coef=0.5, holdout=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    T, N = data['rew'].shape
    n_val = max(1, int(N * holdout))
    tr_cols, va_cols = slice(0, N - n_val), slice(N - n_val, N)
    G = mc_returns(data['rew'], data['term_next'], gamma)
    cut = min(T // 2, int(math.ceil(3.0 / (1.0 - gamma))))
    vmask = np.zeros((T, N), np.float32)
    vmask[:T - cut] = 1.0

    def segs(cols):
        return {k: torch.as_tensor(to_segments(v[:, cols], horizon)) for k, v in
                (('obs', data['obs']), ('act', data['act']), ('done', data['done']),
                 ('G', G), ('vmask', vmask))}

    tr, va = segs(tr_cols), segs(va_cols)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    S = tr['obs'].shape[0]
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(S, generator=g)
        tot = {'ce': 0.0, 'v': 0.0, 'n': 0}
        for i in range(0, S, mb_segments):
            idx = perm[i:i + mb_segments]
            loss, ce, vl = _loss(policy, {k: v[idx] for k, v in tr.items()}, value, vf_coef)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot['ce'] += ce
            tot['v'] += vl
            tot['n'] += 1
        m = metrics(policy, va)
        if not value:
            m['value_ev'] = float('nan')  # value head not trained
        row = {'epoch': ep + 1, 'train_ce': tot['ce'] / tot['n'], 'train_vloss': tot['v'] / tot['n'], **m}
        hist.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    return hist


def _forward(policy, b):
    st = {'lstm_h': None, 'terminals': b['done']}
    if getattr(policy, 'recurrent', False):
        return policy(b['obs'], st)
    return policy(b['obs'].reshape(-1, b['obs'].shape[-1]), st)


def _loss(policy, b, value, vf_coef):
    logits, v = _forward(policy, b)
    act = b['act'].reshape(-1, 3)
    ce = sum(torch.nn.functional.cross_entropy(l, act[:, h]) for h, l in enumerate(logits))
    vl = torch.zeros(())
    if value:
        m = b['vmask'].reshape(-1)
        vl = (0.5 * (v.reshape(-1) - b['G'].reshape(-1)) ** 2 * m).sum() / m.sum().clamp(min=1)
    return ce + vf_coef * vl, float(ce), float(vl)


@torch.no_grad()
def metrics(policy, b, chunk=512):
    S = b['obs'].shape[0]
    correct = np.zeros(3)
    joint = 0.0
    n = 0
    ys, vs, ms = [], [], []
    for i in range(0, S, chunk):
        sb = {k: v[i:i + chunk] for k, v in b.items()}
        logits, v = _forward(policy, sb)
        act = sb['act'].reshape(-1, 3)
        pred = torch.stack([l.argmax(-1) for l in logits], 1)
        eq = pred == act
        correct += eq.float().sum(0).numpy()
        joint += float(eq.all(1).float().sum())
        n += act.shape[0]
        ys.append(sb['G'].reshape(-1))
        vs.append(v.reshape(-1))
        ms.append(sb['vmask'].reshape(-1) > 0)
    y, vv, m = torch.cat(ys)[torch.cat(ms)], torch.cat(vs)[torch.cat(ms)], None
    ev = float(1 - (y - vv).var() / y.var()) if y.numel() > 1 and y.var() > 0 else float('nan')
    out = {f'acc_head{h}': float(correct[h] / n) for h in range(3)}
    out.update(acc_joint=joint / n, value_ev=ev)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='auto', choices=['auto', 'standin', 'real'])
    p.add_argument('--teacher', default=None, help="teacher name (real: a baselines.py name, default "
                   "fair_value; stand-in: 'standin')")
    p.add_argument('--teacher-params', default=None, help='JSON file of teacher parameters')
    p.add_argument('--env-kw', nargs='*', help='env kwargs key=value for the collection env')
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--steps', type=int, default=1024, help='steps per env to collect')
    p.add_argument('--policy', default='mlp', choices=['mlp', 'puffernet'])
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=None)
    p.add_argument('--act', default='tanh')
    p.add_argument('--shared', dest='shared', action='store_true', default=True,
                   help='MLP: one trunk (the train_ppo.py default)')
    p.add_argument('--separate', dest='shared', action='store_false')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--minibatch', type=int, default=2048, help='samples per gradient step')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--horizon', type=int, default=64)
    p.add_argument('--value', action='store_true', help='also fit the value head on MC returns')
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--eval-train', action='store_true',
                   help='score the student greedily on every train market (not val)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', default=os.path.join(HERE, 'runs', 'bc', 'bc.pt'))
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    adapter = get_adapter(a.env)
    params = json.load(open(a.teacher_params)) if a.teacher_params else None
    teacher_name = a.teacher or ('standin' if adapter.name == 'standin' else 'fair_value')
    env_kw = {}
    for it in a.env_kw or []:
        k, v = it.split('=', 1)
        env_kw[k] = json.loads(v)
    teacher = adapter.teacher(teacher_name, params, env_kw)
    params = getattr(adapter, 'teacher_params', params)

    t0 = time.time()
    data = collect(adapter, teacher, a.num_envs, a.steps, a.seed, env_kw)
    t_col = time.time() - t0
    acts = data['act'].reshape(-1, 3)
    teacher_hist = {f'head{h}': np.bincount(acts[:, h], minlength=n).tolist()
                    for h, n in enumerate(adapter.nvec)}
    print(json.dumps({'collected': int(acts.shape[0]), 'seconds': round(t_col, 1),
                      'teacher_actions': teacher_hist}), flush=True)

    pk = {'kind': a.policy, 'hidden': a.hidden}
    if a.policy == 'mlp':
        pk.update(layers=a.layers or 2, act=a.act, shared=a.shared)
    else:
        pk.update(layers=a.layers or 1)
    kw = {k: v for k, v in pk.items() if k != 'kind'}
    policy = make_policy(a.policy, obs_size=adapter.obs_size, nvec=adapter.nvec, **kw)
    hist = fit(policy, data, epochs=a.epochs, lr=a.lr, horizon=a.horizon, value=a.value,
               gamma=a.gamma, seed=a.seed, mb_segments=max(1, a.minibatch // a.horizon))
    tr_eval = None
    if a.eval_train:
        tr_eval = evaluate(policy, adapter, split=0, env_kw=env_kw, seed=a.seed)
        print(json.dumps({'student_train_markets': {k: tr_eval[k] for k in ('pnl_total', 'pnl_mean', 'cvar95')}}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'policy_kwargs': pk, 'obs_size': adapter.obs_size,
                'nvec': list(adapter.nvec), 'cfg': {'env': {'backend': adapter.name, 'kw': env_kw}},
                'bc': {'teacher': teacher_name, 'teacher_params': params, 'samples': int(acts.shape[0]),
                       'collect_seconds': t_col, 'teacher_action_counts': teacher_hist, 'history': hist,
                       'value': a.value, 'gamma': a.gamma, 'train_eval': tr_eval, 'args': vars(a)}},
               a.out)
    print(json.dumps({'saved': a.out, **hist[-1]}), flush=True)


if __name__ == '__main__':
    main()
