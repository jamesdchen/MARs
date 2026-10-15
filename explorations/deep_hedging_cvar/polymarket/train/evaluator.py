"""Deterministic evaluation over every market of a split, each played once.

`evaluate(policy, adapter, split)` builds the adapter's eval env (one market per
env, start offset 0, no randomness in market choice), steps it with the
argmax action of every head until each market has finished once, and
summarizes the per-market records:

    n_markets, pnl_total (sum over markets), pnl_mean, pnl_median, pnl_std,
    cvar95 (stats.cvar of per-market P&L: mean of the worst 5%, negative = loss),
    frac_positive, max_abs_inv (max over markets), score_mean, and for every other
    numeric field of the records its sum ('sum/<k>') and mean ('mean/<k>'),
    plus action histograms of the greedy actions ('act/<head>/<category>').

Recurrent policies carry their state across steps and reset it on each env's
terminal flag, exactly as in the rollout.
"""

import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'eval'))
from stats import cvar  # noqa: E402  (polymarket/eval/stats.py, the protocol's helper)

from policy import greedy_actions  # noqa: E402
from puffer3 import action_histogram  # noqa: E402


def summarize(records, group_key=None):
    recs = list(records)
    out = {'n_markets': len(recs)}
    if not recs:
        return out
    pnl = np.array([r['pnl'] for r in recs], dtype=np.float64)
    out.update(pnl_total=float(pnl.sum()), pnl_mean=float(pnl.mean()),
               pnl_median=float(np.median(pnl)), pnl_std=float(pnl.std()),
               cvar95=cvar(pnl, 0.95), frac_positive=float((pnl > 0).mean()))
    if 'max_abs_inv' in recs[0]:
        out['max_abs_inv'] = float(max(r['max_abs_inv'] for r in recs))
    if 'score' in recs[0]:
        out['score_mean'] = float(np.mean([r['score'] for r in recs]))
    skip = {'pnl', 'market', 'env', 'score', 'max_abs_inv'}
    for k, v in recs[0].items():
        if k in skip or not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        vals = np.array([r[k] for r in recs], dtype=np.float64)
        out[f'sum/{k}'] = float(vals.sum())
        out[f'mean/{k}'] = float(vals.mean())
    if group_key and group_key in recs[0]:
        for g in sorted({r[group_key] for r in recs}):
            gp = pnl[[r[group_key] == g for r in recs]]
            out[f'pnl_total/{g}'] = float(gp.sum())
    return out


@torch.no_grad()
def evaluate(policy, adapter, split=1, env_kw=None, seed=0, device='cpu', max_steps=None,
             return_records=False):
    t0 = time.perf_counter()
    env, n_markets = adapter.make_eval(split, seed, **(env_kw or {}))
    obs, _ = env.reset(seed)
    N = env.num_agents
    recurrent = getattr(policy, 'recurrent', False)
    state = {'lstm_h': None} if recurrent else None
    done = torch.zeros(N)
    records, steps = {}, 0
    nvec = tuple(int(n) for n in env.single_action_space.nvec)
    hist = [np.zeros(n) for n in nvec]
    active = np.ones(N, bool)
    was = policy.training
    policy.eval()
    try:
        while len(records) < n_markets:
            o = torch.as_tensor(obs).to(device)
            if recurrent:
                state['done'] = done
            logits, _ = policy.forward_eval(o, state)
            act = greedy_actions(logits).cpu().numpy()
            for h, n in enumerate(nvec):
                hist[h] += np.bincount(act[active, h], minlength=n)[:n]
            obs, _, term, _, _ = env.step(act)
            done = torch.as_tensor(np.asarray(term, dtype=np.float32))
            for rec in adapter.eval_records(env):
                if rec['market'] not in records:
                    records[rec['market']] = rec
                    if 'env' in rec:
                        active[rec['env']] = False
            steps += 1
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError(f'eval did not finish {n_markets} markets in {max_steps} steps '
                                   f'({len(records)} done)')
    finally:
        policy.train(was)
        env.close()
    recs = [records[k] for k in sorted(records)]
    out = summarize(recs, group_key='contract_len_s')
    out.update(action_histogram(hist, getattr(adapter, 'action_labels', None), prefix='act/'))
    out['eval_steps'] = steps
    out['eval_seconds'] = time.perf_counter() - t0
    return (out, recs) if return_records else out


def nan_to_none(d):
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}
