"""Evaluate policy checkpoints (BC students, PPO runs) on val v2, one ledger row per evaluation.

    python eval_ckpt.py --name bc --ckpts ckpt/bc_m0.pt ckpt/bc_m1.pt ckpt/bc_m2.pt
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tune  # noqa: E402
from tune import VARIANTS, V2, LEDGER, KEYS, seed_summary  # noqa: E402,F401
import ledger  # noqa: E402
from evaluator import evaluate, nan_to_none  # noqa: E402
from envs import get_adapter  # noqa: E402
from policy import load_checkpoint  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--ckpts', nargs='+')
    p.add_argument('--variants', nargs='*', default=['central_1s', 'pess_1s', 'central_2s'])
    a = p.parse_args()
    torch.set_num_threads(1)
    adapter = get_adapter('real')
    res = {}
    for v in a.variants:
        rows = []
        for i, c in enumerate(a.ckpts):
            pol, ck = load_checkpoint(c)
            kw = {**(ck.get('cfg', {}).get('env', {}).get('kw', {}) or {}), **VARIANTS[v], 'data_dir': V2}
            r = evaluate(pol, adapter, split=1, env_kw=kw)
            assert r['n_markets'] == 374
            ledger.append({'kind': 'val_eval', 'split': 1, 'protocol': 'v2', 'source': 'workers/w4/eval_ckpt.py',
                           'backend': 'real', 'tag': f'w4_{a.name}', 'name': f'{a.name}_{v}_{i}', 'seed': i + 1,
                           'data_dir': V2, 'variant': v, 'ckpt': os.path.relpath(c, HERE), 'ckpt_sha256': sha(c),
                           'val': nan_to_none(r)}, LEDGER)
            rows.append({k: r.get(k) for k in KEYS})
        res[v] = {'per_seed': rows, 'summary': seed_summary(rows)}
        s = res[v]['summary']
        print(json.dumps({'variant': v, 'pnl_mean': round(s['pnl_mean'], 4), 'se': round(s['pnl_mean_se'], 4),
                          'fills': round(s['fills_per_market'], 1), 'mm_all': s['makes_markets_all'],
                          'len': [round(s[f'pnl_mean_{L}'], 2) for L in ('5m', '15m', '4h')]}), flush=True)
    json.dump(res, open(os.path.join(HERE, 'runs', f'{a.name}_eval.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
