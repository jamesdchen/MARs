"""w2: per-contract-length specialists. Job queue of train_ppo.py runs (4 parallel, 1 thread).

    python sweep_w2.py sweep            # 4 masks x 10 configs x seeds 1-3, 25M steps
    python sweep_w2.py extra NAME... --seeds 4 5   # more seeds for given mask/config names
    python sweep_w2.py check4h          # 4h steps-sufficiency check (50M) on given configs

Every run starts from train/selected.json (reward_mode 2), trains on split 0 restricted by
len_mask (bit 1 5m, 2 15m, 4 4h), and is evaluated once (final, central fills 1 s) on all
374 v2 val markets; analyze_w2.py derives the length-restricted numbers from val_markets.jsonl.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

W2 = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.dirname(os.path.dirname(W2))
PM = os.path.dirname(TRAIN)
V2 = os.path.join(PM, 'data', 'build_v2')
LEDGER = os.path.join(W2, 'trials.jsonl')
PY = os.path.join(os.path.dirname(PM), '.venv', 'bin', 'python')

MASKS = {'4h': 4, '15m': 2, '5m': 1, '4h15m': 6}
BASE = {'lr': 6.776573310250399e-4, 'ent_coef': 3.9747456923718835e-4,
        'inv_penalty': 5.203196400420489e-05, 'markout_h': 120, 'hidden': 64}
CONFIGS = {
    'c00': {},                                   # selected.json as is
    'c01': {'markout_h': 60},
    'c02': {'markout_h': 300},
    'c03': {'hidden': 128},
    'c04': {'lr': 3e-4},
    'c05': {'lr': 1.5e-3},
    'c06': {'ent_coef': 3e-3},
    'c07': {'inv_penalty': 1e-3},
    'c08': {'inv_penalty': 1e-2},
    'c09': {'markout_h': 300, 'hidden': 128, 'inv_penalty': 1e-3},
}


def cmd(mask_name, cname, seed, steps=25e6, suffix=''):
    p = {**BASE, **CONFIGS[cname]}
    name = f'{mask_name}_{cname}{suffix}'
    out = os.path.join(W2, 'runs', f'{name}_s{seed}')
    return name, out, [
        PY, os.path.join(TRAIN, 'train_ppo.py'), '--config', os.path.join(TRAIN, 'selected.json'),
        '--env', 'real', '--reward-mode', '2', '--markout-h', str(p['markout_h']),
        '--lr', repr(p['lr']), '--ent-coef', repr(p['ent_coef']), '--inv-penalty', repr(p['inv_penalty']),
        '--hidden', str(p['hidden']), '--timesteps', str(int(steps)),
        '--env-kw', f'data_dir={V2}', f'len_mask={MASKS[mask_name]}',
        '--seed', str(seed), '--threads', '1', '--no-train-eval', '--variants', 'central_1s',
        '--out', out, '--tag', f'w2_{name}', '--ledger', LEDGER, '--quiet']


def run(job):
    name, out, c = job
    if os.path.exists(os.path.join(out, 'result.json')):
        return name, out, 0
    os.makedirs(out, exist_ok=True)
    env = {**os.environ, 'OMP_WAIT_POLICY': 'passive', 'OMP_NUM_THREADS': '1'}
    t = time.time()
    with open(os.path.join(out, 'stdout.log'), 'w') as f:
        rc = subprocess.call(c, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=TRAIN)
    print(f'{time.strftime("%H:%M:%S")} {os.path.basename(out)} rc={rc} {time.time() - t:.0f}s', flush=True)
    return name, out, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['sweep', 'extra', 'check4h'])
    ap.add_argument('names', nargs='*', help='mask_cfg names, e.g. 4h_c02')
    ap.add_argument('--seeds', type=int, nargs='*', default=[1, 2, 3])
    ap.add_argument('--steps', type=float, default=25e6)
    ap.add_argument('--masks', nargs='*', default=list(MASKS))
    ap.add_argument('-j', type=int, default=4)
    a = ap.parse_args()
    jobs = []
    if a.what == 'sweep':
        # seed-major so every config gets a first seed early
        for seed, m, c in itertools.product(a.seeds, a.masks, CONFIGS):
            jobs.append(cmd(m, c, seed))
    else:
        suffix = '' if a.what == 'extra' else f'_{int(a.steps / 1e6)}M'
        for n, seed in itertools.product(a.names, a.seeds):
            m, c = n.rsplit('_', 1)
            jobs.append(cmd(m, c, seed, a.steps, suffix))
    print(f'{len(jobs)} jobs', flush=True)
    with ThreadPoolExecutor(a.j) as ex:
        res = list(ex.map(run, jobs))
    bad = [r for r in res if r[2]]
    print('failed:', bad, flush=True)


if __name__ == '__main__':
    main()
