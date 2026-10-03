"""Pick the PufferLib 5.0 sweep run to report, using validation paths.

Protein ranks the runs of `./puffer sweep` by the env's score, minus the
Rockafellar-Uryasev objective at each env's own w with sampled actions.
That is a training-side proxy, so we take the top_k runs by it, re-score
each on the validation paths of evaluate.py (deterministic actions, grid
search on w, CVaR of the exact model), and copy the best to
<out>/puffer5_sweep_best.bin with a JSON file of its run facts, which
evaluate.py reads like any other 5.0 run. The test paths are not used.

    python select_sweep.py --pufferlib /content/PufferLib --sweep-log sweep.log \
        --sweep-wall 5400 --out ../results/gpu
"""

import argparse
import configparser
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from market import BatesConfig, market_noise, simulate, cvar  # noqa: E402
from evaluate import w_search  # noqa: E402
from puffer5.puffernet import PufferNetPolicy  # noqa: E402


def sweep_runs(pufferlib, sweep_log):
    """(run number, Protein score, cost in seconds, steps) of finished runs."""
    pat = re.compile(r'sweep run=(\d+) score=(\S+) cost=(\S+) steps=(\S+)')
    text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', open(sweep_log, errors='replace').read())
    runs = []
    for m in pat.finditer(text):
        run, score, cost, steps = int(m[1]), float(m[2]), float(m[3]), float(m[4])
        inis = sorted(glob.glob(f'{pufferlib}/logs/deep_hedging/sweep_*_{run:04d}.ini'),
                      key=os.path.getmtime)
        if not inis or not np.isfinite(score):
            continue
        run_id = os.path.splitext(os.path.basename(inis[-1]))[0]
        bins = sorted(glob.glob(f'{pufferlib}/checkpoints/deep_hedging/{run_id}/*.bin'),
                      key=os.path.getmtime)
        if not bins:
            continue
        ini = configparser.ConfigParser(strict=False)
        with open(inis[-1]) as f:
            ini.read_string(''.join(line for line in f if not line.startswith('#')))
        runs.append(dict(run=run, run_id=run_id, score=score, cost=cost, steps=steps,
                         checkpoint=bins[-1],
                         hidden=int(float(ini['policy']['hidden_size'])),
                         layers=int(float(ini['policy']['num_layers']))))
    return runs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pufferlib', default='/content/PufferLib')
    p.add_argument('--sweep-log', required=True)
    p.add_argument('--sweep-wall', type=float, required=True, help='sweep wall time, seconds')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--n-val', type=int, default=100_000)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'results', 'gpu'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    runs = sweep_runs(args.pufferlib, args.sweep_log)
    assert runs, 'no finished sweep runs with a checkpoint'
    cfg = BatesConfig()
    val = market_noise(args.n_val, cfg, torch.Generator().manual_seed(999), args.device)
    w_grid = np.linspace(0, 0.8, 33) * cfg.scale
    top = sorted(runs, key=lambda r: -r['score'])[:args.top_k]
    for r in top:
        pol = PufferNetPolicy(r['checkpoint'], hidden=r['hidden'], layers=r['layers'],
                              device=args.device)
        r['w'], _ = w_search(val, pol, w_grid, cfg)
        r['val_cvar'] = cvar(simulate(val, pol, r['w'], cfg).cpu(), cfg.alpha)
        print(json.dumps({k: r[k] for k in ['run', 'score', 'val_cvar', 'w', 'hidden', 'layers',
                                            'steps', 'cost']}), flush=True)
    best = min(top, key=lambda r: r['val_cvar'])
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(best['checkpoint'], f'{args.out}/puffer5_sweep_best.bin')
    meta = {'wall_time': best['cost'], 'steps': best['steps'], 'hidden': best['hidden'],
            'layers': best['layers'], 'device': torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else 'cpu', 'tune_wall_time': args.sweep_wall,
            'trials': len(runs), 'run_id': best['run_id'], 'protein_score': best['score'],
            'val_cvar': best['val_cvar'], 'top_k': top}
    with open(f'{args.out}/puffer5_sweep_best.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"selected run {best['run']} ({best['run_id']}): validation CVaR {best['val_cvar']:.3f}")


if __name__ == '__main__':
    main()
