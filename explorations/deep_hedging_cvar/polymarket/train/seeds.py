"""Train the selected config for 5 fresh seeds and report validation across seeds.

    python seeds.py --selected runs/sweeps/s1/selected.json --jobs 2 --threads 2

Seeds default to 100..104, none of which the sweep used, so the reported
validation numbers are not inflated by the selection (the selected config's
sweep seeds were part of the choice). Each seed is a train_ppo.py run
(--config with the selected config, same step budget), which saves its
checkpoints, appends its validation evaluations to the ledger and its summary to
runs.jsonl (tag 'seeds:<cfg hash>'). Writes
<out>/seeds.json: per seed the validation metrics of every variant, the final
checkpoint and its sha256; across seeds the mean, min and max of the main
metrics (EVALUATION.md section 4: report mean and min/max, never the best seed).
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
KEYS = ('pnl_total', 'pnl_mean', 'pnl_median', 'cvar95', 'max_abs_inv', 'frac_positive',
        'score_mean', 'sum/spread_capture', 'sum/markout_60', 'sum/inventory_pnl', 'sum/rebates',
        'sum/liq_rewards', 'sum/pnl_trading', 'sum/fill_vol', 'mean/uptime_two_sided')


def run(seed, cfg_path, out, args, tag):
    rd = os.path.join(out, f'seed_{seed}')
    res_path = os.path.join(rd, 'result.json')
    if os.path.exists(res_path):  # resumable: finished seeds are not retrained
        return json.load(open(res_path))
    from sweep import set_aside
    set_aside(rd)  # a killed seed run: keep its partial logs, retrain
    cmd = [PY, os.path.join(HERE, 'train_ppo.py'), '--config', cfg_path, '--seed', str(seed),
           '--threads', str(args.threads), '--out', rd, '--tag', tag, '--quiet',
           '--eval-every', str(args.eval_every)]
    if args.variants:
        cmd += ['--variants', *args.variants]
    if args.ledger:
        cmd += ['--ledger', args.ledger]
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, 'stdout.txt'), 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, 'OMP_NUM_THREADS': str(args.threads)})
    if proc.returncode != 0:
        raise RuntimeError(f'seed {seed} failed: {rd}/stdout.txt')
    return json.load(open(res_path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selected', required=True, help='selected.json from select.py (or a config JSON)')
    p.add_argument('--seeds', type=int, nargs='*', default=[100, 101, 102, 103, 104])
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--eval-every', type=float, default=0)
    p.add_argument('--variants', nargs='*', default=None)
    p.add_argument('--ledger', default=None)
    p.add_argument('--timesteps', type=float, default=None,
                   help='train longer than the sweep (changes total_timesteps, hence the config hash)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    if len(a.seeds) < 5:
        print('warning: the protocol asks for >= 5 seeds', file=sys.stderr)

    doc = json.load(open(a.selected))
    cfg = doc.get('cfg') or doc
    if cfg is None:
        raise SystemExit('selected.json has no selected config')
    import ledger
    if a.timesteps:
        cfg = {**cfg, 'total_timesteps': int(a.timesteps)}
    h = ledger.config_hash(cfg)
    out = a.out or os.path.join(HERE, 'runs', 'seeds', h[:12])
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, 'config.json')
    json.dump({'cfg': cfg}, open(cfg_path, 'w'), indent=1)
    tag = f'seeds:{h[:12]}'

    results = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run, s, cfg_path, out, a, tag): s for s in a.seeds}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            results[s] = fut.result()
            hv = results[s]['val'][results[s]['headline']]
            print(json.dumps({'seed': s, 'pnl_total': hv.get('pnl_total'), 'cvar95': hv.get('cvar95')}), flush=True)

    per_seed = {s: {'val': r['val'], 'ckpt': r['ckpt'], 'ckpt_sha256': r['ckpt_sha256'],
                    'cfg_hash': r['cfg_hash'], 'steps': r['steps'], 'wall_s': r['wall_s']}
                for s, r in sorted(results.items())}
    hashes = {r['cfg_hash'] for r in results.values()}
    across = {}
    for variant in next(iter(results.values()))['val']:
        for k in KEYS:
            vals = [r['val'][variant].get(k) for r in results.values()]
            if all(v is not None for v in vals):
                across[f'{variant}/{k}'] = {'mean': float(np.mean(vals)), 'min': float(np.min(vals)),
                                            'max': float(np.max(vals)), 'std': float(np.std(vals))}
    summary = {'cfg_hash': h, 'run_cfg_hashes': sorted(hashes), 'seeds': sorted(results),
               'selected_from': os.path.abspath(a.selected), 'across_seeds': across, 'per_seed': per_seed}
    with open(os.path.join(out, 'seeds.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({'out': os.path.join(out, 'seeds.json'),
                      **{k: v['mean'] for k, v in across.items() if k.endswith(('pnl_total', 'cvar95'))}}))


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    main()
