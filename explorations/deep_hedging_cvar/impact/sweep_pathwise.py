"""Hyperparameter sweep for the pathwise baseline (train_pathwise.train).

The PPO baseline is tuned by a sweep, so the pathwise baseline gets the same
treatment: a fixed number of trials proposed by Optuna's TPE sampler over
the learning rate, the straight-through temperature, the batch size, the
hidden width and the number of iterations. Each trial trains with its own
seed (--seed + trial number) and is scored like evaluate.py scores every
method: CVaR_alpha of the exact model (hard gate, deterministic actions) on
the validation paths (seed 999), with the trial's learned w. The test paths
(seed 12345) are never touched here.

Writes to --out-dir: trial_XXX.pt for every trial (same keys as
train_pathwise.py), trials.json, and best.pt, the best trial's checkpoint
plus tune_wall_time (training wall time summed over all trials), tune_steps,
trials (count), trial (its number) and val_cvar. Both files are rewritten
after every trial, so an interrupted sweep still leaves a usable best.pt.

--max-iters and --max-batch cap the sampled values; they exist only for
quick smoke tests.
"""

import argparse
import json
import math
import os
import time

import optuna
import torch

from market import ImpactConfig, fundamental_noise, simulate, cvar
from policy import ImpactHedgePolicy
from train_pathwise import train

VAL_SEED, TEST_SEED = 999, 12345


def val_cvar(ckpt, val, cfg):
    """Validation CVaR of a checkpoint, loaded the way evaluate.py loads it."""
    pol = ImpactHedgePolicy(hidden=ckpt['hidden'])
    pol.actor.load_state_dict(ckpt['actor'])
    pol = pol.to(val.device).eval()
    with torch.no_grad():
        return cvar(simulate(val, pol.mean_action, ckpt['w'], cfg).cpu(), cfg.alpha)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=32)
    p.add_argument('--gate', choices=['hard', 'ste', 'sigmoid'], default='ste')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='results/gpu/pathwise_sweep')
    p.add_argument('--seed', type=int, default=0, help='sampler seed; trial i trains with seed + i')
    p.add_argument('--n-val', type=int, default=100_000)
    p.add_argument('--max-iters', type=int, default=None, help='cap on iters (smoke tests)')
    p.add_argument('--max-batch', type=int, default=None, help='cap on batch (smoke tests)')
    args = p.parse_args()
    if {VAL_SEED, TEST_SEED} & set(range(args.seed, args.seed + args.trials)):
        raise ValueError('a training seed would reuse the validation or test noise')

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = ImpactConfig()
    val = fundamental_noise(args.n_val, cfg, torch.Generator().manual_seed(VAL_SEED), args.device)
    records, best = [], None
    t0 = time.time()

    def objective(trial):
        nonlocal best
        hp = dict(lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                  temp=trial.suggest_float('temp', 0.01, 1.0, log=True),
                  batch=trial.suggest_categorical('batch', [4096, 8192, 16384, 32768, 65536]),
                  hidden=trial.suggest_categorical('hidden', [32, 64, 128, 256]),
                  iters=trial.suggest_categorical('iters', [1000, 2000, 4000]))
        if args.max_iters:
            hp['iters'] = min(hp['iters'], args.max_iters)
        if args.max_batch:
            hp['batch'] = min(hp['batch'], args.max_batch)
        seed = args.seed + trial.number
        ckpt = train(args.gate, seed=seed, device=args.device, log=None, **hp)
        path = f'{args.out_dir}/trial_{trial.number:03d}.pt'
        torch.save(ckpt, path)
        value = val_cvar(ckpt, val, cfg)
        finite = math.isfinite(value)
        records.append({'trial': trial.number, 'seed': seed, 'params': trial.params,
                        'train_args': ckpt['args'], 'val_cvar': value if finite else None,
                        'w': ckpt['w'], 'wall_time': ckpt['wall_time'], 'steps': ckpt['steps'],
                        'path': path})
        if finite and (best is None or value < best['val_cvar']):
            best = {'trial': trial.number, 'val_cvar': value, 'path': path}

        tune_wall_time = sum(r['wall_time'] for r in records)
        tune_steps = sum(r['steps'] for r in records)
        if best is not None:
            best_ckpt = torch.load(best['path'], weights_only=False, map_location='cpu')
            best_ckpt.update(tune_wall_time=tune_wall_time, tune_steps=tune_steps,
                             trials=len(records), trial=best['trial'],
                             val_cvar=best['val_cvar'])
            torch.save(best_ckpt, f'{args.out_dir}/best.pt')
        with open(f'{args.out_dir}/trials.json', 'w') as f:
            json.dump({'gate': args.gate, 'sampler': 'TPE', 'sampler_seed': args.seed,
                       'n_val': args.n_val, 'val_seed': VAL_SEED, 'alpha': cfg.alpha,
                       'device': args.device, 'tune_wall_time': tune_wall_time,
                       'tune_steps': tune_steps, 'best': best, 'trials': records}, f, indent=2)

        line = {'trial': trial.number, **hp, 'seed': seed,
                'val_cvar': value, 'w': ckpt['w'], 'wall_time': ckpt['wall_time'],
                'steps': ckpt['steps'], 'best_trial': best and best['trial'],
                'best_val_cvar': best and best['val_cvar'], 'elapsed': time.time() - t0}
        print(json.dumps({k: float(f'{v:.5g}') if isinstance(v, float) else v
                          for k, v in line.items()}), flush=True)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)


if __name__ == '__main__':
    main()
