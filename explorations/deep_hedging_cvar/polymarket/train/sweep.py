"""Optuna (TPE) sweep of PPO hyperparameters, scored on validation.

    python sweep.py --name s1 --trials 24 --steps 20e6 --jobs 2 --threads 2
    python sweep.py --name s1 --trials 24 --steps 20e6 --reseed-top 3 --reseed-n 3   # resumes
    python select.py --tag s1

Fixed budget: --trials trials, each training for --steps agent steps
(train_ppo.py, one subprocess per run, --jobs at a time with --threads torch
threads each). Each trial trains --seeds-per-trial seeds (default 1, seed 0
for every trial: common random numbers) and its objective is the seed-mean of
the validation metric (default: total P&L over all val markets, headline
variant central fills / 1 s). Every run appends its validation evaluations to
the ledger (train/trials.jsonl) and its summary to train/runs.jsonl, tagged with
the sweep name, so the ledger's N counts all of them. Then --reseed-top K re-trains the K best configs with more seeds until
each has --reseed-n seeds, so select.py can pick by the seed-mean over >= 3
seeds rather than by a lucky seed.

Search space (SPACE): learning rate, entropy coefficient, gamma, GAE lambda,
the env's inventory penalty, MLP hidden size, and the reward mark (mark_mode 0
book mid / 1 fair value). Other settings come from
--base-config (a JSON config as train_ppo.py --config takes) and the
train_ppo.py defaults; extra train_ppo.py flags can follow `--`.
State: runs/sweeps/<name>/ (optuna.db, sweep_log.jsonl, one run dir per run).
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SPACE = {
    'lr': ('log', 1e-4, 3e-3),
    'ent_coef': ('log', 1e-4, 5e-2),
    'one_minus_gamma': ('log', 5e-4, 5e-2),   # gamma in [0.95, 0.9995]
    'gae_lambda': ('float', 0.8, 0.99),
    'inv_penalty': ('log', 1e-5, 3e-2),
    'hidden': ('cat', [64, 128, 256]),
    'mark_mode': ('cat', [0, 1]),             # reward mark: book mid / fair value
}


def suggest(trial):
    p = {}
    for k, spec in SPACE.items():
        if spec[0] == 'log':
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=True)
        elif spec[0] == 'float':
            p[k] = trial.suggest_float(k, spec[1], spec[2])
        else:
            p[k] = trial.suggest_categorical(k, spec[1])
    return p


def run_args(params):
    return ['--lr', str(params['lr']), '--ent-coef', str(params['ent_coef']),
            '--gamma', str(1.0 - params['one_minus_gamma']), '--gae-lambda', str(params['gae_lambda']),
            '--inv-penalty', str(params['inv_penalty']), '--hidden', str(params['hidden'])] + (
                ['--mark', str(params['mark_mode'])] if 'mark_mode' in params else [])


def train_run(args, params, seed, run_dir, extra):
    os.makedirs(run_dir, exist_ok=True)
    res_path = os.path.join(run_dir, 'result.json')
    if os.path.exists(res_path):  # resumed sweep: do not retrain (and do not re-ledger)
        return json.load(open(res_path))
    cmd = [PY, os.path.join(HERE, 'train_ppo.py'), '--timesteps', str(args.steps), '--seed', str(seed),
           '--threads', str(args.threads), '--out', run_dir, '--tag', args.name, '--quiet',
           '--env', args.env, *run_args(params)]
    if args.base_config:
        cmd += ['--config', args.base_config]
    if args.ledger:
        cmd += ['--ledger', args.ledger]
    cmd += extra
    with open(os.path.join(run_dir, 'stdout.txt'), 'w') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': str(args.threads)})
    if proc.returncode != 0 or not os.path.exists(res_path):
        raise RuntimeError(f'run failed ({proc.returncode}): {run_dir}/stdout.txt')
    return json.load(open(res_path))


def metric_of(res, metric, variant=None):
    v = res['val'][variant or res['headline']]
    return v.get(metric)


def main():
    import optuna
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--env', default='auto', choices=['auto', 'standin', 'real'])
    p.add_argument('--trials', type=int, default=24, help='trial budget (same as the baselines)')
    p.add_argument('--steps', type=float, default=20e6, help='agent steps per run')
    p.add_argument('--seeds-per-trial', type=int, default=1)
    p.add_argument('--reseed-top', type=int, default=0, help='re-train the K best configs ...')
    p.add_argument('--reseed-n', type=int, default=3, help='... until each has this many seeds')
    p.add_argument('--metric', default='pnl_total')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--threads', type=int, default=1, help='torch threads per run (1: see bench.py)')
    p.add_argument('--base-config', default=None)
    p.add_argument('--ledger', default=None)
    p.add_argument('--sampler-seed', type=int, default=0)
    args, extra = p.parse_known_args()
    if extra and extra[0] == '--':
        extra = extra[1:]
    args.steps = int(args.steps)

    root = os.path.join(HERE, 'runs', 'sweeps', args.name)
    os.makedirs(root, exist_ok=True)
    log_path = os.path.join(root, 'sweep_log.jsonl')  # not 'trials.jsonl': final/ counts every such file
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.name, storage=f'sqlite:///{os.path.join(root, "optuna.db")}',
        load_if_exists=True, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, constant_liar=True,
                                           n_startup_trials=max(4, args.trials // 4)))

    def log(row):
        with open(log_path, 'a') as f:
            f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **row}, default=float) + '\n')

    def run_trial(trial):
        params = suggest(trial)
        vals, runs = [], []
        for s in range(args.seeds_per_trial):
            rd = os.path.join(root, f't{trial.number:03d}_s{s}')
            res = train_run(args, params, s, rd, extra)
            vals.append(metric_of(res, args.metric))
            runs.append({'seed': s, 'run_dir': rd, 'cfg_hash': res['cfg_hash'], 'value': vals[-1],
                         'sps': res['sps_median']})
        return params, float(np.mean(vals)), runs

    done = len([t for t in study.trials if t.state.is_finished()])
    todo = max(0, args.trials - done)
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        pending = {}
        while todo or pending:
            while todo and len(pending) < args.jobs:
                trial = study.ask()
                pending[ex.submit(run_trial, trial)] = trial
                todo -= 1
            fin, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in fin:
                trial = pending.pop(fut)
                try:
                    params, value, runs = fut.result()
                    study.tell(trial, value)
                    trial_cfg = runs[0]['cfg_hash']
                    log({'trial': trial.number, 'params': params, 'value': value, 'runs': runs,
                         'cfg_hash': trial_cfg})
                    print(json.dumps({'trial': trial.number, 'value': value, 'params': params}), flush=True)
                except Exception as e:  # noqa: BLE001
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    log({'trial': trial.number, 'failed': str(e)})
                    print(json.dumps({'trial': trial.number, 'failed': str(e)}), flush=True)

    if args.reseed_top:
        ok = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top = sorted(ok, key=lambda t: t.value, reverse=True)[:args.reseed_top]
        jobs = []
        for t in top:
            for s in range(args.seeds_per_trial, args.reseed_n):
                jobs.append((t, s))
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(train_run, args, t.params, s, os.path.join(root, f't{t.number:03d}_s{s}'),
                              extra): (t, s) for t, s in jobs}
            for fut in cf.as_completed(futs):
                t, s = futs[fut]
                try:
                    res = fut.result()
                    v = metric_of(res, args.metric)
                    log({'reseed_of': t.number, 'seed': s, 'value': v, 'cfg_hash': res['cfg_hash']})
                    print(json.dumps({'reseed_of': t.number, 'seed': s, 'value': v}), flush=True)
                except Exception as e:  # noqa: BLE001
                    log({'reseed_of': t.number, 'seed': s, 'failed': str(e)})

    best = study.best_trial if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None
    summary = {'name': args.name, 'trials': len(study.trials), 'steps_per_run': args.steps,
               'metric': args.metric, 'best_single_trial': best and {'number': best.number,
                                                                     'value': best.value,
                                                                     'params': best.params},
               'note': 'best_single_trial is NOT the selection; run select.py (seed-mean over >= 3 seeds)'}
    json.dump(summary, open(os.path.join(root, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
