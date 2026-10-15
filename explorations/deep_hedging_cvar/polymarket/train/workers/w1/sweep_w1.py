"""w1: risk-aware objectives on reward_mode 2 (PROTOCOL.md, DECISIONS.md D8).

Optuna TPE over inv_penalty, close-inventory shaping, markout_h, mark_mode, lr and ent_coef,
starting from the v1 best config (train/selected.json, R7 c02_h120: reward_mode 2,
markout_h 120, MLP 64x2). Each trial = one config trained on 3 seeds (0, 1, 2) at 25M steps,
run one after another inside a slot; 4 slots run in parallel (study.optimize n_jobs=4).
Every run calls the shared train/train_ppo.py unchanged, with val v2 (data/build_v2 split 1),
ledger train/workers/w1/trials.jsonl and tag w1_<experiment>.

Objective (maximised): seed-mean val-v2 P&L total (central fills, 1 s). Configs failing
D1.3 on seed means (fills/market >= 5, T6 <= 0.5, T5 <= 0.25) get P&L - 1000 so TPE
does not drift to no-quote.

    python sweep_w1.py sweep --trials 48 --enqueue   # first slot; then 3 more without --enqueue
    python sweep_w1.py seeds --top 3 --seeds 3 4     # extra seeds of the top configs
    python sweep_w1.py table                          # aggregate -> runs/agg.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, '..', '..'))
PM = os.path.dirname(TRAIN)
RUNS = os.path.join(HERE, 'runs')
LEDGER = os.path.join(HERE, 'trials.jsonl')
V2 = os.path.join(PM, 'data', 'build_v2')
BASE = json.load(open(os.path.join(TRAIN, 'selected.json')))['selected']['cfg']
STEPS = 25e6
PY = sys.executable
_lock = threading.Lock()


def cfg_args(c):
    """CLI args for train_ppo.py from a w1 config dict."""
    a = ['--hidden', '64', '--layers', '2', '--off-prior', str(BASE['policy']['off_prior']),
         '--gamma', str(BASE['ppo']['gamma']), '--gae-lambda', str(BASE['ppo']['gae_lambda']),
         '--reward-mode', '2', '--markout-h', str(c['markout_h']), '--mark-mode', str(c['mark_mode']),
         '--inv-penalty', repr(c['inv_penalty']), '--lr', repr(c['lr']), '--ent-coef', repr(c['ent_coef'])]
    if c.get('close_inv_coef', 0) > 0:
        a += ['--close-inv-coef', repr(c['close_inv_coef']), '--close-ramp', str(c['close_ramp'])]
    return a


def run_one(name, c, seed, steps=STEPS):
    out = os.path.join(RUNS, name, f's{seed}')
    res = os.path.join(out, 'result.json')
    if not os.path.exists(res):
        os.makedirs(out, exist_ok=True)
        cmd = [PY, os.path.join(TRAIN, 'train_ppo.py'), '--env', 'real', '--timesteps', str(steps),
               '--threads', '1', '--seed', str(seed), '--out', out, '--tag', f'w1_{name}',
               '--ledger', LEDGER, '--variants', 'central_1s', '--no-train-eval', '--quiet',
               '--env-kw', f'data_dir={V2}'] + cfg_args(c)
        env = dict(os.environ, OMP_WAIT_POLICY='passive', OMP_NUM_THREADS='1')
        with open(os.path.join(out, 'stdout.log'), 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=TRAIN)
        if r.returncode != 0 or not os.path.exists(res):
            raise RuntimeError(f'{name} s{seed} failed ({r.returncode}); see {out}/stdout.log')
    return json.load(open(res))


def metrics(result):
    v = result['val']['central_1s']
    keys = ('n_markets', 'pnl_total', 'pnl_mean', 'cvar95', 'fills_per_market', 't6_share', 't5_max_share',
            'makes_markets', 'max_abs_inv', 'sum/spread_capture', 'sum/markout_60', 'sum/inventory_pnl',
            'sum/rebates', 'mean/uptime_two_sided', 'len/5m/pnl_total', 'len/15m/pnl_total',
            'len/4h/pnl_total', 'len/5m/n', 'len/15m/n', 'len/4h/n', 'len/5m/fills_per_market',
            'len/15m/fills_per_market', 'len/4h/fills_per_market')
    return {k: v.get(k) for k in keys}


def aggregate(ms):
    out = {'n_seeds': len(ms)}
    for k in ms[0]:
        x = [m[k] for m in ms if m[k] is not None and not (isinstance(m[k], float) and math.isnan(m[k]))]
        if k == 't6_share':  # protocol_flags: T6 only applies when total P&L > 0
            x = [m[k] if (m[k] is not None and m['pnl_total'] > 0) else 0.0 for m in ms]
        if not x:
            out[k] = None
            continue
        out[k] = float(np.mean(x))
        if k in ('pnl_total', 'pnl_mean') or k.startswith('len/') and k.endswith('pnl_total'):
            out[k + '_se'] = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    t5 = out.get('t5_max_share') or 0.0
    out['d13'] = bool(out['fills_per_market'] >= 5 and out['t6_share'] <= 0.5 and t5 <= 0.25)
    return out


def record(name, c, seeds, agg):
    with _lock:
        with open(os.path.join(HERE, 'configs.jsonl'), 'a') as f:
            f.write(json.dumps({'name': name, 'cfg': c, 'seeds': seeds, 'agg': agg}) + '\n')


def sample(trial):
    c = {'inv_penalty': trial.suggest_float('inv_penalty', 1e-4, 3e-2, log=True),
         'markout_h': trial.suggest_categorical('markout_h', [60, 120, 300]),
         'mark_mode': trial.suggest_categorical('mark_mode', [1, 2]),
         'lr': trial.suggest_float('lr', 1e-4, 1.5e-3, log=True),
         'ent_coef': trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)}
    if trial.suggest_categorical('close', [0, 1]):
        c['close_inv_coef'] = trial.suggest_float('close_inv_coef', 1e-3, 1e-1, log=True)
        c['close_ramp'] = trial.suggest_categorical('close_ramp', [0.1, 0.2, 0.4])
    return c


def cmd_sweep(a):
    import optuna
    os.makedirs(RUNS, exist_ok=True)
    study = optuna.create_study(study_name='w1', direction='maximize', load_if_exists=True,
                                storage=f'sqlite:///{os.path.join(RUNS, "optuna.db")}',
                                sampler=optuna.samplers.TPESampler(seed=a.sampler_seed or os.getpid(), n_startup_trials=10,
                                                                   constant_liar=True))
    if not study.trials and a.enqueue:
        # trial 0: the v1 best (R7 c02_h120); trials 1-3: hand-picked risk-aware variants
        study.enqueue_trial({'inv_penalty': 1e-4, 'markout_h': 120, 'mark_mode': 1,
                             'lr': BASE['ppo']['learning_rate'], 'ent_coef': BASE['ppo']['ent_coef'], 'close': 0})
        study.enqueue_trial({'inv_penalty': 3e-3, 'markout_h': 120, 'mark_mode': 1,
                             'lr': BASE['ppo']['learning_rate'], 'ent_coef': BASE['ppo']['ent_coef'],
                             'close': 1, 'close_inv_coef': 1e-2, 'close_ramp': 0.2})
        study.enqueue_trial({'inv_penalty': 1e-2, 'markout_h': 300, 'mark_mode': 1,
                             'lr': BASE['ppo']['learning_rate'], 'ent_coef': BASE['ppo']['ent_coef'], 'close': 0})
        study.enqueue_trial({'inv_penalty': 1e-3, 'markout_h': 60, 'mark_mode': 2,
                             'lr': BASE['ppo']['learning_rate'], 'ent_coef': 3e-3,
                             'close': 1, 'close_inv_coef': 3e-2, 'close_ramp': 0.4})
    for t in (study.trials if a.requeue else []):  # a killed run leaves RUNNING trials: mark failed, re-queue its params
        if t.state == optuna.trial.TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            study.enqueue_trial(t.params)

    def objective(trial):
        c = sample(trial)
        name = f't{trial.number:02d}'
        trial.set_user_attr('cfg', c)
        ms = [metrics(run_one(name, c, s)) for s in a.seeds]
        agg = aggregate(ms)
        record(name, c, a.seeds, agg)
        for k in ('pnl_total', 'pnl_total_se', 'fills_per_market', 't6_share', 't5_max_share', 'd13'):
            trial.set_user_attr(k, agg.get(k))
        return agg['pnl_total'] - (0.0 if agg['d13'] else 1000.0)

    # one thread per process (n_jobs threads raced on enqueued trials: 3 threads ran trial 0);
    # parallel slots = several processes on the same sqlite study, capped at --trials in total
    study.optimize(objective, n_jobs=1,
                   callbacks=[optuna.study.MaxTrialsCallback(a.trials, states=None)])


def load_configs():
    rows = {}
    for line in open(os.path.join(HERE, 'configs.jsonl')):
        r = json.loads(line)
        rows[r['name']] = r  # later rows (more seeds) supersede
    return rows


def cmd_seeds(a):
    from concurrent.futures import ThreadPoolExecutor
    rows = load_configs()
    if a.names:
        top = [rows[n] for n in a.names]
    else:
        ok = [r for r in rows.values() if r['agg']['d13']]
        top = sorted(ok, key=lambda r: -r['agg']['pnl_total'])[:a.top]
    print('top:', [(r['name'], round(r['agg']['pnl_total'], 1)) for r in top])
    jobs = [(r, s) for r in top for s in a.seeds]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: run_one(j[0]['name'], j[0]['cfg'], j[1]), jobs))
    for r in top:
        seeds = sorted(set(r['seeds']) | set(a.seeds))
        ms = [metrics(run_one(r['name'], r['cfg'], s)) for s in seeds]
        record(r['name'], r['cfg'], seeds, aggregate(ms))


def cmd_table(a):
    rows = load_configs()
    json.dump(rows, open(os.path.join(RUNS, 'agg.json'), 'w'), indent=1)
    for r in sorted(rows.values(), key=lambda r: -r['agg']['pnl_total']):
        g = r['agg']
        print(f"{r['name']} n={g['n_seeds']} pnl={g['pnl_total']:8.1f}±{g.get('pnl_total_se') or 0:6.1f} "
              f"$/mkt={g['pnl_mean']:6.3f} fills={g['fills_per_market']:6.1f} t6={g['t6_share']:.2f} "
              f"inv={g['max_abs_inv']:.0f} d13={g['d13']} {r['cfg']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sweep')
    s.add_argument('--trials', type=int, default=32)
    s.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    s.add_argument('--sampler-seed', type=int, default=None,
                   help='TPE seed (default: pid; equal seeds in parallel processes sample equal configs)')
    s.add_argument('--enqueue', action='store_true', help='new study: enqueue the 4 hand-picked trials')
    s.add_argument('--requeue', action='store_true', help='re-queue trials a kill left RUNNING')
    s = sub.add_parser('seeds')
    s.add_argument('--top', type=int, default=3)
    s.add_argument('--names', nargs='*')
    s.add_argument('--seeds', type=int, nargs='*', default=[3, 4])
    s.add_argument('--jobs', type=int, default=4)
    sub.add_parser('table')
    a = p.parse_args()
    {'sweep': cmd_sweep, 'seeds': cmd_seeds, 'table': cmd_table}[a.cmd](a)
