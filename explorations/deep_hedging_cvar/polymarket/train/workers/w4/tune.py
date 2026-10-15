"""w4: tune the learned quoter's knobs on val v2 (64 TPE trials, the baselines' budget) and evaluate.

Each trial evaluates the same knobs with each data-seed model (3 models = 3 seeds), greedy, every
val v2 market once (train/evaluator.py, central fills, 1 s latency, data/build_v2 split 1), and
logs one ledger row per evaluation to workers/w4/trials.jsonl. Objective: seed-mean pnl_total,
minus a penalty of 1000 when the seed-mean policy fails D1.3 (fills/market < 5, T6 > 0.5 or
T5 > 0.25), so the tuner looks for the best market-making setting rather than no-quote.

    python tune.py tune --models models/edge_s1.pt models/edge_s2.pt models/edge_s3.pt --trials 64
    python tune.py eval --params runs/tune_best.json --variants central_1s pess_1s central_2s
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, '..', '..'))
PM = os.path.dirname(TRAIN)
sys.path.insert(0, TRAIN)
sys.path.insert(0, HERE)
import pufferlib.pufferl  # noqa: E402,F401
import ledger  # noqa: E402
from envs import get_adapter  # noqa: E402
from evaluator import evaluate, nan_to_none  # noqa: E402
from quoter import Quoter  # noqa: E402

V2 = os.path.join(PM, 'data', 'build_v2')
LEDGER = os.path.join(HERE, 'trials.jsonl')
VARIANTS = {'central_1s': {'fill_model': 1, 'latency': 1}, 'pess_1s': {'fill_model': 2, 'latency': 1},
            'central_2s': {'fill_model': 1, 'latency': 2}}
KEYS = ('n_markets', 'pnl_total', 'pnl_mean', 'fills_per_market', 't6_share', 't6_share_pos', 't5_max_share',
        'makes_markets', 'sum/spread_capture', 'sum/markout_60', 'sum/inventory_pnl', 'sum/rebates',
        'mean/uptime_two_sided', 'len/5m/pnl_mean', 'len/15m/pnl_mean', 'len/4h/pnl_mean',
        'len/5m/fills_per_market', 'len/15m/fills_per_market', 'len/4h/fills_per_market', 'cvar95')


class FnPolicy(torch.nn.Module):
    recurrent = False

    def __init__(self, fn, nvec=(7, 7, 3)):
        super().__init__()
        self.fn, self.nvec = fn, nvec

    def forward_eval(self, obs, state=None):
        a = torch.as_tensor(np.asarray(self.fn(obs.numpy()), dtype=np.int64))
        logits = tuple(torch.nn.functional.one_hot(a[:, h], n).float() * 20.0 for h, n in enumerate(self.nvec))
        return logits, torch.zeros(obs.shape[0], 1)


_ADAPTER = None


def run_eval(fn, variant='central_1s', tag='w4_quoter', name='', spec=None, seed=None):
    global _ADAPTER
    _ADAPTER = _ADAPTER or get_adapter('real')
    kw = {**VARIANTS[variant], 'data_dir': V2}
    t0 = time.time()
    res = evaluate(FnPolicy(fn), _ADAPTER, split=1, env_kw=kw)
    assert res['n_markets'] == 374, res['n_markets']
    ledger.append({'kind': 'val_eval', 'split': 1, 'protocol': 'v2', 'source': 'workers/w4/tune.py',
                   'backend': 'real', 'tag': tag, 'name': name, 'seed': seed, 'data_dir': V2,
                   'variant': variant, 'policy_spec': spec,
                   'config_hash': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                   'val': nan_to_none(res)}, LEDGER)
    out = {k: res.get(k) for k in KEYS}
    out['seconds'] = round(time.time() - t0, 1)
    return out


def seed_summary(rows):
    pn = np.array([r['pnl_mean'] for r in rows])
    out = {'pnl_mean': float(pn.mean()), 'pnl_mean_se': float(pn.std(ddof=1) / np.sqrt(len(pn))) if len(pn) > 1 else 0.0,
           'pnl_total': float(np.mean([r['pnl_total'] for r in rows])),
           'fills_per_market': float(np.mean([r['fills_per_market'] for r in rows])),
           'makes_markets_all': bool(all(r['makes_markets'] for r in rows)),
           't6_share': [r['t6_share'] for r in rows], 't5_max_share': [r['t5_max_share'] for r in rows]}
    for L in ('5m', '15m', '4h'):
        out[f'pnl_mean_{L}'] = float(np.mean([r[f'len/{L}/pnl_mean'] for r in rows]))
    return out


def tune(a):
    import optuna
    models = a.models
    quoters = [Quoter(m) for m in models]
    log = open(os.path.join(HERE, 'runs', f'{a.study}_log.jsonl'), 'a')

    def objective(trial):
        if a.per_length:
            P = dict(k_inv=trial.suggest_float('k_inv', 0.0, 0.12), cap=trial.suggest_float('cap', 0.1, 1.0),
                     max_size=trial.suggest_int('max_size', 0, 2), mode=trial.suggest_categorical('mode', [0, 1, 2]),
                     thr=0.0, edge_min=0.0)
            for L in ('5m', '15m', '4h'):
                P[f'thr_{L}'] = trial.suggest_float(f'thr_{L}', -0.01, 0.05)
                P[f'edge_min_{L}'] = trial.suggest_float(f'edge_min_{L}', -1.0, 6.0)
            return run_trial(trial, P)
        P = dict(thr=trial.suggest_float('thr', -0.01, 0.05),
                 edge_min=trial.suggest_float('edge_min', -1.0, 3.0),
                 k_inv=trial.suggest_float('k_inv', 0.0, 0.10),
                 cap=trial.suggest_float('cap', 0.1, 1.0),
                 max_size=trial.suggest_int('max_size', 0, 2),
                 mode=trial.suggest_categorical('mode', [0, 1, 2]))
        return run_trial(trial, P)

    def run_trial(trial, P):
        rows = []
        for i, q in enumerate(quoters):
            q.params.update(P)
            spec = {'kind': 'w4_quoter', 'model': os.path.basename(models[i]), 'params': P, 'study': a.study,
                    'trial': trial.number}
            rows.append(run_eval(q, 'central_1s', tag=f'w4_{a.study}', name=f't{trial.number}_m{i}', spec=spec, seed=i + 1))
        s = seed_summary(rows)
        mm = s['fills_per_market'] >= 5 and all(r['makes_markets'] for r in rows)
        obj = s['pnl_total'] - (0 if mm else 1000)
        log.write(json.dumps({'trial': trial.number, 'params': P, 'summary': s, 'per_seed': rows, 'obj': obj}) + '\n')
        log.flush()
        print(json.dumps({'trial': trial.number, 'obj': round(obj, 2), **P, 'pnl_mean': round(s['pnl_mean'], 4),
                          'se': round(s['pnl_mean_se'], 4), 'fills': round(s['fills_per_market'], 1), 'mm': mm}),
              flush=True)
        return obj

    storage = f'sqlite:///{os.path.join(HERE, "runs", a.study + ".db")}'
    study = optuna.create_study(study_name=a.study, storage=storage, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12))
    if len(study.trials) == 0 and a.per_length:
        b = json.load(open(os.path.join(HERE, 'runs', 'q1_best.json')))['params']   # start from q1's best
        study.enqueue_trial({**{k: b[k] for k in ('k_inv', 'cap', 'max_size', 'mode')},
                             **{f'thr_{L}': b['thr'] for L in ('5m', '15m', '4h')},
                             **{f'edge_min_{L}': b['edge_min'] for L in ('5m', '15m', '4h')}})
    elif len(study.trials) == 0:
        study.enqueue_trial(dict(thr=0.0, edge_min=0.0, k_inv=0.02, cap=0.6, max_size=2, mode=0))
    left = a.trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    best = study.best_trial
    bp = dict(best.params)
    if a.per_length:
        bp.update(thr=0.0, edge_min=0.0)
    json.dump({'params': bp, 'value': best.value, 'models': models, 'study': a.study},
              open(os.path.join(HERE, 'runs', f'{a.study}_best.json'), 'w'), indent=1)
    print('best', best.number, best.value, best.params)


def evalcmd(a):
    b = json.load(open(a.params))
    P = b['params']
    models = a.models or b['models']
    res = {}
    for v in a.variants:
        rows = []
        for i, m in enumerate(models):
            q = Quoter(m, **P)
            spec = {'kind': 'w4_quoter', 'model': os.path.basename(m), 'params': P, 'study': b.get('study'), 'final': True}
            rows.append(run_eval(q, v, tag=f'w4_{a.tag}', name=f'{a.tag}_{v}_m{i}', spec=spec, seed=i + 1))
            print(json.dumps({'variant': v, 'model': m, **rows[-1]}, default=float), flush=True)
        res[v] = {'per_seed': rows, 'summary': seed_summary(rows)}
        print(json.dumps({'variant': v, **res[v]['summary']}, default=float), flush=True)
    json.dump(res, open(os.path.join(HERE, 'runs', f'{a.tag}_eval.json'), 'w'), indent=1, default=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['tune', 'eval'])
    p.add_argument('--models', nargs='*')
    p.add_argument('--trials', type=int, default=64)
    p.add_argument('--study', default='q1')
    p.add_argument('--params')
    p.add_argument('--tag', default='final')
    p.add_argument('--variants', nargs='*', default=['central_1s', 'pess_1s', 'central_2s'])
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--per-length', action='store_true', help='thr / edge_min per contract length')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    tune(a) if a.cmd == 'tune' else evalcmd(a)


if __name__ == '__main__':
    main()
