"""Tune the no-trade band (baselines.band) on validation paths.

The learned hedgers get a sweep of --trials trials, so the band gets the
same budget: Optuna's TPE sampler (Bergstra et al. 2011) over eight of its
parameters (SEARCH; eta_s stays at 1, see baselines.band). The first trial
is the theory point BAND_DEFAULTS (risk aversion 1 / scale). Each trial is scored the way every method
is scored: CVaR_alpha of market.simulate with the hard gate and
deterministic actions on the validation paths (seed 999); the classical
policies ignore w. The test paths (seed 12345) are never touched here.

Also scores the untuned baselines (baselines.untuned) on the same paths.
Writes --out (rewritten after every trial, so an interrupted run still
leaves the best band so far): the config, the greeks grid, the untuned
baselines, every trial (params, val CVaR, mean loss, trades, time), the
best parameters (all nine, ready for baselines.band or tuned_band) and the
tuning wall time. Prints one JSON line for each untuned baseline and each
trial.

    python tune_baselines.py --trials 32 --device cuda
"""

import argparse
import json
import os
import time
from dataclasses import asdict

import optuna
import torch

from market import BatesConfig, market_noise, simulate, cvar
from greeks import BookGreeks
from baselines import BAND_DEFAULTS, band, untuned

VAL_SEED, TEST_SEED = 999, 12345
# name: (low, high, log scale). The width multipliers scale with a power of
# the risk aversion, a scale parameter, so they get log-uniform priors.
SEARCH = {'a_p': (0.01, 3.0, True), 'a_f': (0.01, 3.0, True), 'h_min': (0.0, 0.2, False),
          'a_y': (0.01, 3.0, True), 'y_min': (0.0, 1.0, False), 'eta_y': (0.0, 1.0, False),
          'm_delta': (0.5, 1.5, False), 'm_vega': (0.5, 2.0, False)}


@torch.no_grad()
def score(noise, fn, cfg):
    t0 = time.time()
    loss, rec = simulate(noise, fn, 0.0, cfg, record=True)
    if noise.is_cuda:
        torch.cuda.synchronize()
    return {'val_cvar': cvar(loss.cpu(), cfg.alpha), 'mean': loss.mean().item(),
            'trades_stock': rec['trade_s'].float().sum(1).mean().item(),
            'trades_swap': rec['trade_y'].float().sum(1).mean().item(),
            'time': time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n-val', type=int, default=100_000)
    p.add_argument('--seed', type=int, default=0, help='TPE sampler seed')
    p.add_argument('--out', default='results/baselines.json')
    args = p.parse_args()

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cfg = BatesConfig()
    greeks = BookGreeks(cfg)
    val = market_noise(args.n_val, cfg, torch.Generator().manual_seed(VAL_SEED), args.device)
    out = {'cfg': cfg.to_dict(), 'n_val': args.n_val, 'val_seed': VAL_SEED,
           'device': args.device, 'sampler_seed': args.seed,
           'greeks': {'cache': greeks.path, 'grid': asdict(greeks.spec)},
           'search_space': SEARCH,
           'fixed': {k: v for k, v in BAND_DEFAULTS.items() if k not in SEARCH},
           'untuned': {}, 'trials': [], 'best': None}

    def save():
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)

    for name, fn in untuned(cfg, greeks).items():
        out['untuned'][name] = score(val, fn, cfg)
        print(json.dumps({'baseline': name, **out['untuned'][name]}), flush=True)
    save()

    t0 = time.time()

    def objective(trial):
        params = {**BAND_DEFAULTS,
                  **{k: trial.suggest_float(k, lo, hi, log=log)
                     for k, (lo, hi, log) in SEARCH.items()}}
        r = score(val, band(cfg, greeks, params), cfg)
        rec = {'trial': trial.number, 'params': params, **r}
        out['trials'].append(rec)
        if out['best'] is None or r['val_cvar'] < out['best']['val_cvar']:
            out['best'] = {'trial': trial.number, 'params': params, 'val_cvar': r['val_cvar']}
        out['tune_wall_time'] = time.time() - t0
        out['tune_steps'] = len(out['trials']) * args.n_val * cfg.n_steps
        save()
        print(json.dumps(rec), flush=True)
        return r['val_cvar']

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial({k: BAND_DEFAULTS[k] for k in SEARCH})
    study.optimize(objective, n_trials=args.trials)
    print(json.dumps({'best': out['best'], 'tune_wall_time': out['tune_wall_time']}), flush=True)


if __name__ == '__main__':
    main()
