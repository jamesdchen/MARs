"""Pick the config to carry forward, by the pre-registered rule (EVALUATION.md T4).

    python select.py --tag s1                       # -> runs/sweeps/s1/selected.json
    python select.py --tag s1 --cvar-floor -12.5 --max-inv 100

Reads the ledger, groups the training runs by config hash (configs that
differ only in seed), keeps configs with at least --min-seeds distinct seeds
(the first ledger row of each seed counts; reruns of a seed are ignored), and
picks the config that maximizes the SEED-MEAN of the validation metric (default
total P&L, headline variant central fills / 1 s), subject to
  - seed-mean CVaR95 of per-market P&L >= --cvar-floor (the best baseline's
    CVaR95, when given), and
  - max |inventory| over every seed and market <= --max-inv (when given).
Ties (within --tie-tol) go to the higher seed-mean pessimistic-fill P&L
(--tiebreak-variant). Never the best seed. Writes selected.json with the
config, its per-seed results, the full candidate table and the ledger's trial
count N (all rows, for the deflated Sharpe).
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ledger  # noqa: E402


def val_of(row, variant, key):
    v = (row.get('val') or {}).get(variant or row.get('headline')) or {}
    return v.get(key)


def candidates(rows, metric, variant, tiebreak_variant):
    groups = OrderedDict()
    for r in rows:
        if r.get('kind') != 'train_run':
            continue
        g = groups.setdefault(r['cfg_hash'], {'cfg': r['cfg'], 'runs': OrderedDict()})
        g['runs'].setdefault(r['seed'], r)  # first row per seed
    out = []
    for h, g in groups.items():
        runs = list(g['runs'].values())
        vals = [val_of(r, variant, metric) for r in runs]
        if any(v is None for v in vals):
            continue
        cv = [val_of(r, variant, 'cvar95') for r in runs]
        inv = [val_of(r, variant, 'max_abs_inv') for r in runs]
        tb = [val_of(r, tiebreak_variant, metric) for r in runs]
        out.append({
            'cfg_hash': h, 'cfg': g['cfg'], 'n_seeds': len(runs), 'seeds': [r['seed'] for r in runs],
            'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'min': float(np.min(vals)),
            'max': float(np.max(vals)), 'per_seed': dict(zip([r['seed'] for r in runs], vals)),
            'cvar95_mean': float(np.mean(cv)) if all(c is not None for c in cv) else None,
            'max_abs_inv': float(np.max(inv)) if all(i is not None for i in inv) else None,
            'tiebreak_mean': float(np.mean(tb)) if tb and all(t is not None for t in tb) else None,
            'runs': [{'seed': r['seed'], 'run_dir': r.get('run_dir'), 'ckpt': r.get('ckpt'),
                      'ckpt_sha256': r.get('ckpt_sha256'), 'tag': r.get('tag')} for r in runs],
        })
    return out


def choose(cands, min_seeds, cvar_floor=None, max_inv=None, tie_tol=0.0):
    ok = []
    for c in cands:
        reasons = []
        if c['n_seeds'] < min_seeds:
            reasons.append(f'{c["n_seeds"]} < {min_seeds} seeds')
        if cvar_floor is not None and (c['cvar95_mean'] is None or c['cvar95_mean'] < cvar_floor):
            reasons.append(f'cvar95 {c["cvar95_mean"]} < floor {cvar_floor}')
        if max_inv is not None and (c['max_abs_inv'] is None or c['max_abs_inv'] > max_inv):
            reasons.append(f'max_abs_inv {c["max_abs_inv"]} > {max_inv}')
        c['excluded'] = reasons
        if not reasons:
            ok.append(c)
    if not ok:
        return None
    best = max(c['mean'] for c in ok)
    tied = [c for c in ok if c['mean'] >= best - tie_tol]
    return max(tied, key=lambda c: (c['tiebreak_mean'] if c['tiebreak_mean'] is not None else -np.inf,
                                    c['mean']))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ledger', default=ledger.LEDGER)
    p.add_argument('--tag', default=None, help='only runs with this ledger tag (sweep name)')
    p.add_argument('--metric', default='pnl_total')
    p.add_argument('--variant', default=None, help='val variant (default: each run\'s headline)')
    p.add_argument('--tiebreak-variant', default='pess_1s')
    p.add_argument('--min-seeds', type=int, default=3)
    p.add_argument('--cvar-floor', type=float, default=None)
    p.add_argument('--max-inv', type=float, default=None)
    p.add_argument('--tie-tol', type=float, default=0.0)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    if a.min_seeds < 3:
        raise SystemExit('the protocol selects on the mean over >= 3 seeds')

    all_rows = ledger.read(a.ledger)
    rows = [r for r in all_rows if a.tag is None or r.get('tag') == a.tag]
    cands = candidates(rows, a.metric, a.variant, a.tiebreak_variant)
    sel = choose(cands, a.min_seeds, a.cvar_floor, a.max_inv, a.tie_tol)
    cands.sort(key=lambda c: c['mean'], reverse=True)
    out = a.out or os.path.join(HERE, 'runs', 'sweeps', a.tag or 'all', 'selected.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = {
        'rule': {'metric': a.metric, 'variant': a.variant or 'headline', 'statistic': 'seed mean',
                 'min_seeds': a.min_seeds, 'cvar_floor': a.cvar_floor, 'max_inv': a.max_inv,
                 'tiebreak_variant': a.tiebreak_variant, 'tie_tol': a.tie_tol, 'tag': a.tag},
        'ledger': os.path.abspath(a.ledger),
        'n_trials_ledger': len(all_rows),
        'n_val_evals_ledger': int(sum(r.get('n_val_evals', 1) for r in all_rows)),
        'selected': sel,
        'cfg': sel['cfg'] if sel else None,
        'candidates': [{k: v for k, v in c.items() if k != 'cfg'} for c in cands],
    }
    with open(out, 'w') as f:
        json.dump(doc, f, indent=1, default=float)
    if sel is None:
        print(json.dumps({'selected': None, 'candidates': len(cands), 'out': out,
                          'why': [c['excluded'] for c in cands][:10]}))
        sys.exit(1)
    print(json.dumps({'selected': sel['cfg_hash'], 'mean': sel['mean'], 'std': sel['std'],
                      'seeds': sel['seeds'], 'candidates': len(cands), 'out': out}))


if __name__ == '__main__':
    main()
