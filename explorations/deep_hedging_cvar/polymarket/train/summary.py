"""Table of training runs: validation results, attribution, protocol flags and diagnostics.

    python summary.py --tag explore             # runs.jsonl rows with this tag
    python summary.py --runs runs/explore/*      # or run directories
    python summary.py --tag s1 --by-config       # seed means per config hash

Per run (or per config: seed means): the sweep knobs that differ, val (headline)
P&L total / per market, CVaR95, fills per market, attribution sums (spread capture,
60 s markout, inventory/resolution, rebates), T6 directional share, T5 artifact
share, makes_markets (DECISIONS.md D1.3), two-sided uptime, greedy action mix on
val (share 'off' per price head), pessimistic-fill P&L, train-split P&L (gap), and
the last training diagnostics (entropy per head, approx KL, explained variance).
Prints markdown; --json writes the rows.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ledger  # noqa: E402

KNOBS = [('ppo', 'learning_rate', 'lr'), ('ppo', 'ent_coef', 'ent'), ('ppo', 'gamma', 'gamma'),
         ('ppo', 'gae_lambda', 'lam'), ('env', 'kw', 'inv_penalty', 'invpen'), ('env', 'kw', 'mark_mode', 'mark'),
         ('policy', 'hidden', 'H'), ('total_timesteps', 'steps')]


def knob(cfg, path):
    x = cfg
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def last_diag(run_dir):
    p = os.path.join(run_dir or '', 'log.jsonl')
    if not os.path.exists(p):
        return {}
    ev = [json.loads(l) for l in open(p) if '"type": "eval"' in l]
    return ev[-1] if ev else {}


def row_of(r):
    h = r.get('headline') or 'central_1s'
    v = (r.get('val') or {}).get(h, {})
    pv = (r.get('val') or {}).get('pess_1s', {})
    te = r.get('train_eval') or {}
    d = last_diag(r.get('run_dir'))
    out = {'run': os.path.basename(r.get('run_dir') or ''), 'seed': r.get('seed'), 'cfg': r.get('cfg_hash', '')[:8]}
    for *path, name in KNOBS:
        out[name] = knob(r['cfg'], path)
    out.update({
        'val_pnl': v.get('pnl_total'), 'val_pnl/mkt': v.get('pnl_mean'), 'cvar95': v.get('cvar95'),
        'fills/mkt': v.get('fills_per_market'), 'spread': v.get('sum/spread_capture'),
        'markout60': v.get('sum/markout_60'), 'inventory': v.get('sum/inventory_pnl'),
        'rebates': v.get('sum/rebates'), 't6': v.get('t6_share'), 't6_pos': v.get('t6_share_pos'),
        't5': v.get('t5_max_share'), 'mm': v.get('makes_markets'), 'uptime2': v.get('mean/uptime_two_sided'),
        'off_bid': v.get('act/bid/off'), 'off_ask': v.get('act/ask/off'),
        'pess_pnl': pv.get('pnl_total'), 'train_pnl/mkt': te.get('pnl_mean'),
        'H_heads': [d.get(f'diag/losses/entropy_head{i}') for i in range(3)],
        'kl': d.get('diag/losses/approx_kl'), 'ev': d.get('diag/losses/explained_variance'),
        'sps': r.get('sps_median'),
    })
    return out


def fmt(x):
    if x is None:
        return '-'
    if isinstance(x, list):
        return '/'.join(fmt(y) for y in x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return 'nan'
        if abs(x) >= 100:
            return f'{x:.0f}'
        if abs(x) >= 1:
            return f'{x:.2f}'
        return f'{x:.3g}'
    return str(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs-index', default=ledger.RUNS)
    p.add_argument('--tag', default=None)
    p.add_argument('--runs', nargs='*', default=None)
    p.add_argument('--by-config', action='store_true')
    p.add_argument('--sort', default='val_pnl')
    p.add_argument('--json', default=None)
    a = p.parse_args()
    if a.runs:
        rows = [json.load(open(os.path.join(d, 'result.json'))) for ds in a.runs for d in glob.glob(ds)
                if os.path.exists(os.path.join(d, 'result.json'))]
    else:
        rows = [r for r in ledger.read(a.runs_index) if r.get('kind') == 'train_run'
                and (a.tag is None or r.get('tag') == a.tag)]
    table = [row_of(r) for r in rows]
    if a.by_config:
        groups = {}
        for t in table:
            groups.setdefault(t['cfg'], []).append(t)
        table = []
        for h, ts in groups.items():
            m = dict(ts[0])
            m['seed'] = f'{len(ts)} seeds'
            for k, v in ts[0].items():
                vals = [t[k] for t in ts if isinstance(t[k], (int, float)) and not isinstance(t[k], bool)
                        and t[k] is not None and math.isfinite(t[k])]
                if vals and isinstance(v, float):
                    m[k] = float(np.mean(vals))
            m['val_pnl_min'] = min((t['val_pnl'] for t in ts if t['val_pnl'] is not None), default=None)
            m['val_pnl_max'] = max((t['val_pnl'] for t in ts if t['val_pnl'] is not None), default=None)
            table.append(m)
    table.sort(key=lambda t: -1e18 if t.get(a.sort) is None else t[a.sort], reverse=True)
    if not table:
        print('no rows')
        return
    cols = list(table[0].keys())
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for t in table:
        print('| ' + ' | '.join(fmt(t.get(c)) for c in cols) + ' |')
    if a.json:
        json.dump(table, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
