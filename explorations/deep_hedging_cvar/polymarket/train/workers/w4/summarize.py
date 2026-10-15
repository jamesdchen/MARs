"""Markdown tables for REPORT.md from workers/w4/trials.jsonl (final PPO evals) and runs/*_eval.json."""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(x, d=3):
    return 'nan' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def row_from(name, variant, vals, seeds):
    pn = np.array([v['pnl_mean'] for v in vals])
    se = pn.std(ddof=1) / np.sqrt(len(pn)) if len(pn) > 1 else float('nan')
    fills = np.mean([v['fills_per_market'] for v in vals])
    t6 = [v.get('t6_share') for v in vals]
    mm = [bool(v.get('makes_markets')) for v in vals]
    lens = [np.mean([v.get(f'len/{L}/pnl_mean') or 0.0 for v in vals]) for L in ('5m', '15m', '4h')]
    t6s = '/'.join(fmt(t, 2) for t in t6)
    return {'name': name, 'variant': variant, 'n': len(vals), 'seeds': seeds, 'pnl_mean': float(pn.mean()), 'se': float(se),
            'fills': float(fills), 't6': t6s, 'mm': f'{sum(mm)}/{len(mm)}', 'mm_all': all(mm),
            'len': lens, 'per_seed': pn.tolist()}


def rl_rows():
    g = defaultdict(dict)
    for l in open(os.path.join(HERE, 'trials.jsonl')):
        r = json.loads(l)
        if r.get('source') != 'train' or not r.get('final') or not str(r.get('tag', '')).startswith('w4_rl_'):
            continue
        g[(r['tag'][6:], r['variant'])][r['seed']] = r['val']
    return [row_from(k[0], k[1], [v[s] for s in sorted(v)], sorted(v)) for k, v in sorted(g.items())]


def file_rows(fname, label):
    p = os.path.join(HERE, 'runs', fname)
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    return [row_from(label, v, x['per_seed'], [1, 2, 3]) for v, x in d.items()]


def table(rows, variant):
    out = ['| config | seeds | P&L/market (mean ± SE) | per seed | fills/mkt | T6 per seed | D1.3 pass | 5m | 15m | 4h |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted([r for r in rows if r['variant'] == variant], key=lambda r: -r['pnl_mean']):
        out.append(f"| {r['name']} | {r['n']} | {r['pnl_mean']:+.3f} ± {fmt(r['se'])} | "
                   f"{', '.join(f'{x:+.2f}' for x in r['per_seed'])} | {r['fills']:.1f} | {r['t6']} | {r['mm']} | "
                   + ' | '.join(f'{x:+.2f}' for x in r['len']) + ' |')
    return '\n'.join(out)


def main():
    rows = file_rows('q1best_eval.json', 'learned quoter (q1 best)') + file_rows('bc_eval.json', 'BC student of the quoter') \
        + rl_rows()
    for v in ('central_1s', 'pess_1s', 'central_2s'):
        print(f'\n### {v}\n')
        print(table(rows, v))
    json.dump(rows, open(os.path.join(HERE, 'results', 'summary_rows.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
