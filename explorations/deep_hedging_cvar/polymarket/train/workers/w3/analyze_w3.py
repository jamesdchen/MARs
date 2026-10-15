"""Aggregate w3 runs: seed means per config + training diagnostics.

    python train/workers/w3/analyze_w3.py [--md out.md] [--json out.json]

Reads runs/<config>_s<seed>/{result.json,log.jsonl,val_markets.jsonl}. Per config:
val v2 central_1s P&L per market (seed mean +- SE), total, fills/market, T6, T5,
D1.3 (on the seed-mean of the flags and per seed), per-length P&L per market,
pessimistic P&L, and training diagnostics averaged over the last 20% of epochs
(entropy per head, approx KL, clipfrac, explained variance) plus the train-val gap.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEN = {300: '5m', 900: '15m', 14400: '4h'}


def se(x):
    x = [v for v in x if v is not None and math.isfinite(v)]
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan')


def mean(x):
    x = [v for v in x if v is not None and math.isfinite(v)]
    return float(np.mean(x)) if x else float('nan')


def run_stats(d):
    r = json.load(open(os.path.join(d, 'result.json')))
    v = r['val']['central_1s']
    out = {'pnl_mean': v['pnl_mean'], 'pnl_total': v['pnl_total'], 'fills': v.get('fills_per_market'),
           't6': v.get('t6_share'), 't5': v.get('t5_max_share'), 'mm': v.get('makes_markets'),
           'n_markets': v.get('n_markets', v.get('n')), 'cvar95': v.get('cvar95'),
           'pess_pnl_mean': r['val'].get('pess_1s', {}).get('pnl_mean'),
           'train_pnl_mean': (r.get('train_eval') or {}).get('pnl_mean'), 'wall_s': r['wall_s'],
           'sps': r['sps_median'], 'ckpt': r['ckpt'], 'cfg_hash': r['cfg_hash']}
    if out['train_pnl_mean'] is not None:
        out['gap'] = out['train_pnl_mean'] - out['pnl_mean']
    recs = [json.loads(l) for l in open(os.path.join(d, 'val_markets.jsonl'))]
    out['n_markets'] = len(recs)
    by = defaultdict(list)
    for m in recs:
        by[LEN.get(m['contract_len_s'], str(m['contract_len_s']))].append(m['pnl'])
    for k, xs in by.items():
        out[f'pnl_{k}'] = float(np.mean(xs))
    rows = [json.loads(l) for l in open(os.path.join(d, 'log.jsonl')) if '"type": "train"' in l]
    tail = rows[int(len(rows) * 0.8):] or rows
    for k in ('losses/entropy_head0', 'losses/entropy_head1', 'losses/entropy_head2', 'losses/approx_kl',
              'losses/clipfrac', 'losses/explained_variance', 'losses/value_loss'):
        out[k.split('/')[1]] = mean([t.get(k) for t in tail])
    return out


def collect(pattern='*_s*'):
    C = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(HERE, 'runs', pattern))):
        m = re.match(r'(.+)_s(\d+)$', os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, 'result.json')):
            continue
        C[m.group(1)][int(m.group(2))] = run_stats(d)
    return C


def agg(seeds):
    S = list(seeds.values())
    a = {'n': len(S), 'seeds': sorted(seeds)}
    for k in S[0]:
        if isinstance(S[0][k], (int, float)) and not isinstance(S[0][k], bool):
            a[k] = mean([s.get(k) for s in S])
    a['pnl_mean_se'] = se([s['pnl_mean'] for s in S])
    a['pnl_min'] = min(s['pnl_mean'] for s in S)
    a['pnl_max'] = max(s['pnl_mean'] for s in S)
    a['mm_seeds'] = sum(1 for s in S if s.get('mm'))
    t6 = a.get('t6') if a['pnl_mean'] > 0 else 0.0
    t6 = 0.0 if t6 is None or not math.isfinite(t6) else t6
    t5 = a.get('t5')
    t5 = 0.0 if t5 is None or not math.isfinite(t5) else t5
    a['d13'] = bool(a['fills'] >= 5 and t6 <= 0.5 and t5 <= 0.25)
    return a


def f(x, n=2):
    return '—' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{n}f}'


def table(A):
    hdr = ('| config | n | P&L/mkt (mean ± SE) | min / max seed | total | fills/mkt | T6 | T5 | D1.3 (agg; seeds) '
           '| 5m | 15m | 4h | pess | ent h0/h1/h2 | KL | EV | gap (train−val) | SPS |')
    lines = [hdr, '|' + '---|' * hdr.count('|')[:-1] if False else '|' + '---|' * (hdr.count('|') - 1)]
    for k, a in sorted(A.items(), key=lambda kv: -kv[1]['pnl_mean']):
        lines.append(
            f"| {k} | {a['n']} | {f(a['pnl_mean'], 3)} ± {f(a['pnl_mean_se'], 3)} | {f(a['pnl_min'])} / {f(a['pnl_max'])} "
            f"| {f(a['pnl_total'], 0)} | {f(a['fills'], 1)} | {f(a.get('t6'))} | {f(a.get('t5'))} "
            f"| {'pass' if a['d13'] else 'FAIL'}; {a['mm_seeds']}/{a['n']} | {f(a.get('pnl_5m'))} | {f(a.get('pnl_15m'))} "
            f"| {f(a.get('pnl_4h'))} | {f(a.get('pess_pnl_mean'))} "
            f"| {f(a.get('entropy_head0'))}/{f(a.get('entropy_head1'))}/{f(a.get('entropy_head2'))} "
            f"| {f(a.get('approx_kl'), 4)} | {f(a.get('explained_variance'))} | {f(a.get('gap'))} | {f(a.get('sps', 0) / 1e3, 0)}k |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--md')
    p.add_argument('--json')
    p.add_argument('--min-seeds', type=int, default=1)
    a = p.parse_args()
    C = collect()
    A = {k: agg(v) for k, v in C.items() if len(v) >= a.min_seeds}
    t = table(A)
    print(t)
    if a.md:
        open(a.md, 'w').write(t + '\n')
    if a.json:
        json.dump({'configs': A, 'runs': C}, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
