"""w2 analysis: length-restricted val v2 numbers per specialist run, seed aggregates, and the
'portfolio of specialists' (per length: the best specialist by val v2 seed mean, or off).

    python analyze_w2.py            # writes summary.json and prints tables

Each run's final evaluation covers all 374 v2 val markets (one ledger row, written by
train_ppo.py). Here the per-market records are filtered to the specialist's own lengths
(evaluator.filter_records: excluded markets = no-quote), and each such derived view gets one
ledger row in workers/w2/trials.jsonl (derived: length_filter), once per run.
"""
import glob
import json
import math
import os
import sys

import numpy as np

W2 = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.dirname(os.path.dirname(W2))
sys.path.insert(0, TRAIN)
import ledger  # noqa: E402
from evaluator import filter_records, nan_to_none, summarize  # noqa: E402

LEDGER = os.path.join(W2, 'trials.jsonl')
LENS = {'4h': ('4h',), '15m': ('15m',), '5m': ('5m',), '4h15m': ('15m', '4h')}
LEN_S = {'5m': 300, '15m': 900, '4h': 14400}


def load_runs():
    logged = set()
    if os.path.exists(LEDGER):
        for l in open(LEDGER):
            r = json.loads(l)
            if r.get('derived') == 'length_filter':
                logged.add(r['run_dir'])
    runs = []
    for d in sorted(glob.glob(os.path.join(W2, 'runs', '*_s*'))):
        rj, vm = os.path.join(d, 'result.json'), os.path.join(d, 'val_markets.jsonl')
        if not (os.path.exists(rj) and os.path.exists(vm)):
            continue
        res = json.load(open(rj))
        base = os.path.basename(d)
        name, seed = base.rsplit('_s', 1)
        mask = name.split('_')[0]
        recs = [json.loads(l) for l in open(vm)]
        assert len(recs) == 374, (d, len(recs))
        filt = summarize(filter_records(recs, LENS[mask]), group_key='contract_len_s')
        own = summarize([r for r in recs if any(r['contract_len_s'] == LEN_S[x] for x in LENS[mask])])
        if d not in logged:
            ledger.append({'kind': 'val_eval', 'split': 1, 'source': 'w2_analyze', 'backend': 'real',
                           'tag': f'w2_{name}', 'cfg_hash': res['cfg_hash'], 'seed': int(seed),
                           'variant': f'central_1s|quote:{"+".join(LENS[mask])}', 'final': True,
                           'derived': 'length_filter', 'quote_lengths': list(LENS[mask]),
                           'data_dir': res['cfg']['env']['kw'].get('data_dir'), 'run_dir': d,
                           'ckpt': res['ckpt'], 'val': nan_to_none(filt)}, LEDGER)
        runs.append({'name': name, 'mask': mask, 'cfg': name.split('_', 1)[1], 'seed': int(seed), 'dir': d,
                     'ckpt': res['ckpt'], 'all': res['val']['central_1s'], 'filt': filt, 'own': own,
                     'recs': recs})
    return runs


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan'))


def aggregate(runs, seeds=None):
    groups = {}
    for r in runs:
        if seeds is None or r['seed'] in seeds:
            groups.setdefault(r['name'], []).append(r)
    rows = {}
    for n, rs in groups.items():
        f = [r['filt'] for r in rs]
        o = [r['own'] for r in rs]
        row = {'name': n, 'mask': rs[0]['mask'], 'n_seeds': len(rs), 'seeds': sorted(r['seed'] for r in rs)}
        row['pnl_total'] = ms([x['pnl_total'] for x in f])
        row['own_n'] = o[0]['n_markets']
        row['own_pnl_per_mkt'] = ms([x['pnl_mean'] for x in o])
        row['own_fills_per_mkt'] = ms([x['fills_per_market'] for x in o])
        row['own_t6'] = ms([x['t6_share'] if x['t6_share'] == x['t6_share'] else 0.0 for x in o])
        row['own_t5'] = ms([x['t5_max_share'] if x['t5_max_share'] == x['t5_max_share'] else 0.0 for x in o])
        row['own_d13'] = float(np.mean([x['makes_markets'] for x in o]))
        row['all_pnl_total'] = ms([r['all']['pnl_total'] for r in rs])
        for L in ('5m', '15m', '4h'):
            row[f'len/{L}'] = ms([r['all'].get(f'len/{L}/pnl_total', 0.0) for r in rs])
            row[f'fills/{L}'] = ms([r['all'].get(f'len/{L}/fills_per_market', 0.0) for r in rs])
        rows[n] = row
    return rows


def portfolio(rows, min_seeds=2, need_d13=False):
    """Per length: the specialist (any mask covering it) with the best seed-mean val P&L on that length
    (need_d13: only specialists whose own-length view passes D1.3 on every seed)."""
    pick = {}
    for L in ('5m', '15m', '4h'):
        cands = [(r[f'len/{L}'][0], n) for n, r in rows.items() if L in LENS[r['mask']] and r['n_seeds'] >= min_seeds
                 and (not need_d13 or r['own_d13'] == 1.0)]
        best = max(cands) if cands else (float('-inf'), None)
        pick[L] = {'model': best[1] if best[0] > 0 else None, 'val_pnl': best[0] if best[0] > 0 else 0.0,
                   'best_candidate': best[1], 'best_candidate_pnl': best[0]}
    pick['total'] = sum(pick[L]['val_pnl'] for L in ('5m', '15m', '4h'))
    return pick


def portfolio_eval(runs, pick, seeds):
    """Combined per-market records for the picked models, seed by seed (seed k of each model),
    over all 374 markets: D1.3 flags and totals of the portfolio as a single policy."""
    by = {(r['name'], r['seed']): r for r in runs}
    out = []
    for s in seeds:
        recs = None
        ok = True
        for L in ('5m', '15m', '4h'):
            m = pick[L]['model']
            if m is None:
                continue
            if (m, s) not in by:
                ok = False
                break
            fr = filter_records(by[(m, s)]['recs'], (L,))
            recs = fr if recs is None else [a if a['contract_len_s'] == LEN_S[L] else b for a, b in zip(fr, recs)]
        if ok and recs is not None:
            out.append((s, summarize(recs, group_key='contract_len_s')))
    return out


def fmt(m):
    return f'{m[0]:+.2f} ± {m[1]:.2f}' if m[1] == m[1] else f'{m[0]:+.2f}'


def main():
    runs = load_runs()
    rows = aggregate(runs)
    pick = portfolio(rows)
    pick_d13 = portfolio(rows, need_d13=True)
    print('| run | seeds | own-length val P&L $ | per mkt | fills/mkt | T6 | D1.3 | 5m | 15m | 4h |')
    for n in sorted(rows, key=lambda n: -rows[n]['pnl_total'][0]):
        r = rows[n]
        print(f"| {n} | {r['n_seeds']} | {fmt(r['pnl_total'])} | {fmt(r['own_pnl_per_mkt'])} | "
              f"{r['own_fills_per_mkt'][0]:.1f} | {r['own_t6'][0]:.2f} | {r['own_d13']:.2f} | "
              f"{fmt(r['len/5m'])} | {fmt(r['len/15m'])} | {fmt(r['len/4h'])} |")
    out = {'rows': rows}
    for nm, pk in (('portfolio', pick), ('portfolio_d13', pick_d13)):
        print(nm, json.dumps(pk))
        port = portfolio_eval(runs, pk, sorted({r['seed'] for r in runs}))
        for s, p in port:
            print('  seed', s, {k: p.get(k) for k in ('pnl_total', 'fills_per_market', 't6_share',
                                                       't5_max_share', 'makes_markets', 'len/5m/pnl_total',
                                                       'len/15m/pnl_total', 'len/4h/pnl_total')})
        out[nm] = pk
        out[nm + '_by_seed'] = [{'seed': s, **nan_to_none(p)} for s, p in port]
    json.dump(out,
              open(os.path.join(W2, 'summary.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
