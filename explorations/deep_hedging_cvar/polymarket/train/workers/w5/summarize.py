"""w5: seed-mean table of every config (val v2, central fills, 1 s) from runs/*/result.json.

    python summarize.py [--md] [--min-seeds 3]

D1.3 on the seed mean: fills/market >= 5, T6 = mean inventory P&L / mean total <= 0.5 when the
mean total is > 0 (n/a otherwise), worst seed T5 artifact share <= 0.25 (NaN = no positive fill edge).
"""
import argparse, glob, json, math, os
import numpy as np

W5 = os.path.dirname(os.path.abspath(__file__))


def load():
    plan = {c['name']: c for c in json.load(open(os.path.join(W5, 'plan.json')))}
    by = {}
    for p in sorted(glob.glob(os.path.join(W5, 'runs', '*_s*', 'result.json'))):
        d = os.path.basename(os.path.dirname(p))
        if '.killed' in d:
            continue
        name, seed = d.rsplit('_s', 1)
        r = json.load(open(p))
        v = r['val']['central_1s']
        assert v['n_markets'] == 374, (p, v['n_markets'])
        by.setdefault(name, {})[int(seed)] = {'v': v, 'p': r['val'].get('pess_1s', {}), 'ckpt': r['ckpt'],
                                              'sha': r['ckpt_sha256'], 'cfg_hash': r['cfg_hash']}
    return plan, by


def agg(seeds):
    S = sorted(seeds)
    V = [seeds[s]['v'] for s in S]
    pm = np.array([v['pnl_mean'] for v in V])
    tot = np.array([v['pnl_total'] for v in V])
    inv = np.mean([v['sum/inventory_pnl'] for v in V])
    fills = float(np.mean([v['fills_per_market'] for v in V]))
    t6 = float(inv / tot.mean()) if tot.mean() > 0 else float('nan')
    t5s = [v['t5_max_share'] for v in V if v.get('t5_max_share') is not None and math.isfinite(v['t5_max_share'])]
    t5 = max(t5s) if t5s else float('nan')
    d13 = fills >= 5 and (not math.isfinite(t6) or t6 <= 0.5) and (not math.isfinite(t5) or t5 <= 0.25)
    se = float(pm.std(ddof=1) / math.sqrt(len(pm))) if len(pm) > 1 else float('nan')
    out = {'n': len(S), 'seeds': S, 'pnl_mean': float(pm.mean()), 'pnl_se': se, 'pnl_total': float(tot.mean()),
           'pnl_total_se': se * 374, 'fills': fills, 't6': t6, 't5': t5, 'd13': d13,
           'n_mm_seeds': int(sum(bool(v['makes_markets']) for v in V)),
           'uptime2': float(np.mean([v['mean/uptime_two_sided'] for v in V])),
           'pess_pnl_mean': float(np.mean([seeds[s]['p'].get('pnl_mean', float('nan')) for s in S])),
           'seed_pnl_mean': [float(x) for x in pm]}
    for L in ('5m', '15m', '4h'):
        out[f'len/{L}/pnl_total'] = float(np.mean([v[f'len/{L}/pnl_total'] for v in V]))
        out[f'len/{L}/pnl_mean'] = float(np.mean([v[f'len/{L}/pnl_mean'] for v in V]))
        out[f'len/{L}/fills'] = float(np.mean([v[f'len/{L}/fills_per_market'] for v in V]))
    for k in ('sum/spread_capture', 'sum/markout_60', 'sum/inventory_pnl', 'sum/rebates'):
        out[k] = float(np.mean([v[k] for v in V]))
    return out


def f(x, d=2):
    return '-' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:+.{d}f}'


def table(plan, by, min_seeds=1):
    rows = [(n, agg(s)) for n, s in by.items() if len(s) >= min_seeds]
    rows.sort(key=lambda r: (not r[1]['d13'], -r[1]['pnl_mean']))
    hdr = ['config', 'teacher', 'H', 'kl', 'decay', 'inv_pen', 'off', 'lr', 'ent', 'seeds', '$/mkt (±SE)', 'total',
           'fills/mkt', 'T6', 'T5', 'D1.3', '2-sided', '5m $/mkt', '15m $/mkt', '4h $/mkt', 'pess $/mkt']
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for n, a in rows:
        c = plan.get(n, {})
        out.append('| ' + ' | '.join([n, c.get('teacher', '?'), str(c.get('markout_h', '?')), f"{c.get('kl_coef', 0):.3f}",
                   f"{c.get('kl_decay', 0):g}", f"{c.get('inv_penalty', 0):.1e}", f"{c.get('off_bias', 0):g}",
                   f"{c.get('lr', 0):.1e}", f"{c.get('ent_coef', 0):.1e}", str(a['n']),
                   f"{f(a['pnl_mean'], 3)} ± {a['pnl_se']:.3f}", f(a['pnl_total'], 1), f"{a['fills']:.1f}",
                   f(a['t6']), f(a['t5']), 'pass' if a['d13'] else 'fail', f"{a['uptime2']:.2f}",
                   f(a['len/5m/pnl_mean'], 3), f(a['len/15m/pnl_mean'], 3), f(a['len/4h/pnl_mean'], 3),
                   f(a['pess_pnl_mean'], 3)]) + ' |')
    return '\n'.join(out), rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--min-seeds', type=int, default=1)
    p.add_argument('--json', default=None)
    a = p.parse_args()
    plan, by = load()
    t, rows = table(plan, by, a.min_seeds)
    print(t)
    if a.json:
        json.dump({n: r for n, r in rows}, open(a.json, 'w'), indent=1)
