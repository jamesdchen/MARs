"""Tables for REPORT.md from configs.jsonl + each run's val_markets.jsonl (val v2, central 1 s).

    python make_report.py > runs/tables.md
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUPLICATES = {'t05'}  # same params as t04 (two sweep processes shared a TPE seed); ledger rows kept
LENS = ((300, '5m'), (900, '15m'), (14400, '4h'))


def ms(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    return x.mean(), se


def per_seed(name, seed):
    recs = [json.loads(l) for l in open(os.path.join(HERE, 'runs', name, f's{seed}', 'val_markets.jsonl'))]
    pnl = np.array([r['pnl'] for r in recs])
    traded = [r for r in recs if r['fills'] > 0]
    out = {'pnl_mkt': pnl.mean(), 'fills': np.mean([r['fills'] for r in recs]),
           'cap_share': np.mean([r['max_abs_inv'] >= 99 for r in traded]) if traded else 0.0,
           'final_inv': np.mean([r['final_abs_inv'] for r in traded]) if traded else 0.0,
           'spread': sum(r['spread_capture'] for r in recs) / len(recs),
           'mk60': sum(r['markout_60'] for r in recs) / len(recs),
           'invpnl': sum(r['inventory_pnl'] for r in recs) / len(recs),
           'uptime2': np.mean([r['uptime_two_sided'] for r in recs])}
    for L, n in LENS:
        out[f'{n}'] = np.mean([r['pnl'] for r in recs if r['contract_len_s'] == L])
    return out


def load():
    rows = {}
    for line in open(os.path.join(HERE, 'configs.jsonl')):
        r = json.loads(line)
        rows[r['name']] = r
    return rows


def fmt_cfg(c):
    s = f"λ={c['inv_penalty']:.1e} H={c['markout_h']} mark={c['mark_mode']} lr={c['lr']:.1e} ent={c['ent_coef']:.1e}"
    if c.get('close_inv_coef'):
        s += f" close={c['close_inv_coef']:.1e}/{c['close_ramp']}"
    return s


def main():
    rows = load()
    print('| config | seeds | $/market (± SE) | 5m | 15m | 4h | fills/mkt | T6 | T5 | D1.3 | at-cap share | '
          'spread/mkt | markout60/mkt | inv P&L/mkt | 2-sided uptime | params |')
    print('|' + '---|' * 16)
    for r in sorted(rows.values(), key=lambda r: -r['agg']['pnl_total']):
        if r['name'] in DUPLICATES:
            continue
        ps = [per_seed(r['name'], s) for s in r['seeds']]
        g = r['agg']
        m, se = ms([p['pnl_mkt'] for p in ps])
        cell = lambda k: f"{ms([p[k] for p in ps])[0]:+.2f}"  # noqa: E731
        t5 = g.get('t5_max_share')
        print(f"| {r['name']} | {len(ps)} | {m:+.3f} ± {se:.3f} | {cell('5m')} | {cell('15m')} | {cell('4h')} | "
              f"{g['fills_per_market']:.1f} | {g['t6_share']:.2f} | {'-' if t5 is None else f'{t5:.2f}'} | "
              f"{'pass' if g['d13'] else 'FAIL'} | {ms([p['cap_share'] for p in ps])[0]:.2f} | "
              f"{cell('spread')} | {cell('mk60')} | {cell('invpnl')} | {ms([p['uptime2'] for p in ps])[0]:.2f} | "
              f"{fmt_cfg(r['cfg'])} |")


if __name__ == '__main__':
    main()
