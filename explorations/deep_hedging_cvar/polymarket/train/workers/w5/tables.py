"""w5: markdown tables for REPORT.md (teachers / BC students, every config, top-3 detail)."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load, table, agg, f

W5 = os.path.dirname(os.path.abspath(__file__))


def ref_table():
    d = json.load(open(os.path.join(W5, 'teachers.json')))
    hdr = ['policy', '$/mkt', 'total', 'fills/mkt', 'T6', 'D1.3', '5m total', '15m total', '4h total']
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for k, m in d.items():
        out.append('| ' + ' | '.join([k, f(m['pnl_mean'], 3), f(m['pnl_total'], 1), f"{m['fills_per_market']:.2f}",
                   f(m['t6_share']), 'pass' if m['makes_markets'] else 'fail', f(m['len/5m/pnl_total'], 1),
                   f(m['len/15m/pnl_total'], 1), f(m['len/4h/pnl_total'], 1)]) + ' |')
    return '\n'.join(out)


def detail(names):
    plan, by = load()
    hdr = ['config', 'seeds', '$/mkt ± SE', 'per-seed $/mkt', 'fills/mkt', '2-sided uptime', 'T6', 'T5',
           '5m $/mkt (fills)', '15m $/mkt (fills)', '4h $/mkt (fills)', 'spread', 'markout60', 'inventory', 'rebates']
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for n in names:
        a = agg(by[n])
        out.append('| ' + ' | '.join([n, ','.join(map(str, a['seeds'])), f"{f(a['pnl_mean'], 3)} ± {a['pnl_se']:.3f}",
                   ' '.join(f'{x:+.3f}' for x in a['seed_pnl_mean']), f"{a['fills']:.1f}", f"{a['uptime2']:.2f}",
                   f(a['t6']), f(a['t5']),
                   *[f"{f(a[f'len/{L}/pnl_mean'], 3)} ({a[f'len/{L}/fills']:.1f})" for L in ('5m', '15m', '4h')],
                   f(a['sum/spread_capture'], 1), f(a['sum/markout_60'], 1), f(a['sum/inventory_pnl'], 1),
                   f(a['sum/rebates'], 1)]) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    print(ref_table())
    print()
    plan, by = load()
    print(table(plan, by, 3)[0])
