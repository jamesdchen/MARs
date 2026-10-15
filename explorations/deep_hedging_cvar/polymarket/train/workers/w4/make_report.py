"""Fill REPORT_template.md's placeholders from the ledger -> REPORT.md."""
import json
import os

import summarize as S

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = S.file_rows('q1best_eval.json', 'learned quoter (q1 best)') + S.file_rows('bc_eval.json', 'BC student of the quoter') \
        + S.rl_rows()
    t = open(os.path.join(HERE, 'REPORT_template.md')).read()
    short = [r for r in rows if '100M' not in r['name']]
    long_ = [r for r in rows if '100M' in r['name']]
    t = t.replace('RL_TABLE_CENTRAL', S.table(short, 'central_1s')).replace('RL_TABLE_PESS', S.table(short, 'pess_1s'))
    t = t.replace('RL_TABLE_2S', S.table(short, 'central_2s'))
    if long_:
        lt = ['### Round 2: c2 and c8 at 100M steps (3 seeds)', '']
        for v in ('central_1s', 'pess_1s', 'central_2s'):
            lt += [f'{v}:', '', S.table(long_, v), '']
        t = t.replace('LONG_RUNS', '\n'.join(lt))
    else:
        t = t.replace('LONG_RUNS', '')
    n = fpos = inv = passpos = 0
    ex = []
    for l in open(os.path.join(HERE, 'trials.jsonl')):
        r = json.loads(l)
        v = r['val']
        if v.get('sum/spread_capture') is None:
            continue
        F = v['sum/spread_capture'] + v['sum/markout_60'] + v['sum/rebates'] + (v.get('sum/liq_rewards') or 0)
        n += 1
        inv += v['sum/inventory_pnl'] > 0
        passpos += bool(v.get('makes_markets')) and v['pnl_total'] > 0
        if F > 0:
            fpos += 1
            ex.append(f"{r['tag']} {r.get('name')}, {v['fills_per_market']:.1f} fills/market")
    t = t.replace('FPOS_COUNT', f'F > 0 in {fpos} of {n} evaluations ({"; ".join(ex)})')
    t = t.replace('INVPOS_COUNT', f'{inv} of {n}').replace('PASSPOS_COUNT', f'{passpos} of {n}')
    t = t.replace('LEDGER_COUNT', f'Ledger: {n} val v2 evaluations in `trials.jsonl` (all of them, including smoke and '
                  'killed-trial rows).')
    open(os.path.join(HERE, 'REPORT.md'), 'w').write(t)


if __name__ == '__main__':
    main()
