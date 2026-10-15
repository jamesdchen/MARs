#!/usr/bin/env python3
"""Share of pre-game maker inventory that nets out before kickoff, and the flatten-cost drag (D15 follow-up).

  python netting.py --legs cache/legs_sep24.parquet cache/legs_oct8.parquet --out out/netting.csv

Unit: one maker wallet in one condition (outcome-0 frame, so trades on both tokens net, as a merge would).
  gross = sum of filled shares before kickoff; net = |sum of signed shares| at kickoff
  netted share = 1 - sum(net) / sum(gross)
  drag (bps of notional) = flat_c * sum(net) / sum(q * size) * 1e4, flat_c = 0.5c per share (half a 1c spread)
  edge to kickoff (C3 of RECONCILE.md, own price, micro-price in [0.05, 0.95]) and edge net of the drag.
Only conditions whose kickoff falls inside the window (mid_kick observed). Positions carried in from before the
window start are not seen (they are small for games listed < 2 weeks ahead).
Groups: league; wallet-condition gross size; wallet activity tier (by the wallet's total pre-game notional).
CIs: Poisson bootstrap over gamma events (500).
"""
import argparse

import numpy as np
import pandas as pd

FLAT = 0.005
SIZE_B = ([0, 50, 250, 1000, 5000, 25000, np.inf], ['<50', '50-250', '250-1k', '1k-5k', '5k-25k', '>25k'])
PERIODS = {'Sep24-Oct07': ('2026-09-24', '2026-10-08'), 'Oct08-14': ('2026-10-08', '2026-10-15')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs', nargs='+', required=True)
    ap.add_argument('--out', default='out/netting.csv')
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(p) for p in a.legs], ignore_index=True)
    d = d[d.mid_kick.notna() & d.mid_pre.notna() & (d.m0 >= .05) & (d.m0 <= .95)].copy()
    d['day'] = pd.to_datetime(d.t, unit='s')
    d['ev'] = d.ev.fillna(d.cond)
    d['n'] = d.q * d['size']
    d['pk'] = d.s * (d.mid_kick - d.q) * d['size'] + d.rebate_ps * d['size']
    d['sgn'] = d.s0 * d['size']
    rng = np.random.default_rng(0)
    rows = []
    for per, (lo, hi) in PERIODS.items():
        x = d[(d.day >= lo) & (d.day < hi)]
        wc = x.groupby(['wallet', 'cond'], sort=False).agg(ev=('ev', 'first'), league=('league', 'first'), gross=('size', 'sum'),
                                                          net=('sgn', 'sum'), n=('n', 'sum'), pk=('pk', 'sum')).reset_index()
        wc['net'] = wc.net.abs()
        wtot = wc.groupby('wallet').n.sum()
        tier = pd.Series(pd.qcut(wtot.rank(method='first', ascending=False), [0, 20, 200, 2000, 1e9], labels=False) if False else
                         np.select([wtot.rank(ascending=False) <= 20, wtot.rank(ascending=False) <= 200,
                                    wtot.rank(ascending=False) <= 2000], ['top20', '21-200', '201-2000'], 'rest'), index=wtot.index)
        wc['tier'] = wc.wallet.map(tier)
        wc['size_b'] = pd.cut(wc.gross, SIZE_B[0], labels=SIZE_B[1], right=False).astype(str)
        groups = [('all', 'all', np.ones(len(wc), bool))]
        groups += [('league', g, (wc.league == g).to_numpy()) for g in wc.groupby('league').n.sum().sort_values(ascending=False).index[:8]]
        groups += [('wc_gross_shares', g, (wc.size_b == g).to_numpy()) for g in SIZE_B[1]]
        groups += [('wallet_tier', g, (wc.tier == g).to_numpy()) for g in ('top20', '21-200', '201-2000', 'rest')]
        for by, g, m in groups:
            y = wc[m]
            if not len(y):
                continue
            e = y.groupby('ev')[['gross', 'net', 'n', 'pk']].sum()
            W = rng.poisson(1.0, (500, len(e)))
            edge_bs = (W @ e.pk.to_numpy()) / np.maximum(W @ e.n.to_numpy(), 1e-9) * 1e4
            drag_bs = FLAT * (W @ e.net.to_numpy()) / np.maximum(W @ e.n.to_numpy(), 1e-9) * 1e4
            net_bs = edge_bs - drag_bs
            rows.append({'period': per, 'by': by, 'group': g, 'wallet_conds': len(y), 'games': len(e),
                         'notional_k': y.n.sum() / 1e3, 'gross_k_sh': y.gross.sum() / 1e3,
                         'netted_share': 1 - y.net.sum() / y.gross.sum(),
                         'drag_bps': FLAT * y.net.sum() / y.n.sum() * 1e4,
                         'edge_kick_bps': y.pk.sum() / y.n.sum() * 1e4,
                         'edge_after_drag_bps': (y.pk.sum() - FLAT * y.net.sum()) / y.n.sum() * 1e4,
                         'after_lo': np.percentile(net_bs, 2.5), 'after_hi': np.percentile(net_bs, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    print(out.round(2).to_string())


if __name__ == '__main__':
    main()
