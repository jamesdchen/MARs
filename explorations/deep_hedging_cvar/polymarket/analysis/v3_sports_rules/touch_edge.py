#!/usr/bin/env python3
"""Census-style edge of maker volume at the touch, pre-game (book-free, from events.parquet).

  python touch_edge.py --events cache/events.parquet --out out/touch_edge.csv

Every taker event's volume at or beyond the touch level (ceil/floor of the micro-price, d = 0) is valued at
the touch: edge per share = spread to the pre-event micro-price + markout to mid_h (+ rebate), per $ of
notional. Horizons 30 min (capped at kickoff) and to kickoff (no flatten cost: this is the maker pool, not a
strategy). Grouped by period (sel = Sep 24 - Oct 2, check = Oct 3 - 7), day, league, time to kickoff and
market type; 95% CIs from a Poisson bootstrap over gamma events (1000).
"""
import argparse

import numpy as np
import pandas as pd

TTK = ([0, 300, 1800, 7200, 21600, 86400, np.inf], ['0-5m', '5-30m', '30m-2h', '2-6h', '6-24h', '>24h'])


def boot(g, rng, nb=1000):
    """g: frame with ev, n, p30, nk, pk -> bps and CIs."""
    e = g.groupby('ev')[['n', 'p30', 'nk', 'pk', 'sp']].sum()
    W = rng.poisson(1.0, (nb, len(e)))
    out = {'notional_k': e.n.sum() / 1e3, 'games': len(e), 'spread_bps': e.sp.sum() / e.n.sum() * 1e4}
    for p, n, nm in (('p30', 'n', '30m'), ('pk', 'nk', 'kick')):
        out[f'bps_{nm}'] = e[p].sum() / e[n].sum() * 1e4
        bs = (W @ e[p].to_numpy()) / np.maximum(W @ e[n].to_numpy(), 1e-9) * 1e4
        out[f'bps_{nm}_lo'], out[f'bps_{nm}_hi'] = np.percentile(bs, [2.5, 97.5])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='cache/events.parquet')
    ap.add_argument('--sel-end', default='2026-10-02')
    ap.add_argument('--out', default='out/touch_edge.csv')
    a = ap.parse_args()
    d = pd.read_parquet(a.events)
    d = d[(d.m >= .05) & (d.m <= .95)].copy()
    d['ev'] = d.ev.fillna(d.cond)
    q, lv = d.vol_ge_0, d.lvl_0
    d['n'] = q * np.where(d.dir > 0, 1 - lv, lv)
    reb = q * d.rebate_rate * d.fee_rate * lv * (1 - lv)
    d['sp'] = q * d.dir * (lv - d.m)
    d['p30'] = (q * d.dir * (lv - d.mid_1800) + reb).where(d.mid_1800.notna(), 0.0)
    d['n'] = d.n.where(d.mid_1800.notna(), 0.0)
    d['pk'] = (q * d.dir * (lv - d.mid_kick) + reb).where(d.mid_kick.notna(), 0.0)
    d['nk'] = (q * np.where(d.dir > 0, 1 - lv, lv)).where(d.mid_kick.notna(), 0.0)
    cut = pd.Timestamp(a.sel_end, tz='UTC').timestamp() + 86400
    d['period'] = np.where(d.t < cut, 'sel', 'check')
    d['day'] = pd.to_datetime(d.t, unit='s').dt.date.astype(str)
    d['ttk'] = pd.cut(d.ttk_s, TTK[0], labels=TTK[1], right=False).astype(str)
    rng = np.random.default_rng(0)
    rows = []
    for per in ('sel', 'check', 'all'):
        dp = d if per == 'all' else d[d.period == per]
        rows.append({'period': per, 'by': 'all', 'group': 'all', **boot(dp, rng)})
        for by in ('league', 'ttk', 'sports_type', 'day'):
            vc = dp[by].value_counts()
            for g in vc.index[:12]:
                rows.append({'period': per, 'by': by, 'group': g, **boot(dp[dp[by] == g], rng)})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option('display.width', 200)
    print(out.round(1).to_string())


if __name__ == '__main__':
    main()
