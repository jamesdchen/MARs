#!/usr/bin/env python3
"""Quoting-window selection with the corrected (leg-proportional) estimator (RECONCILE.md §4).

  python select_windows.py --legs cache/legs_sep24.parquet cache/legs_oct8.parquet --out out/select_windows.csv

Estimator: pre-game maker legs at their own price (census definition), restricted to the quote level relative
to the micro-price touch, with P&L to kickoff (last pre-kickoff mid) and micro-price filter [0.05, 0.95].
This is the per-$ edge of a small quoter at that level whose fills are proportional to the volume traded at
the level (random queue position). The rule simulator in rules.py instead gave a fill of min(S, volume at or
beyond the level) per taker event, which credits sweep volume at deeper levels to our price; see RECONCILE.md.
No flatten cost (pool inventory is not ours) and no queue/capacity: per-$ edge only.

Candidates: level {touch, +1, touch or +1} x start A {inf, 24, 6, 2} h x pull T {0, 5, 15, 60} min x league
{all, soccer, nfl, mlb, nhl, esports, tennis, cfb}. Selection: best point estimate on sel (Sep 24 - Oct 2) among
candidates with >= $2M notional on sel; reported with check (Oct 3 - 7) and census-window (Oct 8 - 14) values.
The D14 window (T-24h .. T-1h) and the full window are reported for every level and league, unselected.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

PERIODS = {'sel': ('2026-09-24', '2026-10-03'), 'check': ('2026-10-03', '2026-10-08'), 'oct8_14': ('2026-10-08', '2026-10-15')}
LEAGUES = ('all', 'soccer', 'nfl', 'mlb', 'nhl', 'esports', 'tennis', 'cfb')
LEVELS = {'touch': (0,), '+1': (1,), 'touch|+1': (0, 1)}
STARTS = (np.inf, 24, 6, 2)
PULLS = (0, 5, 15, 60)


def prep(d):
    d = d[(d.m0 >= .05) & (d.m0 <= .95) & d.mid_pre.notna()].copy()
    lvl0 = np.where(d.s0 < 0, np.ceil(d.m0 * 100 - 1e-9) / 100, np.floor(d.m0 * 100 + 1e-9) / 100)
    p0 = np.where(d.oi == 0, d.q, 1 - d.q)
    d['ticks'] = np.round((-d.s0) * (p0 - lvl0) * 100)
    d['ev'] = d.ev.fillna(d.cond)
    d['n'] = d.q * d['size']
    ok = d.mid_kick.notna()
    d['pk'] = np.where(ok, d.s * (d.mid_kick - d.q) * d['size'] + d.rebate_ps * d['size'], 0.0)
    d['nk'] = np.where(ok, d.n, 0.0)
    ok3 = d.mid_30cap.notna()
    d['p3'] = np.where(ok3, d.s * (d.mid_30cap - d.q) * d['size'] + d.rebate_ps * d['size'], 0.0)
    d['n3'] = np.where(ok3, d.n, 0.0)
    return d


def stat(x, rng, nb=1000):
    e = x.groupby('ev')[['pk', 'nk', 'p3', 'n3']].sum()
    if not len(e) or e.nk.sum() <= 0:
        return {'notional_k': 0.0, 'games': 0, 'bps_kick': np.nan, 'lo': np.nan, 'hi': np.nan, 'bps_30cap': np.nan}
    W = rng.poisson(1.0, (nb, len(e)))
    bs = (W @ e.pk.to_numpy()) / np.maximum(W @ e.nk.to_numpy(), 1e-9) * 1e4
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'notional_k': e.nk.sum() / 1e3, 'games': len(e), 'bps_kick': e.pk.sum() / e.nk.sum() * 1e4, 'lo': lo, 'hi': hi,
            'bps_30cap': e.p3.sum() / max(e.n3.sum(), 1e-9) * 1e4}


def mask(d, lev, A, T, lg):
    m = d.ticks.isin(LEVELS[lev]) & (d.ttk_s <= A * 3600) & (d.ttk_s >= T * 60)
    if lg != 'all':
        m &= d.league == lg
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs', nargs='+', required=True)
    ap.add_argument('--out', default='out/select_windows.csv')
    a = ap.parse_args()
    d = prep(pd.concat([pd.read_parquet(p) for p in a.legs], ignore_index=True))
    day = pd.to_datetime(d.t, unit='s')
    per = {k: d[(day >= lo) & (day < hi)] for k, (lo, hi) in PERIODS.items()}
    rng = np.random.default_rng(0)
    rows = []
    for lev, A, T, lg in itertools.product(LEVELS, STARTS, PULLS, LEAGUES):
        if T * 60 >= A * 3600:
            continue
        r = {'level': lev, 'A_h': A, 'T_min': T, 'league': lg}
        for k, x in per.items():
            s = stat(x[mask(x, lev, A, T, lg)], rng, nb=300 if k != 'sel' else 300)
            r.update({f'{k}_{c}': v for c, v in s.items()})
        rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(a.out, index=False)
    pd.set_option('display.width', 250)
    cols = ['level', 'A_h', 'T_min', 'league'] + [f'{k}_{c}' for k in PERIODS for c in ('notional_k', 'bps_kick', 'lo', 'hi')]
    elig = g[g.sel_notional_k >= 2000]
    print('== selected on sel (best point estimate, >= $2M notional), all leagues pooled vs per league')
    print(elig[elig.league == 'all'].sort_values('sel_bps_kick', ascending=False).head(5)[cols].round(0).to_string())
    print(elig.sort_values('sel_bps_kick', ascending=False).head(8)[cols].round(0).to_string())
    print('== D14 window (T-24h .. T-1h) and full window, unselected')
    x = g[((g.A_h == 24) & (g.T_min == 60)) | ((g.A_h == np.inf) & (g.T_min == 0))]
    print(x.sort_values(['league', 'level', 'A_h'])[cols].round(0).to_string())
    print('== rank correlation of sel vs check bps over eligible candidates:',
          round(elig.sel_bps_kick.rank().corr(elig.check_bps_kick.rank()), 3),
          '; sel vs oct8_14:', round(elig.sel_bps_kick.rank().corr(elig.oct8_14_bps_kick.rank()), 3))


if __name__ == '__main__':
    main()
