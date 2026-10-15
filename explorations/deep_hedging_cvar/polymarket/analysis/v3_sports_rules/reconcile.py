#!/usr/bin/env python3
"""Reconcile the census pre-game edge (+64 bps [1, 130] at 30 min, Oct 8-14) with w2's at-touch pool (RECONCILE.md).

  python reconcile.py --legs cache/legs_sep24.parquet cache/legs_oct8.parquet --out out/reconcile.csv

Per window (and per week), the ex-rewards edge per $ of maker legs (event-bootstrap 95% CI, 1000) under a ladder
of definitions, each switching one census convention to w2's:
  C0 census:  all pre-game maker legs at their own price q; 30 min mark NOT capped at kickoff (in-play mid,
              settlement if closed); notional q*size; no price filter.            [census.py edge_xr_30m]
  C1          + 30 min mark capped at kickoff (last mid before game start)
  C2          + micro-price filter [0.05, 0.95]
  C3          + to-kickoff horizon (last pre-kickoff mid) instead of 30 min
  C4          C2 with notional = cost of the token bought (w2's weighting)
  C5          C2 with every leg re-priced at the touch level of its event (ceil/floor of the micro-price, 1c grid)
  C6          C5 restricted to legs at or beyond the touch (w2's at-touch pool at 30 min)
  C7          C6 to kickoff (w2's at-touch pool to kickoff)
  C8          C2 restricted to legs at or beyond the touch, at their own price (what the restriction alone does)
Also by time to kickoff (C0 vs C1: the in-play contamination sits in legs < 30 min before kickoff) and by league.
"""
import argparse

import numpy as np
import pandas as pd

TTK = ([0, 300, 1800, 7200, 21600, 86400, np.inf], ['0-5m', '5-30m', '30m-2h', '2-6h', '6-24h', '>24h'])


def edge(d, mark, notional, rng, nb=1000, price='q'):
    ok = d.mid_pre.notna() & d[mark].notna()
    x = d[ok]
    num = x.s * (x[mark] - x[price]) * x['size'] + x.rebate_ps * x['size']
    den = notional[ok]
    e = pd.DataFrame({'ev': x.ev.fillna(x.cond), 'num': num, 'den': den,
                      'sp': x.s * (x.mid_pre - x[price]) * x['size']}).groupby('ev').sum()
    W = rng.poisson(1.0, (nb, len(e)))
    bs = (W @ e.num.to_numpy()) / np.maximum(W @ e.den.to_numpy(), 1e-9) * 1e4
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'bps': e.num.sum() / e.den.sum() * 1e4, 'lo': lo, 'hi': hi, 'spread_bps': e.sp.sum() / e.den.sum() * 1e4,
            'notional_k': e.den.sum() / 1e3, 'games': len(e), 'legs': int(ok.sum())}


def ladder(d, rng):
    nq = d.q * d['size']
    nb = np.where(d.s > 0, d.q, 1 - d.q) * d['size']
    f = (d.m0 >= .05) & (d.m0 <= .95)
    nb = pd.Series(nb, index=d.index)
    # touch level in frame 0 for the taker direction (taker dir0 = -s0), then in the leg's own frame
    lvl0 = np.where(d.s0 < 0, np.ceil(d.m0 * 100 - 1e-9) / 100, np.floor(d.m0 * 100 + 1e-9) / 100)
    p0 = np.where(d.oi == 0, d.q, 1 - d.q)
    d = d.assign(q_touch=np.where(d.oi == 0, lvl0, 1 - lvl0),
                 beyond=(-d.s0) * (p0 - lvl0) >= -1e-9)
    nqt = d.q_touch * d['size']
    g = f & d.beyond
    return {'C0 census': edge(d, 'mid_30', nq, rng),
            'C1 +cap at kickoff': edge(d, 'mid_30cap', nq, rng),
            'C2 +price filter': edge(d[f], 'mid_30cap', nq[f], rng),
            'C3 +to kickoff': edge(d[f], 'mid_kick', nq[f], rng),
            'C4 C2, w2 notional': edge(d[f], 'mid_30cap', nb[f], rng),
            'C5 C2, touch price': edge(d[f], 'mid_30cap', nqt[f], rng, price='q_touch'),
            'C6 C5, legs beyond touch': edge(d[g], 'mid_30cap', nqt[g], rng, price='q_touch'),
            'C7 C6 to kickoff': edge(d[g], 'mid_kick', nqt[g], rng, price='q_touch'),
            'C8 C2, legs beyond touch, own price': edge(d[g], 'mid_30cap', nq[g], rng)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs', nargs='+', required=True)
    ap.add_argument('--out', default='out/reconcile.csv')
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(p) for p in a.legs], ignore_index=True)
    d['day'] = pd.to_datetime(d.t, unit='s')
    wk = {'Sep24-30': ('2026-09-24', '2026-10-01'), 'Oct01-07': ('2026-10-01', '2026-10-08'),
          'Sep24-Oct07': ('2026-09-24', '2026-10-08'), 'Oct08-14 (census)': ('2026-10-08', '2026-10-15')}
    rng = np.random.default_rng(0)
    rows = []
    for w, (lo, hi) in wk.items():
        dw = d[(d.day >= lo) & (d.day < hi)]
        if not len(dw):
            continue
        for k, v in ladder(dw.reset_index(drop=True), rng).items():
            rows.append({'window': w, 'by': 'all', 'group': 'all', 'def': k, **v})
        dw = dw.assign(ttk=pd.cut(dw.ttk_s, TTK[0], labels=TTK[1], right=False).astype(str))
        # per price level relative to the micro-price touch (legs at their own price; C2 filter): the per-$ edge
        # of a maker whose fills are proportional to the volume traded at its level
        f = ((dw.m0 >= .05) & (dw.m0 <= .95)).to_numpy()
        lvl0 = np.where(dw.s0 < 0, np.ceil(dw.m0 * 100 - 1e-9) / 100, np.floor(dw.m0 * 100 + 1e-9) / 100)
        p0 = np.where(dw.oi == 0, dw.q, 1 - dw.q)
        ticks = np.round((-dw.s0) * (p0 - lvl0) * 100)
        lv = pd.cut(ticks, [-100, -0.5, 0.5, 1.5, 2.5, 100], labels=['inside', 'touch', '+1', '+2', '+3 or more']).astype(str)
        nq0 = dw.q * dw['size']
        for g in ('inside', 'touch', '+1', '+2', '+3 or more'):
            m = f & (lv == g)
            for mark, nm in (('mid_30cap', 'level, 30 min capped'), ('mid_kick', 'level, to kickoff')):
                rows.append({'window': w, 'by': 'level', 'group': g, 'def': nm, **edge(dw[m], mark, nq0[m], rng)})
        for g in TTK[1]:
            m = f & (lv == 'touch') & (dw.ttk == g).to_numpy()
            rows.append({'window': w, 'by': 'touch_ttk', 'group': g, 'def': 'level, to kickoff', **edge(dw[m], 'mid_kick', nq0[m], rng)})
        for g in dw.league.value_counts().index[:6]:
            m = f & (lv == 'touch') & (dw.league == g).to_numpy()
            rows.append({'window': w, 'by': 'touch_league', 'group': g, 'def': 'level, to kickoff', **edge(dw[m], 'mid_kick', nq0[m], rng)})
        nq = dw.q * dw['size']
        for g in TTK[1]:
            m = dw.ttk == g
            for mark, nm in (('mid_30', 'C0 census'), ('mid_30cap', 'C1 +cap at kickoff')):
                rows.append({'window': w, 'by': 'ttk', 'group': g, 'def': nm, **edge(dw[m], mark, nq[m], rng)})
        for g in dw.league.value_counts().index[:6]:
            m = dw.league == g
            for mark, nm in (('mid_30', 'C0 census'), ('mid_30cap', 'C1 +cap at kickoff')):
                rows.append({'window': w, 'by': 'league', 'group': g, 'def': nm, **edge(dw[m], mark, nq[m], rng)})
        print(w, 'done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    print(out.round(1).to_string())


if __name__ == '__main__':
    main()
