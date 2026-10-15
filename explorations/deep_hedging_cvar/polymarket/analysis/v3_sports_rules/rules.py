#!/usr/bin/env python3
"""Book-free backtest of simple pre-game quoting rules on on-chain taker events (build_events.py).

  python rules.py --events cache/events.parquet --sel-end 2026-10-02 --out out/

A rule quotes both sides of every pre-game sports condition (outcome-0 frame) around the trade micro-price:
  d     distance from the micro-price in cents (ask = ceil(m + d), bid = floor(m - d) on the 1c grid),
  S     size per side in shares (refilled after every fill: one fill per taker event at most S shares),
  T     pull time: no quotes in the last T minutes before kickoff,
  A     start: quote only when kickoff is at most A hours away (inf = from listing),
  fill  'ge': a taker event fills us if it traded at or beyond our price (we join the level with no queue,
        optimistic); 'gt': only volume strictly beyond our price fills us (queue at our price ahead of us).
For each event that reaches our level we fill q = min(S, volume at/beyond the level) at our price.
P&L per share at horizon h = dir * (level - mid_h) (+ maker rebate rebate_rate * fee_rate * p(1-p)),
with mid_h capped at kickoff. 'kick' = the whole position marked at the last pre-kickoff micro-price, minus a
flatten cost of --flat-c cents per share of each condition's net inventory: the strategy's P&L when it
must be flat at kickoff (DECISIONS D11). Edge per $ = P&L / notional (cost of the token bought).
CIs: Poisson bootstrap over gamma events (1000 replicates), as in analysis/census.
Selection uses only events with t < --sel-end + 1 day (selection period); the rest of the window is the
check period. Every window day is older than the last 7 days (data reserved for the untouched test).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIST = (0, 0.5, 1, 1.5, 2, 3, 5)
SIZES = (10, 50, 250)
PULL_MIN = (0, 1, 5, 15, 60)
START_H = (np.inf, 24, 6, 2)
FILLS = ('ge', 'gt')
TTK_BUCKETS = [(0, 300), (300, 1800), (1800, 7200), (7200, 6 * 3600), (6 * 3600, 24 * 3600), (24 * 3600, np.inf)]
TTK_NAMES = ['0-5m', '5-30m', '30m-2h', '2-6h', '6-24h', '>24h']
NB = 1000


def key(k):
    return str(k).replace('.', 'p')


class Book:
    def __init__(self, df, flat_c, seed=0):
        self.df = df
        self.ev_codes, self.ev_names = pd.factorize(df['ev'].fillna(df['cond']))
        self.cond_codes, _ = pd.factorize(df['cond'])
        self.n_ev = len(self.ev_names)
        self.flat = flat_c / 100
        rng = np.random.default_rng(seed)
        self.W = rng.poisson(1.0, (NB, self.n_ev)).astype(np.float32)
        self.dir = df['dir'].to_numpy(np.float64)
        self.ttk = df['ttk_s'].to_numpy(np.float64)
        self.reb = (df['rebate_rate'] * df['fee_rate']).to_numpy(np.float64)
        self.mid = {h: df[f'mid_{h}'].to_numpy(np.float64) for h in ('60', '300', '1800', 'kick')}
        self.cond_ev = pd.Series(self.ev_codes).groupby(self.cond_codes).first().to_numpy()

    def run(self, d, S, T, A, fill, mask=None):
        lvl = self.df[f'lvl_{key(d)}'].to_numpy(np.float64)
        vol = self.df[f'vol_{fill}_{key(d)}'].to_numpy(np.float64)
        act = (self.ttk >= T * 60) & (self.ttk <= A * 3600) & (lvl > 0) & (lvl < 1) & (vol > 0)
        if mask is not None:
            act &= mask
        q = np.where(act, np.minimum(S, vol), 0.0)
        notional = q * np.where(self.dir > 0, 1 - lvl, lvl)
        rebate = q * self.reb * lvl * (1 - lvl)
        out = {'fills': int(act.sum()), 'shares': float(q.sum()), 'notional': float(notional.sum())}
        evn = np.bincount(self.ev_codes, notional, self.n_ev)
        for h, mh in self.mid.items():
            ok = act & np.isfinite(mh)
            pnl = np.where(ok, q * self.dir * (lvl - np.nan_to_num(mh)), 0.0) + np.where(ok, rebate, 0.0)
            nt = np.where(ok, notional, 0.0)
            pe = np.bincount(self.ev_codes, pnl, self.n_ev)
            if h == 'kick':
                inv = np.bincount(self.cond_codes, np.where(ok, -self.dir * q, 0.0))
                pe = pe - np.bincount(self.cond_ev, self.flat * np.abs(inv), self.n_ev)
                out['flat_cost'] = float(self.flat * np.abs(inv).sum())
                out['max_abs_inv'] = float(np.abs(inv).max()) if len(inv) else 0.0
            ne = np.bincount(self.ev_codes, nt, self.n_ev)
            tot, ntot = pe.sum(), ne.sum()
            bs = (self.W @ pe) / np.maximum(self.W @ ne, 1e-9) * 1e4
            out[f'pnl_{h}'] = float(tot)
            out[f'bps_{h}'] = float(tot / ntot * 1e4) if ntot > 0 else float('nan')
            out[f'bps_{h}_lo'], out[f'bps_{h}_hi'] = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
            out[f'pnl_{h}_lo'], out[f'pnl_{h}_hi'] = (float(x) for x in np.percentile(self.W @ pe, [2.5, 97.5]))
        out['rebate'] = float(rebate.sum())
        out['spread'] = float((q * self.dir * (lvl - self.df['m'].to_numpy())).sum())
        return out


def grid(book, mask=None, label=''):
    rows = []
    for d, S, T, A, fl in itertools.product(DIST, SIZES, PULL_MIN, START_H, FILLS):
        r = book.run(d, S, T, A, fl, mask)
        rows.append({'d': d, 'S': S, 'T': T, 'A': A, 'fill': fl, 'subset': label, **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='cache/events.parquet')
    ap.add_argument('--sel-end', default='2026-10-02', help='last day (inclusive) of the selection period')
    ap.add_argument('--flat-c', type=float, default=0.5, help='flatten cost at kickoff, cents per share')
    ap.add_argument('--pmin', type=float, default=0.05)
    ap.add_argument('--out', default='out')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df = pd.read_parquet(a.events)
    df = df[(df.m >= a.pmin) & (df.m <= 1 - a.pmin)].reset_index(drop=True)
    df['mid_kick'] = df['mid_kick']
    df = df.rename(columns={'mid_kick': 'mid_kick'})
    cut = pd.Timestamp(a.sel_end, tz='UTC').timestamp() + 86400
    periods = {'sel': df[df.t < cut].reset_index(drop=True), 'check': df[df.t >= cut].reset_index(drop=True)}
    days = {k: (pd.to_datetime(v.t.min(), unit='s').date().isoformat(), pd.to_datetime(v.t.max(), unit='s').date().isoformat(),
                len(v), v.ev.nunique()) for k, v in periods.items()}
    print('periods', days, flush=True)
    books = {k: Book(v, a.flat_c) for k, v in periods.items()}
    # 1. global grid on both periods (selection on 'sel' only)
    g = {k: grid(b) for k, b in books.items()}
    for k, v in g.items():
        v.to_csv(out / f'grid_{k}.csv', index=False)
    sel = g['sel']
    best = sel[sel.fill == 'gt'].sort_values('pnl_kick', ascending=False).iloc[0]
    best_ge = sel[sel.fill == 'ge'].sort_values('pnl_kick', ascending=False).iloc[0]
    res = {'periods': days, 'flat_c': a.flat_c, 'pmin': a.pmin}
    rk = ['d', 'S', 'T', 'A', 'fill']
    for nm, b in (('best_gt', best), ('best_ge', best_ge)):
        rule = {k: (b[k] if k != 'A' else float(b[k])) for k in rk}
        chk = g['check']
        c = chk[(chk.d == b.d) & (chk.S == b.S) & (chk.T == b.T) & (chk.A == b.A) & (chk.fill == b.fill)].iloc[0]
        res[nm] = {'rule': rule, 'sel': b.drop(rk + ['subset']).to_dict(), 'check': c.drop(rk + ['subset']).to_dict()}
    # 2. per league and per time-to-kickoff for the selected rule; per-league selection on 'sel'
    r0 = res['best_gt']['rule']
    per = []
    for k, b in books.items():
        dfk = periods[k]
        for lg in dfk.league.value_counts().index[:12]:
            per.append({'period': k, 'by': 'league', 'group': lg,
                        **b.run(r0['d'], r0['S'], r0['T'], r0['A'], r0['fill'], (dfk.league == lg).to_numpy())})
        for (lo, hi), nmb in zip(TTK_BUCKETS, TTK_NAMES):
            per.append({'period': k, 'by': 'ttk', 'group': nmb,
                        **b.run(r0['d'], r0['S'], 0, np.inf, r0['fill'], ((dfk.ttk_s >= lo) & (dfk.ttk_s < hi)).to_numpy())})
    pd.DataFrame(per).to_csv(out / 'breakdown_selected_rule.csv', index=False)
    lg_sel = []
    for lg in periods['sel'].league.value_counts().index[:8]:
        gs = grid(books['sel'], (periods['sel'].league == lg).to_numpy(), lg)
        gs = gs[gs.fill == 'gt'].sort_values('pnl_kick', ascending=False).iloc[0]
        mc = (periods['check'].league == lg).to_numpy()
        c = books['check'].run(gs.d, gs.S, gs.T, gs.A, 'gt', mc) if mc.any() else {}
        lg_sel.append({'league': lg, 'rule': {k: (gs[k] if k != 'A' else float(gs[k])) for k in rk},
                       'sel_pnl_kick': gs.pnl_kick, 'sel_bps_kick': gs.bps_kick,
                       'sel_ci': [gs.bps_kick_lo, gs.bps_kick_hi], 'check': c})
    res['per_league_selected'] = lg_sel
    json.dump(res, open(out / 'rules_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
