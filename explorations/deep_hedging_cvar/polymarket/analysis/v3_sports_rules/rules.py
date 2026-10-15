#!/usr/bin/env python3
"""Book-free backtest of simple pre-game quoting rules on on-chain taker events (build_events.py).

  python rules.py --events cache/events.parquet --sel-end 2026-10-02 --out out

A rule quotes both sides of every pre-game sports condition (outcome-0 frame) around the trade micro-price m:
  d     distance in cents: ask = ceil(m + d), bid = floor(m - d) on the 1c grid,
  S     size per side in shares, re-posted after every fill (one fill per taker event, at most S shares),
  Q     inventory cap per condition (shares): the side that would take |inventory| above Q is cut,
  T     pull time: no quotes in the last T minutes before kickoff,
  A     start: quote only when kickoff is at most A hours away (inf = from listing),
  age   pull while the last print of the condition is older than `age` seconds (stale micro-price),
  fill  'gt': only volume that traded strictly beyond our price fills us (the queue at our price is ahead
        of us; conservative); 'ge': volume at our price fills us too (front of queue; optimistic).
A taker event that reaches our level fills q = min(S, volume at/beyond the level, cap room) at our price.
P&L per share at horizon h = dir * (level - mid_h) + maker rebate (rebate_rate * fee_rate * p(1-p)), mid_h
capped at kickoff. 'kick' = the position marked at the last pre-kickoff micro-price minus --flat-c cents per
share of each condition's net inventory at kickoff (flat at kickoff, DECISIONS D11). This is the strategy P&L.
Edge per $ = P&L / notional (cost of the token bought). CIs: Poisson bootstrap over gamma events (1000).
Selection uses only the selection period (t < --sel-end + 1 day); the check period is the rest of the window.
The whole window is older than the last 7 days, which are reserved for the untouched test.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

DIST = (0, 0.5, 1, 2, 3)
SIZES = (10, 50, 250)
CAPS = (100, 500, 2500)
PULL_MIN = (0, 5, 15, 60)
START_H = (np.inf, 24, 6)
AGES = (30, 300, np.inf)
FILLS = ('gt', 'ge')
HZ = ('60', '300', '1800', 'kick')
TTK_BUCKETS = [(0, 300), (300, 1800), (1800, 7200), (7200, 6 * 3600), (6 * 3600, 24 * 3600), (24 * 3600, np.inf)]
TTK_NAMES = ['0-5m', '5-30m', '30m-2h', '2-6h', '6-24h', '>24h']
RK = ['d', 'S', 'Q', 'T', 'A', 'age', 'fill']
NB = 1000


def key(k):
    return f'{float(k):g}'.replace('.', 'p')


@numba.njit(cache=True)
def simulate(cond, dirn, act, vol, S, Q):
    """Sequential fills with an inventory cap; events sorted by (cond, t). Returns shares filled (>= 0)."""
    n = len(cond)
    q = np.zeros(n)
    inv = 0.0
    prev = -1
    for i in range(n):
        if cond[i] != prev:
            inv = 0.0
            prev = cond[i]
        if not act[i]:
            continue
        room = Q + inv if dirn[i] > 0 else Q - inv   # taker buys -> we sell (inv falls)
        x = min(S, vol[i], room)
        if x > 0:
            q[i] = x
            inv -= dirn[i] * x
    return q


class Book:
    def __init__(self, df, flat_c, seed=0):
        df = df.sort_values(['cond', 't', 'block']).reset_index(drop=True)
        self.df = df
        self.ev_codes, self.ev_names = pd.factorize(df['ev'].fillna(df['cond']))
        self.cond_codes, _ = pd.factorize(df['cond'])
        self.n_ev, self.n_cond = len(self.ev_names), int(self.cond_codes.max()) + 1
        self.cond_ev = pd.Series(self.ev_codes).groupby(self.cond_codes).first().to_numpy()
        self.flat = flat_c / 100
        self.W = np.random.default_rng(seed).poisson(1.0, (NB, self.n_ev)).astype(np.float64)
        self.dir = df['dir'].to_numpy(np.float64)
        self.ttk = df['ttk_s'].to_numpy(np.float64)
        self.age = df['pre_age'].to_numpy(np.float64)
        self.m = df['m'].to_numpy(np.float64)
        self.reb = (df['rebate_rate'] * df['fee_rate']).to_numpy(np.float64)
        self.mid = {h: df[f'mid_{h}'].to_numpy(np.float64) for h in HZ}
        self.lvl = {key(d): df[f'lvl_{key(d)}'].to_numpy(np.float64) for d in DIST}
        self.vol = {(f, key(d)): df[f'vol_{f}_{key(d)}'].to_numpy(np.float64) for f in FILLS for d in DIST}

    def run(self, d, S, Q, T, A, age, fill, mask=None, ci=False):
        lvl, vol = self.lvl[key(d)], self.vol[(fill, key(d))]
        act = (self.ttk >= T * 60) & (self.ttk <= A * 3600) & (self.age <= age) & (lvl > 0) & (lvl < 1) & (vol > 0)
        if mask is not None:
            act &= mask
        q = simulate(self.cond_codes, self.dir, act, vol, float(S), float(Q))
        notional = q * np.where(self.dir > 0, 1 - lvl, lvl)
        rebate = q * self.reb * lvl * (1 - lvl)
        out = {'fills': int((q > 0).sum()), 'shares': float(q.sum()), 'notional': float(notional.sum()),
               'spread': float((q * self.dir * (lvl - self.m)).sum()), 'rebate': float(rebate.sum()),
               'conds': int(len(np.unique(self.cond_codes[q > 0])))}
        for h in HZ:
            mh = self.mid[h]
            ok = (q > 0) & np.isfinite(mh)
            pnl = np.where(ok, q * self.dir * (lvl - np.where(ok, mh, 0.0)) + rebate, 0.0)
            nt = np.where(ok, notional, 0.0)
            if h == 'kick':
                inv = np.bincount(self.cond_codes, np.where(ok, -self.dir * q, 0.0), self.n_cond)
                fc = self.flat * np.abs(inv)
                out['flat_cost'] = float(fc.sum())
                out['inv_abs_mean'] = float(np.abs(inv[inv != 0]).mean()) if (inv != 0).any() else 0.0
            tot, ntot = float(pnl.sum()) - (out['flat_cost'] if h == 'kick' else 0.0), float(nt.sum())
            out[f'pnl_{h}'] = tot
            out[f'bps_{h}'] = tot / ntot * 1e4 if ntot > 0 else float('nan')
            if ci:
                pe = np.bincount(self.ev_codes, pnl, self.n_ev)
                if h == 'kick':
                    pe -= np.bincount(self.cond_ev, fc, self.n_ev)
                ne = np.bincount(self.ev_codes, nt, self.n_ev)
                bs = (self.W @ pe) / np.maximum(self.W @ ne, 1e-9) * 1e4
                out[f'bps_{h}_ci'] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
                out[f'pnl_{h}_ci'] = [float(x) for x in np.percentile(self.W @ pe, [2.5, 97.5])]
        return out


def grid(book, mask=None, fills=FILLS):
    rows = []
    for d, S, Q, T, A, age, fl in itertools.product(DIST, SIZES, CAPS, PULL_MIN, START_H, AGES, fills):
        if Q < S:
            continue
        rows.append({'d': d, 'S': S, 'Q': Q, 'T': T, 'A': A, 'age': age, 'fill': fl,
                     **book.run(d, S, Q, T, A, age, fl, mask)})
    return pd.DataFrame(rows)


def rule_of(row):
    return {k: (row[k] if k == 'fill' else float(row[k])) for k in RK}


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
    cut = pd.Timestamp(a.sel_end, tz='UTC').timestamp() + 86400
    periods = {'sel': df[df.t < cut].reset_index(drop=True), 'check': df[df.t >= cut].reset_index(drop=True)}
    info = {k: {'first': pd.to_datetime(v.t.min(), unit='s').isoformat(), 'last': pd.to_datetime(v.t.max(), unit='s').isoformat(),
                'events': len(v), 'conds': int(v.cond.nunique()), 'games': int(v.ev.nunique()),
                'days': int(np.ceil((v.t.max() - v.t.min()) / 86400))} for k, v in periods.items()}
    print('periods', info, flush=True)
    books = {k: Book(v, a.flat_c) for k, v in periods.items()}
    for k, b in books.items():
        periods[k] = b.df   # sorted copy, aligned with the book's arrays
    res = {'periods': info, 'flat_c': a.flat_c, 'pmin': a.pmin}

    # 1. global grid on both periods; selection on 'sel' only
    g = {k: grid(b) for k, b in books.items()}
    for k, v in g.items():
        v.to_csv(out / f'grid_{k}.csv', index=False)
    m = g['sel'].merge(g['check'], on=RK, suffixes=('_sel', '_chk'))
    for fl in FILLS:
        mm = m[m.fill == fl]
        res[f'grid_rank_corr_{fl}'] = float(mm['pnl_kick_sel'].rank().corr(mm['pnl_kick_chk'].rank()))
        top = mm.sort_values('pnl_kick_sel', ascending=False).head(10)
        res[f'top10_{fl}'] = top[RK + ['fills_sel', 'pnl_kick_sel', 'bps_kick_sel', 'pnl_kick_chk', 'bps_kick_chk',
                                      'bps_1800_sel', 'bps_1800_chk']].to_dict('records')
        best = rule_of(top.iloc[0])
        r = {'rule': best}
        for k, b in books.items():
            r[k] = b.run(**best, ci=True)
        res[f'selected_{fl}'] = r
        print(fl, json.dumps(r, default=float)[:1500], flush=True)
    # reference rules (not selected): join the touch, every event, both fill models
    res['reference_join'] = {fl: {k: b.run(0, 50, 500, 0, np.inf, np.inf, fl, ci=True) for k, b in books.items()}
                             for fl in FILLS}

    # 2. breakdowns of the selected (gt) rule: by league and by time to kickoff
    r0 = res['selected_gt']['rule']
    per = []
    for k, b in books.items():
        dfk = periods[k]
        for lg in dfk.league.value_counts().index[:12]:
            per.append({'period': k, 'by': 'league', 'group': lg, **b.run(**r0, mask=(dfk.league == lg).to_numpy(), ci=True)})
        for st in dfk.sports_type.value_counts().index[:6]:
            per.append({'period': k, 'by': 'market_type', 'group': st,
                        **b.run(**r0, mask=(dfk.sports_type == st).to_numpy(), ci=True)})
        for (lo, hi), nmb in zip(TTK_BUCKETS, TTK_NAMES):
            rr = dict(r0, T=0, A=np.inf)
            per.append({'period': k, 'by': 'ttk', 'group': nmb,
                        **b.run(**rr, mask=((dfk.ttk_s >= lo) & (dfk.ttk_s < hi)).to_numpy(), ci=True)})
    pd.DataFrame(per).to_json(out / 'breakdown_selected_rule.json', orient='records', indent=1)

    # 3. per-league rule selection on 'sel', evaluated on 'check'
    lg_sel = []
    for lg in periods['sel'].league.value_counts().index[:8]:
        ms_ = (periods['sel'].league == lg).to_numpy()
        gl = grid(books['sel'], ms_, fills=('gt',)).sort_values('pnl_kick', ascending=False)
        best = rule_of(gl.iloc[0])
        mc = (periods['check'].league == lg).to_numpy()
        lg_sel.append({'league': lg, 'rule': best, 'sel': books['sel'].run(**best, mask=ms_, ci=True),
                       'check': books['check'].run(**best, mask=mc, ci=True) if mc.any() else None})
        print('league', lg, best, round(lg_sel[-1]['sel']['pnl_kick']), round((lg_sel[-1]['check'] or {}).get('pnl_kick', 0)),
              flush=True)
    res['per_league_selected'] = lg_sel
    json.dump(res, open(out / 'rules_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
