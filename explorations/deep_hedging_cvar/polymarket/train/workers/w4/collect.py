"""w4 step 1: exploratory quoting dataset on TRAIN markets (split 0) through the C env.

One env per train market (baselines/runner.py's Runner: cycle_stride = n, fixed_start = 0, so env i
always replays market i). Every env quotes ONE side at a time with a random (level 1..6, size 0..2),
held for a geometric number of decisions (mean --hold), with an off segment now and then, so queue
positions evolve as they would for a resting quote. The other side is off, so every fill of a step
belongs to the quoted side.

Labels come from the core itself: reward_mode 2 with markout_h = 60, inv_penalty = 0 credits each
fill at its print row with sq * (p_model[tau + 60] - p) plus its rebate (ENV.md section 5), i.e.
spread capture + 60 s markout (on fair value) + rebate. Q is huge so the inventory cap never blocks
a fill (the per-fill edge does not depend on inventory: no market impact). dt = 1 s for every
length, so a decision at step i is first matched by the prints booked at step i + LAG (measured by
--find-lag). Row j of the dataset: obs at decision step i, the action, and the credit / filled
shares / fill count booked at step i + LAG.

    python collect.py --find-lag
    python collect.py --seed 1 --out runs/data_s1.npz
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(PM, 'baselines'))
from runner import Runner  # noqa: E402

SCALE = 0.01   # reward_scale: reward = 0.01 * USD, clipped at +-1 => +-$100 per step
Q_BIG = 1e5
ENV = dict(dt=1, Q=Q_BIG, reward_mode=2, markout_h=60, inv_penalty=0.0, reward_scale=SCALE)


def make_runner(fill_model=1, latency=1):
    r = Runner(split=0, fill_model=fill_model, latency=latency, env=ENV)
    return r


def find_lag(r, steps=3000):
    """Quote a join bid on exactly one random step in 20; see at which later step fills appear."""
    b = r.binding
    b.vec_reset(r.vec, 0)
    rng = np.random.default_rng(0)
    n = r.n
    hist_on = []
    fills = []
    prev_nf = r.obs[:, 46].copy()
    for t in range(steps):
        on = rng.random(n) < 0.05
        a = np.zeros((n, 3), np.int32)
        a[on, 0] = 3
        r.act[:] = a
        b.vec_step(r.vec)
        nf = r.obs[:, 46].copy()
        d = np.round(np.expm1(5 * nf) - np.expm1(5 * prev_nf))
        d[r.term != 0] = 0
        prev_nf = nf
        hist_on.append(on)
        fills.append(d > 0)
    H = np.array(hist_on)
    F = np.array(fills)
    for lag in range(0, 9):
        num = (H[:steps - lag] & F[lag:]).sum()
        print(f'lag {lag}: fills given on {num / max(1, H[:steps - lag].sum()):.5f}')


def collect(r, seed, hold=15.0, p_off=0.1, keep_nofill=0.25):
    b = r.binding
    rng = np.random.default_rng(seed)
    n = r.n
    LAG = 5
    b.vec_reset(r.vec, 0)
    open_ts = np.array([m['open_ts'] for m in r.meta], np.int64)
    mkt = np.array([m['market'] for m in r.meta], np.int64)
    side = rng.integers(0, 2, n)
    lvl = rng.integers(1, 7, n)
    size = rng.integers(0, 3, n)
    off = rng.random(n) < p_off
    row = np.zeros(n, np.int64)
    ring_obs = np.zeros((LAG + 1, n, r.obs.shape[1]), np.float32)
    ring_act = np.zeros((LAG + 1, n, 3), np.int8)   # side, level (0 = off), size
    ring_row = np.zeros((LAG + 1, n), np.int64)
    ring_valid = np.zeros((LAG + 1, n), bool)
    prev_nf = r.obs[:, 46].copy()
    prev_inv = r.obs[:, 33].astype(np.float64) * Q_BIG
    out = {k: [] for k in ('obs', 'side', 'level', 'size', 'credit', 'shares', 'nfill', 'market', 'hour', 'row')}
    t0 = time.time()
    for t in range(r.max_steps):
        # segment switches
        sw = rng.random(n) < 1.0 / hold
        k = int(sw.sum())
        side[sw] = rng.integers(0, 2, k)
        lvl[sw] = rng.integers(1, 7, k)
        size[sw] = rng.integers(0, 3, k)
        off[sw] = rng.random(k) < p_off
        a = np.zeros((n, 3), np.int32)
        L = np.where(off, 0, lvl)
        a[side == 0, 0] = L[side == 0]
        a[side == 1, 1] = L[side == 1]
        a[:, 2] = size
        slot = t % (LAG + 1)
        ring_obs[slot] = r.obs
        ring_act[slot, :, 0] = side
        ring_act[slot, :, 1] = L
        ring_act[slot, :, 2] = size
        ring_row[slot] = row
        ring_valid[slot] = True
        r.act[:] = a
        b.vec_step(r.vec)
        term = r.term != 0
        nf = r.obs[:, 46].copy()
        inv = r.obs[:, 33].astype(np.float64) * Q_BIG
        dn = np.round(np.expm1(5 * nf.astype(np.float64)) - np.expm1(5 * prev_nf.astype(np.float64)))
        dsh = np.abs(inv - prev_inv)
        credit = r.rew.astype(np.float64) / SCALE
        # step t's prints were matched against the quote decided at step t - LAG + 1 ... the label
        # convention: attribute step t's booking to the decision LAG - 1 steps back (see find_lag)
        src = (t - (LAG - 1)) % (LAG + 1)
        ok = ring_valid[src] & ~term & (ring_act[src, :, 1] > 0) & (ring_row[src] == row - (LAG - 1))
        filled = ok & (dn > 0)
        keep = filled | (ok & (rng.random(n) < keep_nofill))
        idx = np.flatnonzero(keep)
        if idx.size:
            out['obs'].append(ring_obs[src, idx].astype(np.float16))
            out['side'].append(ring_act[src, idx, 0])
            out['level'].append(ring_act[src, idx, 1])
            out['size'].append(ring_act[src, idx, 2])
            out['credit'].append(np.where(dn[idx] > 0, credit[idx], 0.0).astype(np.float32))
            out['shares'].append(np.where(dn[idx] > 0, dsh[idx], 0.0).astype(np.float32))
            out['nfill'].append(dn[idx].astype(np.int16))
            out['market'].append(mkt[idx].astype(np.int32))
            out['hour'].append(((open_ts[idx] + ring_row[src, idx]) // 3600).astype(np.int32))
            out['row'].append(ring_row[src, idx].astype(np.int32))
        row += 1
        row[term] = 0
        ring_valid[:, term] = False
        prev_nf = nf
        prev_inv = inv
        if t % 2000 == 0:
            nrow = sum(len(x) for x in out['side'])
            print(f't={t} rows={nrow} {time.time() - t0:.0f}s', flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--find-lag', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--hold', type=float, default=15.0)
    p.add_argument('--keep-nofill', type=float, default=0.25)
    p.add_argument('--out', default=os.path.join(HERE, 'runs', 'data_s1.npz'))
    a = p.parse_args()
    r = make_runner()
    if a.find_lag:
        find_lag(r)
        return
    d = collect(r, a.seed, a.hold, keep_nofill=a.keep_nofill)
    d['keep_nofill'] = np.float32(a.keep_nofill)
    np.savez(a.out, **d)
    f = d['nfill'] > 0
    print(f'rows {len(f)} fills {f.sum()} credit/fill-row {d["credit"][f].mean():.4f} '
          f'per-share {d["credit"][f].sum() / d["shares"][f].sum():.5f}')


if __name__ == '__main__':
    main()
