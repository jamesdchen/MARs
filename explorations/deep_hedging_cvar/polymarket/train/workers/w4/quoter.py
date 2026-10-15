"""w4 step 3: the learned toxicity-aware quoter, fn(obs) -> int actions (B, 3), like baselines.py.

For each side the edge model (model.py) gives, per (level 1..6, size 0..2), the fill probability,
expected shares and expected edge per share (spread capture + 60 s markout + rebate). The quoter
scores every cell with an inventory-skewed edge

    fac    = p_fill * shares * (edge -/+ k_inv * inv / Q)        (- for the bid, + for the ask)
    direct = E[credit] (direct head) -/+ p_fill * shares * k_inv * inv / Q
    score  = fac (mode 0), direct (mode 1) or their average (mode 2)

and per side takes the best cell among sizes <= max_size. It quotes it only if score > thr (USD
per decision) and the skewed edge per share > edge_min (cents); otherwise that side is off.
Above |inv| / Q > cap the side that would grow the position is off. Several models (data seeds)
can be averaged (ensemble) but the 3-seed rule uses one model per seed.

Knobs (tuned on val v2): thr, edge_min, k_inv, cap, max_size, mode; optionally per contract length
thr_<L> and edge_min_<L> (L in 5m, 15m, 4h), which override thr / edge_min on that length.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import load_model  # noqa: E402

DEFAULT = dict(thr=0.0, edge_min=-100.0, k_inv=0.02, cap=0.6, max_size=2, mode=0)


class Quoter:
    def __init__(self, model_paths, **params):
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        self.nets = [load_model(p) for p in model_paths]
        self.params = {**DEFAULT, **params}

    def predict(self, obs):
        outs = [n.ev(obs) for n in self.nets]
        if len(outs) == 1:
            return outs[0]
        return {k: np.mean([o[k] for o in outs], 0) for k in outs[0]}

    def __call__(self, obs):
        obs = np.asarray(obs, np.float32)
        P = self.params
        o = self.predict(obs)
        inv = obs[:, 33].astype(np.float64)                    # inventory / Q
        skew = P['k_inv'] * inv                                 # $/share
        edge = o['edge'].astype(np.float64).copy()              # (B, 2, 6, 3)
        edge[:, 0] -= skew[:, None, None]
        edge[:, 1] += skew[:, None, None]
        ps = o['p'] * o['shares']
        fac = ps * edge
        direct = o['ev_direct'].astype(np.float64).copy()
        direct[:, 0] -= ps[:, 0] * skew[:, None, None]
        direct[:, 1] += ps[:, 1] * skew[:, None, None]
        mode = int(P['mode'])
        score = fac if mode == 0 else direct if mode == 1 else 0.5 * (fac + direct)
        ms = int(P['max_size'])
        score[:, :, :, ms + 1:] = -np.inf
        B = obs.shape[0]
        flat = score.reshape(B, 2, 18)
        best = flat.argmax(-1)                                  # (B, 2)
        bs = np.take_along_axis(flat, best[..., None], -1)[..., 0]
        be = np.take_along_axis(edge.reshape(B, 2, 18), best[..., None], -1)[..., 0]
        thr = np.full(B, P['thr'], np.float64)
        emin = np.full(B, P['edge_min'], np.float64)
        for j, L in enumerate(('5m', '15m', '4h')):      # optional per-length knobs (obs 26-28 one-hot)
            m = obs[:, 26 + j] > 0.5
            thr[m] = P.get(f'thr_{L}', P['thr'])
            emin[m] = P.get(f'edge_min_{L}', P['edge_min'])
        on = (bs > thr[:, None]) & (be * 100.0 > emin[:, None])
        on[:, 0] &= ~(inv > P['cap'])
        on[:, 1] &= ~(inv < -P['cap'])
        level = best // 3 + 1
        size = best % 3
        act = np.zeros((B, 3), np.int32)
        act[:, 0] = np.where(on[:, 0], level[:, 0], 0)
        act[:, 1] = np.where(on[:, 1], level[:, 1], 0)
        # one size head for both sides: the size of the side with the larger score
        pick = np.where(bs[:, 0] >= bs[:, 1], size[:, 0], size[:, 1])
        act[:, 2] = pick
        return act
