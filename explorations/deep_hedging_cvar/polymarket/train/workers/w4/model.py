"""w4 step 2: toxicity / edge model fitted on the exploratory TRAIN dataset (collect.py).

For every action cell c = (side, level 1..6, size 0..2) (36 cells) a small MLP on the observation
predicts
  * p_fill(c | obs): the probability that the quote decided now gets at least one fill in the
    second it governs (BCE; the non-fill rows kept with probability keep_nofill are re-weighted),
  * shares(c | obs, fill): filled shares given a fill (MSE on log1p shares),
  * edge(c | obs, fill): USD per filled share = (spread capture + 60 s fair-value markout + rebate)
    / shares (Huber, in cents, weight = shares).
Expected edge per decision EV(c) = p_fill * shares * edge (USD).

Features: the market block (obs 0-32), our order state (36-43, 45) and the touch-vs-fair-value
columns (47-48). Inventory, wealth and fill counters are left out: the per-fill edge does not depend
on them (no market impact) and the collection env used a huge Q.

    python model.py --data runs/data_s1.npz --cv           # blocked CV by UTC hour (4 folds)
    python model.py --data runs/data_s1.npz --out models/edge_s1.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

FEAT = list(range(0, 33)) + list(range(36, 44)) + [45, 47, 48]
NC = 36   # 2 sides x 6 levels x 3 sizes


def cell(side, level, size):
    return (side.astype(np.int64) * 6 + (level.astype(np.int64) - 1)) * 3 + size.astype(np.int64)


class EdgeNet(nn.Module):
    def __init__(self, nin=len(FEAT), hidden=256, layers=3):
        super().__init__()
        mods, d = [], nin
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.trunk = nn.Sequential(*mods)
        self.head = nn.Linear(d, 4 * NC)
        self.register_buffer('mu', torch.zeros(nin))
        self.register_buffer('sd', torch.ones(nin))

    def forward(self, x):
        h = self.trunk((x - self.mu) / self.sd)
        o = self.head(h).view(-1, 4, NC)
        # fill logit, log1p shares | fill, edge cents/share | fill, direct E[credit] (cents)
        return o[:, 0], o[:, 1], o[:, 2], o[:, 3]

    @torch.no_grad()
    def ev(self, obs):
        """obs (B, 50) float32 -> dict of (B, 2, 6, 3) arrays: p, shares, edge (USD/share), ev (USD)."""
        x = torch.as_tensor(np.asarray(obs, np.float32)[:, FEAT])
        lf, ls, le, ld = self.forward(x)
        p = torch.sigmoid(lf)
        sh = torch.expm1(ls).clamp(min=0)
        e = le / 100.0
        sh_ = lambda t: t.view(-1, 2, 6, 3).numpy()  # noqa: E731
        return {'p': sh_(p), 'shares': sh_(sh), 'edge': sh_(e), 'ev': sh_(p * sh * e), 'ev_direct': sh_(ld / 100.0)}


def load_data(path):
    d = dict(np.load(path))
    keep = float(d.get('keep_nofill', 0.25))
    x = d['obs'].astype(np.float32)[:, FEAT]
    c = cell(d['side'], d['level'], d['size'])
    fill = (d['nfill'] > 0).astype(np.float32)
    w = np.where(fill > 0, 1.0, 1.0 / keep).astype(np.float32)
    sh = d['shares'].astype(np.float32)
    edge_c = np.where(fill > 0, 100.0 * d['credit'] / np.maximum(sh, 1e-6), 0.0).astype(np.float32)
    edge_c = np.clip(edge_c, -50, 50)
    return dict(x=x, c=c, fill=fill, w=w, sh=sh, edge=edge_c, credit=d['credit'].astype(np.float32),
                hour=d['hour'], market=d['market'])


def fit(D, idx, hidden=256, layers=3, epochs=4, lr=1e-3, bs=4096, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EdgeNet(hidden=hidden, layers=layers)
    x = torch.as_tensor(D['x'][idx])
    net.mu.copy_(x.mean(0))
    net.sd.copy_(x.std(0).clamp(min=1e-3))
    c = torch.as_tensor(D['c'][idx])
    f = torch.as_tensor(D['fill'][idx])
    w = torch.as_tensor(D['w'][idx])
    ls = torch.as_tensor(np.log1p(D['sh'][idx]))
    e = torch.as_tensor(D['edge'][idx])
    shw = torch.as_tensor(D['sh'][idx])
    cr = torch.as_tensor(np.clip(100.0 * D['credit'][idx], -1000, 1000))   # cents per decision
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(idx)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        t0, tot = time.time(), 0.0
        for k in range(n // bs):
            b = perm[k * bs:(k + 1) * bs]
            lf, lsh, le, ld = net(x[b])
            cb = c[b].unsqueeze(1)
            lf, lsh, le, ld = (t.gather(1, cb)[:, 0] for t in (lf, lsh, le, ld))
            l4 = (((ld - cr[b]) ** 2) * w[b]).sum() / w[b].sum()
            fb = f[b]
            l1 = (bce(lf, fb) * w[b]).sum() / w[b].sum()
            m = fb > 0
            if m.any():
                l2 = ((lsh[m] - ls[b][m]) ** 2).mean()
                sw = shw[b][m]
                l3 = (((le[m] - e[b][m]) ** 2) * sw).sum() / sw.sum()
            else:
                l2 = l3 = torch.zeros(())
            loss = l1 + 0.5 * l2 + 0.01 * l3 + 0.0001 * l4
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if verbose:
            print(f'  epoch {ep} loss {tot / max(1, n // bs):.4f} {time.time() - t0:.0f}s', flush=True)
    return net


@torch.no_grad()
def evaluate(net, D, idx):
    x = torch.as_tensor(D['x'][idx])
    c = torch.as_tensor(D['c'][idx]).unsqueeze(1)
    outs = [net(x[i:i + 65536]) for i in range(0, len(idx), 65536)]
    lf = torch.cat([o[0] for o in outs]).gather(1, c)[:, 0].numpy()
    ld = torch.cat([o[3] for o in outs]).gather(1, c)[:, 0].numpy()
    ls = torch.cat([o[1] for o in outs]).gather(1, c)[:, 0].numpy()
    le = torch.cat([o[2] for o in outs]).gather(1, c)[:, 0].numpy()
    f, w = D['fill'][idx], D['w'][idx]
    p = 1 / (1 + np.exp(-lf))
    ev = p * np.expm1(ls) * le / 100.0
    credit = D['credit'][idx]
    from sklearn.metrics import roc_auc_score
    m = f > 0
    out = {'n': int(len(idx)), 'fills': int(m.sum()),
           'auc_fill': float(roc_auc_score(f, p, sample_weight=w)),
           'calib_fill': float((p * w).sum() / (f * w).sum()),
           'edge_corr_on_fills': float(np.corrcoef(le[m], D['edge'][idx][m])[0, 1]),
           'realised_edge_cents_per_share': float(100 * credit[m].sum() / D['sh'][idx][m].sum()),
           'pred_edge_cents_per_share_on_fills': float((le[m] * D['sh'][idx][m]).sum() / D['sh'][idx][m].sum())}
    # decision value: weighted realised credit of rows the model would quote at each EV threshold
    for nm, val in (('fac', ev), ('dir', ld / 100.0)):
        for thr in (-0.005, 0.0, 0.002, 0.005):
            sel = val > thr
            out[f'{nm}_sel_frac@{thr}'] = round(float((w * sel).sum() / w.sum()), 4)
            out[f'{nm}_credit@{thr}'] = round(float((credit * w * sel).sum() / max(1e-9, (w * sel).sum())), 5)
        out[f'{nm}_corr'] = float(np.corrcoef(val, credit)[0, 1])
        # top decile by prediction: realised credit
        q = np.quantile(val, 0.9)
        out[f'{nm}_credit_top10'] = round(float(credit[val >= q].mean()), 5)
    out['realised_credit_per_decision_all'] = float((credit * w).sum() / w.sum())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--cv', action='store_true')
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    D = load_data(a.data)
    print(f'rows {len(D["c"])} fills {int(D["fill"].sum())} hours {len(np.unique(D["hour"]))}', flush=True)
    res = {'args': vars(a)}
    if a.cv:
        hours = np.unique(D['hour'])
        rng = np.random.default_rng(123)
        fold_of = dict(zip(hours, rng.permutation(len(hours)) % a.folds))
        fold = np.array([fold_of[h] for h in D['hour']])
        cv = []
        for k in range(a.folds):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            net = fit(D, tr, a.hidden, a.layers, a.epochs, seed=a.seed, verbose=False)
            r = evaluate(net, D, te)
            print(json.dumps({'fold': k, **r}), flush=True)
            cv.append(r)
        res['cv'] = cv
    if a.out:
        net = fit(D, np.arange(len(D['c'])), a.hidden, a.layers, a.epochs, seed=a.seed)
        res['train_fit'] = evaluate(net, D, np.arange(0, len(D['c']), 7))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save({'state': net.state_dict(), 'hidden': a.hidden, 'layers': a.layers, 'feat': FEAT,
                    'meta': res}, a.out)
        print(json.dumps(res['train_fit']), flush=True)
    if a.out or a.cv:
        path = (a.out or a.data) + '.json'
        json.dump(res, open(path, 'w'), indent=1)


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = EdgeNet(hidden=ck['hidden'], layers=ck['layers'])
    net.load_state_dict(ck['state'])
    net.eval()
    return net


if __name__ == '__main__':
    main()
