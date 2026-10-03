"""Score all hedgers on common held-out paths under the exact model.

Every method gets the same treatment: anything it tunes (band width, the
threshold w) is chosen on validation paths, then frozen and scored on
separate test paths, with deterministic actions and the hard trade gate.

Trained runs are passed as kind|label|path with kind one of
  pathwise  a train_pathwise.py checkpoint
  ppo3      a train_ppo.py checkpoint (PufferLib 3.0, Cython env)
  ppo5      a PufferLib 5.0 .bin checkpoint, with a JSON file of run facts
            (wall_time, steps, device, hidden, layers) next to it
Without --runs, every checkpoint found under --results is used.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from market import (ImpactConfig, fundamental_noise, simulate, bs_delta_band, no_hedge,
                    ru_objective, summarize, cvar, bs_call)
from policy import ImpactHedgePolicy


def load_torch(path, device):
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    pol = ImpactHedgePolicy(hidden=ckpt.get('hidden', 64))
    if 'actor' in ckpt:
        pol.actor.load_state_dict(ckpt['actor'])
    else:
        pol.load_state_dict(ckpt['policy'])
    return pol.to(device).eval().mean_action, ckpt


def load_puffer5(path, device):
    from puffer5.puffernet import PufferNetPolicy
    with open(os.path.splitext(path)[0] + '.json') as f:
        meta = json.load(f)
    pol = PufferNetPolicy(path, hidden=meta['hidden'], layers=meta['layers'], device=device)
    return pol, meta


def default_runs(results):
    runs = []
    for gate in ['hard', 'ste', 'sigmoid']:
        for sub, where in [('', 'CPU'), ('gpu/', 'GPU')]:
            path = f'{results}/{sub}pathwise_{gate}.pt'
            if os.path.exists(path):
                runs.append(('pathwise', f'pathwise, {gate} gate ({where})', path))
    if os.path.exists(f'{results}/ppo.pt'):
        runs.append(('ppo3', 'PPO, PufferLib 3.0 + Cython env (CPU)', f'{results}/ppo.pt'))
    if os.path.exists(f'{results}/gpu/puffer5.bin'):
        runs.append(('ppo5', 'PPO, PufferLib 5.0 (GPU)', f'{results}/gpu/puffer5.bin'))
    return runs


def w_search(val, fn, w_grid, cfg):
    j = [ru_objective(simulate(val, fn, w, cfg), w, cfg.alpha).item() for w in w_grid]
    return float(w_grid[int(np.argmin(j))]), j


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--runs', nargs='*', default=None, help='kind|label|path entries')
    p.add_argument('--out', default=None, help='output dir (default: --results)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n-test', type=int, default=200_000)
    p.add_argument('--n-val', type=int, default=100_000)
    args = p.parse_args()
    out = args.out or args.results
    dev = args.device
    cfg = ImpactConfig()
    a = cfg.alpha
    val = fundamental_noise(args.n_val, cfg, torch.Generator().manual_seed(999), dev)
    test = fundamental_noise(args.n_test, cfg, torch.Generator().manual_seed(12345), dev)

    # Tuned no-transaction band: grid search on validation CVaR.
    t0 = time.time()
    grid = [(h, e) for h in np.round(np.arange(0, 0.32, 0.02), 2) for e in [0, 0.25, 0.5, 0.75, 1]]
    band_val = {(h, e): cvar(simulate(val, bs_delta_band(cfg, h, e), 0, cfg), a) for h, e in grid}
    h_best, e_best = min(band_val, key=band_val.get)
    band_name = f'tuned band (h={h_best:.2f}, edge={e_best:g})'
    strategies = {
        'no hedge': dict(fn=no_hedge, w=0.0),
        'BS delta, daily': dict(fn=bs_delta_band(cfg), w=0.0),
        band_name: dict(fn=bs_delta_band(cfg, h_best, e_best), w=0.0, device=dev,
                        wall_time=time.time() - t0, steps=len(grid) * args.n_val * cfg.n_steps),
    }

    ru_curves = {}
    for kind, label, path in [r.split('|') if isinstance(r, str) else r
                              for r in (args.runs or default_runs(args.results))]:
        if kind == 'pathwise':
            fn, ck = load_torch(path, dev)
            st = dict(fn=fn, w=ck['w'], history=ck['history'])
        elif kind == 'ppo3':
            fn, ck = load_torch(path, dev)
            w_grid = ck['w_center'] + np.linspace(-0.1, 0.1, 21) * cfg.scale
            w, j = w_search(val, fn, w_grid, cfg)
            ru_curves[label] = (w_grid, j)
            st = dict(fn=fn, w=w, history=ck['history'])
        else:
            fn, ck = load_puffer5(path, dev)
            # Each 5.0 env tracks its own w, so search a wide range.
            w_grid = np.linspace(0, 0.8, 33) * cfg.scale
            w, j = w_search(val, fn, w_grid, cfg)
            ru_curves[label] = (w_grid, j)
            st = dict(fn=fn, w=w)
        st.update(wall_time=ck.get('wall_time', float('nan')), steps=ck.get('steps', float('nan')),
                  device=ck.get('device', 'cpu'), tune_time=ck.get('tune_wall_time', float('nan')))
        strategies[label] = st

    rows, losses, recs = {}, {}, {}
    for name, st in strategies.items():
        loss, rec = simulate(test, st['fn'], st['w'], cfg, record=True)
        losses[name], recs[name] = loss.cpu(), {k: v.cpu() for k, v in rec.items()}
        rows[name] = {f'val CVaR{a:g}': cvar(simulate(val, st['fn'], st['w'], cfg).cpu(), a),
                      **summarize(loss.cpu(), a),
                      'trades': rec['trade'].float().sum(1).mean().item(),
                      'w': st['w'], 'device': st.get('device', '–'),
                      'train wall time (s)': st.get('wall_time', float('nan')),
                      'train sim steps': st.get('steps', float('nan')),
                      'tuning wall time (s)': st.get('tune_time', float('nan'))}

    os.makedirs(out, exist_ok=True)
    with open(f'{out}/summary.json', 'w') as f:
        json.dump({'cfg': cfg.to_dict(), 'rows': rows,
                   'ru_curves': {k: [list(map(float, g)), j] for k, (g, j) in ru_curves.items()},
                   'band_val_cvar': {f'{h},{e}': v for (h, e), v in band_val.items()}},
                  f, indent=2)
    cols = [f'val CVaR{a:g}', 'mean', 'std', f'VaR{a:g}', f'CVaR{a:g}', 'trades', 'device',
            'train wall time (s)', 'train sim steps', 'tuning wall time (s)']
    num = lambda v: f'{v:.3f}'
    fmt = {'device': str,
           'train sim steps': lambda v: '–' if np.isnan(v) else f'{v / 1e6:.0f}M',
           'train wall time (s)': lambda v: '–' if np.isnan(v) else f'{v:.0f}',
           'tuning wall time (s)': lambda v: '–' if np.isnan(v) else f'{v:.0f}'}
    lines = ['| strategy | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for name, r in rows.items():
        lines.append(f'| {name} | ' + ' | '.join(fmt.get(c, num)(r[c]) for c in cols) + ' |')
    table = '\n'.join(lines)
    with open(f'{out}/summary.md', 'w') as f:
        f.write(table + '\n')
    print(table)

    learned = [n for n in strategies if n.startswith(('pathwise', 'PPO')) and 'hard' not in n]
    main_names = ['BS delta, daily', band_name, *learned]

    # 1. Loss distributions.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(-2, 6, 140)
    for name in main_names:
        ax.hist(losses[name].numpy(), bins=bins, histtype='step', density=True, lw=1.3,
                label=f"{name}  (CVaR {rows[name][f'CVaR{a:g}']:.2f})")
    ax.set_yscale('log')
    ax.set_xlabel('hedging loss L')
    ax.set_ylabel('density (log)')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{out}/loss_hist.png', dpi=130)

    # 2. Positions along two test paths.
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    t = np.arange(cfg.n_steps)
    tau = cfg.maturity * (1 - torch.arange(cfg.n_steps) / cfg.n_steps)
    for ax, i in zip(axs, [0, 3]):
        s = recs['BS delta, daily']['s'][i]
        ax.plot(t, bs_call(s, tau, cfg)[1], 'k:', lw=1, label='BS delta of this path')
        for name in main_names[1:]:
            ax.step(t, recs[name]['delta'][i], where='post', lw=1.3, label=name)
        ax.set_xlabel('hedging date')
        ax.set_title(f'test path {i}: S_T = {recs[band_name]["s_T"][i]:.1f}')
    axs[0].set_ylabel('position after trading')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{out}/positions.png', dpi=130)

    # 3. Learning curves against wall-clock time.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, st in strategies.items():
        h = st.get('history')
        if not h:
            continue
        key = f'CVaR{a:g}' if f'CVaR{a:g}' in h[0] else f'loss_cvar{a:g}'
        ax.plot([r['time'] / 60 for r in h], [r[key] for r in h], label=name)
    ax.axhline(rows[band_name][f'CVaR{a:g}'], ls='--', c='gray', lw=1, label='tuned band (test)')
    ax.set_ylim(top=4)
    ax.set_xlabel('wall-clock minutes')
    ax.set_ylabel(f'CVaR{a:g} of training batch')
    ax.set_title('pathwise: exact model on 4096 paths; PPO: sampled actions', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{out}/training.png', dpi=130)

    # 4. Outer minimization over w for the PPO policies.
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, (g, j) in ru_curves.items():
        ax.plot(g, j, marker='o', ms=3, label=f'{name} (w* = {strategies[name]["w"]:.2f})')
    ax.set_xlabel('threshold w (policy input)')
    ax.set_ylabel(r'$w + E[(L-w)^+]/(1-\alpha)$')
    ax.set_ylim(top=5)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title('PPO policies, validation paths')
    fig.tight_layout()
    fig.savefig(f'{out}/ru_objective.png', dpi=130)


if __name__ == '__main__':
    main()
