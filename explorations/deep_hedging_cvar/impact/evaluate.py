"""Score all hedgers on common held-out paths under the exact model.

Every method gets the same treatment: anything it tunes (band width, the
threshold w) is chosen on validation paths, then frozen and scored on
separate test paths, with deterministic actions and the hard trade gate.
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


def load(path):
    ckpt = torch.load(path, weights_only=False)
    pol = ImpactHedgePolicy()
    if 'actor' in ckpt:
        pol.actor.load_state_dict(ckpt['actor'])
    else:
        pol.load_state_dict(ckpt['policy'])
    pol.eval()
    return pol, ckpt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--n-test', type=int, default=200_000)
    p.add_argument('--n-val', type=int, default=100_000)
    args = p.parse_args()
    R = args.results
    cfg = ImpactConfig()
    a = cfg.alpha
    val = fundamental_noise(args.n_val, cfg, torch.Generator().manual_seed(999))
    test = fundamental_noise(args.n_test, cfg, torch.Generator().manual_seed(12345))

    # Tuned no-transaction band: grid search on validation CVaR.
    t0 = time.time()
    grid = [(h, e) for h in np.round(np.arange(0, 0.32, 0.02), 2) for e in [0, 0.25, 0.5, 0.75, 1]]
    band_val = {(h, e): cvar(simulate(val, bs_delta_band(cfg, h, e), 0, cfg), a) for h, e in grid}
    h_best, e_best = min(band_val, key=band_val.get)
    band_time = time.time() - t0

    strategies = {
        'no hedge': dict(fn=no_hedge, w=0.0),
        'BS delta, daily': dict(fn=bs_delta_band(cfg), w=0.0),
        f'tuned band (h={h_best:.2f}, edge={e_best:g})': dict(
            fn=bs_delta_band(cfg, h_best, e_best), w=0.0, wall_time=band_time,
            steps=len(grid) * args.n_val * cfg.n_steps),
    }
    band_name = list(strategies)[-1]
    for gate in ['hard', 'ste', 'sigmoid']:
        pol, ck = load(f'{R}/pathwise_{gate}.pt')
        strategies[f'pathwise, {gate} gate'] = dict(
            fn=pol.mean_action, w=ck['w'], wall_time=ck['wall_time'], steps=ck['steps'],
            history=ck['history'])

    # PPO: grid search on w around the center the env converged to.
    pol, ck = load(f'{R}/ppo.pt')
    w_grid = ck['w_center'] + np.linspace(-0.1, 0.1, 21) * cfg.scale
    j_val = [ru_objective(simulate(val, pol.mean_action, w, cfg), w, a).item() for w in w_grid]
    w_star = float(w_grid[int(np.argmin(j_val))])
    strategies['PPO (PufferLib, Cython env)'] = dict(
        fn=pol.mean_action, w=w_star, wall_time=ck['wall_time'], steps=ck['steps'],
        history=ck['history'])

    rows, losses, recs = {}, {}, {}
    for name, st in strategies.items():
        loss, rec = simulate(test, st['fn'], st['w'], cfg, record=True)
        losses[name], recs[name] = loss, rec
        rows[name] = {**summarize(loss, a),
                      'trades': rec['trade'].float().sum(1).mean().item(),
                      'w': st['w'],
                      'train wall time (s)': st.get('wall_time', float('nan')),
                      'train sim steps': st.get('steps', float('nan'))}

    os.makedirs(R, exist_ok=True)
    with open(f'{R}/summary.json', 'w') as f:
        json.dump({'cfg': cfg.to_dict(), 'rows': rows, 'ppo_w_grid': w_grid.tolist(),
                   'ppo_ru_objective_val': j_val,
                   'band_val_cvar': {f'{h},{e}': v for (h, e), v in band_val.items()}},
                  f, indent=2)
    cols = ['mean', 'std', f'VaR{a:g}', f'CVaR{a:g}', 'trades', 'train wall time (s)',
            'train sim steps']
    fmt = {'train sim steps': lambda v: '–' if np.isnan(v) else f'{v / 1e6:.0f}M',
           'train wall time (s)': lambda v: '–' if np.isnan(v) else f'{v:.0f}'}
    lines = ['| strategy | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for name, r in rows.items():
        cells = [fmt.get(c, lambda v: f'{v:.3f}')(r[c]) for c in cols]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    table = '\n'.join(lines)
    with open(f'{R}/summary.md', 'w') as f:
        f.write(table + '\n')
    print(table)

    main_names = ['BS delta, daily', band_name, 'pathwise, ste gate', 'pathwise, sigmoid gate',
                  'PPO (PufferLib, Cython env)']

    # 1. Loss distributions.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(-2, 6, 140)
    for name in main_names:
        ax.hist(losses[name].numpy(), bins=bins, histtype='step', density=True, lw=1.3,
                label=f"{name}  (CVaR {rows[name][f'CVaR{a:g}']:.2f})")
    ax.set_yscale('log')
    ax.set_xlabel('hedging loss L')
    ax.set_ylabel('density (log)')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{R}/loss_hist.png', dpi=130)

    # 2. Positions along two test paths.
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    t = np.arange(cfg.n_steps)
    for ax, i in zip(axs, [0, 3]):
        s = recs['BS delta, daily']['s'][i]
        tau = cfg.maturity * (1 - torch.arange(cfg.n_steps) / cfg.n_steps)
        ax.plot(t, bs_call(s, tau, cfg)[1], 'k:', lw=1, label='BS delta of this path')
        for name in main_names[1:]:
            ax.step(t, recs[name]['delta'][i], where='post', lw=1.3, label=name)
        ax.set_xlabel('hedging date')
        ax.set_title(f'test path {i}: S_T = {recs[band_name]["s_T"][i]:.1f}')
    axs[0].set_ylabel('position after trading')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{R}/positions.png', dpi=130)

    # 3. Learning curves against wall-clock time (4 CPU threads each).
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for gate in ['hard', 'ste', 'sigmoid']:
        h = strategies[f'pathwise, {gate} gate']['history']
        ax.plot([r['time'] / 60 for r in h], [r[f'CVaR{a:g}'] for r in h],
                label=f'pathwise, {gate} gate (exact model, 4096 paths)')
    h = strategies['PPO (PufferLib, Cython env)']['history']
    ax.plot([r['time'] / 60 for r in h], [r[f'loss_cvar{a:g}'] for r in h],
            label='PPO (sampled actions, all 4096 envs)')
    ax.axhline(rows[band_name][f'CVaR{a:g}'], ls='--', c='gray', lw=1, label='tuned band (test)')
    ax.set_ylim(top=4)
    ax.set_xlabel('wall-clock minutes')
    ax.set_ylabel(f'CVaR{a:g} of training batch')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{R}/training.png', dpi=130)

    # 4. Outer minimization over w for PPO.
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(w_grid, j_val, marker='o', ms=3)
    ax.axvline(w_star, ls='--', c='gray', lw=1)
    ax.set_xlabel('threshold w (policy input)')
    ax.set_ylabel(r'$w + E[(L-w)^+]/(1-\alpha)$')
    ax.set_title(f'PPO, validation paths (w* = {w_star:.2f})')
    fig.tight_layout()
    fig.savefig(f'{R}/ru_objective.png', dpi=130)


if __name__ == '__main__':
    main()
