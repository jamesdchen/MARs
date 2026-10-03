"""Compare hedgers on common out-of-sample paths and make plots.

For the PPO policy pi(a | s, w) the outer Rockafellar-Uryasev minimization
over w is a grid search on validation paths; the chosen w* is then frozen
and the hedger is scored on separate test paths.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hedging import (MarketConfig, simulate_gbm, hedge_loss, ru_objective, summarize,
                     bs_delta_policy, no_hedge_policy)
from policy import HedgePolicy


def load_actor(path, key):
    ckpt = torch.load(path, weights_only=False)
    pol = HedgePolicy()
    if key == 'actor':
        pol.actor.load_state_dict(ckpt['actor'])
    else:
        pol.load_state_dict(ckpt['policy'])
    pol.eval()
    return pol, ckpt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pathwise', default='results/pathwise.pt')
    p.add_argument('--ppo', nargs='+', default=['shaped reward=results/ppo.pt',
                                                 'terminal reward=results/ppo_terminal.pt',
                                                 'shaped, fixed w=results/ppo_fixed_w.pt'],
                   help='label=path pairs for PPO checkpoints')
    p.add_argument('--n-test', type=int, default=200_000)
    p.add_argument('--n-val', type=int, default=100_000)
    p.add_argument('--out', default='results')
    args = p.parse_args()

    pw, pw_ckpt = load_actor(args.pathwise, 'actor')
    ppos = {}
    for item in args.ppo:
        label, path = item.split('=')
        ppos[f'PPO, {label}'] = load_actor(path, 'policy')
    cfg = MarketConfig(**pw_ckpt['cfg'])
    a = cfg.alpha
    val = simulate_gbm(args.n_val, cfg, torch.Generator().manual_seed(999))
    test = simulate_gbm(args.n_test, cfg, torch.Generator().manual_seed(12345))

    # Outer minimization over w for each PPO policy, on the w range it was
    # trained on (a single point when w was fixed during training).
    w_grid, j_val, w_star = {}, {}, {}
    for name, (pol, ck) in ppos.items():
        lo, hi = ck['cfg']['w_low'], ck['cfg']['w_high']
        w_grid[name] = np.linspace(lo, hi, 26 if hi > lo else 1) * cfg.scale
        j_val[name] = [ru_objective(hedge_loss(val, pol.mean_action, w, cfg), w, a).item()
                       for w in w_grid[name]]
        w_star[name] = float(w_grid[name][int(np.argmin(j_val[name]))])

    strategies = {
        'no hedge': (no_hedge_policy, 0.0),
        'BS delta': (bs_delta_policy(cfg), 0.0),
        'pathwise CVaR': (pw.mean_action, pw_ckpt['w']),
        **{name: (pol.mean_action, w_star[name]) for name, (pol, _) in ppos.items()},
    }
    rows, losses, deltas = {}, {}, {}
    for name, (fn, w) in strategies.items():
        loss, deltas[name] = hedge_loss(test, fn, w, cfg, return_deltas=True)
        losses[name] = loss
        rows[name] = {**summarize(loss, a), 'w': w,
                      'RU objective at w': ru_objective(loss, w, a).item()}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'cfg': cfg.to_dict(), 'w_star': w_star,
                   'w_grid': {k: v.tolist() for k, v in w_grid.items()},
                   'ru_objective_val': j_val, 'strategies': rows}, f, indent=2)

    cols = ['mean', 'std', f'VaR{a:g}', f'CVaR{a:g}', 'w']
    lines = ['| strategy | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for name, r in rows.items():
        lines.append(f'| {name} | ' + ' | '.join(f'{r[c]:.3f}' for c in cols) + ' |')
    table = '\n'.join(lines)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(table + '\n')
    print(table)

    # 1. Loss distributions.
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1.5, 4, 120)
    for name in ['BS delta', 'pathwise CVaR', *ppos]:
        ax.hist(losses[name].numpy(), bins=bins, histtype='step', density=True, lw=1.4,
                label=f"{name}  (CVaR {rows[name][f'CVaR{a:g}']:.2f})")
    ax.set_yscale('log')
    ax.set_xlabel('hedging loss L')
    ax.set_ylabel('density (log)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'loss_hist.png'), dpi=130)

    # 2. Rockafellar-Uryasev objective of the w-conditioned policy.
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name in ppos:
        if len(w_grid[name]) == 1:
            continue
        ax.plot(w_grid[name], j_val[name], marker='o', ms=3, label=f'{name} (w* = {w_star[name]:.2f})')
    ax.axhline(rows['pathwise CVaR'][f'CVaR{a:g}'], ls='--', c='gray', lw=1, label='pathwise CVaR')
    ax.set_xlabel('threshold w (policy input)')
    ax.set_ylabel(r'$w + E[(L-w)^+]/(1-\alpha)$')
    ax.set_ylim(top=4)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('w-conditioned PPO policies, validation paths')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'ru_objective.png'), dpi=130)

    # 3. Hedge ratio chosen at mid-life along test paths, against spot.
    k = cfg.n_steps // 2
    idx = slice(0, 3000)
    s = test[idx, k]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ['BS delta', 'pathwise CVaR', *ppos]:
        ax.scatter(s, deltas[name][idx, k], s=2, alpha=0.4, label=name)
    ax.set_xlabel(f'spot at step {k} of {cfg.n_steps}')
    ax.set_ylabel('hedge ratio')
    ax.legend(frameon=False, markerscale=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'hedge_ratio.png'), dpi=130)

    # 4. Training curves on the training batches (CVaR of the sampled losses).
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    h = pw_ckpt['history']
    axs[0].plot([r['iter'] for r in h], [r[f'CVaR{a:g}'] for r in h])
    axs[0].set_xlabel('gradient step')
    axs[0].set_title('pathwise: batch CVaR')
    key = f'loss_cvar{a:g}'
    for name, (_, ck) in ppos.items():
        h = ck['history']
        axs[1].plot([r['step'] / 1e6 for r in h], [r[key] for r in h], label=name)
    axs[1].set_xlabel('env steps (M)')
    axs[1].set_title('PPO: CVaR of sampled losses (with exploration noise)')
    for ax in axs:
        ax.axhline(rows['BS delta'][f'CVaR{a:g}'], ls='--', c='gray', lw=1, label='BS delta')
        ax.set_ylim(top=4)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'training.png'), dpi=130)


if __name__ == '__main__':
    main()
