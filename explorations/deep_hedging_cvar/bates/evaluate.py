"""Score every hedger on common held-out paths under the exact model.

Anything a method tunes (band constants, the threshold w, sweep picks) is
chosen on validation paths (seed 999), then frozen and scored on separate
test paths (seed 12345), with deterministic actions and hard trade gates.

Runs are passed as kind|label|path:
  baselines  tune_baselines.py's JSON: no hedge, BS delta, Bates delta,
             Bates delta-vega and the tuned no-trade band
  pathwise   a train_pathwise.py checkpoint (or a sweep's best.pt)
  ppo3       a train_ppo.py checkpoint (PufferLib 3.0, C env)
  ppo5       a PufferLib 5.0 .bin checkpoint with its run facts in a .json
             next to it (wall_time, steps, device, hidden, layers)
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

from market import (BatesConfig, market_noise, simulate, ru_objective, summarize, cvar)
from policy import HedgePolicy

VAL_SEED, TEST_SEED = 999, 12345


def w_search(val, fn, w_grid, cfg):
    j = [ru_objective(simulate(val, fn, w, cfg), w, cfg.alpha).item() for w in w_grid]
    return float(w_grid[int(np.argmin(j))]), j


def load_torch_policy(path, device):
    ck = torch.load(path, weights_only=False, map_location='cpu')
    pol = HedgePolicy(hidden=ck.get('hidden', 64))
    if 'actor' in ck:
        pol.actor.load_state_dict(ck['actor'])
    else:
        pol.load_state_dict(ck['policy'])
    return pol.to(device).eval().mean_action, ck


def baseline_strategies(path, cfg, device):
    """The fixed classical hedgers and the tuned band (baselines.py)."""
    import baselines as B
    from greeks import BookGreeks
    with open(path) as f:
        res = json.load(f)
    greeks = BookGreeks(cfg)
    out = {name: dict(fn=fn, w=0.0) for name, fn in B.untuned(cfg, greeks).items()}
    out['tuned no-trade band'] = dict(
        fn=B.tuned_band(cfg, greeks, path), w=0.0, device=res.get('device', device),
        tune_time=res.get('tune_wall_time', float('nan')), steps=res.get('tune_steps', float('nan')))
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, help='kind|label|path entries')
    p.add_argument('--out', default='results/combined')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n-test', type=int, default=200_000)
    p.add_argument('--n-val', type=int, default=100_000)
    args = p.parse_args()
    dev = args.device
    cfg = BatesConfig()
    a = cfg.alpha
    val = market_noise(args.n_val, cfg, torch.Generator().manual_seed(VAL_SEED), dev)
    test = market_noise(args.n_test, cfg, torch.Generator().manual_seed(TEST_SEED), dev)

    strategies, ru_curves = {}, {}
    for kind, label, path in (r.split('|') for r in args.runs):
        if not os.path.exists(path):
            print(f'skipping {label}: {path} not found')
            continue
        if kind == 'baselines':
            strategies.update(baseline_strategies(path, cfg, dev))
            continue
        if kind == 'pathwise':
            fn, ck = load_torch_policy(path, dev)
            st = dict(fn=fn, w=ck['w'], history=ck.get('history'))
        elif kind == 'ppo3':
            fn, ck = load_torch_policy(path, dev)
            w_grid = ck['w_center'] + np.linspace(-0.3, 0.3, 25) * cfg.scale
            w, j = w_search(val, fn, w_grid, cfg)
            ru_curves[label] = (w_grid, j)
            st = dict(fn=fn, w=w, history=ck.get('history'))
        elif kind == 'ppo5':
            from puffer5.puffernet import PufferNetPolicy
            with open(os.path.splitext(path)[0] + '.json') as f:
                ck = json.load(f)
            fn = PufferNetPolicy(path, hidden=ck['hidden'], layers=ck['layers'], device=dev)
            w_grid = np.linspace(0, 1.5, 31) * cfg.scale
            w, j = w_search(val, fn, w_grid, cfg)
            ru_curves[label] = (w_grid, j)
            st = dict(fn=fn, w=w)
        else:
            raise ValueError(kind)
        st.update(wall_time=ck.get('wall_time', float('nan')), steps=ck.get('steps', float('nan')),
                  device=ck.get('device', 'cpu'), tune_time=ck.get('tune_wall_time', float('nan')))
        strategies[label] = st

    rows, losses, recs = {}, {}, {}
    for name, st in strategies.items():
        t0 = time.time()
        loss, rec = simulate(test, st['fn'], st['w'], cfg, record=True)
        losses[name], recs[name] = loss.cpu(), {k: v.cpu() for k, v in rec.items()}
        rows[name] = {f'val CVaR{a:g}': cvar(simulate(val, st['fn'], st['w'], cfg).cpu(), a),
                      **summarize(loss.cpu(), a),
                      'stock trades': rec['trade_s'].float().sum(1).mean().item(),
                      'swap trades': rec['trade_y'].float().sum(1).mean().item(),
                      'w': st['w'], 'device': str(st.get('device', '–')),
                      'train wall time (s)': st.get('wall_time', float('nan')),
                      'train sim steps': st.get('steps', float('nan')),
                      'tuning wall time (s)': st.get('tune_time', float('nan')),
                      'eval time (s)': time.time() - t0}

    os.makedirs(args.out, exist_ok=True)
    with open(f'{args.out}/summary.json', 'w') as f:
        json.dump({'cfg': cfg.to_dict(), 'rows': rows,
                   'ru_curves': {k: [list(map(float, g)), j] for k, (g, j) in ru_curves.items()}},
                  f, indent=2)
    cols = [f'val CVaR{a:g}', 'mean', 'std', f'VaR{a:g}', f'CVaR{a:g}', 'stock trades',
            'swap trades', 'device', 'train wall time (s)', 'train sim steps',
            'tuning wall time (s)']
    dash = lambda f: (lambda v: '–' if isinstance(v, float) and np.isnan(v) else f(v))  # noqa: E731
    fmt = {'device': str, 'train sim steps': dash(lambda v: f'{v / 1e6:.0f}M'),
           'train wall time (s)': dash(lambda v: f'{v:.0f}'),
           'tuning wall time (s)': dash(lambda v: f'{v:.0f}'),
           'stock trades': lambda v: f'{v:.1f}', 'swap trades': lambda v: f'{v:.1f}'}
    lines = ['| strategy | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for name, r in rows.items():
        lines.append(f'| {name} | ' + ' | '.join(fmt.get(c, lambda v: f'{v:.3f}')(r[c])
                                                  for c in cols) + ' |')
    table = '\n'.join(lines)
    with open(f'{args.out}/summary.md', 'w') as f:
        f.write(table + '\n')
    print(table)

    best = sorted((n for n in rows if n != 'no hedge'), key=lambda n: rows[n][f'val CVaR{a:g}'])[:6]

    # 1. Loss distributions of the six best on validation.
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(-5, 15, 160)
    for name in best:
        ax.hist(losses[name].numpy(), bins=bins, histtype='step', density=True, lw=1.3,
                label=f"{name}  (CVaR {rows[name][f'CVaR{a:g}']:.2f})")
    ax.set_yscale('log')
    ax.set_xlabel('hedging loss L')
    ax.set_ylabel('density (log)')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{args.out}/loss_hist.png', dpi=130)

    # 2. Stock and swap positions along one test path with a jump.
    jumps = (test[:, :, :, 2] < cfg.lam * cfg.dt_sub).any(-1).any(-1).cpu()
    i = int(torch.nonzero(jumps)[0]) if jumps.any() else 0
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t = np.arange(cfg.n_steps)
    for name in best[:4]:
        axs[0].step(t, recs[name]['delta'][i], where='post', lw=1.3, label=name)
        axs[1].step(t, recs[name]['y'][i], where='post', lw=1.3, label=name)
    ax2 = axs[0].twinx()
    ax2.plot(t, recs[best[0]]['s'][i], 'k:', lw=1)
    ax2.set_ylabel('stock price (dotted)')
    axs[0].set_ylabel('stock position')
    axs[1].set_ylabel('variance swap position')
    axs[1].set_xlabel('hedging date')
    axs[0].set_title(f'test path {i} (contains a jump)')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{args.out}/positions.png', dpi=130)

    # 3. Learning curves against wall-clock time.
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, st in strategies.items():
        h = st.get('history')
        if not h:
            continue
        key = f'CVaR{a:g}' if f'CVaR{a:g}' in h[0] else 'ru'
        what = 'CVaR, exact model' if key != 'ru' else 'RU objective, sampled actions'
        ax.plot([r['time'] / 60 for r in h], [r[key] for r in h], label=f'{name} ({what})')
    for name in rows:
        if name.startswith('tuned no-trade band'):
            ax.axhline(rows[name][f'CVaR{a:g}'], ls='--', c='gray', lw=1, label=f'{name} (test)')
    ax.set_xlabel('wall-clock minutes')
    ax.set_ylabel('training objective')
    ax.set_ylim(top=12)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{args.out}/training.png', dpi=130)

    # 4. Outer minimization over w for the PPO policies.
    if ru_curves:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for name, (g, j) in ru_curves.items():
            ax.plot(g, j, marker='o', ms=3, label=f'{name} (w* = {strategies[name]["w"]:.2f})')
        ax.set_xlabel('threshold w (policy input)')
        ax.set_ylabel(r'$w + E[(L-w)^+]/(1-\alpha)$')
        ax.legend(frameon=False, fontsize=7)
        ax.set_title('PPO policies, validation paths')
        fig.tight_layout()
        fig.savefig(f'{args.out}/ru_objective.png', dpi=130)


if __name__ == '__main__':
    main()
