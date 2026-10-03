"""How far the pathwise gradient is from the true gradient once the crowd
reacts to our trades.

Policy: hold theta times the Black-Scholes book delta in stock, rebalanced
on every date, no swap. Trading every date removes the trade decisions, so
any gap below comes from the market alone. On each path the loss is
discontinuous in theta wherever our impact moves a dealer across the edge
of its band, or moves the last return across a step of the Poisson count
of the random orders. Autograd sees only the slope between those jumps.

For E[L] and for the Rockafellar-Uryasev objective (w fixed at the VaR of
the loss at theta = 1, so its theta-derivative is the CVaR's), at theta = 1:
- pathwise: autograd of the sample mean;
- finite difference: central difference of the sample mean on the same
  paths (common random numbers), which does see the jumps, at two steps h.
Standard errors come from the spread over paths. Without the crowd the two
estimates must agree, which checks the finite differences.

    python crowd_gradient.py
"""

import argparse
from dataclasses import replace

import torch

from market import BatesConfig, market_noise, simulate, _norm_cdf
from baselines import read_obs, time_left, _actions


def scaled_delta(cfg, theta):
    """theta (B,) times the BS book delta at sigma0, stock only, every date."""
    def fn(obs):
        k, s, _, _, _ = read_obs(obs, cfg)
        sd = cfg.sigma0 * torch.sqrt(time_left(k, cfg, s))
        d = sum(_norm_cdf(torch.log(s / kk) / sd + 0.5 * sd)
                for kk in (cfg.strike_call, cfg.strike_put)) - 1
        on = torch.ones_like(s, dtype=torch.bool)
        return _actions(theta * d, on, torch.zeros_like(s), ~on)
    return fn


def losses(noise, theta, cfg):
    return simulate(noise, scaled_delta(cfg, theta), 0.0, cfg)


def path_gradients(noise, cfg, chunk):
    """Loss and dL/dtheta on each path at theta = 1."""
    out_l, out_g = [], []
    for z in noise.split(chunk):
        theta = torch.ones(z.shape[0], dtype=z.dtype, requires_grad=True)
        loss = losses(z, theta, cfg)
        g, = torch.autograd.grad(loss.sum(), theta)
        out_l.append(loss.detach())
        out_g.append(g)
    return torch.cat(out_l), torch.cat(out_g)


def mean_se(x):
    return x.mean().item(), (x.std() / x.numel() ** 0.5).item()


def check(cfg, noise, steps, chunk):
    """{quantity: (pathwise, [finite difference for each h], [finite
    difference minus pathwise for each h])}, each a (mean, standard error);
    the gaps are paired on the paths."""
    loss, g = path_gradients(noise, cfg, chunk)
    w = torch.quantile(loss, cfg.alpha)
    tail = (loss > w).to(loss.dtype) / (1 - cfg.alpha)
    ru = lambda l: w + (l - w).clamp_min(0) / (1 - cfg.alpha)
    pw = {'E[L]': g, 'CVaR': tail * g}
    fd = {'E[L]': [], 'CVaR': []}
    with torch.no_grad():
        for h in steps:
            up = torch.cat([losses(z, 1 + h, cfg) for z in noise.split(chunk)])
            dn = torch.cat([losses(z, 1 - h, cfg) for z in noise.split(chunk)])
            fd['E[L]'].append((up - dn) / (2 * h))
            fd['CVaR'].append((ru(up) - ru(dn)) / (2 * h))
    return {q: (mean_se(pw[q]), [mean_se(d) for d in fd[q]], [mean_se(d - pw[q]) for d in fd[q]])
            for q in pw}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', type=int, default=200_000)
    p.add_argument('--chunk', type=int, default=25_000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=float, nargs='+', default=[0.05, 0.02])
    args = p.parse_args()

    base = BatesConfig()
    off = dict(arrival_base=0.0, arrival_move=0.0)
    strong = dict(crowd_books=12.0, dealers=16, dealer_band_low=0.001, arrival_base=3.0,
                  arrival_move=2.0, arrival_follow=0.9)
    cases = [('no crowd', replace(base, crowd_books=0.0, **off)),
             ('dealers', replace(base, **off)),
             ('random orders', replace(base, crowd_books=0.0)),
             ('both (default)', base),
             ('both, strong', replace(base, **strong))]
    noise = market_noise(args.paths, base, generator=torch.Generator().manual_seed(args.seed))
    noise = noise.double()
    hs = ' | '.join(f'finite diff, h = {h:g}' for h in args.steps)
    gaps = ' | '.join(f'gap, h = {h:g}' for h in args.steps)
    print(f'{args.paths} paths, policy theta x BS delta, derivatives at theta = 1; '
          f'gap = finite difference - pathwise, paired on the paths\n')
    print(f'| crowd | derivative | pathwise | {hs} | {gaps} |')
    print('|---|---|---|' + '---|' * 2 * len(args.steps))
    fmt = lambda v: f'{v[0]:.3f} ± {v[1]:.3f}'
    for name, cfg in cases:
        for q, (pw, fd, gap) in check(cfg, noise, args.steps, args.chunk).items():
            label = 'dCVaR/dtheta' if q == 'CVaR' else 'dE[L]/dtheta'
            print(f'| {name} | {label} | {fmt(pw)} | ' + ' | '.join(map(fmt, fd + gap)) + ' |',
                  flush=True)

if __name__ == '__main__':
    main()
