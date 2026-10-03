"""Pathwise deep hedging for the Bates book, with a hybrid gradient estimator
for the trade decisions.

Backpropagates the Rockafellar-Uryasev objective
J(theta, w) = w + E[(L - w)^+] / (1 - alpha) through market.simulate,
jointly over the network and w, as in ../impact/train_pathwise.py (Buehler
et al. 2019). The market noise, jumps included, does not depend on the
policy, and impact and costs are piecewise smooth in the clamped targets,
so the pathwise gradient through the targets is the usual one. The two
trade decisions 1{signal > 0} on each date are not: with the fixed fee the
loss jumps when a decision flips, and the indicator's gradient is zero
almost everywhere. Four ways to handle them:

    --gate hard     the exact model, so the signals never learn
    --gate ste      exact forward pass, sigmoid(signal / temp) gradient
                    (straight-through, Bengio et al. 2013); temp fixed
    --gate sigmoid  train on the relaxed model that trades a sigmoid fraction
                    of the way and pays that fraction of the fee; temp
                    annealed. Part 2 showed that this fails on the exact model
    --gate hybrid   unbiased for a stochastic gate (below); temp annealed

Hybrid estimator. In training each decision is sampled,
g ~ Bernoulli(p) with p = sigmoid(signal / temp), so the expected objective
is smooth in the signals. Let c_i = w + (L_i - w)^+ / (1 - alpha) be path
i's contribution to the RU objective and l_i = sum_k log p(g_ik) the summed
log-probability of path i's sampled decisions (both instruments, every
date). Each p depends on theta through the network and through the state it
observes, which earlier targets moved. The surrogate

    S = mean_i c_i + mean_i (stop_grad(c_i) - b_i) l_i

satisfies E[grad S] = grad E[c_i], the gradient of the RU objective of the
stochastic policy, for the network and for w. The first term is the
pathwise derivative with the sampled decisions held fixed, the second is
the score-function (likelihood-ratio) term for the decisions: Schulman,
Heess, Weber & Abbeel (2015), Gradient estimation using stochastic
computation graphs; Williams (1992). The only cost is terminal, so c_i is
also the cost downstream of every decision.

Baseline. Most of the spread of c_i comes from the market path, not from
the decisions, so a constant baseline leaves the score term very noisy (at
initialization, temp 0.1 and batch 2048, the standard deviation of the
gradient for the signal biases is about ten times its mean). The batch of
B paths is therefore K = --samples independent decision samples on each of
B / K market paths, and b_i is the leave-one-out mean of c over the other
K - 1 samples on path i's market path (Kool, van Hoof & Welling 2019). It
depends on the market noise and on other samples' decisions, never on path
i's, so it adds no bias; with K = 4 that standard deviation falls about
fiftyfold. With K = 1, b_i is the mean of c over the other paths of the
batch. test_pathwise.py checks unbiasedness against exact enumeration of
the decisions on a 2-date book.

The stochastic policy is only a device for training. temp is annealed
geometrically from --temp to --temp-end so the decisions become nearly
deterministic, and the reported policy is the deterministic one: trade iff
signal > 0, the mode of each Bernoulli. The two agree once the network has
sharpened its signals as temp falls; in short runs many signals can stay
within a few temp of 0, leaving the deterministic policy worse than the
stochastic one it was trained as. Every variant is logged, validated and
scored on the exact model (hard gate), so the sweep picks schedules for
which the deterministic policy is good. The log uses a fixed monitoring set
of 4096 paths, drawn first from the training generator; for the hybrid gate
it also reports the stochastic policy at the current temp on those paths
(sampled_*), which keeps that gap visible.

train() is importable, so sweep_pathwise.py can tune it with the same code.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from market import BatesConfig, market_noise, simulate, ru_objective, summarize
from policy import HedgePolicy

VAL_SEED, TEST_SEED = 999, 12345
MONITOR_PATHS = 4096


class SampledGates:
    """gates function for market.simulate: g ~ Bernoulli(sigmoid(signal / temp))
    for the stock (action 1) and the swap (action 3), decided by uniforms u
    (B, n_steps, 2). logp accumulates, for each path, the log-probability of
    the sampled decisions; the decisions themselves carry no gradient."""

    def __init__(self, u, temp):
        self.u, self.temp, self.logp = u, temp, 0.0

    def __call__(self, act, k):
        out = []
        for j, col in enumerate((1, 3)):
            logit = act[:, col] / self.temp
            g = (self.u[:, k, j] < torch.sigmoid(logit)).to(act.dtype)
            self.logp = self.logp + g * F.logsigmoid(logit) + (1 - g) * F.logsigmoid(-logit)
            out.append(g)
        return tuple(out)


def hybrid_surrogate(loss, logp, w, alpha, samples=1):
    """(surrogate S, RU objective mean_i c_i); see the module docstring.
    loss and logp are (B,), laid out as `samples` blocks of the same B /
    samples market paths."""
    c = w + (loss - w).clamp_min(0) / (1 - alpha)
    cd = c.detach().view(samples, -1)
    if samples > 1:
        b = (cd.sum(0) - cd) / (samples - 1)
    else:
        b = (cd.sum() - cd) / (cd.numel() - 1)
    return c.mean() + ((cd - b).view(-1) * logp).mean(), c.mean()


def monitor_noise(seed, cfg, device='cpu'):
    """The monitoring paths of a run with this seed (the first draw of its
    training generator)."""
    return market_noise(MONITOR_PATHS, cfg, torch.Generator().manual_seed(seed), device)


def train(gate, iters=2000, batch=16384, lr=1e-3, temp=0.1, temp_end=0.01, hidden=64,
          seed=0, device='cpu', log=print, cfg=None, log_every=100, samples=4):
    """Train one pathwise hedger and return its checkpoint dict.

    The dict holds the actor's state_dict (on the CPU), w in price units, the
    logged history, wall time and simulated steps, and the arguments, so
    main() and sweep_pathwise.py save the same keys. log receives one JSON
    line every log_every iterations and after the last one; pass None to
    train silently. cfg defaults to BatesConfig() (tests pass smaller books).
    samples is K of the hybrid gate (batch counts all simulated paths).
    """
    if seed in (VAL_SEED, TEST_SEED):
        raise ValueError(f'seed {seed} would reuse the validation or test noise')
    if gate == 'hybrid' and batch % samples:
        raise ValueError('batch must be a multiple of samples')
    args = dict(gate=gate, iters=iters, batch=batch, lr=lr, temp=temp, temp_end=temp_end,
                hidden=hidden, seed=seed, device=device, samples=samples)
    torch.manual_seed(seed)
    cfg = cfg or BatesConfig()
    policy = HedgePolicy(hidden=hidden).to(device)
    w = torch.nn.Parameter(torch.tensor(0.3, device=device))  # in units of cfg.scale
    opt = torch.optim.Adam(list(policy.actor.parameters()) + [w], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    gen = torch.Generator().manual_seed(seed)
    monitor = market_noise(MONITOR_PATHS, cfg, gen, device)
    if gate == 'hybrid':
        monitor_u = torch.rand(MONITOR_PATHS, cfg.n_steps, 2, generator=gen).to(device)

    history, t0 = [], time.time()
    for it in range(iters):
        t = temp
        if gate in ('sigmoid', 'hybrid'):
            t = temp * (temp_end / temp) ** (it / max(1, iters - 1))
        if gate == 'hybrid':
            z = market_noise(batch // samples, cfg, gen, device).repeat(samples, 1, 1, 1)
            u = torch.rand(batch, cfg.n_steps, 2, generator=gen).to(device)
            gates = SampledGates(u, t)
            loss = simulate(z, policy.mean_action, w * cfg.scale, cfg, gates=gates)
            surrogate, obj = hybrid_surrogate(loss, gates.logp, w * cfg.scale, cfg.alpha,
                                              samples)
        else:
            z = market_noise(batch, cfg, gen, device)
            loss = simulate(z, policy.mean_action, w * cfg.scale, cfg, gate_mode=gate, temp=t)
            surrogate = obj = ru_objective(loss, w * cfg.scale, cfg.alpha)
        opt.zero_grad()
        surrogate.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters - 1:
            with torch.no_grad():
                exact, rec = simulate(monitor, policy.mean_action, w * cfg.scale, cfg,
                                      record=True)
                if gate == 'hybrid':
                    sampled = simulate(monitor, policy.mean_action, w * cfg.scale, cfg,
                                       gates=SampledGates(monitor_u, t))
            row = {'iter': it, 'objective': obj.item(), 'w': w.item() * cfg.scale,
                   'temp': t, 'time': time.time() - t0,
                   'trades_stock': rec['trade_s'].float().sum(1).mean().item(),
                   'trades_swap': rec['trade_y'].float().sum(1).mean().item(),
                   **{f'train_{k}': v for k, v in summarize(loss.detach(), cfg.alpha).items()},
                   **summarize(exact, cfg.alpha)}
            if gate == 'hybrid':
                row.update({f'sampled_{k}': v for k, v in summarize(sampled, cfg.alpha).items()})
            history.append(row)
            if log is not None:
                log(json.dumps({k: round(v, 4) for k, v in row.items()}))

    return {'actor': policy.actor.cpu().state_dict(), 'w': w.item() * cfg.scale,
            'cfg': cfg.to_dict(), 'gate': gate, 'history': history,
            'wall_time': time.time() - t0, 'device': device,
            'steps': iters * batch * cfg.n_steps, 'hidden': hidden, 'args': args}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', choices=['hard', 'ste', 'sigmoid', 'hybrid'], required=True)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16384)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temp', type=float, default=0.1,
                   help='STE temperature / sigmoid and hybrid start')
    p.add_argument('--temp-end', type=float, default=0.01, help='sigmoid and hybrid end')
    p.add_argument('--hidden', type=int, default=64, help='actor hidden width')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--samples', type=int, default=4,
                   help='hybrid: decision samples on each market path')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    out = args.out or f'results/pathwise_{args.gate}.pt'

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    ckpt = train(args.gate, iters=args.iters, batch=args.batch, lr=args.lr, temp=args.temp,
                 temp_end=args.temp_end, hidden=args.hidden, seed=args.seed,
                 device=args.device, log=lambda line: print(line, flush=True),
                 log_every=args.log_every, samples=args.samples)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(ckpt, out)


if __name__ == '__main__':
    main()
