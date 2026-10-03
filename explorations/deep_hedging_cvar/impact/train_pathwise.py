"""Pathwise deep hedging baselines under impact and fixed costs.

Backpropagates the Rockafellar-Uryasev objective through market.simulate,
jointly over the network and w, as in ../train_pathwise.py. Impact is
smooth, so its gradient is exact. The trade decision 1{signal > 0} is not:
its gradient is zero almost everywhere. Three ways to handle it:

    --gate hard     the exact model, so the trade signal never learns
    --gate ste      exact forward pass, sigmoid gradient (straight-through)
    --gate sigmoid  train on the relaxed model in which the agent trades a
                    sigmoid(signal / temp) fraction of the way and pays that
                    fraction of the fee; temp is annealed geometrically

Every variant is scored on the exact model (hard gate) in evaluate.py.
train() is importable, so sweep_pathwise.py can tune it with the same code.
"""

import argparse
import json
import os
import time

import torch

from market import ImpactConfig, fundamental_noise, simulate, ru_objective, summarize
from policy import ImpactHedgePolicy


def train(gate, iters=2000, batch=16384, lr=1e-3, temp=0.1, temp_end=0.01, hidden=64,
          seed=0, device='cpu', log=print):
    """Train one pathwise hedger and return its checkpoint dict.

    The dict holds the actor's state_dict (on the CPU), w in price units, the
    logged history, wall time and simulated steps, and the arguments, so
    main() and sweep_pathwise.py save the same keys. log receives one JSON
    line every 100 iterations; pass None to train silently.
    """
    args = dict(gate=gate, iters=iters, batch=batch, lr=lr, temp=temp, temp_end=temp_end,
                hidden=hidden, seed=seed, device=device)
    torch.manual_seed(seed)
    cfg = ImpactConfig()
    policy = ImpactHedgePolicy(hidden=hidden).to(device)
    w = torch.nn.Parameter(torch.tensor(0.3, device=device))  # in units of cfg.scale
    opt = torch.optim.Adam(list(policy.actor.parameters()) + [w], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    gen = torch.Generator().manual_seed(seed)

    history, t0 = [], time.time()
    for it in range(iters):
        t = temp
        if gate == 'sigmoid':
            t = temp * (temp_end / temp) ** (it / max(1, iters - 1))
        z = fundamental_noise(batch, cfg, gen, device)
        loss = simulate(z, policy.mean_action, w * cfg.scale, cfg, gate_mode=gate, temp=t)
        obj = ru_objective(loss, w * cfg.scale, cfg.alpha)
        opt.zero_grad()
        obj.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == iters - 1:
            with torch.no_grad():
                exact, rec = simulate(z[:4096], policy.mean_action, w * cfg.scale, cfg,
                                      record=True)
            row = {'iter': it, 'objective': obj.item(), 'w': w.item() * cfg.scale,
                   'temp': t, 'time': time.time() - t0,
                   'trades': rec['trade'].float().sum(1).mean().item(),
                   **{f'train_{k}': v for k, v in summarize(loss.detach(), cfg.alpha).items()},
                   **summarize(exact, cfg.alpha)}
            history.append(row)
            if log is not None:
                log(json.dumps({k: round(v, 4) for k, v in row.items()}))

    return {'actor': policy.actor.cpu().state_dict(), 'w': w.item() * cfg.scale,
            'cfg': cfg.to_dict(), 'gate': gate, 'history': history,
            'wall_time': time.time() - t0, 'device': device,
            'steps': iters * batch * cfg.n_steps, 'hidden': hidden, 'args': args}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', choices=['hard', 'ste', 'sigmoid'], required=True)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16384)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temp', type=float, default=0.1, help='STE temperature / sigmoid start')
    p.add_argument('--temp-end', type=float, default=0.01, help='sigmoid end temperature')
    p.add_argument('--hidden', type=int, default=64, help='actor hidden width')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    out = args.out or f'results/pathwise_{args.gate}.pt'

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    ckpt = train(args.gate, iters=args.iters, batch=args.batch, lr=args.lr, temp=args.temp,
                 temp_end=args.temp_end, hidden=args.hidden, seed=args.seed,
                 device=args.device, log=lambda line: print(line, flush=True))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(ckpt, out)


if __name__ == '__main__':
    main()
