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
"""

import argparse
import json
import os
import time

import torch

from market import ImpactConfig, fundamental_noise, simulate, ru_objective, summarize
from policy import ImpactHedgePolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', choices=['hard', 'ste', 'sigmoid'], required=True)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16384)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temp', type=float, default=0.1, help='STE temperature / sigmoid start')
    p.add_argument('--temp-end', type=float, default=0.01, help='sigmoid end temperature')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    out = args.out or f'results/pathwise_{args.gate}.pt'

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    cfg = ImpactConfig()
    policy = ImpactHedgePolicy().to(args.device)
    w = torch.nn.Parameter(torch.tensor(0.3, device=args.device))  # in units of cfg.scale
    opt = torch.optim.Adam(list(policy.actor.parameters()) + [w], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    gen = torch.Generator().manual_seed(args.seed)

    history, t0 = [], time.time()
    for it in range(args.iters):
        temp = args.temp
        if args.gate == 'sigmoid':
            temp = args.temp * (args.temp_end / args.temp) ** (it / max(1, args.iters - 1))
        z = fundamental_noise(args.batch, cfg, gen, args.device)
        loss = simulate(z, policy.mean_action, w * cfg.scale, cfg, gate_mode=args.gate, temp=temp)
        obj = ru_objective(loss, w * cfg.scale, cfg.alpha)
        opt.zero_grad()
        obj.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == args.iters - 1:
            with torch.no_grad():
                exact, rec = simulate(z[:4096], policy.mean_action, w * cfg.scale, cfg,
                                      record=True)
            row = {'iter': it, 'objective': obj.item(), 'w': w.item() * cfg.scale,
                   'temp': temp, 'time': time.time() - t0,
                   'trades': rec['trade'].float().sum(1).mean().item(),
                   **{f'train_{k}': v for k, v in summarize(loss.detach(), cfg.alpha).items()},
                   **summarize(exact, cfg.alpha)}
            history.append(row)
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'actor': policy.actor.cpu().state_dict(), 'w': w.item() * cfg.scale,
                'cfg': cfg.to_dict(), 'gate': args.gate, 'history': history,
                'wall_time': time.time() - t0, 'device': args.device,
                'steps': args.iters * args.batch * cfg.n_steps}, out)


if __name__ == '__main__':
    main()
