"""Classic deep hedging baseline: backpropagate CVaR through simulated paths.

Minimizes the Rockafellar-Uryasev objective jointly over the network and w,
as in Buehler et al. (2019). The network sees the same observation as the
PPO agent, with w set to the learned threshold.
"""

import argparse
import json
import os
import time

import torch

from hedging import MarketConfig, simulate_gbm, hedge_loss, ru_objective, summarize
from policy import HedgePolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--batch', type=int, default=16384)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--cost', type=float, default=MarketConfig.cost)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/pathwise.pt')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    cfg = MarketConfig(cost=args.cost)
    policy = HedgePolicy()
    w = torch.nn.Parameter(torch.tensor(0.2))  # in units of cfg.scale
    opt = torch.optim.Adam(list(policy.actor.parameters()) + [w], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    gen = torch.Generator().manual_seed(args.seed)

    history, t0 = [], time.time()
    for it in range(args.iters):
        paths = simulate_gbm(args.batch, cfg, gen)
        loss = hedge_loss(paths, policy.mean_action, w * cfg.scale, cfg)
        obj = ru_objective(loss, w * cfg.scale, cfg.alpha)
        opt.zero_grad()
        obj.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == args.iters - 1:
            row = {'iter': it, 'objective': obj.item(), 'w': w.item() * cfg.scale,
                   'time': time.time() - t0, **summarize(loss.detach(), cfg.alpha)}
            history.append(row)
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'actor': policy.actor.state_dict(), 'w': w.item() * cfg.scale,
                'cfg': cfg.to_dict(), 'history': history}, args.out)


if __name__ == '__main__':
    main()
