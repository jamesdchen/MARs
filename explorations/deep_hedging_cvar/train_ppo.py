"""Train the w-conditioned CVaR hedger with PufferLib's PPO (PuffeRL)."""

import argparse
import json
import os
import time

import torch

import pufferlib.pufferl as pufferl

from hedging import MarketConfig
from env import HedgingEnv
from policy import HedgePolicy


def ppo_config(args, horizon, num_agents):
    batch = num_agents * horizon
    return dict(
        env='deep_hedging_cvar', seed=args.seed, torch_deterministic=True,
        device='cpu', cpu_offload=False, use_rnn=False, compile=False,
        compile_mode='default', compile_fullgraph=False, precision='float32',
        optimizer='adam', learning_rate=args.lr, anneal_lr=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        total_timesteps=args.timesteps,
        batch_size=batch, bptt_horizon=horizon,
        minibatch_size=batch // args.minibatches, max_minibatch_size=batch,
        update_epochs=args.update_epochs,
        # Terminal-only reward on a fixed horizon: no discounting.
        gamma=1.0, gae_lambda=args.gae_lambda,
        clip_coef=0.2, vf_coef=0.5, vf_clip_coef=10.0, ent_coef=0.0,
        max_grad_norm=0.5,
        # PufferLib's V-trace clips and prioritized segment sampling. alpha=0
        # makes sampling uniform, which is plain PPO.
        vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.0, prio_beta0=1.0,
        data_dir=args.ckpt_dir, checkpoint_interval=10**9,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=40_000_000)
    p.add_argument('--num-agents', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--minibatches', type=int, default=8)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--cost', type=float, default=MarketConfig.cost)
    p.add_argument('--no-shaping', action='store_true')
    p.add_argument('--w-low', type=float, default=MarketConfig.w_low,
                   help='threshold range in units of sigma sqrt(T) S0; equal ends fix w')
    p.add_argument('--w-high', type=float, default=MarketConfig.w_high)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/ppo.pt')
    p.add_argument('--ckpt-dir', default='results/ppo_ckpt')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    cfg = MarketConfig(cost=args.cost, w_low=args.w_low, w_high=args.w_high)
    env = HedgingEnv(cfg, num_agents=args.num_agents, shaping=not args.no_shaping, seed=args.seed)
    policy = HedgePolicy()
    horizon = cfg.n_steps + 1  # one episode in each PufferLib segment, see env.py

    # Replace the full-screen rich dashboard with a JSON line every 5 epochs.
    pufferl.PuffeRL.print_dashboard = lambda self, *a, **k: None
    trainer = pufferl.PuffeRL(ppo_config(args, horizon, args.num_agents), env, policy)

    history, t0 = [], time.time()
    while trainer.global_step < args.timesteps:
        trainer.evaluate()
        logs = trainer.train()
        if logs and trainer.epoch % 5 == 0:
            row = {'step': trainer.global_step, 'epoch': trainer.epoch,
                   'time': time.time() - t0, 'std': policy.logstd.exp().item()}
            row.update({k.split('/')[-1]: float(v) for k, v in logs.items()
                        if k.startswith(('environment/', 'losses/'))})
            history.append(row)
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    trainer.vecenv.close()
    trainer.utilization.stop()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'cfg': cfg.to_dict(),
                'history': history, 'args': vars(args)}, args.out)


if __name__ == '__main__':
    main()
