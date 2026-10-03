"""Train the CVaR hedger with PufferLib's PPO (PuffeRL) on the Cython env."""

import argparse
import json
import os
import time

import torch

import pufferlib.pufferl as pufferl

from market import ImpactConfig
from env import ImpactHedgingEnv
from policy import ImpactHedgePolicy


def ppo_config(args, horizon, num_agents):
    """Same PPO settings as ../train_ppo.py."""
    batch = num_agents * horizon
    return dict(
        env='deep_hedging_cvar_impact', seed=args.seed, torch_deterministic=True,
        device='cpu', cpu_offload=False, use_rnn=False, compile=False,
        compile_mode='default', compile_fullgraph=False, precision='float32',
        optimizer='adam', learning_rate=args.lr, anneal_lr=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        total_timesteps=args.timesteps,
        batch_size=batch, bptt_horizon=horizon,
        minibatch_size=batch // args.minibatches, max_minibatch_size=batch,
        update_epochs=args.update_epochs,
        gamma=1.0, gae_lambda=args.gae_lambda,
        clip_coef=0.2, vf_coef=0.5, vf_clip_coef=10.0, ent_coef=0.0,
        max_grad_norm=0.5,
        vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.0, prio_beta0=1.0,
        data_dir=args.ckpt_dir, checkpoint_interval=10**9,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=120_000_000)
    p.add_argument('--num-agents', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--minibatches', type=int, default=8)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--no-shaping', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/ppo.pt')
    p.add_argument('--ckpt-dir', default='results/ppo_ckpt')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    cfg = ImpactConfig()
    env = ImpactHedgingEnv(cfg, num_agents=args.num_agents,
                           shaping=not args.no_shaping, seed=args.seed)
    policy = ImpactHedgePolicy()
    horizon = cfg.n_steps + 1  # one episode in each PufferLib segment

    pufferl.PuffeRL.print_dashboard = lambda self, *a, **k: None
    trainer = pufferl.PuffeRL(ppo_config(args, horizon, args.num_agents), env, policy)

    history, t0 = [], time.time()
    while trainer.global_step < args.timesteps:
        trainer.evaluate()
        logs = trainer.train()
        if logs and trainer.epoch % 5 == 0:
            row = {'step': trainer.global_step, 'epoch': trainer.epoch,
                   'time': time.time() - t0,
                   'std_target': policy.logstd[0, 0].exp().item(),
                   'std_signal': policy.logstd[0, 1].exp().item()}
            row.update({k.split('/')[-1]: float(v) for k, v in logs.items()
                        if k.startswith(('environment/', 'losses/', 'performance/'))})
            history.append(row)
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    trainer.utilization.stop()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'policy': policy.state_dict(), 'cfg': cfg.to_dict(),
                'w_center': env.w_center, 'history': history, 'args': vars(args),
                'wall_time': time.time() - t0, 'steps': trainer.global_step}, args.out)


if __name__ == '__main__':
    main()
