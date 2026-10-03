"""Train the CVaR hedger with PufferLib 3.0's PPO (PuffeRL) on the C env.

The env (env.py, hedge.h behind PufferLib's native binding) runs 32-step
episodes, so bptt_horizon = 32 puts exactly one episode in each segment.
Same PPO settings as the earlier parts: Adam, gamma = 1, uniform segment
sampling (prio_alpha = 0, i.e. plain PPO).

    python setup.py build_ext --inplace && python train_ppo.py
"""

import argparse
import json
import os
import time

import torch

import pufferlib.pufferl as pufferl

from market import BatesConfig
from env import BatesEnv, EPISODE
from policy import HedgePolicy


def ppo_config(args, num_agents):
    batch = num_agents * EPISODE
    return dict(
        env='deep_hedging_bates', seed=args.seed, torch_deterministic=True,
        device=args.device, cpu_offload=False, use_rnn=False, compile=False,
        compile_mode='default', compile_fullgraph=False, precision='float32',
        optimizer='adam', learning_rate=args.lr, anneal_lr=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        total_timesteps=args.timesteps,
        batch_size=batch, bptt_horizon=EPISODE,
        minibatch_size=batch // args.minibatches, max_minibatch_size=batch,
        update_epochs=args.update_epochs,
        gamma=1.0, gae_lambda=args.gae_lambda,
        clip_coef=0.2, vf_coef=0.5, vf_clip_coef=10.0, ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.0, prio_beta0=1.0,
        data_dir=args.ckpt_dir, checkpoint_interval=10**9,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=200_000_000)
    p.add_argument('--num-agents', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--minibatches', type=int, default=8)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--ent-coef', type=float, default=0.0)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--no-shaping', action='store_true')
    p.add_argument('--w-eta', type=float, default=0.01)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/ppo.pt')
    p.add_argument('--ckpt-dir', default='results/ppo_ckpt')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count())))
    cfg = BatesConfig()
    env = BatesEnv(cfg, num_agents=args.num_agents, shaping=not args.no_shaping,
                   w_eta=args.w_eta, seed=args.seed)
    policy = HedgePolicy(hidden=args.hidden).to(args.device)

    pufferl.PuffeRL.print_dashboard = lambda self, *a, **k: None
    trainer = pufferl.PuffeRL(ppo_config(args, args.num_agents), env, policy)

    history, t0, w = [], time.time(), float('nan')
    while trainer.global_step < args.timesteps:
        trainer.evaluate()
        logs = trainer.train()
        if logs and trainer.epoch % 5 == 0:
            row = {'step': trainer.global_step, 'epoch': trainer.epoch, 'time': time.time() - t0}
            row.update({k.split('/')[-1]: float(v) for k, v in logs.items()
                        if k.startswith(('environment/', 'losses/'))})
            w = row.get('w', w)
            history.append(row)
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    trainer.utilization.stop()
    env.close()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'policy': policy.cpu().state_dict(), 'cfg': cfg.to_dict(), 'hidden': args.hidden,
                'w_center': w, 'history': history, 'args': vars(args),
                'wall_time': time.time() - t0, 'steps': trainer.global_step,
                'device': args.device}, args.out)


if __name__ == '__main__':
    main()
