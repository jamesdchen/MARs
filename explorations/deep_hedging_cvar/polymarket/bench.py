"""Throughput of the market-making replay env.

1. env-only, single thread, through PufferLib 3.0's native binding (env.py: one Python call
   per vectorized step of all envs; every env step in C), random actions;
2. the bare C core (mm_harness.c via ctypes, no Python in the loop), same settings;
3. end to end: PufferLib 3.0 PPO (PuffeRL, pufferlib.models.Default MLP) on CPU, rollout plus
   update, reported as agent steps per second of wall time.

    python bench.py                       # all three, 4096 envs
    python bench.py --skip-ppo --seconds 20
    python bench.py --fill-model 2 --dt 1

Numbers depend on other load on the machine; run it on an idle box (or take the max of a few
repeats, --repeats).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def env_kwargs(args):
    return dict(fill_model=args.fill_model, latency=args.latency, dt=args.dt, split=0)


def bench_env(args):
    from env import MmEnv
    env = MmEnv(num_envs=args.num_envs, seed=0, log_interval=10**9, **env_kwargs(args))
    env.reset(0)
    rng = np.random.default_rng(0)
    acts = np.stack([rng.integers(0, 7, (64, args.num_envs)), rng.integers(0, 7, (64, args.num_envs)),
                     rng.integers(0, 3, (64, args.num_envs))], -1).astype(np.int32)
    for i in range(200):                     # warm the page cache and the caches
        env.step(acts[i & 63])
    best = 0.0
    for _ in range(args.repeats):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            for _ in range(50):
                env.step(acts[n & 63])
                n += 1
        best = max(best, n * args.num_envs / (time.perf_counter() - t0))
    env.close()
    return best


def bench_core(args):
    import env as E
    with tempfile.TemporaryDirectory() as tmp:
        lib = os.path.join(tmp, "libmm_bench.so")
        subprocess.run([os.environ.get("CC", "cc"), "-O3", "-march=native", "-ffp-contract=off", "-std=c99",
                        "-D_POSIX_C_SOURCE=200809L", "-shared", "-fPIC", "-o", lib, str(HERE / "mm_harness.c"),
                        "-lm"], check=True)
        lib = ctypes.CDLL(lib)
        err = ctypes.create_string_buffer(512)
        assert lib.mm_h_open(str(E.DATA_DIR).encode(), err, 512) == 0, err.value
        lib.mm_h_bench.restype = ctypes.c_double
        p = np.array(list(E.params(**env_kwargs(args)).values()))
        out = np.zeros(64)
        steps = max(1, int(args.seconds * 6e6 / args.num_envs))
        best = 0.0
        for _ in range(args.repeats):
            best = max(best, lib.mm_h_bench(p.ctypes.data_as(ctypes.c_void_p), args.num_envs, steps,
                                            ctypes.c_uint64(1), out.ctypes.data_as(ctypes.c_void_p)))
        return best


def bench_ppo(args):
    import torch
    import pufferlib.models
    import pufferlib.pufferl as pufferl
    from env import MmEnv
    torch.set_num_threads(args.threads)
    env = MmEnv(num_envs=args.num_envs, seed=0, log_interval=10**9, **env_kwargs(args))
    policy = pufferlib.models.Default(env, hidden_size=args.hidden)
    batch = args.num_envs * args.horizon
    cfg = dict(
        env="polymarket_mm", seed=0, torch_deterministic=False, device="cpu", cpu_offload=False,
        use_rnn=False, compile=False, compile_mode="default", compile_fullgraph=False, precision="float32",
        optimizer="adam", learning_rate=3e-4, anneal_lr=False, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        total_timesteps=batch * (args.ppo_epochs + 2) * 10, batch_size=batch, bptt_horizon=args.horizon,
        minibatch_size=args.minibatch, max_minibatch_size=args.minibatch, update_epochs=args.update_epochs,
        gamma=0.995, gae_lambda=0.95, clip_coef=0.2, vf_coef=0.5, vf_clip_coef=0.2, ent_coef=0.01,
        max_grad_norm=0.5, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.0, prio_beta0=1.0,
        data_dir=tempfile.mkdtemp(), checkpoint_interval=10**9,
    )
    pufferl.PuffeRL.print_dashboard = lambda self, *a, **k: None
    trainer = pufferl.PuffeRL(cfg, env, policy)
    trainer.evaluate()                       # warm-up epoch
    trainer.train()
    t_eval = t_train = 0.0
    s0 = trainer.global_step
    for _ in range(args.ppo_epochs):
        a = time.perf_counter()
        trainer.evaluate()
        b = time.perf_counter()
        trainer.train()
        c = time.perf_counter()
        t_eval += b - a
        t_train += c - b
    steps = trainer.global_step - s0
    trainer.utilization.stop()
    env.close()
    return dict(sps_end_to_end=steps / (t_eval + t_train), sps_rollout=steps / t_eval,
                rollout_share=t_eval / (t_eval + t_train), steps=steps, threads=args.threads,
                hidden=args.hidden, horizon=args.horizon, minibatch=args.minibatch,
                update_epochs=args.update_epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fill-model", type=int, default=1)
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--dt", type=int, default=0, help="0 = per contract length (1/1/5 s)")
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--minibatch", type=int, default=32768)
    ap.add_argument("--update-epochs", type=int, default=1)
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()
    res = {"num_envs": args.num_envs, "fill_model": args.fill_model, "latency": args.latency, "dt": args.dt,
           "load_avg_1m_at_start": os.getloadavg()[0]}
    res["env_only_binding_sps"] = bench_env(args)
    print(f"env-only, PufferLib 3.0 binding, 1 thread, {args.num_envs} envs: "
          f"{res['env_only_binding_sps'] / 1e6:.2f}M steps/s", flush=True)
    res["env_only_core_sps"] = bench_core(args)
    print(f"env-only, bare C core (ctypes harness), 1 thread:        "
          f"{res['env_only_core_sps'] / 1e6:.2f}M steps/s", flush=True)
    if not args.skip_ppo:
        res["ppo"] = bench_ppo(args)
        p = res["ppo"]
        print(f"PPO end to end (PufferLib 3.0, CPU, {p['threads']} torch threads, MLP {p['hidden']}): "
              f"{p['sps_end_to_end'] / 1e3:.0f}k steps/s (rollout alone {p['sps_rollout'] / 1e3:.0f}k steps/s, "
              f"{100 * p['rollout_share']:.0f}% of wall time)", flush=True)
    res["load_avg_1m_at_end"] = os.getloadavg()[0]
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
