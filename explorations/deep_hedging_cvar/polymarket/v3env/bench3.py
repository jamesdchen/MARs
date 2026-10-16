"""Throughput of mm3 against v2 (../mm.h) on the same data, single thread.

    python bench3.py                      # bare cores (ctypes harnesses) + PufferLib 3.0 bindings
    python bench3.py --seconds 5 --repeats 5 --out bench3.json

Rows are interleaved (v2, v3, v2, v3, ...) so both see the same machine load, and every number is
the best of --repeats. Thread CPU time is reported next to wall time: on a shared machine the CPU
rate is the steadier one. Cases:
  core v2 / core v3 (options off)    the crypto data/build train split, random actions, central, L 1
  core v3 obs 80                      + the 29 extra observation slots
  core v3 obs 80 + liq + size_mode 1  + the reward hook (every row matched; no quiet-row skipping)
  binding v2 / binding v3             env-only through PufferLib 3.0's native binding (env.py / env3.py)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PM = HERE.parent
sys.path.insert(0, str(HERE))
from reference3 import Config3  # noqa: E402

FLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-std=c99", "-D_POSIX_C_SOURCE=200809L", "-shared", "-fPIC"]


def build(src, out):
    subprocess.run([os.environ.get("CC", "cc"), *FLAGS, "-o", str(out), str(src), "-lm"], check=True)
    lib = ctypes.CDLL(str(out))
    return lib


def core(lib, fn, vals, n, seconds):
    getattr(lib, fn).restype = ctypes.c_double
    out = np.zeros(64)
    cpu = ctypes.c_double(0.0)
    steps = max(1, int(seconds * 4e6 / n))
    p = vals.ctypes.data_as(ctypes.c_void_p)
    w = getattr(lib, fn)(p, n, steps, ctypes.c_uint64(1), out.ctypes.data_as(ctypes.c_void_p), ctypes.byref(cpu))
    return w, cpu.value


CHILD = r"""
import sys, time, json, numpy as np
sys.path.insert(0, %(pm)r); sys.path.insert(0, %(here)r)
kind, n, seconds, kw = %(kind)r, %(n)d, %(seconds)f, json.loads(%(kw)r)
if kind == "v2":
    from env import MmEnv as E
else:
    from env3 import MmEnv3 as E
env = E(num_envs=n, seed=0, log_interval=10**9, **kw)
env.reset(0)
rng = np.random.default_rng(0)
acts = np.stack([rng.integers(0, 7, (64, n)), rng.integers(0, 7, (64, n)), rng.integers(0, 3, (64, n))], -1).astype(np.int32)
for i in range(200):
    env.step(acts[i & 63])
k, t0, c0 = 0, time.perf_counter(), time.thread_time()
while time.perf_counter() - t0 < seconds:
    for _ in range(50):
        env.step(acts[k & 63]); k += 1
print(json.dumps([k * n / (time.perf_counter() - t0), k * n / (time.thread_time() - c0)]))
"""


def binding(kind, n, seconds, kw):
    code = CHILD % dict(pm=str(PM), here=str(HERE), kind=kind, n=n, seconds=seconds, kw=json.dumps(kw))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(PM))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-binding", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    n = args.num_envs
    v2_names = Config3.names()[:27]
    with tempfile.TemporaryDirectory() as tmp:
        l2 = build(PM / "mm_harness.c", Path(tmp) / "v2.so")
        l3 = build(HERE / "harness3.c", Path(tmp) / "v3.so")
        err = ctypes.create_string_buffer(512)
        data = str(PM / "data" / "build").encode()
        assert l2.mm_h_open(data, err, 512) == 0, err.value
        assert l3.m3_h_open(data, err, 512) == 0, err.value
        base = Config3(split=0, fill_model=1, latency=1)
        v2_vals = base.values()[:27].copy()
        v2_vals[v2_names.index("len_mask")] = 7
        cases = {
            "core v2 (mm.h)": (l2, "mm_h_bench", v2_vals),
            "core v3 (options off)": (l3, "m3_h_bench", base.values()),
            "core v3 obs 80": (l3, "m3_h_bench", Config3(obs_size=80).values()),
            "core v3 obs 80 + liq + size_mode 1": (l3, "m3_h_bench", Config3(obs_size=80, liq_rewards=1, size_mode=1,
                                                                             Q=300).values()),
        }
        res = {k: [0.0, 0.0] for k in cases}
        for _ in range(args.repeats):
            for k, (lib, fn, vals) in cases.items():
                w, c = core(lib, fn, np.ascontiguousarray(vals, np.float64), n, args.seconds)
                res[k] = [max(res[k][0], w), max(res[k][1], c)]
                print(f"{k:40s} wall {w / 1e6:6.2f} M/s   cpu {c / 1e6:6.2f} M/s", flush=True)
    if not args.skip_binding:
        bcases = {"binding v2 (env.py)": ("v2", {}), "binding v3 (env3.py, options off)": ("v3", {}),
                  "binding v3 obs 80 + liq + size_mode 1": ("v3", dict(obs_size=80, liq_rewards=1, size_mode=1,
                                                                       Q=300))}
        for k in bcases:
            res[k] = [0.0, 0.0]
        for _ in range(args.repeats):
            for k, (kind, kw) in bcases.items():
                w, c = binding(kind, n, args.seconds, kw)
                res[k] = [max(res[k][0], w), max(res[k][1], c)]
                print(f"{k:40s} wall {w / 1e6:6.2f} M/s   cpu {c / 1e6:6.2f} M/s", flush=True)
    print("\nbest of", args.repeats, f"({n} envs, single thread, load {os.getloadavg()[0]:.1f}):")
    for k, (w, c) in res.items():
        print(f"  {k:40s} wall {w / 1e6:6.2f} M steps/s   cpu {c / 1e6:6.2f} M steps/s")
    r2, r3 = res["core v2 (mm.h)"], res["core v3 (options off)"]
    print(f"  v3 / v2 (core, cpu): {r3[1] / r2[1]:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"num_envs": n, "seconds": args.seconds, "repeats": args.repeats,
                                              "load": os.getloadavg(), "results": res}, indent=1))


if __name__ == "__main__":
    main()
