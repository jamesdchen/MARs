"""Per-task executor for the bootstrap-coverage sweep.

When run under hpc-agent, the cluster dispatcher sets per-task env vars:
- RESULT_DIR             : output directory (read by write_metrics; do NOT pass)
- HPC_KW_DISTRIBUTION    : "normal" | "exponential" | "lognormal"
- HPC_KW_SAMPLE_SIZE     : "30" | "100" | "300"
- HPC_KW_SEED            : "0".."49"

The HPC_KW_* prefix + uppercasing is how hpc-agent surfaces the dict that
.hpc/tasks.py's `resolve(i)` returns. `read_kw_env()` strips the prefix and
lowercases.

To test locally without HPC, set the env vars by hand:
    RESULT_DIR=./_local HPC_KW_DISTRIBUTION=normal HPC_KW_SAMPLE_SIZE=30 \\
        HPC_KW_SEED=0 uv run python scripts/run.py

Import boundary: inside an executor that ships to the cluster, only
`hpc_agent.mapreduce.metrics_io` and `hpc_agent.executor_cli` are stable
imports from the hpc_agent package.
"""

from __future__ import annotations

import numpy as np

from hpc_agent.mapreduce.metrics_io import read_kw_env, write_metrics


N_BOOTSTRAP_ITERS = 500
CI_LEVEL = 0.95
BASE_SEED = 42


def draw_zero_mean_sample(
    distribution: str, n: int, rng: np.random.Generator
) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(n)
    if distribution == "exponential":
        return rng.exponential(scale=1.0, size=n) - 1.0
    if distribution == "lognormal":
        sigma = 1.0
        return rng.lognormal(mean=0.0, sigma=sigma, size=n) - np.exp(sigma**2 / 2)
    raise ValueError(f"unknown distribution: {distribution}")


def percentile_bootstrap_ci(
    sample: np.ndarray, n_iter: int, level: float, rng: np.random.Generator
) -> tuple[float, float]:
    n = len(sample)
    boot_means = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = sample[idx].mean()
    alpha = (1.0 - level) / 2.0
    return (
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def main() -> int:
    kw = read_kw_env()
    distribution = kw["distribution"]
    sample_size = int(kw["sample_size"])
    seed = int(kw["seed"])

    rng = np.random.default_rng(BASE_SEED + seed)
    sample = draw_zero_mean_sample(distribution, sample_size, rng)
    lo, hi = percentile_bootstrap_ci(sample, N_BOOTSTRAP_ITERS, CI_LEVEL, rng)
    covered = lo <= 0.0 <= hi

    write_metrics(
        {
            "covered": int(covered),
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_width": hi - lo,
            "sample_mean": float(sample.mean()),
            "distribution": distribution,
            "sample_size": sample_size,
            "seed": seed,
            "n_samples": 1,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
