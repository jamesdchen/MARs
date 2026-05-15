"""Per-task executor for the bootstrap-coverage sweep.

Run as:
    uv run python scripts/run.py --task-id <i> --out <result_dir>

Local mode (no HPC): the agent loops over i in [0, total()) sequentially.
Cluster mode: claude-hpc dispatches one process per task across the cluster.
Each process writes its metrics via claude_hpc.mapreduce.metrics_io.write_metrics
so that idempotent skip-on-resubmit and aggregation work without extra glue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make .hpc/tasks.py importable without packaging it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".hpc"))
import tasks  # noqa: E402

from claude_hpc.mapreduce.metrics_io import write_metrics  # type: ignore


N_BOOTSTRAP_ITERS = 500
CI_LEVEL = 0.95
BASE_SEED = 42


def draw_sample(distribution: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(n)
    if distribution == "exponential":
        return rng.exponential(scale=1.0, size=n) - 1.0  # zero-mean
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
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kwargs = tasks.resolve(args.task_id)
    distribution: str = kwargs["distribution"]
    sample_size: int = kwargs["sample_size"]
    seed: int = kwargs["seed"]

    rng = np.random.default_rng(BASE_SEED + seed)
    sample = draw_sample(distribution, sample_size, rng)
    lo, hi = percentile_bootstrap_ci(sample, N_BOOTSTRAP_ITERS, CI_LEVEL, rng)
    true_mean = 0.0  # all distributions above are centered at zero
    covered = lo <= true_mean <= hi

    args.out.mkdir(parents=True, exist_ok=True)
    write_metrics(
        args.out,
        task_id=args.task_id,
        kwargs=kwargs,
        metrics={
            "covered": int(covered),
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_width": hi - lo,
            "sample_mean": float(sample.mean()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
