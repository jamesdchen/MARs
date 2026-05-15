"""Task fan-out for the bootstrap-coverage sweep.

claude-hpc imports this module and calls `total()` to size the array job,
then `resolve(i)` per task to recover the kwargs for task #i. The agent
writes this file from meta.json's parameter axes; do NOT edit by hand
unless meta.json changes.
"""

import itertools

_DISTRIBUTIONS = ["normal", "exponential", "lognormal"]
_SAMPLE_SIZES = [30, 100, 300]
_SEEDS = list(range(50))

_TASKS = [
    {"distribution": dist, "sample_size": n, "seed": seed}
    for dist, n, seed in itertools.product(_DISTRIBUTIONS, _SAMPLE_SIZES, _SEEDS)
]


def total() -> int:
    return len(_TASKS)


def resolve(i: int) -> dict:
    return _TASKS[i]
