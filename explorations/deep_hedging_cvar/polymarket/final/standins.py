"""Stand-in policies (method kind "callable") for dry runs and tests. Never evaluated in a real plan
unless someone puts them in train/selected.json, which make_plan.py reports as a stand-in."""
from __future__ import annotations

import numpy as np


def jittered_fair_value(entry, ctx):
    """A fair-value quoter whose parameters depend on the seed: a cheap way to get 5 "RL seeds"
    that differ, so the seed-mean / seed-spread code paths have something to chew on."""
    import baselines as bl
    rng = np.random.default_rng(int(entry.get("seed", 0)) + 1000)
    params = dict(bl.DEFAULTS["fair_value"])
    params["spread_mult"] = float(params["spread_mult"] * np.exp(rng.normal(0, 0.2)))
    params["k_skew"] = float(params["k_skew"] * np.exp(rng.normal(0, 0.5)))
    return bl.make("fair_value", ctx["spec"](), params)


def always_join(entry, ctx):
    """Join both touches at size index 0 (test stand-in)."""
    def fn(obs):
        a = np.full((np.asarray(obs).shape[0], 3), 3, dtype=np.int32)
        a[:, 2] = 0
        return a
    return fn
