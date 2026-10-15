"""Adapter layer between the final-test runner and the other components.

Two registries, each keyed by a name stored in TEST_PLAN.json:

POLICY_KINDS: method `kind` -> factory(entry, ctx) -> Policy
-------------------------------------------------------------
A Policy is deterministic and batched:

    pol.reset(B)                          new batch of B episodes (zero recurrent state)
    actions = pol.act(obs, first)         obs float32 (B, OBS); first bool (B,) True on the
                                          first observation of an episode; -> int32 (B, 3)

  kind "baseline"  entry {name, params}: baselines/baselines.py `make(name, spec, params)`,
                   spec = the backend's ObsSpec (feature name -> obs index), fn(obs)->actions.
  kind "ppo3"      entry {path}: a PufferLib 3.0 checkpoint read by train/policy.py
                   `load_checkpoint` (MLPPolicy or recurrent PufferNetPolicy); greedy argmax
                   per head (policy.greedy_actions); recurrent state reset on `first`.
  kind "puffer5"   entry {path, hidden, layers[, obs_size, align]}: a PufferLib 5.0 .bin
                   checkpoint, puffer5/puffernet_mm.py `PufferNetMMPolicy(deterministic=True)`.
  kind "callable"  entry {target: "module:attr", ...}: attr(entry, ctx) -> Policy-like object
                   with act(obs, first) (stand-ins in tests and dry runs).

BACKENDS: plan["backend"]["name"] -> backend(ds, options)
---------------------------------------------------------
A backend runs one policy over a list of markets under one variant and yields one
MarketTrace per market. Both backends drive a *stepper*, the interface the runner needs from
an env (pyreplay.ReplayEnv implements it; the C env must expose the same):

    env = make_env(markets, variant, split, allow_test)
        markets: global indices into markets.bin; env i plays markets[i] from its row 0
        (start offset 0) to its last row, then idles. variant: fill_model ("optimistic" |
        "central" | "pessimistic"), latency_s (0/1/2), size_mult (0.5/1/2), mirror (bool).
        allow_test must be True for split 2 (the C core also requires MM_FINAL=1).
    env.obs_size; env.obs_spec() -> baselines ObsSpec (or None: baselines/obs_spec.load_spec())
    obs = env.reset()                     (B, obs_size) float32, observation of row 0
    env.trace() -> dict of (B,) arrays, the state AFTER the current row was processed:
        active bool (False once the market is finished), row int64 (global ticks row, -1 idle),
        cash (USDC; includes rebates, liquidity rewards and fees paid; NOT the settlement of
        inventory), inv (Up shares, signed, Up frame of the episode), reward_cum, rebate_cum,
        fee_cum (cumulative, >= 0 fee is a cost), bid_live/ask_live (our quote live during
        this second, after latency), bid_px/ask_px (its price, NaN when off)
    obs = env.step(actions)               decision on the current row, advance one row
    env.drain_fills() -> dict of arrays, one entry per fill of OUR resting orders since the
        last call: env (index into markets), row (global row = second the fill is credited),
        side (+1 we bought Up), qty, price (our price, Up frame), fee, rebate,
        print_px (price of the taker print / maker leg that filled us; NaN if unknown)
    env.settlement() -> (B,) Up-frame payoff of each market (1 - winner_up when mirrored)

The C env of record is cenv.CEnvBackend (cenv.py): it does not use the stepper loop above,
because the core exports no per-step cash/inventory or per-fill records. It runs the policy in
closed loop through the C core (PufferLib binding), replays the recorded actions in the C harness
(bit-identical observations; per-market C Logs and per-step C state) and in reference.py (every
fill, verified step by step against the C state). It returns the same MarketTrace dicts, plus
`log` (the market's C Log) and `checks` (replay agreement).
"""
from __future__ import annotations

import importlib
import sys

import numpy as np

from common import CODE_ROOT, FINAL, Dataset, resolve, train_policy_module

TRACE_FIELDS = ("cash", "inv", "reward_cum", "rebate_cum", "fee_cum", "bid_live", "ask_live",
                "bid_px", "ask_px")
FILL_FIELDS = ("row", "side", "qty", "price", "fee", "rebate", "print_px")


class BackendUnavailable(RuntimeError):
    pass


class ContractError(RuntimeError):
    """The env broke the stepper contract (missing rows, wrong shapes, ...)."""


# ============================================================================ policies
class FnPolicy:
    def __init__(self, fn, name=""):
        self.fn, self.name = fn, name

    def reset(self, B):
        pass

    def act(self, obs, first):
        return np.asarray(self.fn(obs), dtype=np.int32)


class TorchPPO3Policy:
    """Greedy actions of a train/policy.py checkpoint (MLP or recurrent PufferNet)."""

    def __init__(self, path, obs_size=None):
        import torch
        tp = train_policy_module()  # train/policy.py
        self.torch, self.tp = torch, tp
        self.net, self.ck = tp.load_checkpoint(str(path))
        self.net.eval()
        self.recurrent = bool(getattr(self.net, "recurrent", False))
        if obs_size is not None and int(self.ck["obs_size"]) != int(obs_size):
            raise ContractError(f"{path}: checkpoint obs_size {self.ck['obs_size']} != env {obs_size}")
        self.state = None

    def reset(self, B):
        self.state = {}

    def act(self, obs, first):
        torch = self.torch
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs, np.float32))
            if self.recurrent:
                self.state["done"] = torch.as_tensor(np.asarray(first, np.float32))
                logits, _ = self.net.forward_eval(o, self.state)
            else:
                logits, _ = self.net.forward_eval(o)
            return self.tp.greedy_actions(logits).numpy().astype(np.int32)


class Puffer5Policy:
    def __init__(self, entry, obs_size=None):
        import puffernet_mm as p5  # puffer5/puffernet_mm.py
        kw = dict(hidden=int(entry.get("hidden", 128)), layers=int(entry.get("layers", 4)),
                  deterministic=True)
        if entry.get("obs_size") or obs_size:
            kw["obs_size"] = int(entry.get("obs_size") or obs_size)
        if "align" in entry:
            kw["align"] = int(entry["align"])
        self.pol = p5.PufferNetMMPolicy(entry["_abspath"], **kw)
        if obs_size is not None and self.pol.obs_size != int(obs_size):
            raise ContractError(f"5.0 checkpoint obs_size {self.pol.obs_size} != env {obs_size}")
        self.state = None

    def reset(self, B):
        self.state = None

    def act(self, obs, first):
        a, self.state = self.pol(obs, self.state, np.asarray(first, np.float32))
        return np.asarray(a, dtype=np.int32)


def _baseline(entry, ctx):
    import baselines as bl
    spec = ctx["spec"]()
    return FnPolicy(bl.make(entry["name"], spec, entry.get("params") or {}), entry["id"])


def _ppo3(entry, ctx):
    return TorchPPO3Policy(entry["_abspath"], ctx.get("obs_size"))


def _puffer5(entry, ctx):
    return Puffer5Policy(entry, ctx.get("obs_size"))


def _callable(entry, ctx):
    mod, attr = entry["target"].split(":")
    if str(FINAL) not in sys.path:
        sys.path.insert(0, str(FINAL))
    obj = getattr(importlib.import_module(mod), attr)(entry, ctx)
    if not hasattr(obj, "act"):
        obj = FnPolicy(obj, entry["id"])
    if not hasattr(obj, "reset"):
        obj.reset = lambda B: None
    return obj


POLICY_KINDS = {"baseline": _baseline, "ppo3": _ppo3, "puffer5": _puffer5, "callable": _callable}


def make_policy(entry: dict, ctx: dict):
    """ctx: spec (callable -> ObsSpec), obs_size, root (artifact root for relative paths)."""
    if entry["kind"] not in POLICY_KINDS:
        raise KeyError(f"unknown method kind {entry['kind']!r}; known {sorted(POLICY_KINDS)}")
    e = dict(entry)
    if "path" in e:
        e["_abspath"] = str(resolve(e["path"], ctx.get("root"), CODE_ROOT))
    return POLICY_KINDS[e["kind"]](e, ctx)


# ============================================================================ driver
def drive(env, policy, markets, ds: Dataset) -> list[dict]:
    """Run one policy over the env's markets; return one MarketTrace per market."""
    B = len(markets)
    recs = ds.markets[np.asarray(markets)]
    off = recs["row_offset"].astype(np.int64)
    n = recs["n_rows"].astype(np.int64)
    buf = {k: [np.full(int(n[i]), np.nan) for i in range(B)] for k in TRACE_FIELDS}
    seen = [np.zeros(int(n[i]), bool) for i in range(B)]
    fills = {k: [] for k in ("env",) + FILL_FIELDS}

    def record():
        tr = env.trace()
        act = np.asarray(tr["active"], bool)
        rows = np.asarray(tr["row"], np.int64)
        for i in np.flatnonzero(act):
            k = rows[i] - off[i]
            if not 0 <= k < n[i]:
                raise ContractError(f"env {i}: row {rows[i]} outside market {markets[i]}")
            for f in TRACE_FIELDS:
                buf[f][i][k] = tr[f][i]
            seen[i][k] = True
        return act

    def take_fills():
        fd = env.drain_fills()
        if len(fd.get("row", ())):
            for k in fills:
                v = fd.get(k)
                if v is None:
                    v = np.full(len(fd["row"]), np.nan)
                fills[k].append(np.asarray(v))

    policy.reset(B)
    obs = env.reset()
    act = record()
    first = np.ones(B, bool)
    steps, max_steps = 0, int(n.max()) + 8
    while act.any():
        a = policy.act(obs, first)
        a = np.asarray(a, np.int32).reshape(B, -1)
        obs = env.step(a)
        first[:] = False
        act = record()
        take_fills()
        steps += 1
        if steps > max_steps:
            raise ContractError("env did not finish its markets within n_rows steps")
    take_fills()
    for i in range(B):
        if not seen[i].all():
            raise ContractError(f"market {markets[i]}: {int((~seen[i]).sum())} rows never traced")
    F = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in fills.items()}
    settle = np.asarray(env.settlement(), dtype=np.float64)
    out = []
    for i in range(B):
        sel = F["env"].astype(np.int64) == i if F["env"].size else np.zeros(0, bool)
        out.append(dict(market=int(markets[i]), settlement=float(settle[i]),
                        **{f: buf[f][i] for f in TRACE_FIELDS},
                        fills={f: F[f][sel] for f in FILL_FIELDS}))
    return out


# ============================================================================ backends
class StepperBackend:
    name = "stepper"
    of_record = False

    def __init__(self, ds: Dataset, options: dict | None = None):
        self.ds, self.options = ds, dict(options or {})
        self._probe = None

    def make_env(self, markets, variant, split, allow_test):
        raise NotImplementedError

    def probe(self, split):
        """A tiny env for the obs spec / size (built on the first market of the split)."""
        if self._probe is None:
            m = self.ds.split_markets(split)[:1]
            self._probe = self.make_env(m, dict(fill_model="central", latency_s=1, size_mult=1.0),
                                        split, allow_test=(split == 2))
        return self._probe

    def obs_spec(self, split):
        spec = getattr(self.probe(split), "obs_spec", lambda: None)()
        if spec is None:
            from obs_spec import load_spec
            spec = load_spec()
        return spec

    def obs_size(self, split):
        return int(self.probe(split).obs_size)

    def run(self, policy, markets, variant, split, allow_test):
        """MarketTraces for `markets`, batched by contract length (equal episode lengths)."""
        markets = np.asarray(markets, dtype=np.int64)
        lens = self.ds.markets["contract_len_s"][markets]
        out = {}
        for L in np.unique(lens):
            grp = markets[lens == L]
            env = self.make_env(grp, variant, split, allow_test)
            for tr in drive(env, policy, grp, self.ds):
                out[tr["market"]] = tr
        return [out[int(m)] for m in markets]


class PyReplayBackend(StepperBackend):
    name = "pyreplay"
    of_record = False

    def make_env(self, markets, variant, split, allow_test):
        import pyreplay
        return pyreplay.make_env(self.ds, markets, variant, split, allow_test)


def _cenv(ds, options):
    import cenv
    return cenv.CEnvBackend(ds, options)


BACKENDS = {"pyreplay": PyReplayBackend, "cenv": _cenv}


def make_backend(name, ds, options=None):
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known {sorted(BACKENDS)}")
    return BACKENDS[name](ds, options)
