"""Decode the replay env's float32 observation into the raw quantities the baselines use.

The policies in `baselines.py` never index the observation directly. They read a `State`
(a dict of float64 arrays of shape (B,)), produced here from the env's documented
feature-name -> obs-index mapping. The mapping is loaded at runtime from (first found):

  1. `../data/build/feat_meta.json`   (machine-readable; preferred)
  2. `../ENV.md`                      (a fenced ```json block holding the same object, or a
                                       markdown table whose rows are `| idx | name | ... |`)

and loading fails with a clear error if neither exists or a required feature is absent.

Accepted spec shapes (whichever the env writes):
  {"features": [{"name": str, "index": int, <decode keys>}, ...]}
  {"obs_index" | "column_index" | "feature_index": {name: idx}}
  {"obs_names" | "obs_columns" | "features": [name, ...]}      (index = position)
Decode keys of a feature entry (all optional; identity if absent):
  "mean"/"std"          obs = (raw - mean) / std
  "shift"/"scale"       obs = (raw - shift) / scale       ("offset" is a synonym of "shift")
  "transform"           applied after the linear inverse: "logit" (raw = sigmoid(x)),
                        "log" (raw = exp(x)), "log1p" (raw = expm1(x)), "signed_log1p".

Raw quantities the baselines need (`State` keys) and the env feature names accepted for each
(first match wins). A quantity is derived when it is not observed directly:

  bid, ask      best Up-frame bid / ask price in $        best_bid | bid_px_0 ...
                derived from mid +- spread/2 if only those are observed, or from
                mid and (bid|ask) offsets in ticks.
  bid_ok/ask_ok side present                              derived: bid > 0, ask < 1
  tick          current tick size in $                    tick | tick_size; derived from the
                Polymarket rule (0.001 if bid < 0.04 or ask > 0.96, else 0.01) if absent
  q             our inventory, Up shares (signed)         inventory | inv | q | position
  tau           seconds to close                          seconds_to_close | tau
                (or tau_frac * T)
  T             contract length, seconds                  contract_len_s | T; or one-hot
                                                          len_5m/len_15m/len_4h
  p_model       model fair value of Up                    p_model | fair_value | fv;
                (or mid + gap if only `gap` is observed)  (only the fair-value quoters)
  valid         book usable                               book_valid (default 1)
  min_size      base order size in shares                 min_order_size (default 5)

A required feature that cannot be found or derived raises `SpecError` naming what is missing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PM = HERE.parent
FEAT_META = PM / "data" / "build" / "feat_meta.json"
ENV_MD = PM / "ENV.md"

ALIASES = {
    "bid": ["best_bid", "bid_px_0", "bid0", "bid", "best_bid_px"],
    "ask": ["best_ask", "ask_px_0", "ask0", "ask", "best_ask_px"],
    "mid": ["mid", "mid_px", "book_mid"],
    "spread": ["spread", "spread_px"],
    "spread_ticks": ["spread_ticks"],
    "bid_ticks": ["bid_ticks_from_mid", "bid_off_ticks", "bid0_ticks"],
    "ask_ticks": ["ask_ticks_from_mid", "ask_off_ticks", "ask0_ticks"],
    "tick": ["tick", "tick_size", "tick_px"],
    "q": ["inventory", "inv", "q", "position", "inventory_shares"],
    "tau": ["seconds_to_close", "tau", "tau_s"],
    "tau_frac": ["tau_frac", "tau_over_T", "frac_left", "time_left_frac"],
    "T": ["contract_len_s", "T", "contract_len"],
    "len_5m": ["len_5m", "is_5m"],
    "len_15m": ["len_15m", "is_15m"],
    "len_4h": ["len_4h", "is_4h"],
    "p_model": ["p_model", "fair_value", "fv", "p_fair"],
    "gap": ["gap", "fv_gap", "gap_model_mid"],
    "valid": ["book_valid", "valid"],
    "min_size": ["min_order_size", "min_size"],
}

# action head layout of MultiDiscrete([7, 7, 3]); overridable by the spec ("actions" key)
BID_ASK_OFFSETS = [None, 2, 1, 0, -1, -2, -4]     # index -> ticks of improvement (None = off)
SIZE_MULTS = [1, 2, 5]                            # index -> multiple of min order size


class SpecError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- spec loading
def _from_obj(obj: dict) -> dict:
    """Normalise any accepted spec shape to {name: entry-dict with 'index'}."""
    feats = {}
    if isinstance(obj.get("features"), list) and obj["features"] and isinstance(obj["features"][0], dict):
        for f in obj["features"]:
            idx = f.get("index", f.get("idx", f.get("obs_index")))
            if idx is None or "name" not in f:
                raise SpecError(f"feature entry without name/index: {f}")
            if isinstance(idx, (list, tuple)):          # a block of features
                for k, i in enumerate(range(int(idx[0]), int(idx[1]) + 1)):
                    feats[f"{f['name']}_{k}"] = {**f, "index": i}
            else:
                feats[f["name"]] = {**f, "index": int(idx)}
        return feats
    for key in ("obs_index", "column_index", "feature_index"):
        if isinstance(obj.get(key), dict):
            return {n: {"index": int(i)} for n, i in obj[key].items()}
    for key in ("obs_names", "obs_columns", "features"):
        if isinstance(obj.get(key), list):
            return {n: {"index": i} for i, n in enumerate(obj[key])}
    for v in obj.values():                               # nested, e.g. {"obs": {...}}
        if isinstance(v, dict):
            try:
                out = _from_obj(v)
                if out:
                    return out
            except SpecError:
                pass
    return {}


def _from_markdown(text: str) -> tuple[dict, dict]:
    for block in re.findall(r"```json\s*(.*?)```", text, flags=re.S):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        feats = _from_obj(obj)
        if feats:
            return feats, obj
    feats = {}
    for line in text.splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"\d+", cells[0]) and re.fullmatch(r"[A-Za-z_][\w]*", cells[1]):
            feats[cells[1]] = {"index": int(cells[0])}
    return feats, {}


def load_spec(path: str | Path | None = None) -> "ObsSpec":
    """ObsSpec from an explicit path, else feat_meta.json, else ENV.md."""
    cands = [Path(path)] if path else [FEAT_META, ENV_MD]
    for p in cands:
        if not p.exists():
            continue
        if p.suffix == ".json":
            obj = json.loads(p.read_text())
            feats = _from_obj(obj)
        else:
            feats, obj = _from_markdown(p.read_text())
        if feats:
            return ObsSpec(feats, meta=obj, source=str(p))
    raise SpecError("no observation spec found: looked for "
                    + ", ".join(str(c) for c in cands)
                    + " (the env must document its feature-name -> obs-index mapping)")


# ----------------------------------------------------------------------------- decoding
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ObsSpec:
    """feature-name -> obs-index mapping plus per-feature inverse normalisation."""

    def __init__(self, features: dict, meta: dict | None = None, source: str = "<dict>"):
        self.features = features
        self.meta = meta or {}
        self.source = source
        acts = self.meta.get("actions", {}) if isinstance(self.meta.get("actions"), dict) else {}
        self.offsets = acts.get("price_offsets_ticks", BID_ASK_OFFSETS)
        self.size_mults = acts.get("size_mults", SIZE_MULTS)
        self.obs_size = self.meta.get("obs_size")

    def find(self, logical: str) -> str | None:
        for n in ALIASES.get(logical, [logical]):
            if n in self.features:
                return n
        return None

    def raw(self, obs: np.ndarray, name: str) -> np.ndarray:
        f = self.features[name]
        x = np.asarray(obs[:, f["index"]], dtype=np.float64)
        if "std" in f or "mean" in f:
            x = x * float(f.get("std", 1.0)) + float(f.get("mean", 0.0))
        if "scale" in f or "shift" in f or "offset" in f:
            x = x * float(f.get("scale", 1.0)) + float(f.get("shift", f.get("offset", 0.0)))
        t = f.get("transform")
        if t in (None, "", "none", "linear"):
            return x
        if t == "logit":
            return _sigmoid(x)
        if t == "log":
            return np.exp(x)
        if t == "log1p":
            return np.expm1(x)
        if t == "signed_log1p":
            return np.sign(x) * np.expm1(np.abs(x))
        raise SpecError(f"unknown transform {t!r} for feature {name!r}")

    def get(self, obs, logical):
        n = self.find(logical)
        return None if n is None else self.raw(obs, n)

    def decode(self, obs: np.ndarray, need_fair: bool = False) -> dict:
        """State dict of (B,) float64 arrays. Raises SpecError if something required is missing."""
        obs = np.asarray(obs)
        if obs.ndim != 2:
            raise ValueError(f"obs must be (B, OBS_SIZE), got {obs.shape}")
        B = obs.shape[0]
        g = lambda k: self.get(obs, k)  # noqa: E731
        missing = []

        tick = g("tick")
        bid, ask = g("bid"), g("ask")
        if bid is None or ask is None:
            mid, spread = g("mid"), g("spread")
            if spread is None and g("spread_ticks") is not None and tick is not None:
                spread = g("spread_ticks") * tick
            bt, at = g("bid_ticks"), g("ask_ticks")
            if mid is not None and spread is not None:
                bid, ask = mid - spread / 2, mid + spread / 2
            elif mid is not None and bt is not None and at is not None and tick is not None:
                bid, ask = mid - np.abs(bt) * tick, mid + np.abs(at) * tick
            else:
                missing.append("best bid/ask (or mid + spread)")
        q = g("q")
        if q is None:
            missing.append("inventory")
        T = g("T")
        if T is None:
            oh = [g(k) for k in ("len_5m", "len_15m", "len_4h")]
            if all(o is not None for o in oh):
                T = np.select([oh[0] > 0.5, oh[1] > 0.5, oh[2] > 0.5], [300.0, 900.0, 14400.0], np.nan)
            else:
                missing.append("contract length")
        tau = g("tau")
        if tau is None:
            tf = g("tau_frac")
            if tf is not None and T is not None:
                tau = tf * T
            else:
                missing.append("seconds to close")
        p_model = g("p_model")
        if need_fair and p_model is None:
            gap = g("gap")
            if gap is not None and bid is not None:
                p_model = 0.5 * (bid + ask) + gap
            else:
                missing.append("p_model (or gap)")
        if missing:
            raise SpecError(f"obs spec {self.source} lacks: {', '.join(missing)}")

        bid_ok = (bid > 1e-9) & (bid < 1)
        ask_ok = (ask < 1 - 1e-9) & (ask > 0)
        if tick is None:
            tick = np.where((bid_ok & (bid < 0.04 - 1e-9)) | (ask_ok & (ask > 0.96 + 1e-9)), 0.001, 0.01)
        valid = g("valid")
        valid = np.ones(B, bool) if valid is None else valid > 0.5
        ms = g("min_size")
        ms = np.full(B, float(self.meta.get("min_order_size", 5.0))) if ms is None else ms
        st = dict(bid=bid, ask=ask, bid_ok=bid_ok & valid, ask_ok=ask_ok & valid, tick=tick, q=q,
                  tau=tau, T=T, valid=valid, min_size=ms)
        if p_model is not None:
            st["p_model"] = p_model
        return st


def spec_from_names(names: list[str], **entries) -> ObsSpec:
    """Build a spec for tests: features at their list positions, optional per-feature decode keys."""
    feats = {n: {"index": i, **entries.get(n, {})} for i, n in enumerate(names)}
    return ObsSpec(feats, meta={"obs_size": len(names)}, source="<test>")
