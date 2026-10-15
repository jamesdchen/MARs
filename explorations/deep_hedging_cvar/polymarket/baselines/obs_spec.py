"""Decode the replay env's float32 observation into the raw quantities the baselines use.

The policies in `baselines.py` never index the observation directly. They read a `State`
(a dict of float64 arrays of shape (B,)), produced here from the env's documented
feature-name -> obs-index mapping, loaded at runtime:

  * market block: `../data/build/feat_meta.json` (`column_index`, `n_obs_features`: columns
    0..n_obs_features-1 of feat.bin are copied into the observation). Each column used here has
    a known inverse of its documented encoding (`FEAT_META_DECODERS`), checked against
    `column_docs` at load time, so a changed encoding fails loudly instead of silently.
  * agent block (inventory, ...): `../ENV.md`, either a fenced ```json block holding a spec object
    (shapes below) or a markdown table whose rows start `| idx | name |`. The inventory slot and
    its scale are also parsed from the C source (`../mm.h`, mm_write_obs: `o[k] = (float)(m->inv
    / Q)` after the MM_FEAT_OBS market columns); that is used when ENV.md is absent or silent, and
    must agree with ENV.md's index when both exist.

Loading fails with `SpecError` naming what is missing if a file or a required feature is absent.

Accepted spec-object shapes:
  {"features": [{"name": str, "index": int, <decode keys>}, ...]}
  {"obs_index" | "column_index" | "feature_index": {name: idx}}
  {"obs_names" | "obs_columns" | "features": [name, ...]}      (index = position)
Decode keys of a feature entry (all optional; identity if absent):
  "mean"/"std"          obs = (raw - mean) / std
  "shift"/"scale"       obs = (raw - shift) / scale       ("offset" is a synonym of "shift")
  "transform"           applied after the linear inverse: "logit" (raw = sigmoid(x)),
                        "log" (raw = exp(x)), "log1p" (raw = expm1(x)), "signed_log1p".

Raw quantities the baselines need (`State` keys), and where they come from:

  bid, ask        best Up-frame bid / ask ($). From best_bid/bid_px_0 if observed; with the
                  feat_meta block: mid = sigmoid(3 mark_logit), spread = tick exp(2 spread_ticks),
                  bid/ask = mid -+ spread/2 snapped to the tick grid; NaN unless two-sided
                  (mark_is_fv = 0), since the obs carries no price level for a one-sided book.
  bid_ok, ask_ok  side present (bid_present / ask_present; else bid > 0 / ask < 1)
  tick            0.001 if tick_fine else 0.01 (else the tick feature, else the 0.04/0.96 rule)
  q               our inventory, Up shares, signed       (agent block: inventory | inv | q ...)
  tau             seconds to close = tau_frac * T (rounded to whole seconds)
  T               contract length from len_5m/len_15m/len_4h (or contract_len_s)
  p_model         sigmoid(3 p_logit) (or p_model / mid + gap)   (fair-value quoters only)
  min_size        5 shares unless observed / documented (`min_order_size`)
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
MM_H = PM / "mm.h"

ALIASES = {
    "bid": ["best_bid", "bid_px_0", "bid0", "bid", "best_bid_px"],
    "ask": ["best_ask", "ask_px_0", "ask0", "ask", "best_ask_px"],
    "mid": ["mid", "mid_px", "book_mid", "mark"],
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
    "bid_present": ["bid_present"],
    "ask_present": ["ask_present"],
    "tick_fine": ["tick_fine"],
    "mark_is_fv": ["mark_is_fv"],
}

# action head layout of MultiDiscrete([7, 7, 3]); overridable by the spec ("actions" key)
BID_ASK_OFFSETS = [None, 2, 1, 0, -1, -2, -4]     # index -> ticks of improvement (None = off)
SIZE_MULTS = [1, 2, 5]                            # index -> multiple of min order size


class SpecError(RuntimeError):
    pass


def _sigmoid3(x):
    return 1.0 / (1.0 + np.exp(-3.0 * x))


# feat_meta.json column -> (logical name, documented encoding (must match column_docs), inverse)
FEAT_META_DECODERS = {
    "p_logit": ("p_model", "logit(p_model) / 3, p clipped to [1e-4, 1 - 1e-4]", _sigmoid3),
    "mark_logit": ("mark", "logit(mark) / 3; mark = book mid if two-sided and valid, else p_model",
                   _sigmoid3),
    "gap_c": ("gap", "(p_model - mark) in cents / 5", lambda x: x * 0.05),
    "tau_frac": ("tau_frac", "seconds_to_close / contract_len_s", lambda x: x),
    "spread_ticks": ("spread_ticks", "ln(spread / tick) / 2 if two-sided, else 0",
                     lambda x: np.exp(2.0 * x)),
    "bid_present": ("bid_present", "best bid exists (book valid and bid_px_0 > 0)", lambda x: x),
    "ask_present": ("ask_present", "best ask exists (book valid and ask_px_0 < 1)", lambda x: x),
    "len_5m": ("len_5m", "contract length one-hot", lambda x: x),
    "len_15m": ("len_15m", "contract length one-hot", lambda x: x),
    "len_4h": ("len_4h", "contract length one-hot", lambda x: x),
    "tick_fine": ("tick_fine", "1 once the data shows sub-cent prices (tick 0.001), sticky",
                  lambda x: x),
    "mark_is_fv": ("mark_is_fv", "1 if the mark fell back to p_model (one-sided or no valid book)",
                   lambda x: x),
}


def _from_feat_meta(obj: dict, obs_offset: int = 0) -> dict:
    """Market-block features of feat_meta.json (observed columns only), with decoders."""
    n_obs = int(obj["n_obs_features"])
    ci, docs = obj["column_index"], obj.get("column_docs", {})
    feats = {}
    for col, (logical, doc, inv) in FEAT_META_DECODERS.items():
        if col not in ci or ci[col] >= n_obs:
            continue
        if docs.get(col) != doc:
            raise SpecError(f"feat_meta.json encoding of {col!r} changed: {docs.get(col)!r} "
                            f"(baselines decode {doc!r}); update obs_spec.FEAT_META_DECODERS")
        feats[logical] = {"index": obs_offset + int(ci[col]), "decoder": inv, "column": col}
    return feats


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


def inventory_from_core(mm_h: Path = MM_H) -> dict | None:
    """{"index", "scale_param"} of the inventory slot, from mm.h's observation writer."""
    if not Path(mm_h).exists():
        return None
    src = Path(mm_h).read_text()
    base = re.search(r"#define MM_FEAT_OBS (\d+)", src)
    blk = re.search(r"float\* o = obs \+ MM_FEAT_OBS;", src)
    inv = re.search(r"o\[(\d+)\] = \(float\)\(m->inv / (\w+)\);", src)
    if not (base and blk and inv):
        return None
    return {"index": int(base.group(1)) + int(inv.group(1)), "scale_param": inv.group(2)}


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


def load_spec(path: str | Path | None = None, feat_meta: str | Path | None = None,
              env_md: str | Path | None = None, env_params: dict | None = None,
              mm_h: str | Path | None = None) -> "ObsSpec":
    """ObsSpec from an explicit spec file (json or md), else feat_meta.json (market block)
    merged with ENV.md (agent block and any overrides); the inventory slot is cross-checked
    against (or, if ENV.md is absent or silent, taken from) mm.h, scaled by the env parameter it
    is divided by (`env_params`, e.g. Q=100)."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise SpecError(f"no observation spec found at {p}")
        if p.suffix == ".json":
            obj = json.loads(p.read_text())
            feats = _from_feat_meta(obj) if "n_obs_features" in obj else _from_obj(obj)
        else:
            feats, obj = _from_markdown(p.read_text())
        if not feats:
            raise SpecError(f"{p} holds no feature-name -> obs-index mapping")
        return ObsSpec(feats, meta=obj, source=str(p))
    fm, md = Path(feat_meta or FEAT_META), Path(env_md or ENV_MD)
    feats, meta, src = {}, {}, []
    md_feats, md_obj = ({}, {}) if not md.exists() else _from_markdown(md.read_text())
    if fm.exists():
        obj = json.loads(fm.read_text())
        off = int(md_obj.get("market_obs_offset", 0)) if isinstance(md_obj, dict) else 0
        feats.update(_from_feat_meta(obj, off))
        src.append(str(fm))
    if md_feats:
        feats.update(md_feats)
        meta.update(md_obj if isinstance(md_obj, dict) else {})
        src.append(str(md))
    if not feats:
        raise SpecError(f"no observation spec found: looked for {fm} and {md} "
                        "(the env must document its feature-name -> obs-index mapping)")
    core = inventory_from_core(Path(mm_h or MM_H))
    inv_name = next((n for n in ALIASES["q"] if n in feats), None)
    if core is not None:
        if env_params is None or core["scale_param"] not in env_params:
            raise SpecError(f"inventory is obs / {core['scale_param']}: pass env_params with "
                            f"{core['scale_param']!r} to load_spec")
        scale = float(env_params[core["scale_param"]])
        if inv_name is None:
            feats["inventory"] = {"index": core["index"], "scale": scale}
            src.append(f"mm.h mm_write_obs (inventory = obs[{core['index']}] * {core['scale_param']}; "
                       "not in ENV.md)")
        else:
            if feats[inv_name]["index"] != core["index"]:
                raise SpecError(f"ENV.md puts {inv_name} at obs[{feats[inv_name]['index']}] but mm.h "
                                f"writes inventory at obs[{core['index']}]")
            if not any(k in feats[inv_name] for k in ("scale", "std", "decoder")):
                feats[inv_name] = {**feats[inv_name], "scale": scale}
    return ObsSpec(feats, meta=meta, source=" + ".join(src))


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

    @property
    def width(self) -> int:
        return int(self.obs_size or 1 + max(f["index"] for f in self.features.values()))

    def check(self, need_fair: bool = False) -> None:
        """Raise SpecError now (not mid-rollout) if a required quantity cannot be decoded."""
        with np.errstate(all="ignore"):
            self.decode(np.zeros((1, self.width), np.float32), need_fair=need_fair)

    def find(self, logical: str) -> str | None:
        for n in ALIASES.get(logical, [logical]):
            if n in self.features:
                return n
        return None

    def raw(self, obs: np.ndarray, name: str) -> np.ndarray:
        f = self.features[name]
        x = np.asarray(obs[:, f["index"]], dtype=np.float64)
        if "decoder" in f:
            return f["decoder"](x)
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
        if tick is None and g("tick_fine") is not None:
            tick = np.where(g("tick_fine") > 0.5, 0.001, 0.01)
        bp, ap = g("bid_present"), g("ask_present")
        bid, ask = g("bid"), g("ask")
        if bid is None or ask is None:
            mid, spread = g("mid"), g("spread")
            if spread is None and g("spread_ticks") is not None and tick is not None:
                spread = np.rint(g("spread_ticks")) * tick
            bt, at = g("bid_ticks"), g("ask_ticks")
            if mid is not None and spread is not None:
                bid, ask = mid - spread / 2, mid + spread / 2
            elif mid is not None and bt is not None and at is not None and tick is not None:
                bid, ask = mid - np.abs(bt) * tick, mid + np.abs(at) * tick
            else:
                missing.append("best bid/ask (or mid + spread)")
            if bid is not None and tick is not None:     # snap to the grid (float32 obs)
                bid, ask = np.rint(bid / tick) * tick, np.rint(ask / tick) * tick
            if bid is not None:                          # price levels known only if two-sided
                two = np.ones(B, bool)
                if bp is not None and ap is not None:
                    two &= (bp > 0.5) & (ap > 0.5)
                if g("mark_is_fv") is not None:
                    two &= g("mark_is_fv") < 0.5
                bid, ask = np.where(two, bid, np.nan), np.where(two, ask, np.nan)
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
                tau = np.rint(tf * T)
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

        valid = g("valid")
        valid = np.ones(B, bool) if valid is None else valid > 0.5
        bid_ok = (bp > 0.5) if bp is not None else (bid > 1e-9) & (bid < 1)
        ask_ok = (ap > 0.5) if ap is not None else (ask < 1 - 1e-9) & (ask > 0)
        if tick is None:
            tick = np.where((bid_ok & (bid < 0.04 - 1e-9)) | (ask_ok & (ask > 0.96 + 1e-9)), 0.001, 0.01)
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
