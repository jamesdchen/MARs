"""Play every market of a split once, from offset 0, through the C env, with a batched numpy policy.

All fills, P&L and attribution come from the C core (../mm.h through the PufferLib 3.0 binding
../binding.c); nothing here re-implements a fill rule. One env per market, each pinned to its
market (`fixed_market`, `fixed_start = 0`). The envs step together through one vectorized handle,
so the policy sees one (n_markets, OBS) batch per step, and each env's Log is read through its own
single-env handle right after its first terminal. The env auto-resets onto the same market, and
later episodes are discarded. Envs whose market has ended keep stepping until the longest market
ends; their actions do not matter.

Env parameters: the core's defaults (MM_PARAM_NAMES / MM_DEFAULTS, parsed from mm.h, so they
cannot drift from the C code) with these overrides: split, fill_model (0 optimistic, 1 central,
2 pessimistic), latency (s), fixed_market, fixed_start = 0, plus anything passed in `env`. So the
decision interval (dt_5m, dt_15m, dt_4h), inventory limit Q, rebates (on, 20%) and liquidity
rewards (off) are those of the headline env.

    r = Runner(split=1, fill_model=1, latency=1)
    res = r.run(policy_fn)          # {"markets": [per-market Log dicts], "summary": {...}}

Env snapshots: the env's data and binary may be rebuilt while a sweep runs (a rewritten mmapped
file kills the process with SIGBUS). `make_snapshot(dir)` copies the data files, mm.h and the
compiled binding into `dir` with a sha256 manifest; `Runner(env_dir=dir)` then runs that exact
env version (tune.py --env-dir).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PM = HERE.parent
DATA_DIR = PM / "data" / "build"
sys.path.insert(0, str(PM / "eval"))
sys.path.insert(0, str(PM / "data"))

import stats  # noqa: E402  (eval/stats.py)
from load import load as load_data  # noqa: E402  (data/load.py)

FILL_MODELS = {"optimistic": 0, "central": 1, "pessimistic": 2}
LENS = {300: "5m", 900: "15m", 14400: "4h"}
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]
VOL_FLAGS = ["vol_crossed", "vol_stale", "vol_onesided", "vol_last30", "vol_last10",
             "vol_inside2", "vol_extreme", "vol_outside"]
SUM_KEYS = ["pnl", "pnl_trading", "spread_capture", "markout_5", "markout_30", "markout_60",
            "inventory_pnl", "rebates", "liq_rewards", "fills", "fill_vol", "buy_vol", "sell_vol",
            "rejects"] + VOL_FLAGS


def core_params(mm_h: Path = PM / "mm.h") -> dict:
    """{name: default} of the C core's parameter block, parsed from mm.h."""
    src = mm_h.read_text()
    names = re.search(r"MM_PARAM_NAMES\[\]\s*=\s*\{(.*?)\};", src, re.S).group(1)
    vals = re.search(r"MM_DEFAULTS\[\]\s*=\s*\{(.*?)\};", src, re.S).group(1)
    names = re.findall(r'"(\w+)"', names)
    enum = {"MM_OPT": 0, "MM_CENTRAL": 1, "MM_PESS": 2}
    vals = [v.strip() for v in re.sub(r"//[^\n]*", "", vals).split(",") if v.strip()]
    vals = [float(enum.get(v, v)) for v in vals]
    if len(names) != len(vals):
        raise RuntimeError(f"mm.h: {len(names)} parameter names but {len(vals)} defaults")
    return dict(zip(names, vals))


def obs_size(mm_h: Path = PM / "mm.h") -> int:
    return int(re.search(r"#define MM_OBS (\d+)", mm_h.read_text()).group(1))


DATA_FILES = ["markets.bin", "feat.bin", "book.bin", "fills.bin", "fill_index.bin", "meta.json",
              "feat_meta.json", "splits.json", "ticks.bin"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def make_snapshot(dest: str | Path) -> dict:
    """Copy the env (data files, mm.h, binding.c, the compiled binding) into dest + manifest.json."""
    dest = Path(dest)
    so = sorted(PM.glob("binding*.so"))
    if not so:
        raise RuntimeError(f"no compiled binding in {PM}")
    if so[-1].stat().st_mtime < (PM / "mm.h").stat().st_mtime:
        raise RuntimeError("binding .so is older than mm.h: rebuild the env first")
    dest.mkdir(parents=True, exist_ok=True)
    files = [(DATA_DIR / f, dest / f) for f in DATA_FILES] + [(PM / "mm.h", dest / "mm.h"),
             (PM / "binding.c", dest / "binding.c"), (so[-1], dest / so[-1].name)]
    for src, dst in files:
        shutil.copy2(src, dst)
    man = {"source": str(PM), "files": {dst.name: _sha(dst) for _, dst in files},
           "mtimes": {dst.name: src.stat().st_mtime for src, dst in files}}
    (dest / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def load_binding(env_dir: Path | None = None):
    """The compiled PufferLib 3.0 extension `binding` (built by ../setup.py build_ext --inplace),
    from env_dir if given (a snapshot), else from the env's folder."""
    d = str(env_dir or PM)
    if "binding" in sys.modules:
        mod = sys.modules["binding"]
        if Path(mod.__file__).resolve().parent != Path(d).resolve():
            raise RuntimeError(f"binding already loaded from {mod.__file__}, not {d}")
        return mod
    sys.path.insert(0, d)
    try:
        return importlib.import_module("binding")
    except ImportError as e:
        raise RuntimeError("C env binding not built: run `python setup.py build_ext --inplace` in "
                           f"{PM} (or the env's own build step)") from e
    finally:
        sys.path.remove(d)


def split_markets(split: int, len_mask: int = 7, data_dir: Path = DATA_DIR):
    """Global market indices of a split (markets.bin order), plus their records."""
    _, markets, _ = load_data(data_dir)
    bit = {300: 1, 900: 2, 14400: 4}
    idx = [i for i, m in enumerate(markets)
           if int(m["split"]) == split and bit[int(m["contract_len_s"])] & len_mask]
    return np.array(idx, dtype=np.int64), markets


class Runner:
    def __init__(self, split: int = 1, fill_model: int | str = "central", latency: int = 1,
                 env: dict | None = None, len_mask: int = 7, env_dir: str | Path | None = None):
        if split == 2:
            raise ValueError("baselines are tuned and evaluated on val only; test is run once "
                             "from the frozen plan (EVALUATION.md section 5)")
        self.env_dir = Path(env_dir) if env_dir else None
        data_dir = self.env_dir or DATA_DIR
        mm_h = (self.env_dir or PM) / "mm.h"
        self.binding = load_binding(self.env_dir)
        fm = FILL_MODELS.get(fill_model, fill_model)
        self.params = {**core_params(mm_h), **(env or {}), "split": split, "fill_model": int(fm),
                       "latency": int(latency), "fixed_start": 0, "len_mask": len_mask}
        self.idx, markets = split_markets(split, len_mask, data_dir)
        self.meta = [dict(market=int(i), len=LENS[int(markets[i]["contract_len_s"])],
                          coin=COINS[int(markets[i]["coin_id"])], open_ts=int(markets[i]["open_ts"]),
                          end_ts=int(markets[i]["end_ts"])) for i in self.idx]
        n = len(self.idx)
        self.n = n
        self.obs = np.zeros((n, obs_size(mm_h)), np.float32)
        self.act = np.zeros((n, 3), np.int32)
        self.rew = np.zeros(n, np.float32)
        self.term = np.zeros(n, np.uint8)
        self.trunc = np.zeros(n, np.uint8)
        b = self.binding
        kw = {k: float(v) for k, v in self.params.items()}
        self.handles = [b.env_init(self.obs[i:i + 1], self.act[i:i + 1], self.rew[i:i + 1],
                                   self.term[i:i + 1], self.trunc[i:i + 1], i,
                                   data_dir=str(data_dir), **{**kw, "fixed_market": float(m)})
                        for i, m in enumerate(self.idx)]
        self.vec = b.vectorize(*self.handles)
        self.single = [b.vectorize(h) for h in self.handles]
        steps = {300: 300 / max(1, self.params["dt_5m"]), 900: 900 / max(1, self.params["dt_15m"]),
                 14400: 14400 / max(1, self.params["dt_4h"])}
        if self.params.get("dt", 0) > 0:
            steps = {k: k / self.params["dt"] for k in steps}
        self.max_steps = int(max(np.ceil(steps[int(markets[i]["contract_len_s"])]) for i in self.idx)) + 5

    def run(self, policy) -> dict:
        b = self.binding
        b.vec_log(self.vec)                       # drop anything left from a previous run
        b.vec_reset(self.vec, 0)
        done = np.zeros(self.n, bool)
        recs = [None] * self.n
        t0 = time.perf_counter()
        steps = 0
        while not done.all():
            if steps > self.max_steps:
                raise RuntimeError(f"markets did not end after {steps} steps: {np.flatnonzero(~done)[:10]}")
            self.act[:] = policy(self.obs)
            b.vec_step(self.vec)
            steps += 1
            new = np.flatnonzero((self.term != 0) & ~done)
            for i in new:
                recs[i] = {**self.meta[i], **{k: float(v) for k, v in b.vec_log(self.single[i]).items()}}
                done[i] = True
        wall = time.perf_counter() - t0
        return {"markets": recs, "summary": summarize(recs), "steps": steps, "wall_s": wall}

    def close(self):
        if self.vec is not None:
            self.binding.vec_close(self.vec)      # frees the envs (single handles now dangle)
            self.vec, self.single, self.handles = None, [], []

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def summarize(recs: list[dict]) -> dict:
    """Per-market means of the Log fields, distribution stats of per-market P&L, attribution shares,
    exploit-flag volume shares and P&L by contract length."""
    pnl = np.array([r["pnl"] for r in recs])
    out = {"n_markets": len(recs), "pnl_mean": float(pnl.mean()), "pnl_total": float(pnl.sum()),
           "pnl_median": float(np.median(pnl)), "pnl_sd": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
           "cvar95": stats.cvar(pnl, 0.95), "frac_positive": float((pnl > 0).mean())}
    for k in SUM_KEYS:
        out[f"{k}_mean"] = float(np.mean([r.get(k, np.nan) for r in recs]))
    out["max_abs_inv_max"] = float(np.max([r["max_abs_inv"] for r in recs]))
    out["max_abs_inv_mean"] = float(np.mean([r["max_abs_inv"] for r in recs]))
    for k in ("uptime_two_sided", "uptime_any", "final_abs_inv"):
        out[f"{k}_mean"] = float(np.mean([r[k] for r in recs]))
    vol = float(np.sum([r["fill_vol"] for r in recs]))
    for k in VOL_FLAGS:
        out[f"share_{k}"] = float(np.sum([r[k] for r in recs]) / vol) if vol > 0 else 0.0
    for L in LENS.values():
        sub = [r["pnl"] for r in recs if r["len"] == L]
        out[f"pnl_mean_{L}"] = float(np.mean(sub)) if sub else float("nan")
    return out


def config_hash(cfg: dict) -> str:
    return stats.sha256_json(cfg)
