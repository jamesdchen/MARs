"""Play every market of a split once, from offset 0, through the C env, with a batched numpy policy.

All fills, P&L and attribution come from the C core (../mm.h through the PufferLib 3.0 binding
../binding.c); nothing here re-implements a fill rule. One env per market, all sharing one config:
the core's market cycle (`cycle_stride = n_markets`, env seed i) starts env i on market i of the
split's list (markets.bin order) from `fixed_start = 0`; after reset each env's market observation
is checked against feat.bin row 0 of the market it should be playing. The envs step together
through one vectorized handle, so the policy sees one (n_markets, OBS) batch per step, and each
env's Log is read through its own single-env handle right after its first terminal. The env then
auto-resets onto the same market, and later episodes are discarded. Envs whose market has ended
keep stepping until the longest market ends; their actions do not matter.

Env parameters: the core's defaults (MM_PARAM_NAMES / MM_DEFAULTS, parsed from mm.h, so they
cannot drift from the C code) with these overrides: split, fill_model (0 optimistic, 1 central,
2 pessimistic), latency (s), fixed_start = 0, cycle_stride = n_markets, plus anything in `env`. So the
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
FLAGS = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside", "inspread"]
VOL_FLAGS = [f"vol_{f}" for f in FLAGS]
PNL_FLAGS = [f"pnl_{f}" for f in FLAGS]
# T5 as amended by eval/DECISIONS.md D2: the 25% P&L rule applies to the replay-artifact flags only
# (last10 is a subset of last30); `outside` (sweeps through the touch) is reported, not judged.
T5_ARTIFACT_FLAGS = ["inspread", "crossed", "stale", "onesided", "last30"]
T5_MAX_SHARE = 0.25
T6_MAX_INVENTORY_SHARE = 0.50
D1_MIN_FILLS = 5.0
SUM_KEYS = ["pnl", "pnl_trading", "spread_capture", "markout_5", "markout_30", "markout_60",
            "inventory_pnl", "rebates", "liq_rewards", "fills", "fill_vol", "buy_vol", "sell_vol",
            "rejects"] + VOL_FLAGS + PNL_FLAGS


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
    if (PM / "ENV.md").exists():
        files.append((PM / "ENV.md", dest / "ENV.md"))
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
        self.params["cycle_stride"] = n
        self.params["fixed_market"] = -1
        self.data_dir = Path(data_dir)
        self.obs = np.zeros((n, obs_size(mm_h)), np.float32)
        self.act = np.zeros((n, 3), np.int32)
        self.rew = np.zeros(n, np.float32)
        self.term = np.zeros(n, np.uint8)
        self.trunc = np.zeros(n, np.uint8)
        b = self.binding
        kw = {k: float(v) for k, v in self.params.items()}
        self.handles = [b.env_init(self.obs[i:i + 1], self.act[i:i + 1], self.rew[i:i + 1],
                                   self.term[i:i + 1], self.trunc[i:i + 1], i,
                                   data_dir=str(data_dir), **kw)
                        for i in range(n)]
        self.vec = b.vectorize(*self.handles)
        self.single = [b.vectorize(h) for h in self.handles]
        steps = {300: 300 / max(1, self.params["dt_5m"]), 900: 900 / max(1, self.params["dt_15m"]),
                 14400: 14400 / max(1, self.params["dt_4h"])}
        if self.params.get("dt", 0) > 0:
            steps = {k: k / self.params["dt"] for k in steps}
        self._check_markets(markets)
        self.max_steps = int(max(np.ceil(steps[int(markets[i]["contract_len_s"])]) for i in self.idx)) + 5

    def _check_markets(self, markets):
        """After reset, env i must be on market idx[i] at offset 0 (its obs equals feat.bin row 0)."""
        fm = json.loads((self.data_dir / "feat_meta.json").read_text())
        feat = np.memmap(self.data_dir / "feat.bin", np.float32, "r", shape=(fm["N"], fm["G"]))
        k = int(fm["n_obs_features"])
        self.binding.vec_reset(self.vec, 0)
        want = feat[markets["row_offset"][self.idx].astype(np.int64), :k]
        bad = np.flatnonzero(~(self.obs[:, :k] == want).all(1))
        if bad.size:
            raise RuntimeError(f"envs {bad[:10].tolist()} are not on their expected markets at offset 0")

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
    tot = lambda k: float(np.sum([r.get(k, 0.0) for r in recs]))  # noqa: E731
    vol = tot("fill_vol")
    # T5: share of fill volume and of fill P&L (spread capture + 60 s markout, as env.py's
    # pnl_share_X; NaN when that P&L is <= 0) per flag
    fill_pnl = tot("spread_capture") + tot("markout_60")
    out["fill_pnl_total"] = fill_pnl
    for f in FLAGS:
        out[f"share_vol_{f}"] = tot(f"vol_{f}") / vol if vol > 0 else 0.0
        out[f"pnl_total_{f}"] = tot(f"pnl_{f}")
        out[f"pnl_share_{f}"] = tot(f"pnl_{f}") / fill_pnl if fill_pnl > 0 else float("nan")
    art = {f: out[f"pnl_share_{f}"] for f in T5_ARTIFACT_FLAGS}
    ok = {f: v for f, v in art.items() if v == v}
    worst = max(ok, key=ok.get) if ok else None
    out["t5_worst_artifact_flag"] = worst
    out["t5_worst_artifact_share"] = ok[worst] if worst else float("nan")
    out["t5_invalid"] = bool(worst is not None and ok[worst] > T5_MAX_SHARE)
    # T6 directional (final/reduce.py directional_verdict, on totals over markets)
    comp = {"spread_capture": tot("spread_capture"), "adverse_selection": tot("markout_60"),
            "inventory": tot("inventory_pnl"), "rebates": tot("rebates"), "rewards": tot("liq_rewards")}
    total, inv = tot("pnl"), comp["inventory"]
    pos = sum(max(v, 0.0) for v in comp.values())
    out["t6_inventory_share_of_total"] = inv / total if total > 0 else float("nan")
    out["t6_inventory_share_of_positive"] = inv / pos if pos > 0 else float("nan")
    out["t6_directional"] = bool(inv > 0 and ((total > 0 and inv / total > T6_MAX_INVENTORY_SHARE)
                                             or (pos > 0 and inv / pos > T6_MAX_INVENTORY_SHARE)))
    # D1.3: counts as market making only with >= 5 fills/market, not directional, T5 artifact <= 25%
    out["d1_makes_markets"] = bool(out["fills_mean"] >= D1_MIN_FILLS and not out["t6_directional"]
                                   and not out["t5_invalid"])
    for L in LENS.values():
        sub = [r["pnl"] for r in recs if r["len"] == L]
        out[f"pnl_mean_{L}"] = float(np.mean(sub)) if sub else float("nan")
        sub = [r["fills"] for r in recs if r["len"] == L]
        out[f"fills_mean_{L}"] = float(np.mean(sub)) if sub else float("nan")
    return out


def config_hash(cfg: dict) -> str:
    return stats.sha256_json(cfg)
