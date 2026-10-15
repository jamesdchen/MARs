"""Shared pieces of the pre-registered final test (EVALUATION.md section 5).

Pipeline (each script's docstring has the details):

    make_plan.py   writes TEST_PLAN.json (methods, variants, metrics, endpoint, decision rule,
                   ledger trial count, data and code hashes) and plan_sha256.txt; never overwrites.
    run_test.py    verifies plan hash, data hashes, code hashes and MM_FINAL=1, runs every
                   method x variant over every market of the split once, writes results/<split>/.
    report.py      results/<split>/ -> RESULTS.md + figures/.

`--split 1` (val) is a dry run of the same code paths; the test split (2) is touched only by
run_test.py with MM_FINAL=1 and only once (a second run needs --force-rerun REASON, recorded).

Paths: CODE_ROOT is the polymarket/ folder these scripts live in (env, baselines, train,
puffer5, eval code is imported from there). `root` (artifacts: baselines/tuned.json,
train/selected.json, trials.jsonl ledgers, checkpoints) and `build_dir` (data) default to the
real ones and are overridable so the unit tests run everything on a synthetic tiny dataset.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parent
CODE_ROOT = FINAL.parent                      # polymarket/
BUILD = CODE_ROOT / "data" / "build"
if str(FINAL) not in sys.path:
    sys.path.insert(0, str(FINAL))
# Other folders are only ever APPENDED (never shadow final/'s modules); generic module names
# (stats, policy) are loaded by file path under unique names.
for _d in ("baselines", "puffer5"):
    if str(CODE_ROOT / _d) not in sys.path:
        sys.path.append(str(CODE_ROOT / _d))


def load_module(name: str, path):
    """Import a source file under a unique module name (cached in sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = load_module("pm_eval_stats", CODE_ROOT / "eval" / "stats.py")


def train_policy_module():
    return load_module("pm_train_policy", CODE_ROOT / "train" / "policy.py")

SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "purged"}
FILL_MODELS = {"optimistic": 0, "central": 1, "pessimistic": 2}
FILL_MODEL_NAMES = {v: k for k, v in FILL_MODELS.items()}
LATENCIES = (0, 1, 2)
SIZE_SWEEP = (0.5, 2.0)
HEADLINE = dict(fill_model="central", latency_s=1, size_mult=1.0, mirror=False)
MARKOUT_HORIZONS = (1, 5, 10, 30, 60)
MID_CLOSE_STC = 10            # pre-close reference mark: seconds_to_close = 10 (attribution carry)
STALE_S = 10.0
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]
LENGTH_NAMES = {300: "5m", 900: "15m", 14400: "4h"}
PLAN_SCHEMA = "polymarket-mm-test-plan/1"
RESULTS_SCHEMA = "polymarket-mm-test-results/1"


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def variant_id(v: dict) -> str:
    s = f"{v['fill_model']}_lat{int(v['latency_s'])}_x{float(v['size_mult']):g}"
    return s + ("_mirror" if v.get("mirror") else "")


def make_variants() -> list[dict]:
    """3 fill models x 3 latencies at size x1, the size sweep and the mirror run at the
    headline cell. `roles` say which parts of the protocol each variant serves."""
    out = []
    for fm in FILL_MODELS:
        for lat in LATENCIES:
            v = dict(fill_model=fm, fill_model_id=FILL_MODELS[fm], latency_s=lat, size_mult=1.0,
                     mirror=False, roles=["grid"])
            if fm == "central" and lat == 1:
                v["roles"] += ["headline", "primary_endpoint"]
            if (fm, lat) in (("pessimistic", 1), ("central", 2)):
                v["roles"] += ["success_rule"]
            if lat == 0:
                v["roles"] += ["never_headline"]
            out.append(v)
    for sm in SIZE_SWEEP:
        out.append(dict(fill_model="central", fill_model_id=1, latency_s=1, size_mult=sm,
                        mirror=False, roles=["size_sweep"]))
    out.append(dict(fill_model="central", fill_model_id=1, latency_s=1, size_mult=1.0,
                    mirror=True, roles=["mirror_test"]))
    for v in out:
        v["id"] = variant_id(v)
    return out


HEADLINE_ID = variant_id(HEADLINE)


# ----------------------------------------------------------------------------- hashing / json
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, sort_keys=False, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def clean_json(obj):
    """Replace NaN/inf by None recursively (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    return obj


def append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(clean_json(rec), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    out, bad = [], 0
    p = Path(path)
    if not p.exists():
        return out, bad
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return out, bad


# ----------------------------------------------------------------------------- git / code
def git_info(path=CODE_ROOT) -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=path, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    prefix = run("rev-parse", "--show-prefix") or ""
    status = run("status", "--porcelain", "-uall", "--", ".")
    dirty = []
    for ln in (status or "").splitlines():
        p = ln[3:].split(" -> ")[-1].strip('"')
        dirty.append(p[len(prefix):] if p.startswith(prefix) else p)
    return dict(commit=commit, dirty=bool(dirty), n_dirty=len(dirty), dirty_files=dirty[:500])


def git_log_contains(text: str, path=CODE_ROOT) -> bool:
    r = subprocess.run(["git", "log", "--all", "--fixed-strings", f"--grep={text}", "--format=%H"],
                       cwd=path, capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


# Everything the evaluation imports or loads (the data itself is frozen by its own hashes;
# training scripts are not, the checkpoints are). EVALUATION.md is not here: the plan hash is
# appended to its amendment log after the plan is written.
CODE_GLOBS = [
    "final/*.py", "eval/stats.py", "baselines/baselines.py", "baselines/obs_spec.py",
    "train/policy.py", "puffer5/puffernet_mm.py", "env.py", "reference.py", "*.h", "*.c", "*.so",
    "ENV.md", "data/load.py",
]


def code_hashes(code_root=CODE_ROOT) -> dict:
    """sha256 of every evaluation-relevant source file (frozen by the plan; tests excluded)."""
    code_root = Path(code_root)
    out = {}
    for g in CODE_GLOBS:
        for p in sorted(glob.glob(str(code_root / g))):
            rel = os.path.relpath(p, code_root)
            if os.path.basename(p).startswith("test_"):
                continue
            out[rel] = S.sha256_file(p)
    return out


# ----------------------------------------------------------------------------- dataset
_NP = {"int64_t": "<i8", "int32_t": "<i4", "float": "<f4", "int8_t": "i1", "uint8_t": "u1",
       "int16_t": "<i2"}


def _record_dtype(rec: dict) -> np.dtype:
    names, formats, offsets = [], [], []
    for f in rec["fields"]:
        t = f["type"]
        shape = ()
        if "[" in t:
            t, n = t.rstrip("]").split("[")
            shape = (int(n),)
        names.append(f["name"])
        formats.append((_NP[t], shape) if shape else _NP[t])
        offsets.append(f["offset"])
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": rec["itemsize"]})


class Dataset:
    """Read-only view of a data/build directory (real or synthetic, same layout).

    ticks.bin / markets.bin / meta.json are required; feat.bin + feat_meta.json,
    fills.bin + fill_index.bin and book.bin are used when present."""

    REQUIRED = ("meta.json", "markets.bin", "ticks.bin")
    HASHED = ("meta.json", "markets.bin", "ticks.bin", "splits.json", "feat.bin", "feat_meta.json",
              "book.bin", "fills.bin", "fill_index.bin")

    def __init__(self, build_dir=BUILD):
        self.dir = Path(build_dir)
        for f in self.REQUIRED:
            if not (self.dir / f).exists():
                raise FileNotFoundError(self.dir / f)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.N, self.F = int(self.meta["N"]), int(self.meta["F"])
        self.ticks = np.memmap(self.dir / "ticks.bin", dtype=np.float32, mode="r",
                               shape=(self.N, self.F))
        self.markets = np.fromfile(self.dir / "markets.bin",
                                   dtype=_record_dtype(self.meta["market_record"]))
        assert len(self.markets) == self.meta["n_markets"], "markets.bin vs meta.json"
        self.ci = self.meta["column_index"]
        self.synthetic = bool(self.meta.get("synthetic", False))
        self.feat = self.fmeta = None
        if (self.dir / "feat.bin").exists() and (self.dir / "feat_meta.json").exists():
            self.fmeta = json.loads((self.dir / "feat_meta.json").read_text())
            assert int(self.fmeta["N"]) == self.N
            self.feat = np.memmap(self.dir / "feat.bin", dtype=np.float32, mode="r",
                                  shape=(self.N, int(self.fmeta["G"])))
        self.book = None
        if self.fmeta and "book_record" in self.fmeta and (self.dir / "book.bin").exists():
            bd = _record_dtype(self.fmeta["book_record"])
            if (self.dir / "book.bin").stat().st_size == self.N * bd.itemsize:
                self.book = np.memmap(self.dir / "book.bin", dtype=bd, mode="r", shape=(self.N,))
        self.fills = self.fill_index = None
        if self.fmeta and (self.dir / "fills.bin").exists() and (self.dir / "fill_index.bin").exists():
            self.fills = np.fromfile(self.dir / "fills.bin", dtype=_record_dtype(self.fmeta["fill_record"]))
            self.fill_index = np.fromfile(self.dir / "fill_index.bin", dtype="<i4")
            assert self.fill_index.size == self.N + 1

    # --- columns
    def tcol(self, name):
        return self.ci[name]

    def fcol(self, name):
        return None if self.fmeta is None else self.fmeta["column_index"].get(name)

    @property
    def obs_feature_names(self):
        if self.fmeta is None:
            return []
        return list(self.fmeta["columns"][: int(self.fmeta["n_obs_features"])])

    # --- markets
    def split_markets(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.markets["split"] == int(split))

    def market_rows(self, m: int) -> slice:
        r = self.markets[m]
        o = int(r["row_offset"])
        return slice(o, o + int(r["n_rows"]))

    def slug(self, m: int) -> str:
        slugs = self.meta.get("slugs")
        idx = int(self.markets[m]["market_idx"])
        return slugs[idx] if slugs and idx < len(slugs) else f"m{m}"

    def coin(self, m: int) -> str:
        inv = {v: k for k, v in self.meta.get("coin_ids", {}).items()}
        return inv.get(int(self.markets[m]["coin_id"]), str(int(self.markets[m]["coin_id"])))

    def split_time_range(self, split: int, bin_seconds: float = 3600.0):
        """[t_start, t_end) of the zero-filled time grid: from the UTC midnight of the first
        open to the end of the bin holding the last close row (row t = end is included)."""
        idx = self.split_markets(split)
        t0 = int(self.markets["open_ts"][idx].min()) // 86400 * 86400
        t1 = int(self.markets["end_ts"][idx].max()) + 1
        t1 = int(-(-(t1 - t0) // bin_seconds) * bin_seconds + t0)
        return t0, t1

    # --- marks (mid used for MTM and markouts)
    def mark(self, m: int) -> np.ndarray:
        """Mark per row of market m (float64): feat 'mark' (book mid if two-sided and valid,
        else the fair value) when available, else the two-sided ticks mid forward-filled
        (0.5 before the first two-sided row). Settlement is NOT applied here."""
        sl = self.market_rows(m)
        j = self.fcol("mark")
        if j is not None:
            return np.asarray(self.feat[sl, j], dtype=np.float64)
        t = self.ticks[sl]
        bid, ask = t[:, self.ci["bid_px_0"]].astype(np.float64), t[:, self.ci["ask_px_0"]].astype(np.float64)
        ok = (t[:, self.ci["book_valid"]] > 0.5) & (bid > 0) & (ask < 1)
        mid = np.where(ok, 0.5 * (bid + ask), np.nan)
        idx = np.where(ok, np.arange(mid.size), -1)
        idx = np.maximum.accumulate(idx)
        out = np.where(idx >= 0, mid[np.maximum(idx, 0)], 0.5)
        return out

    def delta_usd(self, m: int):
        j = self.fcol("delta_usd_raw")
        if j is None:
            return None
        return np.asarray(self.feat[self.market_rows(m), j], dtype=np.float64)

    # --- hashing
    def test_rows_sha256(self) -> str:
        """Exactly as data/load.py verify(): sha256 of the C-contiguous bytes of
        ticks[row_lo:row_hi] (split 2 rows). Reads the test rows; called only by the runner."""
        t = self.meta["test_rows"]
        return sha256_bytes(np.ascontiguousarray(self.ticks[t["row_lo"]: t["row_hi"]]).tobytes())

    def split_rows_sha256(self, split: int) -> str:
        idx = self.split_markets(split)
        lo = int(self.markets["row_offset"][idx].min())
        hi = int((self.markets["row_offset"][idx] + self.markets["n_rows"][idx]).max())
        return sha256_bytes(np.ascontiguousarray(self.ticks[lo:hi]).tobytes())

    def file_hashes(self) -> dict:
        return {f: S.sha256_file(self.dir / f) for f in self.HASHED if (self.dir / f).exists()}


# ----------------------------------------------------------------------------- ledgers
def find_ledgers(root) -> list[Path]:
    """Every trials.jsonl under `root` (the polymarket folder), except final/ itself and caches."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob("trials.jsonl")):
        parts = p.relative_to(root).parts
        if parts[0] == "final" or ".venv" in parts or "cache" in parts:
            continue
        out.append(p)
    return out


def _is_val(rec) -> bool | None:
    for k in ("split", "eval_split", "data_split"):
        if k in rec:
            v = rec[k]
            return v in (1, "1", "val", "validation")
    return None


def _config_key(rec) -> str:
    for k in ("config_sha256", "config_hash", "params_sha256", "cfg_sha256"):
        if isinstance(rec.get(k), str):
            return rec[k]
    for k in ("config", "params", "cfg"):
        if isinstance(rec.get(k), dict):
            return S.sha256_json({"kind": rec.get("kind", rec.get("method", "")), "cfg": rec[k]})
    return S.sha256_json(rec)


def _sharpe_of(rec):
    srcs = [rec] + [rec[k] for k in ("metrics", "val", "result") if isinstance(rec.get(k), dict)]
    for d in srcs:
        for k in ("sharpe_hourly", "hourly_sharpe", "sharpe_1h", "sharpe"):
            v = d.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)
    return None


NOTE_KINDS = {"note", "comment", "meta", "header", "annotation"}


def count_trials(root) -> dict:
    """Trial count N for T4: every val evaluation in every ledger, plus distinct configs.
    A record without a split field is counted as val (conservative) and reported."""
    files, n_val, n_all, n_assumed, keys, sharpes, kinds = [], 0, 0, 0, set(), [], {}
    n_notes = 0
    for p in find_ledgers(root):
        recs, bad = read_jsonl(p)
        fv = fa = 0
        for r in recs:
            if str(r.get("kind", "")).lower() in NOTE_KINDS:     # annotations, not evaluations
                n_notes += 1
                continue
            n_all += 1
            v = _is_val(r)
            if v is False:
                continue
            if v is None:
                fa += 1
            fv += 1
            keys.add(_config_key(r))
            sh = _sharpe_of(r)
            if sh is not None:
                sharpes.append(sh)
            kind = str(r.get("kind", r.get("method", r.get("group", "unknown"))))
            kinds[kind] = kinds.get(kind, 0) + 1
        n_val += fv
        n_assumed += fa
        files.append(dict(path=p.relative_to(root).as_posix(), sha256=S.sha256_file(p),
                          n_records=len(recs), n_val_records=fv, n_assumed_val=fa,
                          n_unparseable_lines=bad))
    return dict(files=files, n_trials=n_val, n_records_all_splits=n_all, n_annotation_rows=n_notes,
                n_assumed_val=n_assumed, n_distinct_configs=len(keys), by_kind=kinds,
                n_with_sharpe=len(sharpes),
                var_trial_sharpes=float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else None)


# ----------------------------------------------------------------------------- plan
class PlanError(RuntimeError):
    pass


def load_plan(plan_path) -> tuple[dict, str]:
    """(plan, sha256) after checking the hash against plan_sha256.txt next to it."""
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise PlanError(f"no plan at {plan_path}: run make_plan.py first")
    h = S.sha256_file(plan_path)
    hp = plan_path.parent / "plan_sha256.txt"
    if not hp.exists():
        raise PlanError(f"{hp} missing: the plan hash must be recorded before the run")
    rec = hp.read_text().split()[0].strip()
    if rec != h:
        raise PlanError(f"TEST_PLAN.json sha256 {h} != recorded {rec} ({hp}): plan changed")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise PlanError(f"unknown plan schema {plan.get('schema')!r}")
    return plan, h


def resolve(path, *bases) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for b in bases:
        if b is not None and (Path(b) / p).exists():
            return Path(b) / p
    return Path(bases[0]) / p if bases else p
