"""The C env of record (env.py / binding.c over mm.h) behind the runner's backend interface.

For one policy, one variant and the markets of one contract length, `CEnvBackend.run`:

1. Closed loop through the C core. PufferLib 3.0's native binding (the `binding` extension
   env.py wraps) with num_envs = n markets, the env parameters frozen in the plan plus
   split / allow_test / fill_model / latency / size_mult, fixed_start = 0 and
   cycle_stride = len(list) with len_mask = this length, so env i plays the i-th market of the
   core's (split, length) list from row 0 (checked: each first observation's market block equals
   feat.bin's row 0 of the expected market, bit for bit, and every env ends on the same step).
   Greedy policy actions are recorded.
2. Replay in the C harness (mm_harness.c, compiled from the frozen source): the same core,
   the same parameters, the recorded actions. Its observations must equal the closed loop's
   bit for bit, so the replay is the same trajectory. It yields each market's C Log (the
   numbers of record: P&L, attribution sums, flag volumes, uptime, max |inventory|) and the
   core's state after every step.
3. Replay in reference.py (the numpy twin test_env.py checks against the core) with the
   recorded actions, to get what the core does not export: every individual fill (match row,
   side, size, price, rebate, the print that hit us) and the per-second trace. Its state is
   compared with the C state after every step (cash, inventory, rebates, rewards, wealth) and its
   final P&L with the C Log; the maxima are recorded per market and run_test refuses results
   whose replay drifts beyond tolerance.

The core reads its data through a static mmap opened once per process, so each data directory
(the snapshot, its mirrored copy) gets its own copy of the binding and the harness library.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.machinery
import importlib.util
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import CODE_ROOT, FILL_MODELS, Dataset, load_module

LEN_BIT = {300: 1, 900: 2, 14400: 4}
ENUMS = {"MM_OPT": 0, "MM_CENTRAL": 1, "MM_PESS": 2}
STATE_TOL = 1e-7          # |C - reference| on cash / inventory / rebates / rewards / wealth (USD, shares)
LOG_TOL = 2e-3            # |C Log P&L (float32) - reference P&L| (USD), plus 1e-6 relative


class BackendUnavailable(RuntimeError):
    pass


class ContractError(RuntimeError):
    pass


class Core:
    """What the runner needs to know about mm.h / mm_harness.c, parsed from the frozen sources."""

    def __init__(self, code_root=CODE_ROOT):
        code_root = Path(code_root)
        src = (code_root / "mm.h").read_text()
        names = re.findall(r'"(\w+)"', re.search(r"MM_PARAM_NAMES\[\]\s*=\s*\{(.*?)\};", src, re.S)[1])
        vals = re.sub(r"//[^\n]*", "", re.search(r"MM_DEFAULTS\[\]\s*=\s*\{(.*?)\};", src, re.S)[1])
        vals = [float(ENUMS.get(v.strip(), v.strip())) for v in vals.split(",") if v.strip()]
        if len(names) != len(vals):
            raise ContractError("mm.h: parameter names and defaults differ in length")
        self.param_names, self.defaults = names, dict(zip(names, vals))
        self.obs = int(re.search(r"#define MM_OBS (\d+)", src)[1])
        self.n_feat_obs = int(re.search(r"#define MM_FEAT_OBS (\d+)", src)[1])
        body = re.sub(r"//[^\n]*", "", re.search(r"struct Log \{(.*?)\};", src, re.S)[1])
        self.log_fields = re.findall(r"float\s+(\w+)\s*;", body)
        hs = (code_root / "mm_harness.c").read_text()
        self.nstate = int(re.search(r"#define MM_NSTATE (\d+)", hs)[1])
        rec = re.search(r"static void record\(.*?\{(.*?)\n\}", hs, re.S)[1]
        st = {int(i): n for i, n in re.findall(r"s\[(\d+)\]\s*=\s*m->([\w\[\]\.]+)\s*;", rec)}
        self.state_index = {n: i for i, n in st.items()}
        for need in ("t", "inv", "cash", "rebates", "liq", "wealth"):
            if need not in self.state_index:
                raise ContractError(f"mm_harness.c record() lacks m->{need}")


def _cc(code_root: Path, out: Path):
    cmd = [os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off", "-D_POSIX_C_SOURCE=200809L",
           "-shared", "-fPIC", "-o", str(out), str(code_root / "mm_harness.c"), "-lm"]
    subprocess.run(cmd, check=True, capture_output=True)


def _binding_path(code_root: Path) -> Path:
    cands = sorted(code_root.glob("binding*.so"))
    if not cands:
        raise BackendUnavailable(f"no compiled binding*.so in {code_root} (python setup.py build_ext --inplace)")
    return cands[0]


def _load_binding(so: Path):
    loader = importlib.machinery.ExtensionFileLoader("binding", str(so))
    spec = importlib.util.spec_from_file_location("binding", str(so), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class CEnvBackend:
    name = "cenv"
    of_record = True
    _n = 0

    def __init__(self, ds: Dataset, options: dict | None = None):
        o = dict(options or {})
        self.ds = ds
        self.code_root = Path(o.get("code_root", CODE_ROOT))
        for f in ("mm.h", "mm_harness.c", "reference.py"):
            if not (self.code_root / f).exists():
                raise BackendUnavailable(f"{self.code_root / f} missing")
        self.core = Core(self.code_root)
        self.params = dict(self.core.defaults)
        self.params.update({k: float(v) for k, v in (o.get("env_params") or {}).items()})
        bad = set(self.params) - set(self.core.param_names)
        if bad:
            raise ContractError(f"unknown env parameters {sorted(bad)}")
        work = Path(o.get("work_dir") or (Path(ds.dir).parent / ".cenv_work"))
        # a fresh copy of each library per backend instance: its own static data mmap, and a
        # library already loaded in this process is never overwritten
        CEnvBackend._n += 1
        tag = hashlib.sha256(str(Path(ds.dir).resolve()).encode()).hexdigest()[:10]
        self.work = work / f"{tag}_{os.getpid()}_{CEnvBackend._n}"
        self.work.mkdir(parents=True, exist_ok=True)
        so = _binding_path(self.code_root)
        local = self.work / so.name
        shutil.copyfile(so, local)
        self.binding = _load_binding(local)
        lib = self.work / "libmm_final_harness.so"
        _cc(self.code_root, lib)
        self.lib = ctypes.CDLL(str(lib))
        err = ctypes.create_string_buffer(512)
        if self.lib.mm_h_open(str(ds.dir).encode(), err, 512) != 0:
            raise ContractError(f"harness could not open {ds.dir}: {err.value.decode()}")
        if self.lib.mm_h_sizeof_log() != 4 * len(self.core.log_fields):
            raise ContractError("Log layout parsed from mm.h does not match the harness")
        self.ref = load_module("pm_reference", self.code_root / "reference.py")
        self.refdata = self.ref.Data(ds.dir)
        cfg_names = list(self.ref.Config.names())
        if cfg_names != self.core.param_names:
            raise ContractError("reference.Config fields differ from mm.h MM_PARAM_NAMES")

        ref = self.ref

        class TracingEpisode(ref.Episode):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.trace = []

            def book_fill(self, side, qty, tau, u, book, print_px):
                px, r0 = self.orders[side].px, self.rebates
                super().book_fill(side, qty, tau, u, book, print_px)
                self.trace.append((tau, u, side, qty, px, self.rebates - r0, print_px))

        self.Episode = TracingEpisode
        self._spec = None

    # ------------------------------------------------------------------ interface
    def obs_size(self, split=None):
        return self.core.obs

    def obs_spec(self, split=None):
        if self._spec is None:
            from obs_spec import load_spec
            self._spec = load_spec(feat_meta=Path(self.ds.dir) / "feat_meta.json", mm_h=self.code_root / "mm.h",
                                   env_params=self.params)
        return self._spec

    def variant_params(self, variant, split, allow_test):
        p = dict(self.params)
        p.update(split=float(split), allow_test=1.0 if allow_test else 0.0,
                 fill_model=float(FILL_MODELS[variant["fill_model"]]), latency=float(variant["latency_s"]),
                 size_mult=float(self.params["size_mult"]) * float(variant["size_mult"]),
                 fixed_start=0.0, fixed_market=-1.0)
        return p

    def run(self, policy, markets, variant, split, allow_test):
        if variant.get("mirror"):
            raise ContractError("the C backend mirrors through a mirrored data directory, not a flag")
        if allow_test and os.environ.get("MM_FINAL") != "1":
            raise PermissionError("split 2 needs MM_FINAL=1")
        markets = np.asarray(markets, np.int64)
        lens = self.ds.markets["contract_len_s"][markets]
        out = {}
        for L in np.unique(lens):
            grp = markets[lens == L]
            for tr in self._run_group(policy, grp, int(L), variant, split, allow_test):
                out[tr["market"]] = tr
        return [out[int(m)] for m in markets]

    # ------------------------------------------------------------------ one length group
    def _run_group(self, policy, grp, L, variant, split, allow_test):
        ds, core = self.ds, self.core
        M = ds.markets
        lst = np.flatnonzero((M["split"] == split) & (M["contract_len_s"] == L))
        n = len(grp)
        if not np.array_equal(lst[:n], grp):
            raise ContractError("the C backend plays a prefix of the core's (split, length) market list")
        p = self.variant_params(variant, split, allow_test)
        p.update(cycle_stride=float(len(lst)), len_mask=float(LEN_BIT[L]))
        dt = int(p["dt"]) if p["dt"] > 0 else int({300: p["dt_5m"], 900: p["dt_15m"], 14400: p["dt_4h"]}[L])
        T = int(math.ceil((L) / dt))           # decisions from row 0 to the last row (n_rows - 1 = L)

        # 1. closed loop through the binding
        obs = np.zeros((n, core.obs), np.float32)
        act = np.zeros((n, 3), np.int32)
        rew = np.zeros(n, np.float32)
        term = np.zeros(n, np.uint8)
        trunc = np.zeros(n, np.uint8)
        b = self.binding
        c = b.vec_init(obs, act, rew, term, trunc, n, 0, data_dir=str(ds.dir), **p)
        try:
            b.vec_reset(c, 0)
            r0 = M["row_offset"][grp]
            want = np.asarray(ds.feat[r0, : core.n_feat_obs], np.float32)
            if not np.array_equal(obs[:, : core.n_feat_obs], want):
                raise ContractError("closed loop: an env is not on its expected market (first observation)")
            O = np.zeros((T, n, core.obs), np.float32)
            A = np.zeros((T, n, 3), np.int32)
            policy.reset(n)
            first = np.ones(n, bool)
            for k in range(T):
                O[k] = obs
                a = np.asarray(policy.act(obs, first), np.int32).reshape(n, 3)
                A[k] = a
                act[:] = a
                b.vec_step(c)
                first[:] = False
                done = term.astype(bool)
                if k < T - 1 and done.any():
                    raise ContractError(f"closed loop: terminal at step {k} before {T}")
            if not term.astype(bool).all():
                raise ContractError(f"closed loop: not every env ended after {T} steps")
        finally:
            b.vec_close(c)

        # 2. C harness replay
        vals = np.array([p[k] for k in core.param_names], np.float64)
        ho = np.zeros((T + 1, n, core.obs), np.float32)
        hr = np.zeros((T, n), np.float32)
        ht = np.zeros((T, n), np.float32)
        hs = np.zeros((T + 1, n, core.nstate), np.float64)
        hl = np.zeros((n, len(core.log_fields)), np.float32)
        err = ctypes.create_string_buffer(512)
        P = lambda x: x.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        mk = np.ascontiguousarray(grp, np.int32)
        st = np.zeros(n, np.int32)
        rc = self.lib.mm_h_run(P(vals), n, T, P(mk), P(st), P(np.ascontiguousarray(A)), ctypes.c_uint64(0),
                               P(ho), P(hr), P(ht), P(hs), P(hl), err, 512)
        if rc != 0:
            raise ContractError(f"harness: {err.value.decode()}")
        if not np.array_equal(ho[:T], O):
            raise ContractError("harness replay diverged from the closed loop (observations differ)")
        if not (ht[T - 1] != 0).all() or (ht[: T - 1] != 0).any():
            raise ContractError("harness replay: terminal flags differ from the closed loop")
        si = core.state_index

        # 3. reference replay per market
        cfg = self.ref.Config(**{k: p[k] for k in core.param_names})
        out = []
        for i, m in enumerate(grp):
            m = int(m)
            ep = self.Episode(self.refdata, cfg, market=m, start=0)
            nrow = int(M["n_rows"][m])
            rows0 = int(M["row_offset"][m])
            srow = np.zeros(T, np.int64)
            sv = np.zeros((T, 6))
            sl = np.zeros((T, 4))
            dmax = 0.0
            for k in range(T):
                ep.step(A[k, i])
                srow[k] = ep.t - rows0
                sv[k] = (ep.cash, ep.inv, ep.rebates, ep.liq, ep.wealth, 0.0)
                o0, o1 = ep.orders
                sl[k] = (float(o0.live), o0.px / 1000.0, float(o1.live), o1.px / 1000.0)
                if k < T - 1:
                    cs = hs[k + 1, i]
                    for j, name in enumerate(("cash", "inv", "rebates", "liq", "wealth")):
                        dmax = max(dmax, abs(cs[si[name]] - sv[k, j]))
                    if int(cs[si["t"]]) - rows0 != srow[k]:
                        raise ContractError(f"market {m}: reference and C rows differ at step {k}")
            log = dict(zip(core.log_fields, hl[i].astype(float)))
            log_diff = abs(log["pnl"] - ep.wealth)
            # per-row trace (rows inside a multi-second step hold the previous step's state)
            rows = np.arange(nrow)
            j = np.searchsorted(srow, rows, side="right") - 1
            has = j >= 0
            jj = np.maximum(j, 0)

            def col(v, fill=0.0):
                return np.where(has, v[jj], fill)
            cash = col(sv[:, 0] + sv[:, 2] + sv[:, 3])
            tr = dict(market=m, settlement=float(M["winner_up"][m]), cash=cash, inv=col(sv[:, 1]),
                      reward_cum=col(sv[:, 3]), rebate_cum=col(sv[:, 2]), fee_cum=np.zeros(nrow),
                      bid_live=col(sl[:, 0]), ask_live=col(sl[:, 2]),
                      bid_px=np.where(col(sl[:, 0]) > 0.5, col(sl[:, 1], np.nan), np.nan),
                      ask_px=np.where(col(sl[:, 2]) > 0.5, col(sl[:, 3], np.nan), np.nan),
                      log=log, checks=dict(state_max_abs_diff=float(dmax), log_pnl_abs_diff=float(log_diff),
                                           steps=T, dt=dt, obs_bit_identical=True))
            if dmax > STATE_TOL or log_diff > LOG_TOL + 1e-6 * abs(log["pnl"]):
                tr["checks"]["replay_ok"] = False
            else:
                tr["checks"]["replay_ok"] = True
            f = np.array(ep.trace, np.float64).reshape(-1, 7)
            tr["fills"] = dict(row=f[:, 0].astype(np.int64), side=np.where(f[:, 2] == 0, 1, -1).astype(np.int64),
                               qty=f[:, 3], price=f[:, 4] / 1000.0, fee=np.zeros(len(f)), rebate=f[:, 5],
                               print_px=f[:, 6] / 1000.0)
            out.append(tr)
        return out


def default_env_params(code_root=CODE_ROOT) -> dict:
    """The core's parameter defaults (mm.h), for the plan."""
    return Core(code_root).defaults

