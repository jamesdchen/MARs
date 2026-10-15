"""No-RL pre-game quoting baselines on recorded sports games (track S, eval/DECISIONS.md D11).

One command: build the recorder output into env binaries with build_sports.py (episodes end at
kickoff, inventory settles at the last pre-kickoff mid), then play a small fixed grid of
fixed-rule quoters through the patched env (env_sports/: the central mm.h fill model,
fractional settlement), one env per market from offset 0, and report per market and aggregated
by league and overall.

    python baseline_pregame.py --rec rec_out --out baseline_out            # real: kicked-off games
    python baseline_pregame.py --rec rec_out --out baseline_test --test    # mechanics test, open games

Eligible games (real mode): kickoff >= --settle-wait (30 min) in the past AND finalized trades in
the recorder output (the slug appears in rec_out/trades). Trades come from the recorder only
(build_sports --no-fetch-trades). --test (TESTING ONLY) uses build_sports --allow-open: open games
are cut at their last book row ("kickoff" = last book row) and trades come from data-api.

Strategy grid (fixed, no tuning; the full list is STRATEGIES below), pull_s in {0, 300, 900, 1800, 3600}:
  no_quote
  join_s{10,100}_pull{pull_s}     join best bid and best ask
  inside_s{10,100}_pull{pull_s}   one tick inside each touch when the spread is >= 3 ticks; at exactly
                                  2 ticks only the side that reduces |inventory| improves (both one
                                  tick inside would lock), with zero inventory both join; 1 tick: join
Every strategy quotes only on a two-sided book, holds at most Q = 100 shares (env cap), and pulls
all quotes pull_s seconds before kickoff (obs-8 kickoff clock; 0 = at kickoff). The pull time is a
val-selected hyperparameter (DECISIONS.md D15), not a build cut. Headline P&L ("pnl_flat"): the
inventory left is crossed out at the pull (or at the episode end, if earlier): marked at the mid
there minus half the spread x |inv|. "pnl" is the env's (marked at the kickoff mid, no crossing).
Sizes: 10 shares = size head 1 (2 x min size 5), 100 shares = size head 2 x size_mult 4 (5 x 5 x 4);
the effective size is reported per market (gamma orderMinSize may differ from 5).
Episodes: 4 h segments tiling [kickoff - max_hours, kickoff] (default 24 h); each is reported in
its majority time-to-kickoff bucket (>24h, 6-24h, 1-6h, 1h-30m, 30-5m, 5-0m; build_sports meta.json),
plus overall and by league.

Env: dt = 1 s decisions for every length, latency 1 s, fill model central (headline) and
pessimistic (sensitivity), Q 100, rebates from markets.bin (gamma feeSchedule rate x rebateRate),
liquidity rewards off.
"""
from __future__ import annotations

import argparse
import glob
import importlib.machinery
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PM = HERE.parent.parent                       # polymarket/
ENV_DIR = HERE / "env_sports"
sys.path.insert(0, str(PM / "recorder"))
sys.path.insert(0, str(PM / "data"))
sys.path.insert(0, str(PM / "eval"))
sys.path.insert(0, str(PM / "train"))

FILL_MODELS = {"optimistic": 0, "central": 1, "pessimistic": 2}
SEG = 14400
LOG_TAU_DEN = math.log1p(14400.0)             # obs 8 = log1p(seconds to kickoff) / log1p(14400)
# obs indices (ENV.md section 6)
O_LOG_TAU, O_SPREAD, O_BID, O_ASK, O_INV = 8, 11, 12, 13, 33

PULLS_S = (0, 300, 900, 1800, 3600)            # D15: the pull before kickoff is a val-selected hyperparameter
BUCKET_ORDER = [">24h", "6-24h", "1-6h", "1h-30m", "30-5m", "5-0m"]   # build_sports.BUCKETS

# name -> (kind, shares, pull seconds before kickoff)
STRATEGIES = {"no_quote": ("off", 10, 0)}
for _k in ("join", "inside"):
    for _s in (10, 100):
        for _n in PULLS_S:
            STRATEGIES[f"{_k}_s{_s}_pull{_n}"] = (_k, _s, _n)
SIZE_ACT = {10: (1, 1.0), 100: (2, 4.0)}      # shares -> (size head, size_mult) at min size 5


# ----------------------------------------------------------------------------- data
def finalized_slugs(rec: Path) -> set[str]:
    from polyrec.sinks import read_dataset
    t = read_dataset(rec, "trades", columns=["slug"])
    return set() if t is None else set(t.column("slug").to_pylist())


def run_build(args, out: Path) -> Path:
    bdir = out / "build"
    splits = out / "splits_all_val.json"
    # exploratory only: every game day goes to split 1 so the env opens it without the test guard;
    # no D11 split is fixed yet and this is NOT a val/test assignment.
    splits.write_text(json.dumps({"val": ["2000-01-01..2100-12-31"]}))
    cmd = [sys.executable, str(HERE / "build_sports.py"), "--out", str(bdir), "--dt", "1", "--seg", str(SEG),
           "--max-hours", str(args.max_hours), "--min-cover", str(args.min_cover), "--splits", str(splits),
           "--settle-wait", str(args.settle_wait), "--verify"]
    for r in args.rec:
        cmd += ["--rec", r]
    if args.test:
        cmd += ["--allow-open"]
        if args.max_markets:
            cmd += ["--max-markets", str(args.max_markets)]
        if args.slugs:
            cmd += ["--slugs", args.slugs]
    else:
        slugs = set()
        for r in args.rec:
            slugs |= finalized_slugs(Path(r))
        if args.slugs:
            slugs = {s for s in slugs if re.search(args.slugs, s)}
        print(f"slugs with finalized recorder trades: {len(slugs)}")
        if not slugs:
            raise SystemExit("no game has finalized trades in the recorder output yet (rec_out/trades); "
                             "rerun after kickoff + ~30 min")
        cmd += ["--no-fetch-trades", "--slugs", "^(" + "|".join(re.escape(s) for s in sorted(slugs)) + ")$"]
        if args.max_markets:
            cmd += ["--max-markets", str(args.max_markets)]
    if args.offline:
        cmd += ["--offline"]
    print("build:", " ".join(cmd[:14]), "...", flush=True)
    r = subprocess.run(cmd, cwd=str(HERE), text=True, capture_output=True)
    (out / "build.log").write_text(r.stdout + "\n" + r.stderr)
    print("\n".join(r.stdout.strip().splitlines()[-6:]))
    if r.returncode != 0:
        raise SystemExit(f"build_sports failed (see {out / 'build.log'}):\n{r.stderr[-3000:]}")
    return bdir


def ensure_env() -> None:
    so = glob.glob(str(ENV_DIR / "binding*.so"))
    src_t = max((ENV_DIR / f).stat().st_mtime for f in ("mm.h", "binding.c"))
    if not so or Path(so[0]).stat().st_mtime < src_t:
        r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(ENV_DIR),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("env_sports build failed:\n" + r.stderr[-3000:])


def load_binding():
    so = sorted(glob.glob(str(ENV_DIR / "binding*.so")))[0]
    loader = importlib.machinery.ExtensionFileLoader("binding", so)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("binding", so, loader=loader))
    loader.exec_module(mod)
    return mod


def core_defaults() -> dict:
    src = (ENV_DIR / "mm.h").read_text()
    names = re.findall(r'"(\w+)"', re.search(r"MM_PARAM_NAMES\[\] = \{(.*?)\};", src, re.S)[1])
    vals = [float(x) for x in re.search(r"MM_DEFAULTS\[\] = \{(.*?)\};", src, re.S)[1].split(",") if x.strip()]
    assert len(names) == len(vals)
    return dict(zip(names, vals))


# ----------------------------------------------------------------------------- policies
def make_policy(kind: str, shares: int, pull_s: float):
    size_head = SIZE_ACT[shares][0]

    def pol(obs: np.ndarray) -> np.ndarray:
        n = len(obs)
        act = np.zeros((n, 3), np.int32)
        act[:, 2] = size_head
        if kind == "off":
            return act
        two = (obs[:, O_BID] > 0.5) & (obs[:, O_ASK] > 0.5)
        secs = np.expm1(obs[:, O_LOG_TAU].astype(np.float64) * LOG_TAU_DEN)
        on = two & (secs > pull_s + 1e-6)
        bid = np.full(n, 3, np.int32)
        ask = np.full(n, 3, np.int32)
        if kind == "inside":
            s = np.rint(np.exp(2.0 * obs[:, O_SPREAD].astype(np.float64)))
            inv = obs[:, O_INV]
            both = s >= 3
            bid[both | ((s == 2) & (inv < 0))] = 2
            ask[both | ((s == 2) & (inv > 0))] = 2
        act[:, 0] = np.where(on, bid, 0)
        act[:, 1] = np.where(on, ask, 0)
        return act
    return pol


# ----------------------------------------------------------------------------- runner
class Runner:
    """One env per market, all from offset 0 (cycle_stride = n, env seed i -> market i), stepped together;
    each env's Log is read through its own handle at its first terminal (baselines/runner.py pattern).
    Besides the Log it tracks the inventory obs per step to rebuild the signed fill flow per print row,
    which gives the 5-min markout (the Log has 5/30/60 s); the rebuilt 30 s markout is checked against
    the Log's."""

    def __init__(self, b, bdir: Path, fill_model: int, size_mult: float, extra: dict | None = None):
        from load import load as load_data
        self.b = b
        _, markets, meta = load_data(bdir)
        self.markets, self.meta = markets, meta
        self.idx = np.flatnonzero(markets["split"] == 1)
        n = self.n = len(self.idx)
        fm = json.loads((bdir / "feat_meta.json").read_text())
        self.feat = np.memmap(bdir / "feat.bin", np.float32, "r", shape=(fm["N"], fm["G"]))
        self.mark_col = 34
        self.sko = np.fromfile(bdir / "sports_feat.bin", np.float32).reshape(fm["N"], -1)[:, 0]  # s to kickoff
        tk = np.memmap(bdir / "ticks.bin", np.float32, "r", shape=(meta["N"], meta["F"]))
        ci = meta["column_index"]
        bb, aa = np.asarray(tk[:, ci["bid_px_0"]], np.float64), np.asarray(tk[:, ci["ask_px_0"]], np.float64)
        two = (np.asarray(tk[:, ci["book_valid"]]) > 0) & (bb > 0) & (aa < 1)
        self.half_spread = np.where(two, 0.5 * (aa - bb), np.nan)     # forward-filled per market in flatten()
        p = {**core_defaults(), "split": 1, "fill_model": fill_model, "latency": 1, "dt": 1, "Q": 100,
             "size_mult": size_mult, "fixed_start": 0, "cycle_stride": n, "fixed_market": -1,
             "liq_rewards": 0, **(extra or {})}
        self.params = p
        self.D = int(math.ceil((p["match_delay"] if p["match_delay"] >= 0 else {0: 0.0, 1: 3.5, 2: 5.0}[fill_model])
                               + 0.5))
        self.Q = p["Q"]
        self.obs = np.zeros((n, 50), np.float32)
        self.act = np.zeros((n, 3), np.int32)
        self.rew = np.zeros(n, np.float32)
        self.term = np.zeros(n, np.uint8)
        self.trunc = np.zeros(n, np.uint8)
        kw = {k: float(v) for k, v in p.items()}
        self.handles = [b.env_init(self.obs[i:i + 1], self.act[i:i + 1], self.rew[i:i + 1], self.term[i:i + 1],
                                   self.trunc[i:i + 1], i, data_dir=str(bdir), **kw) for i in range(n)]
        self.vec = b.vectorize(*self.handles)
        self.single = [b.vectorize(h) for h in self.handles]
        b.vec_reset(self.vec, 0)
        k = int(fm["n_obs_features"])
        want = self.feat[markets["row_offset"][self.idx].astype(np.int64), :k]
        bad = np.flatnonzero(~(self.obs[:, :k] == want).all(1))
        if bad.size:
            raise RuntimeError(f"envs {bad[:10].tolist()} are not on their expected markets at offset 0")

    def flatten(self, recs: list[dict], pull_s: float) -> None:
        """D15 flatten cost: the inventory is crossed out at the pull (pull_s before kickoff; the episode end if that
        comes first): marked at the mid there minus half the spread x |inv| (the last two-sided spread, else 1 c)."""
        off = self.markets["row_offset"][self.idx].astype(np.int64)
        nrows = self.markets["n_rows"][self.idx].astype(np.int64)
        for i, rec in enumerate(recs):
            o, n = int(off[i]), int(nrows[i])
            sk = self.sko[o:o + n]
            fr = int(np.argmax(sk <= pull_s + 1e-6)) if (sk <= pull_s + 1e-6).any() else n - 1
            hs = self.half_spread[o:o + fr + 1]
            hs = hs[np.isfinite(hs)]
            h = float(hs[-1]) if len(hs) else 0.005
            mid_f = float(self.feat[o + fr, self.mark_col])
            settle = float(self.feat[o + n - 1, self.mark_col])
            inv = float(rec["final_inv"])
            rec["flat_row"] = fr
            rec["flat_half_spread_c"] = 100 * h
            rec["flat_cost"] = abs(inv) * h
            rec["inv_mark_shift"] = inv * (mid_f - settle)        # re-mark from the kickoff mid to the pull mid
            rec["pnl_flat"] = rec["pnl"] + rec["inv_mark_shift"] - rec["flat_cost"]

    def run(self, policy) -> list[dict]:
        b, n = self.b, self.n
        b.vec_log(self.vec)
        b.vec_reset(self.vec, 0)
        off = self.markets["row_offset"][self.idx].astype(np.int64)
        nrows = self.markets["n_rows"][self.idx].astype(np.int64)
        mo = {h: np.zeros(n) for h in (30, 300)}
        done = np.zeros(n, bool)
        recs = [None] * n
        inv_prev = np.zeros(n)
        step = 0
        self.ko_err = 0.0          # |decoded seconds-to-kickoff (obs 8) - sports_feat| at decision rows, <= 1 h
        mark = lambda i, row: float(self.feat[off[i] + min(row, nrows[i] - 1), self.mark_col])  # noqa: E731
        while not done.all():
            if step > SEG + 5:
                raise RuntimeError("markets did not end")
            self.act[:] = policy(self.obs)
            rows = off + np.minimum(step, nrows - 1)
            sk = self.sko[rows]
            dec = np.expm1(self.obs[:, O_LOG_TAU].astype(np.float64) * LOG_TAU_DEN)
            near = ~done & (sk <= 3600)
            if near.any():
                self.ko_err = max(self.ko_err, float(np.abs(dec - sk)[near].max()))
            b.vec_step(self.vec)
            step += 1
            u = step                                   # print row of the fills booked in this step (dt = 1)
            term = (self.term != 0) & ~done
            inv_now = self.obs[:, O_INV].astype(np.float64) * self.Q
            for i in np.flatnonzero(term):
                log = {k: float(v) for k, v in b.vec_log(self.single[i]).items()}
                inv_now[i] = log["buy_vol"] - log["sell_vol"]          # final signed inventory
                recs[i] = log
            live = ~done
            dq = np.where(live, inv_now - inv_prev, 0.0)
            for i in np.flatnonzero(np.abs(dq) > 1e-6):
                tau = u - self.D
                m0 = mark(i, tau)
                for h in mo:                           # past the end: the settlement = last-row mark
                    mo[h][i] += dq[i] * (mark(i, tau + h) - m0)
            inv_prev = np.where(live, inv_now, inv_prev)
            for i in np.flatnonzero(term):
                recs[i]["markout_300"] = mo[300][i]
                recs[i]["markout_30_rebuilt"] = mo[30][i]
                recs[i]["final_inv"] = round(float(inv_now[i]), 4)
                done[i] = True
        return recs

    def close(self):
        self.b.vec_close(self.vec)


# ----------------------------------------------------------------------------- report
SUMK = ["pnl", "pnl_flat", "flat_cost", "pnl_trading", "spread_capture", "markout_30", "markout_60", "markout_300", "inventory_pnl",
        "rebates", "fills", "fill_vol", "buy_vol", "sell_vol", "rejects"]


def aggregate(df: pd.DataFrame) -> dict:
    from evaluator import protocol_flags
    recs = df.to_dict("records")
    out = {"n": len(df)}
    for k in SUMK:
        out[f"{k}_total"] = float(df[k].sum())
    out["pnl_mean"] = float(df.pnl.mean())
    out["frac_pos"] = float((df.pnl > 0).mean())
    vol = df.fill_vol.sum()
    for h in (30, 60, 300):
        out[f"mo{h}_per_sh_c"] = 100 * float(df[f"markout_{h}"].sum() / vol) if vol > 0 else float("nan")
    out["spread_per_sh_c"] = 100 * float(df.spread_capture.sum() / vol) if vol > 0 else float("nan")
    out["fills_per_mkt"] = float(df.fills.mean())
    out["vol_per_mkt"] = float(df.fill_vol.mean())
    out["abs_inv_ko_mean"] = float(df.final_inv.abs().mean())
    out["inv_ko_nonzero"] = float((df.final_inv.abs() > 1e-6).mean())
    out["uptime_two_sided"] = float(df.uptime_two_sided.mean())
    pf = protocol_flags(recs)
    out.update({k: pf.get(k) for k in ("fills_per_market", "t6_share", "t6_share_pos", "t5_max_share",
                                       "makes_markets")})
    shares = {k.split("/")[1]: v for k, v in pf.items() if k.startswith("t5_share/")}
    out["t5_worst_flag"] = max(shares, key=shares.get) if shares else None
    return out


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def table(rows: list[dict], cols: list[tuple[str, str, int]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r.get(c[0]), c[2]) for c in cols) + " |\n" for r in rows)


COLS = [("strategy", "strategy", 0), ("group", "group", 0), ("n", "n", 0), ("pnl_flat_total", "P&L flat $", 2),
        ("pnl_total", "P&L KO-mark $", 2), ("flat_cost_total", "flat cost $", 2),
        ("spread_capture_total", "spread $", 2), ("markout_60_total", "mo60 $", 2),
        ("inventory_pnl_total", "inv $", 2), ("rebates_total", "rebates $", 3), ("fills_per_mkt", "fills/mkt", 1),
        ("vol_per_mkt", "vol/mkt", 0), ("spread_per_sh_c", "spread c/sh", 2), ("mo30_per_sh_c", "mo30 c/sh", 2),
        ("mo300_per_sh_c", "mo5m c/sh", 2), ("abs_inv_ko_mean", "|inv| KO", 1), ("t6_share", "T6", 2),
        ("t5_max_share", "T5 max", 2), ("t5_worst_flag", "T5 flag", 0), ("makes_markets", "D1.3", 0)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", action="append", required=True, help="recorder out dir (repeatable)")
    ap.add_argument("--out", default=str(HERE / "baseline_out"))
    ap.add_argument("--max-hours", type=float, default=24.0, help="window before kickoff (4 h segments tiling it "
                    "backwards from kickoff; each is reported in its majority time-to-kickoff bucket)")
    ap.add_argument("--min-cover", type=float, default=0.05,
                    help="keep a 4 h segment if this fraction has a book (recording may start < 4 h before KO)")
    ap.add_argument("--settle-wait", type=int, default=1800)
    ap.add_argument("--slugs", help="regex filter on slugs")
    ap.add_argument("--max-markets", type=int, default=0)
    ap.add_argument("--fill-models", default="central,pessimistic")
    ap.add_argument("--skip-build", action="store_true", help="reuse <out>/build")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="TESTING ONLY: open games via build_sports --allow-open (cut at the last book row), "
                         "trades from data-api")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out).resolve()
    if "jun15" in str(out).lower():
        raise SystemExit("refusing a *jun15* path")
    out.mkdir(parents=True, exist_ok=True)
    bdir = out / "build" if args.skip_build else run_build(args, out)
    ensure_env()
    b = load_binding()
    from load import load as load_data
    _, markets, meta = load_data(bdir)
    sp = meta["sports"]
    assert sp["dt_s"] == 1 and sp["seg_rows"] == SEG
    leagues = {v: k for k, v in meta["coin_ids"].items()}
    sfeat = np.fromfile(bdir / "sports_feat.bin", np.float32).reshape(meta["N"], -1)
    idx = np.flatnonzero(markets["split"] == 1)
    segm = meta.get("segments")                   # build_sports sports-2: per segment kickoff_ts, bucket, settle
    base = []
    for j, i in enumerate(idx):
        m = markets[i]
        o, nr = int(m["row_offset"]), int(m["n_rows"])
        sid = meta["slugs"][i]
        sm = segm[i]
        assert sm["sid"] == sid and int(round(float(sfeat[o + nr - 1, 0]))) == sm["kickoff_ts"] - int(m["end_ts"]), sid
        base.append(dict(market=sid, game=sid.split("#")[0], league=leagues[int(m["coin_id"])],
                         kickoff_utc=pd.Timestamp(sm["kickoff_ts"], unit="s", tz="UTC").strftime("%m-%d %H:%M"),
                         bucket=sm["bucket"], bucket_pure=sm["bucket_pure"], at_kickoff=bool(sm["at_cut"]),
                         to_ko_start_h=sm["to_kickoff_start_s"] / 3600, to_ko_end_h=sm["to_kickoff_end_s"] / 3600,
                         min_size=float(m["min_order_size"]), fee_rate=float(m["taker_fee_rate"]),
                         rebate_rate=float(m["maker_rebate_rate"]), settle=float(sm["settle_price"]),
                         book_hours=float(sfeat[o + nr - 1, 3])))
    assert len(meta["slugs"]) == len(markets)
    print(f"{len(idx)} markets ({len(set(r['game'] for r in base))} games), leagues "
          f"{sorted(set(r['league'] for r in base))}; min sizes {sorted(set(r['min_size'] for r in base))}")
    rows, checks = [], {}
    for fm_name in [x for x in args.fill_models.split(",") if x]:
        fm = FILL_MODELS[fm_name]
        for mult in sorted({SIZE_ACT[v[1]][1] for v in STRATEGIES.values()}):
            r = Runner(b, bdir, fm, mult)
            for name, (kind, shares, pull) in STRATEGIES.items():
                if SIZE_ACT[shares][1] != mult:
                    continue
                ts = time.time()
                recs = r.run(make_policy(kind, shares, pull))
                r.flatten(recs, pull)
                for bm, rec in zip(base, recs):
                    rows.append({"strategy": name, "fill_model": fm_name, "kind": kind, "shares": shares,
                                 "pull_s": pull, **bm,
                                 "size_shares": SIZE_ACT[shares][1] * bm["min_size"] * (1, 2, 5)[SIZE_ACT[shares][0]],
                                 **rec})
                d = np.array([x["markout_30_rebuilt"] - x["markout_30"] for x in recs])
                checks[f"{fm_name}/{name}"] = {
                    "max_abs_mo30_rebuild_err": float(np.abs(d).max()) if len(d) else 0.0,
                    "max_abs_kickoff_clock_err_s": r.ko_err,
                    # fills in the last 30 s of the episodes ending at kickoff: must be 0 when pull >= 60 s
                    "vol_last30_at_kickoff": float(sum(x["vol_last30"] for x, bm in zip(recs, base) if bm["at_kickoff"])),
                    "wall_s": round(time.time() - ts, 2)}
                if pull >= 60 and checks[f"{fm_name}/{name}"]["vol_last30_at_kickoff"] > 0:
                    raise RuntimeError(f"{name}: fills in the last 30 s despite pulling {pull} s before kickoff")
                print(f"  {fm_name:11s} {name:18s} pnl {sum(x['pnl'] for x in recs):9.2f} flat "
                      f"{sum(x['pnl_flat'] for x in recs):9.2f}  fills "
                      f"{sum(x['fills'] for x in recs):6.0f}  mo30 rebuild err {checks[f'{fm_name}/{name}']['max_abs_mo30_rebuild_err']:.2e}  KO clock err "
                      f"{r.ko_err:.3f}s"
                      f"  {time.time() - ts:.1f}s", flush=True)
            r.close()
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_market.csv", index=False)
    summ = []
    for (fm_name, st), g in df.groupby(["fill_model", "strategy"], sort=False):
        summ.append({"fill_model": fm_name, "strategy": st, "group": "ALL", **aggregate(g)})
        for lg, gl in g.groupby("league"):
            summ.append({"fill_model": fm_name, "strategy": st, "group": lg, **aggregate(gl)})
        for bk, gb in g.groupby("bucket"):
            summ.append({"fill_model": fm_name, "strategy": st, "group": "bucket " + bk, **aggregate(gb)})
    for x in summ:
        k_, s_, p_ = STRATEGIES[x["strategy"]]
        x.update(kind=k_, shares=s_, pull_s=p_)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "summary.csv", index=False)
    res = {"mode": "TEST (--allow-open: open games cut at the last book row; data-api trades)" if args.test
           else "real (kickoff >= settle_wait past, finalized recorder trades)",
           "n_markets": len(idx), "n_games": len(set(r["game"] for r in base)), "build": str(bdir),
           "env": {"fill_models": args.fill_models, "latency_s": 1, "dt_s": 1, "Q": 100,
                   "rebates": "markets.bin taker_fee_rate x maker_rebate_rate x q x p(1-p)", "liq_rewards": 0},
           "flatten": "pnl_flat = pnl (inventory marked at the kickoff mid) + inv x (mid at the pull - kickoff mid) - "
                      "|inv| x half spread at the pull (last two-sided spread, else 1 c); pull = pull_s before kickoff "
                      "or the episode end, whichever first (D15)",
           "buckets": "each episode in its majority time-to-kickoff bucket (meta.json segments[i].bucket)",
           "strategies": {k: {"kind": v[0], "shares": v[1], "pull_s": v[2]} for k, v in STRATEGIES.items()},
           "checks": checks, "wall_s": round(time.time() - t0, 1),
           "build_stats": sp.get("stats")}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    md = [f"# Pre-game baseline ({res['mode']})\n",
          f"{len(idx)} markets, {res['n_games']} games, built {meta.get('built_at_utc')}. "
          f"Env: env_sports central/pessimistic, latency 1 s, dt 1 s, Q 100, rebates per markets.bin.\n"]
    for fm_name in df.fill_model.unique():
        md.append(f"\n## {fm_name}: overall (P&L flat = crossed out at the pull, half spread)\n")
        md.append(table([x for x in summ if x["fill_model"] == fm_name and x["group"] == "ALL"], COLS))
        md.append(f"\n## {fm_name}: P&L flat $ by pull_s x bucket\n")
        sub = [x for x in summ if x["fill_model"] == fm_name and x["strategy"] != "no_quote"
               and (x["group"] == "ALL" or x["group"].startswith("bucket "))]
        if sub:
            pv = pd.DataFrame(sub).pivot_table(index=["kind", "shares", "pull_s"], columns="group",
                                               values="pnl_flat_total", aggfunc="sum")
            order = ["ALL"] + [f"bucket {b}" for b in BUCKET_ORDER if f"bucket {b}" in pv.columns]
            pr = pv[order].reset_index()
            md.append(table(pr.to_dict("records"), [("kind", "kind", 0), ("shares", "shares", 0), ("pull_s", "pull_s", 0)]
                            + [(c, c.replace("bucket ", ""), 2) for c in order]))
        md.append(f"\n## {fm_name}: by bucket\n")
        md.append(table([x for x in summ if x["fill_model"] == fm_name and x["group"].startswith("bucket ")
                         and x["strategy"] != "no_quote"], COLS))
        md.append(f"\n## {fm_name}: by league\n")
        md.append(table([x for x in summ if x["fill_model"] == fm_name and x["group"] != "ALL"
                         and not x["group"].startswith("bucket ") and x["strategy"] != "no_quote"], COLS))
    pm = df[(df.fill_model == df.fill_model.iloc[0])]
    md.append("\n## per market (first fill model; join_s10_pull0 and join_s10_pull900)\n")
    cols = [("market", "market", 0), ("league", "lg", 0), ("kickoff_utc", "KO", 0), ("bucket", "bucket", 0),
            ("book_hours", "book h", 1), ("strategy", "strategy", 0), ("pnl_flat", "P&L flat", 2), ("pnl", "P&L", 2), ("spread_capture", "spread", 2),
            ("markout_60", "mo60", 2), ("inventory_pnl", "inv", 2), ("rebates", "reb", 3), ("fills", "fills", 0),
            ("fill_vol", "vol", 0), ("final_inv", "inv KO", 1), ("settle", "settle", 3)]
    md.append(table(pm[pm.strategy.isin(["join_s10_pull0", "join_s10_pull900"])].sort_values(
        ["market", "strategy"]).to_dict("records"), cols))
    (out / "summary.md").write_text("".join(md))
    print("".join(md[:3]))
    print(f"wrote {out}/summary.md, summary.csv, per_market.csv, results.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
