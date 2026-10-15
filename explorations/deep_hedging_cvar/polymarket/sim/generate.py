#!/usr/bin/env python3
"""Generate synthetic Polymarket up/down markets in the replay env's on-disk format.

Writes to --out (default data/build_sim/), byte-compatible with data/build/:
  ticks.bin     float32 [N, 40], columns as data/build/meta.json (book in the Up frame, trade aggregates at
                block time, Binance underlying columns)
  markets.bin   88-byte MarketRec per market (data/README.md), sorted by (split, open_ts, slug); split = 0
                (synthetic markets are training material), flags |= 16 (SYNTHETIC)
  meta.json     same keys as data/build/meta.json (+ "synthetic": generator settings and params sha256)
  splits.json   {slug: 0}
  fills.parquet every fill leg in data/raw/trades.parquet's schema (taker + maker legs; maker legs at resting
                prices, timestamp = block time = match + 3..4 s), the input data/features.py expects
  sim_markets.parquet  per-market generator draws (coin, length, segment, strike, randomisation knobs)

Schedules:
  --schedule random  markets with random coin / length / aligned open time inside the cached Binance days
                     (sim/cache/binance, excludes 2026-08-22..25)
  --schedule slice   random 4 h UTC-aligned slices with EVERY market of each coin x length live in them
                     (realistic concurrency / cross-coin correlation)
  --schedule train   the 431 train markets' exact schedule and Binance paths, --reps replications
                     (for the sim-vs-replay consistency gate; reads Binance < 2026-08-23 only)

    python sim/generate.py --n-markets 2000 --coins btc eth sol xrp --lengths 5m 15m 4h --seed 1 --randomize
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import common as cm
import engine as en
import fetch_binance as fb

WARMUP_S = 3600                         # EWMA vol seed window before the first market open in a segment
ROWS_PER_BATCH = 2_000_000


# --------------------------------------------------------------------------------------------- schedules
def load_segments(coins, start=None, end=None):
    """{coin: [(t0, close, sigma), ...]} contiguous cached Binance segments."""
    out = {}
    for c in coins:
        segs = fb.load_closes(c, start, end)
        if not segs:
            raise SystemExit(f"no cached Binance days for {c}; run sim/fetch_binance.py --coins {c}")
        out[c] = [(t0, a, cm.coin_sigma(a)) for t0, a in segs]
    return out


def _windows(segs, L):
    """All aligned [open, end] windows of length L fully inside a segment (after warm-up)."""
    res = []
    for si, (t0, a, _) in enumerate(segs):
        lo, hi = t0 + WARMUP_S, t0 + len(a) - 1
        first = -(-lo // L) * L
        opens = np.arange(first, hi - L + 1, L, dtype=np.int64)
        res.append((si, opens))
    return res


def schedule_random(n, coins, lens, weights, segs, rng):
    rows = []
    li = rng.choice(len(lens), n, p=weights)
    ci = rng.integers(0, len(coins), n)
    for k in range(n):
        c, L = coins[ci[k]], lens[li[k]]
        wins = _windows(segs[c], L)
        tot = sum(len(o) for _, o in wins)
        j = rng.integers(tot)
        for si, o in wins:
            if j < len(o):
                rows.append(dict(coin=c, L=L, open_ts=int(o[j]), seg=si))
                break
            j -= len(o)
    return pd.DataFrame(rows)


def schedule_slice(n, coins, lens, segs, rng, slice_s=4 * 3600):
    """Pick random 4 h-aligned slices common to all coins; every market of every coin x length in them."""
    per_slice = len(coins) * sum(slice_s // L for L in lens)
    n_slices = max(1, int(round(n / per_slice)))
    # slices available in all coins' segments
    cand = None
    for c in coins:
        s = set()
        for t0, a, _ in segs[c]:
            lo, hi = t0 + WARMUP_S, t0 + len(a) - 1
            first = -(-lo // slice_s) * slice_s
            s |= set(range(first, hi - slice_s + 1, slice_s))
        cand = s if cand is None else cand & s
    cand = np.array(sorted(cand))
    pick = rng.choice(cand, min(n_slices, len(cand)), replace=False)
    rows = []
    for s0 in pick:
        for c in coins:
            si = next(i for i, (t0, a, _) in enumerate(segs[c]) if t0 + WARMUP_S <= s0 and s0 + slice_s <= t0 + len(a) - 1)
            for L in lens:
                for o in range(int(s0), int(s0) + slice_s, L):
                    rows.append(dict(coin=c, L=L, open_ts=o, seg=si))
    return pd.DataFrame(rows)


def schedule_train(reps):
    """The train split's schedule (all 8 coins) on the train Binance paths; reps replications."""
    _, tr, meta = cm.train_build()
    coins = {v: k for k, v in meta["coin_ids"].items()}
    base = pd.DataFrame(dict(coin=[coins[int(c)] for c in tr["coin_id"]], L=tr["contract_len_s"].astype(int),
                             open_ts=tr["open_ts"].astype(np.int64), seg=0,
                             real_slug=[meta["slugs"][int(i)] for i in tr["market_idx"]]))
    return pd.concat([base.assign(rep=r) for r in range(reps)], ignore_index=True)


# --------------------------------------------------------------------------------------------- pricing
def price_markets(sch: pd.DataFrame, segs) -> list[dict]:
    out = []
    for r in sch.itertuples():
        t0, a, sg = segs[r.coin][r.seg]
        pr = cm.market_pricing(a, sg, t0, r.open_ts, r.open_ts + r.L)
        out.append(pr)
    return out


# --------------------------------------------------------------------------------------------- run
def run_batches(model, sch, prices, rng, randomize, on_batch):
    """Simulate the schedule grouped by length in batches; on_batch(idx, result, knobs) per batch."""
    for L in sorted(sch.L.unique()):
        idx_all = np.flatnonzero(sch.L.values == L)
        bs = max(1, ROWS_PER_BATCH // (L + 1))
        for s in range(0, len(idx_all), bs):
            idx = idx_all[s:s + bs]
            inp = dict(p=np.stack([prices[i]["p"] for i in idx]), S=np.stack([prices[i]["S"] for i in idx]),
                       S_prev=np.stack([prices[i]["S_prev"] for i in idx]),
                       sigma=np.stack([prices[i]["sigma"] for i in idx]),
                       K=np.array([prices[i]["K"] for i in idx]), twap_end=np.array([prices[i]["twap_end"] for i in idx]),
                       coin_id=np.array([cm.COINS.index(c) for c in sch.coin.values[idx]]))
            knobs = en.draw_knobs(model, len(idx), rng, randomize)
            res = en.simulate(model, int(L), inp, rng, knobs)
            on_batch(idx, res, knobs)


def write_build(out: Path, model, sch, prices, rng, randomize, args_dict):
    out.mkdir(parents=True, exist_ok=True)
    sch = sch.copy()
    sch["end_ts"] = sch.open_ts + sch.L
    tag = f"s{args_dict.get('seed', 0)}"
    if "rep" in sch:
        sch["slug"] = [f"sim{tag}-r{r}-{s}" for r, s in zip(sch.rep, sch.real_slug)]
    else:
        sch["slug"] = [f"sim{tag}-{c}-updown-{cm.LEN_NAME[L]}-{o}-{k}" for k, (c, L, o) in
                       enumerate(zip(sch.coin, sch.L, sch.open_ts))]
    order = np.lexsort((sch.slug.values, sch.open_ts.values))
    sch = sch.iloc[order].reset_index(drop=True)
    prices = [prices[i] for i in order]
    n_rows = (sch.L.values + 1).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    N = int(n_rows.sum())
    ticks = np.lib.format.open_memmap(out / "_ticks.npy", mode="w+", dtype=np.float32, shape=(N, cm.F))
    recs = np.zeros(len(sch), dtype=cm.MARKET_DTYPE)
    legs_parts = []
    knob_rows = [None] * len(sch)
    extra = {}
    fees = model.fees

    def on_batch(idx, res, knobs):
        T = res["ticks"]
        for j, i in enumerate(idx):
            ticks[offs[i]: offs[i] + n_rows[i]] = T[j]
        legs = en.trade_legs(model, res["trades"], rng)
        if len(legs["b"]):
            legs_parts.append(pd.DataFrame(dict(m=idx[legs["b"]], stamp=legs["stamp"], trade=legs["trade"],
                                                outcome_idx=legs["outcome_idx"], side=legs["side"],
                                                price=legs["price"], size=legs["size"], is_taker=legs["is_taker"],
                                                aggressor_up=legs["aggressor_up"])))
        for j, i in enumerate(idx):
            knob_rows[i] = {k: float(v[j]) for k, v in knobs.items()}
            extra[i] = dict(winner_up=int(res["winner_up"][j]), fine=bool(res["fine_final"][j]),
                            ptb=float(res["ptb"][j]), ls=float(res["log_strike_vs_ul_open"][j]),
                            n_tr=int(res["n_trades_in_window"][j]))

    t0 = time.time()
    run_batches(model, sch, prices, rng, randomize, on_batch)
    ticks.flush()
    sim_time = time.time() - t0
    rw = fees["rewards"]
    for i, r in enumerate(sch.itertuples()):
        e, kn = extra[i], knob_rows[i]
        rp = rw[cm.LEN_NAME[r.L]]
        pair = np.asarray(rp["pairs"])[rng.choice(len(rp["p"]), p=np.asarray(rp["p"]) / np.sum(rp["p"]))]
        fee_rate = 0.0 if kn["fee_free"] else fees["taker_fee_rate"] * kn["fee_rate_mult"]
        flags = ((cm.FLAG_HAS_TRADES if e["n_tr"] > 0 else 0) | cm.FLAG_HAS_UNDERLYING | cm.FLAG_HAS_STRIKE
                 | (cm.FLAG_UL_PERP_PROXY if r.coin == "hype" else 0) | cm.FLAG_SYNTHETIC)
        recs[i] = (offs[i], r.open_ts, r.end_ts, n_rows[i], r.L, cm.COINS.index(r.coin), e["winner_up"], 0, flags,
                   0.01, fee_rate, fees["fee_exponent"], fees["min_order_size"], pair[0], pair[1], e["ls"],
                   fees["maker_rebate_rate"] * kn["rebate_mult"], i, 0.001 if e["fine"] else 0.01)
    del ticks
    # raw float32 [N, F] (drop the .npy header) -> ticks.bin
    arr = np.load(out / "_ticks.npy", mmap_mode="r")
    with open(out / "ticks.bin", "wb") as f:
        for s in range(0, N, 1 << 20):
            f.write(np.ascontiguousarray(arr[s: s + (1 << 20)]).tobytes())
    del arr
    (out / "_ticks.npy").unlink()
    recs.tofile(out / "markets.bin")

    # fills table (trades.parquet schema)
    if legs_parts:
        lg = pd.concat(legs_parts, ignore_index=True)
    else:
        lg = pd.DataFrame(columns=["m", "stamp", "trade", "outcome_idx", "side", "price", "size", "is_taker",
                                   "aggressor_up"])
    seed_bytes = hashlib.sha256(json.dumps({k: v for k, v in args_dict.items() if not k.startswith("_")},
                                           sort_keys=True, default=str).encode()).digest()
    txid = (lg.m.values.astype(np.int64) << 24) + lg.trade.values.astype(np.int64)
    tx = [hashlib.sha256(seed_bytes + int(v).to_bytes(8, "little")).digest() for v in np.unique(txid)]
    uniq, inv = np.unique(txid, return_inverse=True)
    tx_arr = np.array(tx, dtype=object)[inv] if len(uniq) else np.array([], dtype=object)
    fills = pa.table({
        "slug": pa.array(sch.slug.values[lg.m.values.astype(int)]).dictionary_encode(),
        "timestamp": pa.array((sch.open_ts.values[lg.m.values.astype(int)] + lg.stamp.values).astype(np.int64)),
        "outcome_idx": pa.array(lg.outcome_idx.values.astype(np.int8)),
        "side": pa.array(lg.side.values.astype(np.int8)),
        "price": pa.array(lg.price.values.astype(np.float32)),
        "size": pa.array(lg["size"].values.astype(np.float32)),
        "is_taker": pa.array(lg.is_taker.values.astype(bool)),
        "aggressor_up": pa.array(lg.aggressor_up.values.astype(np.int8)),
        "tx": pa.array(list(tx_arr), type=pa.binary(32)),
    })
    pq.write_table(fills, out / "fills.parquet", compression="zstd")

    # per-market draws
    sm = sch.drop(columns=[c for c in ("seg",) if c in sch]).copy()
    sm["winner_up"] = [extra[i]["winner_up"] for i in range(len(sch))]
    sm["price_to_beat"] = [extra[i]["ptb"] for i in range(len(sch))]
    sm["K_binance"] = [prices[i]["K"] for i in range(len(sch))]
    sm["twap_end_binance"] = [prices[i]["twap_end"] for i in range(len(sch))]
    sm["n_taker_tx_in_window"] = [extra[i]["n_tr"] for i in range(len(sch))]
    for k in en.DEFAULT_KNOBS:
        sm[f"knob_{k}"] = [knob_rows[i][k] for i in range(len(sch))]
    sm.to_parquet(out / "sim_markets.parquet", index=False)

    # meta + splits, mirroring data/build/meta.json
    real_meta = json.loads((cm.BUILD / "meta.json").read_text())
    layout = real_meta["market_record"]
    layout["flags"]["SYNTHETIC"] = cm.FLAG_SYNTHETIC
    meta = {k: real_meta[k] for k in ("columns", "column_index", "coin_ids", "contract_len_s", "sentinels",
                                      "conventions", "normalisation_train_only")}
    meta.update({
        "build_version": real_meta["build_version"], "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "F": cm.F, "N": N, "n_markets": int(len(recs)), "dtype": "float32", "order": "row-major [N, F]",
        "market_record": layout,
        "splits": {"ids": real_meta["splits"]["ids"], "rule": "synthetic: every market is split 0 (train material)",
                   "counts": {"train": {"markets": int(len(recs)), "rows": N,
                                        **{k: int((sch.L == v).sum()) for k, v in cm.CLEN.items()}}}},
        "market_order": "sorted by (split, open_ts, slug); each split's rows are contiguous",
        "slugs": list(sch.slug), "test_rows": {"row_lo": 0, "row_hi": 0, "sha256": None},
        "row_counts": {"ticks": N, "markets": int(len(recs))},
        "synthetic": {"generator": "sim/generate.py", "args": {k: v for k, v in args_dict.items() if not k.startswith("_")},
                      "params_sha256": hashlib.sha256(cm.PARAMS.read_bytes()).hexdigest(),
                      "params_fitted_on": model.raw["meta"]["fitted_on"],
                      "underlying": args_dict["_underlying"],
                      "fills": "fills.parquet (data/raw/trades.parquet schema)",
                      "differences_vs_replay": "no crossed/stale capture artefacts (crossed_raw = book_from_down = 0, "
                                               "book_valid = 1); strike = Binance TWAP60 at open * exp(coin basis); "
                                               "winner from Binance TWAP60 at end"}})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "splits.json").write_text(json.dumps({s: 0 for s in sch.slug}, indent=0, sort_keys=True))
    return dict(N=N, markets=len(recs), legs=len(lg), sim_s=sim_time)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=1000)
    ap.add_argument("--lengths", nargs="+", default=["5m", "15m", "4h"], choices=list(cm.CLEN))
    ap.add_argument("--length-weights", nargs="+", type=float, default=None,
                    help="random schedule only; default = train mix (248/146/37)")
    ap.add_argument("--coins", nargs="+", default=["btc", "eth", "sol", "xrp"])
    ap.add_argument("--start", default=None, help="first UTC day of Binance history to sample (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last UTC day (inclusive)")
    ap.add_argument("--schedule", choices=["random", "slice", "train"], default="random")
    ap.add_argument("--reps", type=int, default=1, help="train schedule: replications")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=True,
                    help="domain randomisation (params.json 'randomization' table)")
    ap.add_argument("--out", default=str(cm.OUT_DEFAULT))
    ap.add_argument("--features", action=argparse.BooleanOptionalAction, default=True,
                    help="run data/features.py on the output (via sim/features_sim.py): feat/book/fills/fill_index")
    a = ap.parse_args()
    t0 = time.time()
    model = en.Model(cm.load_params())
    rng = np.random.default_rng(a.seed)
    lens = [cm.CLEN[x] for x in a.lengths]
    if a.schedule == "train":
        sch = schedule_train(a.reps)
        ul = cm.train_underlying()
        segs = {c: [ul[c]] for c in ul}
        a._underlying = {"source": "train", "note": "fair_value.load_underlying(): Binance 2026-08-20..22"}
    else:
        segs = load_segments(a.coins, a.start, a.end)
        t0s = {c: [s[0] for s in v] for c, v in segs.items()}
        if any(len(v) != 1 for v in t0s.values()) or len({v[0] for v in t0s.values()}) != 1:
            raise SystemExit(f"need one contiguous cached segment common to all coins (got {t0s}); "
                             "narrow --start/--end")
        a._underlying = {"source": "cache", "t0": int(next(iter(t0s.values()))[0]), "coins": a.coins,
                         "start": a.start, "end": a.end, "days": len(segs[a.coins[0]][0][1]) // 86400}
        if a.schedule == "random":
            w = a.length_weights or [{300: 248, 900: 146, 14400: 37}[L] for L in lens]
            w = np.asarray(w, float) / np.sum(w)
            sch = schedule_random(a.n_markets, a.coins, lens, w, segs, rng)
        else:
            sch = schedule_slice(a.n_markets, a.coins, lens, segs, rng)
    print(f"schedule: {len(sch)} markets {sch.groupby('L').size().to_dict()}  {time.time()-t0:.1f}s")
    prices = price_markets(sch, segs)
    print(f"priced  {time.time()-t0:.1f}s")
    info = write_build(Path(a.out), model, sch, prices, rng, a.randomize, vars(a))
    print(f"wrote {a.out}: {info['markets']} markets, {info['N']} rows, {info['legs']} fill legs; "
          f"engine {info['sim_s']:.1f}s, total {time.time()-t0:.1f}s", flush=True)
    if a.features:
        import subprocess
        import sys
        subprocess.run([sys.executable, str(cm.HERE / "features_sim.py"), str(Path(a.out).resolve())],
                       check=True, cwd=str(cm.HERE))
        print(f"features done, total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
