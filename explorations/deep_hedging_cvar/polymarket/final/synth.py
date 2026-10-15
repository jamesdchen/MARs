"""Synthetic tiny dataset + stand-in components for tests and dry runs (never real test rows).

    build_dataset(dir)            data/build-like folder: meta.json (synthetic: true, test_rows
                                  hash of the SYNTHETIC test rows), markets.bin, ticks.bin,
                                  splits.json, feat.bin + feat_meta.json, fills.bin + fill_index.bin.
                                  Splits: train 2026-08-21, val 08-23, test 08-24, 8 markets each
                                  (5m and 15m, btc/eth/sol), spread over ~4 h of each day.
    build_root(dir, ds_dir, ...)  polymarket-like artifact root: baselines/tuned.json,
                                  train/selected.json (5 seeds: real tiny train/policy.py MLP
                                  checkpoints when torch is available, else callable stand-ins),
                                  optional puffer5/selected.json (.bin PufferNet checkpoints),
                                  trials.jsonl ledgers in several folders.
    standin_root(dir)             artifact root for a dry run on the REAL val split before the
                                  other components exist: default baseline params, 5 "RL" seeds
                                  that are jittered fair-value quoters (callable kind).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import CODE_ROOT, S, sha256_bytes, train_policy_module, write_json

COLS = ["seconds_to_close"] + [f"bid_px_{i}" for i in range(5)] + [f"bid_sz_{i}" for i in range(5)] \
    + [f"ask_px_{i}" for i in range(5)] + [f"ask_sz_{i}" for i in range(5)] \
    + ["bid_depth_total", "ask_depth_total", "book_age", "book_valid", "crossed_raw", "book_from_down",
       "buy_vol", "buy_vwap", "buy_max_px", "sell_vol", "sell_vwap", "sell_min_px", "trade_count",
       "ul_logp_rel_open", "ul_logret_1s", "ul_logp_vs_strike", "const_one", "pad_0", "pad_1"]
MARKET_FIELDS = [("row_offset", "int64_t"), ("open_ts", "int64_t"), ("end_ts", "int64_t"),
                 ("n_rows", "int32_t"), ("contract_len_s", "int32_t"), ("coin_id", "int32_t"),
                 ("winner_up", "int32_t"), ("split", "int32_t"), ("flags", "int32_t"),
                 ("tick_size", "float"), ("taker_fee_rate", "float"), ("fee_exponent", "float"),
                 ("min_order_size", "float"), ("rewards_min_size", "float"),
                 ("rewards_max_spread", "float"), ("log_strike_vs_ul_open", "float"),
                 ("maker_rebate_rate", "float"), ("market_idx", "int32_t"), ("tick_size_final", "float")]
FEAT_OBS = ["p_logit", "gap_c", "tau_frac", "spread_ticks", "imb_l1", "flow5", "ulz5", "len_5m",
            "len_15m", "tick_fine", "crossed_raw"]
FEAT_AUX = ["p_model", "mark", "delta_usd_raw"]
DAY = {0: 1787270400, 1: 1787443200, 2: 1787529600}       # 2026-08-21, 08-23, 08-24 00:00 UTC
COIN_IDS = {"btc": 0, "eth": 1, "sol": 2, "xrp": 3, "bnb": 4, "doge": 5, "hype": 6, "zec": 7}


def _market_record():
    size = {"int64_t": 8, "int32_t": 4, "float": 4}
    fields, off = [], 0
    for n, t in MARKET_FIELDS:
        fields.append(dict(name=n, offset=off, type=t, size=size[t]))
        off += size[t]
    return dict(itemsize=off, fields=fields,
                flags={"HAS_TRADES": 1, "HAS_UNDERLYING": 2, "UL_PERP_PROXY": 4, "HAS_STRIKE": 8})


def _feat_layout(layout):
    """(columns, n_obs, feat_meta extras). 'cenv': the C core's exact layout, column names and
    docs taken from the real feat_meta.json (metadata only, no data); 'small': a short list."""
    if layout == "cenv":
        real = json.loads((CODE_ROOT / "data" / "build" / "feat_meta.json").read_text())
        keep = ("columns", "column_index", "column_docs", "n_obs_features", "book_record", "fill_record")
        return list(real["columns"]), int(real["n_obs_features"]), {k: real[k] for k in keep if k in real}
    cols = FEAT_OBS + FEAT_AUX
    return cols, len(FEAT_OBS), {}


def _feat_values(name, v):
    """Synthetic value of feature `name` from the market's raw arrays v (encodings as documented)."""
    pm, mark, stc, L = v["pm"], v["mark"], v["stc"], v["L"]
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
    two = v["bid_ok"] & v["ask_ok"]
    table = {
        "p_logit": lg(pm) / 3, "mark_logit": lg(mark) / 3, "gap_c": (pm - mark) * 100 / 5,
        "z": np.clip(v["rng"].normal(0, 1, stc.size), -5, 5) / 2, "sdv_bp": np.ones(stc.size),
        "sigma_1m_bp": np.ones(stc.size), "tau_frac": stc / L, "log_tau": np.log1p(stc) / np.log1p(14400),
        "in_twap": (stc < 60).astype(float), "delta_usd": np.full(stc.size, 0.5),
        "spread_ticks": np.where(two, np.log(np.maximum(v["ask"] - v["bid"], 0.01) / 0.01) / 2, 0.0),
        "bid_present": v["bid_ok"].astype(float), "ask_present": v["ask_ok"].astype(float),
        "imb_l1": v["rng"].uniform(-1, 1, stc.size), "bid_sz0": np.log1p(v["bsz"]) / 5,
        "ask_sz0": np.log1p(v["asz"]) / 5, "flow5": v["rng"].uniform(-1, 1, stc.size),
        "ulz5": v["rng"].normal(0, 1, stc.size), "len_5m": float(L == 300) * np.ones(stc.size),
        "len_15m": float(L == 900) * np.ones(stc.size), "len_4h": float(L == 14400) * np.ones(stc.size),
        "book_age": np.log1p(v["age"]) / 4, "crossed_raw": v["crossed"].astype(float),
        "mark_is_fv": (~two).astype(float), "p_model": pm, "mark": mark,
        "delta_usd_raw": v["rng"].uniform(10, 200, stc.size), "sigma": np.full(stc.size, 1e-4),
        "sqrt_v": np.full(stc.size, 0.01), "liq_mid": np.full(stc.size, -1.0),
        "bid_minus_p_c": np.where(v["bid_ok"], (v["bid"] - pm) * 100 / 5, 0.0),
        "ask_minus_p_c": np.where(v["ask_ok"], (v["ask"] - pm) * 100 / 5, 0.0),
    }
    return table.get(name, np.zeros(stc.size))


BOOK_FIELDS = [("bid_px", "<i2", (5,)), ("ask_px", "<i2", (5,)), ("bid_sz", "<f4", (5,)),
               ("ask_sz", "<f4", (5,)), ("flags", "u1"), ("tick", "u1"), ("tick_alt", "u1"), ("pad", "u1")]


def build_dataset(out_dir, n_per_split=8, seed=7, lengths=(300, 900), layout="small"):
    """Write a synthetic build dir; returns its Path. layout 'cenv' matches the C core's files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ci = {c: i for i, c in enumerate(COLS)}
    fcols, n_obs, fextra = _feat_layout(layout)
    specs = []
    for split in (0, 1, 2):
        for i in range(n_per_split):
            L = lengths[i % len(lengths)]
            coin = ("btc", "eth", "sol")[i % 3]
            open_ts = DAY[split] + 1800 * i + 600
            specs.append((split, open_ts, coin, L))
    specs.sort(key=lambda s: (s[0], s[1], s[2]))
    rows, feats, books, legs, mrecs, slugs = [], [], [], [], [], []
    off = 0
    for mi, (split, open_ts, coin, L) in enumerate(specs):
        n = L + 1
        x = np.cumsum(rng.normal(0, 0.04, n)) + rng.normal(0, 0.4)
        fv = 1 / (1 + np.exp(-x))
        mid = np.clip(np.round(fv / 0.01) * 0.01, 0.03, 0.97)
        half = rng.choice([1, 1, 2, 3], n) * 0.01 / 2
        bid0 = np.round((mid - half) / 0.01) * 0.01
        ask0 = np.round((mid + half) / 0.01) * 0.01
        ask0 = np.where(ask0 <= bid0, bid0 + 0.01, ask0)
        one = rng.random(n) < 0.03                       # occasional one-sided book (no asks)
        t = np.zeros((n, len(COLS)), np.float32)
        t[:, ci["seconds_to_close"]] = L - np.arange(n)
        for k in range(5):
            t[:, ci[f"bid_px_{k}"]] = np.clip(bid0 - 0.01 * k, 0, 1)
            t[:, ci[f"ask_px_{k}"]] = np.where(one, 1.0, np.clip(ask0 + 0.01 * k, 0, 1))
            t[:, ci[f"bid_sz_{k}"]] = rng.uniform(5, 80, n)
            t[:, ci[f"ask_sz_{k}"]] = np.where(one, 0.0, rng.uniform(5, 80, n))
        t[:, ci["bid_depth_total"]] = t[:, [ci[f"bid_sz_{k}"] for k in range(5)]].sum(1)
        t[:, ci["ask_depth_total"]] = t[:, [ci[f"ask_sz_{k}"] for k in range(5)]].sum(1)
        t[:, ci["book_valid"]] = 1
        crossed = rng.random(n) < 0.02
        t[:, ci["crossed_raw"]] = crossed
        age = rng.exponential(3, n)
        t[:, ci["book_age"]] = age
        ul = np.cumsum(rng.normal(0, 1e-4, n))
        t[:, ci["ul_logp_rel_open"]] = ul - ul[0]
        t[:, ci["ul_logret_1s"]] = np.diff(ul, prepend=ul[0])
        t[:, ci["const_one"]] = 1
        # taker prints: maker legs at the touch (or one level through) of the book 3 s earlier
        for k in range(n):
            if rng.random() < 0.35:
                kk = max(k - 3, 0)
                side = int(rng.choice([1, -1]))
                depth = int(rng.random() < 0.3)
                px = float(ask0[kk] + 0.01 * depth) if side > 0 else float(bid0[kk] - 0.01 * depth)
                px = round(min(max(px, 0.01), 0.99), 2)
                sz = float(rng.uniform(5, 60))
                legs.append((off + k, side, 1, int(round(px * 1000)), px, sz, sz))
                col = "buy_vol" if side > 0 else "sell_vol"
                t[k, ci[col]] += sz
                t[k, ci["trade_count"]] += 1
        winner = int(fv[-1] + rng.normal(0, 0.05) > 0.5)
        pm = np.clip(fv, 1e-4, 1 - 1e-4)
        bid_ok, ask_ok = bid0 > 0, ~one
        mark = np.where(ask_ok, 0.5 * (bid0 + ask0), pm)
        vals = dict(pm=pm, mark=mark, stc=(L - np.arange(n)).astype(float), L=float(L), bid=bid0, ask=ask0,
                    bid_ok=bid_ok, ask_ok=ask_ok, bsz=t[:, ci["bid_sz_0"]], asz=t[:, ci["ask_sz_0"]],
                    age=age, crossed=crossed, rng=rng)
        f = np.stack([_feat_values(c, vals) for c in fcols], 1).astype(np.float32)
        f[:, :n_obs] = np.clip(f[:, :n_obs], -5, 5)
        feats.append(f)
        bk = np.zeros(n, np.dtype(BOOK_FIELDS))
        bk["bid_px"] = np.round(t[:, [ci[f"bid_px_{k}"] for k in range(5)]] * 1000).astype(np.int16)
        bk["ask_px"] = np.round(t[:, [ci[f"ask_px_{k}"] for k in range(5)]] * 1000).astype(np.int16)
        bk["bid_sz"] = t[:, [ci[f"bid_sz_{k}"] for k in range(5)]]
        bk["ask_sz"] = t[:, [ci[f"ask_sz_{k}"] for k in range(5)]]
        bk["flags"] = 1 | (crossed.astype(np.uint8) << 1) | ((age > 10).astype(np.uint8) << 2)
        bk["tick"] = bk["tick_alt"] = 10
        books.append(bk)
        slug = f"{coin}-updown-{L // 60}m-{open_ts}"
        mrecs.append((off, open_ts, open_ts + L, n, L, COIN_IDS[coin], winner, split, 3, 0.01, 0.07,
                      1.0, 5.0, 0.0, 0.0, 0.0, 0.2, mi, 0.001))
        slugs.append(slug)
        rows.append(t)
        off += n
    ticks = np.concatenate(rows)
    feat = np.concatenate(feats)
    N = ticks.shape[0]
    rec = _market_record()
    mdt = np.dtype({"names": [x[0] for x in MARKET_FIELDS],
                    "formats": [{"int64_t": "<i8", "int32_t": "<i4", "float": "<f4"}[x[1]] for x in MARKET_FIELDS],
                    "offsets": [f["offset"] for f in rec["fields"]], "itemsize": rec["itemsize"]})
    markets = np.array(mrecs, dtype=mdt)
    ticks.tofile(out / "ticks.bin")
    markets.tofile(out / "markets.bin")
    feat.tofile(out / "feat.bin")
    np.concatenate(books).tofile(out / "book.bin")
    test = markets[markets["split"] == 2]
    lo = int(test["row_offset"].min())
    hi = int((test["row_offset"] + test["n_rows"]).max())
    meta = dict(build_version=1, synthetic=True, F=len(COLS), N=int(N), n_markets=len(markets),
                dtype="float32", order="row-major [N, F]", columns=COLS, column_index=ci,
                market_record=rec, coin_ids=COIN_IDS, contract_len_s={"5m": 300, "15m": 900, "4h": 14400},
                splits={"ids": {"0": "train", "1": "val", "2": "test", "3": "purged"}}, slugs=slugs,
                test_rows=dict(row_lo=lo, row_hi=hi,
                               sha256=sha256_bytes(np.ascontiguousarray(ticks[lo:hi]).tobytes())))
    write_json(out / "meta.json", meta)
    write_json(out / "splits.json", {s: int(m["split"]) for s, m in zip(slugs, markets)})
    fdt = [("row", "int32_t", 0), ("side", "int8_t", 4), ("first", "uint8_t", 5), ("px", "int16_t", 6),
           ("price", "float", 8), ("size", "float", 12), ("tx_size", "float", 16)]
    fmeta = dict(features_version=1, synthetic=True, layout=layout, N=int(N), G=len(fcols), n_obs_features=n_obs,
                 columns=fcols, column_index={c: i for i, c in enumerate(fcols)},
                 fill_record=dict(itemsize=20, fields=[dict(name=a, type=b, offset=c) for a, b, c in fdt]),
                 book_record=dict(itemsize=64, fields=[
                     dict(name="bid_px", type="int16_t[5]", offset=0), dict(name="ask_px", type="int16_t[5]", offset=10),
                     dict(name="bid_sz", type="float[5]", offset=20), dict(name="ask_sz", type="float[5]", offset=40),
                     dict(name="flags", type="uint8_t", offset=60), dict(name="tick", type="uint8_t", offset=61),
                     dict(name="tick_alt", type="uint8_t", offset=62), dict(name="pad", type="uint8_t", offset=63)]))
    fmeta.update({k: v for k, v in fextra.items() if k not in ("fill_record", "book_record")})
    write_json(out / "feat_meta.json", fmeta)
    fdtype = np.dtype({"names": [a for a, _, _ in fdt], "formats": ["<i4", "i1", "u1", "<i2", "<f4", "<f4", "<f4"],
                       "offsets": [c for _, _, c in fdt], "itemsize": 20})
    legs.sort(key=lambda r: r[0])
    fills = np.array(legs, dtype=fdtype)
    fills.tofile(out / "fills.bin")
    idx = np.searchsorted(fills["row"], np.arange(N + 1), side="left").astype("<i4")
    idx.tofile(out / "fill_index.bin")
    return out


# ----------------------------------------------------------------------------- components
def _save_mlp(path, obs_size, seed):
    import torch
    tp = train_policy_module()
    torch.manual_seed(seed)
    net = tp.make_policy("mlp", obs_size=obs_size, nvec=(7, 7, 3), hidden=16, layers=1)
    with torch.no_grad():                         # larger logits so argmax is not all-off
        net.actor_head.weight.mul_(100.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(policy_kwargs=dict(kind="mlp", hidden=16, layers=1), obs_size=obs_size,
                    nvec=[7, 7, 3], policy=net.state_dict(), seed=seed), path)


def _save_p5(path, obs_size, seed):
    import torch
    import puffernet_mm as p5
    torch.manual_seed(seed)
    net = p5.PufferNetMM(obs_size, hidden=16, layers=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.save_bin(str(path))


def have_torch():
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def build_root(out_dir, obs_size, torch_ckpts=None, puffer5=True, n_seeds=5):
    out = Path(out_dir)
    torch_ckpts = have_torch() if torch_ckpts is None else torch_ckpts
    write_json(out / "baselines" / "tuned.json", {
        "sym_touch": {"params": {"size": 0, "q_cap": 10.0, "cap_mode": "pull"},
                      "val": {"total_pnl": 1.0}, "config_sha256": "a" * 64},
        "as_logit": {"params": {"gamma": 0.05, "k": 50.0, "h_risk": 60.0, "size": 0},
                     "val": {"total_pnl": -2.0}},
        "fair_value": {"params": {"k_skew": 0.001, "spread_mult": 1.0, "gap_max": 0.1, "pull_s": 30.0,
                                  "size": 0}, "val": {"total_pnl": 3.0}},
    })
    write_json(out / "baselines" / "tuned_constrained.json", {      # DECISIONS.md D1.2 (stand-in)
        "sym_touch": {"best_params": {"size": 0, "q_cap": 20.0, "cap_mode": "back"}, "val_pnl_mean": -0.5,
                      "constraints": {"min_fills_per_market": 5, "min_uptime_two_sided": 0.25}}})
    seeds = []
    for s in range(n_seeds):
        if torch_ckpts:
            p = out / "train" / "checkpoints" / f"seed{s}.pt"
            _save_mlp(p, obs_size, s)
            seeds.append({"seed": s, "checkpoint": f"checkpoints/seed{s}.pt"})
        else:
            seeds.append({"seed": s, "kind": "callable", "target": "standins:jittered_fair_value"})
    cfg = {"policy": "mlp", "hidden": 16, "layers": 1, "lr": 3e-4}
    write_json(out / "train" / "selected.json", {"config": cfg, "config_sha256": S.sha256_json(cfg),
                                                  "kind": "ppo3", "seeds": seeds,
                                                  "selection_rule": "seed-mean val P&L (stand-in)"})
    if puffer5 and torch_ckpts:
        cks = []
        for s in range(n_seeds):
            p = out / "puffer5" / "checkpoints" / f"seed{s}.bin"
            _save_p5(p, obs_size, s)
            cks.append({"seed": s, "checkpoint": f"checkpoints/seed{s}.bin"})
        write_json(out / "puffer5" / "selected.json", {"hidden": 16, "layers": 1, "seeds": cks,
                                                        "config": {"hidden": 16, "layers": 1}})
    led = [(out / "results" / "trials.jsonl", [{"split": 1, "kind": "ppo3", "config_sha256": "c1", "seed": 0,
                                                "metrics": {"sharpe_hourly": 0.1}},
                                               {"split": 1, "kind": "ppo3", "config_sha256": "c1", "seed": 1,
                                                "metrics": {"sharpe_hourly": 0.3}},
                                               {"split": 0, "kind": "ppo3", "config_sha256": "c2"}]),
           (out / "baselines" / "trials.jsonl", [{"split": "val", "kind": "baseline", "params": {"a": 1}},
                                                 {"split": "val", "kind": "baseline", "params": {"a": 2},
                                                  "sharpe": -0.2},
                                                 {"kind": "baseline", "params": {"a": 3}}]),
           (out / "train" / "runs" / "x" / "trials.jsonl", [{"split": 1, "config": {"lr": 1}}])]
    for p, recs in led:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in recs) + "not json\n")
    return out


def standin_root(out_dir, n_seeds=5):
    """Artifact root for a val dry run with the real data before tuned.json/selected.json exist."""
    import baselines as bl
    out = Path(out_dir)
    write_json(out / "baselines" / "tuned.json",
               {k: {"params": v, "note": "untuned DEFAULTS (stand-in)"} for k, v in bl.DEFAULTS.items()
                if k != "no_quote"})
    seeds = [{"seed": s, "kind": "callable", "target": "standins:jittered_fair_value"} for s in range(n_seeds)]
    write_json(out / "train" / "selected.json", {"config": {"standin": "jittered fair_value"},
                                                  "kind": "callable", "seeds": seeds})
    return out
