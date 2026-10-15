"""Mirrored copy of a data/build directory for the mirror test (EVALUATION.md T6).

The Down token becomes "Up": every price p -> 1 - p with the bid and ask sides swapped (sizes
follow their side), taker sides negated, underlying returns negated, winner_up -> 1 - winner_up.
A market maker's P&L should be statistically unchanged on the mirrored day. Both the C env and
the stand-in read the mirrored files exactly as they read the originals, so the mirror run goes
through the same code path as every other variant.

    mirror_build(src_dir, dst_dir) -> dict of output sha256

Feature columns are transformed by FEAT_RULES ("neg", "same", "one_minus", "swap:<col>",
"liq_mid", "negswap:<col>"); an unknown column raises, so a new feature cannot be mirrored silently wrong.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from common import S, Dataset, _record_dtype, sha256_bytes, write_json

FEAT_RULES = {
    "p_logit": "neg", "mark_logit": "neg", "gap_c": "neg", "dgap5_c": "neg", "z": "neg",
    "sdv_bp": "same", "sigma_1m_bp": "same", "tau_frac": "same", "log_tau": "same", "in_twap": "same",
    "delta_usd": "same", "spread_ticks": "same", "bid_present": "swap:ask_present",
    "ask_present": "swap:bid_present", "imb_l1": "neg", "imb_l5": "neg", "bid_sz0": "swap:ask_sz0",
    "ask_sz0": "swap:bid_sz0", "flow5": "neg", "flow30": "neg", "flow120": "neg", "mret5_c": "neg",
    "mret30_c": "neg", "mret120_c": "neg", "ulz5": "neg", "ulz30": "neg", "ulz120": "neg",
    "len_5m": "same", "len_15m": "same", "len_4h": "same", "tick_fine": "same", "book_age": "same",
    "crossed_raw": "same", "mark_is_fv": "same", "gross120": "same",
    # auxiliaries read by the env (never observed)
    "p_model": "one_minus", "mark": "one_minus", "delta_usd_raw": "same", "sigma": "same",
    "sqrt_v": "same", "liq_mid": "liq_mid", "liq_comp": "same", "liq_pi": "same",
    "bid_minus_p_c": "negswap:ask_minus_p_c", "ask_minus_p_c": "negswap:bid_minus_p_c",
}


def _feat(x: np.ndarray, cols: list[str]) -> np.ndarray:
    out = np.empty_like(x)
    idx = {c: i for i, c in enumerate(cols)}
    for i, c in enumerate(cols):
        rule = FEAT_RULES.get(c, "same" if c.startswith("pad") else None)
        if rule is None:
            raise NotImplementedError(f"no mirror rule for feature column {c!r} (final/mirror.py)")
        if rule == "same":
            out[:, i] = x[:, i]
        elif rule == "neg":
            out[:, i] = -x[:, i]
        elif rule == "one_minus":
            out[:, i] = 1.0 - x[:, i]
        elif rule == "liq_mid":
            out[:, i] = np.where(x[:, i] < 0, x[:, i], 1.0 - x[:, i])
        elif rule.startswith("swap:"):
            out[:, i] = x[:, idx[rule[5:]]]
        elif rule.startswith("negswap:"):
            out[:, i] = -x[:, idx[rule[8:]]]
        else:
            raise ValueError(rule)
    return out


def _ticks(t: np.ndarray, ci: dict) -> np.ndarray:
    o = t.copy()
    for k in range(5):
        o[:, ci[f"bid_px_{k}"]] = 1.0 - t[:, ci[f"ask_px_{k}"]]
        o[:, ci[f"ask_px_{k}"]] = 1.0 - t[:, ci[f"bid_px_{k}"]]
        o[:, ci[f"bid_sz_{k}"]] = t[:, ci[f"ask_sz_{k}"]]
        o[:, ci[f"ask_sz_{k}"]] = t[:, ci[f"bid_sz_{k}"]]
    o[:, ci["bid_depth_total"]] = t[:, ci["ask_depth_total"]]
    o[:, ci["ask_depth_total"]] = t[:, ci["bid_depth_total"]]
    bv, sv = t[:, ci["buy_vol"]], t[:, ci["sell_vol"]]
    o[:, ci["buy_vol"]], o[:, ci["sell_vol"]] = sv, bv
    o[:, ci["buy_vwap"]] = np.where(sv > 0, 1.0 - t[:, ci["sell_vwap"]], 0.0)
    o[:, ci["buy_max_px"]] = np.where(sv > 0, 1.0 - t[:, ci["sell_min_px"]], 0.0)
    o[:, ci["sell_vwap"]] = np.where(bv > 0, 1.0 - t[:, ci["buy_vwap"]], 0.0)
    o[:, ci["sell_min_px"]] = np.where(bv > 0, 1.0 - t[:, ci["buy_max_px"]], 0.0)
    for c in ("ul_logp_rel_open", "ul_logret_1s", "ul_logp_vs_strike"):
        o[:, ci[c]] = -t[:, ci[c]]
    return o


def mirror_build(src, dst) -> dict:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    ds = Dataset(src)
    meta = json.loads((src / "meta.json").read_text())
    ticks = _ticks(np.asarray(ds.ticks), ds.ci)
    ticks.tofile(dst / "ticks.bin")
    mk = ds.markets.copy()
    mk["winner_up"] = 1 - mk["winner_up"]
    mk["log_strike_vs_ul_open"] = -mk["log_strike_vs_ul_open"]
    mk.tofile(dst / "markets.bin")
    t = meta["test_rows"]
    meta["test_rows"] = dict(t, sha256=sha256_bytes(np.ascontiguousarray(ticks[t["row_lo"]: t["row_hi"]]).tobytes()))
    meta["mirrored"] = True
    meta["mirror_of"] = {f: S.sha256_file(src / f) for f in ("meta.json", "ticks.bin", "markets.bin")}
    write_json(dst / "meta.json", meta)
    for f in ("splits.json", "fill_index.bin"):
        if (src / f).exists():
            shutil.copyfile(src / f, dst / f)
    if ds.fmeta is not None:
        fm = json.loads((src / "feat_meta.json").read_text())
        _feat(np.asarray(ds.feat), fm["columns"]).astype(np.float32).tofile(dst / "feat.bin")
        fm["mirrored"] = True
        write_json(dst / "feat_meta.json", fm)
        if (src / "fills.bin").exists():
            f = ds.fills.copy()
            f["side"] = -f["side"]
            if "px" in f.dtype.names:
                f["px"] = 1000 - f["px"]
            f["price"] = 1.0 - f["price"]
            f.tofile(dst / "fills.bin")
        if (src / "book.bin").exists() and "book_record" in fm:
            bd = _record_dtype(fm["book_record"])
            b = np.fromfile(src / "book.bin", dtype=bd)
            o = b.copy()
            o["bid_px"], o["ask_px"] = 1000 - b["ask_px"], 1000 - b["bid_px"]
            o["bid_sz"], o["ask_sz"] = b["ask_sz"], b["bid_sz"]
            o.tofile(dst / "book.bin")
    return {f: S.sha256_file(dst / f) for f in Dataset.HASHED if (dst / f).exists()}
