"""Numpy loader for the built env data (data/build/).

    from load import load, episodes
    ticks, markets, meta = load()            # ticks: np.memmap float32 [N, F]
    for m, rows in episodes(ticks, markets, split=0):
        ...                                   # rows: view [n_rows, F] for one market

`markets` is a structured array matching the C struct documented in
meta.json["market_record"] / data/README.md (88 bytes, little endian).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BUILD = Path(__file__).resolve().parent / "build"
_NP = {"int64_t": "<i8", "int32_t": "<i4", "float": "<f4"}
SPLITS = {"train": 0, "val": 1, "test": 2, "purged": 3}


def market_dtype(meta: dict) -> np.dtype:
    rec = meta["market_record"]
    f = rec["fields"]
    return np.dtype({"names": [x["name"] for x in f], "formats": [_NP[x["type"]] for x in f],
                     "offsets": [x["offset"] for x in f], "itemsize": rec["itemsize"]})


def load(build_dir: str | Path = BUILD, mode: str = "r"):
    build_dir = Path(build_dir)
    meta = json.loads((build_dir / "meta.json").read_text())
    ticks = np.memmap(build_dir / "ticks.bin", dtype=np.float32, mode=mode, shape=(meta["N"], meta["F"]))
    markets = np.fromfile(build_dir / "markets.bin", dtype=market_dtype(meta))
    assert len(markets) == meta["n_markets"]
    return ticks, markets, meta


def episodes(ticks, markets, split: int | str | None = None, shuffle: bool = False, seed: int = 0):
    """Yield (market_record, rows[n_rows, F]) per market, optionally for one split."""
    if isinstance(split, str):
        split = SPLITS[split]
    idx = np.arange(len(markets)) if split is None else np.flatnonzero(markets["split"] == split)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in idx:
        m = markets[i]
        o = int(m["row_offset"])
        yield m, ticks[o: o + int(m["n_rows"])]


def col(meta: dict, name: str) -> int:
    return meta["column_index"][name]


def verify(build_dir: str | Path = BUILD) -> None:
    """Integrity checks; raises AssertionError on failure."""
    import hashlib
    build_dir = Path(build_dir)
    ticks, markets, meta = load(build_dir)
    N, F = ticks.shape
    assert F % 4 == 0 and F == len(meta["columns"])
    assert np.isfinite(ticks).all(), "NaN/inf in ticks"
    forbidden = {"winner", "winner_up", "is_winning_outcome", "final_price"}
    assert not forbidden & set(meta["columns"]), "label leaked into ticks columns"
    off, n = markets["row_offset"], markets["n_rows"]
    assert off[0] == 0 and (off[1:] == off[:-1] + n[:-1]).all() and int(off[-1] + n[-1]) == N, "offsets"
    assert (n == markets["contract_len_s"] + 1).all()
    assert (markets["end_ts"] - markets["open_ts"] == markets["contract_len_s"]).all()
    c = meta["column_index"]
    stc = ticks[:, c["seconds_to_close"]]
    assert (ticks[:, c["const_one"]] == 1).all()
    for m in markets:
        s = stc[m["row_offset"]: m["row_offset"] + m["n_rows"]]
        assert s[0] == m["contract_len_s"] and s[-1] == 0 and (np.diff(s) == -1).all(), "seconds_to_close"
    v = ticks[:, c["book_valid"]] == 1
    bp = ticks[:, [c[f"bid_px_{k}"] for k in range(5)]]
    ap = ticks[:, [c[f"ask_px_{k}"] for k in range(5)]]
    assert ((bp >= 0) & (bp <= 1) & (ap >= 0) & (ap <= 1)).all()
    assert (np.diff(bp, axis=1) <= 1e-7).all() and (np.diff(ap, axis=1) >= -1e-7).all(), "ladder order"
    assert (bp[v, 0] < ap[v, 0]).all(), "valid book crossed"
    for k in ("buy_vol", "sell_vol", "trade_count", "book_age"):
        assert (ticks[:, c[k]] >= 0).all(), k
    assert set(np.unique(markets["split"])) <= {0, 1, 2, 3}
    splits = json.loads((build_dir / "splits.json").read_text())
    assert [splits[s] for s in meta["slugs"]] == markets["split"].tolist()
    t = meta["test_rows"]
    if t["sha256"]:
        assert hashlib.sha256(np.ascontiguousarray(ticks[t["row_lo"]: t["row_hi"]]).tobytes()).hexdigest() == t["sha256"]
    print(f"verify OK: ticks {ticks.shape}, markets {len(markets)}, "
          + ", ".join(f"{k}={int((markets['split'] == v).sum())}" for k, v in SPLITS.items()))


if __name__ == "__main__":
    verify()
