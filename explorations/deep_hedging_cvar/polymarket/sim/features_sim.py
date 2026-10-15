#!/usr/bin/env python3
"""Run data/features.py, unmodified, on a synthetic build directory.

features.py is written for data/build: it calls `load()` with the default directory, reads
raw/markets.parquet (market life for the liquidity-reward pools), raw/trades.parquet (fill legs) and
raw/underlying.parquet (Binance, indexed from fair_value.T0), and writes to BUILD. This wrapper repoints
those module globals at a sim build and then calls features.main(), so the sim's feat.bin / book.bin /
fills.bin / fill_index.bin / feat_meta.json come from exactly the same code as the replay's:

  BUILD      -> <build_dir>
  RAW        -> <build_dir>/raw   (markets.parquet: slug, created_start = open - 24 h, end;
                                   trades.parquet -> the build's fills.parquet, same schema)
  load()     -> data/load.load(<build_dir>)
  underlying -> the Binance closes the generator used, from meta.json["synthetic"]["underlying"]:
                "train": fair_value.load_underlying() (2026-08-20..22, asserts < 08-23; fv.T0 unchanged)
                "cache": sim/cache/binance segments; fair_value.T0 is set to the segment start so the
                         pricer's indexing (model_at, twap_binance) addresses that array

    python sim/features_sim.py data/build_sim
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as cm


def build_features(build_dir: str | Path):
    bd = Path(build_dir).resolve()
    meta = json.loads((bd / "meta.json").read_text())
    syn = meta["synthetic"]
    sys.path.insert(0, str(cm.FVDIR.parent / "rewards"))
    import features as F                      # data/features.py (on sys.path via common)
    from load import load
    fv = F.fv
    src = syn["underlying"]
    if src["source"] == "train":
        u = fv.load_underlying()
        arrays = {c: a for c, a in u.items()}
    else:
        import fetch_binance as fb
        t0 = int(src["t0"])
        arrays = {}
        for c in src["coins"]:
            segs = [(s0, a) for s0, a in fb.load_closes(c, src.get("start"), src.get("end"))]
            assert len(segs) == 1 and segs[0][0] == t0, f"{c}: expected one contiguous segment from {t0}"
            arrays[c] = segs[0][1]
        fv.T0 = t0                            # pricer indexing: close[i] <-> kline open_time T0 + i
    raw = bd / "raw"
    raw.mkdir(exist_ok=True)
    mk = pd.read_parquet(bd / "sim_markets.parquet", columns=["slug", "open_ts", "end_ts"])
    pd.DataFrame({"slug": mk.slug.values,
                  "created_start": pd.to_datetime(mk.open_ts.values - 86400, unit="s", utc=True),
                  "end": pd.to_datetime(mk.end_ts.values, unit="s", utc=True)}).to_parquet(raw / "markets.parquet")
    tp = raw / "trades.parquet"
    if tp.exists() or tp.is_symlink():
        tp.unlink()
    tp.symlink_to(Path("..") / "fills.parquet")          # relative: survives renaming the build dir
    F.BUILD = bd
    F.RAW = raw
    F.load = lambda: load(bd)
    F.underlying = lambda: {c: (a, 0) for c, a in arrays.items()}
    F.main()


if __name__ == "__main__":
    build_features(sys.argv[1])
