"""Which census period a script runs on, from environment variables (defaults: the original Oct 8-14 run).

  CENSUS_TAGS  comma list of build_legs tags; '' = cache/legs.parquet (Oct 8-14 as first built)
  CENSUS_OUT   output dir name (default out)
  CENSUS_DAYS  days in the period (default 7)
  CENSUS_PAY   'start_ts,end_ts' of REWARD payouts attributable to the period
"""
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TAGS = [t for t in os.environ.get("CENSUS_TAGS", "").split(",") if t]
OUT = HERE / os.environ.get("CENSUS_OUT", "out")
DAYS = float(os.environ.get("CENSUS_DAYS", "7"))
PAY = tuple(int(x) for x in os.environ.get("CENSUS_PAY", "1791504000,1792044000").split(","))
_files = [HERE / "cache" / f"legs_{t}.parquet" for t in TAGS] or [HERE / "cache" / "legs.parquet"]
LEGS = "read_parquet([" + ", ".join(f"'{f}'" for f in _files) + "])"


def taker() -> pd.DataFrame:
    fs = [HERE / "cache" / f"taker_{t}.parquet" for t in TAGS] or [HERE / "cache" / "taker.parquet"]
    return pd.concat([pd.read_parquet(f) for f in fs]).groupby("cond", as_index=False).sum(numeric_only=True)
