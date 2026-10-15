#!/usr/bin/env python3
"""CLOB /rewards/markets/current (all pages) -> cache/rewards_current.parquet: condition_id, rate_per_day,
min size, max spread, start date. Snapshot of today's programs (2026-10-15)."""
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
URL = "https://clob.polymarket.com/rewards/markets/current"
s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) Chrome/126.0"
rows, cur = [], ""
while True:
    for k in range(6):
        r = s.get(URL, params={"next_cursor": cur} if cur else {}, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(2 ** k)
    d = r.json()
    for m in d.get("data", []):
        rc = m.get("rewards_config") or [{}]
        rows.append({"condition_id": m["condition_id"], "rate_per_day": float(m.get("total_daily_rate") or 0),
                     "native_rate": float(m.get("native_daily_rate") or 0),
                     "min_size": m.get("rewards_min_size"), "max_spread": m.get("rewards_max_spread"),
                     "start_date": min((c.get("start_date") or "") for c in rc), "n_cfg": len(rc)})
    cur = d.get("next_cursor") or ""
    if not cur or cur == "LTE=" or not d.get("data"):
        break
df = pd.DataFrame(rows).drop_duplicates("condition_id")
(HERE / "cache").mkdir(exist_ok=True)
df.to_parquet(HERE / "cache" / "rewards_current.parquet", index=False)
print(len(df), "markets; total $/day", df.rate_per_day.sum().round(0))
print(df.rate_per_day.describe())
