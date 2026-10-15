"""Compact hourly per-market summaries of the polyrec recording -> out/rec_summary/books_hourly.csv
(market_stats of live_books.py per market x UTC hour) and out/rec_summary/recorder_status.json."""
import json, shutil, sys
from pathlib import Path
import pandas as pd
sys.argv = [sys.argv[0], "politics"]
import live_books as L

out = L.OUT / "rec_summary"; out.mkdir(exist_ok=True)
b = L.load_politics()
b["hour"] = b.captured_at.dt.floor("h")
U = {r["slug"]: r for r in json.load(open(L.OUT / "universe.json"))}
rows = []
for (slug, h), g in b.groupby(["market_slug", "hour"]):
    r = L.market_stats(g, U.get(slug, {}).get("rewards_max_spread"))
    if r:
        r.update(slug=slug, hour=h, category=U.get(slug, {}).get("category"))
        rows.append(r)
pd.DataFrame(rows).to_csv(out / "books_hourly.csv", index=False)
shutil.copy(L.HERE / "rec_out/status.json", out / "recorder_status.json")
print(len(rows), "market-hours")
