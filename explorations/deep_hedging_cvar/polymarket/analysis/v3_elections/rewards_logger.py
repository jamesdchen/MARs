#!/usr/bin/env python3
"""Log the CLOB liquidity-reward configs of the election markets every hour (D11 track E).

    python rewards_logger.py --out rec [--every 3600]

Each snapshot: CLOB GET /rewards/markets/current (all pages), restricted to the condition ids of
sel/selection.json's `all_midterm_with_rewards` (every midterm market with a reward program at selection
time) plus any current reward market whose id is in the recorder's tracked set. Written to
<out>/rewards/<day>/rewards_<day>T<HH>.parquet with columns: taken_at, condition_id, selected (recorded by
polyrec), total_daily_rate, native_daily_rate, rewards_min_size, rewards_max_spread, n_cfg,
cfg_json (the full rewards_config list as JSON). A market that drops out of the program shows up as
missing in later snapshots (also logged: in_program=False rows for previously seen ids).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from select_markets import rewards_current

HERE = Path(__file__).resolve().parent


def snapshot(out: Path, sel_path: Path):
    sel = json.loads(sel_path.read_text())
    watch = {r["condition_id"] for r in sel["all_midterm_with_rewards"]}
    selected = {r["condition_id"] for r in sel["selected"]}
    rw = rewards_current()
    now = time.gmtime()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", now)
    rows = []
    for cid in sorted(watch):
        m = rw.get(cid)
        rows.append({"taken_at": ts, "condition_id": cid, "selected": cid in selected, "in_program": m is not None,
                     "total_daily_rate": float((m or {}).get("total_daily_rate") or 0),
                     "native_daily_rate": float((m or {}).get("native_daily_rate") or 0),
                     "rewards_min_size": (m or {}).get("rewards_min_size"),
                     "rewards_max_spread": (m or {}).get("rewards_max_spread"),
                     "n_cfg": len((m or {}).get("rewards_config") or []),
                     "cfg_json": json.dumps((m or {}).get("rewards_config"))})
    df = pd.DataFrame(rows)
    day = time.strftime("%Y-%m-%d", now)
    d = out / "rewards" / day
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"rewards_{day}T{time.strftime('%H', now)}.parquet"
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(f)
    print(ts, f.name, len(df), "rows; in program", int(df.in_program.sum()),
          "; pool selected $%.0f/day" % df[df.selected].total_daily_rate.sum(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "rec"))
    p.add_argument("--sel", default=str(HERE / "sel" / "selection.json"))
    p.add_argument("--every", type=float, default=3600)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    while True:
        t0 = time.time()
        try:
            snapshot(Path(a.out), Path(a.sel))
        except Exception as e:  # keep logging; a failed hour is visible as a missing file
            print("snapshot failed:", repr(e), flush=True)
        if a.once:
            return
        time.sleep(max(60.0, a.every - (time.time() - t0)))


if __name__ == "__main__":
    main()
