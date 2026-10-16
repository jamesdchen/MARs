"""One-off density probe for the Starter plan (key from env only; never printed)."""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "v3_sports"))
import backfill_pob as bp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--n", type=int, default=4)
a = ap.parse_args()
args = argparse.Namespace(start=a.start, end=a.end, market_types=["moneyline"], leagues=bp.LEAGUES,
                          hours_before=168, post_min=0, min_volume=50000, order="volume", max_markets=0,
                          out=str(HERE.parent.parent / "v3_sports" / "raw_pob"))
ms = bp.select_markets(args)
floor = time.time() - 3 * 86400 + 600
ms = [m for m in ms if m["kickoff"] - 6 * 3600 > floor]
print("eligible (6h fine window inside history):", len(ms))
pob = bp.POB(os.environ[bp.ENV_KEY], rpm=40, category="sports")
seen = set(); rep = []
for m in ms:
    if m["league"] in seen: continue
    seen.add(m["league"])
    r = {"slug": m["slug"], "vol": round(m["volume"]), "ko": bp.iso(m["kickoff"])}
    try:
        n0 = pob.n
        p6, pg6, _ = pob.token_books(m["tokens"][0], m["kickoff"] - 6 * 3600, m["kickoff"], "1s")
        r["fine6h_points"], r["fine6h_pages"] = len(p6), pg6
        if p6:
            ts = np.array([bp.parse_ts(x["t"]) for x in p6])
            r["first_t"] = p6[0]["t"]; r["last_t"] = p6[-1]["t"]
            r["levels_med"] = float(np.median([len(x.get("bids") or []) + len(x.get("asks") or []) for x in p6]))
            r["pts_last30m"] = int((ts >= m["kickoff"] - 1800).sum())
        lo = max(int(floor) // 3600 * 3600 + 3600, m["kickoff"] - 168 * 3600)
        pc, pgc, _ = pob.token_books(m["tokens"][0], lo, m["kickoff"] - 6 * 3600, "1m")
        r["coarse_hours"] = round((m["kickoff"] - 6 * 3600 - lo) / 3600, 1)
        r["coarse_points"], r["coarse_pages"] = len(pc), pgc
        d1 = pob.get(f"/v1/tokens/{m['tokens'][1]}/books", {"start_ts": m["kickoff"] - 600, "end_ts": m["kickoff"], "resolution": "1s", "limit": 200})
        q = {x["t"]: x for x in d1.get("data") or []}
        top3 = lambda pt: [[round(p, 6) for p in bp._ladder(pt.get(sd), sd == "bids")[0][:3]] for sd in ("bids", "asks")]
        same = [top3(bp.mirror_point(x)) == top3(q[x["t"]]) for x in p6 if x["t"] in q]
        r["mirror_top3"] = f"{sum(same)}/{len(same)}"
        r["requests"] = pob.n - n0
    except bp.ApiError as e:
        r["error"] = str(e)
    print(json.dumps(r)); rep.append(r)
    if len(rep) >= a.n: break
print("total requests", pob.n)
