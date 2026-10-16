"""Which leagues does PolyOrderBooks have, and from when before kickoff? 1 request per market (1h res)."""
import os, sys, time, json
from pathlib import Path
import argparse
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "v3_sports"))
import backfill_pob as bp  # noqa: E402
args = argparse.Namespace(start=sys.argv[1], end=sys.argv[2], market_types=["moneyline"], leagues=bp.LEAGUES,
                          hours_before=168, post_min=0, min_volume=50000, order="volume", max_markets=0,
                          out=str(HERE.parent.parent / "v3_sports" / "raw_pob"))
per = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ms = bp.select_markets(args)
floor = time.time() - 3 * 86400 + 300
ms = [m for m in ms if m["kickoff"] - 3600 > floor]
pob = bp.POB(os.environ[bp.ENV_KEY], rpm=40, category="sports")
cnt = {}
for m in ms:
    if cnt.get(m["league"], 0) >= per: continue
    cnt[m["league"]] = cnt.get(m["league"], 0) + 1
    lo = max(int(floor) + 60, m["kickoff"] - 168 * 3600)
    r = {"slug": m["slug"], "vol": round(m["volume"]), "ko": bp.iso(m["kickoff"]), "hist_h": round((m["kickoff"] - lo) / 3600, 1)}
    try:
        p, pg, meta = pob.token_books(m["tokens"][0], lo, m["kickoff"], "1h")
        r["pts_1h"] = len(p); r["first_t"] = p[0]["t"] if p else None
    except bp.ApiError as e:
        r["error"] = str(e)
    print(json.dumps(r), flush=True)
print("requests", pob.n)
