import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "v3_sports"))
import backfill_pob as bp  # noqa: E402
tok = sys.argv[1]; ko = bp.parse_ts(sys.argv[2])
pob = bp.POB(os.environ[bp.ENV_KEY], rpm=30, category="sports")
for a, b, res in [(ko - 12*3600, ko - 6*3600, "1m"), (ko - 12*3600, ko - 6*3600, "5m"), (ko - 7*3600, ko - 6*3600 - 3000, "1s"), (ko - 3600, ko, "1m"), (ko - 12*3600, ko, "15m")]:
    d = pob.get(f"/v1/tokens/{tok}/books", {"start_ts": int(a), "end_ts": int(b), "resolution": res, "limit": 200})
    p = d.get("data") or []
    print(res, bp.iso(a), bp.iso(b), "n", len(p), "first", p[0]["t"] if p else None, "last", p[-1]["t"] if p else None, "more", bool((d.get("metadata") or {}).get("next_cursor")))
