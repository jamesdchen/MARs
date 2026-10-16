import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "v3_sports"))
import backfill_pob as bp  # noqa: E402
tok = sys.argv[1]; pob = bp.POB(os.environ[bp.ENV_KEY], rpm=30, category="sports")
for i in range(2, len(sys.argv), 3):
    a, b, res = bp.parse_ts(sys.argv[i]), bp.parse_ts(sys.argv[i+1]), sys.argv[i+2]
    d = pob.get(f"/v1/tokens/{tok}/books", {"start_ts": int(a), "end_ts": int(b), "resolution": res, "limit": 200})
    p = d.get("data") or []
    print(res, sys.argv[i], sys.argv[i+1], "n", len(p), "first", p[0]["t"] if p else None, "last", p[-1]["t"] if p else None, "meta", {k: v for k, v in (d.get("metadata") or {}).items() if k != "next_cursor"}, "more", bool((d.get("metadata") or {}).get("next_cursor")))
