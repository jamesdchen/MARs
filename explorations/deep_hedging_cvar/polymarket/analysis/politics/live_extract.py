"""onchain/extract.py for a window ending near the chain head (today's partial UTC day).

extract.plan() asks for the block at the end of every touched UTC day, which raises for today. This wrapper
clamps that lookup to the head (the day is then never 'full', so segments stay as part files).
  python live_extract.py --from-block A --to-block B --tokens out/tokens.parquet --events OrderFilled --slim --out <dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "onchain"))
import rpc  # noqa: E402

_orig = rpc.BlockClock.block_at_or_after


def _clamped(self, t, head=None, *a, **k):
    try:
        return _orig(self, t, head, *a, **k)
    except ValueError:
        return (head if head is not None else self.latest()) + 1


rpc.BlockClock.block_at_or_after = _clamped
import extract  # noqa: E402

if __name__ == "__main__":
    sys.exit(extract.main())
