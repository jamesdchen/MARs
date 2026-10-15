"""Launch polyrec for this study without modifying the shared recorder code.

Patch: polyrec's browser-like User-Agent gets a Cloudflare 403 on clob.polymarket.com REST (/book) from aiohttp in
this container (curl with the same UA, and aiohttp with a plain UA, get 200), so every REST reconcile failed.
  python run_recorder.py record --config recorder_politics.toml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recorder"))
import polyrec.config, polyrec.http  # noqa: E402
polyrec.config.UA = polyrec.http.UA = "polyrec-research/1.0"
from polyrec.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
