"""CLI:  python -m polyrec record   [--config FILE] [--out DIR] [--duration S] [--coins btc,eth] [--lengths 5m,15m]
       python -m polyrec finalize [--config FILE] [--out DIR] [--slug S ...]
       python -m polyrec status   [--out DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import sys

from .config import load_config


def setup_logging(level: str, log_file: str = ""):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level.upper())
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    root.addHandler(h)
    if log_file:
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=20 * 2**20, backupCount=5)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="polyrec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "finalize", "status"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--log-level", default=None)
    rp = sub.choices["record"]
    rp.add_argument("--duration", type=float, default=None, help="stop after N seconds (smoke tests)")
    rp.add_argument("--coins", default=None, help="restrict the default up/down source, e.g. btc,eth")
    rp.add_argument("--lengths", default=None, help="restrict the default up/down source, e.g. 5m,15m")
    rp.add_argument("--slug-regex", default=None, help="override the first source's slug regex")
    rp.add_argument("--pre-open", type=int, default=None)
    rp.add_argument("--settle-delay", type=float, default=None)
    rp.add_argument("--run-id", default=None)
    fp = sub.choices["finalize"]
    fp.add_argument("--slug", action="append", default=None)
    fp.add_argument("--require-resolved", action="store_true")
    a = ap.parse_args(argv)

    ov = {}
    if a.out:
        ov["out_dir"] = a.out
    if a.log_level:
        ov["log_level"] = a.log_level
    cfg = load_config(a.config, ov)
    setup_logging(cfg.log_level, cfg.log_file)

    if a.cmd == "status":
        print((cfg.out / "status.json").read_text())
        return 0
    if a.cmd == "finalize":
        from .offline import finalize_offline
        res = asyncio.run(finalize_offline(cfg, a.slug, require_resolved=a.require_resolved))
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}, indent=1))
        return 0 if not res["errors"] else 1

    src = cfg.sources[0]
    if a.coins or a.lengths:
        coins = "|".join(a.coins.split(",")) if a.coins else "[a-z0-9]+"
        lens = "|".join(a.lengths.split(",")) if a.lengths else "5m|15m|4h"
        src.slug_regex = rf"^(?P<coin>{coins})-updown-(?P<len>{lens})-(?P<ts>\d+)$"
    if a.slug_regex:
        src.slug_regex = a.slug_regex
    if a.pre_open is not None:
        for s in cfg.sources:
            s.pre_open_s = a.pre_open
    if a.settle_delay is not None:
        cfg.trades.settle_delay_s = a.settle_delay
    from .recorder import Recorder
    rec = Recorder(cfg, run_id=a.run_id, duration_s=a.duration)
    try:
        import uvloop  # optional: lower per-frame overhead
        factory = uvloop.new_event_loop
    except ImportError:
        factory = None
    return asyncio.run(rec.run(), loop_factory=factory)


if __name__ == "__main__":
    sys.exit(main())
