"""Dead-man watchdog thread, systemd sd_notify, and the coverage log."""
from __future__ import annotations

import faulthandler
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

import orjson

log = logging.getLogger("polyrec.watchdog")

EXIT_LOOP_STALL = 70
EXIT_NO_DATA = 75
EXIT_RSS = 76


def sd_notify(msg: str) -> bool:
    """Minimal sd_notify(3) without libsystemd (no-op unless NOTIFY_SOCKET is set)."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
        return True
    except OSError:
        return False


class LoopWatchdog(threading.Thread):
    """Independent thread: if the asyncio loop stops beating for `stall_s`, dump stacks and hard-exit
    so the supervisor (systemd / docker restart policy) starts a fresh process."""

    def __init__(self, stall_s: float):
        super().__init__(name="polyrec-watchdog", daemon=True)
        self.stall_s = stall_s
        self.beat = time.monotonic()
        self.stopped = threading.Event()

    def kick(self):
        self.beat = time.monotonic()

    def run(self):
        while not self.stopped.wait(5):
            lag = time.monotonic() - self.beat
            if lag > self.stall_s:
                log.critical("event loop stalled for %.0fs; dumping stacks and exiting", lag)
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                finally:
                    os._exit(EXIT_LOOP_STALL)


class CoverageLog:
    """Append-only JSONL of market liveness transitions, used by merge.py to keep only markets whose
    whole life was recorded:
        {"t": unix_s, "run": id, "ev": "live"|"down", "slug": ..., "why": ...}
        {"t": ..., "run": id, "ev": "hb"}           every status interval (bounds a crash)
        {"t": ..., "run": id, "ev": "start"|"stop"}
    A market is live while both of its token books hold a snapshot on a connected websocket."""

    def __init__(self, root: Path, run_id: str):
        self.dir = Path(root) / "coverage"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run_id
        self.day = None
        self.fh = None

    def write(self, ev: str, t: float | None = None, **kw):
        t = time.time() if t is None else t
        day = time.strftime("%Y-%m-%d", time.gmtime(t))
        if day != self.day:
            if self.fh:
                self.fh.close()
            self.fh = open(self.dir / f"coverage_{day}.jsonl", "ab")
            self.day = day
        self.fh.write(orjson.dumps({"t": round(t, 3), "run": self.run, "ev": ev, **kw}) + b"\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self.fh.close()
            self.fh = None
