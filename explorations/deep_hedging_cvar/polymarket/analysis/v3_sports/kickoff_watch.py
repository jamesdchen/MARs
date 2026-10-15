#!/usr/bin/env python
"""Track S (D11): record ACTUAL game start times for the games the pre-game recorder records.

This runs on its own, alongside the polyrec recorder (it only reads rec_out/coverage). Two inputs:

1. wss://sports-api.polymarket.com/ws, the live-score stream. Nothing has to be subscribed: the
   server pushes one JSON object per game update ({gameId, leagueAbbreviation, homeTeam, awayTeam,
   status, score, period, elapsed?, live, ended}) for every sport, and sends a WS PING frame every
   ~15 s, which aiohttp's autoping answers. Every message goes to kickoff/ws_<UTC day>.jsonl as
   {"t": local receive unix time, "m": message}, and a new file starts at each UTC day.
2. gamma /markets?slug=a&slug=b..., in batches of 20. The moneyline market embeds its event, which
   carries live/ended/score/period/elapsed/gameId. Each recorded game is polled every 30 s from its
   scheduled kickoff -20 min to +40 min. The window is also opened around gamma endDate, the
   recorder's book cutoff, when that differs from gameStartTime (esports), and is kept open up to +3 h
   while gamma has not shown the game live (late starts; esports often start 10-60 min late). Polls go to
   kickoff/gamma_<UTC day>.jsonl.

The recorded slugs come from rec_out/coverage/coverage_*.jsonl ("live" events), read incrementally.
Per-slug metadata (gameStartTime, endDate, event slug, gameId) is cached in kickoff/meta.json and
refreshed every 30 min. kickoff/kickoffs.csv is re-derived every 5 min (see derive()).

    nohup nice -n 10 python kickoff_watch.py >> kickoff/watch.log 2>&1 &
    python kickoff_watch.py --derive          # rebuild kickoffs.csv from the logs and exit

For build_sports.py: `load_actual_kickoffs(kick_dir)` returns {slug: {"t": unix, "source", "confidence", ...}}.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import logging
import os
import random
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "rec_out"
KDIR = HERE / "kickoff"
WS_URL = "wss://sports-api.polymarket.com/ws"
GAMMA = "https://gamma-api.polymarket.com/markets"
PRE_S, POST_S = 20 * 60, 40 * 60
LATE_S = 3 * 3600                    # keep polling past +40 min, up to +3 h, while gamma has not shown the game live
POLL_S = 30
BATCH = 20
META_REFRESH_S = 1800
DISCOVER_S = 300
DERIVE_S = 300
WS_IDLE_S = 90                       # no frame (incl. server PING every ~15 s) for this long -> reconnect

log = logging.getLogger("kickoff_watch")

# ----------------------------------------------------------------------------------------- helpers


def to_unix(x) -> float | None:
    """gamma times: '2026-10-15 23:00:00+00', '2026-10-15T23:00:00Z', ISO with offset."""
    if x in (None, ""):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) >= 3 and (s[-3] in "+-") and s[-3:].lstrip("+-").isdigit():   # '+00' -> '+00:00'
        s = s + ":00"
    from datetime import datetime, timezone
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def iso(t: float | None) -> str:
    if t is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def day_of(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t))


def num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class DailyJsonl:
    """Append-only jsonl that starts a new file at each UTC day (by the record's time)."""

    def __init__(self, prefix: str, kdir: Path):
        self.prefix, self.kdir = prefix, kdir
        self.day, self.fh = None, None

    def write(self, t: float, obj: dict):
        d = day_of(t)
        if d != self.day:
            if self.fh:
                self.fh.close()
            self.fh = open(self.kdir / f"{self.prefix}_{d}.jsonl", "a", buffering=1 << 16)
            self.day = d
        self.fh.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

    def flush(self):
        if self.fh:
            self.fh.flush()


def meta_from_market(m: dict) -> dict:
    ev = (m.get("events") or [{}])[0]
    gst = to_unix(m.get("gameStartTime"))
    est = to_unix(m.get("eventStartTime")) or to_unix(ev.get("startTime"))
    end = to_unix(m.get("endDate"))
    sched = gst or est or end
    return dict(slug=m.get("slug"), event_slug=ev.get("slug"), game_id=str(ev.get("gameId") or m.get("gameId") or ""),
                game_start=gst, event_start=est, end=end, scheduled=sched,
                clear_book_on_start=m.get("clearBookOnStart"), title=ev.get("title") or m.get("question"),
                tokens=m.get("clobTokenIds"), fetched=time.time())


def poll_record(t: float, m: dict) -> dict:
    ev = (m.get("events") or [{}])[0]
    return dict(
        t=round(t, 3), slug=m.get("slug"), event_slug=ev.get("slug"), gameId=ev.get("gameId") or m.get("gameId"),
        live=ev.get("live"), ended=ev.get("ended"), score=ev.get("score"), period=ev.get("period"),
        elapsed=ev.get("elapsed"), started=ev.get("started"), gameStartTime=m.get("gameStartTime"),
        eventStartTime=m.get("eventStartTime"), startTime=ev.get("startTime"), endDate=m.get("endDate"),
        closed=m.get("closed"), active=m.get("active"), acceptingOrders=m.get("acceptingOrders"),
        clearBookOnStart=m.get("clearBookOnStart"), bestBid=m.get("bestBid"), bestAsk=m.get("bestAsk"),
        lastTradePrice=m.get("lastTradePrice"), m_updatedAt=m.get("updatedAt"), e_updatedAt=ev.get("updatedAt"),
        ev_closed=ev.get("closed"))


# ------------------------------------------------------------------------------------------- watch


class Watch:
    def __init__(self, rec: Path, kdir: Path):
        self.rec, self.kdir = rec, kdir
        kdir.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, dict] = {}
        mp = kdir / "meta.json"
        if mp.exists():
            try:
                self.meta = json.loads(mp.read_text())
            except ValueError:
                pass
        self.slugs: set[str] = set(self.meta)
        self.cov_off: dict[str, int] = {}
        self.ws_out = DailyJsonl("ws", kdir)
        self.gm_out = DailyJsonl("gamma", kdir)
        self.stop = asyncio.Event()
        self.n_ws = 0
        self.g_live: set[str] = set()     # slugs gamma has shown live/ended
        self.n_poll = 0
        self.ws_connects = 0

    # -- targets
    def read_coverage(self):
        for f in sorted(glob.glob(str(self.rec / "coverage" / "coverage_*.jsonl"))):
            off = self.cov_off.get(f, 0)
            try:
                with open(f, "rb") as fh:
                    fh.seek(off)
                    data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n") + 1
            for ln in data[:end].splitlines():
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if d.get("ev") == "live" and d.get("slug"):
                    self.slugs.add(d["slug"])
            self.cov_off[f] = off + end

    async def fetch(self, sess, slugs: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(slugs), BATCH):
            q = [("slug", s) for s in slugs[i:i + BATCH]]
            for attempt in range(3):
                try:
                    async with sess.get(GAMMA, params=q, timeout=20) as r:
                        if r.status == 200:
                            out += await r.json()
                            break
                        log.warning("gamma %s", r.status)
                except Exception as e:  # noqa: BLE001
                    log.warning("gamma error %r", e)
                await asyncio.sleep(2 * (attempt + 1))
            await asyncio.sleep(0.3)
        return out

    def save_meta(self):
        tmp = self.kdir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=0))
        os.replace(tmp, self.kdir / "meta.json")

    async def discover_loop(self, sess):
        while not self.stop.is_set():
            try:
                self.read_coverage()
                now = time.time()
                todo = [s for s in self.slugs
                        if s not in self.meta or now - self.meta[s].get("fetched", 0) > META_REFRESH_S]
                # skip refreshing games long past (> 6 h after the later of kickoff / endDate)
                todo = [s for s in todo if s not in self.meta or
                        max(filter(None, [self.meta[s].get("scheduled"), self.meta[s].get("end"), now])) > now - 6 * 3600]
                if todo:
                    ms = await self.fetch(sess, todo)
                    for m in ms:
                        if m.get("slug"):
                            self.meta[m["slug"]] = meta_from_market(m)
                    self.save_meta()
                    log.info("meta: %d fetched (%d slugs known)", len(ms), len(self.meta))
            except Exception:  # noqa: BLE001
                log.exception("discover")
            await self.sleep(DISCOVER_S)

    def in_window(self, md: dict, now: float) -> bool:
        for t in {md.get("scheduled"), md.get("game_start"), md.get("end")}:
            if t and t - PRE_S <= now <= t + POST_S:
                return True
        t = md.get("scheduled")
        return bool(t and t + POST_S < now <= t + LATE_S and md.get("slug") not in self.g_live)

    async def poll_loop(self, sess):
        while not self.stop.is_set():
            t0 = time.time()
            try:
                act = sorted(s for s, md in self.meta.items() if self.in_window(md, t0))
                if act:
                    ms = await self.fetch(sess, act)
                    tr = time.time()
                    for m in ms:
                        rec = poll_record(tr, m)
                        self.gm_out.write(tr, rec)
                        if _is_live(rec["live"], rec["ended"], None, rec["elapsed"]):
                            self.g_live.add(rec["slug"])
                    self.gm_out.flush()
                    self.n_poll += len(ms)
            except Exception:  # noqa: BLE001
                log.exception("poll")
            await self.sleep(max(1.0, POLL_S - (time.time() - t0)))

    async def ws_loop(self, sess):
        import aiohttp
        backoff = 1.0
        while not self.stop.is_set():
            t_conn = time.time()
            try:
                async with sess.ws_connect(WS_URL, autoping=True, heartbeat=None, timeout=20,
                                           max_msg_size=1 << 22) as ws:
                    self.ws_connects += 1
                    log.info("ws connected (#%d)", self.ws_connects)
                    self.ws_out.write(time.time(), {"t": round(time.time(), 3), "_ev": "connect"})
                    while not self.stop.is_set():
                        try:
                            msg = await ws.receive(timeout=WS_IDLE_S)
                        except asyncio.TimeoutError:
                            log.warning("ws idle %ss, reconnecting", WS_IDLE_S)
                            break
                        tr = time.time()
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                m = json.loads(msg.data)
                            except ValueError:
                                m = {"_raw": msg.data}
                            self.ws_out.write(tr, {"t": round(tr, 3), "m": m})
                            self.n_ws += 1
                            backoff = 1.0
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING,
                                          aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                            log.warning("ws closed: %s %s", msg.type, msg.data)
                            break
            except Exception as e:  # noqa: BLE001
                log.warning("ws error %r", e)
            self.ws_out.write(time.time(), {"t": round(time.time(), 3), "_ev": "disconnect"})
            self.ws_out.flush()
            if time.time() - t_conn > 60:
                backoff = 1.0
            await self.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60.0)

    async def flush_loop(self):
        last_derive = 0.0
        while not self.stop.is_set():
            await self.sleep(10)
            self.ws_out.flush()
            if time.time() - last_derive > DERIVE_S:
                last_derive = time.time()
                try:
                    rows = await asyncio.to_thread(derive, self.kdir, True)
                    n_obs = sum(1 for r in rows if r["actual_kickoff"])
                    log.info("derive: %d games, %d with an observed start | ws msgs %d, gamma rows %d, ws connects %d",
                             len(rows), n_obs, self.n_ws, self.n_poll, self.ws_connects)
                except Exception:  # noqa: BLE001
                    log.exception("derive")

    async def sleep(self, s: float):
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=s)
        except asyncio.TimeoutError:
            pass

    async def run(self):
        import aiohttp
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        async with aiohttp.ClientSession(trust_env=True) as sess:
            await asyncio.gather(self.discover_loop(sess), self.poll_loop(sess), self.ws_loop(sess), self.flush_loop())
        self.ws_out.flush()
        self.gm_out.flush()
        derive(self.kdir, True)
        log.info("stopped")


# ------------------------------------------------------------------------------------------ derive

LIVE_STATUS = {"inprogress", "in progress", "running", "live", "halftime", "break", "1h", "2h", "ht"}
PRE_STATUS = {"scheduled", "not_started", "notstarted", "pre", "pregame", "upcoming", "created", "delayed", ""}


def _is_live(live, ended, status=None, elapsed=None) -> bool | None:
    """True = game under way (or over), False = not started, None = unknown."""
    if live is True or ended is True:
        return True
    st = (status or "").strip().lower()
    if st in LIVE_STATUS:
        return True
    e = num(elapsed)
    if e is not None and e > 0:
        return True
    if live is False:
        return False
    if st in PRE_STATUS and status is not None:
        return False
    return None


def _ws_up(conn: list[tuple[float, str]], lo: float, hi: float) -> bool:
    """True if the ws was connected throughout [lo, hi] according to the connect/disconnect markers."""
    state = None
    for t, ev in conn:
        if t <= lo:
            state = ev
        elif t <= hi:
            return False
        else:
            break
    return state == "connect"


def _iter_jsonl(pattern: str):
    for f in sorted(glob.glob(pattern)):
        with open(f, "rb") as fh:
            for ln in fh:
                try:
                    yield json.loads(ln)
                except ValueError:
                    continue          # torn tail


def derive(kdir: Path = KDIR, write: bool = True) -> list[dict]:
    """Per recorded game: first live observation from ws and gamma, bracketed by the last not-live
    observation before it. actual_kickoff = the earliest live observation (ws is push, gamma 30 s polls).
    confidence: high (bracket <= 60 s, or a ws transition with the socket up over the whole bracket),
    medium (<= 600 s), low (wider), upper_bound (never seen
    before the start: the game was already running when first observed)."""
    kdir = Path(kdir)
    try:
        meta = json.loads((kdir / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    by_gid: dict[str, list[str]] = {}
    for s, md in meta.items():
        if md.get("game_id"):
            by_gid.setdefault(str(md["game_id"]), []).append(s)
    # ws: per gameId, first live time and not-live observations; also connection coverage
    ws_first: dict[str, tuple[float, dict]] = {}
    ws_notlive: dict[str, list[float]] = {}
    ws_first_seen: dict[str, float] = {}
    conn: list[tuple[float, str]] = []          # ws connect / disconnect markers
    for r in _iter_jsonl(str(kdir / "ws_*.jsonl")):
        if r.get("_ev"):
            conn.append((r["t"], r["_ev"]))
            continue
        m = r.get("m")
        if not isinstance(m, dict) or "gameId" not in m:
            continue
        gid = str(m["gameId"])
        if gid not in by_gid:
            continue
        t = r["t"]
        ws_first_seen.setdefault(gid, t)
        lv = _is_live(m.get("live"), m.get("ended"), m.get("status"), m.get("elapsed"))
        if lv is True:
            if gid not in ws_first:
                ws_first[gid] = (t, m)
        elif lv is False and gid not in ws_first:
            ws_notlive.setdefault(gid, []).append(t)
    # gamma polls: per slug
    g_first: dict[str, tuple[float, dict]] = {}
    g_notlive: dict[str, float] = {}
    g_first_seen: dict[str, float] = {}
    for r in _iter_jsonl(str(kdir / "gamma_*.jsonl")):
        s = r.get("slug")
        if s not in meta:
            continue
        g_first_seen.setdefault(s, r["t"])
        lv = _is_live(r.get("live"), r.get("ended"), None, r.get("elapsed"))
        if lv is True:
            if s not in g_first:
                g_first[s] = (r["t"], r)
        elif s not in g_first:
            g_notlive[s] = r["t"]          # live False/None before the first live poll = not started
    rows = []
    for s, md in sorted(meta.items(), key=lambda kv: (kv[1].get("scheduled") or 0, kv[0])):
        gid = str(md.get("game_id") or "")
        sched = md.get("scheduled")
        wf = ws_first.get(gid)
        gf = g_first.get(s)
        cands = []
        if wf:
            cands.append((wf[0], "ws"))
        if gf:
            cands.append((gf[0], "gamma"))
        actual, src = (min(cands) if cands else (None, ""))
        lo = None
        if actual is not None:
            nl = [t for t in ws_notlive.get(gid, []) if t < actual]
            if s in g_notlive and g_notlive[s] < actual:
                nl.append(g_notlive[s])
            lo = max(nl) if nl else None
        if actual is None:
            conf = ""
        elif lo is None:
            conf = "upper_bound"
        else:
            w = actual - lo
            conf = "high" if w <= 60 else "medium" if w <= 600 else "low"
            # the ws pushes every game update: with the socket up over the whole bracket, the first
            # live message IS the transition (to within push latency), however wide the bracket
            if src == "ws" and _ws_up(conn, lo, actual):
                conf = "high"
        rows.append(dict(
            slug=s, event_slug=md.get("event_slug") or "", game_id=gid,
            scheduled_kickoff=iso(sched), scheduled_unix=sched or "",
            end_date=iso(md.get("end")), end_minus_sched_s=(round(md["end"] - sched) if md.get("end") and sched else ""),
            actual_kickoff=iso(actual), actual_unix=round(actual, 3) if actual else "", source=src, confidence=conf,
            not_live_before=iso(lo), bracket_s=round(actual - lo, 1) if actual and lo else "",
            delta_s=round(actual - sched, 1) if actual and sched else "",
            ws_first_live=iso(wf[0]) if wf else "", ws_status=(wf[1].get("status") if wf else "") or "",
            ws_period=(wf[1].get("period") if wf else "") or "", ws_elapsed=(wf[1].get("elapsed") if wf else "") or "",
            gamma_first_live=iso(gf[0]) if gf else "", gamma_first_poll=iso(g_first_seen.get(s)),
            ws_first_seen=iso(ws_first_seen.get(gid)), clear_book_on_start=md.get("clear_book_on_start"),
        ))
    if write and rows:
        tmp = kdir / "kickoffs.csv.tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, kdir / "kickoffs.csv")
    return rows


def load_actual_kickoffs(kdir: Path = KDIR, min_conf: str = "medium") -> dict[str, dict]:
    """{slug: {"t": actual kickoff unix, "source": "ws"|"gamma", "confidence", "scheduled", "delta_s"}}
    from kickoff_watch's logs. upper_bound rows (game already running when first seen) are included only
    with min_conf="upper_bound"; their t is a LATEST possible start."""
    order = ["high", "medium", "low", "upper_bound"]
    keep = set(order[: order.index(min_conf) + 1])
    out = {}
    for r in derive(kdir, write=False):
        if r["actual_unix"] != "" and r["confidence"] in keep:
            out[r["slug"]] = dict(t=float(r["actual_unix"]), source="watch_" + r["source"], confidence=r["confidence"],
                                  scheduled=r["scheduled_unix"], delta_s=r["delta_s"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=str(REC))
    ap.add_argument("--out", default=str(KDIR))
    ap.add_argument("--derive", action="store_true", help="rebuild kickoffs.csv from the logs and exit")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S")
    logging.Formatter.converter = time.gmtime
    if a.derive:
        rows = derive(Path(a.out), True)
        print(f"{len(rows)} games, {sum(1 for r in rows if r['actual_kickoff'])} with an observed start "
              f"-> {Path(a.out) / 'kickoffs.csv'}")
        return
    asyncio.run(Watch(Path(a.rec), Path(a.out)).run())


if __name__ == "__main__":
    main()
