#!/usr/bin/env python
"""Track S (D11): offline kickoff detection from the recorder's own data (the clearBookOnStart wipe).

Sports game markets have clearBookOnStart=true: the exchange cancels every resting order, the book of
both outcome tokens collapses at once (observed: ~55 levels -> 3+3 within 0.3 s), and market makers
re-quote. OBSERVED 2026-10-15 (val-flyr-mibrf, dota2-amaru-skyrea, scheduled 21:00:00 UTC): the wipe
fires at the SCHEDULED gameStartTime (+0.06..0.32 s), although val went live 8 min later and dota2
later still. The wipe is therefore a schedule-driven event, which matches the docs' warning that
early starts may not be cleared in time. It tells you when the book was cleared, not when the game
began. kickoff_watch.py supplies the game start, and actual_kickoffs() merges the two. This script finds that wipe for each recorded market and compares it with the scheduled
kickoff (gamma gameStartTime, from kickoff/meta.json, else the books' market_end_at) and with
kickoff_watch's observed start (kickoff/kickoffs.csv logic), when that is available.

Sources, per outcome token:
  raw    rec_out/raw/<day>/events_*.jsonl.zst (polyrec.sinks raw log). Each token's book is rebuilt
         from 'book' / 'rest_book' snapshots plus 'price_change' deltas, and the detector looks for
           book_empty   a 'book' snapshot with both ladders empty, the token having had >= MIN_LEVELS levels
           emptied      the rebuilt book reaches 0 bid and 0 ask levels after having had >= MIN_LEVELS
           mass_cancel  within WIN_S seconds, >= MASS_FRAC of the levels on EACH side were removed
                        (size -> 0), and >= MIN_LEVELS levels in all
  books  rec_out/books parquet rows: the first row with both ladders empty after a row with
         levels (fallback when the raw log has rotated out, raw_retention_days = 3).
The search window is [scheduled - --pre (3 h), scheduled + --post (1 h)]. A market's wipe time is
the earliest strong event (book_empty / emptied / books_empty) over its two tokens. If there is none,
it is the earliest mass_cancel.

COVERAGE LIMITS (recorder_pregame.toml): book rows stop at gamma endDate + 300 s (post_close_s) and
the subscription, and with it the raw log, ends at endDate + 600 s (linger_s). For the traditional
sports endDate == gameStartTime, so a game that starts more than ~10 min LATE leaves no wipe in the
recorder's data. Such a late start can only be seen by kickoff_watch.py (the sports websocket and
gamma polls, which run to +40 min). Early starts are fully visible: the book is recorded from
discovery (up to 48 h before) until endDate. Esports are the exception: their gamma endDate is
gameStartTime + 6 h, so the recorder keeps their in-play books for 6 h and late starts show up too.

    python detect_kickoff.py                       # rec_out -> kickoff/detect.csv (+ comparison table)
    python detect_kickoff.py --synthetic           # self-test on a synthetic rec dir

For build_sports.py: actual_kickoffs([rec_dirs]) -> {slug: {"t": unix, "source", "confidence", ...}}.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "rec_out"
KDIR = HERE / "kickoff"
MIN_LEVELS = 4
MASS_FRAC = 0.8
WIN_S = 10.0
PRE_S = 3 * 3600
POST_S = 3600
STRONG = ("book_empty", "emptied", "books_empty")

sys.path.insert(0, str(HERE))
import kickoff_watch as kw  # noqa: E402


# ----------------------------------------------------------------------------------------- inputs


def _book_files(rec: Path) -> list[str]:
    fs = glob.glob(str(rec / "books" / "*" / "*.parquet"))
    fs += glob.glob(str(rec / "books" / "*" / "_parts" / "*.parquet"))
    return sorted(fs)


def load_markets(recs: list[Path]) -> dict[str, dict]:
    """slug -> {tokens: {token: outcome}, end: unix (books market_end_at)} from the books parquet."""
    import pyarrow.parquet as pq
    out: dict[str, dict] = {}
    for rec in recs:
        for f in _book_files(Path(rec)):
            try:
                t = pq.read_table(f, columns=["market_slug", "token_id", "outcome", "market_end_at"]).to_pandas()
            except Exception:  # noqa: BLE001  (a part being written / compacted away)
                continue
            for (s, tok, oc), g in t.groupby(["market_slug", "token_id", "outcome"], sort=False):
                m = out.setdefault(s, dict(tokens={}, end=None))
                m["tokens"][str(tok)] = oc
                e = g.market_end_at.max().timestamp()
                m["end"] = e if m["end"] is None else max(m["end"], e)
    return out


def scheduled(markets: dict, kdir: Path) -> dict[str, tuple[float, str]]:
    """slug -> (scheduled kickoff unix, source): gamma gameStartTime from kickoff/meta.json, else endDate."""
    try:
        meta = json.loads((Path(kdir) / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    out = {}
    for s, m in markets.items():
        md = meta.get(s) or {}
        if md.get("game_start"):
            out[s] = (float(md["game_start"]), "gameStartTime")
        elif md.get("scheduled"):
            out[s] = (float(md["scheduled"]), "meta")
        else:
            out[s] = (float(m["end"]), "market_end_at")
    return out


def raw_files(recs: list[Path]) -> list[str]:
    fs = []
    for rec in recs:
        fs += glob.glob(str(Path(rec) / "raw" / "*" / "events_*.jsonl.zst"))
    return sorted(fs, key=lambda f: (os.path.basename(f).split("_")[1], f))   # by hour, then run


def iter_raw_lines(path: str):
    """Decoded JSON objects of one raw file (polyrec.sinks.iter_raw semantics: tolerates a torn tail)."""
    import orjson
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh, read_across_frames=True)
        pending = b""
        while True:
            try:
                chunk = reader.read(1 << 20)
            except zstd.ZstdError:
                break
            if not chunk:
                break
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for ln in lines:
                if ln:
                    try:
                        yield orjson.loads(ln)
                    except orjson.JSONDecodeError:
                        continue


# --------------------------------------------------------------------------------------- detector


class Tok:
    __slots__ = ("bids", "asks", "peak", "rm", "first", "last", "ev")

    def __init__(self):
        self.bids: dict[str, float] = {}
        self.asks: dict[str, float] = {}
        self.peak = 0                     # max levels since the last strong wipe
        self.rm: deque = deque()          # (t, side) removals in the last WIN_S
        self.first = None                 # first / last update time seen in the window
        self.last = None
        self.ev: dict[str, tuple] = {}    # kind -> (t, levels_before)

    def n(self):
        return len(self.bids) + len(self.asks)

    def hit(self, kind, t, before):
        if kind not in self.ev:
            self.ev[kind] = (t, before)


def _t(o) -> float:
    ts = o.get("timestamp")
    try:
        return int(ts) / 1000.0
    except (TypeError, ValueError):
        return (o.get("r") or 0) / 1000.0


def _snap(tk: Tok, o: dict, t: float):
    before = tk.n()
    tk.bids = {str(x["price"]): float(x["size"]) for x in o.get("bids") or [] if float(x["size"]) > 0}
    tk.asks = {str(x["price"]): float(x["size"]) for x in o.get("asks") or [] if float(x["size"]) > 0}
    if not tk.bids and not tk.asks and tk.peak >= MIN_LEVELS:
        tk.hit("book_empty", t, before)
    tk.peak = max(tk.peak, tk.n())


def _delta(tk: Tok, it: dict, t: float):
    side = tk.bids if it.get("side") == "BUY" else tk.asks
    p, sz = str(it["price"]), float(it["size"])
    before = tk.n()
    if sz > 0:
        side[p] = sz
        tk.peak = max(tk.peak, tk.n())
        return
    if side.pop(p, None) is None:
        return
    tk.rm.append((t, it.get("side")))
    while tk.rm and tk.rm[0][0] < t - WIN_S:
        tk.rm.popleft()
    if not tk.bids and not tk.asks and tk.peak >= MIN_LEVELS:
        tk.hit("emptied", t, before)
    nb = sum(1 for _, s in tk.rm if s == "BUY")
    na = len(tk.rm) - nb
    if (len(tk.rm) >= MIN_LEVELS and nb >= 1 and na >= 1
            and nb >= MASS_FRAC * (nb + len(tk.bids)) and na >= MASS_FRAC * (na + len(tk.asks))):
        tk.hit("mass_cancel", tk.rm[0][0], len(tk.rm) + tk.n())


def scan_raw(files: list[str], windows: dict[str, tuple[float, float]]) -> dict[str, Tok]:
    """windows: token -> (lo, hi) unix. Returns token -> Tok with the first event of each kind in-window."""
    toks: dict[str, Tok] = {}
    lo_all = min((w[0] for w in windows.values()), default=0)
    for f in files:
        for o in iter_raw_lines(f):
            et = o.get("event_type")
            if et == "price_change":
                t = _t(o)
                if t < lo_all - 3600:
                    continue
                for it in o.get("price_changes") or ():
                    a = it.get("asset_id")
                    w = windows.get(a)
                    if w is None or t > w[1]:
                        continue
                    tk = toks.get(a)
                    if tk is None:
                        continue                  # no snapshot yet: wait for one
                    if t >= w[0]:
                        tk.first = tk.first or t
                        tk.last = t
                        _delta(tk, it, t)
                    else:
                        _delta_quiet(tk, it)
            elif et in ("book", "rest_book"):
                a = o.get("asset_id")
                w = windows.get(a)
                if w is None:
                    continue
                t = _t(o)
                if t > w[1]:
                    continue
                tk = toks.setdefault(a, Tok())
                if t >= w[0]:
                    tk.first = tk.first or t
                    tk.last = t
                    _snap(tk, o, t)
                else:
                    tk.bids = {str(x["price"]): float(x["size"]) for x in o.get("bids") or [] if float(x["size"]) > 0}
                    tk.asks = {str(x["price"]): float(x["size"]) for x in o.get("asks") or [] if float(x["size"]) > 0}
                    tk.peak = max(tk.peak, tk.n())
    return toks


def _delta_quiet(tk: Tok, it: dict):
    side = tk.bids if it.get("side") == "BUY" else tk.asks
    p, sz = str(it["price"]), float(it["size"])
    if sz > 0:
        side[p] = sz
        tk.peak = max(tk.peak, tk.n())
    else:
        side.pop(p, None)


def scan_books(recs: list[Path], windows_slug: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """slug -> {t: first books row in-window with both ladders empty after a row with levels, last: last row}."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    out: dict[str, dict] = {}
    rows = []
    for rec in recs:
        for f in _book_files(Path(rec)):
            try:
                t = pq.read_table(f, columns=["market_slug", "token_id", "captured_at", "bid_prices", "ask_prices"])
            except Exception:  # noqa: BLE001
                continue
            t = t.filter(pc.is_in(t["market_slug"], value_set=__import__("pyarrow").array(list(windows_slug))))
            if t.num_rows:
                d = t.to_pandas()
                d["nl"] = d.bid_prices.map(len) + d.ask_prices.map(len)
                rows.append(d[["market_slug", "token_id", "captured_at", "nl"]])
    if not rows:
        return out
    import pandas as pd
    d = pd.concat(rows).sort_values("captured_at")
    d["ts"] = d.captured_at.map(lambda x: x.timestamp())
    for (s, tok), g in d.groupby(["market_slug", "token_id"]):
        lo, hi = windows_slug[s]
        o = out.setdefault(s, dict(t=None, last=float(g.ts.max()), first=float(g.ts.min())))
        o["last"], o["first"] = max(o["last"], float(g.ts.max())), min(o["first"], float(g.ts.min()))
        had = (g.nl >= MIN_LEVELS).cummax().shift(fill_value=False)
        hit = g[(g.nl == 0) & had & (g.ts >= lo) & (g.ts <= hi)]
        if len(hit):
            t0 = float(hit.ts.iloc[0])
            o["t"] = t0 if o["t"] is None else min(o["t"], t0)
    return out


# ------------------------------------------------------------------------------------------- main


def detect(recs: list[Path] | None = None, kdir: Path = KDIR, pre_s: float = PRE_S, post_s: float = POST_S,
           use_cache: bool = True, write: bool = True) -> list[dict]:
    recs = [Path(r) for r in (recs or [REC])]
    kdir = Path(kdir)
    files = raw_files(recs)
    sig = [(f, os.path.getsize(f)) for f in files] + [("books", len(sum((_book_files(r) for r in recs), [])))]
    cache = kdir / "detect_cache.json"
    if use_cache and cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("sig") == [list(x) for x in sig] and c.get("params") == [pre_s, post_s]:
                return c["rows"]
        except (OSError, ValueError):
            pass
    markets = load_markets(recs)
    sched = scheduled(markets, kdir)
    win_tok, win_slug, tok_slug = {}, {}, {}
    for s, m in markets.items():
        k = sched[s][0]
        w = (k - pre_s, k + post_s)
        win_slug[s] = w
        for tok in m["tokens"]:
            win_tok[tok] = w
            tok_slug[tok] = s
    toks = scan_raw(files, win_tok)
    books = scan_books(recs, win_slug)
    try:
        watch = {r["slug"]: r for r in kw.derive(kdir, write=False)}
    except Exception:  # noqa: BLE001
        watch = {}
    rows = []
    for s, m in sorted(markets.items(), key=lambda kv: (sched[kv[0]][0], kv[0])):
        k, ksrc = sched[s]
        evs: dict[str, tuple] = {}
        last = None
        for tok in m["tokens"]:
            tk = toks.get(tok)
            if tk is None:
                continue
            last = max(filter(None, [last, tk.last]), default=None)
            for kind, (t, nb) in tk.ev.items():
                if kind not in evs or t < evs[kind][0]:
                    evs[kind] = (t, nb)
        b = books.get(s) or {}
        if b.get("t") is not None:
            evs["books_empty"] = (b["t"], None)
        strong = sorted((v[0], kind) for kind, v in evs.items() if kind in STRONG)
        weak = sorted((v[0], kind) for kind, v in evs.items() if kind not in STRONG)
        pick = strong[0] if strong else (weak[0] if weak else (None, ""))
        # a mass cancel shortly before the strong event is the start of the same wipe
        if strong and weak and 0 <= strong[0][0] - weak[0][0] <= 30:
            pick = (weak[0][0], f"{weak[0][1]}+{strong[0][1]}")
        wt, kind = pick
        wr = watch.get(s) or {}
        wa = float(wr["actual_unix"]) if wr.get("actual_unix") not in (None, "") else None
        raw_end = last
        data_end = max(filter(None, [raw_end, b.get("last")]), default=None)
        note = ""
        if wt is None and b.get("first") is not None and b["first"] > k:
            note = f"recording began {kw.iso(b['first'])}, after the scheduled kickoff (in-play book only)"
        elif wt is None:
            if data_end is not None and data_end < k + 60:
                note = "data ends before kickoff+60 s (game not started yet, or recorder stopped)"
            elif data_end is not None:
                note = (f"no wipe in data up to {kw.iso(data_end)} (sched{(data_end - k) / 60:+.0f} min): if the game "
                        f"started later, only kickoff_watch sees it")
            else:
                note = "no raw/books data in window"
        rows.append(dict(
            slug=s, scheduled_kickoff=kw.iso(k), scheduled_unix=k, sched_source=ksrc, end_date=kw.iso(m["end"]),
            wipe_time=kw.iso(wt), wipe_unix=round(wt, 3) if wt else "", wipe_kind=kind,
            wipe_delta_s=round(wt - k, 1) if wt else "",
            levels_before=(evs.get(kind.split("+")[0]) or (None, ""))[1] if wt else "",
            events=";".join(f"{kd}@{kw.iso(v[0])}" for kd, v in sorted(evs.items(), key=lambda kv: kv[1][0])),
            data_end=kw.iso(data_end),
            watch_actual=wr.get("actual_kickoff", ""), watch_source=wr.get("source", ""),
            watch_confidence=wr.get("confidence", ""),
            wipe_minus_watch_s=round(wt - wa, 1) if wt and wa else "", note=note))
    if write:
        kdir.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(kdir / "detect.csv.tmp", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            os.replace(kdir / "detect.csv.tmp", kdir / "detect.csv")
        if use_cache:
            cache.write_text(json.dumps(dict(sig=sig, params=[pre_s, post_s], rows=rows)))
    return rows


def actual_kickoffs(recs: list[Path] | None = None, kdir: Path = KDIR, min_watch_conf: str = "medium") -> dict[str, dict]:
    """{slug: {"t", "source", "confidence", "wipe", "game_start", "scheduled", "delta_s"}}.

    t = the END OF THE PRE-GAME REGIME = the earlier of (a) the clearBookOnStart book wipe found in the
    recorder data and (b) the game start observed by kickoff_watch (confidence >= min_watch_conf).
    Observed 2026-10-15: the wipe fires at the SCHEDULED gameStartTime (to 0.3 s) even when the game
    starts later, so (a) pins the scheduled cut and (b) catches early starts. A wipe counts when it is
    strong (empty book) or a mass cancel of >= 20 levels; a smaller mass cancel is used only if
    nothing else exists (confidence low)."""
    watch = kw.load_actual_kickoffs(kdir, min_conf=min_watch_conf) if (Path(kdir) / "meta.json").exists() else {}
    wipes = {}
    for r in detect(recs, kdir, write=False):
        if r["wipe_unix"] == "":
            continue
        strong = any(k in r["wipe_kind"] for k in STRONG) or (r["levels_before"] or 0) >= 20
        wipes[r["slug"]] = (float(r["wipe_unix"]), r["wipe_kind"], "high" if strong else "low", r["scheduled_unix"])
    out = {}
    for s in set(watch) | set(wipes):
        w, g = wipes.get(s), watch.get(s)
        cands = []
        if w and (w[2] == "high" or not g):
            cands.append((w[0], "wipe_" + w[1], w[2]))
        if g:
            cands.append((g["t"], g["source"], g["confidence"]))
        t, src, conf = min(cands)
        sched = (w[3] if w else None) or (float(g["scheduled"]) if g and g["scheduled"] != "" else None)
        out[s] = dict(t=t, source=src, confidence=conf, wipe=w[0] if w else None, game_start=g["t"] if g else None,
                      scheduled=sched, delta_s=round(t - sched, 1) if sched else None)
    return out


# -------------------------------------------------------------------------------------- synthetic


def make_synthetic(root: Path, k0: float) -> dict:
    """Three games at kickoff k0: A wiped by an empty 'book' snapshot 7 min EARLY, B by per-level
    cancels 2 min late, C no wipe at all (data ends at +600 s); D a pre-game mass pull by one maker
    of 60% of levels (must NOT count). Returns the expected wipe times."""
    import orjson
    import pandas as pd
    import zstandard as zstd
    day = time.strftime("%Y-%m-%d", time.gmtime(k0))
    games = {"synA": k0 - 420, "synB": k0 + 120, "synC": None, "synD": None}
    rows, ev = [], []
    lv = [f"0.{i:02d}" for i in range(30, 70)]
    for gi, (g, wipe) in enumerate(games.items()):
        slug = f"nhl-{g.lower()}-x-{day}"
        for oi in range(2):
            tok = f"{gi}{oi}" * 10
            bids, asks = lv[:18], lv[20:]
            t = k0 - 3600
            ev.append(dict(event_type="book", asset_id=tok, timestamp=str(int(t * 1000)), r=int(t * 1000),
                           bids=[dict(price=p, size="100") for p in bids], asks=[dict(price=p, size="100") for p in asks]))
            rows.append(dict(market_slug=slug, token_id=tok, outcome="Yes" if oi == 0 else "No",
                             market_end_at=pd.Timestamp(k0, unit="s", tz="UTC"),
                             captured_at=pd.Timestamp(t, unit="s", tz="UTC"),
                             bid_prices=[float(p) for p in bids], ask_prices=[float(p) for p in asks]))
            for j in range(0, 4200, 30):          # churn: one level updated every 30 s
                tt = k0 - 3600 + j
                p = bids[j // 30 % len(bids)]
                if wipe is None or tt < wipe:
                    ev.append(dict(event_type="price_change", timestamp=str(int(tt * 1000)), r=int(tt * 1000),
                                   price_changes=[dict(asset_id=tok, price=p, size="150", side="BUY")]))
            if g == "synA":
                ev.append(dict(event_type="book", asset_id=tok, timestamp=str(int(wipe * 1000)), r=int(wipe * 1000),
                               bids=[], asks=[]))
            if g == "synB":
                items = [dict(asset_id=tok, price=p, size="0", side="BUY") for p in bids] + \
                        [dict(asset_id=tok, price=p, size="0", side="SELL") for p in asks]
                for c in range(0, len(items), 5):     # spread over 2 s in frames of 5
                    tt = wipe + c * 0.1
                    ev.append(dict(event_type="price_change", timestamp=str(int(tt * 1000)), r=int(tt * 1000),
                                   price_changes=items[c:c + 5]))
            if g == "synD":                           # one maker pulls 60% of each side 30 min before
                tt = k0 - 1800
                items = [dict(asset_id=tok, price=p, size="0", side="BUY") for p in bids[:11]] + \
                        [dict(asset_id=tok, price=p, size="0", side="SELL") for p in asks[:12]]
                ev.append(dict(event_type="price_change", timestamp=str(int(tt * 1000)), r=int(tt * 1000),
                               price_changes=items))
    ev.sort(key=lambda o: o["r"])
    (root / "raw" / day).mkdir(parents=True, exist_ok=True)
    (root / "books" / day).mkdir(parents=True, exist_ok=True)
    data = b"".join(orjson.dumps(o) + b"\n" for o in ev)
    hh = time.strftime("%Y-%m-%dT%H", time.gmtime(k0))
    (root / "raw" / day / f"events_{hh}_synthetic.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(data))
    pd.DataFrame(rows).to_parquet(root / "books" / day / f"books_{hh}.parquet")
    return {f"nhl-{g.lower()}-x-{day}": w for g, w in games.items()}


def selftest(tmp: Path) -> bool:
    k0 = float(int(time.time() // 3600 * 3600) - 86400)
    exp = make_synthetic(tmp, k0)
    rows = detect([tmp], tmp / "kickoff", use_cache=False, write=True)
    ok = True
    for r in rows:
        e = exp[r["slug"]]
        got = r["wipe_unix"] if r["wipe_unix"] != "" else None
        good = (got is None and e is None) or (got is not None and e is not None and abs(got - e) < 1)
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {r['slug']:28s} expected {kw.iso(e) or '-':20s} got {r['wipe_time'] or '-':20s} "
              f"{r['wipe_kind']:24s} delta {r['wipe_delta_s']}  {r['note']}")
    ak = actual_kickoffs([tmp], tmp / "kickoff")
    print("actual_kickoffs:", {s: (kw.iso(v['t']), v['source'], v['confidence']) for s, v in ak.items()})
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", action="append", help="recorder out dir (default rec_out); repeatable")
    ap.add_argument("--kdir", default=str(KDIR))
    ap.add_argument("--pre", type=float, default=PRE_S, help="search from scheduled - PRE seconds")
    ap.add_argument("--post", type=float, default=POST_S, help="search until scheduled + POST seconds")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="self-test on a synthetic rec dir")
    ap.add_argument("--tmp", help="--synthetic: dir to build it in (default: a temp dir)")
    a = ap.parse_args(argv)
    if a.synthetic:
        import tempfile
        d = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="kick_syn_"))
        ok = selftest(d)
        print("SELFTEST", "PASSED" if ok else "FAILED", d)
        sys.exit(0 if ok else 1)
    rows = detect([Path(r) for r in (a.rec or [REC])], Path(a.kdir), a.pre, a.post, use_cache=not a.no_cache)
    now = time.time()
    past = [r for r in rows if r["scheduled_unix"] < now]
    print(f"{len(rows)} markets, {len(past)} with scheduled kickoff in the past, "
          f"{sum(1 for r in rows if r['wipe_time'])} wipes detected -> {Path(a.kdir) / 'detect.csv'}")
    for r in past:
        print(f"  {r['slug']:34s} sched {r['scheduled_kickoff']} wipe {r['wipe_time'] or '-':20s} "
              f"{r['wipe_kind']:22s} d={r['wipe_delta_s']!s:>8s}  watch {r['watch_actual'] or '-'} "
              f"({r['watch_confidence']}) {r['note']}")


if __name__ == "__main__":
    main()
