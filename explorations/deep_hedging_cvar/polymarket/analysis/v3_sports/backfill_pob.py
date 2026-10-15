#!/usr/bin/env python3
"""Backfill historical pre-game SPORTS L2 order books from PolyOrderBooks (api.polyorderbooks.com), plus
keyless trades (data-api) and market metadata (gamma), in the recorder's normalized raw layout.

    OUT/books/<day>/books_<day>T<HH>.parquet      data/raw/updown_*.parquet schema, column for column
    OUT/trades/<day>/trades_<day>T<HH>.parquet    data/raw/trades.parquet schema (hour = kickoff hour)
    OUT/markets/<day>/markets_<day>T<HH>.parquet  data/raw/markets.parquet schema + trailing sports columns
    OUT/_state/...   per-chunk / per-market done markers (resume), OUT/_gamma/ selection cache,
    OUT/backfill.log

Files are written as part files (`<name>/<day>/_parts/...`) and compacted per hour with the recorder's own
PartSink (recorder/polyrec/sinks.py), so `polyrec.sinks.read_dataset(OUT, "books")` and plain
`pd.read_parquet(file)` both work, exactly as for the recorder's out_dir.

Market selection: gamma `/markets?tag_id=100639` (sports "Games") with `sports_market_types` (default
moneyline), closed, kickoff (`gameStartTime`) in [--start, --end] (UTC dates, inclusive), league = slug
prefix restricted to --leagues (priority order = census volume order, see README_backfill.md). Books are
fetched only for the window [kickoff - --hours-before, kickoff + --post-min).

The API key is read ONLY from the env var POLYORDERBOOKS_API_KEY. It is sent as the X-API-Key header and is
never written to disk or logs (a logging filter also masks it, in case a library ever echoes it).

    python backfill_pob.py --start 2026-09-20 --end 2026-10-14 --dry-run        # no key needed
    python backfill_pob.py --start 2026-10-10 --end 2026-10-14 --stage meta     # trades+markets, no key
    POLYORDERBOOKS_API_KEY=pob_... python backfill_pob.py --start ... --end ... --probe
    POLYORDERBOOKS_API_KEY=pob_... python backfill_pob.py --start ... --end ...  # books + meta, resumable
    python backfill_pob.py --self-test                                        # offline mocked-API test
    python backfill_pob.py --verify --out <OUT>                               # schema check of outputs
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]                                   # .../polymarket
sys.path.insert(0, str(P / "recorder"))
from polyrec.discovery import GAMMA_KEYS, jl, parse_ts, resolution  # noqa: E402
from polyrec.metadata import market_row  # noqa: E402
from polyrec.schema import BOOK_SCHEMA, BOOK_SCHEMA_RAW, MARKETS_SCHEMA, TRADES_SCHEMA, UTC_US  # noqa: E402
from polyrec.sinks import PartSink, dataset_files, hour_key  # noqa: E402
from polyrec.trades import LegSet, build_trade_rows, rows_consistency, trade_rows_to_table  # noqa: E402

# data/fetch_trades.py (stdlib HTTP with retries/backoff, data-api constants), loaded by path so that
# data/ is not put on sys.path (it holds generic module names: build, features, load).
_spec = importlib.util.spec_from_file_location("_pm_fetch_trades", P / "data" / "fetch_trades.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

log = logging.getLogger("backfill_pob")

POB_BASE = "https://api.polyorderbooks.com"
ENV_KEY = "POLYORDERBOOKS_API_KEY"
SPORTS_GAMES_TAG = 100639
# Priority order (task list / analysis/sports/out/census_summary.txt volume ranking).
LEAGUES = ["nfl", "cs2", "atp", "lol", "ucl", "epl", "lal", "uel", "bun", "sea", "cfb", "wta", "mlb",
           "nhl", "nba", "dota2", "val"]
RES_S = {"250ms": 0.25, "1s": 1, "1m": 60, "60s": 60, "5m": 300, "10m": 600, "15m": 900, "1h": 3600,
         "6h": 21600, "1d": 86400}
BOOK_PAGE = 200                 # API max points per page (books)
BYTES_PER_ROW = 60              # data/raw: 43 MB / 897k rows ~ 48 B; sports ladders a bit deeper
_S = pa.large_string()
MARKETS_EXTRA = [("kickoff", UTC_US), ("league", _S), ("sports_market_type", _S), ("event_slug", _S),
                 ("outcome_0", _S), ("outcome_1", _S), ("volume", pa.float64()),
                 ("window_start", UTC_US), ("window_end", UTC_US), ("trades_consistent", pa.bool_())]
MARKETS_SCHEMA_SPORTS = pa.schema(list(MARKETS_SCHEMA) + [pa.field(n, t) for n, t in MARKETS_EXTRA])
BOOK_SORT = ["market_slug", "token_id", "captured_at"]
BOOK_DICT = ["market_slug", "coin", "contract_length", "token_id", "outcome"]


# ----------------------------------------------------------------------------------------- utilities
class KeyMask(logging.Filter):
    """Masks the API key in any log record (defence in depth; we never log it on purpose)."""

    def __init__(self, key: str | None):
        super().__init__()
        self.key = key

    def filter(self, record):
        if self.key:
            msg = record.getMessage()
            if self.key in msg:
                record.msg, record.args = msg.replace(self.key, "***"), ()
        return True


def setup_logging(out: Path | None, level: str, key: str | None):
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = time.gmtime
    hs = [logging.StreamHandler(sys.stderr)]
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        hs.append(logging.FileHandler(out / "backfill.log"))
    for h in hs:
        h.setFormatter(fmt)
        h.addFilter(KeyMask(key))
        root.addHandler(h)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def day_start(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    os.replace(tmp, path)


def res_seconds(res: str) -> float:
    if res not in RES_S:
        raise SystemExit(f"unsupported resolution {res!r}; allowed {sorted(RES_S)}")
    return RES_S[res]


# ----------------------------------------------------------------------------- gamma market selection
SEL_KEYS = GAMMA_KEYS + ["sportsMarketType", "gameStartTime", "closed", "enableOrderBook", "secondsDelay",
                         "clearBookOnStart"]


def _trim_market(m: dict) -> dict:
    g = {k: m.get(k) for k in SEL_KEYS}
    ev = (m.get("events") or [{}])[0]
    g["eventMetadata"] = ev.get("eventMetadata")
    g["eventStartTimeEv"] = ev.get("startTime")
    g["eventSlug"] = ev.get("slug")
    return g


def gamma_window(a: int, b: int, types: list[str], closed: bool, cache: Path | None) -> list[dict]:
    """All sports-game markets with endDate in [a, b) (sports endDate == kickoff), offset pagination."""
    key = hashlib.sha1(json.dumps([a, b, sorted(types), closed]).encode()).hexdigest()[:16]
    cp = cache / f"{iso(a)[:13]}_{key}.json.gz" if cache else None
    if cp is not None and cp.exists():
        with gzip.open(cp, "rt") as f:
            return json.load(f)
    out, off = [], 0
    while True:
        q = [("tag_id", SPORTS_GAMES_TAG), ("closed", "true" if closed else "false"),
             ("end_date_min", iso(a)), ("end_date_max", iso(b)), ("limit", 100), ("offset", off)]
        q += [("sports_market_types", t) for t in types]
        pg = ft.http_json(f"{ft.GAMMA}/markets?{urllib.parse.urlencode(q)}")
        out += [_trim_market(m) for m in pg]
        if len(pg) < 100:
            break
        off += 100
    if cp is not None and b < time.time() - 2 * 86400:          # stable once closed for a while
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(out, f, separators=(",", ":"))
        os.replace(tmp, cp)
    return out


def select_markets(args) -> list[dict]:
    """Gamma selection -> list of market dicts (gamma record under 'g' + derived fields), priority order."""
    lo, hi = day_start(args.start), day_start(args.end) + 86400
    types = [] if args.market_types == ["all"] else args.market_types
    leagues = None if args.leagues == ["all"] else args.leagues
    cache = Path(args.out) / "_gamma"
    # gamma endDate == kickoff for most leagues, but kickoff + 6 h (mlbb, ...) or + 7 d (atp, wta, mlb):
    # query endDate over [lo - 1 d, hi + 8 d] and filter on kickoff
    wins = [(t, t + 6 * 3600) for t in range(lo - 86400, hi + 8 * 86400, 6 * 3600)]
    raw = []
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(gamma_window, a, b, types, True, cache) for a, b in wins]
        if hi > time.time() - 3 * 86400:                  # not-yet-closed markets near the range end
            futs += [ex.submit(gamma_window, a, b, types, False, None) for a, b in wins if b > time.time() - 3 * 86400]
        for f in as_completed(futs):
            raw += f.result()
    seen, out = set(), []
    hb, post = int(round(args.hours_before * 3600)), int(round(args.post_min * 60))
    for g in raw:
        slug = g.get("slug") or ""
        if slug in seen:
            continue
        seen.add(slug)
        league = slug.split("-")[0]
        if leagues is not None and league not in leagues:
            continue
        toks, outs = jl(g.get("clobTokenIds") or "[]"), jl(g.get("outcomes") or "[]")
        if len(toks) != 2 or len(outs) != 2 or g.get("enableOrderBook") is False:
            continue
        ko = parse_ts(g.get("gameStartTime")) or parse_ts(g.get("eventStartTimeEv")) or parse_ts(g.get("endDate"))
        if ko is None or not (lo <= ko < hi):
            continue
        if ko + post > time.time() - 600:                 # window not finished yet
            continue
        vol = float(g.get("volumeNum") or g.get("volumeClob") or 0)
        if vol < args.min_volume:
            continue
        ko = int(ko)
        out.append(dict(slug=slug, league=league, smt=g.get("sportsMarketType") or "", kickoff=ko,
                        ws=ko - hb, we=ko + post, tokens=[str(t) for t in toks], outcomes=list(outs),
                        cond=g.get("conditionId"), volume=vol, g=g))
    lrank = {l: i for i, l in enumerate(leagues or [])}
    trank = {t: i for i, t in enumerate(types)}
    if args.order == "day":       # oldest kickoff day first (rolling plan history expires it first)
        out.sort(key=lambda m: (m["kickoff"] // 86400, trank.get(m["smt"], 99), lrank.get(m["league"], 99),
                                m["kickoff"], m["slug"]))
    elif args.order == "priority":
        out.sort(key=lambda m: (trank.get(m["smt"], 99), lrank.get(m["league"], 99), m["kickoff"], m["slug"]))
    elif args.order == "volume":
        out.sort(key=lambda m: -m["volume"])
    else:
        out.sort(key=lambda m: (m["kickoff"], m["slug"]))
    if args.max_markets:
        out = out[: args.max_markets]
    return out


# --------------------------------------------------------------------------------- PolyOrderBooks API
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class BudgetExhausted(Exception):
    pass


class RateLimiter:
    """Evenly spaced requests at <= rpm (shared by all worker threads)."""

    def __init__(self, rpm: float):
        self.dt = 60.0 / max(rpm, 1e-9)
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.dt
        if t > now:
            time.sleep(t - now)

    def pause(self, s: float):
        with self.lock:
            self.next = max(self.next, time.monotonic() + s)


class POB:
    def __init__(self, key: str, rpm: float = 60, max_requests: int = 0, timeout: float = 60, tries: int = 6,
                 category: str | None = "sports"):
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": key, "Accept": "application/json",
                               "User-Agent": "mars-polymarket-backfill/1.0"})
        self.lim = RateLimiter(rpm)
        self.max_requests, self.timeout, self.tries, self.category = max_requests, timeout, tries, category
        self.n = 0
        self.lock = threading.Lock()

    def set_rpm(self, rpm: float):
        self.lim.dt = 60.0 / max(rpm, 1e-9)

    def get(self, path: str, params: dict | None = None) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if self.category and "category" not in params:
            params["category"] = self.category
        delay, k, n429 = 1.0, 0, 0
        while True:
            with self.lock:
                if self.max_requests and self.n >= self.max_requests:
                    raise BudgetExhausted(f"--max-requests {self.max_requests} reached")
                self.n += 1
            self.lim.wait()
            try:
                r = self.s.get(POB_BASE + path, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                k += 1
                if k >= self.tries:
                    raise ApiError(0, "network", type(e).__name__) from None
                log.warning("network error %s on %s, retry %d", type(e).__name__, path, k)
                time.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 200:
                return r.json()
            try:
                body = r.json()
            except ValueError:
                body = {"error": "non_json", "message": r.text[:200]}
            code = body.get("error") or "unknown"
            msg = body.get("message") or json.dumps(body.get("detail"))[:300]
            if r.status_code == 429:
                n429 += 1
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra)
                except (TypeError, ValueError):
                    wait = min(5 * 2 ** min(n429, 6), 300)
                if n429 > 40:                      # > ~1-2 h of 429s: daily quota, most likely
                    raise BudgetExhausted(f"persistent 429 on {path}: {msg}")
                log.warning("429 rate limited (%s); sleeping %.0fs", msg, wait)
                self.lim.pause(wait)
                continue
            if r.status_code >= 500:
                k += 1
                if k >= self.tries:
                    raise ApiError(r.status_code, code, msg)
                log.warning("HTTP %d on %s, retry %d", r.status_code, path, k)
                time.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 60)
                continue
            raise ApiError(r.status_code, code, msg)      # 400/401/403/404/422: not retryable

    def usage(self) -> dict:
        cat, self.category = self.category, None          # /v1/usage takes no category
        try:
            return self.get("/v1/usage")
        finally:
            self.category = cat

    def token_books(self, token_id: str, a: float, b: float, res: str, limit: int = BOOK_PAGE):
        """All pages of GET /v1/tokens/{token_id}/books for [a, b). Returns (points, pages, response meta)."""
        pts, pages, cursor, meta = [], 0, None, {}
        while True:
            d = self.get(f"/v1/tokens/{token_id}/books",
                         {"start_ts": int(a), "end_ts": int(math.ceil(b)), "resolution": res, "limit": limit,
                          "cursor": cursor})
            pages += 1
            pts += d.get("data") or []
            meta = {k: d.get(k) for k in ("market_id", "token_id", "token_label", "resolution")}
            cursor = (d.get("metadata") or {}).get("next_cursor")
            if not cursor:
                return pts, pages, meta


# --------------------------------------------------------------------------------- books conversion
def _ladder(levels, desc: bool):
    lv = []
    for x in levels or []:
        if isinstance(x, dict):
            p, s = x.get("price", x.get("p")), x.get("size", x.get("s"))
        else:
            p, s = x[0], x[1]
        p, s = float(p), float(s)
        if s > 0:
            lv.append((p, s))
    lv.sort(key=lambda z: -z[0] if desc else z[0])
    return [z[0] for z in lv], [z[1] for z in lv]


def mirror_point(pt: dict) -> dict:
    """Outcome-1 book from the outcome-0 book: the two tokens share one CLOB book (bid p == ask 1-p)."""
    return {"t": pt["t"],
            "bids": [[round(1.0 - float(p), 6), s] for p, s in (pt.get("asks") or [])],
            "asks": [[round(1.0 - float(p), 6), s] for p, s in (pt.get("bids") or [])]}


def points_to_columns(m: dict, idx: int, pts: list, res_s: float, shift: bool, lo: float, hi: float,
                      dedup: bool, cols: dict):
    """Append book rows of token idx of market m to columnar dict `cols`.

    captured_at = bucket time t + resolution when `shift` (the API returns the LAST book in each bucket, so
    the row holds the book as of the bucket's end, the recorder's causal convention); rows outside
    [lo, hi) after the shift are dropped. `dedup` drops a row whose ladders equal the previous row's of
    the same token (the recorder writes a row only when the book changed).
    """
    g = m["g"]
    win, _ = resolution(g)
    winner = None if win is None else ("yes" if win == 0 else "no")
    is_win = None if win is None else (win == idx)
    end_us = int(round((parse_ts(g.get("endDate")) or m["kickoff"]) * 1e6))
    sh = res_s if shift else 0.0
    prev = None
    for pt in sorted(pts, key=lambda z: z["t"]):
        t = parse_ts(pt["t"])
        if t is None:
            continue
        cap = pt.get("_cap", t + sh)
        if not (lo <= cap < hi):
            continue
        bp, bs = _ladder(pt.get("bids"), True)
        ap, as_ = _ladder(pt.get("asks"), False)
        sig = (tuple(bp), tuple(bs), tuple(ap), tuple(as_))
        if dedup and sig == prev:
            continue
        prev = sig
        cap_us = int(round(cap * 1e6))
        bb = bp[0] if bp else None
        ba = ap[0] if ap else None
        cols["market_slug"].append(m["slug"])
        cols["coin"].append(m["league"])
        cols["contract_length"].append(m["smt"])
        cols["token_id"].append(m["tokens"][idx])
        cols["outcome"].append(m["outcomes"][idx])
        cols["winner"].append(winner)
        cols["is_winning_outcome"].append(is_win)
        cols["market_end_at"].append(end_us)
        cols["captured_at"].append(cap_us)
        cols["seconds_to_close"].append((end_us - cap_us) / 1e6)
        cols["bid_prices"].append(bp)
        cols["bid_sizes"].append(bs)
        cols["ask_prices"].append(ap)
        cols["ask_sizes"].append(as_)
        cols["best_bid"].append(bb)
        cols["best_ask"].append(ba)
        cols["crossed"].append(bool(bb is not None and ba is not None and bb >= ba))


def columns_to_table(cols: dict, schema: pa.Schema) -> pa.Table:
    arrs = {}
    for f in BOOK_SCHEMA_RAW:
        v = cols[f.name]
        if f.type == UTC_US:
            arrs[f.name] = pa.array(np.asarray(v, np.int64), pa.int64()).cast(UTC_US)
        else:
            arrs[f.name] = pa.array(v, f.type)
    if "reconciled" in schema.names:
        arrs["reconciled"] = pa.array([False] * len(cols["market_slug"]), pa.bool_())
    return pa.table(arrs, schema=schema)


def new_cols():
    return {f.name: [] for f in BOOK_SCHEMA_RAW}


# ------------------------------------------------------------------------------------- books stage
def chunks_for(m: dict, args) -> list[tuple[int, int, str]]:
    """[(a, b, resolution)], hour-aligned chunks of the market window; the last --fine-hours before
    kickoff (and the post-kickoff part) at --resolution, earlier at --coarse-resolution."""
    ws, we = m["ws"], m["we"]
    fine_from = max(ws, m["kickoff"] - int(args.fine_hours * 3600))
    segs = []
    if fine_from > ws:
        segs.append((ws, fine_from, args.coarse_resolution))
    segs.append((fine_from, we, args.resolution))
    out = []
    for a, b, res in segs:
        span = 3600 * max(1, int(BOOK_PAGE * res_seconds(res) // 3600))   # ~1 page per token per chunk at 1m
        t = a
        while t < b:
            nxt = min(b, (t // span + 1) * span)
            out.append((t, nxt, res))
            t = nxt
    return out


def book_state_dir(out: Path, slug: str) -> Path:
    return out / "_state" / "books" / slug


def fetch_chunk(pob: POB, m: dict, a: int, b: int, res: str, args, schema) -> dict:
    """Fetch one chunk (all tokens), write its book rows as hourly part files, return a state record."""
    res_s = res_seconds(res)
    cols = new_cols()
    pages, npts, labels = 0, 0, []
    idxs = [0] if args.mirror else [0, 1]
    for i in idxs:
        pts, pg, meta = pob.token_books(m["tokens"][i], a, b, res)
        pages += pg
        npts += len(pts)
        labels.append(meta.get("token_label"))
        if a == m["ws"] and not args.no_seed:
            # last book before the window (sparse feeds have no row until the first change)
            sp, spg, _ = pob.token_books(m["tokens"][i], a - args.seed_hours * 3600, a, "1h")
            pages += spg
            if sp:     # the last book before a == the book as of a: stamped at a
                last = dict(max(sp, key=lambda z: z["t"]))
                last["_cap"] = float(a)
                pts = [last] + pts
        lo, hi = float(a), float(b)
        points_to_columns(m, i, pts, res_s, not args.no_shift, lo, hi, not args.keep_unchanged, cols)
        if args.mirror:
            points_to_columns(m, 1, [mirror_point(p) | ({"_cap": p["_cap"]} if "_cap" in p else {}) for p in pts],
                              res_s, not args.no_shift, lo, hi, not args.keep_unchanged, cols)
    tab = columns_to_table(cols, schema)
    run_id = f"{m['slug']}-c{a}"
    sink = PartSink(Path(args.out), "books", schema, run_id, level=3, sort_by=BOOK_SORT, dictionary_cols=BOOK_DICT)
    if tab.num_rows:
        cap = tab.column("captured_at").cast(pa.int64()).to_numpy() // 1_000_000
        hours = np.array([hour_key(int(x)) for x in cap])
        for h in sorted(set(hours)):
            sink.write_part(h, tab.filter(pa.array(hours == h)))
    return {"status": "ok" if tab.num_rows else "empty", "rows": tab.num_rows, "points": npts,
            "pages": pages, "res": res, "a": a, "b": b, "labels": labels, "at": int(time.time())}


def run_books(args, markets: list[dict], pob: POB, schema) -> dict:
    out = Path(args.out)
    hist_floor = getattr(args, "_history_floor", None)
    tasks, skipped_done, skipped_missing, skipped_hist = [], 0, 0, 0
    for m in markets:
        sd = book_state_dir(out, m["slug"])
        if (sd / "MISSING.json").exists() and not args.retry_missing:
            skipped_missing += 1
            continue
        for a, b, res in chunks_for(m, args):
            sp = sd / f"{a}_{res}.json"
            if sp.exists() and not (args.retry_forbidden and json.loads(sp.read_text()).get("status") == "forbidden"):
                skipped_done += 1
                continue
            if hist_floor is not None and a < hist_floor:
                skipped_hist += 1
                continue
            tasks.append((m, a, b, res))
    log.info("books: %d chunks to fetch (%d already done, %d markets marked missing, %d chunks older than "
             "the plan's history window)", len(tasks), skipped_done, skipped_missing, skipped_hist)
    stats = {"ok": 0, "empty": 0, "forbidden": 0, "missing": 0, "failed": 0, "rows": 0, "pages": 0}
    missing = set()
    stop = threading.Event()
    lock = threading.Lock()

    def work(t):
        m, a, b, res = t
        if stop.is_set() or m["slug"] in missing:
            return None
        sd = book_state_dir(out, m["slug"])
        try:
            st = fetch_chunk(pob, m, a, b, res, args, schema)
        except ApiError as e:
            if e.status == 404:
                with lock:
                    missing.add(m["slug"])
                atomic_json(sd / "MISSING.json", {"error": e.message, "at": int(time.time())})
                return ("missing", m, a, e)
            if e.status == 403:
                atomic_json(sd / f"{a}_{res}.json", {"status": "forbidden", "error": e.message, "a": a, "b": b,
                                                      "res": res, "at": int(time.time())})
                return ("forbidden", m, a, e)
            if e.status == 401:
                stop.set()
                return ("auth", m, a, e)
            return ("failed", m, a, e)
        except BudgetExhausted as e:
            stop.set()
            return ("budget", m, a, e)
        atomic_json(sd / f"{a}_{res}.json", st)
        return (st["status"], m, a, st)

    t0, nf = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is None:
                continue
            kind, m, a, x = r
            if kind in ("ok", "empty"):
                stats[kind] += 1
                stats["rows"] += x["rows"]
                stats["pages"] += x["pages"]
            elif kind in ("forbidden", "missing", "failed"):
                stats[kind] += 1
                log.warning("%s %s chunk %s: %s", kind, m["slug"], iso(a), x)
                if kind == "forbidden":
                    nf += 1
                    if nf >= 20 and stats["ok"] + stats["empty"] == 0:
                        log.error("20 chunks forbidden and none fetched: plan does not allow this data "
                                  "(granularity/history?). Stopping.")
                        stop.set()
            elif kind == "auth":
                log.error("API key rejected (401): %s. Stopping.", x.message)
            elif kind == "budget":
                log.error("stopping: %s", x)
            if i % 50 == 0 or i == len(futs):
                log.info("books progress %d/%d chunks, %d rows, %d requests, %.0fs", i, len(futs), stats["rows"],
                         pob.n, time.time() - t0)
    stats["requests"] = pob.n
    return stats


def compact(out: Path, names=("books", "trades", "markets")):
    for name in names:
        base = out / name
        if not base.exists():
            continue
        schema = {"trades": TRADES_SCHEMA, "markets": MARKETS_SCHEMA_SPORTS}.get(name)
        if name == "books":
            any_part = next(iter(base.glob("*/_parts/*.parquet")), None)
            if any_part is None:
                continue
            schema = pq.read_schema(any_part).remove_metadata()
        sink = PartSink(out, name, schema, "compact", level=9,
                        sort_by=BOOK_SORT if name == "books" else (["slug"] if name == "markets" else None),
                        dictionary_cols=BOOK_DICT if name == "books" else (["slug"] if name == "trades" else None))
        hours = sink.part_hours()
        for h in hours:
            sink.compact(h)
        if hours:
            log.info("compacted %s: %d hour files", name, len(hours))


# --------------------------------------------------------------------------- trades + markets stage
def dapi_window(cond: str, taker_only: bool, a: int, b: int) -> tuple[list[list], bool]:
    """data-api /trades for [a, b] using its start/end filters; if a window holds more legs than one query
    can page through (offset <= 10000), it is split in halves. Returns (list of per-window row lists,
    truncated). Windows overlap at the split second; merge each list separately into a LegSet (max count
    per key), so the overlap never double counts."""
    rows, off = [], 0
    while True:
        url = (f"{ft.DATA}/trades?market={cond}&limit={ft.PAGE}&offset={off}"
               f"&takerOnly={'true' if taker_only else 'false'}&start={a}&end={b}")
        try:
            pg = ft.http_json(url)
        except urllib.error.HTTPError as e:              # 400 offset cap
            if e.code != 400:
                raise
            pg = None
        if pg is None:
            break
        rows += pg
        if len(pg) < ft.PAGE:
            return [rows], False
        if off + ft.PAGE > ft.MAX_OFFSET:
            break
        off += ft.PAGE
    if b - a < 2:
        return [rows], True
    mid = (a + b) // 2
    l, t1 = dapi_window(cond, taker_only, a, mid)
    r, t2 = dapi_window(cond, taker_only, mid, b)
    return l + r, t1 or t2


def fetch_meta(m: dict, args) -> tuple[pa.Table, dict, dict]:
    a = m["ws"] if not args.trades_full else 0
    b = m["we"]
    taker, allf = LegSet(), LegSet()
    trunc = False
    for only, ls in ((True, taker), (False, allf)):
        wins, t = dapi_window(m["cond"], only, a, b)
        trunc |= t
        for w in wins:
            ls.merge([r for r in w if a <= int(r["timestamp"]) <= b])
    rows = build_trade_rows(m["slug"], taker, allf)
    cons = rows_consistency(rows)
    tab = trade_rows_to_table(rows)
    g = dict(m["g"])
    row = market_row(m["slug"], m["league"], m["smt"], g, len(taker), len(allf), trunc)
    utc = lambda s: dt.datetime.fromtimestamp(s, dt.timezone.utc)  # noqa: E731
    row.update(kickoff=utc(m["kickoff"]), league=m["league"], sports_market_type=m["smt"],
               event_slug=g.get("eventSlug"), outcome_0=str(m["outcomes"][0]), outcome_1=str(m["outcomes"][1]),
               volume=m["volume"], window_start=utc(a), window_end=utc(b), trades_consistent=bool(cons["ok"]))
    return tab, row, cons


def run_meta(args, markets: list[dict]) -> dict:
    out = Path(args.out)
    todo = [m for m in markets if not (out / "_state" / "meta" / f"{m['slug']}.json").exists()]
    log.info("meta: %d markets to fetch trades/metadata for (%d done)", len(todo), len(markets) - len(todo))
    stats = {"ok": 0, "failed": 0, "fills": 0, "truncated": 0, "inconsistent": 0}

    def work(m):
        tab, row, cons = fetch_meta(m, args)
        h = hour_key(m["kickoff"])
        PartSink(out, "trades", TRADES_SCHEMA, m["slug"], level=3, dictionary_cols=["slug"]).write_part(h, tab)
        PartSink(out, "markets", MARKETS_SCHEMA_SPORTS, m["slug"], level=3).write_part(
            h, pa.Table.from_pylist([row], schema=MARKETS_SCHEMA_SPORTS))
        st = {"fills": tab.num_rows, "n_taker": row["n_taker"], "truncated": row["truncated"], "consistency": cons,
              "at": int(time.time())}
        atomic_json(out / "_state" / "meta" / f"{m['slug']}.json", st)
        return st

    with ThreadPoolExecutor(args.meta_workers) as ex:
        futs = {ex.submit(work, m): m for m in todo}
        for i, f in enumerate(as_completed(futs), 1):
            m = futs[f]
            try:
                st = f.result()
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                log.warning("meta failed %s: %s", m["slug"], e)
                continue
            stats["ok"] += 1
            stats["fills"] += st["fills"]
            stats["truncated"] += bool(st["truncated"])
            stats["inconsistent"] += not st["consistency"]["ok"]
            if i % 25 == 0 or i == len(futs):
                log.info("meta progress %d/%d, %d fills", i, len(futs), stats["fills"])
    return stats


# -------------------------------------------------------------------------------- dry-run / probe
def estimate(markets: list[dict], args) -> dict:
    ntok = 1 if args.mirror else 2
    pts = pages = 0
    for m in markets:
        for a, b, res in chunks_for(m, args):
            n = math.ceil((b - a) / res_seconds(res))
            pts += n * ntok
            pages += ntok * max(1, math.ceil(n / BOOK_PAGE))
        if not args.no_seed:
            pages += ntok
    return {"markets": len(markets), "chunks": sum(len(chunks_for(m, args)) for m in markets),
            "points_upper": pts, "requests_upper": pages, "bytes_upper": pts * (2 if args.mirror else 1) * BYTES_PER_ROW}


def print_selection(markets: list[dict], args):
    import pandas as pd
    if not markets:
        print("no markets selected")
        return
    df = pd.DataFrame([{k: m[k] for k in ("slug", "league", "smt", "kickoff", "volume")} for m in markets])
    df["day"] = pd.to_datetime(df.kickoff, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    pd.set_option("display.width", 200)
    print(f"\n## selection: {len(df)} markets, kickoff days {df.day.min()} .. {df.day.max()}, "
          f"window [kickoff - {args.hours_before} h, kickoff + {args.post_min} min)")
    print(df.groupby(["league", "smt"], sort=False).agg(markets=("slug", "size"), vol_M=("volume", lambda v: v.sum() / 1e6))
          .round(2).to_string())
    print("\n## markets per kickoff day")
    print(df.groupby("day").size().to_string())
    e = estimate(markets, args)
    rpd = {"starter (60/min, 1k/day, 3 d history)": 1000, "pro (300/min, 50k/day, 30 d history)": 50000}
    print(f"\n## PolyOrderBooks books estimate ({'mirror: outcome-0 token only' if args.mirror else 'both tokens'}; "
          f"{args.resolution} for the last {args.fine_hours} h, {args.coarse_resolution} before)")
    print(f"chunks {e['chunks']}, requests <= {e['requests_upper']:,} (dense upper bound: one point per bucket; "
          f"sparse feeds need fewer), book rows <= {e['points_upper'] * (2 if args.mirror else 1):,}, "
          f"parquet <= {e['bytes_upper'] / 1e9:.2f} GB")
    for k, v in rpd.items():
        print(f"  {k}: {e['requests_upper'] / v:.1f} days of quota")
    print("alternatives (requests upper bound):")
    for fh in sorted({0.0, 2.0, 6.0, float(args.hours_before)}):
        for mir in (False, True):
            alt = argparse.Namespace(**{**vars(args), "fine_hours": min(fh, args.hours_before), "mirror": mir})
            print(f"  --fine-hours {fh:g}{' --mirror' if mir else ''}: {estimate(markets, alt)['requests_upper']:,}")
    vq = df.volume.quantile([0.25, 0.5, 0.75, 0.9]).round(0).to_dict()
    print(f"market volume quantiles (gamma volumeNum): {vq}; --min-volume trims the long tail")
    print(f"trades/markets (keyless data-api + gamma): ~{2 * len(df)}-{4 * len(df)} data-api requests")
    out = Path(args.out)
    done = sum(1 for m in markets for a, b, r in chunks_for(m, args) if (book_state_dir(out, m["slug"]) / f"{a}_{r}.json").exists())
    print(f"already done: {done} book chunks, "
          f"{sum(1 for m in markets if (out / '_state' / 'meta' / (m['slug'] + '.json')).exists())} meta markets")
    print("\nfirst 15 in fetch order:")
    print(df.head(15).assign(kickoff=lambda d: pd.to_datetime(d.kickoff, unit="s", utc=True)).to_string(index=False))


def probe(args, pob: POB, markets: list[dict]) -> dict:
    rep: dict = {"at": iso(time.time())}
    u = pob.usage()
    rep["usage"] = u
    lim = u.get("limits") or {}
    print(f"plan={u.get('plan')} org={u.get('organization')} rpm={lim.get('requests_per_minute')} "
          f"remaining={lim.get('requests_remaining')} granularity={lim.get('granularity_allowed')} "
          f"history_days={lim.get('max_history_days')} data_points={lim.get('data_points')} reset_at={u.get('reset_at')}")
    try:
        tags = pob.get("/v1/tags").get("data") or []
        rep["sports_tags"] = tags
        print(f"sports tags: {len(tags)}; ours present: {[l for l in LEAGUES if l in tags]}")
    except ApiError as e:
        print("tags:", e)
    # sample: per league, first / middle / last kickoff day of the selection
    by = {}
    for m in markets:
        by.setdefault(m["league"], []).append(m)
    sample = []
    for lg, ms in by.items():
        ms = sorted(ms, key=lambda z: z["kickoff"])
        for j in sorted({0, len(ms) // 2, len(ms) - 1})[: args.probe_n]:
            sample.append(ms[j])
    rows = []
    first = True
    for m in sample:
        r = {"slug": m["slug"], "league": m["league"], "kickoff": iso(m["kickoff"])}
        try:
            if first:
                d = pob.get(f"/v1/markets/{m['slug']}")
                r["market_id"], r["gamma_id"] = d.get("id"), m["g"].get("id")
                r["pob_tokens"] = [t.get("id") for t in d.get("tokens") or []]
                r["tokens_match"] = sorted(r["pob_tokens"]) == sorted(m["tokens"])
                first = False
            pts, pg, meta = pob.token_books(m["tokens"][0], m["ws"], m["we"], "1h")
            r["hourly_points"] = len(pts)
            r["label"] = meta.get("token_label")
            r["first_t"] = pts[0]["t"] if pts else None
            r["last_t"] = pts[-1]["t"] if pts else None
            d = pob.get(f"/v1/tokens/{m['tokens'][0]}/books",
                        {"start_ts": m["kickoff"] - 600, "end_ts": m["kickoff"], "resolution": "1s", "limit": BOOK_PAGE})
            p1 = d.get("data") or []
            r["points_1s_last10min_page1"] = len(p1)
            r["next_cursor"] = bool((d.get("metadata") or {}).get("next_cursor"))
            if p1:
                ts = [parse_ts(x["t"]) for x in p1]
                r["t_frac_nonzero"] = any(t % 1 for t in ts)
                r["min_gap_s"] = float(np.min(np.diff(ts))) if len(ts) > 1 else None
                r["levels_med"] = float(np.median([len(x.get("bids") or []) + len(x.get("asks") or []) for x in p1]))
                r["example_point"] = {k: (v[:3] if isinstance(v, list) else v) for k, v in p1[0].items()}
                if True:      # is outcome 1 the mirror of outcome 0? (decides whether --mirror is safe)
                    d1 = pob.get(f"/v1/tokens/{m['tokens'][1]}/books",
                                 {"start_ts": m["kickoff"] - 600, "end_ts": m["kickoff"], "resolution": "1s", "limit": BOOK_PAGE})
                    q = {x["t"]: x for x in d1.get("data") or []}
                    top3 = lambda pt: [[round(p, 6) for p in _ladder(pt.get(sd), sd == "bids")[0][:3]]  # noqa: E731
                                       for sd in ("bids", "asks")]
                    same = [top3(mirror_point(x)) == top3(q[x["t"]]) for x in p1 if x["t"] in q]
                    r["mirror_match_top3"] = f"{sum(same)}/{len(same)}"
        except ApiError as e:
            r["error"] = str(e)
        rows.append(r)
        print(json.dumps(r, default=str))
    rep["sample"] = rows
    dense = [r["points_1s_last10min_page1"] for r in rows if "points_1s_last10min_page1" in r]
    if dense:
        print(f"1s points in the first page of the last 10 min before kickoff: median {np.median(dense):.0f} "
              f"(page max {BOOK_PAGE}; ~{BOOK_PAGE} with next_cursor => dense or busy, << 600 => sparse on-change)")
    rep["estimate"] = estimate(markets, args)
    return rep


# ------------------------------------------------------------------------------------- verification
def verify(out: Path) -> bool:
    ok = True
    raw = P / "data" / "raw"
    ref = {"books": pq.read_schema(raw / "updown_5m.parquet").remove_metadata(),
           "trades": pq.read_schema(raw / "trades.parquet").remove_metadata(),
           "markets": pq.read_schema(raw / "markets.parquet").remove_metadata()}
    for name, rs in ref.items():
        files = dataset_files(out, name)
        if not files:
            print(f"{name}: no files")
            continue
        n = 0
        for f in files:
            s = pq.read_schema(f).remove_metadata()
            if name == "books" and s.names[-1] == "reconciled":
                s = pa.schema(list(s)[:-1])
            if name == "markets":
                good = pa.schema(list(s)[: len(rs)]).equals(rs) and pa.schema(list(s)[len(rs):]).equals(
                    pa.schema([pa.field(a, b) for a, b in MARKETS_EXTRA]))
            else:
                good = s.equals(rs)
            if not good:
                ok = False
                print(f"SCHEMA MISMATCH {f}\n  got {s}\n  want {rs}")
            n += pq.read_metadata(f).num_rows
        print(f"{name}: {len(files)} files, {n} rows, schema {'== data/raw' if ok else 'MISMATCH'}")
    import pandas as pd
    fb = dataset_files(out, "books")
    if fb:
        df = pd.concat([pd.read_parquet(f) for f in fb[:50]])
        mono_b = df.bid_prices.apply(lambda r: all(r[i] >= r[i + 1] for i in range(len(r) - 1))).all()
        mono_a = df.ask_prices.apply(lambda r: all(r[i] <= r[i + 1] for i in range(len(r) - 1))).all()
        bb = df.bid_prices.apply(lambda r: r[0] if len(r) else np.nan)
        cr = ((df.best_bid >= df.best_ask) & df.best_bid.notna() & df.best_ask.notna())
        stc = (df.market_end_at - df.captured_at).dt.total_seconds()
        checks = {"bids_desc": bool(mono_b), "asks_asc": bool(mono_a),
                  "best_bid==bid_prices[0]": bool(np.allclose(bb.fillna(-1), df.best_bid.fillna(-1))),
                  "crossed derived": bool((cr == df.crossed).all()),
                  "seconds_to_close": bool(np.allclose(stc, df.seconds_to_close)),
                  "whole-second captured_at": bool((df.captured_at.astype("int64") % 1_000_000 == 0).all())}
        print("books invariants:", checks)
        ok &= all(v for k, v in checks.items() if k != "whole-second captured_at")
    return ok


# --------------------------------------------------------------------------------------- self-test
def self_test() -> bool:
    """Offline: synthetic market + mocked PolyOrderBooks responses shaped like the OpenAPI examples
    (BooksResponseSingleToken, metadata.next_cursor pagination), run through the real books stage, then
    compacted and checked against data/raw's schema."""
    tmp = Path(tempfile.mkdtemp(prefix="pob_selftest_", dir=os.environ.get("TMPDIR")))
    ko = 1791764400                                       # 2026-10-12T00:20Z
    toks = ["111", "222"]
    g = {"id": "4024681", "slug": "nfl-bal-atl-2026-10-12", "endDate": iso(ko), "gameStartTime": iso(ko),
         "outcomes": '["Ravens", "Falcons"]', "outcomePrices": '["1", "0"]', "clobTokenIds": json.dumps(toks),
         "closed": True, "conditionId": "0x" + "ab" * 32, "sportsMarketType": "moneyline"}
    m = dict(slug=g["slug"], league="nfl", smt="moneyline", kickoff=ko, ws=ko - 3 * 3600, we=ko, tokens=toks,
             outcomes=["Ravens", "Falcons"], cond=g["conditionId"], volume=1.0, g=g)

    class MockPOB(POB):
        def __init__(self):
            super().__init__("pob_test_key", rpm=1e9)
            self.calls = []

        def get(self, path, params=None):
            self.calls.append((path, dict(params or {})))
            self.n += 1
            if path == "/v1/usage":                       # OpenAPI UsageResponse example
                return {"plan": "starter", "organization": "Acme Research", "reset_at": 1754140860,
                        "limits": {"requests_per_minute": 60, "requests_remaining": 58, "granularity_allowed": "250ms",
                                   "max_history_days": 3, "data_points": {"books": True, "metrics": True, "prices": True}}}
            if path == "/v1/tags":
                return {"count": 2, "data": ["nfl", "sports"]}
            if path.startswith("/v1/markets/"):
                return {"id": "4024681", "slug": path.split("/")[3], "category": "sports",
                        "tokens": [{"id": toks[0], "label": "Ravens"}, {"id": toks[1], "label": "Falcons"}]}
            tid = path.split("/")[3]
            if tid == "999":
                raise ApiError(404, "not_found", "Token not found: 999")
            a, b, res = int(params["start_ts"]), int(params["end_ts"]), params["resolution"]
            step = RES_S[res]
            ts = [t for t in np.arange(a, b, step)]
            if res == "1s":
                ts = [t for t in ts if int(t) % 7 in (0, 3)]      # sparse: on-change buckets only
            off = int(params.get("cursor") or 0)
            lim = int(params.get("limit") or 100)
            page = ts[off: off + lim]
            data = []
            for t in page:
                x = (int(t) // 7) % 20 / 1000
                bids = [[round(0.40 + x - 0.01 * k, 4), 100.0 + k] for k in range(5)]
                asks = [[round(0.41 + x + 0.01 * k, 4), 120.0 + k] for k in range(5)]
                if int(t) % 49 == 0:
                    asks = []                                     # one-sided book
                if tid == toks[1]:                                # same CLOB book seen from outcome 1
                    bids, asks = [[round(1 - p, 4), s] for p, s in asks], [[round(1 - p, 4), s] for p, s in bids]
                data.append({"t": dt.datetime.fromtimestamp(float(t), dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                             "bids": bids[::-1] if int(t) % 2 else bids, "asks": asks})
            nxt = str(off + lim) if off + lim < len(ts) else None
            return {"market_id": "4024681", "token_id": tid, "token_label": "Ravens" if tid == toks[0] else "Falcons",
                    "resolution": res, "data": data, "metadata": {"count": len(data), "limit": lim, "next_cursor": nxt}}

    ns = argparse.Namespace(out=str(tmp), hours_before=3, post_min=0, resolution="1s", fine_hours=1,
                            coarse_resolution="1m", mirror=False, no_seed=False, no_shift=False, seed_hours=6,
                            keep_unchanged=False, retry_missing=False, retry_forbidden=False, workers=2, with_reconciled=False)
    ok = True
    try:
        pob = MockPOB()
        st = run_books(ns, [m], pob, BOOK_SCHEMA_RAW)
        print("books stage:", st)
        st2 = run_books(ns, [m], pob, BOOK_SCHEMA_RAW)            # resume: nothing left to do
        ok &= st2["ok"] + st2["empty"] == 0
        m2 = dict(m, slug="nfl-missing-2026-10-12", tokens=["999", "998"])
        st3 = run_books(ns, [m2], pob, BOOK_SCHEMA_RAW)
        ok &= st3["missing"] >= 1 and (tmp / "_state/books/nfl-missing-2026-10-12/MISSING.json").exists()
        ns_m = argparse.Namespace(**{**vars(ns), "mirror": True, "out": str(tmp / "mirror")})
        stm = run_books(ns_m, [m], MockPOB(), BOOK_SCHEMA_RAW)
        compact(tmp, ("books",))
        compact(tmp / "mirror", ("books",))
        ok &= verify(tmp)
        import pandas as pd
        df = pd.concat(pd.read_parquet(f) for f in dataset_files(tmp, "books"))
        dm = pd.concat(pd.read_parquet(f) for f in dataset_files(tmp / "mirror", "books"))
        assert set(df.token_id) == set(toks) and set(df.outcome) == {"Ravens", "Falcons"}
        assert (df[df.token_id == toks[0]].is_winning_outcome).all() and (df.winner == "yes").all()
        assert df.captured_at.min() >= pd.Timestamp(m["ws"], unit="s", tz="UTC")
        assert df.captured_at.max() < pd.Timestamp(m["we"], unit="s", tz="UTC") + pd.Timedelta(seconds=1)
        fine = df[df.seconds_to_close <= 3600]
        assert fine.groupby("token_id").captured_at.apply(lambda s: s.diff().dt.total_seconds().min()).min() >= 1
        a = df.drop(columns=[]).sort_values(BOOK_SORT).reset_index(drop=True)
        b = dm.sort_values(BOOK_SORT).reset_index(drop=True)
        assert len(a) == len(b) and np.allclose(a.best_bid.fillna(-1), b.best_bid.fillna(-1)), "mirror != fetched"
        seed_paths = [c for c in pob.calls if c[1].get("resolution") == "1h"]
        rep = probe(argparse.Namespace(**{**vars(ns), "probe_n": 2}), MockPOB(), [m])
        assert rep["sample"][0]["tokens_match"] and rep["sample"][0]["mirror_match_top3"].split("/")[0] != "0", rep
        print(f"self-test: {len(df)} rows, {df.market_slug.nunique()} market, mirror rows {len(dm)}, "
              f"one-sided {int(df.best_ask.isna().sum())}, seed calls {len(seed_paths)}, mock requests {pob.n}, "
              f"mirror stage {stm}")
        print("first rows:\n", df.sort_values("captured_at").head(3).to_string())
    except AssertionError as e:
        ok = False
        print("SELF-TEST ASSERTION FAILED:", e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="first kickoff day (UTC, YYYY-MM-DD)")
    ap.add_argument("--end", help="last kickoff day (UTC, inclusive)")
    ap.add_argument("--out", default=str(HERE / "raw_pob"))
    ap.add_argument("--leagues", default=",".join(LEAGUES), help="comma list in priority order, or 'all'")
    ap.add_argument("--market-types", default="moneyline", help="gamma sportsMarketType list (priority order) or 'all'")
    ap.add_argument("--hours-before", type=float, default=168.0,
                    help="H: books for [kickoff - H, kickoff + post); D14 default 7 d (listing to kickoff; the builder cuts episodes at T-1h, the last hour is kept for evaluation only)")
    ap.add_argument("--post-min", type=float, default=0.0, help="minutes after kickoff to include (default 0)")
    ap.add_argument("--resolution", default="1s", help="bucket size for the fine part of the window (D15: 1s for the last --fine-hours)")
    ap.add_argument("--fine-hours", type=float, default=6.0,
                    help="hours before kickoff fetched at --resolution; earlier part of the window at "
                         "--coarse-resolution (set = --hours-before for 1 s everywhere)")
    ap.add_argument("--coarse-resolution", default="1m", help="bucket size before the fine part")
    ap.add_argument("--mirror", action="store_true",
                    help="fetch outcome-0 token only and derive outcome 1 as its mirror (halves requests)")
    ap.add_argument("--no-seed", action="store_true", help="skip the 1h-resolution 'last book before window' request")
    ap.add_argument("--seed-hours", type=int, default=6)
    ap.add_argument("--no-shift", action="store_true", help="captured_at = bucket t (default t + resolution)")
    ap.add_argument("--keep-unchanged", action="store_true", help="keep rows whose book equals the previous row")
    ap.add_argument("--with-reconciled", action="store_true",
                    help="append the recorder's `reconciled` column (False) to book rows")
    ap.add_argument("--min-volume", type=float, default=0.0)
    ap.add_argument("--max-markets", type=int, default=0)
    ap.add_argument("--order", choices=["day", "priority", "time", "volume"], default="day",
                    help="fetch order. day: oldest kickoff day first, then market type / league priority "
                         "(best under the plan's rolling history); priority: type, league, then time")
    ap.add_argument("--stage", choices=["all", "books", "meta"], default="all",
                    help="books = PolyOrderBooks (key), meta = trades + markets (keyless)")
    ap.add_argument("--trades-full", action="store_true", help="all trades of the market up to window end")
    ap.add_argument("--category", default="sports", help="PolyOrderBooks category param ('' to omit)")
    ap.add_argument("--workers", type=int, default=2, help="books threads (rate limit is shared)")
    ap.add_argument("--meta-workers", type=int, default=3)
    ap.add_argument("--rpm", type=float, default=None, help="requests/min (default: plan limit from /v1/usage, x0.9)")
    ap.add_argument("--max-requests", type=int, default=0, help="stop after N PolyOrderBooks requests (0 = no cap)")
    ap.add_argument("--ignore-history-limit", action="store_true")
    ap.add_argument("--retry-missing", action="store_true", help="retry markets PolyOrderBooks returned 404 for")
    ap.add_argument("--retry-forbidden", action="store_true", help="retry chunks that got 403 (e.g. after a plan upgrade)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-n", type=int, default=2, help="sample markets per league in --probe")
    ap.add_argument("--compact", action="store_true", help="only compact part files into hourly files")
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only check outputs in --out against data/raw schemas")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    args.leagues = [s.strip() for s in args.leagues.split(",") if s.strip()]
    args.market_types = [s.strip() for s in args.market_types.split(",") if s.strip()]
    args.fine_hours = min(args.fine_hours, args.hours_before)
    for r in (args.resolution, args.coarse_resolution):
        res_seconds(r)
    key = os.environ.get(ENV_KEY) or None
    out = Path(args.out)
    if args.self_test:
        setup_logging(None, args.log_level, key)
        sys.exit(0 if self_test() else 1)
    if args.verify:
        sys.exit(0 if verify(out) else 1)
    setup_logging(out, args.log_level, key)
    if args.compact:
        compact(out)
        return
    if not (args.start and args.end):
        ap.error("--start and --end are required")
    log.info("args: %s", {k: v for k, v in vars(args).items()})
    t0 = time.time()
    markets = select_markets(args)
    log.info("selected %d markets in %.0fs", len(markets), time.time() - t0)
    if args.dry_run:
        print_selection(markets, args)
        return
    need_key = args.probe or args.stage in ("all", "books")
    pob = None
    if need_key:
        if not key:
            sys.exit(f"{ENV_KEY} is not set (needed for --probe / books; use --stage meta or --dry-run without it)")
        pob = POB(key, rpm=args.rpm or 60, max_requests=args.max_requests, category=args.category or None)
        try:
            u = pob.usage()
        except ApiError as e:
            sys.exit(f"PolyOrderBooks /v1/usage failed: {e}")
        lim = u.get("limits") or {}
        rpm = lim.get("requests_per_minute")
        if args.rpm is None and isinstance(rpm, (int, float)):
            pob.set_rpm(0.9 * rpm)
        elif args.rpm is None and rpm == "unlimited":
            pob.set_rpm(600)
        hd = lim.get("max_history_days")
        if isinstance(hd, (int, float)) and not args.ignore_history_limit:
            args._history_floor = int(time.time() - hd * 86400) + 3600
        log.info("plan=%s rpm=%s history_days=%s granularity=%s books=%s", u.get("plan"), rpm, hd,
                 lim.get("granularity_allowed"), (lim.get("data_points") or {}).get("books"))
        if (lim.get("data_points") or {}).get("books") is False:
            log.error("this plan has no historical order books (data_points.books=false)")
    if args.probe:
        rep = probe(args, pob, markets)
        p = out / f"probe_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"
        atomic_json(p, rep)
        print(f"probe report: {p}  (PolyOrderBooks requests used: {pob.n})")
        return
    schema = BOOK_SCHEMA if args.with_reconciled else BOOK_SCHEMA_RAW
    summary = {}
    if args.stage in ("all", "meta"):
        summary["meta"] = run_meta(args, markets)
    if args.stage in ("all", "books"):
        summary["books"] = run_books(args, markets, pob, schema)
    if not args.no_compact:
        compact(out)
    log.info("done in %.0fs: %s", time.time() - t0, summary)


if __name__ == "__main__":
    main()
