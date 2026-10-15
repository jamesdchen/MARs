#!/usr/bin/env python3
"""token_id -> market map from gamma (cached in sqlite), and market-set selection.

Market sets
  crypto-updown   `{coin}-updown-{5m|15m|4h}-{event_start}` slugs are deterministic (event_start aligned to the
                  contract length, 4h on the 00/04/.. UTC grid), so a time window is resolved by enumerating
                  candidate slugs and asking gamma for them in batches (`/markets?slug=a&slug=b..`, 40 per call).
                  A market is selected for [t0, t1) if its trading lifetime [createdAt, end + 10 min) intersects
                  the window: markets are created ~23.85 h before event start and trade from then on.
  slugs           explicit slugs.
  any token       lookup_tokens(ids) via `/markets?clob_token_ids=..` (40 per call) for any token seen in an
                  unfiltered pull, then filter_slug(df, regex).

Everything gamma returns is cached in cache/gamma.sqlite (positive results forever; "no such market" only for
slots more than 3 days old), so re-running over the same window is free.

CLI
  python markets.py crypto-updown --start 2026-08-21 --end 2026-08-21 [--out m.parquet]
  python markets.py slugs bnb-updown-5m-1787277900
  python markets.py lookup --fills data/all/fills [--out tokens.parquet]
  python markets.py filter --fills data/all/fills --slug-regex '^btc-updown-' --out data/btc/fills
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

import common as C

GAMMA = "https://gamma-api.polymarket.com"
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]
LENS = {"5m": 300, "15m": 900, "4h": 14400}
LEAD_S = 24 * 3600 + 1800     # created ~23.85 h before event start (all 805 markets in data/raw)
POST_S = 600                  # fills settle a few seconds after `end`; generous margin
CRYPTO_RE = r"^(btc|eth|sol|xrp|bnb|doge|hype|zec)-updown-(5m|15m|4h)-\d+$"
BATCH = 40
NEG_AGE_S = 3 * 86400

# same fields data/fetch_trades.py keeps (so to_trades.py can rebuild raw/markets.parquet's schema)
GAMMA_KEYS = ["id", "question", "conditionId", "questionID", "slug", "resolutionSource",
              "startDate", "endDate", "eventStartTime", "closedTime", "outcomes",
              "outcomePrices", "clobTokenIds", "orderPriceMinTickSize", "orderMinSize",
              "makerBaseFee", "takerBaseFee", "feesEnabled", "feeType", "feeSchedule",
              "makerRebatesFeeShareBps", "rewardsMinSize", "rewardsMaxSpread",
              "umaResolutionStatus", "negRisk", "acceptingOrdersTimestamp",
              "cryptoMarketConfig", "volumeNum", "volumeClob", "createdAt"]

TOKEN_COLS = ["token_id", "slug", "outcome", "outcome_idx", "condition_id", "neg_risk", "coin",
              "contract_length", "created_ts", "event_start_ts", "end_ts"]


def _ts(s) -> int | None:
    if not s:
        return None
    try:
        return int(pd.Timestamp(s).timestamp())
    except (ValueError, TypeError):
        return None


def _jl(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def trim(g: dict) -> dict:
    m = {k: g.get(k) for k in GAMMA_KEYS}
    ev = (g.get("events") or [{}])[0]
    m["eventMetadata"] = ev.get("eventMetadata")
    m["eventStartTimeEv"] = ev.get("startTime")
    return m


class Gamma:
    """Polite gamma client + sqlite cache (thread-safe)."""

    def __init__(self, path: Path | None = None, workers: int = 3):
        self.path = path or (C.CACHE / "gamma.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self.workers = workers
        self.calls = 0
        self._tl = threading.local()
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS markets(slug TEXT PRIMARY KEY, condition_id TEXT, created_ts INT,
                event_start_ts INT, end_ts INT, neg_risk INT, outcomes TEXT, tokens TEXT, raw TEXT);
            CREATE TABLE IF NOT EXISTS tokens(token_id TEXT PRIMARY KEY, slug TEXT, outcome TEXT, outcome_idx INT);
            CREATE TABLE IF NOT EXISTS missing(key TEXT PRIMARY KEY, checked INT);
            """)

    # ---------------------------------------------------------------- http
    def _get(self, params: list[tuple[str, str]], tries: int = 7):
        s = getattr(self._tl, "s", None)
        if s is None:
            s = self._tl.s = requests.Session()
            s.headers.update({k: v for k, v in C.HTTP_HEADERS.items() if k != "Content-Type"})
        delay = 1.0
        for k in range(tries):
            try:
                r = s.get(GAMMA + "/markets", params=params, timeout=60)
                self.calls += 1
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (403, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"gamma HTTP {r.status_code}: {r.text[:200]}")
            except (requests.RequestException, ValueError):
                pass
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)
        raise RuntimeError(f"gamma: failed after {tries} tries")

    # ---------------------------------------------------------------- cache
    def _store(self, items: list[dict]):
        rows, toks = [], []
        for g in items:
            m = trim(g)
            outs, tids = _jl(m["outcomes"]), _jl(m["clobTokenIds"])
            cands = [t for t in (_ts(m.get("createdAt")), _ts(m.get("startDate"))) if t]
            created = min(cands) if cands else None
            es = _ts(m.get("eventStartTime") or m.get("eventStartTimeEv"))
            rows.append((m["slug"], m["conditionId"], created, es, _ts(m.get("endDate")),
                         int(bool(m.get("negRisk"))), json.dumps(outs), json.dumps(tids), json.dumps(m)))
            toks += [(str(t), m["slug"], o, i) for i, (o, t) in enumerate(zip(outs, tids))]
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self.db.executemany("INSERT OR REPLACE INTO tokens VALUES (?,?,?,?)", toks)
            self.db.commit()

    def _mark_missing(self, keys: list[str]):
        now = int(time.time())
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO missing VALUES (?,?)", [(k, now) for k in keys])
            self.db.commit()

    def _known(self, table: str, col: str, keys: list[str]) -> set[str]:
        out = set()
        with self.lock:
            for i in range(0, len(keys), 900):
                ch = keys[i:i + 900]
                q = f"SELECT {col} FROM {table} WHERE {col} IN ({','.join('?' * len(ch))})"
                out |= {r[0] for r in self.db.execute(q, ch)}
        return out

    def _fetch(self, param: str, keys: list[str], neg_prefix: str, neg_ok) -> None:
        """Fetch keys (slugs or token ids) not cached yet; closed=true first, then open markets."""
        have = self._known("markets" if param == "slug" else "tokens", "slug" if param == "slug" else "token_id", keys)
        miss = self._known("missing", "key", [neg_prefix + k for k in keys])
        todo = [k for k in keys if k not in have and neg_prefix + k not in miss]
        if not todo:
            return

        def one(batch):
            got, rem = [], list(batch)
            for closed in ("true", "false"):
                q = [(param, k) for k in rem] + [("closed", closed), ("limit", str(2 * len(rem) + 10))]
                got += self._get(q)
                found = {g["slug"] for g in got} if param == "slug" else \
                    {str(t) for g in got for t in _jl(g.get("clobTokenIds"))}
                rem = [k for k in batch if k not in found]
                if not rem:
                    break
            self._store(got)
            self._mark_missing([neg_prefix + k for k in rem if neg_ok(k)])

        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(one, batches))

    # ---------------------------------------------------------------- public
    def markets(self, slugs: list[str]) -> pd.DataFrame:
        now = time.time()

        def neg_ok(s):   # only remember "no such market" for slots well in the past
            m = re.search(r"-(\d{9,})$", s)
            return bool(m) and int(m.group(1)) < now - NEG_AGE_S
        self._fetch("slug", slugs, "slug:", neg_ok)
        rows = []
        with self.lock:
            for i in range(0, len(slugs), 900):
                ch = slugs[i:i + 900]
                rows += self.db.execute(
                    f"SELECT slug, condition_id, created_ts, event_start_ts, end_ts, neg_risk, outcomes, tokens "
                    f"FROM markets WHERE slug IN ({','.join('?' * len(ch))})", ch).fetchall()
        return _token_frame(rows)

    def raw(self, slugs: list[str]) -> dict[str, dict]:
        out = {}
        with self.lock:
            for i in range(0, len(slugs), 900):
                ch = slugs[i:i + 900]
                for s, r in self.db.execute(f"SELECT slug, raw FROM markets WHERE slug IN ({','.join('?' * len(ch))})", ch):
                    out[s] = json.loads(r)
        return out

    def lookup_tokens(self, token_ids) -> pd.DataFrame:
        ids = sorted({str(t) for t in token_ids})
        self._fetch("clob_token_ids", ids, "tok:", lambda k: True)
        slugs = set()
        with self.lock:
            for i in range(0, len(ids), 900):
                ch = ids[i:i + 900]
                slugs |= {r[0] for r in self.db.execute(
                    f"SELECT slug FROM tokens WHERE token_id IN ({','.join('?' * len(ch))})", ch)}
        df = self.markets(sorted(slugs))
        return df[df.token_id.isin(ids)].reset_index(drop=True)


def _token_frame(rows) -> pd.DataFrame:
    out = []
    for slug, cond, cr, es, end, nr, outs, toks in rows:
        m = re.match(CRYPTO_RE, slug)
        for i, (o, t) in enumerate(zip(json.loads(outs), json.loads(toks))):
            out.append((str(t), slug, o, i, cond, bool(nr), m.group(1) if m else None, m.group(2) if m else None,
                        cr, es, end))
    df = pd.DataFrame(out, columns=TOKEN_COLS)
    for c in ("created_ts", "event_start_ts", "end_ts"):
        df[c] = df[c].astype("Int64")
    return df


def crypto_updown_slugs(t0: int, t1: int, coins=None, lens=None) -> list[str]:
    out = []
    for coin in coins or COINS:
        for name in lens or LENS:
            L = LENS[name]
            s = (t0 - L - POST_S) // L * L
            while s < t1 + LEAD_S:
                out.append(f"{coin}-updown-{name}-{s}")
                s += L
    return out


def crypto_updown(t0: int, t1: int, coins=None, lens=None, max_event_start: int | None = None,
                  gamma: Gamma | None = None) -> pd.DataFrame:
    """Token map of crypto up/down markets whose trading lifetime intersects [t0, t1)."""
    g = gamma or Gamma()
    df = g.markets(crypto_updown_slugs(t0, t1, coins, lens))
    live = (df.created_ts.fillna(0) < t1) & (df.end_ts.fillna(2**40) + POST_S > t0)
    if max_event_start is not None:
        live &= df.event_start_ts.fillna(2**40) < max_event_start
    return df[live].sort_values(["event_start_ts", "slug", "outcome_idx"]).reset_index(drop=True)


def filter_slug(df: pd.DataFrame, regex: str) -> pd.DataFrame:
    return df[df.slug.str.match(regex)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("crypto-updown")
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True, help="inclusive date, or exclusive datetime")
    a.add_argument("--coins", default=",".join(COINS))
    a.add_argument("--lens", default=",".join(LENS))
    a.add_argument("--max-event-start", help="drop markets whose event starts at/after this date (e.g. val/test)")
    a.add_argument("--out")
    b = sub.add_parser("slugs")
    b.add_argument("slugs", nargs="+")
    c = sub.add_parser("lookup")
    c.add_argument("--fills", required=True)
    c.add_argument("--out")
    d = sub.add_parser("filter")
    d.add_argument("--fills", required=True)
    d.add_argument("--slug-regex", default=CRYPTO_RE)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    g = Gamma()
    t = time.time()
    if args.cmd == "crypto-updown":
        t0 = C.utc_ts(args.start)
        t1 = C.utc_ts(args.end) + (86400 if len(args.end) == 10 else 0)
        df = crypto_updown(t0, t1, args.coins.split(","), args.lens.split(","),
                           C.utc_ts(args.max_event_start) if args.max_event_start else None, g)
        print(df.groupby("contract_length").slug.nunique().to_string())
        print(f"{df.slug.nunique()} markets, {len(df)} tokens, {g.calls} gamma calls, {time.time()-t:.1f}s")
    elif args.cmd == "slugs":
        df = g.markets(args.slugs)
        print(df.to_string())
    elif args.cmd == "lookup":
        tok = pd.read_parquet(args.fills, columns=["token_id"]).token_id.astype(str).unique()
        df = g.lookup_tokens(tok)
        pat = df.slug.str.replace(r"\d+", "N", regex=True).value_counts()
        print(f"{len(tok)} tokens in fills, {len(df)} mapped, {g.calls} gamma calls\n{pat.head(20).to_string()}")
    else:
        f = pd.read_parquet(args.fills)
        tok = g.lookup_tokens(f.token_id.astype(str).unique())
        keep = filter_slug(tok, args.slug_regex)
        f = f[f.token_id.astype(str).isin(set(keep.token_id))]
        Path(args.out).mkdir(parents=True, exist_ok=True)
        f.to_parquet(Path(args.out) / "part-0.parquet", index=False)
        print(f"kept {len(f)} fills of {keep.slug.nunique()} markets -> {args.out}")
        df = keep
    if getattr(args, "out", None) and args.cmd in ("crypto-updown", "lookup"):
        df.to_parquet(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
