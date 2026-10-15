"""Market discovery through gamma GET /events/keyset (cursor pagination) + client-side slug filters.

Default source: tag `up-or-down` with an event-start-time window, which returns every crypto
up/down market (all coins; 5m / 15m / 4h) that is running or opens within `lookahead_s`. Up/down
markets are listed ~24 h before they open, so a 60 s discovery period is ample; new ones open every
5 minutes and are picked up on the next cycle.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import orjson

from .config import GAMMA, Source

log = logging.getLogger("polyrec.discovery")
LEN_S = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
GAMMA_KEYS = ["id", "question", "conditionId", "questionID", "slug", "resolutionSource",
              "startDate", "endDate", "eventStartTime", "closedTime", "outcomes",
              "outcomePrices", "clobTokenIds", "orderPriceMinTickSize", "orderMinSize",
              "makerBaseFee", "takerBaseFee", "feesEnabled", "feeType", "feeSchedule",
              "makerRebatesFeeShareBps", "rewardsMinSize", "rewardsMaxSpread",
              "umaResolutionStatus", "negRisk", "acceptingOrdersTimestamp",
              "cryptoMarketConfig", "volumeNum", "volumeClob"]


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_ts(s) -> float | None:
    """ISO-8601 (gamma) -> unix seconds."""
    if not s:
        return None
    import datetime as dt
    s = str(s).replace("Z", "+00:00")
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def jl(x):
    return orjson.loads(x) if isinstance(x, str) else x


@dataclass
class MarketInfo:
    slug: str
    condition_id: str
    tokens: list            # [token for outcome 0, token for outcome 1]
    outcomes: list          # e.g. ["Up", "Down"]
    end: int                # unix s
    open: int | None        # eventStartTime (unix s) when the market has one
    coin: str | None
    contract_length: str | None
    source: str
    gamma: dict = field(default_factory=dict)       # trimmed gamma market (GAMMA_KEYS)
    event_meta: dict | None = None                  # event.eventMetadata (priceToBeat, finalPrice)
    event_start_ev: str | None = None

    @property
    def end_us(self) -> int:
        return self.end * 1_000_000


def market_from_gamma(m: dict, ev: dict | None, src: Source, rx: re.Pattern) -> MarketInfo | None:
    slug = m.get("slug") or ""
    mm = rx.match(slug)
    if not mm:
        return None
    toks = jl(m.get("clobTokenIds") or "[]")
    outs = jl(m.get("outcomes") or "[]")
    if len(toks) != 2 or len(outs) != 2:
        return None
    end = parse_ts(m.get("endDate"))
    if end is None:
        return None
    est = parse_ts(m.get("eventStartTime") or (ev or {}).get("startTime"))
    g = mm.groupdict()
    clen = g.get("len")
    if clen is None and est is not None:
        dur = int(round(end - est))
        clen = next((k for k, v in LEN_S.items() if v == dur), None)
    return MarketInfo(
        slug=slug, condition_id=m.get("conditionId"), tokens=[str(t) for t in toks], outcomes=list(outs),
        end=int(round(end)), open=int(round(est)) if est is not None else None,
        coin=g.get("coin"), contract_length=clen, source=src.name,
        gamma={k: m.get(k) for k in GAMMA_KEYS},
        event_meta=(ev or {}).get("eventMetadata"), event_start_ev=(ev or {}).get("startTime"))


async def discover(http, src: Source, now: float | None = None) -> list[MarketInfo]:
    now = time.time() if now is None else now
    rx = re.compile(src.slug_regex)
    ex = re.compile(src.exclude_regex) if src.exclude_regex else None
    params = {"closed": "false", "limit": 100, **src.params}
    if src.window == "start_time":
        params["start_time_min"] = iso(now - src.max_duration_s)
        params["start_time_max"] = iso(now + src.lookahead_s)
    else:
        params["end_date_min"] = iso(now - src.linger_s)
        params["end_date_max"] = iso(now + src.horizon_s)
    out, cursor, pages = [], None, 0
    while pages < src.max_pages:
        q = dict(params)
        if cursor:
            q["after_cursor"] = cursor
        d = await http.get(f"{GAMMA}/events/keyset", params={k: str(v).lower() if isinstance(v, bool) else v
                                                                 for k, v in q.items()})
        pages += 1
        evs = d.get("events", []) if isinstance(d, dict) else d
        for ev in evs:
            for m in ev.get("markets") or []:
                if m.get("closed") or m.get("enableOrderBook") is False:
                    continue
                if ex and ex.search(m.get("slug") or ""):
                    continue
                mi = market_from_gamma(m, ev, src, rx)
                if mi is not None:
                    out.append(mi)
        cursor = d.get("next_cursor") if isinstance(d, dict) else None
        if not cursor or not evs:
            break
    return out


async def fetch_market_final(http, slug: str) -> dict | None:
    """Full gamma record (closed=true first, like data/fetch_trades.py) for metadata finalization."""
    g = await http.get(f"{GAMMA}/markets", params={"slug": slug, "closed": "true"})
    if not g:
        g = await http.get(f"{GAMMA}/markets", params={"slug": slug})
    ev = None
    if not g:
        # right after the close /markets?slug= can briefly return [] while the event endpoint has it
        # (up/down event slug == market slug)
        try:
            e = await http.get(f"{GAMMA}/events/slug/{slug}", tries=2)
        except Exception:  # noqa: BLE001  (404 for markets whose event slug differs)
            e = None
        ms = [m for m in (e or {}).get("markets") or [] if m.get("slug") == slug]
        if not ms:
            return None
        g, ev = ms, e
    g = g[0]
    meta = {k: g.get(k) for k in GAMMA_KEYS}
    ev = ev or (g.get("events") or [{}])[0]
    meta["eventMetadata"] = ev.get("eventMetadata")
    meta["eventStartTimeEv"] = ev.get("startTime")
    meta["closed"] = g.get("closed")
    return meta


def resolution(g: dict) -> tuple[int | None, float | None]:
    """(winner index 0/1 or None if unresolved, outcome-0 price)."""
    try:
        prices = [float(p) for p in jl(g.get("outcomePrices") or "[]")]
    except (TypeError, ValueError):
        return None, None
    if len(prices) != 2:
        return None, None
    p0 = prices[0]
    closed = bool(g.get("closed")) or bool(g.get("closedTime"))
    if closed and {round(p, 6) for p in prices} <= {0.0, 1.0} and sum(prices) == 1.0:
        return (0 if p0 > 0.5 else 1), p0
    return None, p0
