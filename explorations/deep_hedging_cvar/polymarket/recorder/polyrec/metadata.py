"""Market metadata rows in the data/raw/markets.parquet schema (same mapping as data/fetch_trades.py)."""
from __future__ import annotations

import datetime as dt
import math

from .discovery import jl, parse_ts, resolution

NAN = float("nan")


def _ts(s):
    t = parse_ts(s)
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc)


def _f(x, default=NAN):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def market_row(slug: str, coin, contract_length, g: dict, n_taker: int, n_fills: int, truncated: bool) -> dict:
    """g: trimmed gamma market (discovery.GAMMA_KEYS + eventMetadata/eventStartTimeEv/closed).

    Token columns follow outcome order: token_up = outcome 0 ("Up" for up/down markets), token_down =
    outcome 1. winner_up / outcome_price_up / final_price stay null until gamma reports a resolution
    (backfill_winners.py fills them later).
    """
    outcomes = jl(g.get("outcomes") or "[]")
    toks = jl(g.get("clobTokenIds") or "[]")
    fs = g.get("feeSchedule") or {}
    em = g.get("eventMetadata") or {}
    cmc = g.get("cryptoMarketConfig") or {}
    win, p0 = resolution(g)
    est = g.get("eventStartTime") or g.get("eventStartTimeEv")
    return dict(
        slug=slug, coin=coin, contract_length=contract_length,
        gamma_id=int(g["id"]) if g.get("id") is not None else None,
        condition_id=g.get("conditionId"), question_id=g.get("questionID"), question=g.get("question"),
        token_up=str(toks[0]) if len(toks) > 0 else None, token_down=str(toks[1]) if len(toks) > 1 else None,
        created_start=_ts(g.get("startDate")), event_start=_ts(est), end=_ts(g.get("endDate")),
        closed_time=_ts(g.get("closedTime")),
        tick_size=_f(g.get("orderPriceMinTickSize")), min_size=_f(g.get("orderMinSize")),
        maker_base_fee=_f(g.get("makerBaseFee"), 0.0), taker_base_fee=_f(g.get("takerBaseFee"), 0.0),
        fees_enabled=bool(g.get("feesEnabled")), fee_type=g.get("feeType"),
        fee_rate=_f(fs.get("rate")), fee_exponent=_f(fs.get("exponent")),
        fee_taker_only=bool(fs.get("takerOnly", False)), fee_rebate_rate=_f(fs.get("rebateRate")),
        maker_rebates_fee_share_bps=_f(g.get("makerRebatesFeeShareBps"), 0.0),
        rewards_min_size=_f(g.get("rewardsMinSize"), 0.0), rewards_max_spread=_f(g.get("rewardsMaxSpread"), 0.0),
        resolution_source=g.get("resolutionSource"),
        twap_lookback_s=_f(cmc.get("twapLookbackSeconds")),
        outcome_price_up=(p0 if win is not None else None),
        winner_up=(1 if win == 0 else 0) if win is not None else None,
        price_to_beat=_f(em.get("priceToBeat")), final_price=_f(em.get("finalPrice")),
        uma_status=g.get("umaResolutionStatus"),
        n_taker=int(n_taker), n_fills=int(n_fills), truncated=bool(truncated),
    )


def outcome_names(g: dict) -> list:
    return list(jl(g.get("outcomes") or "[]"))


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
