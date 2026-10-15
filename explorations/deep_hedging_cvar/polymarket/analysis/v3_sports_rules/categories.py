"""Robust category for a gamma market row (fee vertical first, tags second)."""
import re

CRYPTO_SHORT = re.compile(r"-updown-(5m|15m|1h|4h)-\d+$")
GEO = {"geopolitics", "middle-east", "iran", "israel", "ukraine", "russia", "military", "china", "gaza", "world"}
ELEC = {"elections", "midterms", "global-elections", "world-elections", "house-elections", "senate-midterms",
        "governor-midterms", "us-presidential-election", "main-election", "midtermmov", "nov-4-elections",
        "international-election-props", "israel-election"}
ECON = {"economy", "fed-rates", "macro-indicators", "fed", "inflation", "economics", "gdp", "jobs-report", "cpi"}
FIN = {"stocks", "equities", "finance-updown", "pyth-finance", "commodities", "oil", "big-tech", "pre-market",
       "finance", "gold", "indices"}
CULT = {"pop-culture", "movies", "music", "awards", "celebrities", "tweets-markets", "mention-markets", "culture",
        "tv", "box-office"}
TECH = {"tech", "ai", "ai-rankings", "ai-releases", "openai", "anthropic", "google"}
SPORT_FEES = {"sports_fees_v3", "sports_fees_v2", "sports_fees_nfl_cfb_oct26"}


def market_category(slug: str, tags: str, fee_type: str, has_game_start: bool):
    """-> (category, subcategory). Sports are split pre-game / in-play per fill later; here 'sports_game'
    (has a game start) or 'sports_futures'."""
    t = set(tags.split("|")) if isinstance(tags, str) else set()
    slug = slug if isinstance(slug, str) else ""
    ft = fee_type if isinstance(fee_type, str) else ""
    if ft == "crypto_fees_v2" or "crypto" in t or "crypto-prices" in t:
        m = CRYPTO_SHORT.search(slug)
        if m:
            return "crypto_updown_short", m.group(1)
        if "up-or-down" in t:
            return "crypto_other", "updown_daily+"
        return "crypto_other", "hit_price" if ("hit-price" in t or "multi-strikes" in t) else "other"
    if ft in SPORT_FEES or "sports" in t or "esports" in t or "games" in t:
        sub = "esports" if "esports" in t else ("soccer" if "soccer" in t else
              next((x for x in ("nfl", "cfb", "ncaaf", "nba", "mlb", "nhl", "tennis", "basketball", "baseball",
                                "hockey", "football") if x in t), "other"))
        return ("sports_game" if has_game_start else "sports_futures"), sub
    if ft == "weather_fees" or "weather" in t:
        return "weather", "temperature" if "daily-temperature" in t else "other"
    if ft == "economics_fees" or t & ECON:
        return "economics", "fed" if "fed-rates" in t else "macro"
    if ft == "finance_prices_fees" or t & FIN:
        return "finance_prices", "updown" if "finance-updown" in t else "other"
    if t & ELEC:
        return "elections", "midterms" if "midterms" in t else "other"
    if t & GEO:
        return "geopolitics", "middle-east" if "middle-east" in t else "other"
    if ft == "politics_fees" or "politics" in t:
        return "politics", "trump" if "trump" in t else "other"
    if ft == "mentions_fees" or "mention-markets" in t:
        return "culture", "mentions"
    if ft == "culture_fees" or t & CULT:
        return "culture", "other"
    if ft == "tech_fees" or t & TECH:
        return "tech_ai", "other"
    return "other", ft or "none"
