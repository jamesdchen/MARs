# External fair-value sources and sports L2 data (tested 2026-10-15 19:43-19:45 UTC)

## Keyless odds (SEA @ DEN moneyline, Polymarket 0.56/0.57 at the time)
| source | keyless | SEA (no-vig / mid) | notes |
|---|---|---|---|
| Pinnacle guest API `guest.api.arcadia.pinnacle.com/0.1` (leagues 889 NFL, 1456 NHL, 487 NBA) | yes | 0.567 (-139/+125, vig 2.6%) | sharp; but from this container only Cloudflare cache hits (max-age 905 s, age 394-772 s); cache-busting gets 204/403 → ≤15 min stale here; may be live from other IPs; no history |
| ESPN `site.api.espn.com/.../scoreboard` `competitions[].odds` (DraftKings) | yes | ~0.571 (-142/+120) | max-age 5 s; NFL/NBA/NHL/MLB/NCAA/soccer; retail, lags; `summary?event=` has play-by-play (5-30 s behind TV, unmeasured) |
| FanDuel `sbapi.*.sportsbook.fanduel.com/api/content-managed-page` | yes | ~0.571 | retail |
| DraftKings direct | 403 (Akamai) | | use ESPN |
| Smarkets `api.smarkets.com/v3/events/…/markets/…/quotes/` | yes (User-Agent) | 0.570 (bid 56.50/ask 57.47) | betting exchange with depth, in-play, low commission → best free live executable reference |
| Kalshi `api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNFLGAME` | yes (429 bursts) | 0.565 | has trade/candle history; highly correlated with Polymarket, not independent |
| Betfair | no (app key + account) | | |
| The Odds API | no (401; free 500 credits/month, history paid) | | availability only |
| OddsPortal | HTML scrape | | historical closing odds |

All agree with Polymarket within ~0.5¢ pre-game.

## Sports L2 for a replay
- **PolyOrderBooks** (docs.polyorderbooks.com/openapi.json): crypto+sports, ~1.16M markets. `/v1/markets/{id}/books`,
  `/v1/tokens/{tid}/books` (ladders, last book per bucket at 250ms…1d), `/prices`, `/metrics`, `/events`, `/series`,
  `POST /v1/ai/backtest`. All `/v1/*` need `X-API-Key`. Starter $0: 60 req/min, 1k/day, 3 days history. $19/29/39/49 per
  month for 30/60/90/120 days of history; +$19/39 for 300 req/min, 50k/day. 200 points/page → a 3 h game at 1 s is
  ~54 pages per token: fine on paid tiers, impossible at scale on the free tier.
- **Marketlens** (api.marketlens.trade/v1, `pip install marketlens`): tick-level full-depth L2 since 2026-03-01;
  NBA/NFL/MLB/NHL/tennis/soccer/UFC through each game, esports (LoL, Dota 2, CS, Valorant). Free: last 7 days, 2M rows
  one-time. Pro $39/month (5B rows, full archive). Parquet export + backtest SDK.
- **Hugging Face (free, ungated)**: `Joseph3222/polymarket-orderbook` raw CLOB WS of every market 2026-02-22..08-10
  (~1.18 TB, Jun 12-17 missing) + 1-min full-depth snapshots (~192 GB); `Houroux/polymarket-l2-history` WS messages
  for reward-eligible markets Jul-Oct 2026 (~310 GB gz JSONL, still updated), includes `sports_ws` (live scores);
  `DineshKumar8399/polymarket-orderbook-dataset` top-of-book ~2-min polls with category column (0.55 GB);
  `Coyevans/polymarket-kalshi-scoresync-orderbook-sample` (ToB joined with scores).
- Others: Telonex (L2 from 2025-10, daily parquet, some free), ZenHodl (score-synced, paid), Apify scrapers.
  Tardis has no Polymarket.
