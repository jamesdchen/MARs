# Fair-value anchor for pre-game sports quoting (w2, 2026-10-15 ~20:20-20:40 UTC)

Question: is there a free external probability a pre-game quoter can centre on, and is it better than Polymarket's
own micro-price? All requests below were made from this container through the agent proxy, with TLS verification on,
no keys and no accounts. Scratch outputs: `/tmp/claude-0/-home-user-MARs/94d31458-3e17-5c33-ba31-1c08290af5cd/scratchpad/fv/`.

## 1. What w8 already found (`analysis/sports/out/sources_notes.md`, `docs_notes.md`, `leadlag.txt`)

- Keyless live odds: Pinnacle guest API, ESPN scoreboard (DraftKings), FanDuel content API, Smarkets exchange
  quotes, Kalshi. DraftKings direct is blocked (Akamai 403). Betfair needs an app key and account. The Odds API
  needs a key (free tier is 500 credits/month, and history is paid only). OddsPortal is HTML scraping only.
- On SEA @ DEN (NFL), every source agreed with Polymarket (0.56/0.57) **within about 0.5¢**: Pinnacle no-vig
  0.567, Smarkets mid 0.570, DraftKings and FanDuel about 0.571, Kalshi 0.565.
- From this container, Pinnacle only serves Cloudflare cache hits (`max-age=905`, age up to about 13 min), so it
  can be up to about 15 min stale. Cache-busting returns 204 or 403.
- The lead/lag test (`scripts/leadlag.py`, Polymarket 1 Hz vs Smarkets 3 s polls) produced **no result**:
  `leadlag.txt` has an empty level-agreement table. **Nobody has measured lead/lag yet.**
- Polymarket mechanics that matter here: makers can cancel freely, takers wait about 1 s on sports, and books are
  wiped at kickoff. Pre-game spreads are 1 tick (1¢) almost 100% of the time, with deep books
  (`live_books.txt`: median 0.1M shares within 1¢ pre-game).

## 2. Sources tested today

| source | keyless? | leagues | live / history | limits and ToS | tested request → result |
|---|---|---|---|---|---|
| **Pinnacle guest** `guest.api.arcadia.pinnacle.com/0.1/leagues/{id}/matchups` and `/markets/straight` (ids: 889 NFL, 246 MLB, 487 NBA, 1456 NHL, 1980 EPL) | yes (keyless, though browser clients send an `X-API-Key` header) | all major leagues, soccer 1X2 | live only, **no history**. Cache-hit only from here (age 762 s of 905). | unofficial website backend. Pinnacle closed its public API to non-customers in 2025, so ToS caution applies. | `leagues/1980/markets/straight` → 200, 740 kB. Everton–Chelsea no-vig H/D/A 0.294/0.262/0.444 (overround 3.3%) vs Polymarket mids 0.295/0.265/0.445 |
| **Smarkets** `api.smarkets.com/v3/events/…`, `/markets/{ids}/quotes/` | yes (needs a User-Agent) | NFL, NBA, NHL, MLB, soccer, tennis | live, exchange depth. **No history**: settled events are not listed, and `last_executed_prices` returns 404. | public read API, polite polling (w8 used 3 s) | `events/?type=american_football_match&state=upcoming` → 200, events 45076214 HOU@JAX etc. |
| **Matchbook** `api.matchbook.com/edge/rest/events?sport-ids=15&include-prices=true` | yes for GET events and prices | soccer and US sports | live exchange, thin. Many markets come back with no prices. | the API officially expects a session, so treat as grey | → 200, 163 kB. Hoffenheim–Besiktas back/lay prices present, Everton–Chelsea empty |
| **ESPN** `site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates=YYYYMMDD` | yes | NFL, MLB, NBA, NHL, NCAA, soccer (eng.1, esp.1, uefa.*) | live DraftKings line (max-age 5 s), **plus historical open and close** through the core API, below | unofficial and undocumented | `scoreboard` NFL → 200, 254 kB. SEA@DEN DK "SEA -2.5", O/U 40.5 |
| **ESPN core** `sports.core.api.espn.com/v2/sports/{s}/leagues/{l}/events/{id}/competitions/{id}/odds` | yes | same as ESPN | **historical DK open and close moneyline / spread / total** per game. `…/odds/100/history/0/movement` returned an empty result. | unofficial | MLB 401817091 (Sep 26) open +109/-131, close -110/-109. NHL Oct 10 open -130/+110, close -148/+124. NFL Sep 27 close -345. EPL Oct 11 open +160, close +190, draw +235 |
| **SportsbookReview (SBR)** `www.sportsbookreview.com/betting-odds/{mlb-baseball,nhl-hockey,nba-basketball}/?date=YYYY-MM-DD` plus `/betting-odds/{league}/line-history/{gameId}/` (`__NEXT_DATA__` JSON) | yes (HTML page with embedded JSON) | NFL, MLB, NBA, NHL, NCAA. No soccer: the soccer page redirects (308) and the EPL page is 404. | **timestamped line history per book** (BetMGM, bet365, DK, FD, Fanatics, BetRivers, Caesars, Hard Rock) and open/current per date. NFL `?date=` is ignored and shows the current week, so NFL needs gameIds. | scraping a website. Check ToS, keep it to a light one-off crawl. | MLB NYM@WAS 2026-09-26 line-history → 200, 13 DK moneyline changes from 09-25 19:36 to 09-26 16:26 (start 16:35). NFL game 379626 → 8 books, up to 90 points. |
| **Kalshi** `api.elections.kalshi.com/trade-api/v2/` `markets?series_ticker=KXNFLGAME|KXMLBGAME|KXNHLGAME|KXNBAGAME|KXEPLGAME|KXUCLGAME|KXLALIGAGAME`, `series/{s}/markets/{t}/candlesticks`, `markets/trades?ticker=` | yes | NFL, MLB, NHL, NBA, EPL, UCL, La Liga, … | live **and full history**: 1-min or 1-h candles with yes bid/ask OHLC, plus every trade | public market data. 429 on bursts, so pace to about 1-2 req/s. | `candlesticks?period_interval=60` for KXNFLGAME-26SEP27LARDEN-DEN → 200, 102 candles with bid 0.45 / ask 0.46. `markets/trades` → 200, tick trades. |
| **football-data.co.uk** `football-data.co.uk/mmz4281/2627/{E0,SP1,D1,I1,F1,…}.csv`, `/new/USA.csv` | yes (free CSV, 302 then 200) | European leagues and MLS. No US majors. | historical, updated about twice a week (E0 last modified Oct 12). Pre-closing and **closing** odds for bet365, Betfair Exchange (`BFE*`, `BFEC*`), BetVictor, bwin, … plus Max/Avg. **No Pinnacle (`PS*`) columns in 2026/27.** | free for research, attribution | `E0.csv` → 61 rows through 2026-10-12, with `BFECH/D/A` closing exchange odds |
| **The Odds API** `api.the-odds-api.com/v4/sports` | **no** | all | live; history is paid only (5-min snapshots) | key needed | → 401 |
| **Betfair** `api.betfair.com/exchange/betting/rest/v1.0/…` | **no** (app key and session). Historic data site needs an account too. | all | live; history needs an account | not usable here | → 403 |
| **OddsPortal** `oddsportal.com/football/england/premier-league/results/` | page yes, odds no | all | historical open and close, per-book movement | odds come from an obfuscated XHR, so it needs a headless browser. ToS forbids scraping. | → 200, 1.17 MB HTML shell, no odds in the static page |
| **Bovada** `bovada.lv/services/sports/event/coupon/events/A/description/football/nfl` | yes | US sports | live retail | grey | → 200, SEA@DEN -115/-105 (spread market) |
| FanDuel `sbapi.*` (w8) | yes | US | live retail | grey | w8: about 0.571 on SEA |
| DraftKings direct (w8), aussportsbetting (historical xlsx) | blocked | – | – | – | 403 / 403 |
| BetDEX / Monaco `prod.api.monacoprotocol.xyz` | – | – | – | – | CONNECT 502 (unreachable) |

## 3. Free historical pre-game odds for late Sep – early Oct 2026

| league | best free source | granularity | independent of Polymarket? |
|---|---|---|---|
| MLB, NHL, NBA | **SBR line-history** (per-book timestamped moneyline changes; list games with `?date=`) | event-driven, every line change | yes (US retail books, no Pinnacle) |
| NFL | **SBR line-history** by gameId (ids are not listable by date, so scan ids near 3796xx or take them from the current-week page), **ESPN core** DK open and close | event-driven / open and close | yes |
| Soccer (EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS) | **football-data.co.uk** pre-closing and closing (incl. Betfair Exchange closing), **ESPN core** DK open and close (incl. draw) | 2 points per match | yes |
| all of the above | **Kalshi candlesticks** (bid/ask OHLC, 1-min) | 1-min, timestamped | **no**: heavily arbitraged against Polymarket. Useful only as a cross-venue check. |

No free source gives timestamped sharp-book (Pinnacle or Betfair) prices for past games. Paid options are The Odds
API historical and Betfair historic data (account needed). For a pure backtest, Polymarket's own L2 history (w8's
list: PolyOrderBooks, Marketlens, HF datasets) is in any case the primary input. External odds only add a check
feature.

## 4. Recommendation

**Do not replace Polymarket's micro-price as the quote centre. Use an external anchor as a guard rail and a feature.**

- **Live anchor:** Pinnacle no-vig where it can be fetched fresh (≤60 s). From this container it is up to 15 min
  stale, so it is only usable as a slow sanity bound.
- **Live fallback (fresh, keyless):** Smarkets exchange mid, using the size-weighted micro-price of best
  back/lay. ESPN/DraftKings is the coverage fallback (5 s cache, all US leagues and soccer).
- **Backtest anchor:** SBR line history for US leagues, football-data.co.uk Betfair-Exchange closing odds for
  soccer, and ESPN DK open and close for both.

**De-vig method:**

- **Exchange mids (Smarkets, Matchbook, Kalshi):** no de-vig. Take mid = (1/back + 1/lay)/2 per runner and
  renormalise to sum 1.
- **Two-way book lines (NFL/NBA/NHL/MLB moneyline):** use **power** (find k with Σ q_i^k = 1) or **Shin**. At
  Pinnacle-level margins (2-3%) both differ from multiplicative by under 0.3¢ near 50/50. With US retail margins
  (4-5%) on heavy favourites, multiplicative overstates the longshot by up to about 1¢.
- **Three-way soccer 1X2:** use **Shin**. It corrects the favourite-longshot bias, which matters most for draws and
  longshots (Štrumbelj 2014: Shin beats basic normalisation for bookmaker odds).

**Rationale.** In the big leagues, Polymarket pre-game books are 1¢ wide and very deep. Every external source checked
today agrees with the Polymarket mid within about 0.5¢: SEA@DEN (w8) and Everton–Chelsea, where Pinnacle no-vig
was 0.294/0.262/0.444 against Polymarket 0.295/0.265/0.445. Polymarket's micro-price already embeds the
sharp-book price at sub-tick precision, so an external centre adds noise, especially from a 15-min-stale Pinnacle
cache or 4-5%-margin retail books. The anchor earns its keep in three places:

- **Guard rail:** pull or widen quotes when |PM micro − anchor| exceeds about 1.5-2¢. This catches stale books,
  news Polymarket has not priced yet, and thin books right after listing or in small leagues.
- **Signal:** an anchor move that Polymarket has not followed is a candidate adverse-selection predictor. This is
  the lead/lag question w8 left unanswered, and it needs a working Smarkets-vs-Polymarket run before anyone relies
  on it.
- **Backtest check:** the drift of the Polymarket mid toward the closing line is a measure of pre-game information
  flow.

**Next step:** rerun `analysis/sports/scripts/leadlag.py` with a pre-game Smarkets log of a few hours (e.g. the
Saturday Oct 17 EPL and NCAA slate). Promote the anchor to a quote-centre input only if Smarkets leads Polymarket
by at least 3 s on moves of 2¢ or more.
