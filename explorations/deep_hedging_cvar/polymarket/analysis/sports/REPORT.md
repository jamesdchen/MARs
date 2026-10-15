# Sports market making on Polymarket: a deep-dive (worker w8, 2026-10-15)

**Question:** is market making in sports markets more promising than in crypto up/down, where our replay study
found no edge (`../../final/RESULTS.md`)? If so, where exactly, and what data would a faithful replay need?

**Short answer: not clearly better per share, but better structured, and in specific places worth a replay.**

Measured with the same keyless on-chain yardstick, makers in aggregate do worse in sports than in crypto up/down.
In sports, spread capture is 0.6¢ vs 1.0¢ per share, rebates are 0.12¢ vs 0.20¢, and the 30-minute markout is
about 0 pre-game and −0.8¢ in-play, against +0.7¢ for crypto. On the other hand, sports has three structural
features crypto lacks:
- a 1 s delay on marketable orders while makers can cancel freely;
- a cheap, sharp, external fair value pre-game (Pinnacle, Smarkets and Kalshi agree with Polymarket to within
  ~0.5¢);
- roughly 30× more maker volume per market.

The informed flow that hurts makers in sports is in-play, around scoring events: maker fills just before a ≥8¢ move
mark out at −1.1¢ after 10 s. External exchanges cannot protect against it; Smarkets suspends on goals and reprices
~60 s after Polymarket. **Promising:** pre-game soccer (UCL/EPL/La Liga), esports pre-match, NBA/NHL pre-game with
thinner queues, prices 0.5–0.7, quoting against a Pinnacle/Smarkets anchor, and pulling quotes before the start.
**Avoid:** in-play quoting without a sub-second score feed, NFL pre-game moneylines (queues of millions of shares
at a 1¢ spread), and tails above 0.9.

Everything here is train-free exploratory analysis. No file under `data/build_jun15` or any val/test set was touched.

## 0. Data and method

| piece | what | where |
|---|---|---|
| census | gamma `/events?tag_id=100639` (sports "Games"), every event with endDate Oct 1–15 (22,340 closed events, 498k markets incl. open) | `scripts/census.py`, `out/census_summary.txt` |
| on-chain fills | `onchain/extract.py --slugs` for 69 high-volume sports markets (games Oct 8–15: NFL, CFB, MLB, NHL, NBA, UCL, EPL, La Liga, ATP/WTA, CS2, LoL, Dota 2, Valorant, WNBA, MLS; 61 moneylines, 8 spreads/totals), blocks of 2026-10-02 → 10-14. 761 segments, ~38 min, ~90 GB JSON streamed, 51 MB kept. Markets with game start after 10-14 20:00 were dropped because their in-play falls outside the pull, leaving **65 markets, 281k maker legs, 130M maker shares, $61M taker notional, $1.06M taker fees** | `scripts/maker_econ.py`, `out/maker_econ.txt` |
| crypto comparison | same extractor and same script on all crypto up/down, 2026-10-14 08:00–14:00 (865 markets, 331k maker legs) | `out/maker_econ_crypto.txt` |
| live books | `recorder/polyrec` (generic source, `recorder_sports.toml`) on 18 sports markets from 19:42 UTC: UEL/UECL in-play, CS2/LoL in-play, NHL ×4, NBA ×2, CFB, NFL SEA@DEN moneyline/spread/total through kickoff | `scripts/live_books.py`, `out/live_books.txt` |
| external odds | Smarkets exchange quotes every 3 s for 13 of those games (keyless) | `scripts/smarkets_poll.py`, `scripts/leadlag.py`, `out/leadlag.txt` |
| docs and vendors | docs.polymarket.com, CLOB rewards endpoints, odds APIs, L2 vendors | `out/docs_notes.md`, `out/sources_notes.md` |

**Markout yardstick (keyless, no book).**
- Every leg is put in the YES frame (outcome 0).
- Each taker tx reveals one touch: the best maker price is the ask when the taker buys YES and the bid when it sells.
  The mid proxy is (last ask + last bid)/2 from touches strictly before t.
- Per maker share:
  - spread capture `sc` = dir·(mid before the block − price);
  - markout at h = dir·(mid(t+h) − price);
  - resolution markout = dir·(payoff − price);
  - rebate = rebateRate·rate·p(1−p).
- Values are size-weighted in ¢/share. SEs are clustered by market.
- Phases: **pre** = block time < `gameStartTime`; **live** = after it.
- "Jump" fills: the mid moves ≥8¢ within the next 60 s.
- Wallet × market pairs are classed **two-sided** (|net|/gross < 0.3 over ≥20 legs, market-maker-like), **one-sided**
  (> 0.7, limit-order bettors) or **mixed**.

Caveats:
- This measures what the **incumbent** maker population earns. A new entrant has worse queue position.
  The crypto RL replay showed that this gap is what matters.
- The touch proxy is noisy, and stale in thin markets.
- Resolution markouts are dominated by 65 binary outcomes: the SE is 1.3¢ in-play and 6¢ pre-game, so use them as a
  sanity check only.

## 1. Structure

### 1.1 Rules (docs.polymarket.com, keyless API; details and URLs in `out/docs_notes.md`)

| item | sports | crypto up/down (for contrast) |
|---|---|---|
| taker fee | `C·rate·p(1−p)`: rate **0.05** (`sports_fees_v3`); **0.03** for NFL/CFB moneylines (`sports_fees_nfl_cfb_oct26`, undocumented); **0** for NFL/CFB spreads, totals and props (`zero_fees`, 10% of sports game volume) | 0.07 |
| maker rebate | **15%** of the maker's fee-equivalent, pro rata per market, daily (cut from 25% on Jul 10). Measured 0.10–0.16¢/maker share; 0 on zero-fee markets | 20%, about 0.2¢/share |
| in-play delay | `secondsDelay` (1 s on 97% of game markets; 0 on MLB and most NBA; 3 on a few). **Only marketable orders** wait. "During either delay, the order is pending and cannot be canceled", and checks re-run at expiry. Resting orders post and cancel immediately, so makers get a 1 s head start on takers | 50 ms crypto taker delay |
| game start | `clearBookOnStart=true` on every game market: all resting orders are cancelled at the official start. The docs warn that an early start may not be cleared in time | n/a |
| liquidity rewards | Live CLOB `/rewards/markets/current`: sports-tagged markets ≈ **$49k/day** in total (esports $23k, NFL $10.5k, soccer $7.3k, NHL $2.5k, CFB $2.2k, NBA $1.35k, MLB $0.9k). Typical per-market rates are $1, $10 or $50/day with max spread 4.5¢ and min size 20–100. Outliers: one LoL Worlds match at $3.4k/day, and NFL/CFB first-half spreads at $514–1,886/day with min size 1,000 and max spread 2.5¢ on markets that trade only ~$2k. The scorer has an undocumented "in-game multiplier". No Sep–Oct sports program (the last was March Madness at $60k/day on live moneylines) | none since September |
| holding rewards | 4% APY exists, but `holdingRewardsEnabled=false` on sports | — |
| ticks | 0.01; 0.001 near the extremes | 0.01 |

Rebates and rewards compared with fees:
- Our 65 markets paid $1.06M of taker fees in 13 days, so the rebate pool was ~$160k (~$2.4k per game market).
- Scaled to the category, about $0.8M/day of sports taker fees gives ~$120k/day of rebates (less after the zero-fee
  NFL/CFB props).
- Liquidity rewards are about $49k/day spread over 7,800 markets.
- **Rebates and rewards are second-order (~0.1¢/share)** except in a few reward outliers (LoL Worlds; NFL/CFB 1H
  spreads at min size 1,000), which deserve a separate look.

### 1.2 Where the volume is (Oct 1–14, gamma volume; `out/census_summary.txt`)

| sport | share | top leagues (share of all sports-game volume) | $M per event |
|---|---|---|---|
| soccer | 28% | Nations League 8.0%, UCL 5.2%, EPL 2.1%, La Liga 1.4%, FIFA qualifiers 1.0% | UCL 0.52, EPL 0.28 |
| esports | 28% | CS2 11.9%, LoL 8.0% (Worlds), Dota 2 6.4%, Valorant 1.8% | LoL 1.24, Dota 0.80, Val 0.93, CS2 0.25 |
| American football | 18% | NFL 12.7%, CFB 5.6% | NFL 1.41 |
| tennis | 16% | ATP 10.1%, WTA 5.0% | 0.15–0.18 |
| baseball | 4% | MLB (postseason) 4.2% | 0.17 |
| hockey / basketball | 2% / 2% | NHL (season start), NBA (preseason), WNBA finals | 0.27 / 0.39 |

By market type: moneyline 62% + child moneyline 14%, totals 11%, spreads 7.5%, other props < 6%.

### 1.3 Pre-game vs in-play (taker notional, 65-market sample)

**58% of taker notional trades in-play.** Including the last pre-game hour, 68% trades within ±3 h of the start.

| league | in-play share | | league | in-play share |
|---|---|---|---|---|
| WNBA (finals) | 98% | | LoL | 68% |
| WTA / ATP | 86% / 82% | | MLB | 67% |
| Dota 2 | 83% | | La Liga | 60% |
| CFB | 80% | | CS2 | 46% |
| NFL | 71% | | NBA / EPL / NHL | 40% / 37% / 34% |
| | | | UCL | 23% |

Taker notional by time to start:
- before −24 h: 2%;
- −24 h to −6 h: 6%;
- −6 h to −1 h: 12%;
- last hour: 22%;
- 0 to +30 min: 9%;
- +30 to +60 min: 8%;
- +1 to +2 h: 16%;
- +2 to +3 h: 14%;
- after +3 h: 11%.

## 2. Maker economics (on-chain, keyless)

### 2.1 Headline (¢ per maker share, size-weighted; market-clustered SE in brackets)

| | spread capture | +10 s | +1 min | +5 min | +30 min | resolution | rebate | maker shares |
|---|---|---|---|---|---|---|---|---|
| **sports pre-game**, all makers | 0.62 (0.09) | 0.33 (0.06) | 0.22 (0.07) | 0.10 (0.04) | 0.09 (0.40) | −2.9 (6.2) | 0.16 | 50M |
| **sports in-play**, all makers | 0.58 (0.04) | 0.10 (0.08) | 0.12 (0.17) | −0.15 (0.23) | −0.83 (0.61) | −1.5 (1.3) | 0.10 | 79M |
| sports pre-game, **two-sided** wallets | 1.04 (0.35) | 0.60 (0.23) | 0.53 (0.26) | 0.47 (0.14) | 1.89 (0.94) | −6.0 (8.7) | 0.15 | 6M |
| sports in-play, **two-sided** wallets | 0.70 (0.06) | 0.31 (0.07) | 0.44 (0.18) | −0.25 (0.29) | −1.47 (0.68) | +1.6 (1.4) | 0.11 | 32M |
| sports in-play, **one-sided** wallets | 0.35 (0.06) | −0.21 (0.16) | −0.25 (0.27) | −0.24 (0.48) | −1.38 (1.25) | −4.5 (2.9) | 0.08 | 30M |
| **crypto up/down** (Oct 14 08–14 UTC), all makers | 1.03 (0.05) | 0.20 (0.09) | 0.15 (0.22) | 0.67 (0.23) | 0.69 (0.26) | 0.63 (0.27) | 0.20 | 4.5M |
| crypto up/down, two-sided wallets | 1.16 (0.08) | 0.33 (0.17) | −0.09 (0.40) | 0.68 (0.21) | 0.82 (0.24) | 0.80 (0.24) | 0.24 | 1.6M |

Reading:
1. **Spread capture is lower in sports.** Pre-game books sit at one 1¢ tick (§4), so the half-spread is at most 0.5¢.
2. **Pre-game, makers keep a small positive edge at short horizons** (+0.2¢ at 1 min, +0.1¢ at 5 min, significant).
   By 30 min it is about 0, with wide error. Two-sided wallets do better (+0.5¢ at 5 min).
3. **In-play, the edge decays to negative by 5–30 min.** This is driven by scoring events (2.3).
4. **Aggregate crypto makers look better on this yardstick** (+0.7¢ at 30 min and to resolution). That does not
   contradict the RL result: incumbents with Binance-speed and queue priority earn what a new entrant cannot
   (`final/RESULTS.md`). Sports makers in aggregate do worse, because the population includes many one-sided
   "limit-order bettors" whose resting orders lose to in-play information.

### 2.2 Where (selected cells; full tables in `out/maker_econ.txt`)

**By league, 30-min markout + rebate (¢/share).** Pre-game:
- EPL +1.50; UCL +0.63; MLB +0.59; LoL +0.31; La Liga +0.46; NFL +0.09;
- CS2 −0.08; ATP −1.78; Valorant −4.3; NHL −5.6. The last two are small samples.

In-play:
- EPL +4.7; WNBA +11.4 (1 game); MLB +1.85; LoL +1.80; CS2 +0.48; La Liga +0.46; UCL −0.31;
- ATP −1.16; WTA −1.59; NHL −1.74; NFL −2.17; Dota 2 −2.64; NBA −3.07; CFB −3.83.

Football-style in-play markets, where possessions change slowly and scores arrive in jumps (NFL, CFB, NBA), and
tennis/Dota are the worst for makers. Soccer and LoL are the least bad.

**By price (favourite probability), two-sided wallets, +30 min:**
- pre-game: 0.5–0.6 +1.48 (spread capture 1.40), 0.6–0.7 +0.59, 0.7–0.8 −0.02;
- in-play: 0.5–0.6 −3.11, 0.6–0.7 −1.72, 0.8–0.9 −3.07.
- Above 0.95, spread capture is 0.2¢ (tick-bound) and the markouts are noise.

**By time to start (all makers):**
- 24 h+ before: +2.6 at 30 min, small sample;
- 6–24 h: 0.0;
- 2–6 h: 0.1;
- 30 min–2 h: +0.5;
- last 30 min: 0.0;
- first hour in-play: −1.1;
- 1–3 h in: +0.5;
- 3–10 h in (long esports/tennis/cricket): −4.2.

**By market type:**
- spreads/totals pre-game: spread capture 0.6¢, +30 min about 0.
- NFL/CFB spreads and totals pay **no fee and no rebate**. Their makers face takers who pay nothing, so expect
  tighter, more informed flow.

### 2.3 Scoring events

In-play maker legs followed within 60 s by a ≥8¢ mid move are **11% of in-play maker shares**. They capture 1.1¢ of
spread but mark out at **−1.1¢ (+10 s)**, −0.5¢ (+1 min) and −1.3¢ (+5 min), and **−8.5¢ to resolution**. The other
in-play fills hold +0.26¢ at 10 s and +0.2¢ at 1 min. This is the sports analogue of the Binance pick-off:
- the informed traders are the fastest viewers of the game feed (TV/stream latency, court-siders, data feeds);
- the 1 s marketable-order delay protects makers only if they see the event within that second.

### 2.4 Top maker wallets (realised to resolution, 65 markets)

- **8,006 wallets** made markets.
- The top 5 hold 29% of maker shares and the top 25 hold 50%. Compare crypto: 2,452 wallets, top 25 = 31%.
- Top-25 maker legs net **−$0.20M** to resolution including rebates. All their legs, counting their taker trades,
  net **+$1.17M**.
- Individual results swing by ±$1M: 0x2c33… −$1.13M; 0xf0f3… +$1.17M; 0x1f3a… −$0.55M.
- Spread capture alone ($45–80k per top wallet) and rebates ($7–11k) are small next to the directional inventory
  P&L.
- The largest sports "makers" are mostly directional books that rest size, not flat market makers. Their per-share
  net (−12¢ … +16¢) is direction, not spread.
- Aggregate maker legs net −$2.5M to resolution. With only 65 outcomes, that is within noise of 0 (−1.9¢/share, SE
  ≈ 2–6¢).

## 3. Fair value: is there a Binance for sports?

**Pre-game: yes, and it is free.** The sources below were tested keylessly; details in `out/sources_notes.md`:
- Pinnacle's guest API: sharp, all sports. From this container it is only reachable through a CDN cache that is
  ≤15 min stale. It may be live from other IPs.
- Smarkets exchange: live quotes with depth, in-play too; needs a User-Agent header.
- ESPN, which carries DraftKings (retail, 5 s cache).
- FanDuel JSON and Kalshi, which are correlated with Polymarket.
- Not keyless: The Odds API (key; 500 free credits/month), Betfair (account and app key).
- OddsPortal has historical closing odds, but only by scraping.

**Lead/lag, tonight (`out/leadlag.txt`; Polymarket books at 1 Hz vs Smarkets at 3 s):**
- **Pre-game (NFL, NHL ×4):** Polymarket mid within **0.3–0.5¢** of the Smarkets mid on average. The Smarkets mid
  was always inside Polymarket's 1¢ touch, while Smarkets' own spread was 1–2.4¢.
  - NBA preseason: Smarkets spreads are 2.6–4¢, and its mid fell outside Polymarket's touch 50–75% of the time.
  - Polymarket is the tighter, more liquid venue. The external price is an anchor for drift and for sanity, not a
    leading signal at the 1 s scale.
- **In-play (5 UEL matches):** Polymarket moved first on every large move.
  - On goals Smarkets suspends, and its quotes catch up **~60 s later**.
  - Cross-correlations of 3 s mid changes are weak and peak at Polymarket-leads lags of 9–30 s.
  - An external exchange cannot serve as a fast in-play reference. Only a low-latency score/data feed can. Polymarket
    streams scores (`wss://sports-api.polymarket.com/ws`), but every taker sees the same feed, so there is no edge
    there.

**Cheap pre-game anchor:** the de-vigged Pinnacle moneyline, or the Smarkets mid, refreshed every few seconds. It
pins fair value to within the 1¢ tick, so a quoter does not need its own model pre-game.

## 4. Live books (polyrec, 18 markets; `out/live_books.txt`)

LIVEBOOKS_PLACEHOLDER

## 5. Data for a faithful replay and env port

The C env is data-source agnostic: it takes ticks/feat/fills/markets binaries. A sports port needs four inputs:

1. **L2 books through the game.** The on-chain data gives fills only (no cancels, queue or book).
   - Free: `Joseph3222/polymarket-orderbook` on Hugging Face (raw CLOB WS of every market, Feb 22–Aug 10 2026,
     ~1.2 TB, plus 1-min snapshots ~190 GB) and `Houroux/polymarket-l2-history` (WS messages of reward-eligible
     markets Jul–Oct 2026, ~310 GB, still updated, with a `sports_ws` score folder).
   - Paid:
     - **Marketlens Pro, $39/month:** tick-level full-depth L2 since Mar 2026 through each game, including esports,
       with parquet export.
     - **PolyOrderBooks:** an API key unlocks `/v1/markets/{id}/books` and `/v1/tokens/{tid}/books` at 250 ms–1 d
       buckets, plus `/prices`, `/metrics` and `/events`. The free tier is 3 days of history at 1,000 requests/day;
       30–120 days cost $19–49/month. 200 points per page means ~54 requests per token-game at 1 s.
   - Ours: `polyrec` with a generic source works as-is for sports, but it needs `post_close_s`/`linger_s` of 4–5 h,
     because sports `endDate` equals the game start. A week of recording gives fresh, exactly matched episodes.
2. **Fills:** `onchain/extract.py --slugs` (exact, keyless). Each day costs ~3 min of a ~90 GB stream, and only the
   kept txs are stored. It can only pull whole UTC days below the chain head; for today, use the data-api.
3. **Fair-value features:**
   - pre-game: the Smarkets/Pinnacle mid sampled at 1–5 s (record it live; there is no free history);
   - in-play: the Polymarket sports WS score/period/elapsed stream, recorded with local receive times;
   - `features.py` needs a sports variant: replace the Binance digital-option fair value with the external no-vig
     probability, and add score-state features.
4. **Market metadata:** `gameStartTime` drives the game-start wipe (`clearBookOnStart`: cancel all quotes at gs).
   The env also needs `secondsDelay` (delay taker arrivals by 1 s relative to maker cancels) and per-market fee and
   rebate rates (0.05/0.15, 0.03/0.15, or 0/0 on NFL/CFB props).

Suggested sequence:
1. Run polyrec on ~200 sports markets per day for 7–10 days, plus the Smarkets/Pinnacle poller and the sports WS,
   and pull on-chain fills for the same markets.
2. Split by day as in `recorder/split_config.json`.
3. Build with a sports `features.py`.
4. Train and evaluate pre-game only first; in-play quoting is the higher-risk extension.

## 6. Verdict and risks

**Is sports more promising than crypto up/down?**

1. **On realised economics: no.** The incumbent maker population earns about 0 pre-game and loses in-play at 30 min.
   Crypto incumbents earn about +0.7¢. Spread is tick-bound at 1¢, and rebates are 15% of a 0.05 (or 0.03, or 0)
   fee.
2. **On structure, sports pre-game is the most tractable place we have seen for a model-light maker:**
   - fair value is observable for free from sharp books and exchanges, and agrees with Polymarket to 0.3–0.5¢;
   - takers are delayed 1 s while makers cancel instantly;
   - prices drift slowly (pre-game mid changes ≈ 0 per minute for 30 min in NFL/NHL books);
   - pre-game two-sided makers keep +0.5¢ at 5 min (SE 0.14).

   **Where:** soccer (UCL, EPL, La Liga) and esports (LoL, CS2) pre-match, NBA/NHL pre-game, prices 0.5–0.7, the last
   0.5–2 h before the start.

   **The binding constraint is queue position, not information.**
   - Pre-game NFL books hold ~2.5M shares within 1¢, against ~80k maker shares filled per hour in the last 6 h.
   - So a new quote at the touch would wait hours.
   - Fills mostly arrive when the price moves through the quote, which is adverse.
   - Thinner books turn over within minutes: NBA (2–6k shares within 1¢ vs 23k/h) and esports/CS2 in-play.
3. **In-play is where volume is (58%), and where the pick-off risk lives.**
   - Fills before ≥8¢ jumps lose 1.1¢ in 10 s and 8.5¢ to resolution.
   - Only a maker with a sub-second score feed and automatic pulls on events should quote in-play.
   - The 1 s taker delay helps exactly that maker.

**Main risks:**
- event jumps and stale quotes after scores;
- game-start wipe timing: early starts may not be cleared, and the docs warn about this;
- one-sided inventory into resolution. Binary outcomes give fat P&L tails, and top wallets swing ±$1M;
- fee regime changes (0.03→0.05 on Jul 10; NFL/CFB special rate; zero-fee props);
- concentration: the top 5 makers hold 29% of maker volume;
- home/away and outcome-order mix-ups (the docs warn assignments can change);
- the delay rule can change by market (0/1/3 s);
- our estimates use a touch proxy, and resolution P&L over 65 outcomes is statistically weak.

**Recommendation for the next step:**
- If any sports work is pursued, build a **pre-game soccer/esports/NBA replay** from a week of polyrec + Smarkets/
  Pinnacle + on-chain fills (or Marketlens at $39), with the game-start wipe and the 1 s taker delay modelled.
- Test a fair-value quoter anchored to the external no-vig price before any RL.
- Treat in-play as out of scope until a low-latency score feed is in the data.
