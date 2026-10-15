# Sports market making on Polymarket: a deep-dive (worker w8, 2026-10-15)

**Question:** is market making in sports markets more promising than in crypto up/down, where our replay study
found no edge (`../../final/RESULTS.md`)? If so, where exactly, and what data would a faithful replay need?

**Short answer: not better per share for an average maker, but better structured, and there is one clearly
profitable segment: two-sided in-play professionals.**

On the same keyless on-chain yardstick (235 sports markets, Oct 2–14), sports makers in aggregate earn less than
crypto up/down makers:
- spread capture 0.6¢ vs 1.0¢ per share;
- rebates 0.09–0.13¢ vs 0.20¢;
- 30-minute markout −0.3¢ pre-game and −0.4¢ in-play, vs +0.7¢.

Pre-game makers keep a significant +0.25¢ at 1 min (SE 0.03), but it is gone by 5 min.

Two features of sports change the picture:
- **In-play two-sided wallets earn +1.6¢/share to resolution** including rebates (SE 0.6, 96M shares). That is about
  $1.5M in 13 days on 235 markets, the only maker segment with a significantly positive result.
- **One-sided "limit-order bettors" pay for it:** −3.4¢/share in-play (SE 1.2).

The pick-off happens at scoring events. Maker fills followed within 60 s by a ≥8¢ move are 10% of in-play maker
shares, and they lose 1.1/1.4/1.7¢ at 10 s/1 min/5 min.

Structurally, sports offers what crypto lacks:
- a 1 s delay on marketable orders, while makers cancel freely;
- a cheap, sharp, external fair value pre-game: Pinnacle, Smarkets and Kalshi agree with Polymarket to within
  ~0.5¢;
- about 30× more maker volume per market.

But external exchanges cannot protect anyone in-play: Smarkets suspends on goals and reprices ~60 s after
Polymarket.

**Where it is promising:**
1. In-play two-sided quoting, with a sub-second score feed and automatic pulls on events. This is where the
   incumbents make money, and the 1 s taker delay helps exactly this maker.
2. Pre-game quoting from listing to T−1 h (the D14 window) against a Pinnacle/Smarkets anchor. The edge is small:
   0 to +30 bps at +30 min in the 1–24 h buckets, and +250 bps (CI from 0) above 24 h on thin volume. The rebate
   adds ~25 bps. Tennis and esports are the most consistently positive leagues; NFL is flat.

**Avoid:**
- one-sided resting size;
- the last hour before the start and the first in-play hour without a feed;
- NFL pre-game moneylines (queues of millions of shares at a 1¢ spread);
- long in-play tails (3 h+ esports/tennis: −2.2¢).

Everything here is train-free exploratory analysis. No file under `data/build_jun15` or any val/test set was touched.

## 0. Data and method

| piece | what | where |
|---|---|---|
| census | gamma `/events?tag_id=100639` (sports "Games"), every event with endDate Oct 1–15 (22,340 closed events, 498k markets incl. open) | `scripts/census.py`, `out/census_summary.txt` |
| on-chain fills, main sample | `onchain/extract.py --slugs` for 239 high-volume sports game markets (`out/sample_slugs_big.txt`; per league top-25 by volume, ≥$150k, games Oct 3–14; 23 leagues; moneylines plus 25 spreads/totals), blocks 2026-10-02 → 10-14. 6 workers, 761 segments, ~20 min, ~40 GB JSON streamed. Markets starting after 10-14 20:00 were dropped (their in-play is outside the pull), leaving **235 markets, 956k maker legs, 380M maker shares, $181M taker notional, $2.67M taker fees** | `scripts/maker_econ.py --big`, `out/maker_econ_big.txt` |
| on-chain fills, first sample | the same for 69 markets (games Oct 8–15), leaving 65 markets: a robustness check with the same conclusions | `out/maker_econ.txt` |
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
- Resolution markouts are dominated by the binary outcomes: the SE is 0.6¢ in-play and 2.9¢ pre-game over 235
  markets.
- The touch proxy was checked against tonight's recorded books (`scripts/proxy_check.py`, `out/proxy_check.txt`).
  Over 7 markets and 1,175 trades it is unbiased (mean error −0.1 to +0.3¢), with mean absolute error 0.3–1.0¢ and
  65–83% of errors within half the 1¢ spread. Markout means are reliable; single fills are noisy.

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
- Our 235 markets paid $2.67M of taker fees in 13 days, so the rebate pool was ~$400k (~$1.7k per market).
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

### 1.3 Pre-game vs in-play (taker notional, 235-market sample)

**59% of taker notional trades in-play.** Including the last pre-game hour, 74% trades within ±3 h of the start.

| league | in-play share | | league | in-play share |
|---|---|---|---|---|
| WNBA (finals) / Euroleague | 97–99% | | NFL | 64% |
| WTA / ATP | 89% / 84% | | La Liga / MLB / Bundesliga | 61% / 55% / 54% |
| Dota 2 | 85% | | CS2 / Valorant / UFC | 52% / 53% / 48% |
| CFB | 75% | | EPL / NBA | 45% / 40% |
| LoL | 69% | | NHL / Nations League / UCL | 34% / 33% / 30% |

Taker notional by time to start:
- before −24 h: 1.5%;
- −24 h to −6 h: 6%;
- −6 h to −1 h: 11%;
- last hour: 22%;
- 0 to +30 min: 10%;
- +30 to +60 min: 9%;
- +1 to +2 h: 20%;
- +2 to +3 h: 13%;
- after +3 h: 7%.

## 2. Maker economics (on-chain, keyless)

### 2.1 Headline (¢ per maker share, size-weighted; market-clustered SE in brackets)

| | spread capture | +10 s | +1 min | +5 min | +30 min | resolution | rebate | maker shares |
|---|---|---|---|---|---|---|---|---|
| **sports pre-game**, all makers | 0.61 (0.04) | 0.38 (0.03) | 0.25 (0.03) | 0.03 (0.08) | −0.27 (0.45) | −0.6 (2.9) | 0.13 | 144M |
| **sports in-play**, all makers | 0.63 (0.03) | 0.10 (0.04) | −0.02 (0.07) | −0.24 (0.11) | −0.41 (0.30) | −0.8 (0.6) | 0.09 | 237M |
| pre-game, **two-sided** wallets | 0.73 (0.15) | 0.48 (0.09) | 0.39 (0.10) | −0.10 (0.37) | 0.39 (0.64) | −2.3 (3.7) | 0.11 | 19M |
| pre-game, **one-sided** wallets | 0.57 (0.03) | 0.36 (0.03) | 0.22 (0.03) | 0.05 (0.07) | −0.13 (0.40) | 0.0 (3.1) | 0.13 | 105M |
| in-play, **two-sided** wallets | 0.75 (0.04) | 0.25 (0.05) | 0.10 (0.09) | −0.19 (0.15) | −0.01 (0.34) | **+1.48 (0.59)** | 0.10 | 96M |
| in-play, **one-sided** wallets | 0.43 (0.04) | −0.09 (0.06) | −0.25 (0.11) | −0.44 (0.21) | −1.56 (0.58) | **−3.36 (1.24)** | 0.08 | 89M |
| **crypto up/down** (Oct 14 08–14 UTC), all makers | 1.03 (0.05) | 0.20 (0.09) | 0.15 (0.22) | 0.67 (0.23) | 0.69 (0.26) | 0.63 (0.27) | 0.20 | 4.5M |
| crypto up/down, two-sided wallets | 1.16 (0.08) | 0.33 (0.17) | −0.09 (0.40) | 0.68 (0.21) | 0.82 (0.24) | 0.80 (0.24) | 0.24 | 1.6M |

The 65-market first sample (`out/maker_econ.txt`) gives the same pattern:
- pre-game: +0.22¢ at 1 min, +0.09¢ at 30 min;
- in-play: −0.83¢ at 30 min;
- two-sided in-play: +1.6¢ to resolution;
- jump fills: −1.1¢ at 10 s.

Reading:
1. **Spread capture is lower in sports** (0.6 vs 1.0¢). Pre-game books sit at one 1¢ tick (§4), so the half-spread
   is at most 0.5¢.
2. **Pre-game, makers keep a small, highly significant edge at 10 s–1 min** (+0.4/+0.25¢), which is gone by 5 min.
   The external anchor (§3) pins the price, so there is little to be picked off on, but also little drift to earn.
3. **In-play, the population splits.**
   - Two-sided wallets, which quote both sides and stay roughly flat, earn +1.5¢ to resolution plus a 0.1¢ rebate.
   - One-sided wallets, whose resting orders take a view, lose 3.4¢.
   - This is a transfer from slow resting money to fast two-sided makers and takers.
4. **Aggregate crypto makers look better on this yardstick** (+0.7¢ at 30 min and to resolution). That does not
   contradict the RL replay: incumbents with Binance speed and queue priority earn what a new entrant cannot
   (`final/RESULTS.md`). The same caution applies to the sports in-play pros.

### 2.2 Where (selected cells; full tables in `out/maker_econ_big.txt`)

**By league, 30-min markout + rebate (¢/share; per-league SEs are large, so read signs only where |x| > 1).**

Pre-game:
- EPL +1.6; WTA +1.2; Dota 2 +0.3; UCL +0.2;
- MLB 0.0; CS2 −0.1; NFL −0.2; ATP −0.7; CFB −1.0; LoL −1.1; Nations League −1.3; NHL −5.6 (small).

In-play:
- EPL +4.0; Nations League +1.3; MLB +0.6; Valorant +0.5; ATP +0.1; CS2 +0.1; UCL +0.1; LoL 0.0;
- Dota 2 −0.8; WTA −0.9; NFL −1.2; CFB −2.2; NBA −3.1 (small); La Liga −3.4; Bundesliga −5.3.

High-scoring US football (NFL/CFB) and some soccer leagues are worst in-play. CS2/LoL/UCL are about flat. EPL is
positive in both phases.

**By price (favourite probability), all makers, +30 min + rebate.**
- Pre-game: 0.5–0.6 −0.4; 0.6–0.7 −0.6; 0.7–0.8 +1.4; 0.8–0.9 +0.7; 0.9–0.95 +1.1.
- In-play: −0.2 to −1.0 in every bin.
- Above 0.95, spread capture is 0.07–0.2¢ (tick-bound), and maker volume there is large (48M in-play shares) but
  worthless.
- Two-sided in-play wallets are positive to resolution in every bin from 0.5 to 0.95 (+1.3 to +2.7¢), except
  0.7–0.8.

**By time to start (all makers, +30 min + rebate):**

| window | value (¢) |
|---|---|
| 24 h+ before | +1.3 |
| 6–24 h before | +0.1 |
| 2–6 h before | +0.2 |
| 30 min–2 h before | +0.45 |
| **last 30 min before** | **−0.9** |
| **first hour in-play** | **−0.9** |
| 1–3 h in-play | +0.4 |
| 3–10 h in-play (long esports/tennis/cricket) | −2.2 |

**D14 pre-game buckets** (`scripts/pregame_buckets.py`, `out/pregame_buckets.txt`). Maker edge at +30 min in bps of
maker notional, ex-rewards, with a market-cluster bootstrap 95% CI. The rebate adds 23–28 bps in every bucket.

| time before kickoff | markets | maker shares | spread capture | +1 min | +30 min (bps) | 95% CI |
|---|---|---|---|---|---|---|
| > 24 h | 183 | 5M | 111 bps | 14 bps | **+251** | [0, +701] |
| 6–24 h | 230 | 22M | 100 | 31 | **0** | [−19, +20] |
| 1–6 h | 234 | 39M | 147 | 79 | **+28** | [−7, +64] |
| last hour | 235 | 78M | 125 | 49 | **−130** | [−463, +178] |

Agreement with w7/w2 (D14):
- **Agrees on the shape:** >24 h positive, 1–6 h mildly positive, last hour about 0 or negative with a wide CI.
- **Disagrees on 6–24 h:** we find 0, against their +41/+43.
- **The last hour is 54% of pre-game maker volume on chain**, so D14's T−1 h cut-off gives up most of the pre-game
  fill volume, but the noisiest and worst part.

>1 h by league, +30 min (bps, CI):
- ATP +61 [26, 99]; Valorant +38 [6, 63]; Dota 2 +33 [14, 63]; UNL +88 [−62, 215]; UCL +36 [−40, 86];
  La Liga +36 [−32, 83]; WTA +31 [−11, 73]; CS2 +27 [−3, 55];
- LoL +9; NFL −3 [−28, 21]; CFB −12; MLB −26; EPL −28.
- Last hour: LoL −710 [−1570, −138]. Most others are negative with wide CIs.

**By market type:**
- Spreads and totals have higher spread capture (0.8¢ pre-game) but are thin.
- NFL/CFB spreads and totals pay **no fee and no rebate**, so their makers face takers who pay nothing.

### 2.3 Scoring events

In-play maker legs followed within 60 s by a ≥8¢ mid move are **10% of in-play maker shares**. They capture 1.6¢ of
spread but then mark out at:
- **−1.1¢** at +10 s;
- **−1.4¢** at +1 min;
- **−1.7¢** at +5 min;
- −3.3¢ to resolution.

The other in-play fills hold +0.23¢ at 10 s and +0.13¢ at 1 min. This is the sports analogue of the Binance
pick-off:
- the informed traders are the fastest viewers of the game feed (TV/stream latency, court-siders, data feeds);
- the 1 s marketable-order delay protects makers only if they see the event within that second;
- the two-sided pros evidently do, or they price the risk in, since their in-play 10 s markout is +0.25¢.

### 2.4 Top maker wallets (realised to resolution, 235 markets)

- **11,345 wallets** made markets.
- The top 5 hold 26% of maker shares and the top 25 hold 45%. Compare crypto: 2,452 wallets in 6 h, top 25 = 31%.
- Top-25 maker legs net **+$0.70M** to resolution including rebates. All their legs, counting their taker trades,
  net **+$2.29M**.
- Individual results swing by ±$1–2M: 0x2c33… −$1.81M on 135 markets; 0xf0f3… +$1.31M on 39; 0x1f3a… +$1.05M on 83;
  0x6754… −$0.68M.
- Spread capture ($15–180k per top wallet) and rebates ($1–26k) are small next to the inventory P&L.
- The biggest sports "makers" are mixed books: large resting size plus active taking (millions of taker shares).
- Broad, flat two-sided operators earn small positive amounts: 0xfe78… +$0.13M on 203 markets; 0x2005… +$0.18M on
  159; 0x5268… +$0.08M on 103.
- Aggregate maker legs net −$2.3M to resolution, about −0.6¢/share, within ~1 SE of zero.

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
- **Pre-game (NFL, NHL ×4, NBA ×2, ~3.5 h each):** the Polymarket mid is within **0.2–0.8¢** of the Smarkets mid on
  average. The Smarkets mid sits inside or at Polymarket's 1¢ touch, and outside it 2–59% of the time (rounding at
  the touch). Smarkets' own spread was 1.3–3¢.
  - NBA preseason: Smarkets spreads are 2.6–4¢, and its mid fell outside Polymarket's touch 50–75% of the time.
  - Polymarket is the tighter, more liquid venue. The external price is an anchor for drift and for sanity, not a
    leading signal at the 1 s scale.
- **In-play (5 UEL matches, 3 NHL games):** Polymarket moved first on large moves.
  - Smarkets suspends on goals. Where it caught up, its half-move came a median **~48 s** after Polymarket's
    (12 of 26 big Polymarket moves matched; in the other 14 Smarkets never caught up within 90 s; range −90 to +180 s).
  - Its in-play spreads were 2–9¢ against Polymarket's 1¢.
  - Cross-correlations of 3 s mid changes are about 0 at every lag from −60 to +60 s. Polling every 3 s plus
    suspensions make the exchange useless as an in-play reference.
  - An external exchange cannot serve as a fast in-play reference. Only a low-latency score/data feed can. Polymarket
    streams scores (`wss://sports-api.polymarket.com/ws`), but every taker sees the same feed, so there is no edge
    there.

**Cheap pre-game anchor:** the de-vigged Pinnacle moneyline, or the Smarkets mid, refreshed every few seconds. It
pins fair value to within the 1¢ tick, so a quoter does not need its own model pre-game.

## 4. Live books (polyrec, 18 markets; `out/live_books.txt`)

Recorded from 19:42 to 01:20 UTC (~5.6 h). 18 markets:
- 5 UEL + 1 UECL matches in-play;
- CS2 and LoL Bo3/Bo5 in-play;
- NHL ×4 (three of them through the start at 23:00);
- NBA ×2, CFB, and the NFL SEA@DEN moneyline, spread and total (kickoff 00:15).

The recorder ran with 0.01–0.08% REST-reconcile mismatches and 9/9 connections; 38 reconnects were recovered. Rows
come at 1 Hz, on change. Tables are in `out/live_books.txt`.

| | pre-game (median over markets) | in-play (median) |
|---|---|---|
| spread | 1¢ (one tick) 100% of the time | 1¢; one tick 92% of the time |
| depth within 1¢ of mid, both sides | 150k shares (NFL moneyline 3.0M, NFL spread 1.8M, NHL 50–210k, NBA 3.7–3.9k) | 9.7k shares (NHL 5–14k, CS2 3.8k, LoL 2.6k) |
| depth within 5¢ | 680k (NFL 10M) | 80k |
| mid changes per minute | **0.005** (NFL: none in 3.6 h; NHL ~0.01) | **5.6** (NHL 1.3–2.6, CS2/LoL 4.7–4.9) |

**Game start (NHL ×3, 23:00:00):** the `clearBookOnStart` wipe shows up to the second.

| time from start | resting shares, both sides (BUF@TOR / NYI@OTT / NYR@NJ) |
|---|---|
| −60 s | 785k / 680k / 802k |
| −1 s | 702k / 597k / 678k |
| **+1 s** | **92k / 106k / 115k** (−85%) |
| +5 s | 269k / 231k / 198k |
| +60 s | 555k / 282k / 273k |
| +5 min | 574k / 317k / 438k |
| +15 min | 226k / 222k / 224k |

- The spread never left 1¢. In-play makers re-quote within a second of the wipe, so the touch is not left empty for
  takers.
- A maker that relied on the wipe instead of pulling its own quotes was protected exactly at the official start.

**In-play jumps (|Δmid| ≥ 5¢ in 30 s): 73 in the recorded in-play hours.**
- The median spread stays 1¢ before and after a jump.
- The maximum spread within 60 s after a jump is typically 2–5¢; the widest was 31¢ on a UEL goal.
- Depth within 2¢ typically falls by 30–90% for about a minute. On the 20¢ Hoffenheim goal it went from 26k to 58
  shares.
- So in-play makers pull or get swept at events, but refill quickly.

**Score feed vs book (`scripts/score_latency.py`, `out/score_latency.txt`).** Polymarket's public sports WebSocket
was logged with receive times.
- For every score change matched to a recorded game, the book had already done half of its move:
  - **NHL goals:** 27–59 s before the feed;
  - **esports map wins:** 96–140 s before the feed.
- **The book moved first in 8/8 cases (median 87 s).**
- The public score feed is far too slow to protect in-play quotes. Takers who pick off makers get scores from faster
  sources (broadcast/data feeds, which are themselves seconds behind the venue).
- An in-play maker needs a commercial low-latency feed, or a book-reactive rule: pull or widen when the mid or the
  opposite touch moves, using the 1 s taker delay as the reaction window.


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

1. **For an average maker: no.**
   - The incumbent population earns +0.25¢ at 1 min pre-game, then about −0.3¢ by 30 min.
   - In-play it loses −0.4¢ at 30 min and −0.8¢ to resolution.
   - Crypto incumbents measured the same way earn about +0.7¢.
   - The spread is tick-bound at 1¢, and the rebate is 15% of a 0.05 (or 0.03, or 0) fee.
2. **For a fast two-sided in-play maker: yes. This is the one significantly profitable segment found.**
   - +1.5¢ + 0.1¢ rebate per share to resolution, SE 0.6, on 96M shares in 13 days across 235 markets.
   - It is paid for by one-sided resting money (−3.4¢) and by the fact that takers face a 1 s delay while makers
     cancel instantly.
   - The risk is concentrated at scoring events: 10% of in-play maker volume, −1.4¢ at 1 min.
   - To play, a maker needs:
     - a score/event feed faster than the takers' (Polymarket's sports WS is public and shared, so it is not enough
       alone);
     - automatic pulls on events;
     - inventory control into resolution.
3. **Pre-game is the most tractable but thinnest opportunity:**
   - fair value is observable for free from sharp books and exchanges, and agrees with Polymarket to 0.3–0.5¢;
   - prices hardly move: no mid change in 30 min of NFL/NHL books.

   **Where:** listing to T−1 h (D14); ATP, esports (Valorant, Dota 2, CS2) and UCL/La Liga/UNL >1 h buckets;
   favourites at 0.7–0.95. Quotes off in the last hour, which is 54% of pre-game maker volume but −130 bps with a
   wide CI.

   **The binding constraint is queue position, not information.**
   - Pre-game NFL books hold ~2.5M shares within 1¢, against ~120k maker shares filled per hour in the last 6 h, so
     a new quote at the touch would wait hours.
   - Fills mostly arrive when the price moves through the quote, which is adverse.
   - NHL, NBA, esports and tennis books turn over far faster. NBA, for example: 2–6k shares within 1¢ vs 23k/h.
   - Liquidity rewards (~$49k/day across sports) are small, except for outliers: LoL Worlds; NFL/CFB 1H spreads at
     min size 1,000.

**Main risks:**
- event jumps and stale quotes after scores;
- game-start wipe timing: early starts may not be cleared, and the docs warn about this;
- one-sided inventory into resolution. Binary outcomes give fat P&L tails, and top wallets swing ±$1–2M;
- fee regime changes (0.03→0.05 on Jul 10; NFL/CFB special rate; zero-fee props);
- concentration: the top 5 makers hold 26% of maker volume;
- home/away and outcome-order mix-ups (the docs warn assignments can change);
- the delay rule can change by market (0/1/3 s);
- our estimates use a validated but noisy touch proxy, and these are incumbent economics, not those of a new entrant
  at the back of the queue.

**Recommendation for the next step:**
1. Build a **sports replay** from 7–10 days of polyrec books, the sports score WS and Smarkets/Pinnacle quotes, plus
   on-chain fills (or Marketlens at $39 for history). Model the game-start wipe, the 1 s taker delay and per-market
   fees.
2. Pre-game: test a fair-value quoter anchored to the no-vig external price.
3. In-play: the question to answer first is **how fast our event detection would be relative to the takers'**.
   Measure the lag of the Polymarket sports WS and ESPN against the book's own jump times in the recorded data.
   Without an edge there, in-play quoting copies the losing one-sided population, not the pros.
