# Where on Polymarket is market making most promising? A maker-economics census from on-chain fills

Worker w7, written 2026-10-15. The window is **Oct 8–14 2026 (7 full UTC days)**, every market on the CTF and
neg-risk exchanges. No order book is needed and no API key: everything comes from complete on-chain fills plus
gamma metadata and the CLOB rewards table. Nothing from `data/build_jun15` or the v2 val/test days was touched.

## TL;DR

1. **Pivoting away from crypto up/down is justified, but no category has a big, clean edge.** In almost every
   category, adverse selection eats the quoted spread almost exactly. At +2 h, adverse selection is 70–190% of the
   spread. Before liquidity rewards, the average maker fill earns about 0 ± 1¢/share. What separates the
   categories is (a) how fast the adverse selection arrives, (b) liquidity rewards, and (c) who the takers are.
2. **First choice: pre-game sports, from listing to kickoff, pulling quotes at kickoff.**
   - It is the only large category whose edge before rewards is positive with the 95% CI above 0 at +30 min:
     **+64 bps of maker notional [1, 130]**. That is about +0.33¢/share: spread 0.72¢ − adverse selection 0.54¢
     + rebate 0.15¢.
   - It is positive on 5 of 7 days, and the top-20 makers earn +236 bps.
   - Capacity is $14.9M/day of maker volume and $247k/day of taker fees. Maker competition is moderate: 7.1k
     makers, top-5 share 21%.
   - The edge to *resolution* is a coin flip (+109 bps, CI ±700), so the strategy must flatten or stop quoting
     at kickoff rather than hold to settlement.
3. **Second choice: liquidity-reward farming in long-dated elections markets (the US midterms are Nov 3).**
   - Before rewards the edge is about 0: +2 h gives −8 bps [−51, 18].
   - With rewards it is +133 bps [94, 165] at +2 h and +122 bps [74, 156] at +1 d.
   - This category has the lowest adverse selection per share of any: 0.5–0.6¢.
   - It has the largest reward pool: about $30k/day across 3.7k rewarded markets.
   - The edge is the reward program's money. It disappears if the program changes, and election night is a
     jump event.
4. **Avoid these:**
   - **In-play sports:** −39 bps [−134, 51]. Non-top makers lose −123 bps, because takers with live data beat
     the book.
   - **Weather:** negative on every day and at every horizon, −87 bps [−115, −56] before rewards. Its takers
     are forecast-model users.
   - **Economics, tech/AI, culture, finance-prices:** adverse selection exceeds the spread, with CIs below 0
     before rewards.
5. **Validation: our crypto up/down result reproduces inside the census.**
   - The top-20 crypto up/down makers are the continuous quoters an RL agent resembles. They lose −23 bps,
     with adverse selection of 1.35¢/share against a spread of 1.02¢. 18 of the 20 are individually profitable,
     but the group loses in aggregate.
   - The category average is ≈ 0: +30 bps [−6, 69], made up of spread 0.99¢, adverse selection 1.03¢ and
     rebate 0.20¢. It is carried by occasional, non-continuous makers.
   - This matches `final/RESULTS.md` (no edge, adverse selection dominates) and `analysis/fair_value` §4.

## 1. Data and method

| step | what | numbers |
|---|---|---|
| fills | `onchain/extract.py --all --events OrderFilled --slim`, Oct 8–14, both exchanges, nice 10 | 403k blocks, 25.72M fill legs (9.4M taker, 16.3M maker), 0 failed segments, 11.6 min wall, 843 MB on disk |
| categories | `gamma_tags.py`: `/markets?clob_token_ids=..&closed=true/false&include_tag=true`, 40 per call, cached | 237,337 tokens, 237,323 resolved (99.99%); tags, feeSchedule, gameStartTime, rewards config, outcome prices |
| rewards | `rewards_current.py`: CLOB `/rewards/markets/current`, all pages | 18,838 markets, $209k/day in total (snapshot of 2026-10-15) |
| legs | `build_legs.py` (DuckDB, 2 threads) | one row per maker leg with pre-fill micro-price, marks at +1m/+5m/+30m/+2h/+1d, settlement, rebate |
| census | `census.py`, `top_makers.py`, `extras.py`, `daily.py` | `out/*.csv` |

**Categories** (`categories.py`) are assigned from the fee vertical first (gamma `feeType`), then from tags:
- crypto up/down short is `*-updown-{5m,15m,1h,4h}-*`; other crypto is daily up/down, hit-price and similar;
- sports markets that have a `gameStartTime` are split per fill into **pre-game** (fill time < game start) and
  **in-play**; sports markets without one are **futures**;
- then weather, economics (Fed and macro), finance-prices (stocks, commodities and finance up/down), elections,
  geopolitics, politics (the rest), culture (including mentions), and tech/AI.

**Per maker leg** (frame: the leg's own token; `s = +1` for a maker buy and −1 for a maker sell; q is the fill price):
- *micro-price* = mean of the latest taker-buy and latest taker-sell prints of the condition. Both tokens are
  folded into one outcome-0 series. If the two sides disagree by more than 10¢ (stale side), the latest print
  is used instead. `mid_pre` is the micro-price as of the block before the fill.
- **spread** = s·(mid_pre − q).
- **markout at h** = s·(mid(t+h) − mid_pre). Its negative is the **adverse selection**. Once the market has
  closed, mid(t+h) is the settlement price. Legs whose t+h falls past Oct 15 00:00 are left out at that horizon.
- **to resolution** = s·(final − mid_pre), for markets resolved by Oct 15 ~20:00 UTC. Coverage is the `res cov`
  column below. It is near 100% for sports, crypto and weather and 1–50% for long-dated categories, which makes
  it selection-biased there; the report uses +2 h and +1 d for those.
- **rebate** = rebateRate·feeRate·q(1−q), from the market's `feeSchedule`:
  - crypto 0.07/20%;
  - sports v3 0.05/15%;
  - politics, finance and tech 0.04/25%;
  - weather, culture and economics 0.05/25%.

  Fee units were checked: taker `fee` = size·rate·p(1−p) USDC, with a median ratio of 1.000 per fee type.
- **reward** = the condition's `rate_per_day` (snapshot) × its active days in the window, spread over the
  condition's maker shares. This is a proxy: real rewards pay resting quotes, not fills. As a check, the top-20
  makers' actual data-api `REWARD` payouts for the window were **2.0×** the proxy for the same wallets
  ($461k vs $230k, corr 0.62). The proxy is therefore conservative for an active quoter.
- **edge** = spread + markout + rebate (+ reward). It is reported per share (¢) and per $ of maker notional (bps).

"Per $ quoted" cannot be observed without a book (cancels and resting size are off chain), so edge is per **$
filled**. Uncertainty is a Poisson bootstrap (1000 replicates) **over gamma events**, because markets of one
game or one election share news and outcomes.

## 2. Validation on crypto up/down

`out/crypto_validation.csv`, `out/top_vs_rest.csv`

| crypto up/down | maker $/day | spread ¢ | adverse sel. 1m / 5m / res ¢ | rebate ¢ | edge to res ¢/share |
|---|---|---|---|---|---|
| 5m (14,687 mkts) | 7.16M | 0.97 | 1.05 / 0.89 / 0.90 | 0.19 | +0.27 |
| 15m (4,997) | 2.29M | 1.00 | 1.03 / 1.33 / 1.42 | 0.22 | −0.20 |
| 4h (342) | 0.07M | 1.85 | 1.35 / 1.59 / 2.50 | 0.23 | −0.42 |
| **top-20 makers** | 2.45M | 1.02 | 1.32 (5m) / 1.35 (res) | 0.20 | **−0.13 (−23 bps)** |
| everyone else | 7.07M | 0.98 | 0.91 / 0.94 | 0.20 | +0.24 (+48 bps) |

Before rebates, the spread minus adverse selection is ≤ 0 for 15m and 4h and +0.07¢ for 5m. The continuous,
high-volume quoters are picked off: adverse selection 1.35¢ against a 1.02¢ spread. This is the population our
agent belongs to, and it reproduces "adverse selection > spread capture". The small positive average comes from
occasional makers whose resting orders get filled on the right side. A replay cannot reproduce that, because
it reflects which orders take the queue position, not a strategy we can run.

Earlier numbers on the train days point the same way:
- the per-share spread here (≈ 1¢) is in line with `analysis/rewards` §2c, which reported 0.2–0.9¢ of maker
  edge against the pre-fill mid;
- `analysis/fair_value` §4 found that a mid-centred maker loses 2–3¢/fill.

## 3. The census

### 3a. Size, competition and per-share decomposition (¢/share, size-weighted; AS = adverse selection)

| category            |   markets |   events | maker $/day   | taker fees $/day   |   makers | top-5 share   |   HHI |   spread |   AS 5m |   AS 2h |   AS 1d |   AS res |   res cov % |   rebate |   reward |   edge 2h |   edge 1d |
|:--------------------|----------:|---------:|:--------------|:-------------------|---------:|:--------------|------:|---------:|--------:|--------:|--------:|---------:|------------:|---------:|---------:|----------:|----------:|
| sports_pregame      |     43664 |     8625 | 14940k        | 247.1k             |     7137 | 21%           |   145 |     0.72 |    0.51 |   -0.36 |    0.27 |     0.22 |          94 |     0.15 |     0.02 |      1.25 |      0.62 |
| sports_futures      |      8791 |     1197 | 576k          | 8.5k               |     3233 | 12%           |    71 |     0.42 |    0.37 |    0.49 |    0.49 |     3    |          11 |     0.05 |     0.29 |      0.26 |      0.27 |
| elections           |      7832 |     1690 | 2173k         | 24.6k              |     5268 | 12%           |    67 |     0.48 |    0.47 |    0.59 |    0.63 |     4.82 |           1 |     0.09 |     0.48 |      0.46 |      0.42 |
| finance_prices      |      3889 |      763 | 618k          | 5.2k               |     3106 | 14%           |    85 |     0.96 |    1.17 |    1.42 |    1.5  |     2.84 |          22 |     0.1  |     0.91 |      0.56 |      0.47 |
| geopolitics         |      1229 |      437 | 2065k         | 3.3k               |     4116 | 17%           |   105 |     0.89 |    0.73 |    0.9  |    0.59 |    -0.83 |          32 |     0.02 |     0.28 |      0.28 |      0.59 |
| crypto_updown_short |     20026 |    20026 | 9515k         | 190.5k             |     7791 | 10%           |    59 |     0.99 |    1    |    1.03 |    1.04 |     1.04 |         100 |     0.2  |     0    |      0.16 |      0.15 |
| crypto_other        |      8997 |     2037 | 2429k         | 26.4k              |     5795 | 39%           |   390 |     0.8  |    0.66 |    0.88 |    0.77 |     0.73 |          73 |     0.11 |     0.04 |      0.07 |      0.19 |
| culture             |      1546 |      262 | 149k          | 2.1k               |     1839 | 13%           |    73 |     1.71 |    1.8  |    2.63 |    2.79 |     3.05 |          41 |     0.14 |     0.74 |     -0.05 |     -0.21 |
| tech_ai             |      1376 |      192 | 195k          | 2.4k               |     2121 | 15%           |    89 |     1.4  |    1.93 |    2.3  |    2.35 |     1.56 |          38 |     0.12 |     0.71 |     -0.07 |     -0.13 |
| politics            |      1382 |      302 | 976k          | 8.5k               |     3064 | 22%           |   169 |     0.7  |    0.77 |    0.97 |    0.96 |     0.78 |          49 |     0.08 |     0.14 |     -0.06 |     -0.04 |
| sports_inplay       |     34264 |     6938 | 37560k        | 547.0k             |    13902 | 15%           |    93 |     0.82 |    0.89 |    1.14 |    1.34 |     1.32 |          99 |     0.11 |     0    |     -0.22 |     -0.41 |
| economics           |      1080 |      251 | 837k          | 11.8k              |     2681 | 21%           |   156 |     0.51 |    0.86 |    1.11 |    1.14 |     5.04 |           2 |     0.13 |     0.32 |     -0.15 |     -0.18 |
| weather             |      8794 |      977 | 1127k         | 9.7k               |     3844 | 17%           |   104 |     1.45 |    1.55 |    1.96 |    1.93 |     1.99 |          92 |     0.1  |     0.14 |     -0.27 |     -0.24 |

The whole venue in this window:
- maker volume about $73M/day, taker fees about $1.09M/day;
- in-play sports is 51% of maker volume and 50% of fees;
- pre-game sports is 20% of volume, crypto up/down 13%.

Pre-game sports' negative adverse selection at 2 h (−0.36, i.e. the price moved in the maker's favour) comes
from 2 h marks that fall after kickoff. It is in-game noise, not a pre-game effect. The 30 min mark (AS 0.54¢)
is the clean pre-game number.

### 3b. Ranking by edge per $ of maker notional (bps, event-bootstrap 95% CI)

| category            | ex-rewards +30m   | ex-rewards +2h    | ex-rewards +1d   | ex-rewards to res   | with rewards +2h   | with rewards +1d   |   reward pool $/day |   maker edge $/day at +2h |
|:--------------------|:------------------|:------------------|:-----------------|:--------------------|:-------------------|:-------------------|--------------------:|--------------------------:|
| sports_pregame      | 64 [1, 130]       | 246 [-108, 627]   | 118 [-580, 800]  | 109 [-627, 808]     | 250 [-107, 629]    | 124 [-579, 801]    |                6461 |                    373600 |
| sports_futures      | 3 [-50, 37]       | -22 [-96, 28]     | -14 [-136, 72]   | -478 [-789, 239]    | 145 [67, 199]      | 155 [43, 207]      |                9457 |                      8400 |
| elections           | 11 [-27, 38]      | -8 [-51, 18]      | -20 [-59, 10]    | -874 [-1450, -214]  | 133 [94, 165]      | 122 [74, 156]      |               29971 |                     28900 |
| finance_prices      | -81 [-137, -43]   | -87 [-148, -40]   | -100 [-223, -7]  | -357 [-745, 13]     | 117 [35, 160]      | 101 [-54, 169]     |               11806 |                      7200 |
| geopolitics         | 10 [-92, 86]      | -1 [-151, 94]     | 63 [-239, 256]   | 337 [-163, 678]     | 56 [-85, 154]      | 115 [-169, 296]    |               11314 |                     11500 |
| crypto_updown_short | 30 [-7, 69]       | 30 [-6, 69]       | 28 [-8, 68]      | 28 [-8, 68]         | 31 [-6, 69]        | 29 [-8, 68]        |                   0 |                     29100 |
| crypto_other        | 9 [-18, 32]       | 3 [-27, 30]       | 25 [-38, 82]     | 36 [-39, 108]       | 14 [-19, 39]       | 36 [-31, 89]       |                1951 |                      3300 |
| culture             | -90 [-193, 24]    | -215 [-339, -89]  | -237 [-463, -28] | -85 [-586, 357]     | -10 [-172, 80]     | -62 [-304, 115]    |                2824 |                      -200 |
| tech_ai             | -188 [-305, -112] | -205 [-342, -118] | -211 [-427, -73] | 92 [-846, 429]      | -17 [-122, 55]     | -33 [-214, 57]     |                3526 |                      -300 |
| politics            | -37 [-141, 39]    | -71 [-174, 4]     | -72 [-185, -5]   | -36 [-271, 71]      | -26 [-125, 45]     | -26 [-129, 26]     |                3793 |                     -2500 |
| sports_inplay       | -38 [-101, 26]    | -40 [-134, 51]    | -76 [-187, 24]   | -72 [-182, 32]      | -39 [-134, 51]     | -75 [-187, 24]     |                  21 |                   -146600 |
| economics           | -210 [-260, -71]  | -128 [-326, -94]  | -158 [-494, -83] | -1098 [-1842, -116] | -41 [-118, 83]     | -77 [-302, 102]    |                7160 |                     -3400 |
| weather             | -51 [-70, -31]    | -87 [-115, -56]   | -77 [-121, -27]  | -90 [-138, -38]     | -56 [-91, -32]     | -47 [-110, -17]    |                3166 |                     -6300 |

"Maker edge $/day" is the with-rewards +2 h edge × maker $/day: the whole maker pool's take, not ours.
"To res" is selection-biased wherever `res cov` is low (elections 1%, economics 2%, sports futures 11%).

Day-by-day stability (`out/daily_edge.csv`): weather, economics, finance-prices and tech are negative before
rewards at +2 h on **7 of 7 days**. Crypto up/down is positive on 6/7, pre-game sports on 5/7, elections on 2/7
(≈ 0).

### 3c. Price level and time to resolution (`out/by_price.csv`, `out/by_ttr.csv`, `out/by_price_ttr.parquet`)

- **Crypto up/down.** Edge is positive at cheap prices (0.05–0.35: +0.9 to +1.7¢) and negative at 0.65–0.95
  (−0.7 to −1.3¢). Takers who buy the favourite late are informed (Binance).
- **Pre-game sports by time to kickoff** (`out/sports_detail.csv`). The spread is flat at 0.6–0.7¢ up to 24 h
  before kickoff and 2.4¢ more than 24 h before. The 5-minute adverse selection is 0.4–0.5¢ inside 24 h and
  1.4¢ beyond. Edge at +2 h and to resolution swings between −436 and +633 bps across the buckets, which is
  game-outcome noise.
- **Pre-game sports by league.** All the big leagues show a positive pre-game edge at +2 h (+250 to +520 bps)
  except tennis (−329), basketball (non-NBA, −298), esports (−24) and college football (≈ 0). Soccer is 47% of pre-game maker volume.
- **In-play by league.** NFL (−466 bps), college football (−592) and basketball (−1316) are strongly negative.
  Soccer (+48), tennis (+185) and NBA (+260) are mildly positive. In-play is a latency game: the top-20 makers
  earn +124 bps there and everyone else −123 bps.
- **Elections** have tight spreads, 0.4–0.8¢ at every price. Most volume (61%) is 7–30 days from resolution,
  i.e. the Nov 3 midterms. Rewards are 1.0–1.1¢/share at prices 0.15–0.85.
- **Weather.** Edge is negative in every price bucket, and strongly negative inside 6 h of resolution
  (≈ −2 to −18¢). Takers there carry forecast information.

### 3d. Realised P&L of the top-20 makers per category (`out/top_makers.csv`, `out/top_makers_by_cat.csv`)

Each wallet's in-window maker fills are marked to settlement, or to the last micro-price for unresolved markets.
Rewards are the wallet's data-api `REWARD` payouts for epochs Oct 8–14, split across categories by its maker
notional; `REWARD` rows carry no market. Positions carried into the window are not included.

| category            | top-20 maker $ (7d)   |   share of category |   spread bps |   adverse sel. bps |   rebates |   rewards |   P&L bps |   P&L $/day |   # profitable /20 |
|:--------------------|:----------------------|--------------------:|---------:|---------------:|----------:|----------:|----------:|------------:|-------------------:|
| sports_pregame      | 41.67M                |                  40 |      133 |             75 |        29 |         6 |       236 |      140300 |                 13 |
| geopolitics         | 5.18M                 |                  36 |       75 |             18 |         2 |       110 |       205 |       15200 |                 18 |
| elections           | 4.44M                 |                  29 |      119 |            -73 |        23 |        85 |       151 |        9600 |                 17 |
| weather             | 2.85M                 |                  36 |      224 |           -324 |        22 |       179 |       103 |        4200 |                 17 |
| tech_ai             | 0.47M                 |                  34 |      273 |           -495 |        31 |       301 |        87 |         600 |                 10 |
| crypto_other        | 9.61M                 |                  56 |      102 |            -50 |        18 |         2 |        71 |        9700 |                 15 |
| sports_inplay       | 89.6M                 |                  34 |      146 |           -113 |        21 |         5 |        56 |       72200 |                 14 |
| crypto_updown_short | 17.15M                |                  26 |      181 |           -240 |        35 |         0 |       -23 |       -5700 |                 18 |
| sports_futures      | 1.19M                 |                  30 |      145 |           -329 |        22 |       112 |       -95 |       -1600 |                 10 |
| politics            | 2.99M                 |                  44 |      149 |           -325 |        20 |        43 |      -122 |       -5200 |                 10 |
| economics           | 2.51M                 |                  43 |      104 |           -343 |        32 |        59 |      -148 |       -5300 |                  7 |
| finance_prices      | 1.41M                 |                  33 |       59 |           -425 |        19 |       150 |      -220 |       -4400 |                  7 |
| culture             | 0.31M                 |                  30 |      257 |           -707 |        32 |       216 |      -232 |       -1000 |                  7 |

- **Weather's top makers are profitable only because of rewards.** Adverse selection (−324 bps) exceeds the
  spread (+224 bps), and rewards (+179 bps) close the gap.
- **Elections and geopolitics:** the top makers earn their spread and keep it, and rewards are about 40–55% of
  P&L. Their mark-to-market in unresolved markets is noisy over one week.
- **Pre-game sports:** the +75 bps "adverse" term is favourable inventory drift at settlement over 7 days. It
  is luck-dominated.

## 4. Recommendation

**Pivot to pre-game sports as the primary target, with an elections reward-farming book as the second.** Drop
crypto up/down, in-play sports, weather, economics, finance-prices, tech and culture as targets.

| rank | target | why | expected edge (ours) | capacity | competition |
|---|---|---|---|---|---|
| 1 | **Pre-game sports**, quoting from listing until about 5 min before kickoff, flat at kickoff. Big leagues: soccer, NFL, MLB, NHL, NBA | the only large category with a positive edge before rewards at +30 min, CI > 0; stable spread (0.6–0.7¢); slow adverse selection (0.5¢ at 5 min) | +0.3¢/share (+64 bps) before rewards at 30 min, an upper bound for a non-top maker | $14.9M/day maker volume, about 8.6k events/week | 7.1k makers, top-5 21%; the top-20 earn +236 bps, so the edge is contested |
| 2 | **Elections (midterms) reward farming**: two-sided quotes inside `rewards_max_spread`, at least `min_size`, priced 0.15–0.85 | lowest adverse selection per share (0.5–0.6¢); largest reward pool (about $30k/day); the rewards scorer is already implemented in `analysis/rewards` | ≈ 0 before rewards; +120–130 bps with rewards (proxy is conservative ×2) | $2.2M/day maker volume | 5.3k makers, top-5 12%, HHI 67, the least concentrated |
| 3 (optional) | sports futures, geopolitics | positive only with rewards (sports futures +145 [67, 199]); geopolitics' CI straddles 0 | small | $0.6M / $2.1M per day | |

Crypto up/down is ≈ 0 on average and negative for continuous quoters; this census agrees with our study. The
remaining move there would be an "occasional maker" that quotes only on the cheap side (0.05–0.35, where the
edge is about +1.2¢). That is a directional filter, not market making.

### Main risks

- **Pre-game sports**
  - *News jumps*: lineups, injuries and weather, typically 1 h before kickoff and at team-sheet release. Our
    +2 h numbers already show positive in-game drift for the top makers; for us that is variance, not edge.
  - *Kickoff*: in-play is −123 bps for non-top makers, so quotes must be cancelled before the game starts.
    Gamma `gameStartTime` can be wrong or the game delayed; use a buffer and a live-status check (`events.live`).
  - *Edge to settlement* is ±700 bps of noise per week of data, so the strategy must not carry inventory into
    the game. A delta-flat book across the moneyline, spread and totals of one game reduces this.
  - *Queue priority*: the edge is concentrated in the top-20 wallets (+411 bps at +2 h, versus +138 bps for
    everyone else).
- **Elections**
  - The program's money can change: rates are a 2026-10-15 snapshot, and the August crypto program ended
    without notice.
  - Election-night jumps on Nov 3. Pull quotes at the first returns, or quote only far from the mid.
  - Resolution and UMA disputes, and capital lock-up until certification.
  - 169 elections markets have holding rewards enabled. These pay on held positions (reported as about 4% APY;
    not verified here), which is negligible for a market maker that turns its inventory over.
- **All categories**
  - Fee and rebate schedules differ by vertical (sports v3: 5% rate, 15% rebate; the NFL/CFB Oct-26 schedule is
    3%/15%) and change often.

### What a faithful queue-aware replay needs

- **Pre-game sports.** L2 from market listing until 30 min after kickoff (the 30 min tail measures what is
  given up when pulling quotes). Cover the top 5 leagues.
  - *Duration*: at least 3–4 weeks: 2 train, 1 val, 1 test. One week gives a ±65 bps CI at +30 min across 8.6k
    events, so 1 week of val/test separates +60 bps from 0 only marginally. Settlement-horizon P&L cannot be
    evaluated at any feasible duration (±700 bps/week), so evaluate flattened-at-kickoff P&L.
  - *Sources*: **PolyOrderBooks covers sports (and crypto) behind a key**. With a key, historical sports books
    could be backfilled at once and a replay started this week. Marketlens covers *recurring* markets since
    March (behind a key), which includes recurring sports games.
  - *Our recorder* can start now on all `sports_fees_*` markets with a `gameStartTime` in the next 48 h.
- **Elections.** L2 for the midterms markets (7,832 conditions are active; the 3,662 rewarded ones are enough)
  at reward-scorer resolution: full book snapshots at least once per minute, plus the trades.
  - *Sources*: neither vendor covers non-recurring election markets, so **only our recorder can do this, and it
    should start now**. The midterms are 19 days away, and every day before Nov 3 is unrepeatable data.
  - *Duration*: Oct 16 → Nov 4, which covers about 2.5 weeks before the election and election night.
- **Both.** The on-chain fills (`onchain/extract.py`) give keyless trade tapes for calibrating fill and markout
  models over any window. The census pipeline here runs in about 20 min per week of all-market data.

## 5. Caveats

- **The micro-price comes from trades, not the book.** In thin long-dated markets the pre-fill price can be
  stale (median age: elections 14 min, economics 7 min, culture 12 min; sports and crypto 2–7 s). That inflates both spread and adverse selection, but
  not their sum.
- **Edge is per $ filled, not per $ quoted.** A maker that quotes and is not filled earns rewards but no spread,
  and that is not observed here.
- **This is the average maker, not a strategy.** The top-20 versus everyone-else split shows the realised edge
  depends strongly on who gets filled (queue position and speed).
- **Rewards are a snapshot.** The rewards table is today's; markets that closed before Oct 15 are missing from
  it (rewards for them = 0), and the attribution is pro rata to fills.
- **Neg-risk conversions and merges are invisible as fills.** Maker legs on complementary tokens (mints) are
  handled in their own token frame.
- **One week, mid-October.** It covers the NFL, MLB playoffs, European soccer, and the run-up to the midterms.
  Category mixes are seasonal.

## Files

- `gamma_tags.py`, `rewards_current.py`, `categories.py`, `build_legs.py`, `census.py`, `top_makers.py`,
  `extras.py`, `daily.py`: the pipeline.
- `out/category_summary.csv`, `subcat_summary.csv`, `by_price.csv`, `by_ttr.csv`, `by_price_ttr.parquet`,
  `crypto_validation.csv`, `top_vs_rest.csv`, `sports_detail.csv`, `top_makers.csv` (wallets truncated),
  `top_makers_by_cat.csv`, `rewards_check.csv`, `holding.csv`, `daily_edge.csv`.
- `cache/` (git-ignored) holds `legs.parquet`, the gamma and reward caches and the per-market sums.

Reproduce (about 20 min of compute plus 12 min of RPC):

```bash
cd polymarket/onchain && nice -n 10 python extract.py --start 2026-10-08 --end 2026-10-14 --all --events OrderFilled --slim
cd ../analysis/census && python gamma_tags.py --fills ../../onchain/data/all/fills && python rewards_current.py
python build_legs.py --fills ../../onchain/data/all/fills --start 2026-10-08 --end 2026-10-14
python census.py && python top_makers.py && python extras.py && python daily.py
```
