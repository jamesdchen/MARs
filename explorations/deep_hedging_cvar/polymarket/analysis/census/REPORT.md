# Where on Polymarket is market making most promising? A maker-economics census from on-chain fills

Worker w7, written 2026-10-15. First pass: **Oct 8–14 2026 (7 full UTC days)**. Replication: **Sep 17–Oct 7**
(21 days, out of sample) and pooled **Sep 17–Oct 14** (28 days), §6. Every market on the CTF and neg-risk exchanges. No order book is needed and no API key: everything comes from complete on-chain fills plus
gamma metadata and the CLOB rewards table. Nothing from `data/build_jun15` or the v2 val/test days was touched.

## TL;DR (updated 2026-10-15 ~22:00 UTC after the 4-week replication, §6)

The first pass (Oct 8–14) was replicated out of sample on **Sep 17–Oct 7** (21 days, 76.3M fill legs), then
pooled over **Sep 17–Oct 14** (28 days, 102.0M fill legs). Edges below are per $ of maker notional (bps)
with an event-clustered bootstrap 95% CI; "ex-rewards" means spread + markout + rebate.

1. **Validation (crypto up/down) replicates in every period.**
   - The category average is a small positive: pooled +33 bps [14, 49]. Per share that is +0.17¢: spread
     1.06¢, adverse selection (AS) 1.08¢ and rebate 0.20¢, so the rebate is the whole margin.
   - The top-20 continuous quoters lose in both periods: −23 bps (Oct), −12 bps (OOS). The spread does not
     cover adverse selection for an always-on maker. This is our study's finding, and the pivot away from
     crypto stands.
2. **Pre-game sports replicates once the mark is capped at kickoff** (§6d; correction of 22:45 UTC, after w2's
   `analysis/v3_sports_rules/RECONCILE.md`).
   - **Why the earlier numbers were noisy.** The census marked a pre-game leg at t + 30 min even when that
     falls after kickoff. Legs in the last 30 min before kickoff then carry in-game price moves.
   - **Uncapped (census) results:** +64 bps [1, 130] for Oct 8–14, +13 [−35, 55] out of sample, +25 [−16, 63]
     pooled.
   - **Capped results (mark = min(t + 30 min, last pre-kickoff micro-price)), ex-rewards at +30 min:**
     - by week: **+53, +50, +49, +71 bps**, each CI above 0;
     - out of sample: **+49 [32, 64]**;
     - pooled: **+57 [34, 79]**;
     - marked to kickoff, pooled: +76 [46, 110].
   - **Time to kickoff (pooled, capped):**
     - last 30 min: **+69 [2, 127]**, and +62 to +69 in every week;
     - 30–60 min: +32 [−6, 91] (weakest);
     - 1–6 h: +49 [28, 66];
     - 6–24 h: +42 [22, 59];
     - more than 24 h: +135 [72, 218].
   - So **the "last hour ≈ 0" and the T−1 h cut were artifacts of the in-play mark.** Quoting all the way to
     kickoff is supported.
   - Caveats:
     - A fill minutes before kickoff has only minutes for adverse selection to show before its mark.
     - Flattening at kickoff costs about half a spread, unless inventory is netted across a game's markets
       (w2 §5).
   - By league (pooled, capped, +30 min), CIs above 0 for:
     - esports +60 [40, 78];
     - MLB +87 [54, 117];
     - NFL +46 [20, 78];
     - NHL +49 [14, 83];
     - tennis +132 [81, 186].
   - Soccer is +42 [−22, 94] and college football +11 [−22, 37].
3. **Election reward farming replicates.**
   - With rewards: +133 bps [94, 165] (Oct), **+101 [63, 130] (OOS)**, pooled +108 [75, 130] at +2 h and
     +164 [106, 219] at +1 d.
   - Ex-rewards it is about 0 to slightly positive: pooled +11 [−25, 31] at +2 h and +67 [1, 132] at +1 d.
     Positive on 16 of 28 days.
   - Within elections, ex-rewards is ≈ 0 at prices 0.15–0.85, which is where the rewards pay. It is +43 bps
     [18, 73] above 0.85.
   - **Caveat:** rewards for the older weeks use today's (Oct 15) rewards snapshot, so they are uncertain. The
     wallet REWARD payouts of the top makers ran at 2.0× (Oct) and 2.7× (OOS) the proxy. The proxy looks
     conservative, but its per-market split is not observable.
4. **Geopolitics: noise, not a regime** (§6e).
   - The pooled +47 bps [11, 102] comes from a handful of military-action longshot events.
   - The top 5 of about 520 events account for 70% of the edge dollars. Without them it is +18 bps; without
     the top 10, +1 bp.
   - Weekly figures: +84, +23, +88, +12. The bulk of volume, priced above 0.85 ($1.2M/day), is −2 [−26, 18].
5. **Avoid (replicated):**
   - weather: pooled ex-rewards at +2 h −64 bps [−79, −48]; 2 of 28 days positive;
   - economics: −124 [−228, −100];
   - tech/AI: −306 [−506, −134];
   - culture: −186 [−250, −127];
   - politics (non-election): −167 [−434, −19];
   - in-play sports: −18 [−65, 26]. Non-top makers lose −52 bps, and only the top-20 earn there (+43 bps).
   - Sports futures is negative before rewards (−57 [−148, −6] at +2 h) and positive with rewards.

The first-pass TL;DR for Oct 8–14 alone is kept in §4a for reference.

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

## 4. Recommendation (first pass, Oct 8–14; see §6c for the update)

### 4a. First-pass TL;DR (Oct 8–14 only; superseded by §6)

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


### 4b. First-pass recommendation

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

## 6. Out-of-sample replication (Sep 17–Oct 7) and pooled results (Sep 17–Oct 14)

**Data.** The fill pull was extended back to Sep 17 with `extract.py --all --events OrderFilled --slim`, using 8
segment workers and 0 failed segments: 21 more days, 1.21M blocks and 76.3M fill legs. Gamma lookups grew
to 757,968 tokens, of which 757,635 resolved (99.96%).

**Method changes.** The pipeline is unchanged except for one thing: legs are now built per week (`build_legs.py
--tag w1..w4`) with a 1-day pad of later fills, so the +2 h and +1 d marks are not truncated at week boundaries.
- Marks are capped at the last fill actually loaded. The Oct 8–14 rerun (`w4`) reproduces the first pass
  exactly.
- `run_periods.sh` runs all three periods. `period.py` selects the period through environment variables.
- `boot_buckets.py` adds event-bootstrap CIs for sub-cells.

**Rewards caveat.** The reward proxy uses the CLOB rewards table of **2026-10-15** for every week, so it is
uncertain for older weeks. Markets that closed before Oct 15 are missing from that table and get 0. As a
check, the top-20 makers' data-api REWARD payouts were **2.0× (Oct), 2.7× (OOS) and 2.4× (pooled)** the proxy for
the same wallets, with correlation 0.62–0.75. With-rewards edges are therefore probably understated in level,
but the split across categories is a modelling assumption.

### 6a. Category edge by period (bps of maker notional, event-bootstrap 95% CI)

`out/`, `out_oos/`, `out_pooled/` → `category_summary.csv`, `daily_edge.csv`

| category            | period             | maker $/day   | ex-rew +30m       | ex-rew +2h        | ex-rew +1d        | with rew +2h    | with rew +1d     | days >0 (+2h ex-rew)   |
|:--------------------|:-------------------|:--------------|:------------------|:------------------|:------------------|:----------------|:-----------------|:-----------------------|
| sports_pregame      | Oct 8–14           | 14.94M        | 64 [1, 130]       | 246 [-108, 627]   | 118 [-580, 800]   | 250 [-107, 629] | 124 [-579, 801]  | 5/7                    |
|                     | Sep 17–Oct 7 (OOS) | 16.65M        | 13 [-35, 55]      | -15 [-138, 114]   | 22 [-269, 324]    | -12 [-138, 114] | 24 [-269, 324]   | 6/21                   |
|                     | pooled             | 16.23M        | 25 [-16, 63]      | 44 [-92, 175]     | 43 [-265, 335]    | 47 [-90, 175]   | 47 [-265, 335]   | 11/28                  |
| elections           | Oct 8–14           | 2.17M         | 11 [-27, 38]      | -8 [-51, 18]      | -20 [-59, 10]     | 133 [94, 165]   | 122 [74, 156]    | 2/7                    |
|                     | Sep 17–Oct 7 (OOS) | 2.75M         | 22 [-10, 39]      | 16 [-29, 39]      | 87 [9, 162]       | 101 [63, 130]   | 173 [100, 242]   | 14/21                  |
|                     | pooled             | 2.60M         | 20 [-9, 37]       | 11 [-25, 31]      | 67 [1, 132]       | 108 [75, 130]   | 164 [106, 219]   | 16/28                  |
| geopolitics         | Oct 8–14           | 2.06M         | 10 [-92, 86]      | -1 [-151, 94]     | 63 [-239, 256]    | 56 [-85, 154]   | 115 [-169, 296]  | 5/7                    |
|                     | Sep 17–Oct 7 (OOS) | 1.68M         | 63 [19, 131]      | 55 [7, 125]       | 44 [-53, 135]     | 110 [55, 179]   | 99 [-4, 189]     | 15/21                  |
|                     | pooled             | 1.78M         | 47 [11, 102]      | 39 [-5, 96]       | 48 [-50, 135]     | 94 [46, 146]    | 101 [-4, 179]    | 20/28                  |
| sports_futures      | Oct 8–14           | 0.58M         | 3 [-50, 37]       | -22 [-96, 28]     | -14 [-136, 72]    | 145 [67, 199]   | 155 [43, 207]    | 4/7                    |
|                     | Sep 17–Oct 7 (OOS) | 0.50M         | -10 [-84, 30]     | -71 [-189, -9]    | -154 [-300, -71]  | 73 [-40, 126]   | -10 [-151, 54]   | 5/21                   |
|                     | pooled             | 0.52M         | -7 [-66, 27]      | -57 [-148, -6]    | -120 [-241, -50]  | 93 [6, 139]     | 32 [-86, 72]     | 9/28                   |
| finance_prices      | Oct 8–14           | 0.62M         | -81 [-137, -43]   | -87 [-148, -40]   | -100 [-223, -7]   | 117 [35, 160]   | 101 [-54, 169]   | 0/7                    |
|                     | Sep 17–Oct 7 (OOS) | 0.85M         | -29 [-109, 60]    | -57 [-142, 32]    | -63 [-144, 17]    | 6 [-90, 92]     | -1 [-89, 70]     | 7/21                   |
|                     | pooled             | 0.79M         | -39 [-109, 28]    | -62 [-135, 12]    | -69 [-146, 8]     | 27 [-56, 87]    | 18 [-69, 76]     | 7/28                   |
| crypto_updown_short | Oct 8–14           | 9.52M         | 30 [-7, 69]       | 30 [-6, 69]       | 28 [-8, 68]       | 31 [-6, 69]     | 29 [-8, 68]      | 6/7                    |
|                     | Sep 17–Oct 7 (OOS) | 9.77M         | 34 [13, 53]       | 34 [13, 53]       | 33 [13, 53]       | 35 [13, 53]     | 34 [13, 53]      | 16/21                  |
|                     | pooled             | 9.71M         | 33 [14, 49]       | 33 [15, 49]       | 32 [14, 48]       | 34 [15, 49]     | 33 [14, 48]      | 22/28                  |
| crypto_other        | Oct 8–14           | 2.43M         | 9 [-18, 32]       | 3 [-27, 30]       | 25 [-38, 82]      | 14 [-19, 39]    | 36 [-31, 89]     | 4/7                    |
|                     | Sep 17–Oct 7 (OOS) | 3.39M         | 29 [12, 45]       | 7 [-20, 31]       | 9 [-30, 44]       | 14 [-15, 35]    | 17 [-26, 49]     | 14/21                  |
|                     | pooled             | 3.15M         | 25 [8, 41]        | 6 [-18, 25]       | 12 [-21, 40]      | 14 [-13, 31]    | 20 [-17, 46]     | 18/28                  |
| sports_inplay       | Oct 8–14           | 37.56M        | -38 [-101, 26]    | -40 [-134, 51]    | -76 [-187, 24]    | -39 [-134, 51]  | -75 [-187, 24]   | 2/7                    |
|                     | Sep 17–Oct 7 (OOS) | 34.31M        | -3 [-34, 28]      | -10 [-60, 40]     | -14 [-66, 37]     | -10 [-60, 40]   | -14 [-66, 37]    | 11/21                  |
|                     | pooled             | 35.12M        | -12 [-41, 16]     | -18 [-65, 26]     | -31 [-82, 18]     | -17 [-65, 26]   | -30 [-82, 18]    | 13/28                  |
| politics            | Oct 8–14           | 0.98M         | -37 [-141, 39]    | -71 [-174, 4]     | -72 [-185, -5]    | -26 [-125, 45]  | -26 [-129, 26]   | 1/7                    |
|                     | Sep 17–Oct 7 (OOS) | 0.67M         | -86 [-218, -3]    | -202 [-518, -16]  | -345 [-930, -29]  | -173 [-486, 1]  | -317 [-898, -10] | 6/21                   |
|                     | pooled             | 0.75M         | -76 [-209, 1]     | -167 [-434, -19]  | -273 [-708, -33]  | -130 [-396, 5]  | -235 [-672, -10] | 7/28                   |
| weather             | Oct 8–14           | 1.13M         | -51 [-70, -31]    | -87 [-115, -56]   | -77 [-121, -27]   | -56 [-91, -32]  | -47 [-110, -17]  | 0/7                    |
|                     | Sep 17–Oct 7 (OOS) | 1.20M         | -24 [-39, -10]    | -57 [-74, -40]    | -71 [-98, -45]    | -46 [-69, -33]  | -60 [-93, -37]   | 2/21                   |
|                     | pooled             | 1.19M         | -31 [-44, -18]    | -64 [-79, -48]    | -72 [-95, -47]    | -48 [-70, -38]  | -57 [-88, -40]   | 2/28                   |
| economics           | Oct 8–14           | 0.84M         | -210 [-260, -71]  | -128 [-326, -94]  | -158 [-494, -83]  | -41 [-118, 83]  | -77 [-302, 102]  | 0/7                    |
|                     | Sep 17–Oct 7 (OOS) | 0.61M         | -94 [-168, -73]   | -122 [-216, -96]  | -202 [-293, -132] | -24 [-73, 20]   | -103 [-154, -10] | 1/21                   |
|                     | pooled             | 0.67M         | -130 [-177, -84]  | -124 [-228, -100] | -195 [-306, -140] | -29 [-66, 25]   | -103 [-151, -5]  | 1/28                   |
| tech_ai             | Oct 8–14           | 0.19M         | -188 [-305, -112] | -205 [-342, -118] | -211 [-427, -73]  | -17 [-122, 55]  | -33 [-214, 57]   | 0/7                    |
|                     | Sep 17–Oct 7 (OOS) | 0.20M         | -263 [-483, -63]  | -339 [-581, -116] | -334 [-578, -90]  | -218 [-469, -8] | -213 [-449, 12]  | 1/21                   |
|                     | pooled             | 0.20M         | -244 [-419, -94]  | -306 [-506, -134] | -307 [-500, -115] | -169 [-381, -8] | -171 [-385, 7]   | 1/28                   |
| culture             | Oct 8–14           | 0.15M         | -90 [-193, 24]    | -215 [-339, -89]  | -237 [-463, -28]  | -10 [-172, 80]  | -62 [-304, 115]  | 1/7                    |
|                     | Sep 17–Oct 7 (OOS) | 0.20M         | -119 [-173, -74]  | -178 [-245, -118] | -179 [-333, -6]   | -79 [-163, -39] | -80 [-245, 77]   | 0/21                   |
|                     | pooled             | 0.19M         | -114 [-160, -72]  | -186 [-250, -127] | -190 [-317, -32]  | -66 [-143, -35] | -73 [-223, 58]   | 1/28                   |

### 6b. Pre-game sports by time to kickoff and league, and elections/geopolitics by price

Ex-rewards, +30 min unless noted (`*/boot_cells.csv`, `*/sports_league_kickoff.csv`). The +2 h mark crosses
kickoff for fills inside the last 2 h, so for those cells it measures in-game noise.

| cell                   | Oct 8–14 +30m     | Sep 17–Oct 7 (OOS) +30m   | pooled +30m     | pooled +2h        | pooled maker $/day   |   pooled events |
|:-----------------------|:------------------|:--------------------------|:----------------|:------------------|:---------------------|----------------:|
| pregame a<1h           | 64 [-54, 209]     | -32 [-143, 66]            | -8 [-95, 77]    | 86 [-193, 376]    | 7179k                |           18272 |
| pregame b1-6h          | 30 [-9, 62]       | 53 [31, 74]               | 49 [30, 66]     | -23 [-169, 77]    | 5204k                |           18738 |
| pregame c6-24h         | 41 [7, 75]        | 43 [17, 64]               | 42 [20, 60]     | 35 [6, 66]        | 2860k                |           17386 |
| pregame d>24h          | 330 [73, 665]     | 81 [55, 114]              | 135 [71, 220]   | 157 [69, 276]     | 982k                 |           10805 |
| pregame soccer c6-24h  | 38 [26, 51]       | 52 [31, 69]               | 48 [32, 61]     | 15 [-5, 31]       | 920k                 |            8558 |
| pregame soccer d>24h   | 67 [53, 81]       | 154 [105, 219]            | 136 [93, 186]   | 140 [90, 189]     | 302k                 |            7328 |
| pregame esports c6-24h | 66 [49, 81]       | 48 [33, 64]               | 52 [40, 64]     | 30 [15, 46]       | 480k                 |            1874 |
| pregame esports d>24h  | 113 [40, 196]     | 74 [50, 103]              | 86 [57, 118]    | 67 [32, 99]       | 57k                  |             598 |
| pregame nfl b1-6h      | 33 [-8, 74]       | 52 [16, 99]               | 50 [18, 91]     | 73 [-73, 228]     | 1373k                |             131 |
| pregame nfl c6-24h     | 19 [-28, 91]      | 28 [1, 67]                | 27 [0, 65]      | 31 [-0, 71]       | 657k                 |             127 |
| pregame nfl d>24h      | 480 [55, 1290]    | 15 [-6, 35]               | 123 [19, 302]   | 128 [7, 332]      | 381k                 |             170 |
| pregame mlb b1-6h      | 97 [70, 124]      | 67 [21, 102]              | 69 [26, 100]    | 87 [-39, 204]     | 503k                 |             432 |
| pregame mlb c6-24h     | 55 [20, 98]       | 41 [14, 65]               | 43 [20, 66]     | 34 [-10, 83]      | 182k                 |             420 |
| pregame nhl b1-6h      | 60 [18, 105]      | 90 [64, 116]              | 77 [53, 104]    | -42 [-183, 89]    | 82k                  |             169 |
| pregame cfb b1-6h      | 52 [19, 85]       | 16 [-1, 32]               | 23 [9, 37]      | -33 [-121, 55]    | 428k                 |             457 |
| pregame tennis b1-6h   | -197 [-576, 75]   | 258 [175, 344]            | 155 [36, 254]   | 268 [91, 412]     | 359k                 |            4933 |
| pregame tennis c6-24h  | -106 [-367, 105]  | 185 [73, 311]             | 124 [13, 238]   | 229 [13, 535]     | 236k                 |            4243 |
| pregame soccer a<1h    | 109 [-37, 286]    | -60 [-339, 180]           | -0 [-188, 184]  | 3 [-646, 644]     | 2738k                |            9587 |
| pregame nfl a<1h       | 24 [-391, 313]    | 28 [-166, 186]            | 28 [-135, 182]  | 323 [-124, 839]   | 1428k                |             131 |
| pregame mlb a<1h       | -205 [-1585, 500] | -188 [-552, 85]           | -189 [-504, 71] | -232 [-1005, 570] | 439k                 |             447 |
| elections p<.15        | 299 [175, 394]    | 315 [-100, 570]           | 309 [38, 482]   | -10 [-536, 276]   | 101k                 |            1998 |
| elections p.15-.85     | -31 [-78, 3]      | -13 [-76, 29]             | -17 [-70, 19]   | -29 [-91, 8]      | 1344k                |            1490 |
| elections p>.85        | 16 [-3, 36]       | 50 [19, 85]               | 43 [18, 73]     | 65 [22, 112]      | 1159k                |            1839 |
| geopolitics p.15-.85   | 12 [-180, 148]    | 89 [10, 226]              | 66 [-3, 166]    | 51 [-41, 166]     | 557k                 |             332 |
| geopolitics p>.85      | -36 [-127, 5]     | 12 [-1, 29]               | -2 [-27, 17]    | 4 [-24, 25]       | 1167k                |             490 |

How to read it:
- **The final hour before kickoff carries no robust edge** in any league. That hour is when lineups land and
  sharp money arrives; the CIs are wide and centred near 0, and MLB is −189 bps.
- **Earlier windows replicate.** They have wider spreads (0.9–1.7¢ beyond 6 h versus 0.65¢ in the last hour)
  and adverse selection that does not keep up. Soccer is the largest addressable cell: $1.2M/day at > 6 h,
  +48 to +136 bps.
- **Tennis** is positive out of sample but negative in Oct 8–14, so it is unstable.
- **Pre-game top-20 makers** earn +84 bps at +2 h pooled (+411 in Oct, −48 OOS) against +21 bps for everyone
  else. The Oct week was a good week.

### 6c. What changes in the recommendation (D11)

- **Keep pre-game sports, from listing to kickoff.** This is superseded by §6d. An earlier draft said to stop at
  T−1 h; that came from the in-play mark artifact.
  - With kickoff-capped marks, the whole pre-game window is about +57 bps [34, 79] of filled notional (ex
    rewards, pooled).
  - The last 30 min is +69 [2, 127]: positive every week, but its marks are short.
  - Cancel at kickoff and stay flat in-play. Flatten cost at kickoff is the open question for the replay.
  - The replay must test whether *our* quotes, which are always on, get the average fill. In crypto the
    top-20 continuous quoters did worse than average. In pre-game sports they did better (+84 vs +21 bps).
- **Keep elections reward farming.** The with-rewards edge replicates (+101 OOS), and adverse selection is the
  lowest anywhere (0.5–0.6¢).
  - The money is the reward program. Quote 0.15–0.85, two-sided, inside `rewards_max_spread`.
  - The risk is the program changing, and Nov 3. Pull quotes at the first returns.
- **Geopolitics:** event-driven noise, not a regime (§6e). Not a candidate.
- **Data needs** are unchanged from §4. For pre-game, record from market listing to T−1 h, plus the last
  hour, since the last 30 min is the best bucket on capped marks. PolyOrderBooks (sports, keyed) can backfill. Elections
  need our recorder, starting now.

### 6d. Pre-game sports with marks capped at kickoff (added 22:45 UTC)

`kickoff_cap.py` → `out*/sports_kickoff_capped.csv` and `out_pooled/sports_kickoff_capped_all_periods.csv`.

The mark is the micro-price at min(t + h, kickoff−). Kickoff− means the micro-price after the last taker print
strictly before `gameStartTime`. Markets are joined only when they have a pre-kickoff print, so event counts
are a little lower than in §6b. Values are ex-rewards, bps of maker notional, event-bootstrap 95% CI. This
follows w2's `analysis/v3_sports_rules/RECONCILE.md`, which found the same artifact and the same capped
numbers on this leg definition (+50/+49/+71 for Sep 24–30, Oct 1–7, Oct 8–14).

| period             | time to kickoff   |   events | uncapped +30m (census)   | capped +30m    | capped +2h    | to kickoff     |
|:-------------------|:------------------|---------:|:-------------------------|:---------------|:--------------|:---------------|
| Sep 17–Oct 7 (OOS) | all pre-game      |    15811 | 16 [-32, 56]             | 49 [32, 64]    | 56 [33, 83]   | 69 [44, 97]    |
| Sep 17–Oct 7 (OOS) | <30m              |    10639 | -49 [-213, 81]           | 62 [18, 99]    | 62 [18, 99]   | 62 [18, 99]    |
| Sep 17–Oct 7 (OOS) | 30-60m            |     8994 | 5 [-16, 26]              | 5 [-16, 26]    | 7 [-21, 35]   | 7 [-21, 35]    |
| Sep 17–Oct 7 (OOS) | 1-6h              |    11965 | 53 [31, 74]              | 53 [31, 74]    | 72 [29, 130]  | 88 [42, 149]   |
| Sep 17–Oct 7 (OOS) | 6-24h             |    10344 | 43 [18, 65]              | 43 [18, 65]    | 41 [7, 81]    | 64 [24, 104]   |
| Sep 17–Oct 7 (OOS) | >24h              |     5863 | 81 [54, 112]             | 81 [54, 112]   | 87 [49, 136]  | 146 [74, 221]  |
| pooled             | all pre-game      |    21587 | 27 [-10, 63]             | 57 [34, 79]    | 62 [35, 92]   | 76 [46, 110]   |
| pooled             | <30m              |    15132 | -27 [-157, 82]           | 69 [2, 128]    | 69 [2, 128]   | 69 [2, 128]    |
| pooled             | 30-60m            |    12655 | 32 [-6, 91]              | 32 [-6, 91]    | 35 [-21, 102] | 35 [-21, 102]  |
| pooled             | 1-6h              |    16576 | 49 [28, 66]              | 49 [28, 66]    | 66 [25, 113]  | 87 [40, 139]   |
| pooled             | 6-24h             |    14459 | 42 [22, 59]              | 42 [22, 59]    | 35 [9, 64]    | 52 [1, 104]    |
| pooled             | >24h              |     7825 | 135 [72, 218]            | 135 [72, 218]  | 157 [71, 270] | 225 [108, 355] |
| Sep 17–23          | all pre-game      |     6864 | 22 [-23, 65]             | 53 [36, 72]    | 46 [25, 66]   | 42 [17, 66]    |
| Sep 17–23          | <30m              |     4110 | -50 [-198, 82]           | 65 [37, 97]    | 65 [37, 97]   | 65 [37, 97]    |
| Sep 17–23          | 30-60m            |     3462 | 10 [-8, 30]              | 10 [-8, 30]    | -13 [-41, 19] | -13 [-41, 19]  |
| Sep 17–23          | 1-6h              |     4788 | 81 [55, 112]             | 81 [55, 112]   | 79 [43, 119]  | 76 [35, 120]   |
| Sep 17–23          | 6-24h             |     4275 | 8 [-61, 51]              | 8 [-61, 51]    | -16 [-86, 31] | -12 [-89, 54]  |
| Sep 17–23          | >24h              |     2638 | 50 [13, 98]              | 50 [13, 98]    | 69 [-9, 163]  | -28 [-125, 56] |
| Sep 24–30          | all pre-game      |     5347 | 52 [6, 94]               | 50 [28, 74]    | 40 [12, 66]   | 48 [16, 74]    |
| Sep 24–30          | <30m              |     3300 | 77 [-92, 250]            | 69 [32, 112]   | 69 [32, 112]  | 69 [32, 112]   |
| Sep 24–30          | 30-60m            |     2829 | 27 [-16, 62]             | 27 [-16, 62]   | 5 [-56, 49]   | 5 [-56, 49]    |
| Sep 24–30          | 1-6h              |     3714 | 31 [-17, 76]             | 31 [-17, 76]   | 13 [-43, 63]  | 27 [-43, 73]   |
| Sep 24–30          | 6-24h             |     3270 | 60 [29, 102]             | 60 [29, 102]   | 47 [15, 87]   | 71 [28, 118]   |
| Sep 24–30          | >24h              |     2053 | 68 [30, 117]             | 68 [30, 117]   | 79 [31, 150]  | 67 [-25, 166]  |
| Oct 1–7            | all pre-game      |     5467 | -27 [-144, 77]           | 49 [33, 67]    | 50 [27, 78]   | 63 [39, 90]    |
| Oct 1–7            | <30m              |     3231 | -158 [-506, 192]         | 63 [26, 98]    | 63 [26, 98]   | 63 [26, 98]    |
| Oct 1–7            | 30-60m            |     2704 | -26 [-89, 23]            | -26 [-89, 23]  | -1 [-84, 72]  | -1 [-84, 72]   |
| Oct 1–7            | 1-6h              |     3522 | 43 [22, 66]              | 43 [22, 66]    | 23 [-14, 61]  | 24 [-23, 70]   |
| Oct 1–7            | 6-24h             |     3088 | 56 [35, 83]              | 56 [35, 83]    | 84 [24, 179]  | 71 [7, 161]    |
| Oct 1–7            | >24h              |     2299 | 120 [75, 183]            | 120 [75, 183]  | 111 [59, 172] | 362 [202, 566] |
| Oct 8–14 (=w4)     | all pre-game      |     7010 | 66 [2, 132]              | 71 [36, 120]   | 57 [17, 112]  | 33 [-8, 84]    |
| Oct 8–14 (=w4)     | <30m              |     4493 | 45 [-118, 215]           | 62 [31, 95]    | 62 [31, 95]   | 62 [31, 95]    |
| Oct 8–14 (=w4)     | 30-60m            |     3661 | 100 [-22, 306]           | 100 [-22, 306] | 78 [-42, 281] | 78 [-42, 281]  |
| Oct 8–14 (=w4)     | 1-6h              |     4636 | 30 [-14, 61]             | 30 [-14, 61]   | -4 [-52, 34]  | -32 [-104, 25] |
| Oct 8–14 (=w4)     | 6-24h             |     4224 | 41 [7, 75]               | 41 [7, 75]     | 14 [-25, 56]  | -66 [-139, 9]  |
| Oct 8–14 (=w4)     | >24h              |     2711 | 330 [74, 657]            | 330 [74, 657]  | 411 [61, 893] | 361 [37, 794]  |

By league (pooled, maker volume > $100k/day):

| league     |   events | maker $/day   | uncapped +30m   | capped +30m    | to kickoff      |
|:-----------|---------:|:--------------|:----------------|:---------------|:----------------|
| soccer     |    10553 | 5332k         | 24 [-84, 107]   | 42 [-22, 94]   | 87 [-1, 181]    |
| nfl        |      163 | 3806k         | 45 [-25, 102]   | 46 [20, 78]    | 70 [27, 113]    |
| esports    |     2277 | 2579k         | 16 [-26, 60]    | 60 [40, 78]    | 40 [-14, 82]    |
| cfb        |      542 | 1298k         | -26 [-107, 51]  | 11 [-22, 37]   | 34 [-11, 75]    |
| mlb        |      475 | 1122k         | -36 [-165, 70]  | 87 [54, 117]   | 88 [47, 127]    |
| tennis     |     5360 | 931k          | 114 [10, 217]   | 132 [81, 186]  | 218 [154, 284]  |
| nhl        |      195 | 313k          | -33 [-270, 139] | 49 [14, 83]    | 39 [-9, 90]     |
| other      |      785 | 231k          | 220 [-245, 777] | 286 [-90, 843] | 129 [-267, 658] |
| basketball |      557 | 173k          | -39 [-136, 60]  | 40 [-11, 90]   | 3 [-101, 80]    |
| nba        |       68 | 123k          | 190 [-20, 498]  | 166 [-28, 448] | 221 [-124, 693] |

How to read it:
- **Uncapped, the last 30 min swings between −158 and +77 bps by week.** That is in-game noise.
- **Capped, the last 30 min is +62 to +69 bps in every week.** It is the most consistent bucket.
- **The window more than 24 h before kickoff stays the strongest per $**, but it is small: $0.9M/day.
- **The earlier "T−1 h" cut (D14) is not supported.** The 30–60 min bucket is the weakest (+32 [−6, 91]
  pooled), but the last 30 min is strong.
- Two caveats keep this from being a free lunch:
  - Fills in the final minutes are marked after only minutes, so slow adverse selection is under-measured.
  - Getting flat at kickoff costs about half a spread, unless the game's moneyline, spread and totals
    inventory nets out.

The replay should measure P&L to kickoff, including the flatten cost.

### 6e. Geopolitics: regime or noise? (added 22:45 UTC)

`geo_note.py` → `out_pooled/geo_note.csv`. Values are ex-rewards, bps, event-bootstrap 95% CI.

| cut            | key      |   events | maker $/day   | +30m ex-rew      | +2h ex-rew      |   top-5 events share of edge $ |   +30m ex top-5 events |   ex top-10 |
|:---------------|:---------|---------:|:--------------|:-----------------|:----------------|-------------------------------:|-----------------------:|------------:|
| week           | w1       |      440 | 2054k         | 84 [-3, 213]     | 56 [-45, 198]   |                           1.01 |                     -1 |         -15 |
| week           | w2       |      427 | 1456k         | 23 [-16, 84]     | 19 [-29, 82]    |                           1.49 |                    -14 |         -23 |
| week           | w3       |      449 | 1535k         | 88 [25, 193]     | 105 [38, 210]   |                           0.88 |                     13 |         -20 |
| week           | w4       |      437 | 2065k         | 12 [-92, 91]     | 1 [-148, 103]   |                           3.75 |                    -46 |         -68 |
| price (pooled) | p.15-.85 |      332 | 557k          | 66 [-10, 174]    | 51 [-36, 171]   |                           1.09 |                     -8 |         -58 |
| price (pooled) | p<.15    |      503 | 54k           | 1070 [461, 2210] | 824 [355, 1580] |                           0.62 |                    413 |         202 |
| price (pooled) | p>.85    |      490 | 1167k         | -2 [-25, 18]     | 4 [-18, 24]     |                          -4.22 |                    -12 |         -18 |
| pooled         | all      |      523 | 1777k         | 51 [16, 108]     | 43 [2, 100]     |                           0.66 |                     18 |           1 |

**Verdict: event-driven noise, not a regime.**
- **Few events.** The pooled +47–52 bps comes from a handful of jumpy military-action markets (Saudi/Yemen,
  Pakistan/Afghanistan, Israel/Lebanon). The top 5 events account for 70% of the edge dollars. Excluding them
  leaves +18 bps, and excluding the top 10 leaves +1.
- **Cheap longshots.** Positive cells sit at prices below 0.15 (+1,070 bps at +30 min on only $54k/day). The
  same cell's resolution markout is strongly negative.
- **Bulk volume is flat.** The bulk, priced above 0.85 ($1.2M/day), is −2 [−26, 18].
- **Week by week**, it is +84, +23, +88, +12; only Oct 1–7 has a CI above 0. The Oct 8–14 "≈ 0" is the typical
  week once the event draw is quiet.

Not a track.

## Files

- `gamma_tags.py`, `rewards_current.py`, `categories.py`, `build_legs.py`, `census.py`, `top_makers.py`,
  `extras.py`, `daily.py`: the pipeline.
- `out/category_summary.csv`, `subcat_summary.csv`, `by_price.csv`, `by_ttr.csv`, `by_price_ttr.parquet`,
  `crypto_validation.csv`, `top_vs_rest.csv`, `sports_detail.csv`, `top_makers.csv` (wallets truncated),
  `top_makers_by_cat.csv`, `rewards_check.csv`, `holding.csv`, `daily_edge.csv`.
- `cache/` (git-ignored) holds `legs*.parquet`, the gamma and reward caches and the per-market sums.
- Replication: `period.py`, `run_periods.sh`, `boot_buckets.py`, `tables.py`. Follow-ups: `kickoff_cap.py`
  (§6d), `geo_note.py` (§6e), `election_rewards.py` (§7). Outputs for the three periods are
  `out/` (Oct 8–14), `out_oos/` (Sep 17–Oct 7) and `out_pooled/` (Sep 17–Oct 14): the same files as above,
  plus `boot_cells.csv`, `sports_league_kickoff.csv` and `out_pooled/replication.md`.

Reproduce (about 20 min of compute plus 12 min of RPC):

```bash
cd polymarket/onchain && nice -n 10 python extract.py --start 2026-10-08 --end 2026-10-14 --all --events OrderFilled --slim
cd ../analysis/census && python gamma_tags.py --fills ../../onchain/data/all/fills && python rewards_current.py
python build_legs.py --fills ../../onchain/data/all/fills --start 2026-10-08 --end 2026-10-14
python census.py && python top_makers.py && python extras.py && python daily.py
# replication: extend the pull to Sep 17, re-run gamma, then
cd ../../onchain && python extract.py --start 2026-09-17 --end 2026-10-07 --all --events OrderFilled --slim --workers 8
cd ../analysis/census && python gamma_tags.py --fills ../../onchain/data/all/fills && REBUILD=1 ./run_periods.sh
for p in "w4 out" "w1,w2,w3 out_oos" "w1,w2,w3,w4 out_pooled"; do set -- $p; CENSUS_TAGS=$1 CENSUS_OUT=$2 python boot_buckets.py; done
python tables.py
```
