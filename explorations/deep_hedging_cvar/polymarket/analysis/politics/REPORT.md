# Politics / economics / geopolitics market making: a book-free feasibility study (w6)

Written 2026-10-15 by worker w6. The question: could politics, Fed and geopolitics markets work better for a
maker than crypto up/down? The crypto up/down study (`../../final/RESULTS.md`, `../fair_value/REPORT.md`,
`../rewards/REPORT.md`) found no edge there.

The data:
- **On-chain fills.** Every `OrderFilled` for 60 markets (120 tokens) over **44 days, 2026-09-01 → 10-14**:
  597k maker legs, 330M shares, $139M maker notional. Source: keyless Polygon archive RPCs via
  `../../onchain/extract.py`.
- **Prices.** CLOB `prices-history`, plus gamma metadata and resolutions.
- **Rewards.** CLOB `/rewards/markets/current`, plus the data-api `/activity` payout rows of the top makers.
- **Live books.** Live L2 books of the 40 active markets, recorded with `polyrec` from 19:34 UTC Oct 15 until
  this session ends.
- **Crypto comparison.** The **same** analysis code on on-chain crypto up/down fills from two samples:
  - **train days**, Aug 21–22: the 401 markets of `data/raw`, 252k maker legs, with recorded L2 for
    validation;
  - **Aug 15–20**, before the train days: a deterministic third of all markets of the same 7 coins, 5.7k
    markets, 3.1M maker legs.

The val and test days (Aug 23–24) and `data/build_jun15` were never touched. Everything is reproducible from
the scripts in this folder (see "Files" at the end).

## TL;DR

1. **Politics market making is more promising than crypto up/down, but the gap is modest and comes mainly from
   rewards, not spread.**
   - Size-weighted over all makers, a politics maker fill is worth **+0.23 ¢/share (+54 bps of notional) after
     5 min, rebate included** (95% CI +0.18 to +0.28 ¢, clustered by market × day). The figure is the same at
     2 h: +0.23 ¢ (+55 bps).
   - The same calculation on crypto up/down gives **+0.01 ¢ (+2 bps), CI −0.18 to +0.20** over Aug 15–20, and
     −0.33 ¢ (−63 bps, CI −1.10 to +0.46) on the train days.
   - So crypto makers are break-even at best after the 20% rebate. Politics makers keep about **0.2 ¢/share
     (≈ 50 bps of notional) more** per fill.
   - On top of that, politics has **large liquidity-reward pools**, which crypto up/down no longer has (§1, §5).
2. **Why the spread survives in politics.**
   - Politics takers are mostly uninformed at minute-to-hour horizons. 5-min adverse selection is 0.33 ¢
     (edge 0.47 ¢ → markout +0.14 ¢).
   - In crypto, takers trading on Binance moves erase a 1.6 ¢ at-fill edge within 5 min (adverse selection
     1.8 ¢), leaving −0.19 ¢.
   - The politics edge is concentrated in **longshot prices (< 0.2 or > 0.8)**. Contested markets (0.2–0.8)
     are break-even to negative at 30 min to 2 h.
3. **Revenue split, all makers in the 60 markets, per day:**
   - Spread net of 5-min adverse selection: **$10.7k**.
   - Maker rebates: **$6.4k** (25% of the 4–5% politics/economics taker fee; geopolitics has no fee).
   - Current liquidity-reward pools: **$11.3k** (on 27 of the 60 markets).
   - That is **38 : 23 : 40**. Rewards are the largest single source and the clearest difference from crypto
     up/down: a $0 pool today, against $1M a month in August.
4. **Are the professional makers profitable? Mostly, but the money is concentrated and partly directional.**
   - The top 25 makers by volume made **+$485k over 44 days before rewards** on these 60 markets. Only 14 of
     the 25 are profitable; the median is +$1.1k. Three wallets make +$517k of it (+$275k, +$195k, +$48k).
     The other 22 together lose $32k before rewards and make about +$104k with their estimated rewards.
   - They were paid $249k in REWARDS (all their markets); about $139k of that is attributable to this universe.
   - The breakdown:
     - spread at fill +$349k;
     - adverse selection to 2 h −$234k;
     - **inventory drift after 2 h +$621k**;
     - taker legs −$430k and taker fees −$141k;
     - rebates +$85k.
   - So professional politics market making is a **spread + rebates + rewards business that a few players
     overlay with large directional positions**. The crypto top 25 over Aug 15–20 (one third of markets) made
     +$62k in 6 days: spread +$174k, rebates +$31k, direction −$168k, taker legs +$78k.
5. **Main risk: jumps.**
   - Politics prices sit still and then jump. For the median politics market, essentially all 10-min price
     variation comes from its largest 1% of moves (36% geopolitics, 48% economics).
   - Moves of 5 ¢ or more in 10 min occur 0.017 (politics), 0.12 (geopolitics) and 0.14 (economics) times per
     market-day. The largest seen was 48.5 ¢.
   - Geopolitics fills followed by a jump lose (−0.23 ¢ at 2 h, against +0.23 ¢ otherwise). In politics and
     economics, makers active before a jump were mostly on the right side in this sample, but the tail is
     under-sampled.
   - Resolution: makers who held to resolution in the 20 resolved markets earned +2.0 ¢/share (CI +1.1 to
     +3.0). That is mostly makers selling longshots that lost; a different set of resolutions could flip it.
     Treat it as inventory risk, not as edge.
6. **Live books are a different regime from crypto.**
   - The politics touch is **one tick wide 100% of the time**, and its price changes **0–1.3 times per hour**
     (crypto: 160–550).
   - Touch depth is $0.8–3k per side (crypto $25–57), and depth within the reward band is $40–133k.
   - A replay must therefore model the **queue**, not the price. The public websocket has no L3: its
     `price_change.hash` is not the order hash.
   - Recommendation: record **≥ 4 weeks** (Oct 16 → Nov 13) covering the Oct 25 Brazil runoff, the Oct 28
     FOMC and the Nov 3 midterms, and snapshot the reward pools daily. The C env needs plumbing, not a
     redesign (§7).

**Caveats:**
- The book-free spread proxy is biased in fast markets: it is +1.3 ¢ too high on crypto when measured against
  the recorded L2 mid, and the bias sits in the price tails. Markouts are robust: the crypto proxy markouts
  match the L2 markouts within about 0.1 ¢. On politics the proxy agrees with the recorded L2 mid: 0.39 vs 0.38 ¢ on 950 fills (§3.3).
- Reward pools are a snapshot of today's configuration. Many were reset on Oct 14–15, before the midterms and
  the FOMC, so pools over the fills window may have been lower.
- Aggregate maker markouts describe the average maker. A newcomer joins the back of a $1–3k queue, so it gets
  more of the sweep fills and fewer of the patient fills, and earns less than the average.

## 1. Universe (`universe.py`, `out/universe.{json,csv}`)

Selection:
- Gamma events were pulled for the tags `politics, elections, geopolitics, economy, fed, fed-rates, world,
  midterms, us-politics, trump, iran, middle-east`, both active (by 1-month volume) and closed with an end
  date in the last 30 days.
- Events with any sports, crypto, pop-culture, tweet, tech or AI tag were dropped, because tag filters leak.
  For example, "elon-musk-of-tweets" carries the `politics` tag.
- The categories are assigned from the tags.
- At most 4 markets were taken per event, giving **40 active markets** (the top 40 by 1-month CLOB volume)
  and **20 resolved** between Sep 16 and Oct 6: the Sep FOMC, Brazil round 1, the Russian Duma, and
  Sep 30 Iran/Israel deadlines.

| | politics | economics (Fed, Hormuz) | geopolitics |
|---|---|---|---|
| markets (active + resolved) | 26 | 10 | 24 |
| fee schedule | `politics_fees`: rate 0.04, exponent 1, takerOnly, **rebateRate 0.25** | `economics_fees`: rate 0.05, rebateRate 0.25 (Hormuz market: no fees) | **no fees** (23 of 24); 1 `politics_fees` |
| taker fee at p = 0.5 | 1.0 ¢/share | 1.25 ¢/share | 0 |
| maker rebate per filled share at p = 0.5 | 0.25 ¢ | 0.31 ¢ | 0 |
| `rewardsMaxSpread` (¢) | 2.5–6.5 | 2.5–4.5 | 2.5–6.5 |
| `rewardsMinSize` (shares) | 50–200 | 200 | 50–200 |
| current daily reward pool (active markets) | $5,993 | $2,302 | $2,962 |
| holding rewards (active markets) | 13 (2028 nominations and midterm markets) | 0 | 3 (Putin / Xi out before 2027, China invades Taiwan) |

Notes:
- **Fees.** The fee is `C·rate·p(1−p)`, paid by takers only. The rebate is 25% of the fee-equivalent of each
  maker fill. Crypto up/down: rate 0.07, rebate 20%.
- **Holding rewards** are 3.25% a year on position value, sampled hourly (help.polymarket.com). That is about
  $0.89 a day per $10k of inventory, negligible for market making.
- **Rewards programme.** It currently covers 18,755 markets with a total of $210k a day. 27 of our 40 active
  markets have a pool: $20–2,004 a day, and $11,257 a day in total. Fee-free geopolitics markets still carry
  reward pools; for example "US × Iran ceasefire through Oct 31" has $2,004 a day.

## 2. Method (`analyze.py`)

**Fills.**
- `extract.py --tokens out/tokens.parquet --events OrderFilled --slim`: 44 days in about 2 h of wall time and
  under 100 MB of disk.
- `to_trades.py --slug-regex '.*'` writes the data/raw trades schema (`onchain/data/politics/trades_*.parquet`).
  Its markets.parquet step is crypto-specific and fails here; it is not needed.
- `markets.py` needed no change, because a gamma slug list → token map works for any market.

**Reference mid.** Every price is put in the frame of outcome 0 (Yes/Up). The trade-implied mid at t averages:
- the latest maker BUY price strictly before t (bid evidence);
- the latest maker SELL price strictly before t (ask evidence);

with both younger than 10 min and not crossed. Otherwise the prices-history point at or before t is used
(politics; 48% of fills), otherwise the fill has no mid.

**Per maker fill (cents/share, size-weighted):**
- edge = side·(mid(t) − p);
- markout(h) = side·(mid(t+h) − p) for h = 1 m, 5 m, 30 m and 2 h, and to resolution. The payout replaces the
  mid once t+h is past a resolved market's end;
- rebate = rebateRate·rate·p(1−p) on days the market charged taker fees on chain.

**Validation against true L2.** For crypto (train days), the recorded L2 mid at block time − 3 s gives
`edge_l2` and `mo_l2_*`. For politics, the same is done on fills during the recording (`analyze.py politics_live`).

**CIs.** `stats.py` bootstraps (1,000 draws) over market × UTC-day clusters.

**Wallets.** Every leg of a wallet (maker or taker), cash plus inventory marked at the window end (payout if
resolved, else the last prices-history mid), minus the on-chain fees it paid, plus modelled rebates. Inventory
held before the window is not observed, so this is the P&L of the window's trades.

## 3. Maker edge: politics vs crypto (`out/compare_ci.csv`, `out/*_edge_by_*.csv`)

### 3.1 Headline (size-weighted over all makers, cents/share, 95% cluster-bootstrap CI)

| | politics (60 mkts, 44 d) | economics | geopolitics | politics only | crypto Aug 15–20 (5.7k mkts) | crypto train Aug 21–22 (401 mkts) |
|---|---|---|---|---|---|---|
| maker legs / notional | 597k / $139M | 159k / $62M | 286k / $46M | 152k / $31M | 3.1M / $20.5M | 252k / $1.8M |
| edge vs trade-implied mid | 0.47 | 0.43 | 0.59 | 0.43 | 1.60 | 1.53 (vs L2 mid: 0.22) |
| markout 1 m | +0.24 | +0.24 | +0.28 | +0.21 | +0.05 | −0.24 (L2: −0.39) |
| markout 5 m | **+0.14** [0.10, 0.19] | +0.12 | +0.17 | +0.17 | **−0.19** [−0.38, −0.00] | −0.53 (L2: −0.46) |
| markout 30 m | +0.07 | −0.01 | +0.17 | +0.13 | −0.15 | −0.46 (L2: −0.56) |
| markout 2 h | **+0.15** [0.07, 0.24] | +0.12 | +0.22 | +0.12 | −0.17 | −0.48 (L2: −0.62) |
| to resolution (resolved mkts) | +1.99 (45% of fills) | +2.32 | +0.81 | +2.81 | −0.16 | −0.54 |
| rebate | 0.085 | 0.118 | 0.001 | 0.111 | 0.203 | 0.202 |
| **net 5 m (markout + rebate)** | **+0.23 [0.18, 0.28] = +54 bps** | +0.23 (+61 bps) | +0.18 (+34 bps) | **+0.28 (+70 bps)** | **+0.01 [−0.18, 0.20] = +2 bps** | −0.33 [−1.10, 0.46] = −63 bps |
| net 2 h | +0.23 [0.14, 0.33] = +55 bps | +0.24 | +0.22 | +0.23 | +0.03 (+6 bps) | −0.28 (−54 bps) |

By crypto contract length, Aug 15–20, net at 5 min: 5m −0.02 ¢, 15m +0.21 ¢, 4h −0.48 ¢. The 4h markets are
badly adverse at 30 min to 2 h (−1.7 to −2.3 ¢).

**Daily dollar totals, all makers.**

| per day | politics universe (60 mkts) | economics | geopolitics | politics only |
|---|---|---|---|---|
| maker notional | $3.17M | $1.42M | $1.05M | $0.70M |
| spread net of 5-min adverse selection | **+$10.7k** | $4.3k | $3.5k | $2.9k |
| rebates | **+$6.4k** | $4.4k | $0.02k | $2.0k |
| current reward pools | **$11.3k** | $2.3k | $3.0k | $6.0k |
| taker fees | $26.3k | | | |

For crypto, a third of the markets in Aug 15–20 give: notional $3.4M/day, spread net of adverse selection
−$11.7k/day and rebates +$13.4k/day. The reward pool was about $32k/day in August across all crypto up/down;
today it is $0.

**Answer.**
- The average crypto up/down maker gives back its entire spread to adverse selection within 5 min. The 20%
  rebate brings it to roughly zero.
- The average politics maker keeps about 30% of its at-fill edge (+0.14 ¢ at 5 min, still +0.15 ¢ at 2 h) and
  adds rebates on top: **+54 bps against +2 bps (crypto Aug 15–20) or −63 bps (crypto train)**.
- Politics also comes with a reward pool about as large as the spread and rebate revenue combined.
- In dollar terms, the 60 politics markets pay their makers about $28k/day from all three sources at current
  pool levels.

### 3.2 By price level and time to resolution (politics; cents/share)

| price | shares | edge | 1 m | 5 m | 30 m | 2 h | resolution | rebate | crypto Aug 15–20, 5 m |
|---|---|---|---|---|---|---|---|---|---|
| ≤ 0.05 | 122M | 0.11 | 0.07 | 0.06 | 0.04 | 0.06 | 0.15 | 0.01 | +0.16 |
| 0.05–0.2 | 71M | 0.62 | 0.37 | 0.31 | 0.32 | 0.36 | 4.61 | 0.08 | −0.03 |
| 0.2–0.4 | 29M | 0.87 | 0.33 | 0.13 | **−0.09** | 0.17 | 7.54 | 0.15 | −0.44 |
| 0.4–0.6 | 34M | 0.70 | 0.30 | 0.03 | **−0.17** | **−0.11** | 7.20 | 0.27 | −0.62 |
| 0.6–0.8 | 29M | 0.84 | 0.37 | 0.13 | **−0.20** | −0.06 | 0.91 | 0.16 | −0.62 |
| 0.8–0.95 | 36M | 0.54 | 0.28 | 0.17 | 0.14 | 0.32 | 0.69 | 0.10 | +0.41 |
| > 0.95 | 10M | 0.51 | 0.40 | 0.41 | 0.47 | 0.54 | 0.75 | 0.00 | +0.31 |

| time to resolution | shares | edge | 5 m | 2 h | resolution |
|---|---|---|---|---|---|
| < 1 h | 1.5M | 1.94 | 1.16 | 0.73 | 5.79 |
| 1 h–1 d | 41M | 0.30 | 0.16 | 0.57 | 1.30 |
| 1–7 d | 85M | 0.42 | 0.16 | 0.17 | 2.03 |
| 7–30 d | 107M | 0.52 | 0.11 | 0.05 | 2.50 |
| > 30 d | 95M | 0.50 | 0.13 | 0.04 | n/a |

- Adverse selection lives in the **contested markets (0.2–0.8)**: the at-fill edge is the largest (0.7–0.9 ¢)
  and is fully given back within 30 min.
- Longshot quotes carry most of the volume and keep their edge at every horizon. That is the
  favourite-longshot bias paying the liquidity provider.
- Crypto shows the same shape (longshots positive, mid prices negative), but it is shifted down by about
  0.3–0.7 ¢.
- Long-dated markets (> 7 d) give up most of the edge by 2 h. The near-resolution buckets look good, but they
  come from a handful of events (the Sep FOMC, Brazil round 1).

### 3.3 How good is the book-free proxy?

- **Crypto** (L2 recorded for every train market):
  - The trade-implied edge is 1.53 ¢ against an L2 edge of 0.22 ¢. The gap comes from the price tails
    (p < 0.05: −7.1 ¢ against L2; p > 0.95: −2.4 ¢), where the book moves in the market's last seconds.
  - Inside 0.05–0.95 the two agree within 0.1–0.8 ¢.
  - The markouts agree much better: 5 m −0.53 (proxy) vs −0.46 (L2), 30 m −0.46 vs −0.56, 2 h −0.48 vs −0.62.
- **Politics** (`analyze.py politics_live`: 1,423 maker legs during 2.6 h of recording, 950 with both mids):
  - The proxy edge is **0.39 ¢ against 0.38 ¢ vs the recorded L2 mid** (block − 3 s). The median per-fill
    difference is 0, and the correlation is 0.79.
  - Markouts: 1 m 0.33 vs 0.21, 5 m 0.29 vs 0.23, 30 m 0.03 vs 0.05.
  - Politics books move about once an hour, so the latest trades describe the book well. The proxy is
    reliable here, unlike in fast crypto tails.

**Conclusion:** compare venues on markouts, not on the edge.

## 4. Who makes money, and from what (`out/*_wallets.csv`, `out/politics_wallet_payouts.csv`)

11,039 wallets made at least one politics maker fill. The top 10 hold 23% of maker shares and the top 25 hold
34%, so the business is not winner-take-all. The top 25 have a median maker share of 82% of their own volume.

| top 25 makers (by maker shares) | politics, 44 d | crypto Aug 15–20 (one third of markets), 6 d | crypto train, 2 d |
|---|---|---|---|
| maker shares | 111M | 14.2M | 1.57M |
| spread at fill (vs proxy mid) | +$349k | +$174k | +$19.4k |
| maker markout at 5 m / 2 h | +$123k / +$115k | +$4.8k / +$8.5k | −$1.1k / −$5.5k |
| maker legs marked to the window end / settlement | +$971k | +$5.9k | −$4.5k |
| taker legs (marked) | −$430k | +$78k | +$8.7k |
| taker fees paid | −$141k | −$53k | −$4.7k |
| rebates (modelled) | +$85k | +$31k | +$3.4k |
| **total before rewards** | **+$485k** (14 of 25 > 0, median +$1.1k) | **+$62k** (18 of 25 > 0) | +$2.9k (17 of 25 > 0) |
| actual REWARD / MAKER_REBATE payouts, all markets | $249k / $149k | (Aug payouts: see `../rewards`) | |
| rewards attributable to this universe (crude, scaled by rebate share) | ≈ $139k | | |

**Politics top 25: where the money comes from.**
- The spread captured at fill is +$349k, but adverse selection takes back $234k within 2 h. That leaves
  +$115k of liquidity-provision profit.
- Rebates (+$85k) and attributable rewards (≈ +$139k) together are about twice that.
- The rest is **direction**: +$621k of inventory drift after 2 h on maker legs, and −$430k on taker legs.
- Three wallets make most of the money, mostly from direction:
  - 0x7b02…: total +$275k, of which direction +$190k;
  - 0x7cf9…: total +$195k, direction +$68k;
  - 0xed10…: total +$48k, direction +$39k.
- Excluding those three, the other 22 net −$32k before rewards and about +$104k after them.
- So the typical professional politics maker **lives on rebates and rewards plus a thin spread**. The big
  winners are directional traders who also quote.
- Reward farming is visible: four wallets each received $24–72k of REWARDS over the window with modest spread
  P&L in these markets.

**Crypto top 25.** The spread captured at fill is about 20× larger than its 2 h markout, so adverse
selection eats almost all of it. They earn mostly from taker legs and rebates, the pattern of
latency-advantaged firms rather than passive makers.

## 5. Live books (`live_books.py`, `out/books_politics.csv`, `out/rewards_live.csv`, `out/books_crypto.csv`, `out/rec_summary/`)

The recording covers 40 markets with REST reconcile every 60 s; the mismatch rate is about 0.05%. Medians
across markets, over 2.7 h of recording (Oct 15, 19:34–22:17 UTC; hourly per-market rows in
`out/rec_summary/books_hourly.csv`):

| | politics | economics | geopolitics | crypto 5m | crypto 15m | crypto 4h |
|---|---|---|---|---|---|---|
| quoted spread, median (¢) | 0.20 | 1.0 | 0.75 | 3 | 2 | 2 |
| time at one tick | 100% | 100% | 100% | 28% | 31% | 36% |
| touch depth per side ($) | 1,706 | 4,325 | 1,048 | 25 | 32 | 57 |
| depth within 1 ¢ of mid ($) | 15.7k | 12.3k | 7.6k | 2.5 | 1.2 | 14 |
| depth within the reward band ($) | 145k | 40k | 57k | 164 | 215 | 256 |
| touch **price** changes per hour | 0 | 0.7 | 1.5 | 549 | 503 | 159 |
| touch **size** changes per hour | 26 | 55 | 50 | 569 | 540 | 167 |
| book updates per hour | 662 | 567 | 644 | 687 | 665 | 485 |

**Liquidity rewards for a hypothetical maker** (one sample per minute, competitors from the recorded book with
the 'hi' bracket, quoting `rewardsMinSize` on both sides):

- **At the touch** it earns a **0.0–1.7%** share of the pool. The median is $0.008/h, rising to
  $0.37–0.55/h in the Fed October markets ($1,000/day pools, 2.5 ¢ band).
- **Per $1k of capital** (a two-sided quote of X shares ties up X dollars) that is a median of **≈ $0.05/h
  (0.14%/day)**. The best markets reach **$1.8–2.7/h** (Fed Oct no-change / +25 bp) and $0.6–0.7/h
  (US–Iran ceasefire, Lula).
- One tick behind the touch, the reward falls by a factor of 1.4–4.

These are gross of fills. An order resting at the touch of a Fed market gets swept on FOMC news. The trade-off
to optimise is reward yield against jump-weighted adverse selection.

## 6. Risk: news jumps and resolution (`risk.py`, `out/risk_*.csv`)

10-min prices-history, Sep 1–Oct 15:

| | market-days | mean abs 10-min move (¢) | 5 ¢+ jumps / market-day | 10 ¢+ jumps / market-day | variation in the top 1% of 10-min moves (median market) | largest 10-min move (¢) |
|---|---|---|---|---|---|---|
| politics | 1,090 | 0.008 | 0.017 | 0.005 | ≈ 100% (most 10-min bars do not move) | 48.5 |
| economics | 303 | 0.058 | 0.143 | 0.051 | 48% | 25 |
| geopolitics | 822 | 0.053 | 0.121 | 0.028 | 36% | 34 |

**Maker fills followed within 2 h by a 5 ¢+ jump.**
- Politics: 6% of fills, markout +0.58 ¢. Economics: 12%, +0.49 ¢. Geopolitics: 3%, **−0.23 ¢** (against
  +0.23 ¢ otherwise).
- In this sample the jumps did not systematically pick off makers outside geopolitics. With a handful of real
  news events, though, the jump tail is **under-sampled**. It is the main risk for any strategy that rests
  size at the touch.

**Resolution.**
- In the 20 resolved markets, makers who held fills to resolution earned +2.0 ¢/share: +1.3 ¢ for fills in the
  last 24 h, +2.0 ¢ for fills 1–7 days out, +2.4 ¢ earlier.
- Fills in the last hour were break-even. 149 fills after the gamma `closedTime` (late UMA settlement) earned
  about 9 ¢: free money for anyone still quoting, but tiny.
- The main resolution risks are:
  - a binary that flips in the final hours, as on election nights (Brazil round 1 resolved six of our markets
    within a day);
  - UMA disputes and late settlement, which tie up capital;
  - the drop of the reward pool at close.

**Risk ranking for a politics maker:**
1. Scheduled-event jumps: FOMC, election nights, ceasefire deadlines. They are predictable in time, so a
   policy can pull its quotes.
2. Unscheduled geopolitical news, the only category with negative jump-window markouts.
3. Inventory carried to resolution.

## 7. What a faithful queue-aware replay needs, and a plan for the C env

### 7.1 Data

| need | source | status |
|---|---|---|
| L2 book at 1 s, all levels, with deltas | `polyrec` with `recorder_politics.toml` (+ `run_recorder.py`: the UA patch needed for CLOB REST from aiohttp) | working; about 7 MB/h (books + raw events) for 40 markets, about 3% of a core |
| queue position | **No L3 in the public feed**: the ws `price_change.hash` is a 20-byte value that differs between the two legs of one change, so it is not the order hash. Estimate it as `mm.h` does: join at the shown size, at-price prints consume it, cancels ahead pro rata. Politics books change slowly (about 30 touch-size changes per hour), so the estimate is much tighter than for crypto. | method exists in `mm.h` (central model) |
| fills with maker and taker identity and exact block time | `onchain/extract.py` (no `--slim` to keep `order_hash`); `--calldata` adds order creation time and original size | working; minutes per day for 60 markets |
| scheduled events (FOMC, polls, debates, election nights) | calendar file | to build (small) |
| reward pool history | `/rewards/markets/current` snapshot every day (it overwrites) | **start snapshotting now**; the pools changed on Oct 14–15 |
| resolution and UMA timing | gamma `closedTime`, `umaResolutionStatuses` | available |

**How many days?**
- Politics touch dynamics are about 400× slower than crypto (about 1 touch price change per hour), so one
  market-day carries roughly the information of a few crypto 15m markets.
- To fit a fill model (queue depletion rate, sweep size distribution), 2 weeks × 40 markets is enough:
  about 13k market-hours and about 13k maker legs a day.
- To evaluate a policy against the jump tail, the bottleneck is events: we see 0.02–0.14 jumps of 5 ¢ or more
  per market-day.
- **Recommendation: ≥ 4 weeks of continuous recording (Oct 16 → Nov 13).** That window includes the Oct 25
  Brazil runoff, the Oct 28 FOMC and the Nov 3 US midterms. Hold out one scheduled event per category as
  test. Record about 100 markets rather than 40: the recorder uses about 3% of a core and about 0.2 GB/day for
  40 markets.

### 7.2 Extending the C env (`mm.h` is data-source agnostic)

`mm.h` consumes per-second rows (`ticks/book/feat/fills/fill_index` .bin) and per-market metadata. Its
mechanics carry over unchanged: latency, match delay, the three fill models, the rebate formula, the reward
hook (`liq_pi`/`liq_comp` columns) and settlement. The steps:

1. **Builder** `data/build_politics.py`, a copy of `build.py` that reads `polyrec` books and on-chain trades.
   - Rows stay at 1 s, but an **episode becomes a window** of a market, for example 6 h, with a random start
     and a stride across the recording. It is no longer the whole market life.
   - The terminal marks inventory to the mid, and settles at the payout only if the window contains the
     resolution.
   - This only needs new `markets.bin` fields: `t_open`, `t_end` and a `settle` flag. It does not touch the
     core loop.
   - The two outcome tokens are already mirrored in the Up frame, so politics Yes/No maps directly.
   - The tick comes from the book: 0.01 or 0.001.
2. **Fees per market.** `fee_rate`, `rebate_rate` and `fee_exponent` are already read from `markets.bin`. Set
   them per category: 0.04/0.25, 0.05/0.25, 0/0.
3. **Rewards.** Enable the existing hook with per-market `rewardsMaxSpread` (2.5–6.5 ¢) and `rewardsMinSize`
   (50–200), and the pool from the daily snapshot. `size_mult` must be large enough for our orders to
   qualify (≥ 200 shares).
   - Precompute `liq_comp` from **all** levels within the band. The top-5 rule is wrong here: the politics
     band holds $40–130k across many levels.
4. **Features.**
   - Drop the Binance and fair-value block (`p_model` = mid, or a slow EMA of mid).
   - Add: time to scheduled event, time to resolution, recent jump indicator, queue-ahead / shown size (already
     obs 40–41), the reward share our current quote would earn, and the 10-min realised volatility.
5. **Decision interval.** dt = 5–30 s is enough: the touch changes about once an hour. Put
   `max_episode_steps` at a 6 h window / dt, which gives 720–4,300 steps, similar to crypto 4h.
6. **Fill model calibration.** Re-fit the central model's cancel-ahead rate from the recorded L2 deltas, and
   validate the replayed fills against on-chain maker fills of real queue joiners. Join on `order_hash`, plus
   calldata creation time to measure their queue age.
7. **Evaluation.** Keep D1–D7, but add an event-stratified test: P&L in windows with a scheduled event vs
   without, and against no-quote and against "join at the touch with min reward size, pull 1 h before
   scheduled events" as the natural baseline. The latter is what the reward-farming wallets do.

Expected outcome: the edge that is learnable here is mostly **quote placement for reward share versus jump
exposure**, plus inventory skew in contested (0.2–0.8) markets. That is a better-posed RL problem than crypto
up/down, where the takers' information from Binance dominated everything. It is still a modest edge: about
+0.2 ¢ per filled share over crypto, and the reward pool is the swing factor.

## Files

| file | what |
|---|---|
| `universe.py` | gamma universe, fee and reward config → `out/universe.{json,csv}`, `out/tokens.parquet` |
| `fetch_meta.py` | rewards (`out/rewards.json`, `out/rewards_all_summary.json`), resolutions (`out/resolution.json`), prices-history (`data/`, git-ignored) |
| `analyze.py {politics,crypto,crypto_more,politics_live}` | maker edge, markouts, rebates, wallet P&L → `out/<set>_edge_by_*.csv`, `out/<set>_wallets.csv` (top 3000), `out/<set>_summary.json`, `out/*_analyze.log` |
| `stats.py` | clustered bootstrap CIs → `out/compare_ci.csv` |
| `wallet_payouts.py` | REWARD / MAKER_REBATE payouts of the top makers → `out/politics_wallet_payouts.csv` |
| `risk.py` | jumpiness, jump-window and resolution markouts → `out/risk_*.csv`, `out/risk_summary.json` |
| `live_books.py {politics,crypto}` | live spread, depth and touch dynamics, and the reward share at the touch → `out/books_*.csv`, `out/rewards_live.csv`, `out/books_politics_summary.json` |
| `rec_summary.py` | hourly per-market book summaries → `out/rec_summary/books_hourly.csv` |
| `recorder_politics.toml`, `run_recorder.py` | `polyrec` config (40 markets, 5 sources) and a launcher with the CLOB User-Agent patch (aiohttp with polyrec's browser UA gets a Cloudflare 403 on `/book`; no shared code changed) |
| `live_extract.py` | `extract.py` wrapper for windows ending at the chain head (`plan()` raises for today's partial day) |
| `rec_out/` (git-ignored) | raw recording |

**Shared-code notes, nothing edited:**
- `recorder/polyrec/http.py`: the browser User-Agent gets a Cloudflare 403 from `clob.polymarket.com` via
  aiohttp.
- `polyrec`: a market whose gamma `endDate` is in the past but which is still open (Brazil after round 1, LA
  mayor) is never subscribed. The workaround is `linger_s`/`post_close_s` per source.
- `onchain/extract.py plan()` fails for today's partial day.
- `onchain/to_trades.py` builds its markets.parquet with crypto-only columns.
