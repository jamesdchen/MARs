# Politics / economics / geopolitics market making: a book-free feasibility study (w6)

Written 2026-10-15 by worker w6. The question: could politics, Fed and geopolitics markets work better for a
maker than crypto up/down? The crypto up/down study (`../../final/RESULTS.md`, `../fair_value/REPORT.md`,
`../rewards/REPORT.md`) found no edge there.

The data:
- **On-chain fills.** Every `OrderFilled` for 60 markets (120 tokens) over **30 days, 2026-09-15 → 10-14**:
  370,926 maker legs, 213M shares, $95.9M maker notional. Source: keyless Polygon archive RPCs via
  `../../onchain/extract.py`.
- **Prices.** CLOB `prices-history`, plus gamma metadata and resolutions.
- **Rewards.** CLOB `/rewards/markets/current`, plus the data-api `/activity` payout rows of the top makers.
- **Live books.** Live L2 books of the 40 active markets, recorded with `polyrec` from 19:34 UTC Oct 15 until
  this session ends.
- **Comparison.** The **same** analysis code on on-chain fills of the crypto up/down train days (Aug 21–22,
  401 markets, 252k maker legs). The val and test days were not touched; `data/build_jun15` was never read.

Everything is reproducible from the scripts in this folder (see "Files" at the end).

## TL;DR

1. **Yes: politics market making is clearly more promising than crypto up/down.**
   - Size-weighted over all makers, a politics maker fill is worth **+0.19 ¢/share (+41 bps of notional) after
     5 min, rebate included** (95% CI +0.14 to +0.24 ¢, clustered by market × day).
   - Crypto up/down, measured the same way, gives **−0.33 ¢/share (−63 bps)**, CI −1.10 to +0.44.
   - The gap is **about 0.5 ¢/share, or about 100 bps of notional**, and politics stays positive at 2 h
     (+0.21 ¢).
   - The source is the counterparty, not the spread. Politics takers are mostly uninformed at minute-to-hour
     horizons. Crypto takers trade on Binance moves, which have no analogue here. The politics 5-min adverse
     selection is 0.36 ¢; crypto's is about 1.7 ¢ against the trade-implied mid.
2. **Revenue split, all makers in the 60 markets, per day:**
   - Spread net of 5-min adverse selection: **≈ $8.0k**.
   - Maker rebates: **≈ $5.2k** (25% of the 4–5% politics/economics taker fee; geopolitics has no fee).
   - Liquidity rewards: **≈ $11.3k** (the current daily pools on 27 of the 60 markets).
   - That is about **33 : 21 : 46**. Rewards are the largest single source, and they are what crypto up/down
     no longer has: zero pool today, against an August pool of $1M a month.
   - The 25 largest politics makers were paid **$238k in rewards** over the 30 days (all their markets),
     against $137k in rebates.
3. **Are the professional makers profitable? Mostly, but the money is concentrated.**
   - The top 25 makers by volume made about **+$227k over 30 days before rewards** on these 60 markets, and
     about +$340k including an estimate of their rewards.
   - That splits into spread at fill +$231k, adverse selection to 2 h −$182k, inventory drift after 2 h +$296k,
     taker legs −$103k, taker fees −$62k and rebates +$47k.
   - Only 14 of the 25 are profitable before rewards (17 of 25 with rewards). The median is about +$0.9k per
     month.
   - Two wallets make $244k, almost all of it directional (inventory carried into moves). Professional
     politics MM is a **spread + rewards business with a fat-tailed directional overlay**, not a pure spread
     business.
   - For comparison, the crypto top 25 made +$2.9k over the 2 train days, with negative markouts that rebates
     only partly offset.
4. **Main risk: jumps.**
   - Politics prices sit still and then jump. 94% of a politics market's absolute 10-min price variation comes
     from its largest 1% of 10-min moves (42% geopolitics, 57% economics).
   - Moves of 5 ¢ or more in 10 min occur 0.02 (politics) to 0.16 (economics) times per market-day. The largest
     seen was 48.5 ¢.
   - Mid-priced markets (0.2–0.8) carry the adverse selection: −0.3 to −0.4 ¢ at 30 min to 2 h. Longshot
     prices (< 0.2, > 0.8) are positive at every horizon.
   - Resolution: makers who held to resolution in the 20 resolved markets earned +1.3 ¢/share (CI +0.5 to
     +2.2). That is mostly makers selling longshots that lost; a different set of resolutions could flip it.
     Treat it as inventory risk, not as edge.
5. **Live books are a different regime from crypto.**
   - The politics touch is **one tick wide 100% of the time**, and its price changes **0–1.3 times per hour**
     (crypto: 160–550).
   - Touch depth is $0.8–3k per side (crypto $25–57), and depth within the reward band is $40–133k (crypto
     $160–260).
   - So a replay must model the **queue**, not the price: a new order joins behind $1–3k at the touch, and
     most fills come from slow queue depletion and rare sweeps.
6. **Replay data.** The CLOB websocket's per-order `hash` is not the on-chain order hash: it is 20 bytes and
   differs between the two legs of one change. The public feed therefore has no L3, and queue position must be
   estimated from L2 deltas plus on-chain fills, as in `mm.h`'s central fill model.
   - Recommendation: record **≥ 4 weeks** of politics books with `polyrec`. The window should cover at least
     one scheduled jump event per category: the **Oct 28 FOMC**, the **Nov 3 US midterms** and the **Oct 25
     Brazil runoff**. That gives about 3–8k market-hours per category and ≥ 50 jumps of 5 ¢ or more for
     stress tests.
   - Pull the matching on-chain fills, which takes minutes per day. The C env needs about a week of plumbing,
     not a redesign (§7).

**Caveats:**
- The book-free spread proxy is biased in fast markets: it is +1.3 ¢ too high on crypto when measured against
  the recorded L2 mid. Markouts are much less sensitive, because they are differences of the same estimator.
  The crypto proxy markouts match the L2 markouts within 0.1–0.2 ¢. The live politics check agrees within
  0.01 ¢ (§3.3).
- Reward pools are a snapshot of today's configuration. Many were reset on Oct 14–15, before the midterms and
  the FOMC, so pools during Sep 15–Oct 14 may have been lower.
- Aggregate maker markouts describe the average maker. A newcomer joins the back of a $1–3k queue, gets more
  of the sweep fills and fewer of the patient fills, and so earns less than the average.

## 1. Universe (`universe.py`, `out/universe.{json,csv}`)

Selection:
- Gamma events were pulled for the tags `politics, elections, geopolitics, economy, fed, fed-rates, world,
  midterms, us-politics, trump, iran, middle-east`, both active (by 1-month volume) and closed with an end
  date in the last 30 days.
- Events with any sports, crypto, pop-culture, tweet, tech or AI tag were dropped, because tag filters leak.
  For example, "elon-musk-of-tweets" carries the `politics` tag.
- The categories are assigned from the tags.
- At most 4 markets were taken per event, giving **40 active markets** (the top 40 by 1-month CLOB volume)
  and **20 that resolved in the window**.

| | politics | economics (Fed, Hormuz) | geopolitics |
|---|---|---|---|
| markets (active + resolved) | 26 | 10 | 24 |
| fee schedule | `politics_fees`: rate 0.04, exponent 1, takerOnly, **rebateRate 0.25** | `economics_fees`: rate 0.05, rebateRate 0.25 (Hormuz market: no fees) | **no fees** (23 of 24); 1 `politics_fees` |
| taker fee at p = 0.5 | 1.0 ¢/share | 1.25 ¢/share | 0 |
| maker rebate per filled share at p = 0.5 | 0.25 ¢ | 0.31 ¢ | 0 |
| `rewardsMaxSpread` (¢) | 2.5–6.5 | 2.5–4.5 | 2.5–6.5 |
| `rewardsMinSize` (shares) | 50–200 | 200 | 50–200 |
| current daily reward pool (active markets) | $5,993 | $2,302 | $2,962 |
| holding rewards | 15 of 40 active (2028 and midterm markets, "out before 2027") | 0 | some ("out before 2027") |

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
- `extract.py --tokens out/tokens.parquet --events OrderFilled --slim`: 30 days, about 1.5 h wall time,
  under 50 MB of disk.
- `to_trades.py --slug-regex '.*'` writes the data/raw trades schema (`onchain/data/politics/trades_*.parquet`).
  Its markets.parquet step is crypto-specific and fails on these markets; it is not needed.
- `markets.py` needed no change, because a gamma slug list → token map works for any market.

**Reference mid.** Every price is put in the frame of outcome 0 (Yes/Up). The trade-implied mid at t averages:
- the latest maker BUY price strictly before t (bid evidence);
- the latest maker SELL price strictly before t (ask evidence);

with both younger than 10 min and not crossed. Otherwise the prices-history point at or before t is used
(politics; 51% of fills), otherwise the fill has no mid.

**Per maker fill (cents/share, size-weighted):**
- edge = side·(mid(t) − p);
- markout(h) = side·(mid(t+h) − p) for h = 1 m, 5 m, 30 m and 2 h, and to resolution. The payout replaces the
  mid once t+h is past a resolved market's end;
- rebate = 0.25·rate·p(1−p) on days the market charged taker fees on chain.

**Validation against true L2.** For crypto, the recorded L2 mid at block time − 3 s gives `edge_l2` and
`mo_l2_*`. For politics, the same is done on the fills during the recording (`analyze.py politics_live`).

**CIs.** `stats.py` bootstraps (1,000 draws) over market × UTC-day clusters.

**Wallets.** Every leg of a wallet (maker or taker), cash plus inventory marked at the window end (payout if
resolved, else the last prices-history mid), minus the on-chain fees it paid, plus modelled rebates. Inventory
held before the window is not observed, so this is the P&L of the window's trades.

## 3. Maker edge: politics vs crypto (`out/compare_ci.csv`, `out/*_edge_by_*.csv`)

### 3.1 Headline (size-weighted, all makers)

| | politics, all (60 mkts, 30 d) | economics | geopolitics | politics only | crypto up/down (401 mkts, 2 d) |
|---|---|---|---|---|---|
| maker legs / shares / notional | 371k / 213M / $95.9M | 75k / 78M / $32M | 174k / 70M / $37M | 122k / 65M / $27M | 252k / 3.5M / $1.8M |
| edge vs trade-implied mid (¢) | 0.47 | 0.42 | 0.57 | 0.41 | 1.53 (L2 mid: 0.22; ex-tails: 1.2–2.5, §3.3) |
| markout 1 m | **+0.22** | +0.22 | +0.25 | +0.19 | **−0.24** |
| markout 5 m | **+0.11** [0.06, 0.16] | +0.04 | +0.14 | +0.17 | **−0.53** [−1.30, 0.35] |
| markout 30 m | +0.03 | −0.15 | +0.14 | +0.14 | −0.46 |
| markout 2 h | **+0.13** [0.02, 0.24] | +0.06 | +0.21 | +0.13 | **−0.48** |
| to resolution (resolved markets only) | +1.30 (26% of fills) | +1.14 | +1.04 | +3.16 | −0.54 |
| rebate | 0.07 | 0.10 | 0.00 | 0.11 | 0.20 |
| **net 5 m (markout + rebate)** | **+0.19 [0.14, 0.24]** = **+41 bps** | +0.15 | +0.14 | **+0.28** (+67 bps) | **−0.33 [−1.10, 0.44]** = **−63 bps** |
| net 2 h | +0.21 [0.10, 0.32] = +45 bps | +0.16 | +0.22 | +0.25 | −0.28 = −54 bps |
| taker fees per day | $21.5k | | | | $17.9k |

Daily dollar totals for all makers:

| | politics universe | crypto train days |
|---|---|---|
| maker notional | $3.20M/day | $0.90M/day |
| spread net of 5-min AS | +$8.0k/day | −$8.5k/day |
| rebates | +$5.2k/day | +$3.5k/day |
| reward pools | ≈ $11.3k/day (current) | $0 today; about $32k/day in August |

By category:

| per day | spread − AS(5 m) | rebates | current reward pool |
|---|---|---|---|
| politics | $3.6k | $2.4k | $6.0k |
| economics | $1.1k | $2.7k | $2.3k |
| geopolitics | $3.4k | $0.02k | $3.0k |

**Answer.**
- In crypto up/down, makers lose at every horizon from 1 min. Rebates cut the loss but do not close it.
- In politics, makers keep about a quarter of their at-fill edge after 5 min. That leaves +0.11 ¢ before
  rebates and +0.19 ¢ after, with CIs clear of zero. On notional, politics is about 100 bps better per filled
  dollar, and it comes with a reward pool about as large as the spread revenue.

### 3.2 By price level and time to resolution (politics; cents/share)

| price | shares | edge | 1 m | 5 m | 30 m | 2 h | resolution | rebate |
|---|---|---|---|---|---|---|---|---|
| ≤ 0.05 | 73M | 0.11 | 0.06 | 0.05 | 0.05 | 0.08 | 0.19 | 0.00 |
| 0.05–0.2 | 52M | 0.61 | 0.34 | 0.27 | 0.31 | 0.35 | 3.37 | 0.07 |
| 0.2–0.4 | 17M | 0.97 | 0.30 | −0.04 | **−0.42** | −0.08 | 0.02 | 0.13 |
| 0.4–0.6 | 15M | 0.71 | 0.23 | −0.01 | **−0.30** | **−0.30** | −2.96 | 0.24 |
| 0.6–0.8 | 19M | 0.69 | 0.26 | −0.06 | **−0.43** | −0.24 | 5.49 | 0.15 |
| 0.8–0.95 | 28M | 0.51 | 0.26 | 0.14 | 0.11 | 0.35 | 0.77 | 0.09 |
| > 0.95 | 9M | 0.51 | 0.43 | 0.43 | 0.49 | 0.56 | 0.71 | 0.00 |

| time to resolution | shares | edge | 5 m | 2 h | resolution |
|---|---|---|---|---|---|
| < 1 h | 1.5M | 2.01 | 1.21 | 0.77 | 5.79 |
| 1 h–1 d | 41M | 0.30 | 0.16 | 0.57 | 1.30 |
| 1–7 d | 32M | 0.43 | 0.11 | 0.11 | 1.51 |
| 7–30 d | 67M | 0.53 | 0.07 | −0.02 | 0.41 |
| > 30 d | 72M | 0.49 | 0.10 | 0.02 | n/a |

- Adverse selection lives in the **contested markets (0.2–0.8)**: the at-fill edge is the largest (0.7–1.0 ¢)
  and is fully given back within 30 min.
- Longshot quotes (< 0.2 or > 0.8) carry most of the volume and keep their edge. That is the favourite-longshot
  bias paying the liquidity provider.
- Long-dated markets (> 7 d) are roughly break-even after 2 h. The near-resolution buckets (≤ 1 d) are
  positive in this sample, which is dominated by the Brazil first round and the September FOMC. With only a
  few events, they are not reliable.
- Crypto by price (same code): from 5 min on, markouts are −0.45 to −2.8 ¢ for 0.05–0.6 and +0.2 to +1.8 ¢
  for 0.6–0.95. Crypto's "time to resolution" is always < 1 d.

### 3.3 How good is the book-free proxy?

- **Crypto** (L2 recorded for every train market):
  - The trade-implied edge is 1.53 ¢ against an L2 edge of 0.22 ¢. The whole gap comes from the price tails
    (p < 0.05: −7.1 ¢ against L2; p > 0.95: −2.4 ¢), where the book moves in the market's last seconds.
  - Inside 0.05–0.95 the two agree within 0.1–0.8 ¢.
  - The markouts agree much better: 5 m −0.53 (proxy) vs −0.46 (L2), 30 m −0.46 vs −0.56, 2 h −0.48 vs −0.62.
- **Politics** (fills during the recording, `analyze.py politics_live`): on the first 35 min (452 maker legs),
  the proxy edge is 0.51 ¢ vs 0.50 ¢ for L2. Politics books move about once an hour, so the latest trades
  describe the book well. *Updated at the end of the session: see §5.*

**Conclusion:** compare venues on markouts, not on the edge. The edge proxy is reliable for politics but
inflated for fast crypto tails.

## 4. Who makes money, and from what (`out/*_wallets.csv`, `out/politics_wallet_payouts.csv`)

8,253 wallets made at least one politics maker fill. The top 10 hold 19% of maker shares and the top 25 hold
31%, so the business is not winner-take-all. The top 25 have a median maker share of 81% of their own volume.

| top 25 makers (by maker shares) | politics, 30 d | crypto, 2 train days |
|---|---|---|
| maker shares | 65.4M | 1.57M |
| spread captured at fill (vs mid) | +$231k | +$19.4k (proxy) |
| maker markout at 5 m / 2 h | +$62k / +$50k | −$1.1k / −$5.5k |
| maker legs marked to window end / settlement | +$346k | −$4.5k |
| taker legs (marked) | −$103k | +$8.7k |
| taker fees paid | −$62k | −$4.7k |
| rebates (modelled, this universe) | +$47k | +$3.4k |
| **total before rewards** | **+$227k** (14 of 25 > 0; median +$0.9k) | **+$2.9k** (17 of 25 > 0) |
| actual REWARD payouts, all markets | $238k | (Aug payouts: see `../rewards`) |
| actual MAKER_REBATE payouts, all markets | $137k | |
| rewards attributable to this universe (crude, rebate-share scaled) | ≈ $112k | |

**Decomposition for the politics top 25:**
- spread +$231k;
- adverse selection to 2 h −$182k;
- drift after 2 h +$296k, which is direction, concentrated in a few wallets: 0x6575… +$139k, 0x4488… +$105k;
- taker activity −$165k including fees;
- rebates +$47k;
- rewards ≈ +$112k.

So after adverse selection, a professional politics maker earns **mostly from rewards and rebates plus a
modest spread**. The big winners add directional bets. The median top maker earns about $1k a month in these
60 markets before rewards.

Rewards are paid per wallet without a market tag. Two of the top 25 receive more than $25k a month in rewards
while showing little spread P&L here (0xf031…, 0xeed8…): they are farming rewards across many markets. The
"attributable" figure assumes rewards in and outside the universe follow the same proportions as rebates. It
assigns 0 to wallets that are active only in fee-free geopolitics markets, so it is a lower bound there.

## 5. Live books (`live_books.py`, `out/books_politics.csv`, `out/rewards_live.csv`, `out/books_crypto.csv`)

The recording covers 40 markets with REST reconcile every 60 s; the mismatch rate is about 0.05%. Medians
across markets (*numbers updated at the end of the session*):

| | politics | economics | geopolitics | crypto 5m | crypto 15m | crypto 4h |
|---|---|---|---|---|---|---|
| quoted spread, median (¢) | 0.25 | 1.0 | 0.75 | 3 | 2 | 2 |
| time at one tick | 100% | 100% | 100% | 28% | 31% | 36% |
| touch depth per side ($) | 1,602 | 3,117 | 780 | 25 | 32 | 57 |
| depth within 1 ¢ of mid ($) | 15.8k | 12.3k | 8.2k | 2.5 | 1.2 | 14 |
| depth within the reward band ($) | 133k | 40k | 64k | 164 | 215 | 256 |
| touch **price** changes per hour | 0 | 0 | 1.3 | 549 | 503 | 159 |
| touch **size** changes per hour | 25 | 32 | 36 | 569 | 540 | 167 |
| book updates per hour | 710 | 503 | 660 | 687 | 665 | 485 |

**Liquidity rewards for a hypothetical maker** (one sample per minute, competitors from the recorded book with
the 'hi' bracket, quoting `rewardsMinSize` on both sides):

- **At the touch** it earns a **0.02–1.1%** share of the pool. The median is about $0.01/h, rising to
  $0.36–0.45/h in the Fed October markets ($1,000/day pools, 2.5 ¢ band).
- **Per $1k of capital** (a two-sided quote of X shares ties up X dollars) that is a median of **≈ $0.05/h
  (0.12%/day)**. The best markets reach **$1.8–2.3/h** (Fed Oct no-change / +25 bp) and $0.5/h (US–Iran
  ceasefire, Brazil).
- One tick behind the touch, the reward falls by a factor of 1.4–4.

These yields come from a few hours of book. They are upper bounds before fills: an order resting at the touch
of a Fed market gets swept on FOMC news. That is the trade-off to optimise: reward yield against
jump-weighted adverse selection.

## 6. Risk: news jumps and resolution (`risk.py`, `out/risk_*.csv`)

10-min prices-history, Sep 15–Oct 15:

| | market-days | mean abs 10-min move (¢) | 5 ¢+ jumps / market-day | 10 ¢+ jumps / market-day | variation in the top 1% of 10-min moves | largest 10-min move (¢) |
|---|---|---|---|---|---|---|
| politics | 740 | 0.006 | 0.020 | 0.007 | **94%** | 48.5 |
| economics | 163 | 0.072 | 0.159 | 0.074 | 57% | 25 |
| geopolitics | 528 | 0.046 | 0.072 | 0.015 | 42% | 33 |

**Maker fills followed within 2 h by a 5 ¢+ jump.**
- Politics: 7% of fills, markout +0.56 ¢. Economics: 15%, +0.45 ¢. Geopolitics: 1.5%, **−0.29 ¢** (against
  +0.22 ¢ otherwise).
- In this sample the jumps did not systematically pick off the makers. In politics and economics the makers
  who traded just before a jump were often on the right side: they were sellers of longshots and buyers of the
  favourite.
- This is a 30-day sample with a handful of real news events, so the jump tail is **under-sampled**. It is
  the main risk for any strategy that rests size at the touch.

**Resolution.**
- In the 20 resolved markets, makers who held fills to resolution earned +1.3 ¢/share overall: +1.3 ¢ for
  fills in the last 24 h, +1.5 ¢ for fills 1–7 days out.
- Fills in the last hour were break-even. 149 fills after the gamma `closedTime` (late UMA settlement) earned
  about 9 ¢, which is free money for anyone still quoting.
- The main resolution risks are:
  - a binary that flips at the final hour (Brazil round 1: four candidate markets resolved No within hours);
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
| L2 book at 1 s, all levels, with deltas | `polyrec` with `recorder_politics.toml` (+ `run_recorder.py`: the UA patch needed for CLOB REST from aiohttp) | working; about 15 MB/h raw for 40 markets |
| queue position | **No L3 in the public feed**: the ws `price_change.hash` is a 20-byte value that differs between the two legs of one change, so it is not the order hash. Estimate it as `mm.h` does: join at the shown size, at-price prints consume it, cancels ahead pro rata. Politics books change slowly (about 30 touch-size changes per hour), so the estimate is much tighter than for crypto. | method exists in `mm.h` (central model) |
| fills with maker and taker identity and exact block time | `onchain/extract.py` (no `--slim` to keep `order_hash`); `--calldata` adds order creation time and original size | working; minutes per day for 60 markets |
| scheduled events (FOMC, polls, debates, election nights) | calendar file | to build (small) |
| reward pool history | `/rewards/markets/current` snapshot every day (it overwrites) | **start snapshotting now**; the pools changed on Oct 14–15 |
| resolution and UMA timing | gamma `closedTime`, `umaResolutionStatuses` | available |

**How many days?**
- Politics touch dynamics are about 400× slower than crypto (about 1 touch price change per hour), so one
  market-day carries roughly the information of a few crypto 15m markets.
- To fit a fill model (queue depletion rate, sweep size distribution), 2 weeks × 40 markets is enough:
  about 13k market-hours and about 10k maker legs a day.
- To evaluate a policy against the jump tail, the bottleneck is events: we see 0.02–0.16 jumps of 5 ¢ or more
  per market-day.
- **Recommendation: ≥ 4 weeks of continuous recording (Oct 16 → Nov 13).** That window includes the Oct 25
  Brazil runoff, the Oct 28 FOMC and the Nov 3 US midterms. Hold out one scheduled event per category as
  test. Record about 100 markets rather than 40: the recorder uses about 5% of a core and well under
  1 GB/day.

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
up/down, where the takers' information from Binance dominated everything.

## Files

| file | what |
|---|---|
| `universe.py` | gamma universe, fee and reward config → `out/universe.{json,csv}`, `out/tokens.parquet` |
| `fetch_meta.py` | rewards (`out/rewards.json`, `out/rewards_all_summary.json`), resolutions (`out/resolution.json`), prices-history (`data/`, git-ignored) |
| `analyze.py {politics,crypto,politics_live}` | maker edge, markouts, rebates, wallet P&L → `out/<set>_edge_by_*.csv`, `out/<set>_wallets.csv` (top 3000), `out/<set>_summary.json`, `out/*_analyze.log` |
| `stats.py` | clustered bootstrap CIs → `out/compare_ci.csv` |
| `wallet_payouts.py` | REWARD / MAKER_REBATE payouts of the top makers → `out/politics_wallet_payouts.csv` |
| `risk.py` | jumpiness, jump-window and resolution markouts → `out/risk_*.csv`, `out/risk_summary.json` |
| `live_books.py {politics,crypto}` | live spread, depth and touch dynamics, and the reward share at the touch → `out/books_*.csv`, `out/rewards_live.csv`, `out/books_politics_summary.json` |
| `recorder_politics.toml`, `run_recorder.py` | `polyrec` config (40 markets, 5 sources) and a launcher with the CLOB User-Agent patch (aiohttp with polyrec's browser UA gets a Cloudflare 403 on `/book`; no shared code changed) |
| `live_extract.py` | `extract.py` wrapper for windows ending at the chain head (`plan()` raises for today's partial day) |
| `rec_out/` (git-ignored), `out/rec_summary/` | raw recording; compact hourly summaries are committed |

**Shared-code notes, nothing edited:**
- `recorder/polyrec/http.py`: the browser User-Agent gets a Cloudflare 403 from `clob.polymarket.com` via
  aiohttp.
- `polyrec`: a market whose gamma `endDate` is in the past but which is still open (Brazil after round 1, LA
  mayor) is never subscribed. The workaround is `linger_s`/`post_close_s` per source.
- `onchain/extract.py plan()` fails for today's partial day.
- `onchain/to_trades.py` builds its markets.parquet with crypto-only columns.
