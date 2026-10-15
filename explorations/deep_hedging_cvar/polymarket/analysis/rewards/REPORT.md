# Maker revenue on Polymarket crypto up/down: spread vs maker rebates vs liquidity rewards

Written 2026-10-15. **All empirical numbers come from the train split only**:
431 markets (248 × 5m, 146 × 15m, 37 × 4h) whose whole window lies inside
Aug 21–22 2026 UTC. No market from Aug 23 or Aug 24 was opened, and no wallet
payouts for those epochs were read.

Adverse selection is not estimated here; see `../fair_value/REPORT.md` §4.

Source tags used below:
- **P**: primary source (docs.polymarket.com, a CLOB or gamma API response,
  or py-clob-client).
- **E**: checked against data in this study.
- **S**: secondary source.

## 1. Rules: what applied to our data (Aug 21–24) vs now

| Item | Aug 21–24 2026 (replay) | Now (2026-10-15, global CLOB) | Tag |
|---|---|---|---|
| Taker fee | `C·0.07·p(1−p)`; makers pay nothing. gamma `feeType=crypto_fees_v2`, `feeSchedule={rate:0.07, exponent:1, takerOnly:true, rebateRate:0.2}` | same | P, E (reproduces the fee table) |
| Maker rebate | 20% of each maker fill's own fee-equivalent `C·0.07·p(1−p)`, computed per market and paid daily (≥ $1). Measured `rebated/fee_eq` = **0.2000** for most makers. Some large makers got **0.350**, seen from Aug 12 to Sep 10 and on no date from Sep 24 onward; this is undocumented. | 0.2000 for every maker sampled (Sep 24, Oct 1/8/13/14) | P, E (`/rebates/current`) |
| Liquidity reward pool | **"Crypto TWAP Rewards"**: $1M across the 5m/15m/4h TWAP markets for August. Split: 5m $550k, 15m $350k, 4h $100k, weighted by coin (BTC largest; ZEC not included). Wallet REWARD payouts show it paid for epochs Aug 7–12 and Aug 15–Sep 5. The docs now say the program has ended. | **No pool.** Crypto up/down markets are absent from CLOB `/rewards/markets/current` and `/sampling-markets`. | P (docs), E (`/activity` REWARD rows) |
| Reward scorer parameters | `rewardsMinSize=50`, **`rewardsMaxSpread=1.5`¢**, `r.moas=1`. These are inferred from the configs that survive on about 20% of closed incentivised markets. | 50 / **4.5**¢, `moas=30` | P (API), inference |
| Reward scoring | `S=((v−s)/v)²·size`, with `s` measured from the size-cutoff-adjusted mid. `Q_one` = bids(m) + asks(m′); `Q_two` = asks(m) + bids(m′). Inside the [0.10, 0.90] band: `Q_min=max(min(Q1,Q2), max(Q1,Q2)/3)`. Outside it: `Q_min=min(Q1,Q2)`. One sample per minute at a random offset; daily epochs, paid at 00:00 UTC. | unchanged | P |
| Tick | 0.01; 0.001 when p < 0.04 or p > 0.96 | 0.01 | E (L2 price grid) |

**The changes on Oct 15 (end of the 17% weekly crypto maker rebate) and
Oct 19 (volume-tiered maker rebates) apply to Polymarket US**, a separate
venue, not to the global CLOB our data comes from. Global rebates sampled on
Oct 13–14 were exactly 20%. Sources: docs.polymarket.us fees and changelog;
secondary press agrees.

None of the endpoints keeps the August allocation for a closed market. The
**per-market August pool is therefore an assumption**: each monthly cap from
the docs, spread evenly over every market of that coin and contract length.

## 2. Empirics (train only)

### 2a. Taker fees and the rebate pool

| | 5m | 15m | 4h |
|---|---|---|---|
| taker fees / market, mean (median) | $95.4 ($8.3) | $60.9 ($6.9) | $73.5 ($23.9) |
| mean fees per market: BTC / alts / ZEC | 600 / 17.8 / 0.00 | 374 / 19.7 / 0.01 | 331 / 55.7 / 0.01 |
| share of fees inside the window | 91% | 93% | 98% |
| 20% rebate pool / market, mean | $19.1 | $12.2 | $14.7 |
| rebate per maker-filled share at 20% (at 35%) | 0.22¢ (0.38¢) | 0.23¢ (0.41¢) | 0.19¢ (0.34¢) |

Total maker fee-equivalent ($35,239) matches total taker fees ($35,274).
Paying makers a pro-rata share of a 20% pool is therefore the same as paying
each maker 20% of its own fee-equivalent.

### 2b. Competing for liquidity rewards

How this was measured:
- One sample per minute at a random offset, using the last Up-book snapshot.
- Competitors' scores are bracketed, because who owns which order is not
  observed.
- Our hypothetical maker posts X shares on each side at the k-th 1¢ price
  away from the adjusted mid.

| setup (August config, X=50) | 5m | 15m | 4h |
|---|---|---|---|
| samples with both sides scored | 30% | 32% | 44% |
| our share at k=1 | 0.62 | 0.69 | 0.72 |
| $/h at k=1 | 0.17 | 0.37 | 1.62 |
| $/h at k=2 | 0.07 | 0.14 | 0.75 |
| per quoted share-hour | 0.17¢ | 0.37¢ | 1.6¢ |
| BTC $/h at k=1 | 0.38 | 1.32 | 6.04 |

**Only 0.35% (5m), 1.0% (15m) and 14% (4h) of a market's reward pool is paid
during the window our L2 data covers.** The rest accrues in the roughly 24 h
before the window opens, which is not in the dataset.

### 2c. Revenue per share, and the revenue split

| | 5m | 15m | 4h |
|---|---|---|---|
| maker edge vs pre-fill mid per filled share, size-weighted | 0.19¢ | 0.87¢ | 0.84¢ |
| of which BTC | 0.09¢ | 0.80¢ | 0.48¢ |
| rebate per filled share (20%) | 0.22¢ | 0.23¢ | 0.19¢ |
| reward per quoted share-hour | 0.17¢ | 0.37¢ | 1.62¢ |
| spread : rebate : reward, at the pool level | 33 : 47 : 20 | 60 : 18 : 22 | 45 : 11 : 44 |

For the 25 largest makers in the train markets:
- spread edge was $10.5k, actual rebates $4.7k, and attributable rewards
  about $1.9k;
- that is a spread : rebate : reward split of about **61 : 28 : 11**, before
  adverse selection.

**Conclusions:**
- **5m:** rebates are the largest component. For BTC 5m, the rebate per
  share (0.22¢) is bigger than the measured edge (0.09¢).
- **15m:** spread dominates.
- **4h:** spread and the (assumed) reward pool are comparable.
- **Inside the replay window**, rewards are negligible for 5m and 15m and
  second-order for 4h. Rebates scale with fills, so they always matter.
- **Live trading today:** rewards are 0 and the rebate is 20%.

## 3. Uncertainty

1. Competitors' reward share is bracketed (hi/lo differ by about 15–35%), and
   the replay book does not react to our quotes.
2. Per-market pool sizes are assumed.
3. The August max spread (1.5¢ vs 4.5¢) is inferred; 4.5¢ is shipped as a
   sensitivity preset.
4. The definition of the adjusted mid is an interpretation.
5. The 35% rebate tier is undocumented and specific to some makers. It is a
   sensitivity case only.
6. `moas` is read as a minimum order age for scoring.
7. The maker edge is measured against stale snapshots.

## 4. Env specification

1. **Rebate.** On each of our maker fills, add
   `rebate = ρ·C·θ·p(1−p)`, with θ = 0.07 and ρ = 0.20. Credit it at fill
   time.
2. **Reward.** At one random second per minute:
   `reward = π_m · Q_min/(Q_min + Q_comp)`, where π_m = pool_m / life_minutes.
   - Our `Q_min` is computed from our resting orders with size ≥ min size and
     age ≥ moas.
   - `Q_comp` is computed from the replay book (hi bound).
   - Reference implementation: `reward_terms.reward_accrual`.
3. **Observations:** per resting quote, d/v, S(d), the min-size flag and the
   age flag. Per market: the [0.10, 0.90] band flag, our share, π_m, and the
   rebate per fill at our bid and at our ask.
4. **Regimes:**
   - `AUG_2026` (replay fidelity): ρ = 0.20, rewards on, v = 1.5¢, min 50,
     moas 1.
   - `LIVE_2026_10` (deployment): ρ = 0.20, rewards off, v = 4.5¢, min 50,
     moas 30.
   - Sensitivity presets: 35% rebate; v = 4.5¢.

## 5. Reproduce

Scripts, run in this order:
1. `rules_snapshot.py`
2. `verify_rebate_rate.py`
3. `empirics.py`
4. `maker_decomp.py`
5. `reward_terms.py`: env reference implementation, with a self-test.

`common.py` holds the train filter and loaders. Outputs go to `out/`; the
`cache/` folder (~97 MB, raw doc and API copies) is git-ignored.

Sources:
- docs.polymarket.com: programs/liquidity-rewards, programs/maker-rebates,
  trading/fees, changelog.
- CLOB API references: `/rewards/markets/current`, `/clob-markets`,
  `/rebates/current`.
- py-clob-client `endpoints.py`.
- docs.polymarket.us: fees, changelog.
