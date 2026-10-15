# Polymarket sports mechanics (docs.polymarket.com + keyless API pulls, 2026-10-15)

- **In-play delay** (concepts/order-lifecycle; market-details `secondsDelay`): only *marketable* orders wait
  `secondsDelay` seconds (status `delayed`) before matching; resting orders post immediately. "During either delay,
  the order is pending and cannot be canceled." Risk/market checks are re-run at expiry. Gamma: secondsDelay 1 on
  ~97% of sports game markets in Oct 1-14 (0 on MLB/most NBA, 3 on a few). Makers can cancel freely, so the delay
  protects makers against takers by 1 s (unlike crypto, where the taker delay is 50 ms).
- **Game-start wipe** (concepts/markets-events): "outstanding limit orders are automatically cancelled once the game
  begins ... if a game starts earlier than scheduled, orders may not be cleared in time." `clearBookOnStart=true` on
  every sports game market in the census.
- **Fees** (trading/fees): taker fee C·rate·p(1-p), makers pay nothing. Sports `sports_fees_v3` rate 0.05, rebate 15%
  (changelog Jul 10 2026: 0.03→0.05, rebate 25%→15%). Undocumented `sports_fees_nfl_cfb_oct26`: NFL/CFB moneylines
  rate 0.03, rebate 15%. NFL/CFB spreads/totals/props are `zero_fees` (rate 0, rebate 0) = 10% of sports game volume.
- **Maker rebates** (programs/maker-rebates): pro-rata share of the market's rebate pool by fee-equivalent, daily in
  pUSD, $1 minimum.
- **Liquidity rewards** (programs/liquidity-rewards): S=((v-s)/v)²·b·size with an undocumented "in-game multiplier" b;
  sampled once per minute; daily. Live CLOB `/rewards/markets/current`: $210.8k/day overall; sports-tagged 7,825
  markets ≈ $49.4k/day (esports $23.2k incl. one LoL Worlds match at $3.4k/day; NFL $10.5k over 2,480 markets;
  soccer $7.3k; NHL $2.5k; CFB $2.2k; NBA $1.35k; MLB $0.9k). Typical per-market daily rates $1/$10/$50; max spread
  4.5¢ (some 5.5¢/2.5¢), min size 20-100. NFL/CFB 1H spreads carry $514-1,886/day with min size 1000, max spread
  2.5¢. Last big sports program in docs: March Madness ($60k/day on live moneylines). No Sep-Oct 2026 sports program.
- **Holding rewards** 4% APY exist but `holdingRewardsEnabled=false` on all sampled sports markets.
- **Rate limits**: 40 orders/s (burst 60), 80 cancels/s per signer (standard tier), warning mode since Jul 24 2026.
- **Ticks**: 0.01 or 0.001 (0.001 near the extremes); World Cup used 0.0025.
- **Live scores**: `wss://sports-api.polymarket.com/ws` streams every game update (score, period, elapsed, live/ended),
  no subscription; gamma events carry `score/period/elapsed`, with OpticOdds (soccer) / PandaScore, GRID (esports) ids.
