# Calibrated generative simulator (spec)

Purpose: **pretraining and stress tests** for the market-making policy, on far more fair-value
paths than four days of replay contain. Not for evaluation: no number of record comes from the
sim (EVALUATION.md). Sim results are trusted for anything at all only after the consistency test
of §5 passes. Same C env interface as the replay env (obs, action, fill engine, accounting), so a
policy moves between them unchanged; the fill engine and its three models are literally shared.

## 1. Underlying and fair value

- **Underlying:** real Binance 1 s closes (spot klines; HYPE from USD-M perp aggTrades, as in
  `../data/fetch_underlying.py`) from data.binance.vision, many months per coin, sampled as joint
  8-coin time slices so cross-coin correlation is real. **Excluded:** Aug 23-24 2026 (val/test)
  plus a 1-day embargo either side, so pretraining never sees the evaluation days' direction.
  Stress sets: top-decile realised-vol days, trending days, flash moves.
- **Settlement:** markets resolve on the Chainlink `<coin>-usd-twap-60s` stream (60 s TWAP at end
  vs `price_to_beat` at open), not Binance. Sim settles Up iff `TWAP_60(S) * exp(eps) > K`,
  eps ~ N(0, sigma_basis^2), with sigma_basis and the open-strike basis fitted from
  (`final_price`, `price_to_beat`, Binance) over all 805 markets. The TWAP matters: in the last
  60 s part of the outcome is already locked in, which changes the digital's delta near close.
- **Fair value:** digital option `FV_t = P(S_T > K | F_t)` from `../analysis/fair_value/` (vol from
  a trailing EWMA of 1 s returns, jump/fat-tail correction as that study recommends). In the sim it
  is computed exactly as the policy's observation computes it, so there is no hidden information.

## 2. Polymarket price process

- **Mid:** on the logit scale, `logit(m_t) = logit(FV_{t-L}) + b(tau) + u_t`, u_t an AR(1)
  (phi_u, sigma_u) and L a lag. The mid updates only at book-change events (Poisson rate
  lambda_upd(tau, len)), otherwise holds: Polymarket mids are sticky, and the staleness is what
  informed takers trade on. Fit L (cross-correlation peak of mid changes vs FV changes),
  phi_u, sigma_u, b(tau) and lambda_upd per contract length on train-day replay.
- **Competing depth (L2 model):** spread in ticks ~ categorical conditional on
  (tau bucket, |FV-0.5| bucket, vol bucket, length); depth per level ~ log-normal with a fitted
  level profile and AR(1) persistence in log depth; P(one-sided) logistic in (tau, |FV-0.5|,
  length) to reproduce the 76-85% one-sided final minute; tick size 0.01, switching to 0.001 once price goes beyond ~0.96 / ~0.04 (and staying).
  Book resilience: after a print removes depth, refill at rate kappa toward the profile.
  Crossed/stale states are **not** generated (they are capture artefacts); a separate stress
  mode injects them at the observed rates to check the policy does not depend on them.
- Up/Down are one book in the Up frame (complementary matching), as in replay.

## 3. Taker flow and fills

- **Arrivals:** two components, both marked point processes on 1 s resolution.
  *Noise takers:* Poisson rate `lambda_n(tau, len, coin, hour-of-day)`, direction 50/50, size
  log-normal (fitted). *Informed takers:* Hawkes intensity
  `lambda_i(t) = mu_i + a * g(gap_t) + sum_k alpha * exp(-beta (t - t_k))`, where
  `gap_t = FV_t - quote` on the side being hit (buy if FV - best ask > c, sell if best bid - FV > c),
  `g` softplus, so takers react to **our** stale quotes too (the one thing replay cannot do).
  Direction and size depend on |gap|, tau and vol. The informed share is
  `lambda_i / (lambda_i + lambda_n)`.
- **Fill engine:** identical code to replay (sim prints carry no block-time delay); our quote sits in a queue at its level (central:
  behind displayed depth, advance on prints at our price and pro-rata cancels), prints walk the
  combined book in price order, latency 0-2 s on placement and cancel.
- **Competitor response (optional, randomised):** with prob p_react per second other makers
  re-price toward FV and match/undercut our improved quotes, so improving the touch is not free.

## 4. Parameters: what is fitted from which data

| parameter | data | method |
|---|---|---|
| sigma_basis, open basis | markets.parquet (all 805) + Binance | moments of log(final/Binance) |
| L, phi_u, sigma_u, b(tau), lambda_upd | L2 replay, **train days only** | least squares / ACF on logit mid |
| spread, depth profile, P(one-sided), kappa | L2 replay, train days | empirical conditional tables, MLE |
| lambda_n, size law, hour-of-day | on-chain fills (data-api / chain) over **months**, excl. Aug 23-24 +/- 1 day | Poisson GLM |
| mu_i, a, c, alpha, beta, informed share | on-chain fills over months joined to Binance FV; cross-checked on train-day trades | Hawkes MLE with gap covariate |
| fee/rebate/reward rules | markets.parquet fields, `../analysis/rewards/` | as documented per market |

Order-book parameters come only from the 2 train days (the only L2 we have), so they get the
widest randomisation. Flow parameters come from months of fills, so they can be conditioned on vol
regime. Nothing is fitted on val or test data.

## 5. Sim-vs-replay consistency test (gate)

Run the four baselines of EVALUATION.md §4 (no-quote, symmetric at touch, AS/GLFT, FV quoter; fixed
parameters) on the **train days** in two ways: (a) replay, central fills, 1 s latency; (b) sim
**conditioned on the same Binance paths and the same market schedule**, 200 replications. Pass
criteria, all required:
1. For each baseline, replay total and median per-market P&L inside the sim 90% predictive
   interval; per-market P&L distributions within a two-sample KS distance threshold fixed now
   (D < 0.15) with block structure respected (compare hourly MTM series, not iid markets).
2. Baselines' **rank order** is the same in sim and replay.
3. Attribution components (spread capture, markout at 5/30/60 s, inventory) and fill rates
   within the sim 90% band; the markout curve of the symmetric-at-touch baseline within the band.
4. Stylised facts match: spread and depth distributions by tau, one-sided fraction by tau, trade
   rate by tau, P(taker buys | gap) curve, autocorrelation of signed flow.
5. Out-of-sample: repeat 1-3 on the **val** day (not used for fitting). Never on test.
If any fails, the sim is used only as an initialiser for pretraining (followed by replay
fine-tuning), and no sim result is quoted. Sim fidelity is re-tested whenever a parameter changes.

## 6. Domain randomisation (per episode, uniform unless stated)

| knob | range |
|---|---|
| mid lag L | 0-3 s (fitted value in the middle) |
| logit noise sigma_u, update rate | x0.5-x2 |
| noise taker rate | x0.5-x2; size scale x0.5-x2 |
| informed share / gap sensitivity a, threshold c | informed share 10-60%; c +/-50% |
| Hawkes branching ratio | 0-0.8 |
| competing depth scale | x0.3-x3; P(one-sided) +/-50% (logit) |
| competitor reaction p_react | 0-0.5 |
| latency (placement, cancel) | 0-2 s each, independent |
| queue position on join | front / central / back (as fill models) |
| fee rate, rebate share, reward share | per-market schedule x0.5-x1.5; reward share 0-central |
| vol regime | Binance slice sampled stratified by realised-vol tercile |
| settlement basis sigma_basis | x0.5-x2 |

## 7. One policy, many markets, portfolio risk

- **Single policy** across 8 coins x 3 lengths: observation includes coin embedding, length
  one-hot, tau / length, FV, mid, book features, own inventory, and portfolio features below.
  Prices and sizes normalised so the same weights apply across coins.
- **Env structure:** one episode = a time slice (e.g. 4 h) with every market live in it as one
  agent (PufferLib multi-agent, shared weights). Replay only contains the sampled subset of
  markets; the sim generates the full schedule, so concurrency in sim is realistic.
- **Portfolio risk:** each position's dollar delta `d_j = q_j * dFV_j/dS * S_c(j)` (digital delta,
  large near the strike close to expiry) and jump-to-settlement loss `q_j * FV_j` or
  `q_j * (1 - FV_j)`. Combined risk `R = sqrt(d' Sigma d)` with Sigma the trailing 1-min return
  covariance across coins (5m/15m/4h on one coin share a row). Reward = MTM P&L
  - k_R * R * dt - k_J * (worst-case settlement loss over markets closing within 60 s); hard limits on
  |q_j| and R. Every agent observes R and its own marginal contribution dR/dq_j, so a BTC 5m quote
  skews against BTC 4h inventory.
- Reward shaping lives only in training; evaluation always uses raw P&L and the metrics of
  EVALUATION.md.
