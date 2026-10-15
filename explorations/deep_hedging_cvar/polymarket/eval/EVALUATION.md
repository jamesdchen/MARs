# Evaluation protocol (pre-registered)

Scope: an RL market maker (PPO, PufferLib 3.0, portable to 5.0) quoting bid/ask on Polymarket
crypto up/down binaries (5m / 15m / 4h; BTC ETH SOL XRP BNB DOGE HYPE ZEC), trained in a C env that
replays 1 s L2 snapshots and taker trades from 2026-08-21..24. The protocol is fixed **before any
validation number is seen** and changed only by a dated amendment at the end of this file. Helpers:
`stats.py` (tested in `test_stats.py`). The sim of `SIMULATOR.md` never produces a number of record.

## 1. Data, splits, units

- Splits by UTC day: **train** Aug 21-22, **val** Aug 23, **test** Aug 24, as built in
  `../data/build/splits.json` (split 3 = purged; 4h windows align to 00/04/../20 UTC, so the
  pipeline purges none). `stats.assign_split` re-derives the assignment independently, optionally
  with an embargo; any disagreement with the pipeline is a blocking bug. Purged markets are
  reported, never used.
- Everything fitted from data (feature normalisers, fair-value vol model, fill-model calibration,
  reward-share model, baseline parameters) is fitted on train only and frozen; there is no
  train+val refit before test. Trailing-window features may look back across a boundary (past data).
- Units: per-market P&L for distributions and pairing; for anything that needs a CI, a
  **mark-to-market (MTM) P&L time series on a zero-filled grid** (1 h primary, 15 min secondary),
  summed over *all* markets live in that bin (`stats.time_bins`). Inventory is marked to mid each
  bin and to the settlement value at close, so P&L is not lumped at settlement times.
- Frame: all accounting in the Up-token frame (Down BUY @ p == Up SELL @ 1-p, `stats.to_up_frame`);
  Up and Down share one book via complementary matching, and the env must use the combined book.

## 2. Threats to validity and mitigations

**T1 Fill optimism of replay.** Our queue position is unknown, the replay does not react to our
quotes (takers that hit us would partly have hit someone else; makers would re-price around us),
and trade-through rules decide which prints could have touched our level. Mitigation: every result
is reported under three fill models, all sharing the same code path (only parameters differ):

| model | queue on join | queue depletion | prints that fill us | size |
|---|---|---|---|---|
| optimistic | front of level | n/a | any taker print at or through our price, after latency | full print |
| central | back of displayed depth at our price | trades at our price + cancels pro-rata to our share of the level | at our price once queue ahead is gone; through our price fully | print size left after better levels |
| pessimistic | back of level | trades at our price only (cancels never advance us) | strictly through our price only | 50% of central, capped at 20% of the print |

**Trade-time alignment** is part of the fill model: trade timestamps are Polygon block times,
~3-4 s after the off-chain match (`../data/README.md`), and 786 prints fall after market end. A
print may fill our quote only if the quote was live at the print's *estimated match time*
(timestamp minus a delay d); d = 3.5 s in central, 0 s in optimistic, 5 s in pessimistic.
Prints after market end never fill. Plus a participation cap (our fills <= 25% of taker volume per market-minute in central) and a
**size sweep** (order size x0.5, x1, x2): P&L that grows linearly in size well beyond the displayed
depth is replay optimism, not edge. Quotes that improve the touch get no extra flow in any model.
The **headline is central**; a claim needs the sign to survive pessimistic. The C env is
cross-checked by an independent Python re-implementation of the three fill rules on 20 random
train markets: fills and P&L must match to 1e-9.

**T2 Latency.** Takers on these markets arbitrage the Binance-to-Polymarket lag, so a stale quote
is the main way to lose money. Every result is reported at quote placement / cancel delay
**0, 1, 2 s** (both directions: our new quote becomes live, and our cancel takes effect, only
after the delay). Headline = 1 s; 0 s is reported but can never be the headline. Observation
timing follows the build's causality conventions (row t uses the Binance kline with
`open_time = t-1` and the latest snapshot with `captured_at <= t`), plus the obs-latency of the
variant; a trade print is observable only at its block timestamp (no back-dating into obs). A **lookahead test** is mandatory:
perturb all data after time t in a copy of an episode and assert the policy's actions up to t are
bit-identical.

**T3 Effective sample size.** 805 markets over 4 days, but markets on 8 coins move together
(crypto beta), 5m/15m/4h windows on one coin share the same underlying path, and the test day is a
single day (~24 hourly blocks). iid-over-markets CIs would be far too narrow (`test_stats.py` shows
iid coverage of a nominal 90% CI falls to ~67% on AR(1) phi=0.6 data). Mitigation: all CIs are
**stationary block bootstraps over time bins spanning all coins and contract lengths**
(`stats.block_bootstrap`, block length by Politis-White with a floor of 2 h; 10,000 replicates).
Any test-day CI is a statement about that day, not about other regimes; generalisation needs §6.

**T4 Multiple testing (sweeps, seeds, early stopping).** Every evaluation on val (any config, seed,
checkpoint, fill/latency variant, baseline parameter set) is appended to `polymarket/results/trials.jsonl`
with the config sha256, seed and metrics; nothing is evaluated on val outside the ledger. The
reported N counts all of them. Selection is on val only, by the pre-registered rule: maximise the
**seed-mean** val total P&L under central fills at 1 s latency, subject to CVaR95 of per-market
P&L >= the best baseline's CVaR95 and max |inventory| within the risk limit; ties broken by the
pessimistic-fill P&L. Never select the best seed. Baselines get the **same trial budget** as the
policy. We also report the deflated Sharpe of the selected config on val (`stats.deflated_sharpe`,
N from the ledger). Test is touched once (§5).

**T5 Simulator-artifact exploitation.** Known artefacts of this data: crossed books (1.5-3.2% of
rows, one stale side), stale books (rows written only on change, full REST reconcile every 60 s,
dropped websocket messages), one-sided books (17% of 5m rows, 76-85% in the final minute of
5m/15m), end-of-window thinning, gaps between trade prints and the book, tick-size rules near 0/1,
and settlement on a Chainlink 60 s TWAP rather than Binance. The build forward-fills the last
uncrossed book over crossed timestamps (`crossed_raw`, growing `book_age`); fills against a
forward-filled book are disallowed in central/pessimistic, and in pessimistic also against any
book with `book_age` > 10 s. **Exploit detectors** (`stats.flag_pnl_share`) report the share of P&L
from fills made while: book crossed; book stale > 10 s; book one-sided; last 30 s / last 10 s of the
window; our quote > 2 ticks inside the touch; our quote at <= 0.02 or >= 0.98; print price outside
the snapshot's [bid, ask]. A result is **invalid** if any single flag accounts for > 25% of total
P&L, or P&L excluding all flagged fills changes sign. Also inspected: quoting/fill rates vs time to
close (a policy that only trades in the last seconds is suspect).

**T6 Directional gambling disguised as market making.** Four days of crypto direction can pay a
policy that leans one way and holds inventory into resolution. Attribution
(`stats.attribute_pnl`, components sum exactly to total P&L, checked against cash accounting):
spread capture vs mid at fill; adverse selection = markout to mid at +60 s (5 s and 30 s also
shown); inventory/resolution = holding from +60 s to settlement (split into carry to a pre-close
mid and the resolution jump); fees; maker rebates; liquidity rewards. Rules: a policy whose
positive P&L comes > 50% from inventory/resolution is classified **directional**, not MM, whatever
its total. Also reported: P&L split by whether Up or Down won; correlation of inventory with the
subsequent Binance return; a **mirror test** (swap Up/Down, p -> 1-p, negate underlying returns;
a market maker's P&L should be statistically unchanged; train with mirror augmentation).

**T7 Concentration.** Report (`stats.concentration`) top-1/5/10 share of P&L by market and by
hour, P&L excluding the top 5 markets and top 3 hours, effective number of markets, the number
needed for half the gross profit, and P&L by coin and contract length. A claim needs ex-top-5-
markets and ex-top-3-hours P&L to stay positive.

**T8 Rewards/rebates model.** Liquidity rewards depend on program rules (`rewards_min_size`,
`rewards_max_spread`, scoring function, our share vs competing qualifying depth, which we can only
estimate from L2). Fees are taker-only (`C * fee_rate * p * (1-p)`, crypto `fee_rate` 0.07); makers
pay 0 and share a rebate pool of 20% of taker fees, so our rebate depends on our share of maker
volume, again an estimate (see `../analysis/rewards/REPORT.md` when present). If the action space
lets the policy cross the spread, those trades pay the taker fee. Every table shows
P&L with rewards+rebates, and **without** them (the floor). A strategy that only pays through
estimated rewards is reported as such.

**T9 Non-stationarity.** Report test-day realised vol, trade counts and spread vs train/val days.
A policy is not claimed to work outside the observed regime until §6.

## 3. Metrics (all under 3 fill models x 3 latencies; headline central / 1 s)

P&L per day and per market (mean, median, distribution); P&L by contract length and coin; hourly
MTM Sharpe (per hour, not annualised) with block-bootstrap CI; **CVaR95 of per-market P&L**
(`stats.cvar`); max |inventory| per market and max combined portfolio delta; fill rate (fills per
quoted market-minute, and our share of taker volume); quote uptime (fraction of live seconds with a
two-sided quote; one-sided uptime separately); the **attribution table** (§T6) per method; markout
curves (0-60 s); exploit-flag table (§T5); concentration table (§T7); the trial count N.

## 4. Comparisons

Baselines, evaluated on identical markets, data and fill/latency variants:
1. **no-quote** (P&L 0; sanity anchor and reward-only check);
2. **symmetric at touch** (join best bid/ask, fixed size, inventory limit);
3. **Avellaneda-Stoikov / GLFT** around mid (risk aversion, intensity params, inventory skew);
4. **fair-value quoter**: quote around the digital-option fair value from Binance
   (`../analysis/fair_value/`), half-spread and skew tuned; this is the bar the RL policy must beat.

All baselines are tuned on val with the same trial budget as the policy. RL: >= 5 training seeds
for the selected config; report mean, min/max across seeds, and block-bootstrap CIs. Policy vs
baseline uses **paired differences on identical markets** (`stats.paired_difference_ci`:
per-hour differences of MTM P&L, block-bootstrapped), which removes the shared market shock
(the test shows paired CIs < 30% the width of unpaired ones under a common factor).

**Primary endpoint (one, fixed now):** test-day total P&L of the selected RL config (seed mean)
minus the best tuned baseline, central fills, 1 s latency, rewards+rebates included; 95%
stationary block-bootstrap CI over hourly bins. **Success** requires all of: CI lower bound > 0;
point estimate > 0 under pessimistic fills at 1 s and under central fills at 2 s; not
classified directional (T6); no invalidating exploit flag (T5); ex-top-5 / ex-top-3h P&L > 0
(T7). Everything else is secondary and descriptive.

## 5. Test-time procedure (run once)

1. Freeze code (git commit hash), configs, checkpoints and baseline parameters, exactly as
   selected on val (no refit).
2. Write `TEST_PLAN.json`: commit hash; methods with config sha256 and checkpoint paths + sha256
   (every seed); baseline parameter sets; metrics; fill models x latencies x size multipliers;
   bootstrap settings (bin, block-length rule, replicates, seed); primary endpoint and success
   rule; ledger trial count N; expected test-data sha256. Record `sha256(TEST_PLAN.json)`
   (`stats.sha256_file`) in this file's amendment log and in a commit message before running.
3. Verify the test-day data: recompute the sha256 of the test rows (`meta.json.test_rows`
   range of `build/ticks.bin`, as `../data/load.py` does) and of `markets.bin`, compare with
   `../data/build/meta.json` and with the hash in TEST_PLAN.json; abort on mismatch.
4. Run every entry of the plan once, in one script, writing `results/test_report.json` that
   embeds the plan hash. No re-runs after seeing output; a crash is fixed only if the fix touches
   no evaluation logic, and the restart is logged.
5. Report everything in the plan, including failed endpoints and every variant.

## 6. After the first test: walk-forward and live shadow

- As more days are captured, **walk-forward**: train on days [1..k], select on k+1, test on k+2,
  roll by one day; never revisit an old test day. Report the distribution of per-day paired
  differences across folds (days are the honest unit of replication for regime risk).
- **Live shadow quoting** is the final reality check: run the frozen policy on live books and
  Binance with the production latency, log the quotes it *would* post and, from the trade tape,
  which would have filled under each fill model; then a small real-money pilot with hard limits.
  Compare realised fills/markouts with the replay central model; a gap > 1 CI kills the claim.

## Amendment log

- 2026-10-15: protocol written (no val or test results seen).
- 2026-10-15: amendments D1–D6 are recorded in eval/DECISIONS.md. D6 was made after R1/R2 val
  results were seen, and that file says so.
- 2026-10-15 13:46 UTC: TEST_PLAN.json frozen, sha256
  `75915990669c7038e81649a4e609ec8c776b0f264f0497fe390113e639eb4b6d`.
  RL: 5 seeds of R7 `c02_h120`. Comparator: baseline/fair_value. Ledger N = 1489 val
  evaluations over 1100 configs.
