# Protocol decisions made during development

Each entry was recorded before any test-split run, together with the reason
it was needed. Every entry extends `EVALUATION.md` and supersedes it where
the two differ.

## D1. Comparators, given that no baseline has a robust edge on val (2026-10-15)

**What we saw.** We tuned four baselines on val (64 TPE trials each; central
fills, 1 s latency). Every one that actually trades loses money: spread
capture is smaller than the 60 s markout. The few positive tuned results are:
- within noise;
- concentrated in a handful of markets;
- directional under T6;
- not robust to the fill model or latency.

So the tuner drives every baseline towards not quoting at all
(`baselines/eval_val.json`).

**Decision.**
1. **Primary comparator.** The best tuned baseline, with no-quote included as
   a candidate. Concretely, the RL policy has to beat about $0 per market.
   The primary endpoint in EVALUATION.md is unchanged.
2. **Secondary comparators.** Baselines that are required to keep trading:
   - at least 5 fills per market on average;
   - two-sided uptime of at least 0.25;
   - tuned with the same 64-trial budget on the frozen env.

   These answer a different question: does RL quote better than the classical
   rules when both are forced to make markets? They are reported alongside the
   primary result, not instead of it.
3. **When the RL result counts as market making.** RL counts as "making
   markets" only if it passes every one of these:
   - at least 5 fills per market;
   - the T6 directional share is at most 50%;
   - the T5 artifact share is at most 25%.

   If it fails any of them, we report what the policy actually does (for
   example "learns to mostly not quote"), whatever its P&L.
4. **Re-tuning on the frozen env.** (Frozen 2026-10-15: `mm.h` sha256
   `746dd7dea250eee035a8cb08a7be479eacf2d44ac1b7251503bc0d0e468566a9`; build
   hashes are in ENV.md.) The baseline tuning above ran on an env
   version from before the freeze. Every baseline will be re-tuned on the
   frozen env and data hashes before the final comparison, and all earlier
   trials stay in the multiple-testing count.

## D2. T5 artifact flags: a sweep through the touch is normal, not an artifact (2026-10-15)

**What we saw.** The env sets a flag on any fill whose trade printed outside
the snapshot's `[bid, ask]`. The env owner checked this on 40 train markets
(central fills, latency 1). The flag compares each trade against the book at
the match row, which is the pre-trade book, as intended. The flag covers 83%
of fill volume for quotes that join the touch, and 93% for quotes 4 ticks
behind it.

The reason is that most of our fill volume comes from takers sweeping through
the touch. A print exactly at our price is mostly absorbed by the displayed
queue ahead of us, which leaves only 5-10% of our volume. These fills are how
a resting order normally gets filled, so they are not an artifact of the
replay. If EVALUATION.md T5 is read literally ("more than 25% of P&L from one
flag makes the result invalid"), almost every policy would be invalid.

**Decision.** T5's 25% P&L rule applies to the flags that point to replay
artifacts:
- `inspread`: a print on the wrong side of the touch, which the snapshot
  cannot explain;
- crossed or stale book;
- one-sided book;
- the last 30 s before close.

The through-the-touch flag is still logged and reported, but it is not an
exploit detector.

## D3. Training run settings (recorded 2026-10-15, before the main sweep)

- **Sweep.** A fresh sweep named `main` with 64 trials, the same budget as
  each baseline, at 25M steps per trial. Sweep `s1` was stopped by the user,
  so it is aborted. Its two finished trials still count towards N.
- **Selection.** The config with the best mean val score over 3 seeds, among
  the top 4 sweep configs (`--reseed-top 4`).
  - Only configs that pass D1.3 can be selected (`--require-mm`): at least 5
    fills per market, and a T6 directional share of at most 0.5.
  - There is no CVaR floor. The env enforces the inventory cap Q = 100 shares.
- **Final seeds.** Seeds 100–104 at 100M steps each.
- **Conditions.** Central fills, 1 s latency, maker rebate 20%, liquidity
  rewards off. Every other parameter is an env default from ENV.md.
- **If PPO underperforms** (task "diagnose and improve"), every follow-up run
  goes into the same val ledger. The improvements to try, in order:
  1. fair-value mark;
  2. warm start from imitating the fair-value baseline, with a decaying KL
     penalty;
  3. an "off" prior at initialisation;
  4. longer runs.

  The test split stays untouched until a single final configuration is
  frozen.

## D4. Overnight improvement loop and when to stop (recorded 2026-10-15 ~07:40 UTC)

The user asked for the work to keep improving overnight without supervision.
These rules stop that from turning into unbounded searching on validation,
and keep the test day unseen.

- **What may run:** every training and selection run uses only the train and
  val splits. Every val evaluation is logged to a ledger, and all of them count
  towards N.
- **Improvement steps:** they follow the D3 order. Each one is a named round
  in the ledger:
  1. warm start by imitating the fair-value baseline, with a decaying KL
     penalty;
  2. an "off" prior at initialisation;
  3. fair-value mark;
  4. longer runs;
  5. pretraining on the simulator, only after its validation gate passes;
  6. recurrent policy.
- **Stopping rule:** stop improving at whichever comes first:
  - **15:30 UTC on 2026-10-15**;
  - two rounds in a row that don't raise the best mean val score (over 3
    seeds, among D1.3-passing configs) by more than its seed standard error.
- **Finishing:** then freeze the best config:
  1. train 5 final seeds;
  2. write TEST_PLAN.json and record its hash;
  3. run the test split **once**, under the EVALUATION.md protocol as
     amended by D1–D3;
  4. publish final/RESULTS.md.

  The result is reported whatever it is, including "PPO does not beat not
  quoting".

## D5. Extra improvement round R7: credit each fill with its markout (recorded 2026-10-15 08:45 UTC)

**What we saw.** At the 08:39 check, about 44 of the 64 main-sweep trials had
finished. All of them lose $2.7–5.9 per market on val. Their sampled policies
stay close to uniform: price-head entropy is 1.6–1.94, against a maximum of
1.95. Each greedy policy fills 40–75 times per market and loses those fills to
adverse selection.

**Our reading.** The dense mark-to-market reward lets inventory revaluation
noise swamp the per-action signal, across episodes of 300–14,400 steps.

**R7.**
- Train with a new env reward mode 2, a training signal only:
  - each of our fills is credited at once with `size × side × (mark_{t+H} − price)`
    plus its rebate, with H = 60 s and the mark at fair value;
  - an inventory-risk penalty is added;
  - there is no per-step mark-to-market.
- The reward may use data after t because it is not an observation. The
  policy still sees only information up to t.
- The objective is a surrogate. Selection and every reported number still use
  true P&L, unchanged.
- Modes 0 and 1 must stay bit-identical, so the env hash changes only by
  adding this mode.

**Order.** R7 runs right after R2 (R1, R2, R7, R3, R4, R5, R6). It follows the
same improvement test and the same stopping rule as D4.

**D5 follow-up (09:36 UTC):** reward_mode 2 was installed by atomic swap. The
new `mm.h` sha256 is `9189f83adbb3f42ab803aca998ff97954c7cc092ffd65401d2b835abaa98a8e9`;
it was `746dd7de…`. Reward modes 0 and 1 are bit-identical to the frozen core:
32 of 32 runs matched exactly, with maximum difference 0. Baseline and main-sweep
results obtained on `746dd7de` therefore stay valid. Hash references are updated
in train/run_all.py, baselines/tune.py and puffer5/README.md.

## D6. Amending the stopping rule after R1 and R2 (recorded 2026-10-15 10:50 UTC, after seeing R1/R2)

**What happened.** `auto` stopped after R1 (BC warm start) and R2 (off-prior),
because neither passed the improvement test, and that met D4's "two
non-improving rounds in a row" rule.
- R7 (D5) never ran. D5 inserted it after R2, but nobody checked how that
  position interacted with the streak rule written for D4's original order.
- R7 is the round aimed at the failure we diagnosed (reward noise / credit
  assignment).
- The user asked for improvement work to continue at full throughput until
  morning.

**This amendment was made after seeing R1 and R2.** We are stating that
plainly. It is acceptable because every val evaluation stays in the ledger
and counts towards N, selection still uses only val, and the test split is
still run exactly once.

**Amendment.**
1. The finalize step that had started was stopped. It was retraining seeds of
   the main config, which already has 5 seeds.
2. Rounds now run in this order: R7, R3, R4, R6. R5 runs only if the sim gate
   passes.
3. The streak counter restarts at R7, and the 15:30 UTC deadline is unchanged.
   Each round uses the same improvement test and the same D1.3 eligibility
   (`--require-mm`).
4. After that comes finalize, then the single test run.

**R1 results to report regardless of selection.** The BC-from-`tuned`
fair_value variants reach a seed mean of about +$110 to +$144 total on val,
i.e. about +$0.45–0.58 per market. They fill only about 4.0–4.7 times per
market. That is below D1.3's threshold of 5, so they cannot be selected.
Their teacher is itself directional (D1). We will report them as "RL matches
the low-activity fair-value baseline", not as market making.

## D7. Protocol v2: improvement continues, with a new untouched test set (recorded 2026-10-15 ~19:15 UTC)

The v1 test day (Aug 24) has been used once (final/RESULTS.md, verdict FAIL). The user asked for
further improvement. Aug 24 can no longer serve as a test, so v2 sets up a new test.

**Data roles in v2:**

| Role | Data | Notes |
|---|---|---|
| Train | Aug 21–22 (431 markets), as before | |
| Validation | Aug 23 + Aug 24 (374 markets) | Aug 24 counts as validation from now on. Its v1 test result is already public, and it is never again called a test. |
| v2 test (untouched) | Marketlens `polymarket-btc-5m-l2-depth` on Hugging Face: every BTC 5-minute up/down market of 2026-06-15 (288 markets) | Converted to our format in `data/build_jun15/`. No policy, baseline or tuning run may touch it before the v2 plan is frozen. It is evaluated once. |

**Caveats on the v2 test, stated before any result exists:**
- It covers BTC 5-minute markets only.
- It comes from a different capture: Marketlens millisecond deltas instead of PolyOrderBooks
  1-second snapshots.
- It is out-of-period, two months before training, rather than walk-forward.

The v2 test therefore measures robustness under a shift in data source and time period. It does
not measure the full portfolio.

**What may change in v2 (selected on validation only, with every evaluation in the ledgers):**
- features (Binance sub-second order flow);
- seed ensembling;
- per-contract-length specialisation;
- reward and objective (fill markout, CVaR).

**Unchanged:** the comparators (D1), the T5 rules (D2) and the D1.3 rule for what counts as
making markets.

**Stopping and freezing:** the v2 plan is frozen and hashed before the Jun 15 evaluation, in the
same way as v1. The test runs once, and every result is reported.

**D7 stopping rule.** v2 improvement stops at whichever comes first:
- two consecutive v2 steps whose best seed-mean val-v2 P&L, among D1.3-passing methods, does not
  beat the previous best by more than its seed SE;
- 2026-10-16 03:00 UTC.

The v2 plan is then frozen and Jun 15 is evaluated once.
