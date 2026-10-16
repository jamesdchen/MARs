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

## D8. Parallel remote workers for v2 (recorded 2026-10-15 ~19:35 UTC, before any v2 result)

The user asked for more parallelism. Five extra cloud sessions (w1–w5), each in its own 4-core
container, run separate v2 research lines under `train/workers/PROTOCOL.md`:

| Worker | Research line |
|---|---|
| w1 | risk-aware objectives on reward mode 2 |
| w2 | per-contract-length specialists |
| w3 | architecture: recurrent PufferNet replica and larger networks |
| w4 | a non-RL toxicity-aware quoter (learned fill-toxicity classifier plus rules) |
| w5 | behaviour-cloning warm starts from other teachers, with markout reward |

Locally, the coordinator runs ensembling, length filters, the main markout sweep, sub-second
features and the Jun 15 build.

**Shared rules for every worker and local run:**
- the same frozen env and data, which each worker verifies by hash;
- selection on val v2 only, never Jun 15;
- each worker writes its own ledger (`train/workers/<name>/trials.jsonl`). Every row counts
  towards N.

**Selection across everything.** The single v2 method is the one with the best seed-mean val-v2
P&L (central fills, 1 s) among methods that:
- pass D1.3;
- have at least 3 seeds.

The coordinator then trains or collects 5 seeds of that method, or uses a pre-specified ensemble
of them, and freezes the plan.

**Unchanged:** D7's stopping rule and the 03:00 UTC deadline.

## D9. Exploring a pivot to other market categories (recorded 2026-10-15 ~19:35 UTC)

The user asked whether market making might work better outside crypto up/down, e.g. in sports
or politics. Three workers study this using keyless data: complete on-chain fills, price history,
live books and reward configs.

| Worker | Scope |
|---|---|
| w6 | politics, economics, geopolitics |
| w7 | census of maker economics across every category |
| w8 | sports deep-dive |

This is exploratory analysis. It selects no model, and it does not affect the crypto v2 protocol
(D7/D8), which continues as planned.

**If we pivot.** If the census shows a category with clearly positive maker economics and we
pivot to it, a new protocol v3 will be pre-registered **before** any model is tuned for that
category. It will fix:
- that category's data;
- its train/validation/test split by date;
- a test set that nobody has touched.

Nothing learned from the census is reported as a test result.

## D10. Rows of the v2 test with no book (recorded 2026-10-15 ~20:00 UTC, before any Jun 15 strategy result)

**What the build found.** 58 of the 288 Jun 15 markets have no order book for the first 3–201 s
after open: 2,987 rows (3.4%) with `book_valid=0`. v1 had the same pattern in 57 of its 471 5m
markets (1.8% of 5m rows), and the env treats both data sets identically.

**Decision: no change to the data or the env, so treatment stays consistent with train, val and
v1.**
- The existing T5 detector already covers fills on these rows through its `onesided` and
  `crossed_or_stale` artifact flags. A result where flagged fills carry more than 25% of the fill
  edge is invalid.
- In addition, the v2 report will show the method's P&L once more with every fill on a
  `book_valid=0` row removed, as a sensitivity check. This is reported, not used to judge.

**Two cosmetic leftovers, not fixed:**
- The env's test banner still mentions "Aug 24", because the text is hard-coded in `mm.h`. It is
  not changed, so the frozen hash stays the same.
- `book_age` on Jun 15 measures time since the last feed message. That differs from the v1
  capture and is part of the source-shift caveat in D7.

## D11. Protocol v3: pivot to pre-game sports and election reward farming (recorded 2026-10-15 ~20:25 UTC, before any v3 tuning)

The user chose to pivot to both tracks named by the census (`analysis/census/REPORT.md`,
`train/PIVOT.md`). Crypto v2 tuning stops: the local sweep in flight finishes, and the crypto v2
freeze and Jun 15 test still run as pre-registered.

### Track S: pre-game sports

- **Data:** PolyOrderBooks sports L2 (the user is getting an API key), plus on-chain fills, gamma
  `gameStartTime`, and our recorder from now on.
- **Episode:** from listing to kickoff. Quotes are forced off at kickoff and inventory is
  flattened or marked there. Holding into the game is not allowed.
- **Split by game date.** The oldest ~60% of days are train, the next ~20% val, and the newest
  ~20% test, untouched. Exact dates are fixed once the backfill range is known, and before any
  model is evaluated on them. Days recorded after that form an extra forward test.

### Track E: election reward farming (US midterms, Nov 3)

- **Data:** no vendor has historical L2 for elections, so books must be recorded from now on with
  our recorder. Liquidity rewards are modelled with the env's reward hook, using the current reward
  configs.
- **Split by date** over the recorded days. The last recorded days before Nov 3 are the untouched
  test.
- Election night is out of scope: it is a jump event.

### Shared rules

- Same comparators logic as D1: no-quote plus tuned classical quoters.
- Same D1.3 market-making rule, same T5/T6 detectors.
- Every val evaluation goes in a ledger.
- Each track's test is run once.

## D12. Sub-second Binance order-flow features: evaluated, not adopted (recorded 2026-10-15 ~20:35 UTC)

Study and code: `analysis/orderflow/`. The sha256 values of `mm.h`, `feat.bin` and `build_v2` are
unchanged.

- **Data:** 28 candidate features built from aggTrades: sub-second returns, activity, staleness, a
  BTC-lead feature and the predicted fair-value change.
- **Method:** fit on train only (6 purged 8-hour blocks). Then a single confirmation on val v2.
- **Rule, fixed before val was looked at:** a candidate counts only if it adds at least +0.01 AUC
  or +0.005 R² on a fill-toxicity or taker-flow target, with the same sign on train and val.
  **Nothing passes.**
- **Why:** Polymarket takers react to Binance moves 2–3.5 s before their print's block time. Under
  central fills with 1 s latency, the quote facing that print is decided 5 s before it, so the
  move hasn't happened yet. The AUC of last-second Binance for taker side, by decision time:

  | decision time (s before print) | 5 | 4 | 3 | 2 |
  |---|---:|---:|---:|---:|
  | AUC | 0.53 | 0.54 | 0.61 | 0.69 |

- **Consequence:** the lever is decision timing (latency, match delay), not features. Any timing
  change must be validated against real fills first (live shadow quoting).
- **Flag for the env owner:**
  - Raw book `captured_at` values are whole seconds. Book rows may therefore reflect the market up
    to about 1 s later than the Binance features, roughly 0.5 s of optimism on average.
  - This may already be absorbed by the 3.5 s central match-delay calibration. Not acted on.

## D13. How D1.3 is judged when selecting a multi-seed method, and the v2 freeze settings (recorded 2026-10-15 ~21:10 UTC, after val v2 results were visible, before any Jun 15 strategy result)

**Stated plainly: this entry was written after the val v2 results of every ledger were visible.**
It changes no data, env or evaluation code. The test split has not been touched.

**What happened.**
- The training agent changed how `train/rounds.py` and `train/v2search.py` compute T6 for a
  multi-seed method. The old code took the mean of the per-seed T6 shares. The new code computes
  T6 on seed-pooled attribution: mean inventory P&L over mean P&L.
- The change was made after the v2s1 val results were seen, and went into commit `4f17e3a`
  together with the ledger rows.
- The reason it gave: under per-seed averaging a losing seed counts as T6 = 0. For example,
  `v2s1/c29` quoting 15m+4h (seeds +236 / −71 / −84, mean +27) had a per-seed-mean T6 of 0.41,
  but all of its seed-mean profit is inventory P&L (pooled share 4.19).

**Decision for the v2 freeze (coordinator, 21:10 UTC).**
1. **D1.3 for a multi-seed method is judged on seed-pooled attribution, with the directional
   rule of `final/reduce.py` `directional_verdict`.** That rule is frozen in the v1
   TEST_PLAN.json, and the test applies it to the RL seed mean. A method is directional if its
   inventory P&L is > 0 and is either:
   - more than 50% of a positive total; or
   - more than 50% of the sum of the positive components (spread capture, 60 s markout,
     inventory, rebates, rewards).

   The T5 artifact share is also pooled, and the fills rule is mean fills/market ≥ 5. This is
   "reading C" in `final/v2/SELECTION.md`. Readings A (per-seed mean) and B (pooled total share
   only, the 4f17e3a code) are reported next to it, with their winners.
2. **Single ensemble evaluations are not D8 candidates.** An ensemble evaluated once has no seed
   mean. Picking the best of the ~200 ensemble × length-filter evaluations would select a single
   draw, which T4 forbids ("never select the best seed"). D8's "pre-specified ensemble" remains a
   way to deploy a method that was selected on its seed mean.
3. **Comparators.** D7 keeps D1 unchanged, and no baseline was re-tuned on val v2. The v1 tuned
   parameter sets are therefore kept: `baselines/tuned.json` (primary candidates) and
   `baselines/tuned_active.json` (D1.2 secondary). The comparator is the best tuned baseline,
   `fair_value`. It is also the best tuned baseline when re-scored on val v2 (+101 against
   as_logit +24 and no-quote 0; `train/runs/v2/baselines.json`).
4. **Env parameters of the plan: `mark_mode = 1`.** This is the env the selected config was
   trained and val-scored with. `mark_mode` changes only the wealth observation and the training
   reward. P&L, attribution and the baselines' inputs do not depend on it.
   - A dry run on Aug 23 reproduces the ledger exactly: seed 0, −387.65 over 5,409 fills on both
     paths.
   - The v1 plan used the `mm.h` default `mark_mode = 0` for R7 `c02_h120`, which had been trained
     with `mark_mode = 1`. So in v1 the RL policies saw a mid-marked wealth input that they were
     not trained on. This is recorded here and not re-run.
5. **D10 sensitivity.** `final/v2/d10_book_valid.py` is a post-hoc script that reads the frozen
   outputs. It is committed before the test.
6. **Cosmetic, not fixed.** `make_plan.py` writes `test_split.dates = "2026-08-24 UTC"`, which is
   hard-coded. The v2 plan's data block (`data/build_jun15`, its file and test-row hashes, and the
   288 market slugs) is what defines the test.

## D14. Track S window refined by the out-of-sample census (recorded 2026-10-15 ~21:40 UTC, before any v3 model is tuned or any v3 split is fixed)

**Evidence.**
- w7's census replication (`analysis/census/REPORT.md` §6, out-of-sample Sep 17–Oct 7) does not reproduce the
  headline pre-game sports edge: +13 bps [−35, 55], against +64 [1, 130] on Oct 8–14.
- w2 (`analysis/v3_sports_rules/REPORT.md`) finds the same on Sep 24–Oct 7: −26 bps [−43, −11] at the touch,
  measured at 30 minutes.
- What replicates in both periods is the time-to-kickoff structure. The ex-rewards maker edge at +30 min:

  | Time before kickoff | Oct 8–14 | Out-of-sample |
  |---|---:|---:|
  | more than 24 h | +330 [73, 665] | +81 [55, 114] |
  | 6–24 h | +41 [7, 75] | +43 [17, 64] |
  | 1–6 h | +30 [−9, 62] | +53 [31, 74] |
  | last hour | about 0, CI ±90 | about 0, CI ±90 |

  The last hour holds 44% of pre-game volume.

**Refinement of D11 Track S:**
- **Quote window:** from listing (or the start of data) until kickoff − 1 h.
- **Last hour:** quotes are off and inventory is flattened or marked at T−1 h. In-play stays out of scope.
- **Time to kickoff** is an observation, and results are reported per time bucket (>24 h, 6–24 h, 1–6 h) and per
  league.
- **Recording and backfill** must cover the windows longer than 1 h, not just the final hours.

**Status of the evidence.** The per-league cells and w2's last-5-minute pocket (+35 bps) are exploratory leads and
not rules; the last-5-minute pocket is negative once realistic size and flatten cost are applied. Track S is still
unconfirmed until it is tested on unseen L2 data.

**Track E** is confirmed out of sample: +101 bps [63, 130] with rewards, and ex-rewards about 0 to slightly
positive. It is unchanged.

## D15. D14's T−1h cut withdrawn: the last-hour result was a measurement artifact (recorded 2026-10-15 ~22:25 UTC, before any v3 model is tuned or any v3 split is fixed)

**Evidence: w2's reconciliation** (`analysis/v3_sports_rules/RECONCILE.md`).

1. **w2's −26 bps was a fill-valuation error, so D14 no longer cites it.** It valued sweep volume at the touch price, when that volume really filled deeper makers at better prices; this is a ~70 bps bias in every week. Measured with the census definitions, w2's pipeline reproduces the census: +66 bps [2, 135] against +64 [1, 130].
2. **w7's "last hour ≈ 0" was an in-play artifact.** The census's +30-minute mark runs past kickoff for legs near kickoff. With the mark capped at kickoff:
   - the pre-game edge is stable week to week: +50 [28, 73], +49 [32, 67] and +71 [34, 116] for Sep 24–30, Oct 1–7 and Oct 8–14;
   - the last 30 minutes are the best bucket in every week. At the touch, to kickoff, 0–5 minutes before kickoff is +79, +70 and +65 for the same three weeks.
3. **The edge is thin to kickoff.** At the touch it is +20 [1, 37] bps; one tick behind it is +81 [34, 124] bps. This is before rewards. Rule and league selections do not transfer: rank correlation is 0.39 from selection to check, and −0.36 to Oct 8–14.

**Decision (replaces D14's quote window):**
- **Episode:** from the start of data to kickoff, as in D11. Quotes are forced off at kickoff and inventory is marked at the kickoff mid, minus a flatten cost.
- **Quote pull time** (`pull_s`) and an inventory penalty that ramps over the final 30 minutes are hyperparameters. They are selected on val, never on test. The candidate pull times are {T−5 min, T−15 min, T−30 min, T−1 h}.
- **Flatten cost** is half a spread on inventory still held at kickoff. A sensitivity check at a full spread is reported but not judged.
- **D14's per-bucket reporting stays:** >24 h, 6–24 h, 1–6 h, 1 h–30 min, 30–5 min and 5–0 min, all marked at no later than kickoff.
- **Recording** keeps D14's 7-day horizon and includes the last hour, which can now be traded.

**Track S status:** the edge is real but thin, about +20 to +30 bps to kickoff before rewards. It depends on netting inventory before kickoff and on queue position. The deciding test is an L2 replay with queue-aware fills at the touch and one tick behind.

## D16. Recording gaps from container restarts (recorded 2026-10-16 ~02:10 UTC, before any v3 split is fixed or any v3 model is tuned)

**Why.** The cloud workers' containers restart without warning: w1 at about 00:44 UTC (recorder down ~2 min) and w3 at about 02:03 UTC (~50 s). On ephemeral hosts this will keep happening, so whole-day episodes that require gap-free data would throw away most days.

**Rule, for both v3 tracks:**
- **Splitting at gaps:** any recording gap longer than 15 s splits an episode at the gap, so no episode spans one. Rows inside a gap are never synthesised.
- **Inventory at a gap:** the policy's inventory is marked at the last pre-gap mid and carried into the next segment. Training and eval treat the gap as a forced quote pull (`pull_s`-style), not as a reset.
- **Day eligibility for splits:**
  - A calendar day stays eligible for train, val or test if its total gap time is 30 min or less, per track, and listed in that track's GAPS log.
  - A day with more than 30 min of gaps is excluded from val and test, but may still be used in train.
- **Elections:** default windows are 4 h. Longer windows are allowed only where gap-free.
- **Sports:** a game whose final pre-kickoff hour contains a gap is excluded from val and test for the last-hour buckets. Its earlier buckets remain usable.
- **Reporting:** all gaps are reported with any result.
