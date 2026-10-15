# v2 selection (DECISIONS.md D8, D13): val v2 only

Recorded 2026-10-15 ~21:15 UTC and refreshed at 22:20 UTC on the ledgers as of commit eb130c1 (the
final w4 and w5 ledgers included; the winner is unchanged), before the v2 plan was frozen and before any Jun 15 strategy
result. Reproduce with `python final/v2/select_v2.py`, which reads only the ledgers and writes
`final/v2/selection_v2.json` with every method row.

## Winner

**`v2s1/c00`, quoting every contract length (no length filter).** It is a reward-mode-2 MLP from
the local sweep v2s1:

| setting | value |
|---|---|
| network | 64×2 |
| markout_h | 60 s |
| inv_penalty | 2.0e-6 |
| lr | 1.4e-3 |
| ent_coef | 1.2e-3 |
| mark | fair-value (`mark_mode` 1) |
| training markets | all lengths |
| config sha256 | `9695dc23…` |

**Val v2 at selection (seeds 0–2):**
- seed mean −35.0 ± 98.1 (SE); the seeds score −185.0, −69.5 and +149.5;
- 17.5 fills per market;
- T6 positive-component share 0.45, T5 0.00, so it passes D1.3 under reading C.

**It loses money on val**: −$35 against no-quote's $0, and −$136 against the comparator
fair_value's +$101 on the same 374 markets. A FAIL on the primary endpoint is the expected outcome.

**Seeds 3 and 4.** Trained after selection with the identical config (ledger tag `v2final`, not
used for selection). They score −362.1 and −87.9. The 5-seed val v2 mean is **−111.0 ± 83.3**.

The checkpoints are copied to `final/v2/ckpt/c00_s{0..4}.pt` (hashes in `final/v2/selected.json`).

**Applied rule (D8 with D13).**
1. Candidates are methods evaluated on val v2: `data/build_v2` split 1, 374 markets (223 5m, 119
   15m, 32 4h), central fills, 1 s latency.
2. Each method needs at least 3 seeds evaluated individually, so that it has a seed mean.
3. It must pass D1.3 on seed-pooled attribution, judged with `final/reduce.py`'s
   `directional_verdict` (reading C). Concretely:
   - mean fills per market ≥ 5;
   - not directional: inventory P&L ≤ 0, or ≤ 50% of a positive total *and* ≤ 50% of the
     positive components;
   - T5: worst D2 artifact flag ≤ 25% of the pooled fill edge.
4. Among the methods that pass, take the best seed mean.

Single ensemble evaluations are not candidates (D13.2). Baselines are comparators, not candidates.

**No candidate that passes D1.3 makes money.**
- Under reading C, 200 methods are eligible, and 0 of them have a positive val v2 seed mean.
- Under reading B (pooled total share only), the best method is −1.6 (w4's quoter t45). Its
  positive-component share is 0.62, so the final/ rule would call it directional.
- Every profitable method on val v2 fails D1.3 through T6 under readings B and C. Their P&L is
  inventory or resolution P&L, mostly on the 32 4h markets.

## The winner depends on the reading (flagged)

| reading | D1.3 T6 definition | single ensemble evaluations excluded | ensembles included |
|---|---|---|---|
| **C (primary)** | pooled; `final/reduce.py` directional_verdict (total share, or positive-component share > 0.5) | **`v2s1/c00` all lengths, −35.0 ± 98.1, 17.5 fills** | `v2s1_ens/c17/ensemble_probs_3`, 15m only: +48.3 (one evaluation), 7.4 fills, T6 0.11, T5 0.17 |
| B (commit 4f17e3a) | pooled; mean inventory P&L / mean P&L if > 0, else 0 | `w4/q1:t45` (non-RL toxicity quoter), −1.6 ± 26.0, 10.4 fills | `v2s1_ens/c17/ensemble_probs_3`, 15m only, +48.3 |
| A (pre-4f17e3a code) | mean of per-seed T6 shares (a losing seed counts 0) | `v2s1/c29`, 15m+4h, +26.7 ± 104.5 (seeds +236 / −71 / −84; pooled T6 4.19) | `v2s1_ens/c17/ensemble_probs_3`, 15m only, +48.3 |

- **Quoting on the test.** The reading-A winner quotes only 15m and 4h, and the ensemble winner
  quotes only 15m. On the Jun 15 test (BTC 5-minute markets only, D7) both would quote nothing:
  P&L exactly $0 and 0 fills. They would fail D1.3 on the test by construction.
- **Why reading C is primary.**
  - D7 and D8 do not define T6 for a seed mean. D7 keeps the D1.3 rule.
  - The directional rule frozen in the v1 TEST_PLAN.json (`directional`) is C. The test applies
    C to the RL seed mean.
  - The coordinator confirmed C at 21:10 UTC (D13).
- **The w4 tuner picked differently.** w4 selected its quoter with reading A and reported q1best
  = trial t54 (−22.2 ± 6.9). t45 beats it under B only.

## Top methods under each reading

#### Reading C (primary), single ensemble evaluations excluded: top 5

| method (length filter) | seeds | val-v2 P&L, seed mean ± SE | fills/mkt | T6 A (per-seed mean) | T6 B (pooled total share) | T6 positive-component share (C) | T5 max (pooled) | D1.3 A | D1.3 B | D1.3 C | P&L 5m / 15m / 4h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2s1/c00 · 5m+15m+4h` | 3 | -35.0 ± 98.1 | 17.55 | 0.82 | 0.00 | 0.45 | 0.00 | fail | pass | pass | -209 / -48 / +222 |
| `main_final/seeds · 15m+4h` | 5 | -37.0 ± 43.5 | 6.02 | 0.36 | 0.00 | 0.47 | 0.00 | pass | pass | pass | +0 / -152 / +115 |
| `v2s1/c18 · 15m+4h` | 3 | -44.0 ± 58.6 | 6.96 | 1.08 | 0.00 | 0.26 | 0.00 | fail | pass | pass | +0 / -82 / +38 |
| `v2s1/c00 · 15m` | 3 | -48.0 ± 25.8 | 5.30 | 0.00 | 0.00 | 0.00 | 0.00 | pass | pass | pass | +0 / -48 / +0 |
| `w4/q2:t23 · 5m+15m+4h` | 3 | -51.4 ± 17.8 | 9.99 | 0.00 | 0.00 | 0.48 | 0.00 | pass | pass | pass | -193 / +6 / +136 |

#### Reading C (primary), ensembles included: top 5

| method (length filter) | seeds | val-v2 P&L, seed mean ± SE | fills/mkt | T6 A (per-seed mean) | T6 B (pooled total share) | T6 positive-component share (C) | T5 max (pooled) | D1.3 A | D1.3 B | D1.3 C | P&L 5m / 15m / 4h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2s1_ens/c17/ensemble_probs_3 · 15m` | ensemble of 3 | +48.3 (one evaluation) | 7.44 | 0.11 | 0.11 | 0.06 | 0.17 | pass | pass | pass | +0 / +48 / +0 |
| `v2s1_ens/c05/ensemble_logits_3 · 15m` | ensemble of 3 | -0.7 (one evaluation) | 9.58 | 0.00 | 0.00 | 0.38 | 0.00 | pass | pass | pass | +0 / -1 / +0 |
| `v2s1_ens/c17/ensemble_logits_3 · 15m` | ensemble of 3 | -13.1 (one evaluation) | 7.46 | 0.00 | 0.00 | 0.00 | 0.00 | pass | pass | pass | +0 / -13 / +0 |
| `v2s1_ens/c05/ensemble_probs_3 · 15m` | ensemble of 3 | -13.7 (one evaluation) | 7.11 | 0.00 | 0.00 | 0.00 | 0.00 | pass | pass | pass | +0 / -14 / +0 |
| `v2s1_ens/c00/ensemble_logits_3 · 5m+15m+4h` | ensemble of 3 | -14.6 (one evaluation) | 12.72 | 0.00 | 0.00 | 0.44 | 0.00 | pass | pass | pass | -218 / -14 / +218 |

#### Reading B, single ensemble evaluations excluded: top 5

| method (length filter) | seeds | val-v2 P&L, seed mean ± SE | fills/mkt | T6 A (per-seed mean) | T6 B (pooled total share) | T6 positive-component share (C) | T5 max (pooled) | D1.3 A | D1.3 B | D1.3 C | P&L 5m / 15m / 4h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `w4/q1:t45 · 5m+15m+4h` | 3 | -1.6 ± 26.0 | 10.37 | 6.55 | 0.00 | 0.62 | 0.00 | fail | pass | fail | -167 / +13 / +152 |
| `w4/q2:t50 · 5m+15m+4h` | 3 | -3.2 ± 30.1 | 7.21 | 1.27 | 0.00 | 0.56 | 0.00 | fail | pass | fail | -136 / +28 / +105 |
| `w4/q1:t23 · 5m+15m+4h` | 3 | -5.3 ± 64.5 | 15.16 | 1.25 | 0.00 | 0.64 | 0.00 | fail | pass | fail | -297 / +102 / +189 |
| `w4/rl_c5_h60_kl0.1_d0.9 · 5m+15m+4h` | 3 | -5.5 ± 30.4 | 12.62 | 1.42 | 0.00 | 0.66 | 0.00 | fail | pass | fail | -233 / +43 / +184 |
| `w4/q1:t14 · 5m+15m+4h` | 3 | -5.7 ± 50.0 | 15.59 | 1.54 | 0.00 | 0.66 | 0.00 | fail | pass | fail | -332 / +115 / +211 |

#### Reading B, ensembles included: top 5

| method (length filter) | seeds | val-v2 P&L, seed mean ± SE | fills/mkt | T6 A (per-seed mean) | T6 B (pooled total share) | T6 positive-component share (C) | T5 max (pooled) | D1.3 A | D1.3 B | D1.3 C | P&L 5m / 15m / 4h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2s1_ens/c17/ensemble_probs_3 · 15m` | ensemble of 3 | +48.3 (one evaluation) | 7.44 | 0.11 | 0.11 | 0.06 | 0.17 | pass | pass | pass | +0 / +48 / +0 |
| `v2s1_ens/c05/ensemble_logits_3 · 15m` | ensemble of 3 | -0.7 (one evaluation) | 9.58 | 0.00 | 0.00 | 0.38 | 0.00 | pass | pass | pass | +0 / -1 / +0 |
| `w4/q1:t45 · 5m+15m+4h` | 3 | -1.6 ± 26.0 | 10.37 | 6.55 | 0.00 | 0.62 | 0.00 | fail | pass | fail | -167 / +13 / +152 |
| `w4/q2:t50 · 5m+15m+4h` | 3 | -3.2 ± 30.1 | 7.21 | 1.27 | 0.00 | 0.56 | 0.00 | fail | pass | fail | -136 / +28 / +105 |
| `w4/q1:t23 · 5m+15m+4h` | 3 | -5.3 ± 64.5 | 15.16 | 1.25 | 0.00 | 0.64 | 0.00 | fail | pass | fail | -297 / +102 / +189 |

#### Reading A, single ensemble evaluations excluded: top 5

| method (length filter) | seeds | val-v2 P&L, seed mean ± SE | fills/mkt | T6 A (per-seed mean) | T6 B (pooled total share) | T6 positive-component share (C) | T5 max (pooled) | D1.3 A | D1.3 B | D1.3 C | P&L 5m / 15m / 4h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2s1/c29 · 15m+4h` | 3 | +26.7 ± 104.5 | 8.26 | 0.41 | 4.19 | 0.53 | 0.00 | pass | fail | fail | +0 / -97 / +124 |
| `w4/q1:t54 · 5m+15m+4h` | 3 | -22.2 ± 6.9 | 13.04 | 0.00 | 0.00 | 0.58 | 0.00 | pass | pass | fail | -192 / +31 / +138 |
| `w4/q1best · 5m+15m+4h` | 3 | -22.2 ± 6.9 | 13.04 | 0.00 | 0.00 | 0.58 | 0.00 | pass | pass | fail | -192 / +31 / +138 |
| `w4/q2:t0 · 5m+15m+4h` | 3 | -22.2 ± 6.9 | 13.04 | 0.00 | 0.00 | 0.58 | 0.00 | pass | pass | fail | -192 / +31 / +138 |
| `w4/q2:t48 · 5m+15m+4h` | 3 | -26.7 ± 11.0 | 13.02 | 0.00 | 0.00 | 0.59 | 0.00 | pass | pass | fail | -187 / +25 / +136 |

#### Reading A, ensembles included: top 5

| method (length filter) | seeds | val-v2 P&L, seed mean ± SE | fills/mkt | T6 A (per-seed mean) | T6 B (pooled total share) | T6 positive-component share (C) | T5 max (pooled) | D1.3 A | D1.3 B | D1.3 C | P&L 5m / 15m / 4h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2s1_ens/c17/ensemble_probs_3 · 15m` | ensemble of 3 | +48.3 (one evaluation) | 7.44 | 0.11 | 0.11 | 0.06 | 0.17 | pass | pass | pass | +0 / +48 / +0 |
| `v2s1/c29 · 15m+4h` | 3 | +26.7 ± 104.5 | 8.26 | 0.41 | 4.19 | 0.53 | 0.00 | pass | fail | fail | +0 / -97 / +124 |
| `v2s1_ens/c05/ensemble_logits_3 · 15m` | ensemble of 3 | -0.7 (one evaluation) | 9.58 | 0.00 | 0.00 | 0.38 | 0.00 | pass | pass | pass | +0 / -1 / +0 |
| `v2s1_ens/c17/ensemble_logits_3 · 15m` | ensemble of 3 | -13.1 (one evaluation) | 7.46 | 0.00 | 0.00 | 0.00 | 0.00 | pass | pass | pass | +0 / -13 / +0 |
| `v2s1_ens/c05/ensemble_probs_3 · 15m` | ensemble of 3 | -13.7 (one evaluation) | 7.11 | 0.00 | 0.00 | 0.00 | 0.00 | pass | pass | pass | +0 / -14 / +0 |

## Top candidates per source (val v2, central fills, 1 s)

For each source, the table shows the best three methods by mean among those with at least 3
seeds, plus the best method passing reading C if it is not already among them. Sources with no
method of at least 3 seeds show their single best method.

Column notes:
- "T6 total-share pooled" is reading B.
- "T6 positive-comp. share" is the second clause of reading C.
- A losing method has T6 total share 0 by the evaluator convention.

| source | method (filter) | seeds / members | val-v2 P&L, seed mean (SE) | fills/mkt | T6 total-share pooled | T6 positive-comp. share | T5 max (pooled) | D1.3 A per-seed | D1.3 B pooled | D1.3 C final/ rule |
|---|---|---|---|---|---|---|---|---|---|---|
| local: R1 tuned-BC configs re-scored on v2 - 3 seeds | `r1_c06_tuned/seeds · 5m+15m+4h` | 3 | +151.9 (+59.1) | 3.99 | 1.59 | 0.55 | 0.00 | fail | fail | fail |
| local: R1 tuned-BC configs re-scored on v2 - 3 seeds | `r1_c00_tuned/seeds · 5m+15m+4h` | 3 | +147.5 (+6.2) | 4.18 | 1.75 | 0.56 | 0.00 | fail | fail | fail |
| local: R1 tuned-BC configs re-scored on v2 - 3 seeds | `r1_c06_tuned/seeds · 5m+15m` | 3 | +139.7 (+55.4) | 3.98 | 1.65 | 0.54 | 0.00 | fail | fail | fail |
| local: R1 tuned-BC configs re-scored on v2 - ensembles | `r1_c10_tuned/ensemble_probs_3 · 5m+15m+4h` | ens of 3 | +188.3 (-) | 4.28 | 1.51 | 0.58 | 0.00 | fail | fail | fail |
| local: R1 tuned-BC configs re-scored on v2 - ensembles | `r1_c12_tuned/ensemble_logits_3 · 5m+15m+4h` | ens of 3 | +184.0 (-) | 4.07 | 1.50 | 0.58 | 0.00 | fail | fail | fail |
| local: R1 tuned-BC configs re-scored on v2 - ensembles | `r1_c10_tuned/ensemble_logits_3 · 5m+15m+4h` | ens of 3 | +181.2 (-) | 4.29 | 1.56 | 0.58 | 0.00 | fail | fail | fail |
| local: R7 c02_h120 (v1 selection) re-scored on v2 - 5 seeds | `r7_final/seeds · 4h` | 5 | +193.3 (+18.8) | 1.10 | 1.06 | 0.90 | 0.00 | fail | fail | fail |
| local: R7 c02_h120 (v1 selection) re-scored on v2 - 5 seeds | `r7_final/seeds · 15m+4h` | 5 | +74.0 (+49.8) | 8.80 | 2.65 | 0.66 | 0.00 | fail | fail | fail |
| local: R7 c02_h120 (v1 selection) re-scored on v2 - 5 seeds | `r7_final/seeds · 5m+15m+4h` | 5 | -117.8 (+125.8) | 22.11 | 0.00 | 0.49 | 0.00 | fail | pass | pass |
| local: R7 c02_h120 (v1 selection) re-scored on v2 - ensembles | `r7_final/ensemble_probs_5 · 5m+15m+4h` | ens of 5 | +150.3 (-) | 7.85 | 0.69 | 0.61 | 0.40 | fail | fail | fail |
| local: R7 c02_h120 (v1 selection) re-scored on v2 - ensembles | `r7_final/ensemble_logits_5 · 5m+15m+4h` | ens of 5 | +127.4 (-) | 8.57 | 1.07 | 0.67 | 0.00 | fail | fail | fail |
| local: R7 c02_h120 (v1 selection) re-scored on v2 - ensembles | `r7_final/ensemble_logits_5 · 15m+4h` | ens of 5 | +127.2 (-) | 3.44 | 1.00 | 0.85 | 0.00 | fail | fail | fail |
| local: c00 seeds 3-4 (v2final, after selection) | `local/v2final: · 5m+15m+4h` (best; < 3 seeds, ineligible) | 2 | -225.0 | 14.44 | 0.00 | 0.27 | 0.00 | pass | pass | pass |
| local: sweep v2s1 (32 configs x 3 seeds, with length filters) | `v2s1/c00 · 4h` | 3 | +222.1 (+9.1) | 0.98 | 1.08 | 0.92 | 0.00 | fail | fail | fail |
| local: sweep v2s1 (32 configs x 3 seeds, with length filters) | `v2s1/c05 · 4h` | 3 | +218.7 (+41.9) | 2.16 | 1.15 | 0.88 | 0.00 | fail | fail | fail |
| local: sweep v2s1 (32 configs x 3 seeds, with length filters) | `v2s1/c31 · 4h` | 3 | +190.2 (+28.2) | 1.91 | 1.12 | 0.88 | 0.00 | fail | fail | fail |
| local: sweep v2s1 (32 configs x 3 seeds, with length filters) | `v2s1/c00 · 5m+15m+4h` | 3 | -35.0 (+98.1) | 17.55 | 0.00 | 0.45 | 0.00 | fail | pass | pass |
| local: v1 main-sweep config re-scored on v2 - 5 seeds | `main_final/seeds · 4h` | 5 | +115.4 (+47.4) | 0.64 | 1.05 | 0.92 | 0.00 | fail | fail | fail |
| local: v1 main-sweep config re-scored on v2 - 5 seeds | `main_final/seeds · 15m+4h` | 5 | -37.0 (+43.5) | 6.02 | 0.00 | 0.47 | 0.00 | pass | pass | pass |
| local: v1 main-sweep config re-scored on v2 - 5 seeds | `main_final/seeds · 5m` | 5 | -136.6 (+51.0) | 9.17 | 0.00 | 0.04 | 0.00 | pass | pass | pass |
| local: v1 main-sweep config re-scored on v2 - ensembles | `main_final/ensemble_logits_5 · 4h` | ens of 5 | +169.2 (-) | 0.64 | 1.00 | 0.97 | 0.00 | fail | fail | fail |
| local: v1 main-sweep config re-scored on v2 - ensembles | `main_final/ensemble_logits_5 · 15m+4h` | ens of 5 | +149.6 (-) | 2.64 | 1.29 | 0.90 | 0.00 | fail | fail | fail |
| local: v1 main-sweep config re-scored on v2 - ensembles | `main_final/ensemble_probs_5 · 4h` | ens of 5 | +106.2 (-) | 0.45 | 1.00 | 0.97 | 0.00 | fail | fail | fail |
| local: v1 main-sweep config re-scored on v2 - ensembles | `main_final/ensemble_logits_5 · 5m+15m` | ens of 5 | -183.7 (-) | 5.73 | 0.00 | 0.00 | 0.00 | pass | pass | pass |
| local: v2s1 3-seed ensembles (v2s1_ens) | `v2s1_ens/c05/ensemble_logits_3 · 4h` | ens of 3 | +322.5 (-) | 1.15 | 1.03 | 0.95 | 0.00 | fail | fail | fail |
| local: v2s1 3-seed ensembles (v2s1_ens) | `v2s1_ens/c05/ensemble_logits_3 · 15m+4h` | ens of 3 | +321.8 (-) | 10.73 | 1.19 | 0.79 | 0.00 | fail | fail | fail |
| local: v2s1 3-seed ensembles (v2s1_ens) | `v2s1_ens/c05/ensemble_probs_3 · 4h` | ens of 3 | +315.4 (-) | 0.93 | 1.01 | 0.95 | 0.00 | fail | fail | fail |
| local: v2s1 3-seed ensembles (v2s1_ens) | `v2s1_ens/c17/ensemble_probs_3 · 15m` | ens of 3 | +48.3 (-) | 7.44 | 0.11 | 0.06 | 0.17 | pass | pass | pass |
| w1 | `w1/t12 · 5m+15m+4h` | 3 | +41.1 (+41.1) | 0.85 | 1.03 | 0.80 | 0.00 | fail | fail | fail |
| w1 | `w1/t04 · 5m+15m+4h` | 3 | -124.8 (+123.6) | 7.89 | 0.00 | 0.10 | 0.00 | pass | pass | pass |
| w1 | `w1/t00 · 5m+15m+4h` | 3 | -183.7 (+105.6) | 14.14 | 0.00 | 0.32 | 0.00 | pass | pass | pass |
| w2 | `w2/4h_c01 · 4h` (best; < 3 seeds, ineligible) | 2 | +169.0 | 1.83 | 1.17 | 0.81 | 0.00 | fail | fail | fail |
| w3 | `w3/pnet_L1_H256 · 5m+15m+4h` | 3 | -507.1 (+368.4) | 33.90 | 0.00 | 0.28 | 0.00 | pass | pass | pass |
| w3 | `w3/pnet_L2_H128 · 5m+15m+4h` | 3 | -555.3 (+195.8) | 28.61 | 0.00 | 0.00 | 0.00 | pass | pass | pass |
| w3 | `w3/pnet_L2_H256 · 5m+15m+4h` | 3 | -796.8 (+182.8) | 37.44 | 0.00 | 0.00 | 0.00 | pass | pass | pass |
| w4: BC(quoter) + markout PPO | `w4/rl_c2_h120_kl0.1_d0.5 · 5m+15m+4h` | 3 | +11.2 (+2.8) | 14.59 | 24.47 | 0.66 | 0.00 | fail | fail | fail |
| w4: BC(quoter) + markout PPO | `w4/rl_c8_h60_kl0.1_d0.5_inv1e-3 · 5m+15m+4h` | 3 | +4.9 (+40.8) | 14.08 | 51.05 | 0.66 | 0.00 | fail | fail | fail |
| w4: BC(quoter) + markout PPO | `w4/rl_c6_h60_kl0.1_d0.5_lr1e-4 · 5m+15m+4h` | 3 | +2.6 (+25.0) | 13.62 | 96.81 | 0.65 | 0.00 | fail | fail | fail |
| w4: toxicity quoter | `w4/q1:t5 · 5m+15m+4h` | 3 | +118.6 (+53.0) | 10.93 | 1.85 | 0.67 | 0.00 | fail | fail | fail |
| w4: toxicity quoter | `w4/q2:t14 · 5m+15m+4h` | 3 | +100.9 (+30.6) | 2.88 | 1.64 | 0.77 | 0.00 | fail | fail | fail |
| w4: toxicity quoter | `w4/q1:t16 · 5m+15m+4h` | 3 | +78.8 (+57.4) | 10.60 | 2.51 | 0.69 | 0.00 | fail | fail | fail |
| w4: toxicity quoter | `w4/q2:t23 · 5m+15m+4h` | 3 | -51.4 (+17.8) | 9.99 | 0.00 | 0.48 | 0.00 | pass | pass | pass |
| w5 | `w5/bc_students:bc_fv · 5m+15m+4h` (best; < 3 seeds, ineligible) | 1 | +28.4 | 4.91 | 6.16 | 0.43 | 0.00 | fail | fail | fail |
| w5: BC warm start + markout PPO | `w5/bcmk_c49_fv_h120_rs0.1 · 5m+15m+4h` | 3 | +110.8 (+32.6) | 4.51 | 2.06 | 0.51 | 0.00 | fail | fail | fail |
| w5: BC warm start + markout PPO | `w5/bcmk_c25_fv_h300 · 5m+15m+4h` | 3 | +66.9 (+7.7) | 5.02 | 2.85 | 0.44 | 0.00 | fail | fail | fail |
| w5: BC warm start + markout PPO | `w5/bcmk_c10_fv_h60 · 5m+15m+4h` | 3 | +60.5 (+7.8) | 4.80 | 3.64 | 0.49 | 0.00 | fail | fail | fail |
| w5: BC warm start + markout PPO | `w5/bcmk_c39_asl_act_h60_rs1 · 5m+15m+4h` | 3 | -87.0 (+8.4) | 13.31 | 0.00 | 0.15 | 0.00 | pass | pass | pass |

The requested candidates, by name:

| candidate | val v2 (central, 1 s) | D1.3 (C) | note |
|---|---|---|---|
| local v2s1 configs | best eligible `c00`, all lengths: −35.0 ± 98.1 | pass | winner |
| local v2s1 length filters | best overall `c00`, 4h only: +222.1 ± 9.1 | fail | 1.0 fills/mkt, T6 1.08: 4h resolution |
| local ensembles (v2eval, v2s1_ens) | best `v2s1_ens/c05/ensemble_logits_3`, 4h: +322.5; best passing C: `c17` probs 15m, +48.3 | – | single evaluations: not D8 candidates (D13.2) |
| R7 `c02_h120` (v1 selection), 5 seeds re-scored on v2 | −117.8 ± 125.8, all lengths | pass (C), fail (A) | the pre-registered incumbent; loses on val v2 |
| w1 `t04` | −124.8 ± 123.6 | pass | best D1.3-passing w1 config |
| w2 specialists | best `4h_c01`, 4h: +169.0 (2 seeds) | fail | no w2 method has 3 seeds |
| w3 PufferNet replicas | best `pnet_L1_H256`: −507.1 | pass | |
| w4 toxicity quoter | `q1best` = t54: −22.2 ± 6.9 (positive-component share 0.58); `t45`: −1.6 ± 26.0 (0.62) | fail (C) | both directional under C; pass under A (t54) or B (both) |
| w4 BC(quoter) + markout PPO (8 configs x 3 seeds, ledgered after 21:00) | best `rl_c2_h120`: +11.2 ± 2.8 (positive-component share 0.66, T6 total share 24); best passing B: `rl_c5`: −5.5 ± 30.4 | fail (C) | all directional under C |
| w5 BC + markout (49 configs x 3 seeds, final ledger 4857cdf) | best `bcmk_c49_fv_h120_rs0.1`: +110.8 (4.5 fills/mkt); best passing C: `bcmk_c39_asl_act_h60_rs1`: −87.0 ± 8.4 | fail (C) for every positive one | the positive fair-value-teacher configs have 4.5–5.0 fills/mkt and T6 total share 2–12 (directional) |

## Checks on the ledger rows

- **Rows kept.** A row is kept only if:
  - `split` = 1;
  - `val.n_markets` = 374;
  - `data_dir` ends in `data/build_v2`;
  - the variant is central fills at 1 s (`central_1s`, or a length-filter row derived from one).
- **Rows set aside.** Rows on other variants (pess_1s, central_2s) and the 253 local v1-val rows
  (`data/build`, 247 markets) are set aside. Every candidate row used above is v2 val.
- **Kept per ledger:**

  | ledger | rows kept | notes |
  |---|---:|---|
  | local | 1,010 | |
  | w1 | 50 | t04 and t05 are one config, de-duplicated by config hash and seed; 11 duplicate rows |
  | w2 | 112 | |
  | w3 | 9 | |
  | w4 | 251 | includes the q2 and RL rows pushed in w4 commits 0e5f564 / 08e3dce |
  | w5 | 156 | final ledger of commit 4857cdf (22:15 UTC) |

  The 54 local `v2_baselines` and 5 `w5_teachers` rows are comparators, not candidates. Duplicate
  rows were checked to carry identical P&L.
- **Seed counts.** A seed count comes from distinct `seed` values. w4's seeds are the three edge
  models (`t<k>_m0..2`). The R7 and main configs have 5 seeds, the R1 configs 3.

## Comparator (D1, D7)

- D7 keeps the D1 comparators unchanged, and no baseline was re-tuned on val v2. There is no
  `baselines/*v2*` file; `train/runs/v2/baselines.json` is a re-scoring of the v1 parameters, not
  a tuning.
- The v1 tuned sets are therefore kept: `baselines/tuned.json` for the primary candidates
  (no-quote included) and `baselines/tuned_active.json` for the D1.2 secondary comparators.
- The primary comparator chosen by `make_plan.py` from tuned.json's val P&L is `fair_value`. It is
  also the best of these on val v2 (+101.1, against as_logit +23.8, no-quote 0, fv_smart −8.0,
  sym_touch 0).

## Stopping

- D11 stopped crypto tuning; the local sweep v2s1 (96/96 runs) was the last step.
- D12 means there is no Binance-feature run.
- The freeze is before the 03:00 UTC deadline of D7 and D8.
