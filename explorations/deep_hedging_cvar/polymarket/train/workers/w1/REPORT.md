# w1: risk-aware objectives on reward_mode 2 (crypto v2): report

**Status: stopped early by the D11 pivot** (coordinator, 20:18 UTC). 15 distinct configs were
trained (13 with 3 seeds, 2 with 2 seeds). Step 2 (top 3 configs × 5 seeds) did not run because
D11 forbids new crypto runs. **No config beats no-quote ($0), and none beats the v1 config by more
than its seed SE.**

## Setup
- Hashes: all six PROTOCOL.md sha256 values matched, and `test_env.py` printed ok.
- Training: split 0 of `data/build_v2`, Aug 21–22. 25M steps, MLP 64×2, off_prior 0.5, with
  gamma and λ_GAE taken from `train/selected.json` (R7 c02_h120). Seeds 0, 1, 2.
- Evaluation: greedy, on val v2 (`data/build_v2` split 1, 374 markets: 223 × 5m, 119 × 15m,
  32 × 4h), with central fills and 1 s latency.
- Ledger: `trials.jsonl` has 50 val evaluations (all rows kept, see "Incidents" below). The run
  index is in `runs.jsonl`, and the per-config seed aggregates are in `configs.jsonl`.
- Search: Optuna TPE (`sweep_w1.py`), maximising seed-mean val P&L, with −1000 for configs that
  fail D1.3 so the search does not drift to no-quote. Search space:
  - inv_penalty λ, log-uniform 1e-4..3e-2;
  - close-inventory shaping on/off, with coef log-uniform 1e-3..1e-1 and ramp {0.1, 0.2, 0.4};
  - markout_h {60, 120, 300};
  - mark_mode {1, 2};
  - lr, log-uniform 1e-4..1.5e-3;
  - ent_coef, log-uniform 1e-4..1e-2.

  Trial t00 replicates the v1 config (λ is clipped to the range minimum of 1e-4; v1 used 5.2e-5).
  Trials t01–t03 are hand-picked risk-aware variants.

## All configs (val v2, central fills, 1 s; seed mean; sorted by P&L)

Column notes:
- **$/market** is the seed mean ± SE across seeds. 5m / 15m / 4h are $/market within each
  contract length.
- **at-cap share** is the share of traded markets where |inventory| reached ≥ 99 of the 100-share
  cap.
- **T5** is "–" when no flagged artifact P&L exists.

| config | seeds | $/market (± SE) | 5m | 15m | 4h | fills/mkt | T6 | T5 | D1.3 | at-cap share | spread/mkt | markout60/mkt | inv P&L/mkt | 2-sided uptime | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t12 | 3 | +0.110 ± 0.110 | +0.02 | +0.14 | +0.63 | 0.9 | 0.34 | - | FAIL | 0.00 | +0.03 | -0.03 | +0.11 | 0.00 | λ=4.6e-03 H=120 mark=1 lr=1.3e-04 ent=9.1e-04 |
| t15 | 2 | -0.096 ± 0.013 | -0.21 | -0.15 | +0.93 | 4.2 | 0.00 | - | FAIL | 0.00 | +0.15 | -0.28 | +0.02 | 0.00 | λ=1.4e-04 H=60 mark=1 lr=1.2e-04 ent=8.9e-04 close=4.0e-03/0.4 |
| t04 | 3 | -0.334 ± 0.330 | -0.32 | -0.58 | +0.50 | 7.9 | 0.00 | - | pass | 0.01 | +0.34 | -0.74 | +0.04 | 0.05 | λ=1.1e-03 H=60 mark=1 lr=1.7e-04 ent=4.9e-04 close=6.9e-03/0.4 |
| t00 | 3 | -0.491 ± 0.282 | -0.91 | -0.19 | +1.30 | 14.1 | 0.47 | - | pass | 0.03 | +0.41 | -1.21 | +0.25 | 0.05 | λ=1.0e-04 H=120 mark=1 lr=6.8e-04 ent=4.0e-04 |
| t09 | 3 | -0.548 ± 0.418 | -0.68 | -0.51 | +0.19 | 8.6 | 0.00 | - | pass | 0.00 | +0.46 | -0.99 | -0.05 | 0.16 | λ=1.0e-03 H=300 mark=2 lr=7.5e-04 ent=3.2e-04 close=3.8e-02/0.4 |
| t01 | 3 | -0.571 ± 0.375 | -0.90 | -0.50 | +1.49 | 17.1 | 0.03 | 0.74 | FAIL | 0.03 | +0.42 | -1.16 | +0.10 | 0.08 | λ=3.0e-03 H=120 mark=1 lr=6.8e-04 ent=4.0e-04 close=1.0e-02/0.2 |
| t02 | 3 | -0.573 ± 0.293 | -0.93 | -0.50 | +1.64 | 13.8 | 0.00 | - | pass | 0.06 | +0.41 | -1.14 | +0.09 | 0.02 | λ=1.0e-02 H=300 mark=1 lr=6.8e-04 ent=4.0e-04 |
| t14 | 2 | -0.662 ± 0.414 | -1.13 | -0.75 | +2.94 | 20.3 | 0.00 | - | pass | 0.06 | +0.47 | -1.65 | +0.44 | 0.15 | λ=4.3e-04 H=60 mark=1 lr=4.0e-04 ent=1.5e-04 close=1.1e-03/0.4 |
| t11 | 3 | -0.697 ± 0.601 | -0.99 | -0.58 | +0.90 | 13.9 | 0.33 | - | pass | 0.00 | +0.54 | -1.27 | -0.02 | 0.22 | λ=1.4e-03 H=300 mark=1 lr=2.2e-04 ent=1.8e-03 close=5.4e-02/0.1 |
| t10 | 3 | -0.875 ± 0.397 | -0.84 | -1.75 | +2.11 | 14.1 | 0.00 | - | pass | 0.05 | +0.51 | -1.21 | -0.26 | 0.04 | λ=9.6e-03 H=120 mark=2 lr=2.8e-04 ent=3.1e-03 |
| t03 | 3 | -1.131 ± 0.445 | -1.46 | -1.29 | +1.72 | 20.8 | 0.00 | - | pass | 0.03 | +0.79 | -1.75 | -0.26 | 0.24 | λ=1.0e-03 H=60 mark=2 lr=6.8e-04 ent=3.0e-03 close=3.0e-02/0.4 |
| t06 | 3 | -1.378 ± 0.625 | -1.52 | -2.57 | +4.08 | 29.3 | 0.00 | - | pass | 0.05 | +1.03 | -2.60 | +0.06 | 0.34 | λ=1.2e-04 H=60 mark=2 lr=8.7e-04 ent=8.6e-03 close=5.7e-02/0.1 |
| t13 | 2 | -1.492 ± 0.330 | -2.00 | -1.57 | +2.31 | 23.5 | 0.00 | - | pass | 0.03 | +0.94 | -2.39 | -0.14 | 0.26 | λ=1.4e-03 H=60 mark=2 lr=8.3e-04 ent=1.5e-03 close=1.3e-02/0.2 |
| t08 | 3 | -1.599 ± 0.202 | -1.86 | -2.94 | +5.17 | 33.3 | 0.00 | - | pass | 0.08 | +1.06 | -2.90 | +0.10 | 0.40 | λ=2.2e-02 H=60 mark=1 lr=9.6e-04 ent=6.4e-03 close=4.4e-02/0.1 |

t05 is not listed. It has the same parameters as t04, because two sweep processes shared a TPE
seed. Its 9 ledger rows are kept.

## What worked and what didn't
- **Nothing reaches $0 while also making markets.**
  - Best D1.3-passing config: **t04** at −$0.33 ± 0.33 per market. Settings: λ 1.1e-3,
    H 60, fair-value mark, lr 1.7e-4, close-inventory shaping 6.9e-3 over the last 40% of the
    window, 7.9 fills/market.
  - The v1 replica t00 scores −$0.49 ± 0.28, so the gap is well within the SE.
  - Only t12 is positive (+$0.11 ± 0.11): λ 4.6e-3 with lr 1.3e-4. It fills 0.9 times per market,
    which means it learned to mostly not quote and fails D1.3.
  - t15 (−$0.10, 4.2 fills, lr 1.2e-4) is the same "nearly off" profile. The lowest lrs
    (1.2–1.7e-4) produce the least active policies.
- **Lower activity is the main thing the risk terms buy.**
  - Across configs, P&L falls almost monotonically with fills/market: about −$0.05 per extra
    fill per market.
  - The 60 s markout is the dominant loss everywhere: −$0.74 to −$2.9 per market against
    +$0.3 to +$1.0 of spread capture.
  - Larger inv_penalty, close shaping and a low lr all reduce activity, and with it the loss. They
    do not improve the edge per fill.
- **Inventory at the cap is mostly solved.**
  - The at-cap share is 0–8% of traded markets.
  - The worst-case max |inventory| still reaches 100 in many configs (one market is enough). The
    best configs (t04, t09, t12, t15) peak at 20–85.
- **Higher entropy, higher lr, and mark_mode 2 with H 60 are worst** (t06, t08, t13): the most
  fills and −$1.4 to −$1.6 per market.
- **Per length:**
  - **4h markets are positive in every config** (+$0.2 to +$5.2 per market, n = 32, noisy).
  - 5m and 15m are negative in every D1.3-passing config.
  - A 4h-only specialist (w2's line) is the one direction these results support.
  - w1 did not evaluate length filters, to avoid adding ledger rows outside its assignment.
- **Close-inventory shaping:** no clear effect beyond lowering activity. t01 (coef 1e-2, ramp 0.2)
  fails T5 at 0.74: the flagged-artifact share of fill P&L concentrates in the shaped close
  window.

## Recommended candidate
For coordinator selection, **t04** is the best D1.3-passing w1 config, but only on 3 seeds and
within noise of v1. It is not an improvement under the D7 rule (it does not beat the previous best
by more than the seed SE). Its checkpoints, and those of t00 and t09, are in `ckpt/`
(`<config>_s<seed>.pt` with `.config.json`).

## Incidents (all affected ledger rows are kept)
1. **Optuna enqueue race.** Optuna `n_jobs=4` threads popped the same enqueued trial. t00
   trained 3× in parallel into the same dirs, which left 6 extra identical ledger rows.
   - Fix: one thread per process, with several processes on one sqlite study.
2. **Shared TPE seed.** Those processes then shared TPE seed 1, so t04 = t05 and t06 = t07
   (t07 was killed after 2 seeds, and its 2 rows are kept).
   - Fix: per-process seed.
3. **Pivot stop.** The D11 stop interrupted t13–t15 after 2 seeds. Their seeds in flight finished
   and were logged.
