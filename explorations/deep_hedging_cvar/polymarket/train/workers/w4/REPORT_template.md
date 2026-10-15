# w4: a learned, toxicity-aware quoter (not pure RL), and RL warm-started from it

Status: final for the crypto v2 line. Per DECISIONS.md D11 crypto tuning has stopped. All w4
runs finished and are ledgered; the 100M round finished after the coordinator's 21:22 check-in,
whose freeze used the q2 rows up to then.

Worker w4, protocol v2 (`../PROTOCOL.md`). Env and data hashes were checked against the table in
PROTOCOL.md and all six matched. Training data is train only (split 0, Aug 21–22). Every
selection and every number below is on **val v2** (`data/build_v2`, split 1, 374 markets),
greedy, every market once. The Jun 15 data was never touched. Every val evaluation is a row in
`trials.jsonl`; the final count is at the end of this report.

## Bottom line

* **The learned quoter is the best D1.3-passing method here:** **−$0.059 ± 0.018 per market**
  (seed mean ± SE over 3 data-seed models; central fills, 1 s).
  - 13.0 fills per market; D1.3 passes on 3 of 3 seeds.
  - It beats the v1 best forced-to-trade baseline (−$0.30/market). It does not beat no-quote ($0).
  - By length: 5m −$0.86, 15m +$0.26, 4h +$4.31 per market.
* **The edge model does not find a positive-edge region in held-out hours.**
  - The 60 s fill credit (spread capture + 60 s fair-value markout + rebate) is about −6¢ per
    filled share on train.
  - Under blocked-by-hour CV, selecting on predicted EV > 0 cuts the loss per quoted decision by
    about 5x (−$0.004 against −$0.023), but it stays negative.
  - The rows with the highest predicted edge realise the **worst** edge out of sample.
  - The quoter therefore works by abstaining. It quotes rarely, one side at a time, and only where
    the model predicts edge.
* **PPO warm-started from the quoter (BC + reward mode 2 + decaying KL), 8 configs x 3 seeds:**
  - The best seed means are +$0.030 ± 0.007 per market (c2, markout_h 120) and +$0.013 ± 0.109 (c8).
  - **No config passes D1.3 on all 3 seeds.** The positive seeds fail T6 (directional).
  - Every config gains on 4h (+$5.8 to +$6.6 per market) and loses more on 5m (−$1.05 to −$1.28)
    than the quoter.
  - Under pessimistic fills every config passes D1.3, but all lose (−$0.15 to −$0.26 per market).
* **D1.3 caps any passing method at roughly $0 in this replay.** Write the P&L as
  F + inv, where F = spread capture + 60 s markout + rebates.
  - If F < 0 and the P&L is positive, then inv = P&L − F > P&L, so T6 > 1 and the method is
    "directional".
  - So a method can pass D1.3 with positive P&L only if its 60 s fill P&L F is itself positive.
  - Across every val evaluation in my ledger: FPOS_COUNT. The inventory
    term (holding past 60 s) is positive in INVPOS_COUNT.
  - **Evaluations that both pass D1.3 and have positive P&L: PASSPOS_COUNT.**
  - So the D1.3-passing seeds are the ones with negative totals. Under this rule the protocol's
    "best passing method" is the least-negative one.
  - Coordinator: please weigh this when comparing workers. A positive seed mean with "0/3 D1.3"
    (for example c2) and a −$0.06 "3/3 D1.3" quoter differ mainly in the sign of the total, not in
    how much they market-make.

**Recommended candidate from w4:** the learned quoter `q1 best` (3 data-seed models,
`results/q1_best.json`, code `quoter.py`). It is the only w4 method with ≥ 3 seeds that passes
D1.3 on every seed, and it has the lowest seed SE of anything here.
- If the coordinator relaxes or re-reads T6 as described above, the strongest w4 RL config is
  c2 (`ckpt/c2_h120_kl0.1_d0.5_s{1,2,3}.pt`): +$0.030 ± 0.007, positive on every seed.
- c2 is directional under T6 as implemented.

## 1. Dataset (train split only): `collect.py`

* **Envs.** One C env per train market, 431 envs (baselines/runner.py's Runner:
  `cycle_stride = n`, `fixed_start = 0`). Within one pass, 5m markets repeat about 48 times and
  15m markets about 16 times, with different random actions each time.
* **Exploration policy.** Each env quotes **one side at a time**: a random level 1–6 and a random
  size 0–2.
  - The choice is held for a geometric number of decisions (mean 15), so queue positions evolve as
    they do for a resting order.
  - 10% of segments are off.
  - The other side is off, so every fill in a step belongs to the quoted side.
* **Labels come from the C core itself.**
  - `reward_mode 2`, `markout_h 60`, `inv_penalty 0` credits each fill at its print row with
    sq · (p_model[tau+60] − p) + rebate, which is spread capture + 60 s markout (on fair value)
    + rebate.
  - Q = 1e5, so the inventory cap never blocks a fill. The per-fill edge does not depend on
    inventory because there is no market impact.
* **Timing.** `dt = 1` for every length. The decision-to-booking lag was measured at 4 steps
  (`--find-lag`: a fill-given-on rate of 6.3% at lag 4, against 0.3% at every other lag). Row j is
  the obs at decision step i, the action, and the credit / shares / fill count booked at step i + 4.
* **Size.** 3 data seeds (exploration RNG 1, 2, 3), each with about 1.65M rows: every fill row plus
  25% of the non-fill rows, which are re-weighted by 4. Each seed has about 358k fill rows over
  48 UTC hours.
* **What the raw data says** (data seed 1):

  | | fill rate | edge (¢/share) |
  |---|---|---|
  | 5m | 5–12% | −5.7 to −9.1 |
  | 15m | 2–5% | −3.2 to −4.9 |
  | 4h | ≈ 0.1–0.3% (dt = 1) | −2.3 to +2.0 (n ≈ 800) |

  - The edge is negative at every level and at every fair-value bucket. It is worst late in the
    window (−9¢ at τ < 10%).

## 2. Edge / toxicity model: `model.py`

* **Features.** An MLP on obs 0–32 (market block), 36–43 and 45 (our order state: live, distance,
  queue, rebate, pending) and 47–48. Inventory, wealth and fill counters are excluded.
* **Heads.** For each of the 36 action cells (side x level x size) the network predicts:
  - p_fill (weighted BCE);
  - E[shares | fill];
  - E[edge per share | fill] (MSE, share-weighted);
  - a direct E[credit] head.
* **EV.** factored = p · shares · edge, or the direct head (mode 1), or their average (mode 2,
  which the tuner chose).
* **Blocked CV by UTC hour** (4 folds over 48 hours, data seed 1; `cv/`). 128x2, 3 epochs was
  chosen over 256x3/4 epochs, 64x2 and 32x1.

  | metric (CV mean) | value |
  |---|---|
  | fill AUC | 0.79 (0.78–0.81) |
  | fill calibration (predicted/actual) | 1.05 (0.96–1.14) |
  | corr(predicted, realised edge \| fill) | 0.17 |
  | realised credit per quoted decision, all rows | −$0.0228 |
  | … rows with factored EV > 0 (31% of rows) | −$0.0097 |
  | … rows with direct EV > 0 (40% of rows) | −$0.0041 |
  | … top decile of factored EV | −$0.088 (worse than average) |
  | in-sample (train fit), direct EV > 0 | +$0.0059 (positive: hour-regime overfit) |

* **Edge given fill, by decile of predicted edge** (held-out fold 0):
  - realised −18.6, −7.6, −7.9, −5.9, −5.2, −5.3, −4.0, −4.4, −4.8, **−8.7** ¢/share;
  - predicted from −56¢ up to +78¢.

  The predicted-positive tail is adverse-selected. Bids far under fair value only fill when fair
  value is falling through them.
* **Earlier attempt.** A first version with a Huber loss on edge was biased upwards: it is
  median-like on a left-skewed target. MSE fixed the bias but not the ranking.
* **Models:** `models/edge_s{1,2,3}.pt` (one per data seed), plus their fit metrics in `*.json`.

## 3. The quoter: `quoter.py`, tuned with `tune.py`

* **Rule.** For each side, take the best (level, size ≤ max_size) cell by an inventory-skewed
  score, with edge − k_inv · inv/Q on the bid and + on the ask.
  - Quote that cell only if score > thr and the skewed edge > edge_min (¢); otherwise that side is
    off.
  - Above |inv|/Q > cap, the side that would grow the position is off.
* **Study q1.** 64 TPE trials. Each trial is evaluated with **all 3 data-seed models** (192 val
  evaluations).
  - Objective: seed-mean P&L, with −1000 if any seed fails D1.3 or the seed-mean fills are < 5.
  - 38 of 64 trials passed; all of the passing top 10 use mode 2.
* **Best (trial 54):** thr 0.0025, edge_min 2.40¢, k_inv 0.087, cap 0.77, max_size 1 (2x), mode 2.
* **Study q2.** Per-length thresholds (thr and edge_min for 5m/15m/4h): 64 trials x 3 seeds
  (192 evaluations), started from q1's best. Nothing beat its start point, so the best is q1's
  setting again.
* **Behaviour on train (BC collection, 2M decisions):**
  - the bid is off 79% of the time, the ask 86%;
  - two-sided uptime on val is ≈ 0, so it quotes one side at a time;
  - when it quotes it usually improves 1–2 ticks;
  - size is 2x.
* **Unconstrained best of q1** (trial 5): +$0.32 ± 0.14 per market with 10.9 fills. It fails T6 on
  every seed: the profit is inventory P&L.

| variant | P&L/market (mean ± SE) | per seed | fills/mkt | D1.3 | 5m | 15m | 4h |
|---|---|---|---|---|---|---|---|
| central 1 s (headline) | −0.059 ± 0.018 | −0.08, −0.02, −0.08 | 13.0 | 3/3 | −0.86 | +0.26 | +4.31 |
| pessimistic 1 s | −0.269 ± 0.042 | −0.23, −0.35, −0.22 | 10.5 | 3/3 | −1.00 | +0.05 | +3.63 |
| central 2 s | +0.114 ± 0.043 | +0.05, +0.19, +0.10 | 11.3 | 0/3 (T6) | −0.66 | +0.45 | +4.26 |

Attribution, central 1 s, per seed, summed over the 374 markets (USD):

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| spread capture | +122 | +115 | +94 |
| 60 s markout | −395 | −307 | −335 |
| rebates | +31 | +23 | +21 |
| inventory | +214 | +161 | +190 |

## 4. BC + RL: `bc_w4.py`, `rl_sweep.sh`

* **BC.** train/bc.py, unchanged, with its teacher replaced by the quoter. There is one student per
  data-seed model (`ckpt/bc_m{0,1,2}.pt`).
  - 1024 envs x 2048 steps on train, with reward mode 2, markout 60 and mark_mode 1.
  - MLP 64x2, matching R7, with a value head on Monte-Carlo returns.
  - Held-out accuracy per head: 94–96% / 96–97% / 92–94%; joint 83–87%.
* **PPO.** train_ppo.py with R7's PPO hyper-parameters (`r7_base_cfg.json`, taken from
  train/selected.json).
  - Init = the BC student of seed s; KL to that student, decaying.
  - reward_mode 2, mark_mode 1, 30M steps, 1 thread.
  - Final evaluations under central 1 s, pessimistic 1 s and central 2 s, on val v2.

Central fills, 1 s (headline):

RL_TABLE_CENTRAL

Pessimistic fills, 1 s:

RL_TABLE_PESS

Central fills, 2 s latency:

RL_TABLE_2S

`T6 per seed` is inventory P&L / total P&L, defined only when the total is > 0; 'nan' means the
total was negative. Per-length columns are seed-mean P&L per market.

LONG_RUNS

## What worked, what did not

* **Worked.**
  - **Abstention driven by the fill and toxicity model.** At 13 fills/market the quoter loses only
    $0.06 per market, with 3 seeds tight (SE 0.018). Every forced-to-trade baseline loses more.
  - **The 4h book.** Every w4 method is clearly positive on 4h (+$4 to +$6.6 per market), and
    15m is mildly positive.
  - **RL from the BC warm start** stays close to the teacher and adds about $1.5–2 per 4h market.
* **Did not work.**
  - **Finding edge.** No positive-edge region generalises across hours. The model's ranking of
    fill edge is weak (corr 0.17), and its most confident positive predictions are the most
    toxic fills.
  - **5m.** 5m loses about $0.9–1.3 per market for every method.
  - **Per-length thresholds** (q2) found nothing better than q1's.
  - **More RL** (kl 0.03–1.0, decay 0.25–0.9, lr 1e-4, inventory penalty 1e-3, markout 120) moves
    P&L by less than the seed noise (SE 0.07–0.13). It mostly adds fills on 5m.
  - **Longer RL** (c2 and c8 at 100M steps instead of 30M) is worse at central 1 s (−$0.19 and
    −$0.23 per market). Fills rise to about 19 per market and the extra fills lose on 5m, while 4h
    stays around +$6–7.
* **Observation.** The 60 s label is pessimistic relative to settlement: inventory P&L past 60 s
  is positive in almost every evaluation. Under D1.3 that part counts against a method (T6), so I
  did not optimise for it.

## Files

| file | what |
|---|---|
| `collect.py` | exploratory dataset (train), labels from the C core (reward mode 2, H = 60) |
| `model.py` | edge / fill MLP, blocked-by-hour CV, fit |
| `quoter.py` | the quoter, `fn(obs) -> (B, 3)` |
| `tune.py` | val v2 tuning (64 TPE trials x 3 models) and variant evaluation |
| `bc_w4.py`, `rl_sweep.sh` | BC from the quoter; PPO sweep (8 x 3) and the 100M round |
| `eval_ckpt.py`, `summarize.py` | checkpoint evaluation; tables |
| `models/edge_s{1,2,3}.pt` | edge models (data seeds 1–3) |
| `results/q1_best.json`, `results/q1_log.jsonl`, `results/q2_*` | tuned knobs, every trial |
| `results/*_eval.json`, `results/tables.md`, `results/rl/*.result.json` | evaluations |
| `ckpt/bc_m{0,1,2}.pt` | BC students |
| `ckpt/<config>_s{1,2,3}.pt` + `_config.json` | top-3 RL configs (c2, c8, c6), all seeds |
| `cv/` | CV results |
| `trials.jsonl` | ledger (every val v2 evaluation), `runs.jsonl` run index |

LEDGER_COUNT
