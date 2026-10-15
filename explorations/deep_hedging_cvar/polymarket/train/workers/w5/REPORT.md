# w5: BC warm starts from three teachers + PPO fine-tuning with the fill-markout reward

Worker w5, protocol v2 (train/workers/PROTOCOL.md). Every number in this report is from **val v2**
(`data/build_v2`, split 1, 374 markets, Aug 23–24), with central fills, 1 s latency and greedy
actions. Training uses the v2 train split. The test data was never touched. Ledger:
`trials.jsonl`, with 311 val evaluations (every run's final variants plus 11 teacher/BC
reference evaluations). Run index: `runs.jsonl`. Hashes of mm.h and the build files matched
PROTOCOL.md.

## Bottom line

- **This combination does not improve on its teachers.** After BC, PPO with the fill-markout
  reward (reward_mode 2) and a decaying KL ends close to the BC student.
- **Configs that pass D1.3 lose money.** All of them use the as_logit-active teacher, and the best
  (seed means) lose about **−$0.23 to −$0.24 per market**. The teacher itself loses
  **−$0.19 per market** on val v2, and no-quote makes $0.
- **Configs that make money fail D1.3.** All of them use the fair_value (tuned) teacher: +$0.1–0.3
  per market, but about 4.5–5 fills per market and T6 ≈ 2–4. That is the same directional,
  low-activity behaviour R1 found.
- **Recommendation:** none of these configs should replace the current candidate. If one has to
  be named, `c39_asl_act_h60_rs1` is the best D1.3-passing config here:
  −0.233 ± 0.022 per market (3 seeds), 13.3 fills per market. It still trails its own teacher
  and no-quote.

## What was run

1. **Teachers on val v2.** Each quoter was run through the env with its tuned parameters
   (`eval_teachers.py teachers`). The BC students before PPO were scored as well
   (`eval_teachers.py ckpt`).
2. **BC** (`sweep.py bc` → train/bc.py), one checkpoint per teacher:
   - teachers: as_logit/tuned_active (`asl_act`), fair_value/tuned (`fv`), fair_value/tuned_active
     (`fv_act`);
   - data: 1024 envs × 2048 steps on the train split;
   - fit: 20 epochs, value head fitted on MC returns;
   - policy: MLP 64×2 with a shared trunk, the same as train/selected.json;
   - held-out joint action accuracy: 0.94 (asl_act), 0.99 (fv), 0.94 (fv_act).

   The "off-bias" variants lower the initial logit of the "off" action on both price heads of the
   BC init by 0.5 or 1. The KL teacher is always the unmodified BC policy.
3. **Round 1** (`plan.json` c00–c35): 36 configs × 3 seeds × 25M steps.
   - Settings: reward_mode 2, mark_mode 1.
   - Sampled: markout_h ∈ {60, 120, 300}; kl_coef ∈ [0.01, 0.3] log-uniform;
     kl_decay ∈ {0.25, 0.5, 1}; inv_penalty ∈ [1e-5, 3e-3] log-uniform; off-bias ∈ {0, −0.5, −1};
     lr around 6.8e-4 × 10^U(−0.7, 0.2); ent_coef ∈ [1e-4, 3e-3].
   - Fixed (from the selected config): gamma, λ, horizon 64, 4096 envs.
4. **Round 2** (c36–c49): 14 configs × 3 seeds.
   - Diagnosis from round 1: the value head barely learned the markout signal. Explained
     variance was about 0.03–0.12, because the default reward scale of 1/Q makes the credits
     about 1e-4.
   - Round 2 therefore raised `reward_scale` to 0.1 or 1.0 (10× and 100×) and lowered the KL range
     to [0.01, 0.05].
   - Result: explained variance rose to about 0.13 at reward_scale 1, but P&L did not move.
5. **Not done: the 5-seed runs of the top 3.** The coordinator's D11 pivot (crypto tuning stopped)
   arrived while round 2 was running, so no new crypto runs were launched. The top 3 have 3 seeds
   each.

## Reference policies on val v2

| policy | $/mkt | total | fills/mkt | T6 | D1.3 | 5m total | 15m total | 4h total |
|---|---|---|---|---|---|---|---|---|
| as_logit (tuned_active) | -0.193 | -72.1 | 14.20 | - | pass | -59.0 | -21.3 | +8.2 |
| fair_value (tuned) | +0.270 | +101.1 | 4.96 | +2.19 | fail | -30.4 | +113.4 | +18.1 |
| fair_value (tuned_active) | -0.643 | -240.7 | 21.68 | - | pass | -102.9 | -150.8 | +13.1 |
| as_logit (tuned) | +0.064 | +23.8 | 5.63 | +0.91 | fail | +0.2 | -10.3 | +33.9 |
| no_quote (tuned) | +0.000 | +0.0 | 0.00 | - | fail | +0.0 | +0.0 | +0.0 |
| bc_asl_act | -0.246 | -91.9 | 14.33 | - | pass | -53.1 | -32.5 | -6.2 |
| bc_fv | +0.076 | +28.4 | 4.91 | +6.16 | fail | -90.3 | +112.1 | +6.5 |
| bc_fv_act | -0.595 | -222.3 | 22.08 | - | pass | -82.7 | -146.7 | +7.1 |
| bc_fv_off-2 | +0.053 | +20.0 | 5.16 | +8.75 | fail | -83.9 | +85.7 | +18.1 |
| bc_fv_off-3 | +0.011 | +4.0 | 5.32 | +41.12 | fail | -106.7 | +92.6 | +18.1 |
| bc_asl_act_off+1 | -0.239 | -89.2 | 13.91 | - | pass | -59.4 | -26.1 | -3.7 |

(`bc_*` rows are the BC students before PPO. `_off-k` means the "off" logit was lowered by k.
The extra off-bias did not lift fv's fills meaningfully, and its profit stays directional.)

## Top configs (3 seeds each), with per-length P&L

| config | seeds | $/mkt ± SE | per-seed $/mkt | fills/mkt | 2-sided uptime | T6 | T5 | 5m $/mkt (fills) | 15m $/mkt (fills) | 4h $/mkt (fills) | spread | markout60 | inventory | rebates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c39_asl_act_h60_rs1 | 0,1,2 | -0.233 ± 0.022 | -0.217 -0.277 -0.204 | 13.3 | 0.32 | - | - | -0.250 (17.2) | -0.251 (7.4) | -0.041 (7.9) | +238.3 | -375.2 | +41.8 | +8.1 |
| c24_asl_act_h300 | 0,1,2 | -0.238 ± 0.025 | -0.248 -0.276 -0.191 | 14.0 | 0.33 | - | - | -0.248 (18.1) | -0.227 (8.3) | -0.210 (7.4) | +244.1 | -388.4 | +46.5 | +8.7 |
| c06_asl_act_h300 | 0,1,2 | -0.240 ± 0.040 | -0.241 -0.309 -0.171 | 13.6 | 0.32 | - | - | -0.256 (17.6) | -0.253 (7.7) | -0.085 (7.9) | +241.5 | -375.6 | +35.9 | +8.4 |
| c49_fv_h120_rs0.1 | 0,1,2 | +0.296 ± 0.087 | +0.123 +0.362 +0.403 | 4.5 | 0.05 | +2.06 | - | -0.017 (4.7) | +0.863 (5.3) | +0.373 (0.3) | +203.8 | -339.9 | +228.0 | +18.8 |
| c25_fv_h300 | 0,1,2 | +0.179 ± 0.021 | +0.206 +0.139 +0.192 | 5.0 | 0.05 | +2.85 | - | -0.197 (5.4) | +0.845 (5.6) | +0.324 (0.3) | +221.2 | -365.9 | +190.9 | +20.7 |
| c44_fv_act_h120_rs1 | 0,1,2 | -0.643 ± 0.015 | -0.671 -0.621 -0.638 | 21.1 | 0.26 | - | - | -0.368 (19.5) | -1.418 (27.5) | +0.317 (8.3) | +121.9 | -351.6 | -39.5 | +28.6 |

The three D1.3-passing configs at the top are within one SE of each other and of most of the
as_logit family. The ranking among them is noise.

## Every config (seed mean over 3 seeds, sorted D1.3-pass first, then by $/market)

Per-length columns are $ per market of that contract length (5m: 223, 15m: 119, 4h: 32 markets).
"pess" is the pessimistic-fill variant. T6 is shown only when the seed-mean total is positive.

| config | teacher | H | kl | decay | inv_pen | off | lr | ent | seeds | $/mkt (±SE) | total | fills/mkt | T6 | T5 | D1.3 | 2-sided | 5m $/mkt | 15m $/mkt | 4h $/mkt | pess $/mkt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c39_asl_act_h60_rs1 | asl_act | 60 | 0.030 | 0.25 | 6.6e-04 | 0 | 3.0e-04 | 4.8e-04 | 3 | -0.233 ± 0.022 | -87.0 | 13.3 | - | - | pass | 0.32 | -0.250 | -0.251 | -0.041 | -0.259 |
| c24_asl_act_h300 | asl_act | 300 | 0.046 | 0.5 | 4.0e-05 | 0 | 7.8e-04 | 8.8e-04 | 3 | -0.238 ± 0.025 | -89.2 | 14.0 | - | - | pass | 0.33 | -0.248 | -0.227 | -0.210 | -0.261 |
| c06_asl_act_h300 | asl_act | 300 | 0.176 | 0.25 | 1.1e-05 | -0.5 | 2.9e-04 | 1.1e-03 | 3 | -0.240 ± 0.040 | -89.9 | 13.6 | - | - | pass | 0.32 | -0.256 | -0.253 | -0.085 | -0.296 |
| c48_asl_act_h120_rs1 | asl_act | 120 | 0.019 | 0.25 | 2.2e-04 | 0 | 4.4e-04 | 1.5e-03 | 3 | -0.242 ± 0.016 | -90.6 | 12.8 | - | - | pass | 0.32 | -0.273 | -0.223 | -0.097 | -0.263 |
| c09_asl_act_h60 | asl_act | 60 | 0.064 | 0.25 | 3.1e-05 | -1 | 1.4e-04 | 1.3e-04 | 3 | -0.243 ± 0.008 | -90.9 | 13.4 | - | - | pass | 0.32 | -0.265 | -0.241 | -0.103 | -0.268 |
| c27_asl_act_h60 | asl_act | 60 | 0.163 | 0.5 | 1.1e-03 | 0 | 8.4e-04 | 1.9e-03 | 3 | -0.253 ± 0.017 | -94.5 | 13.5 | - | - | pass | 0.32 | -0.272 | -0.249 | -0.132 | -0.305 |
| c41_asl_act_h300_rs1 | asl_act | 300 | 0.013 | 0.25 | 6.6e-04 | 0 | 2.7e-04 | 6.1e-04 | 3 | -0.253 ± 0.014 | -94.7 | 13.1 | - | - | pass | 0.33 | -0.242 | -0.324 | -0.067 | -0.270 |
| c21_asl_act_h120 | asl_act | 120 | 0.168 | 1 | 1.8e-03 | -0.5 | 1.8e-04 | 1.4e-03 | 3 | -0.255 ± 0.018 | -95.4 | 14.2 | - | - | pass | 0.33 | -0.258 | -0.283 | -0.131 | -0.313 |
| c12_asl_act_h120 | asl_act | 120 | 0.165 | 1 | 5.1e-04 | -0.5 | 1.5e-04 | 6.6e-04 | 3 | -0.259 ± 0.025 | -96.9 | 14.1 | - | - | pass | 0.33 | -0.279 | -0.269 | -0.086 | -0.296 |
| c40_asl_act_h120_rs1 | asl_act | 120 | 0.045 | 0.5 | 3.2e-04 | 0 | 3.2e-04 | 1.9e-03 | 3 | -0.261 ± 0.015 | -97.6 | 13.5 | - | - | pass | 0.32 | -0.279 | -0.256 | -0.154 | -0.292 |
| c03_asl_act_h120 | asl_act | 120 | 0.041 | 1 | 1.6e-03 | -0.5 | 4.5e-04 | 7.9e-04 | 3 | -0.268 ± 0.014 | -100.3 | 14.0 | - | - | pass | 0.33 | -0.281 | -0.277 | -0.144 | -0.307 |
| c30_asl_act_h120 | asl_act | 120 | 0.064 | 0.5 | 1.7e-03 | -1 | 1.7e-04 | 1.5e-04 | 3 | -0.268 ± 0.026 | -100.4 | 14.0 | - | - | pass | 0.32 | -0.305 | -0.253 | -0.072 | -0.271 |
| c37_asl_act_h120_rs0.1 | asl_act | 120 | 0.016 | 0.5 | 1.0e-04 | 0 | 2.4e-04 | 2.5e-04 | 3 | -0.269 ± 0.011 | -100.7 | 13.8 | - | - | pass | 0.32 | -0.294 | -0.269 | -0.097 | -0.273 |
| c36_asl_act_h60_rs0.1 | asl_act | 60 | 0.011 | 0.5 | 4.3e-05 | 0 | 1.0e-03 | 4.8e-04 | 3 | -0.273 ± 0.014 | -102.2 | 13.0 | - | - | pass | 0.32 | -0.285 | -0.297 | -0.099 | -0.266 |
| c47_asl_act_h300_rs0.1 | asl_act | 300 | 0.016 | 0.5 | 7.1e-05 | 0 | 6.0e-04 | 6.1e-04 | 3 | -0.275 ± 0.036 | -102.9 | 13.9 | - | - | pass | 0.33 | -0.299 | -0.254 | -0.187 | -0.268 |
| c18_asl_act_h60 | asl_act | 60 | 0.046 | 0.5 | 1.0e-03 | 0 | 3.7e-04 | 1.2e-04 | 3 | -0.276 ± 0.010 | -103.3 | 13.6 | - | - | pass | 0.32 | -0.316 | -0.259 | -0.063 | -0.266 |
| c33_asl_act_h300 | asl_act | 300 | 0.144 | 0.5 | 1.1e-04 | 0 | 8.8e-04 | 1.4e-03 | 3 | -0.277 ± 0.017 | -103.6 | 13.7 | - | - | pass | 0.33 | -0.278 | -0.310 | -0.148 | -0.287 |
| c15_asl_act_h300 | asl_act | 300 | 0.244 | 0.25 | 1.9e-05 | -1 | 1.1e-03 | 3.4e-04 | 3 | -0.278 ± 0.032 | -103.8 | 14.8 | - | - | pass | 0.34 | -0.262 | -0.366 | -0.055 | -0.290 |
| c46_asl_act_h120_rs0.1 | asl_act | 120 | 0.032 | 0.25 | 9.4e-04 | 0 | 7.7e-04 | 3.3e-04 | 3 | -0.280 ± 0.014 | -104.5 | 13.5 | - | - | pass | 0.33 | -0.281 | -0.293 | -0.219 | -0.307 |
| c00_asl_act_h60 | asl_act | 60 | 0.021 | 0.5 | 3.3e-05 | 0 | 1.0e-03 | 1.4e-04 | 3 | -0.291 ± 0.019 | -109.0 | 12.7 | - | - | pass | 0.32 | -0.316 | -0.290 | -0.127 | -0.261 |
| c38_asl_act_h300_rs0.1 | asl_act | 300 | 0.022 | 0.25 | 5.5e-04 | 0 | 3.7e-04 | 3.9e-04 | 3 | -0.293 ± 0.047 | -109.5 | 14.0 | - | - | pass | 0.33 | -0.301 | -0.312 | -0.163 | -0.270 |
| c44_fv_act_h120_rs1 | fv_act | 120 | 0.034 | 0.5 | 2.3e-05 | 0 | 2.7e-04 | 1.3e-04 | 3 | -0.643 ± 0.015 | -240.7 | 21.1 | - | - | pass | 0.26 | -0.368 | -1.418 | +0.317 | -0.575 |
| c02_fv_act_h60 | fv_act | 60 | 0.091 | 0.25 | 4.0e-05 | 0 | 8.4e-04 | 2.6e-04 | 3 | -0.646 ± 0.013 | -241.8 | 20.7 | - | - | pass | 0.25 | -0.439 | -1.300 | +0.337 | -0.611 |
| c23_fv_act_h120 | fv_act | 120 | 0.284 | 1 | 5.7e-04 | 0 | 4.7e-04 | 3.2e-04 | 3 | -0.651 ± 0.012 | -243.5 | 22.0 | - | - | pass | 0.27 | -0.448 | -1.260 | +0.199 | -0.607 |
| c32_fv_act_h120 | fv_act | 120 | 0.043 | 1 | 7.4e-04 | 0 | 1.7e-04 | 1.1e-03 | 3 | -0.657 ± 0.030 | -245.8 | 21.5 | - | - | pass | 0.27 | -0.393 | -1.385 | +0.213 | -0.623 |
| c29_fv_act_h60 | fv_act | 60 | 0.018 | 1 | 5.8e-04 | -0.5 | 2.2e-04 | 1.1e-04 | 3 | -0.661 ± 0.011 | -247.2 | 21.5 | - | - | pass | 0.27 | -0.426 | -1.355 | +0.279 | -0.600 |
| c14_fv_act_h120 | fv_act | 120 | 0.149 | 1 | 6.3e-05 | -1 | 3.8e-04 | 7.5e-04 | 3 | -0.669 ± 0.018 | -250.2 | 21.8 | - | - | pass | 0.27 | -0.474 | -1.267 | +0.198 | -0.637 |
| c35_fv_act_h300 | fv_act | 300 | 0.015 | 1 | 5.0e-04 | 0 | 5.8e-04 | 4.4e-04 | 3 | -0.674 ± 0.023 | -252.0 | 21.4 | - | - | pass | 0.26 | -0.418 | -1.399 | +0.240 | -0.647 |
| c17_fv_act_h300 | fv_act | 300 | 0.030 | 1 | 1.9e-05 | -0.5 | 5.8e-04 | 1.5e-03 | 3 | -0.676 ± 0.028 | -252.9 | 21.5 | - | - | pass | 0.27 | -0.465 | -1.320 | +0.244 | -0.615 |
| c11_fv_act_h60 | fv_act | 60 | 0.069 | 0.25 | 9.1e-05 | 0 | 9.3e-04 | 2.3e-03 | 3 | -0.678 ± 0.043 | -253.6 | 21.2 | - | - | pass | 0.26 | -0.448 | -1.381 | +0.332 | -0.561 |
| c26_fv_act_h300 | fv_act | 300 | 0.130 | 0.25 | 1.6e-05 | -0.5 | 1.7e-04 | 3.3e-04 | 3 | -0.684 ± 0.012 | -255.7 | 21.6 | - | - | pass | 0.27 | -0.471 | -1.354 | +0.326 | -0.616 |
| c20_fv_act_h60 | fv_act | 60 | 0.074 | 1 | 6.8e-05 | -1 | 4.5e-04 | 1.5e-03 | 3 | -0.688 ± 0.009 | -257.4 | 21.6 | - | - | pass | 0.27 | -0.466 | -1.353 | +0.240 | -0.599 |
| c05_fv_act_h120 | fv_act | 120 | 0.018 | 1 | 3.0e-05 | 0 | 3.6e-04 | 1.1e-03 | 3 | -0.695 ± 0.037 | -259.8 | 21.7 | - | - | pass | 0.27 | -0.433 | -1.441 | +0.257 | -0.618 |
| c45_fv_act_h300_rs1 | fv_act | 300 | 0.024 | 0.5 | 2.1e-04 | 0 | 3.9e-04 | 4.9e-04 | 3 | -0.695 ± 0.016 | -259.8 | 21.3 | - | - | pass | 0.26 | -0.435 | -1.448 | +0.302 | -0.600 |
| c08_fv_act_h300 | fv_act | 300 | 0.010 | 1 | 2.8e-04 | -0.5 | 3.1e-04 | 1.7e-03 | 3 | -0.704 ± 0.011 | -263.2 | 21.7 | - | - | pass | 0.27 | -0.461 | -1.436 | +0.325 | -0.659 |
| c42_fv_act_h120_rs0.1 | fv_act | 120 | 0.033 | 0.25 | 2.4e-05 | 0 | 2.5e-04 | 5.3e-04 | 3 | -0.717 ± 0.038 | -268.3 | 21.1 | - | - | pass | 0.26 | -0.444 | -1.489 | +0.246 | -0.605 |
| c43_fv_act_h300_rs0.1 | fv_act | 300 | 0.027 | 0.25 | 9.0e-05 | 0 | 3.9e-04 | 1.8e-03 | 3 | -0.739 ± 0.005 | -276.5 | 21.6 | - | - | pass | 0.27 | -0.549 | -1.376 | +0.299 | -0.632 |
| c49_fv_h120_rs0.1 | fv | 120 | 0.030 | 0.25 | 8.1e-04 | 0 | 6.5e-04 | 1.2e-04 | 3 | +0.296 ± 0.087 | +110.8 | 4.5 | +2.06 | - | fail | 0.05 | -0.017 | +0.863 | +0.373 | -0.112 |
| c25_fv_h300 | fv | 300 | 0.020 | 1 | 1.3e-04 | -0.5 | 5.5e-04 | 2.6e-04 | 3 | +0.179 ± 0.021 | +66.9 | 5.0 | +2.85 | - | fail | 0.05 | -0.197 | +0.845 | +0.324 | -0.170 |
| c10_fv_h60 | fv | 60 | 0.015 | 1 | 1.4e-05 | -0.5 | 5.0e-04 | 1.7e-03 | 3 | +0.162 ± 0.021 | +60.5 | 4.8 | +3.64 | - | fail | 0.05 | -0.270 | +0.927 | +0.324 | -0.167 |
| c28_fv_h60 | fv | 60 | 0.018 | 0.5 | 2.7e-04 | 0 | 6.3e-04 | 1.1e-04 | 3 | +0.146 ± 0.072 | +54.5 | 4.9 | +3.33 | - | fail | 0.05 | -0.211 | +0.728 | +0.469 | -0.169 |
| c22_fv_h120 | fv | 120 | 0.239 | 0.25 | 1.7e-03 | -1 | 1.7e-04 | 8.1e-04 | 3 | +0.140 ± 0.034 | +52.3 | 4.9 | +3.97 | - | fail | 0.05 | -0.333 | +0.937 | +0.469 | -0.213 |
| c31_fv_h120 | fv | 120 | 0.117 | 1 | 2.5e-03 | -1 | 9.6e-04 | 1.3e-03 | 3 | +0.133 ± 0.072 | +49.7 | 5.0 | +4.12 | - | fail | 0.05 | -0.380 | +1.042 | +0.324 | -0.140 |
| c34_fv_h300 | fv | 300 | 0.132 | 0.25 | 4.6e-05 | 0 | 2.4e-04 | 1.7e-04 | 3 | +0.128 ± 0.047 | +47.9 | 4.9 | +4.20 | - | fail | 0.05 | -0.298 | +0.873 | +0.324 | -0.162 |
| c04_fv_h120 | fv | 120 | 0.025 | 0.25 | 1.7e-05 | -0.5 | 2.7e-04 | 1.6e-04 | 3 | +0.116 ± 0.097 | +43.2 | 4.9 | +4.39 | - | fail | 0.05 | -0.348 | +0.975 | +0.150 | -0.175 |
| c19_fv_h60 | fv | 60 | 0.023 | 0.25 | 1.0e-03 | 0 | 6.4e-04 | 3.0e-03 | 3 | +0.107 ± 0.037 | +40.1 | 4.6 | +4.13 | - | fail | 0.05 | -0.234 | +0.621 | +0.571 | -0.104 |
| c07_fv_h300 | fv | 300 | 0.161 | 1 | 6.3e-04 | 0 | 8.2e-04 | 2.4e-03 | 3 | +0.105 ± 0.049 | +39.2 | 4.9 | +4.42 | - | fail | 0.05 | -0.302 | +0.841 | +0.203 | -0.182 |
| c01_fv_h60 | fv | 60 | 0.039 | 0.25 | 2.0e-04 | -1 | 1.1e-03 | 1.6e-03 | 3 | +0.088 ± 0.082 | +32.8 | 4.9 | +5.59 | - | fail | 0.05 | -0.284 | +0.721 | +0.326 | -0.066 |
| c13_fv_h120 | fv | 120 | 0.150 | 0.25 | 1.2e-03 | -1 | 2.9e-04 | 2.3e-04 | 3 | +0.073 ± 0.046 | +27.3 | 4.9 | +7.05 | - | fail | 0.05 | -0.354 | +0.903 | -0.042 | -0.206 |
| c16_fv_h300 | fv | 300 | 0.052 | 0.25 | 1.9e-04 | -1 | 1.5e-04 | 1.7e-04 | 3 | +0.038 ± 0.076 | +14.3 | 5.0 | +11.76 | - | fail | 0.05 | -0.412 | +0.772 | +0.445 | -0.250 |

## What worked and what didn't

- **Teacher choice decides almost everything.** Results cluster tightly by teacher, whatever the
  markout horizon, KL schedule, inventory penalty, off-bias or reward scale:
  - asl_act: −0.23 to −0.29 per market;
  - fv_act: −0.64 to −0.74;
  - fv: +0.04 to +0.30, but it fails T6.

  So PPO fine-tuning at 25M steps does not leave the teacher's basin. The sampled policy does
  drift (KL to the teacher reaches 0.04–0.2 and entropy rises), but the greedy policy stays at
  the teacher's actions.
- **Markout horizon** (60/120/300 s): no consistent effect.
- **reward_scale 100×:** improves value learning, but does not improve P&L. c39 is the top row,
  but within noise.
- **Off-bias** (anti-off at init) doesn't move activity either. BC logits are sharp: cross-entropy
  is 0.03 for fv, and even a −3 bias only takes BC-fv from 4.9 to 5.3 fills per market. With the
  fv teacher, more fills just means more losing fills; its profit is inventory/resolution P&L
  (T6 > 2), not spread capture.
- **Where the losses come from:** fill-level edge is negative for every family. Spread capture of
  about +$240 is outweighed by the 60 s markout of about −$380 (as_logit family). The loss is
  spread evenly over 5m and 15m markets; 4h is close to flat.
- **BC alone loses some of the teacher's edge.** For asl_act the student is at −0.246 against the
  teacher's −0.193, and fine-tuning recovers part of that at best.

## Files

- `eval_teachers.py`, `sweep.py`, `summarize.py`, `tables.py`: the scripts used here. Run them
  from this folder with the project venv.
- `plan.json`: every config. `teachers.json`: teacher and BC reference metrics.
- `ckpt/<config>_s<seed>.pt` + `.config.json` for the top 3 configs, seeds 0–2. The KL teacher
  and init are `ckpt/bc_asl_act.pt`; c06's init is `ckpt/bc_asl_act_off-0.5.pt`.
- `trials.jsonl` (ledger) and `runs.jsonl` (run index).
