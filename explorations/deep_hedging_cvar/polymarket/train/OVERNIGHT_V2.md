# Protocol v2 progress (train/)

Updated 2026-10-15 19:43 UTC. Protocol: eval/DECISIONS.md D7. v2 validation =
`data/build_v2` split 1 = Aug 23 + Aug 24, 374 markets (223 5m, 119 15m, 32 4h); its train split is
byte-identical to data/build's (checked). Greedy actions, central fills, 1 s latency, every market once.
The Jun 15 test build is refused by `envs.no_jun15` everywhere in train/. Every evaluation below is a row
in `train/trials.jsonl` (tags `v2_ensemble`, `v2_baselines`, `v2s1`, `v2s1_filter`).

How to resume / reproduce:
```bash
cd explorations/deep_hedging_cvar/polymarket/train && source ../../.venv/bin/activate
python v2eval.py ensembles && python v2eval.py baselines && python v2eval.py report   # steps 1-2 (resumable: runs/v2/*.json)
python v2search.py run --name v2s1 --detach     # step 3 (resumable: finished runs are reused)
python v2search.py status --name v2s1           # progress table -> runs/v2/v2s1/progress.md
python v2search.py select --name v2s1           # -> runs/v2/v2s1_selected.json
```

## Steps 1-2: seed ensembles and per-length filters on v2 val

T6 = inventory/resolution P&L / total P&L (> 0.5 = directional; > 1 means spread capture + 60 s
markout is negative and the whole profit is resolution). D1.3 = fills/market >= 5 and T6 <= 0.5.

**Findings**
1. **No RL policy makes money as a market maker on v2 val.** Every positive P&L (R7 ensemble +127 / +150,
   R1 tuned-BC configs +30 to +255 per seed, the main-sweep ensemble quoting only 4h +169) is
   inventory/resolution P&L: T6 is 0.7-5.5, mostly > 1. The policies that pass D1.3 all lose
   (R7 seeds 101/102 -554/-249, main seeds 100/104 -305/-311, main ensemble -14).
2. **Ensembling removes most of the seed spread** (R7 seeds -554 ... +102 -> ensemble +127 logits /
   +150 probabilities) mainly by quoting less: fills/market 22-27 per seed -> 8 for the ensemble
   (members disagree, the averaged logits more often pick 'off' or a deep quote). Probability averaging
   beat logit averaging in 5 of 10 sets (no consistent winner).
3. **By contract length:** 4h is positive for almost every policy (R7 ensemble +116, main ensemble +169,
   fv_smart (active) +269), 5m is where the losses are (R7 seeds -18 to -515, sym_touch (active) -465).
   But the 4h profits are directional too (T6 ~1.0 with the 4h-only filter): 32 markets, resolution.
4. **Baselines on v2 val:** fair_value +101 (T6 2.2, directional, 5.0 fills), as_logit +24 (T6 0.91),
   every activity-constrained baseline (D1.2) loses (-72 to -778). Best length filter per baseline:
   fair_value 15m+4h +132, fv_smart (active) 4h +269 (T6 1.15).
5. **Length filters change the picture only through 4h resolution exposure**; no (policy, filter) pair
   with positive P&L passes D1.3. Filtering cuts fills/market below 5 (only 32 4h markets of 374).

### r7_final

| policy | P&L | /market | CVaR95 | fills/mkt | T6 | D1.3 | P&L 5m | P&L 15m | P&L 4h |
|---|---|---|---|---|---|---|---|---|---|
| seed_100 | +64.1 | +0.17 | -55.7 | +22.5 | +5.53 | no | -18.2 | -142.8 | +225.1 |
| seed_101 | -554.2 | -1.48 | -50.6 | +27.2 | - | yes | -514.8 | -232.6 | +193.2 |
| seed_102 | -248.8 | -0.67 | -32.0 | +25.8 | - | yes | -214.4 | -210.8 | +176.3 |
| seed_103 | +48.2 | +0.13 | -37.0 | +12.0 | +4.00 | no | -100.4 | +15.8 | +132.8 |
| seed_104 | +101.9 | +0.27 | -43.5 | +23.0 | +4.88 | no | -110.9 | -26.3 | +239.2 |
| ensemble_logits_5 | +127.4 | +0.34 | -37.1 | +8.6 | +1.07 | no | +0.2 | +10.8 | +116.4 |
| ensemble_probs_5 | +150.3 | +0.40 | -32.9 | +7.8 | +0.69 | no | +26.8 | +18.3 | +105.3 |

Length filters (r7_final, ensemble_logits_5):

| policy | P&L | /market | CVaR95 | fills/mkt | T6 | D1.3 | P&L 5m | P&L 15m | P&L 4h |
|---|---|---|---|---|---|---|---|---|---|
| all lengths | +127.4 | +0.34 | -37.1 | +8.6 | +1.07 | no | +0.2 | +10.8 | +116.4 |
| quote 4h | +116.4 | +0.31 | -4.3 | +0.6 | +1.02 | no | +0.0 | +0.0 | +116.4 |
| quote 15m+4h | +127.2 | +0.34 | -20.7 | +3.4 | +1.00 | no | +0.0 | +10.8 | +116.4 |
| quote 15m | +10.8 | +0.03 | -17.0 | +2.8 | +0.85 | no | +0.0 | +10.8 | +0.0 |
| quote 5m | +0.2 | +0.00 | -27.3 | +5.1 | +46.49 | no | +0.2 | +0.0 | +0.0 |
| quote 5m+15m | +11.0 | +0.03 | -34.6 | +7.9 | +1.62 | no | +0.2 | +10.8 | +0.0 |

### main_final

| policy | P&L | /market | CVaR95 | fills/mkt | T6 | D1.3 | P&L 5m | P&L 15m | P&L 4h |
|---|---|---|---|---|---|---|---|---|---|
| seed_100 | -305.3 | -0.82 | -57.9 | +22.2 | - | yes | -219.1 | -236.7 | +150.5 |
| seed_101 | -71.9 | -0.19 | -10.5 | +2.1 | - | no | -13.1 | -76.8 | +18.0 |
| seed_102 | -226.8 | -0.61 | -16.6 | +2.6 | - | no | -88.3 | -126.9 | -11.6 |
| seed_103 | +47.0 | +0.13 | -33.8 | +18.8 | +5.09 | no | -72.4 | -82.0 | +201.4 |
| seed_104 | -310.6 | -0.83 | -52.9 | +30.2 | - | yes | -290.0 | -239.2 | +218.6 |
| ensemble_logits_5 | -14.5 | -0.04 | -26.6 | +6.4 | - | yes | -164.1 | -19.6 | +169.2 |
| ensemble_probs_5 | +4.4 | +0.01 | -19.2 | +2.8 | +2.93 | no | -94.3 | -7.5 | +106.2 |

Length filters (main_final, ensemble_logits_5):

| policy | P&L | /market | CVaR95 | fills/mkt | T6 | D1.3 | P&L 5m | P&L 15m | P&L 4h |
|---|---|---|---|---|---|---|---|---|---|
| all lengths | -14.5 | -0.04 | -26.6 | +6.4 | - | yes | -164.1 | -19.6 | +169.2 |
| quote 4h | +169.2 | +0.45 | -3.7 | +0.6 | +1.00 | no | +0.0 | +0.0 | +169.2 |
| quote 15m+4h | +149.6 | +0.40 | -14.8 | +2.6 | +1.29 | no | +0.0 | -19.6 | +169.2 |
| quote 15m | -19.6 | -0.05 | -11.4 | +2.0 | - | no | +0.0 | -19.6 | +0.0 |
| quote 5m | -164.1 | -0.44 | -17.2 | +3.7 | - | no | -164.1 | +0.0 | +0.0 |
| quote 5m+15m | -183.7 | -0.49 | -23.7 | +5.7 | - | yes | -164.1 | -19.6 | +0.0 |

### baselines

| policy | P&L | /market | CVaR95 | fills/mkt | T6 | D1.3 | P&L 5m | P&L 15m | P&L 4h |
|---|---|---|---|---|---|---|---|---|---|
| no_quote | +0.0 | +0.00 | +0.0 | +0.0 | - | no | +0.0 | +0.0 | +0.0 |
| sym_touch | +0.0 | +0.00 | +0.0 | +0.0 | - | no | +0.0 | +0.0 | +0.0 |
| as_logit | +23.8 | +0.06 | -1.6 | +5.6 | +0.91 | no | +0.2 | -10.3 | +33.9 |
| fair_value | +101.1 | +0.27 | -24.0 | +5.0 | +2.19 | no | -30.4 | +113.4 | +18.1 |
| fv_smart | -8.0 | -0.02 | -3.2 | +0.4 | - | no | -8.0 | +0.0 | +0.0 |
| sym_touch (active) | -723.2 | -1.93 | -11.3 | +46.0 | - | yes | -464.9 | -225.1 | -33.2 |
| as_logit (active) | -72.1 | -0.19 | -5.2 | +14.2 | - | yes | -59.0 | -21.3 | +8.2 |
| fair_value (active) | -240.7 | -0.64 | -11.3 | +21.7 | - | yes | -102.9 | -150.8 | +13.1 |
| fv_smart (active) | -777.8 | -2.08 | -59.7 | +55.9 | - | yes | -709.7 | -337.5 | +269.4 |

Baselines, length filters (best filter per baseline):

| policy | P&L | /market | CVaR95 | fills/mkt | T6 | D1.3 | P&L 5m | P&L 15m | P&L 4h |
|---|---|---|---|---|---|---|---|---|---|
| no_quote | quote 4h | +0.0 | +0.00 | +0.0 | +0.0 | - | no | +0.0 | +0.0 | +0.0 |
| sym_touch | quote 4h | +0.0 | +0.00 | +0.0 | +0.0 | - | no | +0.0 | +0.0 | +0.0 |
| as_logit | quote 4h | +33.9 | +0.09 | -0.1 | +1.3 | +0.71 | no | +0.0 | +0.0 | +33.9 |
| fair_value | quote 15m+4h | +131.5 | +0.35 | -9.3 | +1.8 | +0.86 | no | +0.0 | +113.4 | +18.1 |
| fv_smart | quote 4h | +0.0 | +0.00 | +0.0 | +0.0 | - | no | +0.0 | +0.0 | +0.0 |
| sym_touch (active) | quote 4h | -33.2 | -0.09 | -2.0 | +2.8 | - | no | +0.0 | +0.0 | -33.2 |
| as_logit (active) | quote 4h | +8.2 | +0.02 | -0.1 | +0.4 | +0.39 | no | +0.0 | +0.0 | +8.2 |
| fair_value (active) | quote 4h | +13.1 | +0.04 | -0.7 | +0.8 | +1.30 | no | +0.0 | +0.0 | +13.1 |
| fv_smart (active) | quote 4h | +269.4 | +0.72 | -1.1 | +3.4 | +1.15 | no | +0.0 | +0.0 | +269.4 |

R1 tuned-BC configs (8 configs x 3 seeds + ensembles; all fills/market ~4, T6 1.0-34, none pass D1.3):
see `runs/v2/report_step12.md` (`python v2eval.py report`).

## Step 3: reward_mode-2 search `v2s1` (32 configs x 3 seeds, 25M steps each)

Running since 19:31 UTC (detached, resumable; 4 processes x 1 torch thread; ~110 min). Space: markout_h
{60,120,300}, inv_penalty log [1e-6,1e-2], lr log [1e-4,3e-3], ent_coef log [1e-4,3e-2], hidden {64,128},
close-inventory penalty (off / coef log [1e-5,3e-3] over the last 20% of the window), training len_mask
{all, 4h only, 15m only}; base = the R7 final config (off prior 0.5, fair-value mark). Each run is
scored on v2 val and every length filter is derived from its per-market records (each ledgered).
Selection: best seed mean (3 seeds) of v2 val P&L among D1.3-passing (config, filter) pairs.
Live table: `runs/v2/v2s1/progress.md`.
