# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 9 min. params.json sha256 `bdf21b1fc910f252`, mm.h sha256 `746dd7dea250eee0`.

**Setup**

- Real: the 431 train markets of data/build (opens on 2026-08-21..22).
- Sim: the same schedule on the same Binance paths, 20 replications of the base generator (no randomisation). Each was written in the replay format, passed through data/features.py unmodified (sim/features_sim.py) and played by the C env (mm.h through baselines/runner.py) with identical settings.
- Fill model central, latency 1 s, offset 0, env defaults otherwise.
- Policies: baselines.py `no_quote`, `sym_touch` and `fair_value` at their untuned DEFAULTS.
- SIMULATOR.md asks for 200 replications. 20 were run because CPU is shared with PPO, so the 5-95% bands are rough.

## Overall: **FAIL**

| gate | result |
|---|---|
| G1 P&L inside sim 90% band + hourly KS < 0.15 | no_quote PASS, sym_touch FAIL, fair_value FAIL |
| G2 rank order | FAIL: real ['no_quote', 'fair_value', 'sym_touch'], sim ['fair_value', 'no_quote', 'sym_touch'] |
| G3 attribution, fills, markout curve inside the 90% band | FAIL |
| G4 stylised facts (numpy statistics on the same chunks) | FAIL |
| G5 out-of-sample on the val day | not run: it needs a ledger entry (EVALUATION.md T4) |

Gate definitions: **G1** per policy: real total and median per-market P&L inside the sim 5-95% band of replications, and KS D < 0.15 between the real hourly P&L series and the pooled sim hourly series (markets binned by end hour, so block structure is kept); per-market KS D is reported; **G2** rank order of {no_quote, sym_touch, fair_value} by total P&L identical (real vs sim median); **G3** per-market means of fills, fill volume, spread capture, markout 5/30/60, inventory P&L and rebates, and sym_touch's markout curve (c/share), inside the sim 5-95% band; **G4** stylised facts as in validate.py GATE (spread, depth, one-sided, trade rate, P(buy|gap), sign ACF)

## G1/G2: P&L through the C env ($)

| policy | real total | sim total median [5%, 95%] | real median/mkt | sim median/mkt [5%, 95%] | KS hourly | KS per-market | pass |
|---|---|---|---|---|---|---|---|
| sym_touch | -2957.6 | -2706.3 [-3003.5, -2132.9] | -3.010 | [-1.906, -1.007] | 0.213 | 0.131 | FAIL |
| fair_value | -796.6 | 1288.7 [782.6, 1765.7] | 0.000 | [0.014, 0.622] | 0.397 | 0.201 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -1774.2 [-2065.3, -1413.2] |
| sym_touch | 15m | -952.6 | -794.6 [-1094.8, -591.7] |
| sym_touch | 4h | -93.5 | 2.4 [-93.3, 60.2] |
| fair_value | 5m | -553.0 | 731.7 [469.2, 1095.1] |
| fair_value | 15m | -336.2 | 427.1 [18.6, 649.2] |
| fair_value | 4h | 92.7 | 132.2 [48.3, 188.2] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 127.813 [119.601, 134.365] | False |
| sym_touch | fill_vol | 164.380 | 261.642 [240.184, 276.657] | False |
| sym_touch | spread_capture | 0.675 | 2.192 [1.945, 2.415] | False |
| sym_touch | markout_5 | -4.763 | -5.273 [-5.567, -4.788] | False |
| sym_touch | markout_30 | -7.265 | -9.840 [-11.081, -9.017] | False |
| sym_touch | markout_60 | -7.982 | -10.567 [-11.874, -9.100] | False |
| sym_touch | inventory_pnl | 0.037 | 1.468 [1.103, 2.590] | False |
| sym_touch | rebates | 0.407 | 0.584 [0.536, 0.617] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -1.999 [-2.092, -1.858] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -3.803 [-4.177, -3.571] | False |
| sym_touch | markout 60s (c/share) | -4.856 | -4.033 [-4.678, -3.620] | False |
| fair_value | fills | 49.930 | 54.298 [49.789, 57.793] | True |
| fair_value | fill_vol | 85.714 | 112.814 [104.101, 120.956] | False |
| fair_value | spread_capture | 1.354 | 2.122 [2.008, 2.275] | False |
| fair_value | markout_5 | -2.611 | -1.842 [-1.982, -1.644] | False |
| fair_value | markout_30 | -3.815 | -2.551 [-3.000, -2.253] | False |
| fair_value | markout_60 | -4.047 | -1.513 [-2.118, -0.838] | False |
| fair_value | inventory_pnl | 0.629 | 2.004 [1.074, 3.160] | False |
| fair_value | rebates | 0.216 | 0.240 [0.217, 0.261] | False |
| fair_value | markout 5s (c/share) | -3.046 | -1.624 [-1.694, -1.529] | False |
| fair_value | markout 30s (c/share) | -4.451 | -2.332 [-2.553, -2.032] | False |
| fair_value | markout 60s (c/share) | -4.721 | -1.427 [-1.866, -0.744] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=1, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | FAIL (n_fail=3, n=19) |
| ks_spread|5m | PASS (D=0.040) |
| ks_spread|15m | PASS (D=0.013) |
| ks_spread|4h | PASS (D=0.038) |
| ks_log_top_sz|5m | PASS (D=0.027) |
| ks_log_top_sz|15m | PASS (D=0.014) |
| ks_log_top_sz|4h | PASS (D=0.006) |
| ks_log_top5|5m | PASS (D=0.036) |
| ks_log_top5|15m | PASS (D=0.037) |
| ks_log_top5|4h | PASS (D=0.051) |
| ks_trades_per_market|5m | PASS (D=0.115) |
| ks_trades_per_market|15m | PASS (D=0.134) |
| ks_trades_per_market|4h | PASS (D=0.128) |
| pbuy_gap_curve | FAIL (n_fail=3, n=29) |
| sign_acf1|5m | FAIL (real=0.232, sim=0.305) |
| sign_acf1|15m | PASS (real=0.298, sim=0.291) |
| sign_acf1|4h | PASS (real=0.262, sim=0.254) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.80 | 50/50 | 250/185 | 0.32/0.27 | 0.388/0.352 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.63 | 50/50 | 255/210 | 0.29/0.26 | 0.597/0.595 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.015 | 0.47/0.50 | 50/50 | 260/250 | 0.29/0.26 | 1.424/1.129 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.45 | 50/50 | 300/285 | 0.32/0.28 | 1.434/1.548 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.45 | 50/50 | 322/311 | 0.32/0.28 | 1.652/1.758 |
| 15m | 0 | 0.899/0.884 | 0.067/0.013 | 0.010/0.009 | 0.61/0.85 | 50/50 | 185/168 | 0.30/0.22 | 0.028/0.057 |
| 15m | 1 | 0.834/0.830 | 0.085/0.025 | 0.010/0.009 | 0.58/0.75 | 50/50 | 250/215 | 0.25/0.19 | 0.071/0.091 |
| 15m | 2 | 0.700/0.697 | 0.085/0.041 | 0.020/0.010 | 0.46/0.51 | 50/50 | 250/252 | 0.23/0.18 | 0.221/0.168 |
| 15m | 3 | 0.450/0.448 | 0.056/0.055 | 0.019/0.020 | 0.48/0.46 | 50/50 | 317/323 | 0.26/0.19 | 0.311/0.228 |
| 15m | 4 | 0.175/0.170 | 0.067/0.080 | 0.020/0.020 | 0.43/0.43 | 50/50 | 427/372 | 0.29/0.21 | 0.381/0.347 |
| 15m | 5 | 0.006/0.005 | 0.092/0.096 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/387 | 0.29/0.21 | 0.359/0.338 |
| 4h | 0 | 0.700/0.676 | 0.081/0.027 | 0.008/0.008 | 0.75/0.82 | 50/58 | 265/225 | 0.07/0.07 | 0.065/0.026 |
| 4h | 1 | 0.672/0.641 | 0.081/0.027 | 0.009/0.009 | 0.77/0.77 | 55/57 | 315/238 | 0.11/0.11 | 0.030/0.021 |
| 4h | 2 | 0.532/0.516 | 0.081/0.029 | 0.009/0.009 | 0.66/0.72 | 100/54 | 315/223 | 0.07/0.07 | 0.048/0.034 |
| 4h | 3 | 0.447/0.434 | 0.084/0.054 | 0.009/0.009 | 0.61/0.66 | 103/50 | 294/250 | 0.09/0.10 | 0.034/0.020 |
| 4h | 4 | 0.203/0.215 | 0.095/0.078 | 0.009/0.009 | 0.63/0.62 | 55/52 | 316/250 | 0.06/0.06 | 0.023/0.019 |
| 4h | 5 | 0.093/0.081 | 0.084/0.081 | 0.009/0.009 | 0.58/0.59 | 53/50 | 320/260 | 0.09/0.09 | 0.022/0.018 |
| 4h | 6 | 0.047/0.040 | 0.081/0.081 | 0.011/0.011 | 0.49/0.50 | 50/50 | 338/290 | 0.10/0.10 | 0.015/0.013 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.010 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.431 |
| 5m|1 | 12459 | 0.474 | 0.483 |
| 5m|2 | 12977 | 0.458 | 0.492 |
| 5m|3 | 6481 | 0.457 | 0.514 |
| 5m|4 | 7216 | 0.513 | 0.578 |
| 5m|5 | 8577 | 0.504 | 0.428 |
| 5m|6 | 6910 | 0.538 | 0.493 |
| 5m|7 | 13334 | 0.536 | 0.519 |
| 5m|8 | 11837 | 0.541 | 0.531 |
| 5m|9 | 13657 | 0.568 | 0.563 |
| 15m|0 | 2616 | 0.426 | 0.422 |
| 15m|1 | 4776 | 0.460 | 0.453 |
| 15m|2 | 7510 | 0.489 | 0.483 |
| 15m|3 | 3528 | 0.460 | 0.466 |
| 15m|4 | 3913 | 0.496 | 0.532 |
| 15m|5 | 4166 | 0.528 | 0.463 |
| 15m|6 | 3528 | 0.556 | 0.531 |
| 15m|7 | 7364 | 0.565 | 0.556 |
| 15m|8 | 5177 | 0.583 | 0.575 |
| 15m|9 | 2607 | 0.600 | 0.592 |
| 4h|1 | 455 | 0.433 | 0.473 |
| 4h|2 | 545 | 0.424 | 0.426 |
| 4h|3 | 363 | 0.424 | 0.443 |
| 4h|4 | 532 | 0.451 | 0.469 |
| 4h|5 | 745 | 0.541 | 0.503 |
| 4h|6 | 573 | 0.635 | 0.577 |
| 4h|7 | 1301 | 0.587 | 0.573 |
| 4h|8 | 1147 | 0.616 | 0.609 |
| 4h|9 | 655 | 0.594 | 0.575 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.687 | [0.676, 0.704] |
| align_through|5m|k0 | 0.446 | 0.580 | [0.562, 0.600] |
| align_at_or_through|15m|k0 | 0.561 | 0.698 | [0.679, 0.723] |
| align_through|15m|k0 | 0.407 | 0.527 | [0.504, 0.573] |
| align_at_or_through|4h|k0 | 0.498 | 0.689 | [0.673, 0.707] |
| align_through|4h|k0 | 0.206 | 0.444 | [0.429, 0.466] |
| align_at_or_through|5m|k1 | 0.561 | 0.715 | [0.703, 0.733] |
| align_through|5m|k1 | 0.453 | 0.590 | [0.569, 0.611] |
| align_at_or_through|15m|k1 | 0.587 | 0.715 | [0.694, 0.740] |
| align_through|15m|k1 | 0.424 | 0.527 | [0.503, 0.575] |
| align_at_or_through|4h|k1 | 0.537 | 0.705 | [0.682, 0.719] |
| align_through|4h|k1 | 0.225 | 0.445 | [0.424, 0.463] |
| align_at_or_through|5m|k2 | 0.590 | 0.744 | [0.731, 0.761] |
| align_through|5m|k2 | 0.469 | 0.597 | [0.576, 0.618] |
| align_at_or_through|15m|k2 | 0.621 | 0.728 | [0.712, 0.756] |
| align_through|15m|k2 | 0.453 | 0.525 | [0.502, 0.578] |
| align_at_or_through|4h|k2 | 0.577 | 0.720 | [0.704, 0.736] |
| align_through|4h|k2 | 0.247 | 0.441 | [0.425, 0.467] |
| align_at_or_through|5m|k3 | 0.626 | 0.773 | [0.757, 0.788] |
| align_through|5m|k3 | 0.511 | 0.600 | [0.577, 0.620] |
| align_at_or_through|15m|k3 | 0.665 | 0.738 | [0.724, 0.763] |
| align_through|15m|k3 | 0.507 | 0.524 | [0.499, 0.573] |
| align_at_or_through|4h|k3 | 0.628 | 0.735 | [0.719, 0.752] |
| align_through|4h|k3 | 0.287 | 0.437 | [0.417, 0.464] |
| align_at_or_through|5m|k4 | 0.643 | 0.745 | [0.730, 0.758] |
| align_through|5m|k4 | 0.558 | 0.617 | [0.596, 0.634] |
| align_at_or_through|15m|k4 | 0.682 | 0.715 | [0.702, 0.739] |
| align_through|15m|k4 | 0.557 | 0.540 | [0.520, 0.583] |
| align_at_or_through|4h|k4 | 0.641 | 0.724 | [0.715, 0.741] |
| align_through|4h|k4 | 0.317 | 0.444 | [0.429, 0.468] |
| align_at_or_through|5m|k6 | 0.624 | 0.664 | [0.650, 0.673] |
| align_through|5m|k6 | 0.572 | 0.614 | [0.596, 0.626] |
| align_at_or_through|15m|k6 | 0.659 | 0.657 | [0.642, 0.677] |
| align_through|15m|k6 | 0.567 | 0.563 | [0.549, 0.596] |
| align_at_or_through|4h|k6 | 0.630 | 0.678 | [0.666, 0.695] |
| align_through|4h|k6 | 0.344 | 0.470 | [0.454, 0.491] |
| sign_acf|5m|1 | 0.232 | 0.305 | [0.293, 0.321] |
| sign_acf|5m|2 | 0.196 | 0.212 | [0.198, 0.235] |
| sign_acf|5m|5 | 0.166 | 0.164 | [0.148, 0.185] |
| sign_acf|15m|1 | 0.298 | 0.291 | [0.274, 0.298] |
| sign_acf|15m|2 | 0.206 | 0.152 | [0.135, 0.171] |
| sign_acf|15m|5 | 0.123 | 0.060 | [0.041, 0.097] |
| sign_acf|4h|1 | 0.262 | 0.254 | [0.229, 0.269] |
| sign_acf|4h|2 | 0.222 | 0.071 | [0.044, 0.093] |
| sign_acf|4h|5 | 0.138 | 0.005 | [-0.013, 0.018] |
| markout|5m|0 | 2.975 | 3.712 | [3.247, 4.325] |
| markout|15m|0 | 1.983 | 2.077 | [1.883, 2.497] |
| markout|4h|0 | 1.127 | 1.631 | [1.526, 1.745] |
| markout|5m|5 | 1.611 | 2.889 | [2.454, 3.437] |
| markout|15m|5 | 0.634 | 1.710 | [1.572, 2.091] |
| markout|4h|5 | 0.476 | 1.523 | [1.414, 1.650] |
| markout|5m|30 | 0.527 | 0.915 | [0.042, 1.322] |
| markout|15m|30 | 0.004 | 1.045 | [0.425, 1.240] |
| markout|4h|30 | 0.017 | 0.876 | [0.719, 1.081] |
| markout|5m|60 | 0.111 | 0.799 | [-0.104, 1.249] |
| markout|15m|60 | 0.117 | 0.911 | [0.449, 1.118] |
| markout|4h|60 | -0.087 | 0.637 | [0.477, 0.849] |
| taker_size_med|5m | 5.880 | 5.266 | [5.130, 5.735] |
| taker_size_p90|5m | 40.159 | 38.737 | [36.702, 39.907] |
| levels_per_tx|5m | 1.179 | 1.166 | [1.161, 1.179] |
| buy_share|5m | 0.498 | 0.502 | [0.496, 0.509] |
| taker_size_med|15m | 5.000 | 5.033 | [5.000, 5.992] |
| taker_size_p90|15m | 38.000 | 39.455 | [35.337, 44.251] |
| levels_per_tx|15m | 1.130 | 1.169 | [1.162, 1.175] |
| buy_share|15m | 0.519 | 0.506 | [0.500, 0.510] |
| taker_size_med|4h | 20.000 | 16.893 | [15.654, 19.646] |
| taker_size_p90|4h | 100.000 | 100.000 | [98.882, 100.000] |
| levels_per_tx|4h | 1.136 | 1.329 | [1.317, 1.359] |
| buy_share|4h | 0.541 | 0.517 | [0.501, 0.527] |
| mid|5m|slope_1 | 0.210 | 0.240 | [0.231, 0.257] |
| mid|5m|slope_5 | 0.505 | 0.495 | [0.475, 0.511] |
| mid|5m|slope_30 | 0.747 | 0.772 | [0.754, 0.791] |
| mid|5m|slope_120 | 0.838 | 0.815 | [0.772, 0.838] |
| mid|5m|level_slope | 0.839 | 0.828 | [0.806, 0.849] |
| mid|5m|sd_gap | 0.562 | 0.671 | [0.636, 0.720] |
| mid|5m|acf_5 | 0.759 | 0.816 | [0.795, 0.838] |
| mid|5m|acf_30 | 0.469 | 0.493 | [0.442, 0.562] |
| mid|5m|acf_120 | 0.189 | 0.030 | [-0.027, 0.135] |
| mid|5m|p_change | 0.263 | 0.247 | [0.245, 0.251] |
| mid|15m|slope_1 | 0.173 | 0.181 | [0.170, 0.193] |
| mid|15m|slope_5 | 0.446 | 0.411 | [0.395, 0.424] |
| mid|15m|slope_30 | 0.746 | 0.758 | [0.744, 0.769] |
| mid|15m|slope_120 | 0.843 | 0.849 | [0.835, 0.865] |
| mid|15m|slope_300 | 0.864 | 0.867 | [0.853, 0.878] |
| mid|15m|level_slope | 0.871 | 0.858 | [0.844, 0.869] |
| mid|15m|sd_gap | 0.388 | 0.414 | [0.393, 0.439] |
| mid|15m|acf_5 | 0.814 | 0.824 | [0.807, 0.842] |
| mid|15m|acf_30 | 0.634 | 0.652 | [0.611, 0.681] |
| mid|15m|acf_120 | 0.434 | 0.368 | [0.317, 0.414] |
| mid|15m|p_change | 0.199 | 0.181 | [0.179, 0.182] |
| mid|4h|slope_1 | 0.125 | 0.106 | [0.100, 0.114] |
| mid|4h|slope_5 | 0.362 | 0.267 | [0.252, 0.276] |
| mid|4h|slope_30 | 0.595 | 0.574 | [0.565, 0.581] |
| mid|4h|slope_120 | 0.683 | 0.681 | [0.667, 0.690] |
| mid|4h|slope_300 | 0.721 | 0.709 | [0.691, 0.727] |
| mid|4h|slope_900 | 0.754 | 0.755 | [0.737, 0.779] |
| mid|4h|level_slope | 0.910 | 0.910 | [0.885, 0.931] |
| mid|4h|sd_gap | 0.436 | 0.454 | [0.421, 0.468] |
| mid|4h|acf_5 | 0.986 | 0.978 | [0.976, 0.981] |
| mid|4h|acf_30 | 0.954 | 0.942 | [0.933, 0.947] |
| mid|4h|acf_120 | 0.873 | 0.859 | [0.840, 0.871] |
| mid|4h|p_change | 0.083 | 0.137 | [0.136, 0.138] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -6294.1 [-7558.0, -5196.7] | 0.259 |
| fv | -2585.2 | 2399.2 [1005.2, 3512.7] | 0.308 |

## Diagnosis and use (hand-written for this run; rerunning validate_env.py regenerates only the sections above)

**Verdict.** The gate fails. Per SIMULATOR.md §5, use the sim only to initialise pretraining, followed
by replay fine-tuning, and quote no sim result.

**What matches, to within sampling noise:**
* The book.
  - Spread distributions: KS D <= 0.04.
  - Top-of-book and top-5 depth: KS D <= 0.05.
  - One-sided fraction by time to close: 19/19 cells, e.g. 5m last 10 s 0.91 vs 0.91.
  - Hollow books concentrated in ZEC.
  - Book-update (book_age = 0) rates.
* Trade counts per market: KS D 0.12-0.13.
* Taker sizes and sweep depth on 5m/15m.
* Sign autocorrelation on 15m/4h.
* The sticky-mid dynamics.
  - Slopes of Δmid on Δp_model at 1-900 s.
  - Mid change probability on 5m/15m.
* The post-trade markout drift on 5m.
* sym_touch total P&L: real -2958 inside the sim band [-3004, -2133].

**What fails:**
1. **The fair-value quoter is profitable in the sim and not in the replay.**
   - Total P&L: +1289 [783, 1766] sim vs -797 real.
   - So the rank order flips (G2).
   - Its fills are adversely selected far less in the sim. Markout per share at 5/30/60 s:
     sim -1.6/-2.3/-1.4 c, real -3.0/-4.5/-4.7 c.
   - Real takers carry information beyond the Binance-based p_model, or reach it faster than the
     1 s kline. In the sim, informed takers only know the mid's staleness relative to a target
     built from p_model, so quoting around p_model is too safe.
   - Fix to try next:
     - give informed takers a short look-ahead signal (the Binance return over the next 1-5 s,
       i.e. sub-second information the 1 s kline misses);
     - calibrate its strength to the fair_value baseline's markout curve on train.
2. **Too much fill volume and spread capture for touch quotes** (G3).
   - sym_touch fill volume is 1.6x real. Spread capture is 2.2 vs 0.7 $/market.
   - Prints line up with the captured book more often than in the data: 69-70% at-or-through at
     k = 0, vs 50-56% real. Fewer prints land inside the captured spread or on the wrong side.
   - The per-market P&L distribution is too narrow: real median -3.01 is outside [-1.91, -1.01],
     and the hourly KS is 0.21.
3. **G4 misses.**
   - 3 of 19 trade-rate cells: 15m/4h near close.
   - 3 of 29 P(buy | gap) cells.
   - 1 spread-median cell: 5m tau 10-30 s, 1 tick tighter than real.
   - 5m sign ACF: 0.31 vs 0.23.
4. **Not gated, but off.**
   - 4h prints sweep too deep: 1.33 vs 1.14 levels per tx.
   - 4h mid changes too often: 0.14 vs 0.08 per second.
   - The level of the all-print maker markout stays positive at 60 s on 15m/4h (+0.6 to +0.9 c,
     real about 0).

**Caveats.**
* Only 20 replications (SIMULATOR.md asks for 200), so the bands are rough.
* Every generator parameter is fitted on these same train days, so passing G1-G4 here would be an
  in-sample statement. The out-of-sample check (G5, val day) was not run.
