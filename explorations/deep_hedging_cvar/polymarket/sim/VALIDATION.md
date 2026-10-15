# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 9 min. params.json sha256 `b39c3fc2d0296ed9`, mm.h sha256 `746dd7dea250eee0`.

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
| sym_touch | -2957.6 | -3651.5 [-3930.9, -3347.6] | -3.010 | [-2.890, -1.734] | 0.215 | 0.069 | FAIL |
| fair_value | -796.6 | 249.8 [-53.7, 689.3] | 0.000 | [0.000, 0.000] | 0.207 | 0.107 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -2383.1 [-2630.8, -2153.0] |
| sym_touch | 15m | -952.6 | -1239.8 [-1467.5, -1021.8] |
| sym_touch | 4h | -93.5 | -42.7 [-113.5, 3.8] |
| fair_value | 5m | -553.0 | 84.3 [-211.7, 432.7] |
| fair_value | 15m | -336.2 | 67.4 [-195.4, 306.0] |
| fair_value | 4h | 92.7 | 106.2 [52.0, 179.5] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 106.175 [102.800, 112.803] | False |
| sym_touch | fill_vol | 164.380 | 225.105 [217.172, 245.500] | False |
| sym_touch | spread_capture | 0.675 | 1.773 [1.666, 2.031] | False |
| sym_touch | markout_5 | -4.763 | -5.726 [-6.171, -5.436] | False |
| sym_touch | markout_30 | -7.265 | -10.933 [-11.947, -10.453] | False |
| sym_touch | markout_60 | -7.982 | -11.673 [-12.252, -10.672] | False |
| sym_touch | inventory_pnl | 0.037 | 0.853 [0.084, 1.597] | False |
| sym_touch | rebates | 0.407 | 0.471 [0.457, 0.524] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -2.520 [-2.635, -2.354] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -4.814 [-5.133, -4.603] | False |
| sym_touch | markout 60s (c/share) | -4.856 | -5.068 [-5.481, -4.738] | True |
| fair_value | fills | 49.930 | 46.787 [45.133, 50.284] | True |
| fair_value | fill_vol | 85.714 | 100.272 [96.370, 111.657] | False |
| fair_value | spread_capture | 1.354 | 1.801 [1.712, 2.015] | False |
| fair_value | markout_5 | -2.611 | -2.217 [-2.371, -2.048] | False |
| fair_value | markout_30 | -3.815 | -3.685 [-4.189, -3.268] | True |
| fair_value | markout_60 | -4.047 | -3.279 [-3.725, -2.480] | False |
| fair_value | inventory_pnl | 0.629 | 1.895 [0.763, 2.549] | False |
| fair_value | rebates | 0.216 | 0.210 [0.200, 0.233] | True |
| fair_value | markout 5s (c/share) | -3.046 | -2.177 [-2.277, -2.034] | False |
| fair_value | markout 30s (c/share) | -4.451 | -3.635 [-4.193, -3.191] | False |
| fair_value | markout 60s (c/share) | -4.721 | -3.253 [-3.646, -2.298] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=1, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | PASS (n_fail=0, n=19) |
| ks_spread|5m | PASS (D=0.044) |
| ks_spread|15m | PASS (D=0.017) |
| ks_spread|4h | PASS (D=0.040) |
| ks_log_top_sz|5m | PASS (D=0.025) |
| ks_log_top_sz|15m | PASS (D=0.009) |
| ks_log_top_sz|4h | PASS (D=0.007) |
| ks_log_top5|5m | PASS (D=0.039) |
| ks_log_top5|15m | PASS (D=0.032) |
| ks_log_top5|4h | PASS (D=0.049) |
| ks_trades_per_market|5m | FAIL (D=0.157) |
| ks_trades_per_market|15m | PASS (D=0.108) |
| ks_trades_per_market|4h | PASS (D=0.132) |
| pbuy_gap_curve | FAIL (n_fail=2, n=29) |
| sign_acf1|5m | PASS (real=0.232, sim=0.234) |
| sign_acf1|15m | PASS (real=0.298, sim=0.273) |
| sign_acf1|4h | PASS (real=0.262, sim=0.256) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.78 | 50/50 | 250/128 | 0.32/0.26 | 0.388/0.352 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.62 | 50/50 | 255/208 | 0.29/0.24 | 0.597/0.674 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.020 | 0.47/0.49 | 50/50 | 260/250 | 0.29/0.24 | 1.424/1.390 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.45 | 50/50 | 300/289 | 0.32/0.27 | 1.434/1.394 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.45 | 50/50 | 322/310 | 0.32/0.26 | 1.652/1.543 |
| 15m | 0 | 0.899/0.894 | 0.067/0.007 | 0.010/0.009 | 0.61/0.82 | 50/50 | 185/188 | 0.30/0.22 | 0.028/0.035 |
| 15m | 1 | 0.834/0.831 | 0.085/0.020 | 0.010/0.009 | 0.58/0.75 | 50/50 | 250/216 | 0.25/0.19 | 0.071/0.087 |
| 15m | 2 | 0.700/0.698 | 0.085/0.038 | 0.020/0.010 | 0.46/0.53 | 50/50 | 250/255 | 0.23/0.18 | 0.221/0.227 |
| 15m | 3 | 0.450/0.442 | 0.056/0.054 | 0.019/0.020 | 0.48/0.47 | 50/50 | 317/310 | 0.26/0.19 | 0.311/0.329 |
| 15m | 4 | 0.175/0.174 | 0.067/0.078 | 0.020/0.020 | 0.43/0.45 | 50/50 | 427/388 | 0.29/0.21 | 0.381/0.375 |
| 15m | 5 | 0.006/0.007 | 0.092/0.098 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/387 | 0.29/0.21 | 0.359/0.362 |
| 4h | 0 | 0.700/0.662 | 0.081/0.027 | 0.008/0.009 | 0.75/0.80 | 50/64 | 265/245 | 0.07/0.07 | 0.065/0.054 |
| 4h | 1 | 0.672/0.632 | 0.081/0.027 | 0.009/0.009 | 0.77/0.76 | 55/70 | 315/247 | 0.11/0.11 | 0.030/0.025 |
| 4h | 2 | 0.532/0.504 | 0.081/0.047 | 0.009/0.009 | 0.66/0.72 | 100/68 | 315/250 | 0.07/0.07 | 0.048/0.041 |
| 4h | 3 | 0.447/0.433 | 0.084/0.045 | 0.009/0.009 | 0.61/0.63 | 103/58 | 294/250 | 0.09/0.10 | 0.034/0.031 |
| 4h | 4 | 0.203/0.186 | 0.095/0.070 | 0.009/0.009 | 0.63/0.64 | 55/50 | 316/250 | 0.06/0.06 | 0.023/0.022 |
| 4h | 5 | 0.093/0.080 | 0.084/0.081 | 0.009/0.009 | 0.58/0.60 | 53/53 | 320/253 | 0.09/0.09 | 0.022/0.020 |
| 4h | 6 | 0.047/0.046 | 0.081/0.081 | 0.011/0.011 | 0.49/0.50 | 50/52 | 338/300 | 0.10/0.10 | 0.015/0.013 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.010 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.413 |
| 5m|1 | 12459 | 0.474 | 0.481 |
| 5m|2 | 12977 | 0.458 | 0.478 |
| 5m|3 | 6481 | 0.457 | 0.495 |
| 5m|4 | 7216 | 0.513 | 0.583 |
| 5m|5 | 8577 | 0.504 | 0.437 |
| 5m|6 | 6910 | 0.538 | 0.515 |
| 5m|7 | 13334 | 0.536 | 0.521 |
| 5m|8 | 11837 | 0.541 | 0.534 |
| 5m|9 | 13657 | 0.568 | 0.566 |
| 15m|0 | 2616 | 0.426 | 0.431 |
| 15m|1 | 4776 | 0.460 | 0.461 |
| 15m|2 | 7510 | 0.489 | 0.480 |
| 15m|3 | 3528 | 0.460 | 0.476 |
| 15m|4 | 3913 | 0.496 | 0.516 |
| 15m|5 | 4166 | 0.528 | 0.483 |
| 15m|6 | 3528 | 0.556 | 0.539 |
| 15m|7 | 7364 | 0.565 | 0.546 |
| 15m|8 | 5177 | 0.583 | 0.576 |
| 15m|9 | 2607 | 0.600 | 0.592 |
| 4h|1 | 455 | 0.433 | 0.439 |
| 4h|2 | 545 | 0.424 | 0.433 |
| 4h|3 | 363 | 0.424 | 0.444 |
| 4h|4 | 532 | 0.451 | 0.449 |
| 4h|5 | 745 | 0.541 | 0.538 |
| 4h|6 | 573 | 0.635 | 0.616 |
| 4h|7 | 1301 | 0.587 | 0.580 |
| 4h|8 | 1147 | 0.616 | 0.606 |
| 4h|9 | 655 | 0.594 | 0.588 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.577 | [0.566, 0.588] |
| align_through|5m|k0 | 0.446 | 0.449 | [0.431, 0.464] |
| align_at_or_through|15m|k0 | 0.561 | 0.604 | [0.596, 0.621] |
| align_through|15m|k0 | 0.407 | 0.414 | [0.402, 0.444] |
| align_at_or_through|4h|k0 | 0.498 | 0.676 | [0.661, 0.696] |
| align_through|4h|k0 | 0.206 | 0.422 | [0.398, 0.438] |
| align_at_or_through|5m|k1 | 0.561 | 0.608 | [0.596, 0.618] |
| align_through|5m|k1 | 0.453 | 0.459 | [0.439, 0.473] |
| align_at_or_through|15m|k1 | 0.587 | 0.632 | [0.623, 0.645] |
| align_through|15m|k1 | 0.424 | 0.426 | [0.410, 0.452] |
| align_at_or_through|4h|k1 | 0.537 | 0.689 | [0.676, 0.706] |
| align_through|4h|k1 | 0.225 | 0.417 | [0.398, 0.434] |
| align_at_or_through|5m|k2 | 0.590 | 0.639 | [0.627, 0.650] |
| align_through|5m|k2 | 0.469 | 0.466 | [0.446, 0.482] |
| align_at_or_through|15m|k2 | 0.621 | 0.658 | [0.649, 0.667] |
| align_through|15m|k2 | 0.453 | 0.435 | [0.420, 0.457] |
| align_at_or_through|4h|k2 | 0.577 | 0.707 | [0.694, 0.722] |
| align_through|4h|k2 | 0.247 | 0.414 | [0.393, 0.431] |
| align_at_or_through|5m|k3 | 0.626 | 0.665 | [0.655, 0.675] |
| align_through|5m|k3 | 0.511 | 0.469 | [0.448, 0.486] |
| align_at_or_through|15m|k3 | 0.665 | 0.677 | [0.670, 0.687] |
| align_through|15m|k3 | 0.507 | 0.442 | [0.427, 0.462] |
| align_at_or_through|4h|k3 | 0.628 | 0.721 | [0.709, 0.735] |
| align_through|4h|k3 | 0.287 | 0.408 | [0.390, 0.429] |
| align_at_or_through|5m|k4 | 0.643 | 0.643 | [0.633, 0.652] |
| align_through|5m|k4 | 0.558 | 0.492 | [0.475, 0.508] |
| align_at_or_through|15m|k4 | 0.682 | 0.659 | [0.652, 0.669] |
| align_through|15m|k4 | 0.557 | 0.465 | [0.453, 0.485] |
| align_at_or_through|4h|k4 | 0.641 | 0.713 | [0.699, 0.727] |
| align_through|4h|k4 | 0.317 | 0.418 | [0.399, 0.438] |
| align_at_or_through|5m|k6 | 0.624 | 0.584 | [0.574, 0.592] |
| align_through|5m|k6 | 0.572 | 0.518 | [0.506, 0.528] |
| align_at_or_through|15m|k6 | 0.659 | 0.605 | [0.599, 0.615] |
| align_through|15m|k6 | 0.567 | 0.500 | [0.486, 0.516] |
| align_at_or_through|4h|k6 | 0.630 | 0.666 | [0.656, 0.682] |
| align_through|4h|k6 | 0.344 | 0.446 | [0.426, 0.467] |
| sign_acf|5m|1 | 0.232 | 0.234 | [0.220, 0.249] |
| sign_acf|5m|2 | 0.196 | 0.217 | [0.204, 0.231] |
| sign_acf|5m|5 | 0.166 | 0.186 | [0.171, 0.199] |
| sign_acf|15m|1 | 0.298 | 0.273 | [0.258, 0.285] |
| sign_acf|15m|2 | 0.206 | 0.200 | [0.174, 0.218] |
| sign_acf|15m|5 | 0.123 | 0.126 | [0.088, 0.154] |
| sign_acf|4h|1 | 0.262 | 0.256 | [0.229, 0.274] |
| sign_acf|4h|2 | 0.222 | 0.076 | [0.043, 0.101] |
| sign_acf|4h|5 | 0.138 | 0.012 | [-0.014, 0.023] |
| markout|5m|0 | 2.975 | 1.978 | [1.762, 2.323] |
| markout|15m|0 | 1.983 | 1.312 | [1.177, 1.668] |
| markout|4h|0 | 1.127 | 1.534 | [1.409, 1.652] |
| markout|5m|5 | 1.611 | 0.273 | [0.012, 0.560] |
| markout|15m|5 | 0.634 | 0.357 | [0.281, 0.595] |
| markout|4h|5 | 0.476 | 1.427 | [1.332, 1.563] |
| markout|5m|30 | 0.527 | -1.988 | [-2.903, -1.435] |
| markout|15m|30 | 0.004 | -1.048 | [-1.491, -0.760] |
| markout|4h|30 | 0.017 | 0.950 | [0.866, 1.144] |
| markout|5m|60 | 0.111 | -2.032 | [-2.799, -1.638] |
| markout|15m|60 | 0.117 | -1.099 | [-1.730, -0.832] |
| markout|4h|60 | -0.087 | 0.797 | [0.662, 1.057] |
| taker_size_med|5m | 5.880 | 5.429 | [5.143, 5.962] |
| taker_size_p90|5m | 40.159 | 39.157 | [36.426, 40.706] |
| levels_per_tx|5m | 1.179 | 1.165 | [1.159, 1.173] |
| buy_share|5m | 0.498 | 0.503 | [0.497, 0.509] |
| taker_size_med|15m | 5.000 | 5.050 | [5.000, 5.430] |
| taker_size_p90|15m | 38.000 | 38.860 | [35.263, 42.297] |
| levels_per_tx|15m | 1.130 | 1.174 | [1.165, 1.178] |
| buy_share|15m | 0.519 | 0.507 | [0.503, 0.511] |
| taker_size_med|4h | 20.000 | 18.008 | [15.421, 19.688] |
| taker_size_p90|4h | 100.000 | 100.000 | [93.023, 100.000] |
| levels_per_tx|4h | 1.136 | 1.329 | [1.309, 1.344] |
| buy_share|4h | 0.541 | 0.523 | [0.513, 0.544] |
| mid|5m|slope_1 | 0.210 | 0.224 | [0.211, 0.231] |
| mid|5m|slope_5 | 0.505 | 0.476 | [0.465, 0.494] |
| mid|5m|slope_30 | 0.747 | 0.792 | [0.781, 0.824] |
| mid|5m|slope_120 | 0.838 | 0.864 | [0.842, 0.884] |
| mid|5m|level_slope | 0.839 | 0.887 | [0.868, 0.903] |
| mid|5m|sd_gap | 0.562 | 0.561 | [0.543, 0.583] |
| mid|5m|acf_5 | 0.759 | 0.744 | [0.724, 0.760] |
| mid|5m|acf_30 | 0.469 | 0.442 | [0.406, 0.479] |
| mid|5m|acf_120 | 0.189 | 0.094 | [0.050, 0.136] |
| mid|5m|p_change | 0.263 | 0.232 | [0.229, 0.235] |
| mid|5m|gapdp_30 | -0.037 | -0.074 | [-0.122, 0.006] |
| mid|5m|gapdy_30 | 0.391 | 0.460 | [0.409, 0.493] |
| mid|5m|gapdp_60 | -0.110 | -0.123 | [-0.242, 0.004] |
| mid|5m|gapdy_60 | 0.420 | 0.586 | [0.482, 0.711] |
| mid|15m|slope_1 | 0.173 | 0.178 | [0.166, 0.196] |
| mid|15m|slope_5 | 0.446 | 0.407 | [0.394, 0.425] |
| mid|15m|slope_30 | 0.746 | 0.762 | [0.753, 0.774] |
| mid|15m|slope_120 | 0.843 | 0.871 | [0.867, 0.882] |
| mid|15m|slope_300 | 0.864 | 0.889 | [0.879, 0.902] |
| mid|15m|level_slope | 0.871 | 0.888 | [0.882, 0.894] |
| mid|15m|sd_gap | 0.388 | 0.393 | [0.382, 0.406] |
| mid|15m|acf_5 | 0.814 | 0.807 | [0.795, 0.816] |
| mid|15m|acf_30 | 0.634 | 0.629 | [0.608, 0.647] |
| mid|15m|acf_120 | 0.434 | 0.396 | [0.355, 0.418] |
| mid|15m|p_change | 0.199 | 0.178 | [0.175, 0.179] |
| mid|15m|gapdp_30 | -0.027 | -0.016 | [-0.040, 0.015] |
| mid|15m|gapdy_30 | 0.292 | 0.334 | [0.302, 0.370] |
| mid|15m|gapdp_60 | -0.034 | -0.021 | [-0.075, 0.050] |
| mid|15m|gapdy_60 | 0.333 | 0.421 | [0.359, 0.483] |
| mid|4h|slope_1 | 0.125 | 0.103 | [0.094, 0.110] |
| mid|4h|slope_5 | 0.362 | 0.264 | [0.245, 0.273] |
| mid|4h|slope_30 | 0.595 | 0.559 | [0.549, 0.571] |
| mid|4h|slope_120 | 0.683 | 0.671 | [0.660, 0.684] |
| mid|4h|slope_300 | 0.721 | 0.707 | [0.691, 0.732] |
| mid|4h|slope_900 | 0.754 | 0.786 | [0.764, 0.821] |
| mid|4h|level_slope | 0.910 | 0.892 | [0.874, 0.926] |
| mid|4h|sd_gap | 0.436 | 0.456 | [0.431, 0.465] |
| mid|4h|acf_5 | 0.986 | 0.980 | [0.976, 0.983] |
| mid|4h|acf_30 | 0.954 | 0.948 | [0.936, 0.953] |
| mid|4h|acf_120 | 0.873 | 0.879 | [0.857, 0.886] |
| mid|4h|p_change | 0.083 | 0.136 | [0.135, 0.136] |
| mid|4h|gapdp_30 | -0.002 | -0.011 | [-0.017, -0.009] |
| mid|4h|gapdy_30 | 0.033 | 0.036 | [0.031, 0.042] |
| mid|4h|gapdp_60 | -0.009 | -0.023 | [-0.034, -0.018] |
| mid|4h|gapdy_60 | 0.046 | 0.045 | [0.037, 0.054] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -10067.4 [-10902.0, -8710.6] | 0.100 |
| fv | -2585.2 | -63.0 [-1789.2, 1059.2] | 0.129 |
