# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 9 min. params.json sha256 `90996ad99c6d9073`, mm.h sha256 `746dd7dea250eee0`.

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
| sym_touch | -2957.6 | -3564.5 [-3838.4, -3266.6] | -3.010 | [-3.255, -2.051] | 0.176 | 0.065 | FAIL |
| fair_value | -796.6 | 424.5 [44.8, 828.9] | 0.000 | [0.000, 0.007] | 0.248 | 0.133 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -2408.1 [-2604.0, -2254.3] |
| sym_touch | 15m | -952.6 | -1133.0 [-1394.5, -905.0] |
| sym_touch | 4h | -93.5 | -42.8 [-119.4, 42.6] |
| fair_value | 5m | -553.0 | 192.3 [-11.8, 529.6] |
| fair_value | 15m | -336.2 | 65.7 [-150.5, 260.7] |
| fair_value | 4h | 92.7 | 146.7 [83.2, 266.0] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 106.647 [97.569, 119.813] | False |
| sym_touch | fill_vol | 164.380 | 225.433 [206.798, 259.156] | False |
| sym_touch | spread_capture | 0.675 | 1.772 [1.582, 2.235] | False |
| sym_touch | markout_5 | -4.763 | -5.962 [-6.341, -5.519] | False |
| sym_touch | markout_30 | -7.265 | -11.015 [-11.785, -10.132] | False |
| sym_touch | markout_60 | -7.982 | -11.169 [-12.199, -10.516] | False |
| sym_touch | inventory_pnl | 0.037 | 0.714 [0.123, 1.409] | False |
| sym_touch | rebates | 0.407 | 0.483 [0.444, 0.545] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -2.625 [-2.795, -2.424] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -4.777 [-5.070, -4.482] | False |
| sym_touch | markout 60s (c/share) | -4.856 | -4.980 [-5.232, -4.615] | True |
| fair_value | fills | 49.930 | 47.726 [42.779, 52.180] | True |
| fair_value | fill_vol | 85.714 | 101.414 [91.906, 114.272] | False |
| fair_value | spread_capture | 1.354 | 1.846 [1.642, 2.133] | False |
| fair_value | markout_5 | -2.611 | -2.330 [-2.429, -2.091] | False |
| fair_value | markout_30 | -3.815 | -3.510 [-4.059, -3.244] | True |
| fair_value | markout_60 | -4.047 | -2.983 [-3.608, -2.278] | False |
| fair_value | inventory_pnl | 0.629 | 1.895 [1.027, 2.688] | False |
| fair_value | rebates | 0.216 | 0.217 [0.198, 0.242] | True |
| fair_value | markout 5s (c/share) | -3.046 | -2.248 [-2.367, -2.090] | False |
| fair_value | markout 30s (c/share) | -4.451 | -3.524 [-3.756, -3.177] | False |
| fair_value | markout 60s (c/share) | -4.721 | -2.942 [-3.325, -2.233] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=1, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | FAIL (n_fail=3, n=19) |
| ks_spread|5m | PASS (D=0.043) |
| ks_spread|15m | PASS (D=0.013) |
| ks_spread|4h | PASS (D=0.039) |
| ks_log_top_sz|5m | PASS (D=0.025) |
| ks_log_top_sz|15m | PASS (D=0.009) |
| ks_log_top_sz|4h | PASS (D=0.005) |
| ks_log_top5|5m | PASS (D=0.034) |
| ks_log_top5|15m | PASS (D=0.036) |
| ks_log_top5|4h | PASS (D=0.049) |
| ks_trades_per_market|5m | PASS (D=0.145) |
| ks_trades_per_market|15m | PASS (D=0.134) |
| ks_trades_per_market|4h | PASS (D=0.123) |
| pbuy_gap_curve | FAIL (n_fail=2, n=29) |
| sign_acf1|5m | PASS (real=0.232, sim=0.264) |
| sign_acf1|15m | PASS (real=0.298, sim=0.297) |
| sign_acf1|4h | PASS (real=0.262, sim=0.253) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.80 | 50/50 | 250/164 | 0.32/0.27 | 0.388/0.381 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.62 | 50/50 | 255/209 | 0.29/0.26 | 0.597/0.588 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.020 | 0.47/0.48 | 50/50 | 260/250 | 0.29/0.26 | 1.424/1.110 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.45 | 50/50 | 300/282 | 0.32/0.28 | 1.434/1.462 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.44 | 50/50 | 322/320 | 0.32/0.28 | 1.652/1.669 |
| 15m | 0 | 0.899/0.881 | 0.067/0.007 | 0.010/0.009 | 0.61/0.83 | 50/50 | 185/113 | 0.30/0.22 | 0.028/0.062 |
| 15m | 1 | 0.834/0.833 | 0.085/0.019 | 0.010/0.009 | 0.58/0.76 | 50/50 | 250/145 | 0.25/0.19 | 0.071/0.110 |
| 15m | 2 | 0.700/0.703 | 0.085/0.040 | 0.020/0.010 | 0.46/0.53 | 50/50 | 250/239 | 0.23/0.18 | 0.221/0.190 |
| 15m | 3 | 0.450/0.449 | 0.056/0.055 | 0.019/0.020 | 0.48/0.45 | 50/50 | 317/299 | 0.26/0.19 | 0.311/0.265 |
| 15m | 4 | 0.175/0.187 | 0.067/0.080 | 0.020/0.020 | 0.43/0.44 | 50/50 | 427/379 | 0.29/0.21 | 0.381/0.342 |
| 15m | 5 | 0.006/0.008 | 0.092/0.097 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/384 | 0.29/0.21 | 0.359/0.328 |
| 4h | 0 | 0.700/0.670 | 0.081/0.027 | 0.008/0.009 | 0.75/0.75 | 50/96 | 265/250 | 0.07/0.07 | 0.065/0.043 |
| 4h | 1 | 0.672/0.632 | 0.081/0.030 | 0.009/0.009 | 0.77/0.75 | 55/81 | 315/250 | 0.11/0.11 | 0.030/0.031 |
| 4h | 2 | 0.532/0.518 | 0.081/0.054 | 0.009/0.009 | 0.66/0.68 | 100/64 | 315/250 | 0.07/0.07 | 0.048/0.042 |
| 4h | 3 | 0.447/0.411 | 0.084/0.054 | 0.009/0.009 | 0.61/0.67 | 103/53 | 294/250 | 0.09/0.10 | 0.034/0.028 |
| 4h | 4 | 0.203/0.197 | 0.095/0.075 | 0.009/0.009 | 0.63/0.64 | 55/55 | 316/250 | 0.06/0.06 | 0.023/0.024 |
| 4h | 5 | 0.093/0.067 | 0.084/0.081 | 0.009/0.009 | 0.58/0.60 | 53/53 | 320/259 | 0.09/0.09 | 0.022/0.024 |
| 4h | 6 | 0.047/0.041 | 0.081/0.081 | 0.011/0.010 | 0.49/0.50 | 50/52 | 338/297 | 0.10/0.10 | 0.015/0.017 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.013 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.412 |
| 5m|1 | 12459 | 0.474 | 0.479 |
| 5m|2 | 12977 | 0.458 | 0.475 |
| 5m|3 | 6481 | 0.457 | 0.499 |
| 5m|4 | 7216 | 0.513 | 0.580 |
| 5m|5 | 8577 | 0.504 | 0.433 |
| 5m|6 | 6910 | 0.538 | 0.513 |
| 5m|7 | 13334 | 0.536 | 0.526 |
| 5m|8 | 11837 | 0.541 | 0.535 |
| 5m|9 | 13657 | 0.568 | 0.574 |
| 15m|0 | 2616 | 0.426 | 0.432 |
| 15m|1 | 4776 | 0.460 | 0.461 |
| 15m|2 | 7510 | 0.489 | 0.484 |
| 15m|3 | 3528 | 0.460 | 0.472 |
| 15m|4 | 3913 | 0.496 | 0.519 |
| 15m|5 | 4166 | 0.528 | 0.479 |
| 15m|6 | 3528 | 0.556 | 0.536 |
| 15m|7 | 7364 | 0.565 | 0.552 |
| 15m|8 | 5177 | 0.583 | 0.575 |
| 15m|9 | 2607 | 0.600 | 0.609 |
| 4h|1 | 455 | 0.433 | 0.434 |
| 4h|2 | 545 | 0.424 | 0.430 |
| 4h|3 | 363 | 0.424 | 0.435 |
| 4h|4 | 532 | 0.451 | 0.456 |
| 4h|5 | 745 | 0.541 | 0.539 |
| 4h|6 | 573 | 0.635 | 0.607 |
| 4h|7 | 1301 | 0.587 | 0.585 |
| 4h|8 | 1147 | 0.616 | 0.609 |
| 4h|9 | 655 | 0.594 | 0.579 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.566 | [0.554, 0.581] |
| align_through|5m|k0 | 0.446 | 0.439 | [0.422, 0.458] |
| align_at_or_through|15m|k0 | 0.561 | 0.595 | [0.584, 0.618] |
| align_through|15m|k0 | 0.407 | 0.408 | [0.396, 0.444] |
| align_at_or_through|4h|k0 | 0.498 | 0.679 | [0.669, 0.691] |
| align_through|4h|k0 | 0.206 | 0.424 | [0.405, 0.442] |
| align_at_or_through|5m|k1 | 0.561 | 0.599 | [0.589, 0.614] |
| align_through|5m|k1 | 0.453 | 0.452 | [0.435, 0.468] |
| align_at_or_through|15m|k1 | 0.587 | 0.626 | [0.616, 0.648] |
| align_through|15m|k1 | 0.424 | 0.421 | [0.409, 0.457] |
| align_at_or_through|4h|k1 | 0.537 | 0.691 | [0.682, 0.705] |
| align_through|4h|k1 | 0.225 | 0.419 | [0.402, 0.437] |
| align_at_or_through|5m|k2 | 0.590 | 0.634 | [0.625, 0.647] |
| align_through|5m|k2 | 0.469 | 0.461 | [0.444, 0.476] |
| align_at_or_through|15m|k2 | 0.621 | 0.655 | [0.645, 0.676] |
| align_through|15m|k2 | 0.453 | 0.434 | [0.419, 0.467] |
| align_at_or_through|4h|k2 | 0.577 | 0.705 | [0.694, 0.720] |
| align_through|4h|k2 | 0.247 | 0.415 | [0.398, 0.438] |
| align_at_or_through|5m|k3 | 0.626 | 0.667 | [0.659, 0.678] |
| align_through|5m|k3 | 0.511 | 0.465 | [0.450, 0.480] |
| align_at_or_through|15m|k3 | 0.665 | 0.678 | [0.666, 0.691] |
| align_through|15m|k3 | 0.507 | 0.442 | [0.427, 0.473] |
| align_at_or_through|4h|k3 | 0.628 | 0.720 | [0.708, 0.734] |
| align_through|4h|k3 | 0.287 | 0.410 | [0.391, 0.435] |
| align_at_or_through|5m|k4 | 0.643 | 0.645 | [0.638, 0.656] |
| align_through|5m|k4 | 0.558 | 0.492 | [0.476, 0.505] |
| align_at_or_through|15m|k4 | 0.682 | 0.660 | [0.647, 0.669] |
| align_through|15m|k4 | 0.557 | 0.467 | [0.451, 0.493] |
| align_at_or_through|4h|k4 | 0.641 | 0.713 | [0.699, 0.727] |
| align_through|4h|k4 | 0.317 | 0.417 | [0.397, 0.447] |
| align_at_or_through|5m|k6 | 0.624 | 0.584 | [0.577, 0.594] |
| align_through|5m|k6 | 0.572 | 0.521 | [0.508, 0.532] |
| align_at_or_through|15m|k6 | 0.659 | 0.606 | [0.595, 0.615] |
| align_through|15m|k6 | 0.567 | 0.502 | [0.490, 0.516] |
| align_at_or_through|4h|k6 | 0.630 | 0.669 | [0.656, 0.678] |
| align_through|4h|k6 | 0.344 | 0.443 | [0.424, 0.478] |
| sign_acf|5m|1 | 0.232 | 0.264 | [0.246, 0.279] |
| sign_acf|5m|2 | 0.196 | 0.243 | [0.228, 0.261] |
| sign_acf|5m|5 | 0.166 | 0.205 | [0.185, 0.224] |
| sign_acf|15m|1 | 0.298 | 0.297 | [0.272, 0.316] |
| sign_acf|15m|2 | 0.206 | 0.219 | [0.197, 0.241] |
| sign_acf|15m|5 | 0.123 | 0.141 | [0.101, 0.179] |
| sign_acf|4h|1 | 0.262 | 0.253 | [0.237, 0.273] |
| sign_acf|4h|2 | 0.222 | 0.078 | [0.052, 0.107] |
| sign_acf|4h|5 | 0.138 | 0.009 | [-0.003, 0.032] |
| markout|5m|0 | 2.975 | 1.943 | [1.713, 2.332] |
| markout|15m|0 | 1.983 | 1.326 | [1.240, 1.558] |
| markout|4h|0 | 1.127 | 1.511 | [1.294, 1.666] |
| markout|5m|5 | 1.611 | 0.010 | [-0.199, 0.478] |
| markout|15m|5 | 0.634 | 0.287 | [0.163, 0.510] |
| markout|4h|5 | 0.476 | 1.419 | [1.228, 1.575] |
| markout|5m|30 | 0.527 | -2.159 | [-3.178, -1.697] |
| markout|15m|30 | 0.004 | -1.033 | [-1.751, -0.633] |
| markout|4h|30 | 0.017 | 1.040 | [0.862, 1.202] |
| markout|5m|60 | 0.111 | -2.332 | [-2.928, -1.894] |
| markout|15m|60 | 0.117 | -1.181 | [-1.909, -0.848] |
| markout|4h|60 | -0.087 | 0.927 | [0.720, 1.124] |
| taker_size_med|5m | 5.880 | 5.425 | [5.140, 5.959] |
| taker_size_p90|5m | 40.159 | 39.338 | [36.039, 40.643] |
| levels_per_tx|5m | 1.179 | 1.163 | [1.158, 1.168] |
| buy_share|5m | 0.498 | 0.503 | [0.494, 0.507] |
| taker_size_med|15m | 5.000 | 5.075 | [5.000, 5.985] |
| taker_size_p90|15m | 38.000 | 39.701 | [35.341, 44.732] |
| levels_per_tx|15m | 1.130 | 1.172 | [1.168, 1.178] |
| buy_share|15m | 0.519 | 0.510 | [0.503, 0.514] |
| taker_size_med|4h | 20.000 | 17.171 | [16.045, 19.267] |
| taker_size_p90|4h | 100.000 | 100.000 | [89.909, 100.000] |
| levels_per_tx|4h | 1.136 | 1.329 | [1.313, 1.349] |
| buy_share|4h | 0.541 | 0.526 | [0.514, 0.543] |
| mid|5m|slope_1 | 0.210 | 0.242 | [0.232, 0.259] |
| mid|5m|slope_5 | 0.505 | 0.506 | [0.491, 0.521] |
| mid|5m|slope_30 | 0.747 | 0.806 | [0.797, 0.833] |
| mid|5m|slope_120 | 0.838 | 0.860 | [0.845, 0.882] |
| mid|5m|level_slope | 0.839 | 0.869 | [0.858, 0.887] |
| mid|5m|sd_gap | 0.562 | 0.564 | [0.546, 0.581] |
| mid|5m|acf_5 | 0.759 | 0.749 | [0.732, 0.756] |
| mid|5m|acf_30 | 0.469 | 0.443 | [0.401, 0.479] |
| mid|5m|acf_120 | 0.189 | 0.077 | [0.035, 0.098] |
| mid|5m|p_change | 0.263 | 0.247 | [0.244, 0.249] |
| mid|15m|slope_1 | 0.173 | 0.189 | [0.173, 0.203] |
| mid|15m|slope_5 | 0.446 | 0.420 | [0.409, 0.428] |
| mid|15m|slope_30 | 0.746 | 0.770 | [0.762, 0.781] |
| mid|15m|slope_120 | 0.843 | 0.864 | [0.850, 0.882] |
| mid|15m|slope_300 | 0.864 | 0.878 | [0.853, 0.892] |
| mid|15m|level_slope | 0.871 | 0.868 | [0.856, 0.881] |
| mid|15m|sd_gap | 0.388 | 0.388 | [0.377, 0.401] |
| mid|15m|acf_5 | 0.814 | 0.802 | [0.792, 0.813] |
| mid|15m|acf_30 | 0.634 | 0.629 | [0.612, 0.653] |
| mid|15m|acf_120 | 0.434 | 0.391 | [0.353, 0.420] |
| mid|15m|p_change | 0.199 | 0.180 | [0.178, 0.182] |
| mid|4h|slope_1 | 0.125 | 0.105 | [0.100, 0.135] |
| mid|4h|slope_5 | 0.362 | 0.272 | [0.259, 0.283] |
| mid|4h|slope_30 | 0.595 | 0.576 | [0.565, 0.589] |
| mid|4h|slope_120 | 0.683 | 0.686 | [0.669, 0.701] |
| mid|4h|slope_300 | 0.721 | 0.715 | [0.696, 0.729] |
| mid|4h|slope_900 | 0.754 | 0.757 | [0.722, 0.777] |
| mid|4h|level_slope | 0.910 | 0.903 | [0.877, 0.919] |
| mid|4h|sd_gap | 0.436 | 0.446 | [0.426, 0.457] |
| mid|4h|acf_5 | 0.986 | 0.979 | [0.977, 0.982] |
| mid|4h|acf_30 | 0.954 | 0.944 | [0.939, 0.950] |
| mid|4h|acf_120 | 0.873 | 0.870 | [0.858, 0.880] |
| mid|4h|p_change | 0.083 | 0.137 | [0.136, 0.137] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -9608.9 [-10973.5, -8688.5] | 0.091 |
| fv | -2585.2 | 426.6 [-1613.2, 1572.7] | 0.158 |

## Before / after the informed-taker fix (hand-written; rerunning validate_env.py regenerates only the sections above)

**Before** is the previous gate run: params from before the fix, saved as `cache/params_before.json` and
`cache/validation_env_before.json`. **After** is this run. Both use 20 replications, central fills and L = 1 s.

**Changes**, each calibrated on train only through the C env with `calibrate_env.py`, one train-schedule
replication per candidate (`cache/calib_env.jsonl`):

* **Look-ahead informed takers.** Each arrival is informed with probability 1 - exp(-kappa edge).
  - edge = the Binance-driven mid target 5 s ahead, minus the actual touch, minus the taker fee, in cents.
  - kappa = 0.5 / 3 / 3 per cent (5m/15m/4h). An informed taker takes the side with the edge.
  - The earlier staleness-informed share is now 0.
* **Touch resilience.** A level that a print exhausts is re-posted at the same price with probability 0.8.
* **Impact.** Halved to 0.01 / 0.01 / 0.02.
* **Copy probability.** Re-set to 0 / 0.10 / 0.25, because informed trades already come in same-sign runs.
* **Rates.** The mean-rate shift per length was re-estimated with 10 replications.

| item | before | after | real |
|---|---|---|---|
| Overall | FAIL | FAIL | |
| G1 sym_touch total | -2706 [-3003, -2133] (in) | -3564 [-3838, -3267] (out) | -2958 |
| G1 sym_touch median per market | [-1.91, -1.01] (out) | [-3.26, -2.05] (in) | -3.01 |
| G1 sym_touch KS hourly / per-market | 0.213 / 0.131 | 0.176 / 0.065 | < 0.15 |
| G1 fair_value total | +1289 [783, 1766] | +425 [45, 829] | -797 |
| G1 fair_value KS hourly / per-market | 0.397 / 0.201 | 0.248 / 0.133 | < 0.15 |
| G2 sim rank order | fair_value > no_quote > sym_touch | same (FAIL) | no_quote > fair_value > sym_touch |
| G3 components inside the 90% band | 1 / 22 | 4 / 22 | |
| sym_touch fill volume / market | 262 | 225 | 164 |
| sym_touch spread capture $/market | 2.19 | 1.77 | 0.68 |
| sym_touch markout 5/30/60 c/share | -2.00 / -3.80 / -4.03 | -2.62 / -4.78 / -4.98 | -2.90 / -4.42 / -4.86 |
| fair_value fill volume / market | 113 | 101 | 86 |
| fair_value markout 5/30/60 c/share | -1.62 / -2.33 / -1.43 | -2.25 / -3.52 / -2.94 | -3.05 / -4.45 / -4.72 |
| G4 failed checks | spread-median (1 cell), trade-rate (3), P(buy\|gap) (3), 5m sign ACF | spread-median (1), trade-rate (3), P(buy\|gap) (2) | |
| G4 sign ACF 5m / 15m / 4h | 0.305 / 0.291 / 0.254 | 0.265 / 0.287 / 0.266 (pass) | 0.232 / 0.298 / 0.262 |
| prints at-or-through the touch, book at stamp row (k = 0), 5m / 15m / 4h | 0.69 / 0.70 / 0.69 | 0.57 / 0.60 / 0.67 | 0.54 / 0.56 / 0.50 |
| same at k = 4, 5m / 15m / 4h | 0.75 / 0.72 / 0.72 | 0.65 / 0.66 / 0.71 | 0.64 / 0.68 / 0.64 |

**Verdict.** The gate still fails, so per SIMULATOR.md §5 the sim stays an initialiser for pretraining,
followed by replay fine-tuning, and no sim result is quoted. On balance it improved:
* the G1 KS distances;
* fair_value P&L and markouts;
* fill volume;
* sign ACF;
* the 5m/15m print alignment.

`data/build_sim/` was regenerated with these params, written to a temp dir and swapped in atomically.

**What still fails, and the leads:**
* **The fair_value quoter is still profitable in the sim** (+425 vs -797), so G2 fails.
  - Its 60 s markout reverts in the sim (-3.5 c at 30 s, -2.9 c at 60 s) but not in the data (-4.5, -4.7).
  - Lead: a slower-decaying impact, or informed flow on a longer component of the fair value. h = 10-20 s
    overshot the 30 s markouts of both baselines in the calibration (`cache/calib_refine.log`).
* **sym_touch spread capture is 2.6x real**, and it does not respond to any flow parameter: look-ahead,
  tracking, refill, impact and copy probability were all tried.
  - A per-fill decomposition of through-prints (half-spread at the decision book plus the mid move to the
    match row) matches real by coin and length, e.g. BTC 5m -0.30 real vs -0.38 sim c/share.
  - So the excess comes from fills that decomposition leaves out: probably the env's at-price queue and
    pro-rata cancel inference reacting to the sim's snapshot size redraws. Next step: compare the env's
    queue state on matched real/sim markets.
* **4h prints line up with the captured book far more than in the data** (0.67 vs 0.50 at k = 0).
  - 4h snapshots update with fair-value moves (P(book_age = 0) rises from 0.14 to 0.36 with |Δp|).
    5m/15m books do not. A state-dependent 4h update rate is the lead.
* **Trade rate near close on 4h/15m** (3 cells) and P(buy | gap) (2 cells).
