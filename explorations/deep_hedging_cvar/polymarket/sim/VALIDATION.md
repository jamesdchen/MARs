# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 9 min. params.json sha256 `2fbdc2647ee5648a`, mm.h sha256 `9189f83adbb3f42a`.

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
| G2 rank order | PASS: real ['no_quote', 'fair_value', 'sym_touch'], sim ['no_quote', 'fair_value', 'sym_touch'] |
| G3 attribution, fills, markout curve inside the 90% band | FAIL |
| G4 stylised facts (numpy statistics on the same chunks) | FAIL |
| G5 out-of-sample on the val day | not run: it needs a ledger entry (EVALUATION.md T4) |

Gate definitions: **G1** per policy: real total and median per-market P&L inside the sim 5-95% band of replications, and KS D < 0.15 between the real hourly P&L series and the pooled sim hourly series (markets binned by end hour, so block structure is kept); per-market KS D is reported; **G2** rank order of {no_quote, sym_touch, fair_value} by total P&L identical (real vs sim median); **G3** per-market means of fills, fill volume, spread capture, markout 5/30/60, inventory P&L and rebates, and sym_touch's markout curve (c/share), inside the sim 5-95% band; **G4** stylised facts as in validate.py GATE (spread, depth, one-sided, trade rate, P(buy|gap), sign ACF)

## G1/G2: P&L through the C env ($)

| policy | real total | sim total median [5%, 95%] | real median/mkt | sim median/mkt [5%, 95%] | KS hourly | KS per-market | pass |
|---|---|---|---|---|---|---|---|
| sym_touch | -2957.6 | -3110.2 [-3661.7, -2775.1] | -3.010 | [-2.826, -1.175] | 0.128 | 0.079 | FAIL |
| fair_value | -796.6 | -54.3 [-343.7, 271.0] | 0.000 | [0.000, 0.000] | 0.193 | 0.111 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -2035.0 [-2372.3, -1831.0] |
| sym_touch | 15m | -952.6 | -1116.1 [-1396.1, -865.0] |
| sym_touch | 4h | -93.5 | -7.6 [-75.4, 106.8] |
| fair_value | 5m | -553.0 | -78.4 [-373.9, 174.5] |
| fair_value | 15m | -336.2 | -32.2 [-199.2, 87.6] |
| fair_value | 4h | 92.7 | 93.8 [39.0, 201.0] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 101.784 [92.076, 107.994] | False |
| sym_touch | fill_vol | 164.380 | 211.628 [190.010, 232.904] | False |
| sym_touch | spread_capture | 0.675 | 1.603 [1.468, 1.797] | False |
| sym_touch | markout_5 | -4.763 | -4.882 [-5.154, -4.585] | True |
| sym_touch | markout_30 | -7.265 | -8.832 [-9.716, -8.237] | False |
| sym_touch | markout_60 | -7.982 | -9.235 [-10.181, -8.476] | False |
| sym_touch | inventory_pnl | 0.037 | -0.196 [-0.639, 0.670] | True |
| sym_touch | rebates | 0.407 | 0.473 [0.430, 0.522] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -2.303 [-2.445, -2.185] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -4.254 [-4.474, -3.915] | True |
| sym_touch | markout 60s (c/share) | -4.856 | -4.471 [-4.729, -3.919] | False |
| fair_value | fills | 49.930 | 46.232 [41.252, 49.170] | False |
| fair_value | fill_vol | 85.714 | 97.573 [86.963, 107.875] | False |
| fair_value | spread_capture | 1.354 | 1.752 [1.569, 1.927] | False |
| fair_value | markout_5 | -2.611 | -1.855 [-2.017, -1.708] | False |
| fair_value | markout_30 | -3.815 | -3.153 [-3.571, -2.655] | False |
| fair_value | markout_60 | -4.047 | -2.926 [-3.738, -2.353] | False |
| fair_value | inventory_pnl | 0.629 | 0.979 [0.282, 1.877] | True |
| fair_value | rebates | 0.216 | 0.207 [0.189, 0.230] | True |
| fair_value | markout 5s (c/share) | -3.046 | -1.921 [-2.109, -1.800] | False |
| fair_value | markout 30s (c/share) | -4.451 | -3.235 [-3.652, -3.011] | False |
| fair_value | markout 60s (c/share) | -4.721 | -3.146 [-3.968, -2.563] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=1, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | FAIL (n_fail=3, n=19) |
| ks_spread|5m | PASS (D=0.039) |
| ks_spread|15m | PASS (D=0.013) |
| ks_spread|4h | PASS (D=0.039) |
| ks_log_top_sz|5m | PASS (D=0.029) |
| ks_log_top_sz|15m | PASS (D=0.009) |
| ks_log_top_sz|4h | PASS (D=0.006) |
| ks_log_top5|5m | PASS (D=0.030) |
| ks_log_top5|15m | PASS (D=0.037) |
| ks_log_top5|4h | PASS (D=0.050) |
| ks_trades_per_market|5m | PASS (D=0.146) |
| ks_trades_per_market|15m | PASS (D=0.102) |
| ks_trades_per_market|4h | PASS (D=0.126) |
| pbuy_gap_curve | FAIL (n_fail=2, n=29) |
| sign_acf1|5m | PASS (real=0.232, sim=0.222) |
| sign_acf1|15m | PASS (real=0.298, sim=0.326) |
| sign_acf1|4h | PASS (real=0.262, sim=0.246) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.81 | 50/50 | 250/163 | 0.32/0.26 | 0.388/0.398 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.66 | 50/50 | 255/207 | 0.29/0.24 | 0.597/0.672 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.010 | 0.47/0.52 | 50/50 | 260/250 | 0.29/0.24 | 1.424/1.390 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.47 | 50/50 | 300/292 | 0.32/0.27 | 1.434/1.456 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.45 | 50/50 | 322/320 | 0.32/0.26 | 1.652/1.638 |
| 15m | 0 | 0.899/0.885 | 0.067/0.011 | 0.010/0.009 | 0.61/0.84 | 50/50 | 185/155 | 0.30/0.22 | 0.028/0.048 |
| 15m | 1 | 0.834/0.824 | 0.085/0.023 | 0.010/0.009 | 0.58/0.71 | 50/50 | 250/168 | 0.25/0.19 | 0.071/0.099 |
| 15m | 2 | 0.700/0.697 | 0.085/0.041 | 0.020/0.013 | 0.46/0.51 | 50/50 | 250/254 | 0.23/0.18 | 0.221/0.237 |
| 15m | 3 | 0.450/0.442 | 0.056/0.055 | 0.019/0.020 | 0.48/0.46 | 50/50 | 317/301 | 0.26/0.19 | 0.311/0.357 |
| 15m | 4 | 0.175/0.166 | 0.067/0.078 | 0.020/0.020 | 0.43/0.44 | 50/50 | 427/364 | 0.29/0.21 | 0.381/0.429 |
| 15m | 5 | 0.006/0.007 | 0.092/0.099 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/386 | 0.29/0.21 | 0.359/0.403 |
| 4h | 0 | 0.700/0.703 | 0.081/0.027 | 0.008/0.009 | 0.75/0.75 | 50/56 | 265/212 | 0.07/0.07 | 0.065/0.038 |
| 4h | 1 | 0.672/0.664 | 0.081/0.027 | 0.009/0.009 | 0.77/0.75 | 55/50 | 315/199 | 0.11/0.11 | 0.030/0.019 |
| 4h | 2 | 0.532/0.540 | 0.081/0.031 | 0.009/0.009 | 0.66/0.71 | 100/56 | 315/199 | 0.07/0.07 | 0.048/0.036 |
| 4h | 3 | 0.447/0.438 | 0.084/0.054 | 0.009/0.009 | 0.61/0.66 | 103/55 | 294/200 | 0.09/0.10 | 0.034/0.028 |
| 4h | 4 | 0.203/0.195 | 0.095/0.065 | 0.009/0.009 | 0.63/0.64 | 55/50 | 316/250 | 0.06/0.06 | 0.023/0.019 |
| 4h | 5 | 0.093/0.085 | 0.084/0.081 | 0.009/0.009 | 0.58/0.60 | 53/52 | 320/254 | 0.09/0.09 | 0.022/0.019 |
| 4h | 6 | 0.047/0.040 | 0.081/0.081 | 0.011/0.010 | 0.49/0.50 | 50/50 | 338/300 | 0.10/0.10 | 0.015/0.012 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.009 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.418 |
| 5m|1 | 12459 | 0.474 | 0.477 |
| 5m|2 | 12977 | 0.458 | 0.488 |
| 5m|3 | 6481 | 0.457 | 0.501 |
| 5m|4 | 7216 | 0.513 | 0.595 |
| 5m|5 | 8577 | 0.504 | 0.404 |
| 5m|6 | 6910 | 0.538 | 0.492 |
| 5m|7 | 13334 | 0.536 | 0.520 |
| 5m|8 | 11837 | 0.541 | 0.533 |
| 5m|9 | 13657 | 0.568 | 0.571 |
| 15m|0 | 2616 | 0.426 | 0.436 |
| 15m|1 | 4776 | 0.460 | 0.455 |
| 15m|2 | 7510 | 0.489 | 0.483 |
| 15m|3 | 3528 | 0.460 | 0.476 |
| 15m|4 | 3913 | 0.496 | 0.525 |
| 15m|5 | 4166 | 0.528 | 0.475 |
| 15m|6 | 3528 | 0.556 | 0.519 |
| 15m|7 | 7364 | 0.565 | 0.547 |
| 15m|8 | 5177 | 0.583 | 0.569 |
| 15m|9 | 2607 | 0.600 | 0.598 |
| 4h|1 | 455 | 0.433 | 0.436 |
| 4h|2 | 545 | 0.424 | 0.436 |
| 4h|3 | 363 | 0.424 | 0.434 |
| 4h|4 | 532 | 0.451 | 0.440 |
| 4h|5 | 745 | 0.541 | 0.545 |
| 4h|6 | 573 | 0.635 | 0.597 |
| 4h|7 | 1301 | 0.587 | 0.576 |
| 4h|8 | 1147 | 0.616 | 0.601 |
| 4h|9 | 655 | 0.594 | 0.578 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.603 | [0.585, 0.618] |
| align_through|5m|k0 | 0.446 | 0.468 | [0.450, 0.482] |
| align_at_or_through|15m|k0 | 0.561 | 0.616 | [0.599, 0.659] |
| align_through|15m|k0 | 0.407 | 0.422 | [0.407, 0.461] |
| align_at_or_through|4h|k0 | 0.498 | 0.685 | [0.672, 0.702] |
| align_through|4h|k0 | 0.206 | 0.427 | [0.413, 0.448] |
| align_at_or_through|5m|k1 | 0.561 | 0.631 | [0.612, 0.646] |
| align_through|5m|k1 | 0.453 | 0.475 | [0.458, 0.489] |
| align_at_or_through|15m|k1 | 0.587 | 0.646 | [0.627, 0.684] |
| align_through|15m|k1 | 0.424 | 0.435 | [0.415, 0.473] |
| align_at_or_through|4h|k1 | 0.537 | 0.699 | [0.684, 0.713] |
| align_through|4h|k1 | 0.225 | 0.422 | [0.405, 0.444] |
| align_at_or_through|5m|k2 | 0.590 | 0.658 | [0.642, 0.672] |
| align_through|5m|k2 | 0.469 | 0.480 | [0.461, 0.496] |
| align_at_or_through|15m|k2 | 0.621 | 0.673 | [0.652, 0.705] |
| align_through|15m|k2 | 0.453 | 0.446 | [0.424, 0.481] |
| align_at_or_through|4h|k2 | 0.577 | 0.710 | [0.700, 0.723] |
| align_through|4h|k2 | 0.247 | 0.416 | [0.394, 0.434] |
| align_at_or_through|5m|k3 | 0.626 | 0.680 | [0.668, 0.694] |
| align_through|5m|k3 | 0.511 | 0.480 | [0.462, 0.495] |
| align_at_or_through|15m|k3 | 0.665 | 0.689 | [0.675, 0.718] |
| align_through|15m|k3 | 0.507 | 0.453 | [0.430, 0.487] |
| align_at_or_through|4h|k3 | 0.628 | 0.723 | [0.715, 0.737] |
| align_through|4h|k3 | 0.287 | 0.407 | [0.385, 0.429] |
| align_at_or_through|5m|k4 | 0.643 | 0.656 | [0.644, 0.667] |
| align_through|5m|k4 | 0.558 | 0.500 | [0.484, 0.515] |
| align_at_or_through|15m|k4 | 0.682 | 0.670 | [0.656, 0.700] |
| align_through|15m|k4 | 0.557 | 0.477 | [0.454, 0.505] |
| align_at_or_through|4h|k4 | 0.641 | 0.716 | [0.701, 0.729] |
| align_through|4h|k4 | 0.317 | 0.416 | [0.390, 0.437] |
| align_at_or_through|5m|k6 | 0.624 | 0.598 | [0.586, 0.610] |
| align_through|5m|k6 | 0.572 | 0.527 | [0.514, 0.540] |
| align_at_or_through|15m|k6 | 0.659 | 0.618 | [0.601, 0.649] |
| align_through|15m|k6 | 0.567 | 0.511 | [0.486, 0.528] |
| align_at_or_through|4h|k6 | 0.630 | 0.672 | [0.653, 0.684] |
| align_through|4h|k6 | 0.344 | 0.443 | [0.420, 0.464] |
| sign_acf|5m|1 | 0.232 | 0.222 | [0.212, 0.238] |
| sign_acf|5m|2 | 0.196 | 0.214 | [0.205, 0.227] |
| sign_acf|5m|5 | 0.166 | 0.191 | [0.175, 0.202] |
| sign_acf|15m|1 | 0.298 | 0.326 | [0.303, 0.363] |
| sign_acf|15m|2 | 0.206 | 0.285 | [0.254, 0.329] |
| sign_acf|15m|5 | 0.123 | 0.220 | [0.163, 0.252] |
| sign_acf|4h|1 | 0.262 | 0.246 | [0.224, 0.282] |
| sign_acf|4h|2 | 0.222 | 0.102 | [0.082, 0.126] |
| sign_acf|4h|5 | 0.138 | 0.024 | [0.011, 0.045] |
| markout|5m|0 | 2.975 | 1.838 | [1.668, 1.976] |
| markout|15m|0 | 1.983 | 1.375 | [1.168, 1.502] |
| markout|4h|0 | 1.127 | 1.570 | [1.378, 1.657] |
| markout|5m|5 | 1.611 | 0.337 | [0.228, 0.473] |
| markout|15m|5 | 0.634 | 0.411 | [0.276, 0.663] |
| markout|4h|5 | 0.476 | 1.520 | [1.364, 1.634] |
| markout|5m|30 | 0.527 | -1.563 | [-1.968, -1.246] |
| markout|15m|30 | 0.004 | -1.103 | [-1.431, -0.840] |
| markout|4h|30 | 0.017 | 1.430 | [1.311, 1.566] |
| markout|5m|60 | 0.111 | -1.915 | [-2.369, -1.458] |
| markout|15m|60 | 0.117 | -1.112 | [-1.448, -0.762] |
| markout|4h|60 | -0.087 | 1.395 | [1.286, 1.569] |
| taker_size_med|5m | 5.880 | 5.424 | [5.174, 5.907] |
| taker_size_p90|5m | 40.159 | 39.355 | [36.662, 40.895] |
| levels_per_tx|5m | 1.179 | 1.162 | [1.155, 1.169] |
| buy_share|5m | 0.498 | 0.505 | [0.497, 0.516] |
| taker_size_med|15m | 5.000 | 5.143 | [5.000, 5.672] |
| taker_size_p90|15m | 38.000 | 40.699 | [36.963, 43.336] |
| levels_per_tx|15m | 1.130 | 1.171 | [1.164, 1.176] |
| buy_share|15m | 0.519 | 0.514 | [0.505, 0.523] |
| taker_size_med|4h | 20.000 | 17.782 | [15.552, 18.760] |
| taker_size_p90|4h | 100.000 | 100.000 | [92.350, 100.000] |
| levels_per_tx|4h | 1.136 | 1.332 | [1.313, 1.349] |
| buy_share|4h | 0.541 | 0.526 | [0.501, 0.542] |
| mid|5m|slope_1 | 0.210 | 0.222 | [0.212, 0.232] |
| mid|5m|slope_5 | 0.505 | 0.466 | [0.454, 0.481] |
| mid|5m|slope_30 | 0.747 | 0.752 | [0.741, 0.768] |
| mid|5m|slope_120 | 0.838 | 0.813 | [0.798, 0.844] |
| mid|5m|level_slope | 0.839 | 0.842 | [0.825, 0.861] |
| mid|5m|sd_gap | 0.562 | 0.555 | [0.539, 0.567] |
| mid|5m|acf_5 | 0.759 | 0.751 | [0.737, 0.762] |
| mid|5m|acf_30 | 0.469 | 0.497 | [0.473, 0.530] |
| mid|5m|acf_120 | 0.189 | 0.156 | [0.121, 0.199] |
| mid|5m|p_change | 0.263 | 0.232 | [0.229, 0.235] |
| mid|5m|gapdp_30 | -0.037 | -0.059 | [-0.117, -0.008] |
| mid|5m|gapdy_30 | 0.391 | 0.403 | [0.338, 0.437] |
| mid|5m|gapdp_60 | -0.110 | -0.134 | [-0.258, -0.033] |
| mid|5m|gapdy_60 | 0.420 | 0.479 | [0.368, 0.566] |
| mid|15m|slope_1 | 0.173 | 0.179 | [0.170, 0.196] |
| mid|15m|slope_5 | 0.446 | 0.405 | [0.394, 0.416] |
| mid|15m|slope_30 | 0.746 | 0.745 | [0.735, 0.751] |
| mid|15m|slope_120 | 0.843 | 0.835 | [0.824, 0.845] |
| mid|15m|slope_300 | 0.864 | 0.852 | [0.834, 0.872] |
| mid|15m|level_slope | 0.871 | 0.856 | [0.850, 0.869] |
| mid|15m|sd_gap | 0.388 | 0.394 | [0.376, 0.403] |
| mid|15m|acf_5 | 0.814 | 0.813 | [0.796, 0.821] |
| mid|15m|acf_30 | 0.634 | 0.661 | [0.630, 0.675] |
| mid|15m|acf_120 | 0.434 | 0.452 | [0.402, 0.479] |
| mid|15m|p_change | 0.199 | 0.178 | [0.175, 0.180] |
| mid|15m|gapdp_30 | -0.027 | -0.009 | [-0.052, 0.017] |
| mid|15m|gapdy_30 | 0.292 | 0.301 | [0.268, 0.334] |
| mid|15m|gapdp_60 | -0.034 | -0.009 | [-0.110, 0.031] |
| mid|15m|gapdy_60 | 0.333 | 0.364 | [0.283, 0.401] |
| mid|4h|slope_1 | 0.125 | 0.104 | [0.095, 0.117] |
| mid|4h|slope_5 | 0.362 | 0.259 | [0.250, 0.276] |
| mid|4h|slope_30 | 0.595 | 0.559 | [0.553, 0.578] |
| mid|4h|slope_120 | 0.683 | 0.670 | [0.658, 0.678] |
| mid|4h|slope_300 | 0.721 | 0.711 | [0.693, 0.721] |
| mid|4h|slope_900 | 0.754 | 0.791 | [0.767, 0.811] |
| mid|4h|level_slope | 0.910 | 0.898 | [0.861, 0.920] |
| mid|4h|sd_gap | 0.436 | 0.457 | [0.428, 0.486] |
| mid|4h|acf_5 | 0.986 | 0.980 | [0.975, 0.983] |
| mid|4h|acf_30 | 0.954 | 0.949 | [0.939, 0.957] |
| mid|4h|acf_120 | 0.873 | 0.885 | [0.869, 0.902] |
| mid|4h|p_change | 0.083 | 0.136 | [0.135, 0.137] |
| mid|4h|gapdp_30 | -0.002 | -0.012 | [-0.015, -0.005] |
| mid|4h|gapdy_30 | 0.033 | 0.035 | [0.029, 0.041] |
| mid|4h|gapdp_60 | -0.009 | -0.023 | [-0.030, -0.011] |
| mid|4h|gapdy_60 | 0.046 | 0.042 | [0.035, 0.051] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -9214.2 [-11959.7, -7994.2] | 0.110 |
| fv | -2585.2 | -490.0 [-2648.7, 1135.3] | 0.149 |
