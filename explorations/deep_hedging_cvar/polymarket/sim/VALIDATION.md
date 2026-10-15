# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 9 min. params.json sha256 `847f7096e38c6c47`, mm.h sha256 `746dd7dea250eee0`.

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
| sym_touch | -2957.6 | -3593.2 [-3946.2, -3245.9] | -3.010 | [-3.236, -1.949] | 0.180 | 0.072 | FAIL |
| fair_value | -796.6 | 389.2 [97.8, 710.6] | 0.000 | [0.000, 0.009] | 0.247 | 0.140 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -2442.1 [-2713.6, -2171.9] |
| sym_touch | 15m | -952.6 | -1160.2 [-1324.6, -972.8] |
| sym_touch | 4h | -93.5 | -46.2 [-112.7, 74.1] |
| fair_value | 5m | -553.0 | 216.9 [-88.6, 551.3] |
| fair_value | 15m | -336.2 | 47.9 [-167.2, 272.5] |
| fair_value | 4h | 92.7 | 122.5 [65.1, 204.7] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 107.695 [100.281, 115.097] | False |
| sym_touch | fill_vol | 164.380 | 222.134 [209.091, 244.556] | False |
| sym_touch | spread_capture | 0.675 | 1.787 [1.579, 1.986] | False |
| sym_touch | markout_5 | -4.763 | -6.034 [-6.364, -5.723] | False |
| sym_touch | markout_30 | -7.265 | -11.099 [-11.771, -10.130] | False |
| sym_touch | markout_60 | -7.982 | -11.479 [-12.341, -10.305] | False |
| sym_touch | inventory_pnl | 0.037 | 0.723 [-0.193, 1.505] | True |
| sym_touch | rebates | 0.407 | 0.472 [0.445, 0.515] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -2.663 [-2.792, -2.529] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -4.846 [-5.235, -4.538] | False |
| sym_touch | markout 60s (c/share) | -4.856 | -5.061 [-5.474, -4.576] | True |
| fair_value | fills | 49.930 | 47.231 [44.129, 50.584] | True |
| fair_value | fill_vol | 85.714 | 100.456 [94.534, 111.235] | False |
| fair_value | spread_capture | 1.354 | 1.845 [1.670, 2.009] | False |
| fair_value | markout_5 | -2.611 | -2.272 [-2.535, -2.207] | False |
| fair_value | markout_30 | -3.815 | -3.591 [-4.210, -3.162] | True |
| fair_value | markout_60 | -4.047 | -3.104 [-3.616, -2.256] | False |
| fair_value | inventory_pnl | 0.629 | 1.799 [1.344, 2.511] | False |
| fair_value | rebates | 0.216 | 0.210 [0.200, 0.236] | True |
| fair_value | markout 5s (c/share) | -3.046 | -2.311 [-2.382, -2.208] | False |
| fair_value | markout 30s (c/share) | -4.451 | -3.498 [-3.867, -3.202] | False |
| fair_value | markout 60s (c/share) | -4.721 | -3.031 [-3.456, -2.213] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=1, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | FAIL (n_fail=3, n=19) |
| ks_spread|5m | PASS (D=0.043) |
| ks_spread|15m | PASS (D=0.013) |
| ks_spread|4h | PASS (D=0.039) |
| ks_log_top_sz|5m | PASS (D=0.020) |
| ks_log_top_sz|15m | PASS (D=0.010) |
| ks_log_top_sz|4h | PASS (D=0.005) |
| ks_log_top5|5m | PASS (D=0.036) |
| ks_log_top5|15m | PASS (D=0.030) |
| ks_log_top5|4h | PASS (D=0.050) |
| ks_trades_per_market|5m | PASS (D=0.135) |
| ks_trades_per_market|15m | PASS (D=0.123) |
| ks_trades_per_market|4h | FAIL (D=0.201) |
| pbuy_gap_curve | FAIL (n_fail=2, n=29) |
| sign_acf1|5m | PASS (real=0.232, sim=0.265) |
| sign_acf1|15m | PASS (real=0.298, sim=0.287) |
| sign_acf1|4h | PASS (real=0.262, sim=0.266) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.77 | 50/50 | 250/143 | 0.32/0.27 | 0.388/0.380 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.63 | 50/50 | 255/190 | 0.29/0.26 | 0.597/0.611 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.020 | 0.47/0.48 | 50/50 | 260/250 | 0.29/0.26 | 1.424/1.164 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.44 | 50/50 | 300/280 | 0.32/0.28 | 1.434/1.537 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.44 | 50/50 | 322/316 | 0.32/0.28 | 1.652/1.737 |
| 15m | 0 | 0.899/0.890 | 0.067/0.007 | 0.010/0.009 | 0.61/0.84 | 50/50 | 185/177 | 0.30/0.22 | 0.028/0.067 |
| 15m | 1 | 0.834/0.826 | 0.085/0.020 | 0.010/0.009 | 0.58/0.74 | 50/50 | 250/178 | 0.25/0.19 | 0.071/0.101 |
| 15m | 2 | 0.700/0.696 | 0.085/0.034 | 0.020/0.010 | 0.46/0.51 | 50/50 | 250/250 | 0.23/0.18 | 0.221/0.203 |
| 15m | 3 | 0.450/0.450 | 0.056/0.054 | 0.019/0.020 | 0.48/0.46 | 50/50 | 317/298 | 0.26/0.19 | 0.311/0.267 |
| 15m | 4 | 0.175/0.183 | 0.067/0.075 | 0.020/0.020 | 0.43/0.43 | 50/50 | 427/382 | 0.29/0.21 | 0.381/0.346 |
| 15m | 5 | 0.006/0.009 | 0.092/0.097 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/390 | 0.29/0.21 | 0.359/0.342 |
| 4h | 0 | 0.700/0.649 | 0.081/0.027 | 0.008/0.009 | 0.75/0.79 | 50/52 | 265/208 | 0.07/0.07 | 0.065/0.027 |
| 4h | 1 | 0.672/0.626 | 0.081/0.027 | 0.009/0.009 | 0.77/0.75 | 55/63 | 315/229 | 0.11/0.11 | 0.030/0.025 |
| 4h | 2 | 0.532/0.480 | 0.081/0.041 | 0.009/0.009 | 0.66/0.72 | 100/55 | 315/199 | 0.07/0.07 | 0.048/0.027 |
| 4h | 3 | 0.447/0.384 | 0.084/0.055 | 0.009/0.009 | 0.61/0.67 | 103/50 | 294/246 | 0.09/0.10 | 0.034/0.024 |
| 4h | 4 | 0.203/0.190 | 0.095/0.081 | 0.009/0.009 | 0.63/0.62 | 55/54 | 316/262 | 0.06/0.06 | 0.023/0.016 |
| 4h | 5 | 0.093/0.081 | 0.084/0.081 | 0.009/0.009 | 0.58/0.59 | 53/51 | 320/262 | 0.09/0.09 | 0.022/0.017 |
| 4h | 6 | 0.047/0.030 | 0.081/0.081 | 0.011/0.011 | 0.49/0.50 | 50/50 | 338/300 | 0.10/0.10 | 0.015/0.012 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.010 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.408 |
| 5m|1 | 12459 | 0.474 | 0.475 |
| 5m|2 | 12977 | 0.458 | 0.479 |
| 5m|3 | 6481 | 0.457 | 0.487 |
| 5m|4 | 7216 | 0.513 | 0.584 |
| 5m|5 | 8577 | 0.504 | 0.440 |
| 5m|6 | 6910 | 0.538 | 0.511 |
| 5m|7 | 13334 | 0.536 | 0.526 |
| 5m|8 | 11837 | 0.541 | 0.536 |
| 5m|9 | 13657 | 0.568 | 0.576 |
| 15m|0 | 2616 | 0.426 | 0.430 |
| 15m|1 | 4776 | 0.460 | 0.461 |
| 15m|2 | 7510 | 0.489 | 0.484 |
| 15m|3 | 3528 | 0.460 | 0.479 |
| 15m|4 | 3913 | 0.496 | 0.529 |
| 15m|5 | 4166 | 0.528 | 0.475 |
| 15m|6 | 3528 | 0.556 | 0.534 |
| 15m|7 | 7364 | 0.565 | 0.553 |
| 15m|8 | 5177 | 0.583 | 0.570 |
| 15m|9 | 2607 | 0.600 | 0.604 |
| 4h|1 | 455 | 0.433 | 0.457 |
| 4h|2 | 545 | 0.424 | 0.418 |
| 4h|3 | 363 | 0.424 | 0.460 |
| 4h|4 | 532 | 0.451 | 0.453 |
| 4h|5 | 745 | 0.541 | 0.546 |
| 4h|6 | 573 | 0.635 | 0.606 |
| 4h|7 | 1301 | 0.587 | 0.577 |
| 4h|8 | 1147 | 0.616 | 0.612 |
| 4h|9 | 655 | 0.594 | 0.582 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.570 | [0.555, 0.580] |
| align_through|5m|k0 | 0.446 | 0.443 | [0.426, 0.459] |
| align_at_or_through|15m|k0 | 0.561 | 0.600 | [0.586, 0.621] |
| align_through|15m|k0 | 0.407 | 0.412 | [0.400, 0.440] |
| align_at_or_through|4h|k0 | 0.498 | 0.671 | [0.657, 0.683] |
| align_through|4h|k0 | 0.206 | 0.419 | [0.399, 0.435] |
| align_at_or_through|5m|k1 | 0.561 | 0.606 | [0.588, 0.615] |
| align_through|5m|k1 | 0.453 | 0.455 | [0.436, 0.470] |
| align_at_or_through|15m|k1 | 0.587 | 0.628 | [0.616, 0.648] |
| align_through|15m|k1 | 0.424 | 0.423 | [0.411, 0.451] |
| align_at_or_through|4h|k1 | 0.537 | 0.687 | [0.671, 0.700] |
| align_through|4h|k1 | 0.225 | 0.416 | [0.395, 0.432] |
| align_at_or_through|5m|k2 | 0.590 | 0.641 | [0.626, 0.650] |
| align_through|5m|k2 | 0.469 | 0.465 | [0.445, 0.476] |
| align_at_or_through|15m|k2 | 0.621 | 0.655 | [0.646, 0.673] |
| align_through|15m|k2 | 0.453 | 0.433 | [0.420, 0.460] |
| align_at_or_through|4h|k2 | 0.577 | 0.703 | [0.685, 0.719] |
| align_through|4h|k2 | 0.247 | 0.412 | [0.393, 0.430] |
| align_at_or_through|5m|k3 | 0.626 | 0.671 | [0.661, 0.681] |
| align_through|5m|k3 | 0.511 | 0.470 | [0.450, 0.480] |
| align_at_or_through|15m|k3 | 0.665 | 0.677 | [0.669, 0.692] |
| align_through|15m|k3 | 0.507 | 0.443 | [0.428, 0.466] |
| align_at_or_through|4h|k3 | 0.628 | 0.717 | [0.698, 0.732] |
| align_through|4h|k3 | 0.287 | 0.408 | [0.389, 0.425] |
| align_at_or_through|5m|k4 | 0.643 | 0.648 | [0.639, 0.660] |
| align_through|5m|k4 | 0.558 | 0.495 | [0.479, 0.505] |
| align_at_or_through|15m|k4 | 0.682 | 0.659 | [0.651, 0.674] |
| align_through|15m|k4 | 0.557 | 0.467 | [0.454, 0.489] |
| align_at_or_through|4h|k4 | 0.641 | 0.708 | [0.691, 0.723] |
| align_through|4h|k4 | 0.317 | 0.416 | [0.394, 0.438] |
| align_at_or_through|5m|k6 | 0.624 | 0.585 | [0.577, 0.598] |
| align_through|5m|k6 | 0.572 | 0.522 | [0.510, 0.533] |
| align_at_or_through|15m|k6 | 0.659 | 0.607 | [0.595, 0.618] |
| align_through|15m|k6 | 0.567 | 0.498 | [0.490, 0.521] |
| align_at_or_through|4h|k6 | 0.630 | 0.661 | [0.648, 0.674] |
| align_through|4h|k6 | 0.344 | 0.443 | [0.413, 0.459] |
| sign_acf|5m|1 | 0.232 | 0.265 | [0.252, 0.279] |
| sign_acf|5m|2 | 0.196 | 0.246 | [0.232, 0.261] |
| sign_acf|5m|5 | 0.166 | 0.206 | [0.189, 0.225] |
| sign_acf|15m|1 | 0.298 | 0.287 | [0.263, 0.302] |
| sign_acf|15m|2 | 0.206 | 0.204 | [0.170, 0.229] |
| sign_acf|15m|5 | 0.123 | 0.124 | [0.085, 0.162] |
| sign_acf|4h|1 | 0.262 | 0.266 | [0.242, 0.280] |
| sign_acf|4h|2 | 0.222 | 0.082 | [0.066, 0.101] |
| sign_acf|4h|5 | 0.138 | 0.018 | [-0.003, 0.036] |
| markout|5m|0 | 2.975 | 1.898 | [1.757, 2.247] |
| markout|15m|0 | 1.983 | 1.333 | [1.192, 1.444] |
| markout|4h|0 | 1.127 | 1.458 | [1.273, 1.657] |
| markout|5m|5 | 1.611 | 0.029 | [-0.161, 0.359] |
| markout|15m|5 | 0.634 | 0.357 | [0.187, 0.448] |
| markout|4h|5 | 0.476 | 1.383 | [1.203, 1.588] |
| markout|5m|30 | 0.527 | -2.323 | [-3.232, -1.701] |
| markout|15m|30 | 0.004 | -1.002 | [-1.413, -0.569] |
| markout|4h|30 | 0.017 | 0.982 | [0.830, 1.171] |
| markout|5m|60 | 0.111 | -2.354 | [-2.891, -2.004] |
| markout|15m|60 | 0.117 | -1.070 | [-1.691, -0.615] |
| markout|4h|60 | -0.087 | 0.843 | [0.674, 1.101] |
| taker_size_med|5m | 5.880 | 5.435 | [5.145, 5.915] |
| taker_size_p90|5m | 40.159 | 38.951 | [36.287, 40.470] |
| levels_per_tx|5m | 1.179 | 1.164 | [1.157, 1.171] |
| buy_share|5m | 0.498 | 0.501 | [0.497, 0.507] |
| taker_size_med|15m | 5.000 | 5.030 | [5.000, 5.550] |
| taker_size_p90|15m | 38.000 | 39.081 | [34.565, 42.541] |
| levels_per_tx|15m | 1.130 | 1.173 | [1.168, 1.182] |
| buy_share|15m | 0.519 | 0.509 | [0.505, 0.514] |
| taker_size_med|4h | 20.000 | 17.597 | [16.503, 19.672] |
| taker_size_p90|4h | 100.000 | 100.000 | [100.000, 100.007] |
| levels_per_tx|4h | 1.136 | 1.315 | [1.303, 1.341] |
| buy_share|4h | 0.541 | 0.530 | [0.513, 0.545] |
| mid|5m|slope_1 | 0.210 | 0.243 | [0.232, 0.259] |
| mid|5m|slope_5 | 0.505 | 0.507 | [0.491, 0.521] |
| mid|5m|slope_30 | 0.747 | 0.808 | [0.796, 0.835] |
| mid|5m|slope_120 | 0.838 | 0.860 | [0.839, 0.888] |
| mid|5m|level_slope | 0.839 | 0.871 | [0.855, 0.890] |
| mid|5m|sd_gap | 0.562 | 0.565 | [0.549, 0.585] |
| mid|5m|acf_5 | 0.759 | 0.746 | [0.735, 0.761] |
| mid|5m|acf_30 | 0.469 | 0.444 | [0.411, 0.468] |
| mid|5m|acf_120 | 0.189 | 0.077 | [0.015, 0.112] |
| mid|5m|p_change | 0.263 | 0.247 | [0.244, 0.250] |
| mid|15m|slope_1 | 0.173 | 0.187 | [0.177, 0.205] |
| mid|15m|slope_5 | 0.446 | 0.423 | [0.404, 0.438] |
| mid|15m|slope_30 | 0.746 | 0.772 | [0.760, 0.778] |
| mid|15m|slope_120 | 0.843 | 0.863 | [0.853, 0.882] |
| mid|15m|slope_300 | 0.864 | 0.871 | [0.856, 0.886] |
| mid|15m|level_slope | 0.871 | 0.860 | [0.851, 0.876] |
| mid|15m|sd_gap | 0.388 | 0.395 | [0.384, 0.407] |
| mid|15m|acf_5 | 0.814 | 0.812 | [0.797, 0.823] |
| mid|15m|acf_30 | 0.634 | 0.640 | [0.618, 0.666] |
| mid|15m|acf_120 | 0.434 | 0.413 | [0.372, 0.448] |
| mid|15m|p_change | 0.199 | 0.181 | [0.179, 0.181] |
| mid|4h|slope_1 | 0.125 | 0.106 | [0.096, 0.140] |
| mid|4h|slope_5 | 0.362 | 0.272 | [0.260, 0.299] |
| mid|4h|slope_30 | 0.595 | 0.577 | [0.562, 0.586] |
| mid|4h|slope_120 | 0.683 | 0.683 | [0.669, 0.694] |
| mid|4h|slope_300 | 0.721 | 0.710 | [0.684, 0.724] |
| mid|4h|slope_900 | 0.754 | 0.747 | [0.714, 0.777] |
| mid|4h|level_slope | 0.910 | 0.897 | [0.882, 0.926] |
| mid|4h|sd_gap | 0.436 | 0.460 | [0.422, 0.477] |
| mid|4h|acf_5 | 0.986 | 0.980 | [0.976, 0.984] |
| mid|4h|acf_30 | 0.954 | 0.947 | [0.938, 0.954] |
| mid|4h|acf_120 | 0.873 | 0.878 | [0.857, 0.890] |
| mid|4h|p_change | 0.083 | 0.137 | [0.136, 0.138] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -9849.1 [-11016.7, -8841.3] | 0.086 |
| fv | -2585.2 | 507.3 [-950.9, 1685.2] | 0.158 |
