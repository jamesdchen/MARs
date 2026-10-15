# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 8 min. params.json sha256 `6768863e591dddd1`, mm.h sha256 `746dd7dea250eee0`.

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
| sym_touch | -2957.6 | -3385.0 [-3640.4, -3019.5] | -3.010 | [-3.040, -1.860] | 0.175 | 0.070 | FAIL |
| fair_value | -796.6 | -111.9 [-318.9, 432.9] | 0.000 | [0.000, 0.000] | 0.175 | 0.116 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -2260.6 [-2410.3, -2054.7] |
| sym_touch | 15m | -952.6 | -1113.6 [-1298.8, -952.8] |
| sym_touch | 4h | -93.5 | -27.3 [-116.3, 37.0] |
| fair_value | 5m | -553.0 | -82.7 [-424.3, 95.8] |
| fair_value | 15m | -336.2 | -76.5 [-197.5, 175.5] |
| fair_value | 4h | 92.7 | 116.2 [34.0, 213.5] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 108.449 [100.311, 116.446] | False |
| sym_touch | fill_vol | 164.380 | 232.329 [212.666, 252.786] | False |
| sym_touch | spread_capture | 0.675 | 1.733 [1.556, 1.898] | False |
| sym_touch | markout_5 | -4.763 | -5.688 [-5.932, -5.496] | False |
| sym_touch | markout_30 | -7.265 | -9.230 [-10.157, -8.935] | False |
| sym_touch | markout_60 | -7.982 | -9.720 [-10.546, -9.260] | False |
| sym_touch | inventory_pnl | 0.037 | -0.225 [-0.827, 0.158] | True |
| sym_touch | rebates | 0.407 | 0.530 [0.479, 0.563] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -2.440 [-2.632, -2.312] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -4.046 [-4.298, -3.800] | False |
| sym_touch | markout 60s (c/share) | -4.856 | -4.227 [-4.542, -3.967] | False |
| fair_value | fills | 49.930 | 47.509 [43.825, 52.904] | True |
| fair_value | fill_vol | 85.714 | 103.933 [96.880, 116.856] | False |
| fair_value | spread_capture | 1.354 | 1.893 [1.675, 2.088] | False |
| fair_value | markout_5 | -2.611 | -2.185 [-2.437, -2.072] | False |
| fair_value | markout_30 | -3.815 | -3.386 [-3.773, -3.121] | False |
| fair_value | markout_60 | -4.047 | -3.274 [-4.025, -2.905] | False |
| fair_value | inventory_pnl | 0.629 | 0.972 [0.312, 2.034] | True |
| fair_value | rebates | 0.216 | 0.227 [0.209, 0.257] | True |
| fair_value | markout 5s (c/share) | -3.046 | -2.106 [-2.204, -1.947] | False |
| fair_value | markout 30s (c/share) | -4.451 | -3.228 [-3.462, -3.004] | False |
| fair_value | markout 60s (c/share) | -4.721 | -3.206 [-3.617, -2.560] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=1, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | PASS (n_fail=0, n=19) |
| ks_spread|5m | PASS (D=0.039) |
| ks_spread|15m | PASS (D=0.014) |
| ks_spread|4h | PASS (D=0.041) |
| ks_log_top_sz|5m | PASS (D=0.022) |
| ks_log_top_sz|15m | PASS (D=0.010) |
| ks_log_top_sz|4h | PASS (D=0.007) |
| ks_log_top5|5m | PASS (D=0.032) |
| ks_log_top5|15m | PASS (D=0.031) |
| ks_log_top5|4h | PASS (D=0.049) |
| ks_trades_per_market|5m | FAIL (D=0.156) |
| ks_trades_per_market|15m | PASS (D=0.104) |
| ks_trades_per_market|4h | PASS (D=0.122) |
| pbuy_gap_curve | FAIL (n_fail=2, n=29) |
| sign_acf1|5m | PASS (real=0.232, sim=0.253) |
| sign_acf1|15m | PASS (real=0.298, sim=0.319) |
| sign_acf1|4h | PASS (real=0.262, sim=0.266) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.81 | 50/50 | 250/135 | 0.32/0.26 | 0.388/0.376 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.66 | 50/50 | 255/207 | 0.29/0.24 | 0.597/0.677 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.010 | 0.47/0.51 | 50/50 | 260/250 | 0.29/0.24 | 1.424/1.403 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.47 | 50/50 | 300/283 | 0.32/0.27 | 1.434/1.370 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.45 | 50/50 | 322/315 | 0.32/0.26 | 1.652/1.548 |
| 15m | 0 | 0.899/0.881 | 0.067/0.010 | 0.010/0.009 | 0.61/0.84 | 50/50 | 185/164 | 0.30/0.22 | 0.028/0.039 |
| 15m | 1 | 0.834/0.819 | 0.085/0.023 | 0.010/0.009 | 0.58/0.76 | 50/50 | 250/192 | 0.25/0.19 | 0.071/0.089 |
| 15m | 2 | 0.700/0.684 | 0.085/0.038 | 0.020/0.010 | 0.46/0.54 | 50/50 | 250/250 | 0.23/0.18 | 0.221/0.236 |
| 15m | 3 | 0.450/0.443 | 0.056/0.058 | 0.019/0.020 | 0.48/0.45 | 50/50 | 317/296 | 0.26/0.19 | 0.311/0.327 |
| 15m | 4 | 0.175/0.174 | 0.067/0.083 | 0.020/0.020 | 0.43/0.44 | 50/50 | 427/380 | 0.29/0.21 | 0.381/0.403 |
| 15m | 5 | 0.006/0.005 | 0.092/0.097 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/390 | 0.29/0.21 | 0.359/0.389 |
| 4h | 0 | 0.700/0.689 | 0.081/0.027 | 0.008/0.009 | 0.75/0.78 | 50/57 | 265/225 | 0.07/0.07 | 0.065/0.057 |
| 4h | 1 | 0.672/0.653 | 0.081/0.034 | 0.009/0.009 | 0.77/0.72 | 55/57 | 315/250 | 0.11/0.11 | 0.030/0.033 |
| 4h | 2 | 0.532/0.532 | 0.081/0.055 | 0.009/0.009 | 0.66/0.65 | 100/50 | 315/215 | 0.07/0.07 | 0.048/0.055 |
| 4h | 3 | 0.447/0.437 | 0.084/0.062 | 0.009/0.009 | 0.61/0.65 | 103/52 | 294/236 | 0.09/0.10 | 0.034/0.035 |
| 4h | 4 | 0.203/0.196 | 0.095/0.081 | 0.009/0.009 | 0.63/0.61 | 55/56 | 316/250 | 0.06/0.06 | 0.023/0.026 |
| 4h | 5 | 0.093/0.095 | 0.084/0.081 | 0.009/0.009 | 0.58/0.59 | 53/50 | 320/260 | 0.09/0.09 | 0.022/0.024 |
| 4h | 6 | 0.047/0.054 | 0.081/0.081 | 0.011/0.011 | 0.49/0.50 | 50/52 | 338/300 | 0.10/0.10 | 0.015/0.014 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.011 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.412 |
| 5m|1 | 12459 | 0.474 | 0.477 |
| 5m|2 | 12977 | 0.458 | 0.479 |
| 5m|3 | 6481 | 0.457 | 0.501 |
| 5m|4 | 7216 | 0.513 | 0.605 |
| 5m|5 | 8577 | 0.504 | 0.408 |
| 5m|6 | 6910 | 0.538 | 0.496 |
| 5m|7 | 13334 | 0.536 | 0.522 |
| 5m|8 | 11837 | 0.541 | 0.534 |
| 5m|9 | 13657 | 0.568 | 0.573 |
| 15m|0 | 2616 | 0.426 | 0.431 |
| 15m|1 | 4776 | 0.460 | 0.459 |
| 15m|2 | 7510 | 0.489 | 0.483 |
| 15m|3 | 3528 | 0.460 | 0.471 |
| 15m|4 | 3913 | 0.496 | 0.528 |
| 15m|5 | 4166 | 0.528 | 0.485 |
| 15m|6 | 3528 | 0.556 | 0.539 |
| 15m|7 | 7364 | 0.565 | 0.553 |
| 15m|8 | 5177 | 0.583 | 0.569 |
| 15m|9 | 2607 | 0.600 | 0.601 |
| 4h|1 | 455 | 0.433 | 0.442 |
| 4h|2 | 545 | 0.424 | 0.438 |
| 4h|3 | 363 | 0.424 | 0.417 |
| 4h|4 | 532 | 0.451 | 0.436 |
| 4h|5 | 745 | 0.541 | 0.548 |
| 4h|6 | 573 | 0.635 | 0.606 |
| 4h|7 | 1301 | 0.587 | 0.575 |
| 4h|8 | 1147 | 0.616 | 0.607 |
| 4h|9 | 655 | 0.594 | 0.595 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.588 | [0.570, 0.604] |
| align_through|5m|k0 | 0.446 | 0.449 | [0.433, 0.469] |
| align_at_or_through|15m|k0 | 0.561 | 0.608 | [0.591, 0.619] |
| align_through|15m|k0 | 0.407 | 0.411 | [0.398, 0.431] |
| align_at_or_through|4h|k0 | 0.498 | 0.681 | [0.661, 0.696] |
| align_through|4h|k0 | 0.206 | 0.422 | [0.389, 0.446] |
| align_at_or_through|5m|k1 | 0.561 | 0.619 | [0.604, 0.633] |
| align_through|5m|k1 | 0.453 | 0.459 | [0.443, 0.480] |
| align_at_or_through|15m|k1 | 0.587 | 0.637 | [0.624, 0.649] |
| align_through|15m|k1 | 0.424 | 0.423 | [0.410, 0.444] |
| align_at_or_through|4h|k1 | 0.537 | 0.694 | [0.671, 0.706] |
| align_through|4h|k1 | 0.225 | 0.418 | [0.385, 0.442] |
| align_at_or_through|5m|k2 | 0.590 | 0.650 | [0.636, 0.663] |
| align_through|5m|k2 | 0.469 | 0.466 | [0.452, 0.486] |
| align_at_or_through|15m|k2 | 0.621 | 0.664 | [0.655, 0.675] |
| align_through|15m|k2 | 0.453 | 0.433 | [0.420, 0.453] |
| align_at_or_through|4h|k2 | 0.577 | 0.707 | [0.678, 0.719] |
| align_through|4h|k2 | 0.247 | 0.411 | [0.376, 0.440] |
| align_at_or_through|5m|k3 | 0.626 | 0.676 | [0.663, 0.691] |
| align_through|5m|k3 | 0.511 | 0.469 | [0.454, 0.485] |
| align_at_or_through|15m|k3 | 0.665 | 0.684 | [0.676, 0.694] |
| align_through|15m|k3 | 0.507 | 0.441 | [0.430, 0.463] |
| align_at_or_through|4h|k3 | 0.628 | 0.722 | [0.691, 0.736] |
| align_through|4h|k3 | 0.287 | 0.406 | [0.371, 0.436] |
| align_at_or_through|5m|k4 | 0.643 | 0.653 | [0.640, 0.667] |
| align_through|5m|k4 | 0.558 | 0.493 | [0.480, 0.510] |
| align_at_or_through|15m|k4 | 0.682 | 0.667 | [0.658, 0.676] |
| align_through|15m|k4 | 0.557 | 0.466 | [0.454, 0.483] |
| align_at_or_through|4h|k4 | 0.641 | 0.714 | [0.686, 0.726] |
| align_through|4h|k4 | 0.317 | 0.413 | [0.378, 0.445] |
| align_at_or_through|5m|k6 | 0.624 | 0.595 | [0.583, 0.606] |
| align_through|5m|k6 | 0.572 | 0.524 | [0.510, 0.536] |
| align_at_or_through|15m|k6 | 0.659 | 0.614 | [0.603, 0.620] |
| align_through|15m|k6 | 0.567 | 0.501 | [0.491, 0.509] |
| align_at_or_through|4h|k6 | 0.630 | 0.665 | [0.643, 0.678] |
| align_through|4h|k6 | 0.344 | 0.438 | [0.414, 0.470] |
| sign_acf|5m|1 | 0.232 | 0.253 | [0.241, 0.263] |
| sign_acf|5m|2 | 0.196 | 0.233 | [0.226, 0.252] |
| sign_acf|5m|5 | 0.166 | 0.202 | [0.187, 0.224] |
| sign_acf|15m|1 | 0.298 | 0.319 | [0.296, 0.340] |
| sign_acf|15m|2 | 0.206 | 0.234 | [0.204, 0.269] |
| sign_acf|15m|5 | 0.123 | 0.146 | [0.112, 0.188] |
| sign_acf|4h|1 | 0.262 | 0.266 | [0.253, 0.278] |
| sign_acf|4h|2 | 0.222 | 0.099 | [0.083, 0.108] |
| sign_acf|4h|5 | 0.138 | 0.030 | [0.015, 0.063] |
| markout|5m|0 | 2.975 | 1.780 | [1.599, 2.039] |
| markout|15m|0 | 1.983 | 1.289 | [1.168, 1.365] |
| markout|4h|0 | 1.127 | 1.476 | [1.377, 1.750] |
| markout|5m|5 | 1.611 | 0.071 | [-0.083, 0.367] |
| markout|15m|5 | 0.634 | 0.333 | [0.226, 0.496] |
| markout|4h|5 | 0.476 | 1.448 | [1.328, 1.687] |
| markout|5m|30 | 0.527 | -1.355 | [-1.771, -1.035] |
| markout|15m|30 | 0.004 | -0.576 | [-0.799, -0.373] |
| markout|4h|30 | 0.017 | 1.396 | [1.292, 1.616] |
| markout|5m|60 | 0.111 | -1.704 | [-2.099, -1.192] |
| markout|15m|60 | 0.117 | -0.621 | [-0.846, -0.283] |
| markout|4h|60 | -0.087 | 1.342 | [1.280, 1.614] |
| taker_size_med|5m | 5.880 | 5.456 | [5.144, 5.954] |
| taker_size_p90|5m | 40.159 | 39.435 | [36.515, 40.434] |
| levels_per_tx|5m | 1.179 | 1.163 | [1.156, 1.169] |
| buy_share|5m | 0.498 | 0.505 | [0.496, 0.512] |
| taker_size_med|15m | 5.000 | 5.097 | [5.000, 5.222] |
| taker_size_p90|15m | 38.000 | 39.948 | [35.902, 42.047] |
| levels_per_tx|15m | 1.130 | 1.170 | [1.165, 1.179] |
| buy_share|15m | 0.519 | 0.513 | [0.504, 0.521] |
| taker_size_med|4h | 20.000 | 16.995 | [15.700, 19.621] |
| taker_size_p90|4h | 100.000 | 100.000 | [99.934, 100.000] |
| levels_per_tx|4h | 1.136 | 1.329 | [1.305, 1.354] |
| buy_share|4h | 0.541 | 0.536 | [0.518, 0.557] |
| mid|5m|slope_1 | 0.210 | 0.222 | [0.210, 0.231] |
| mid|5m|slope_5 | 0.505 | 0.465 | [0.453, 0.482] |
| mid|5m|slope_30 | 0.747 | 0.751 | [0.741, 0.767] |
| mid|5m|slope_120 | 0.838 | 0.812 | [0.797, 0.843] |
| mid|5m|level_slope | 0.839 | 0.842 | [0.826, 0.861] |
| mid|5m|sd_gap | 0.562 | 0.556 | [0.540, 0.567] |
| mid|5m|acf_5 | 0.759 | 0.751 | [0.735, 0.762] |
| mid|5m|acf_30 | 0.469 | 0.497 | [0.473, 0.530] |
| mid|5m|acf_120 | 0.189 | 0.155 | [0.125, 0.197] |
| mid|5m|p_change | 0.263 | 0.232 | [0.230, 0.235] |
| mid|5m|gapdp_30 | -0.037 | -0.059 | [-0.115, -0.004] |
| mid|5m|gapdy_30 | 0.391 | 0.405 | [0.339, 0.437] |
| mid|5m|gapdp_60 | -0.110 | -0.135 | [-0.258, -0.034] |
| mid|5m|gapdy_60 | 0.420 | 0.479 | [0.368, 0.564] |
| mid|15m|slope_1 | 0.173 | 0.184 | [0.170, 0.196] |
| mid|15m|slope_5 | 0.446 | 0.409 | [0.390, 0.426] |
| mid|15m|slope_30 | 0.746 | 0.744 | [0.727, 0.754] |
| mid|15m|slope_120 | 0.843 | 0.832 | [0.826, 0.845] |
| mid|15m|slope_300 | 0.864 | 0.856 | [0.844, 0.869] |
| mid|15m|level_slope | 0.871 | 0.861 | [0.844, 0.875] |
| mid|15m|sd_gap | 0.388 | 0.389 | [0.378, 0.405] |
| mid|15m|acf_5 | 0.814 | 0.809 | [0.795, 0.824] |
| mid|15m|acf_30 | 0.634 | 0.651 | [0.628, 0.675] |
| mid|15m|acf_120 | 0.434 | 0.447 | [0.403, 0.485] |
| mid|15m|p_change | 0.199 | 0.177 | [0.176, 0.179] |
| mid|15m|gapdp_30 | -0.027 | -0.028 | [-0.046, 0.010] |
| mid|15m|gapdy_30 | 0.292 | 0.298 | [0.276, 0.323] |
| mid|15m|gapdp_60 | -0.034 | -0.054 | [-0.101, 0.026] |
| mid|15m|gapdy_60 | 0.333 | 0.339 | [0.293, 0.383] |
| mid|4h|slope_1 | 0.125 | 0.104 | [0.097, 0.115] |
| mid|4h|slope_5 | 0.362 | 0.263 | [0.248, 0.274] |
| mid|4h|slope_30 | 0.595 | 0.560 | [0.549, 0.570] |
| mid|4h|slope_120 | 0.683 | 0.669 | [0.659, 0.681] |
| mid|4h|slope_300 | 0.721 | 0.710 | [0.686, 0.726] |
| mid|4h|slope_900 | 0.754 | 0.796 | [0.760, 0.825] |
| mid|4h|level_slope | 0.910 | 0.896 | [0.874, 0.925] |
| mid|4h|sd_gap | 0.436 | 0.447 | [0.425, 0.483] |
| mid|4h|acf_5 | 0.986 | 0.979 | [0.976, 0.983] |
| mid|4h|acf_30 | 0.954 | 0.946 | [0.939, 0.956] |
| mid|4h|acf_120 | 0.873 | 0.880 | [0.866, 0.899] |
| mid|4h|p_change | 0.083 | 0.136 | [0.135, 0.136] |
| mid|4h|gapdp_30 | -0.002 | -0.012 | [-0.017, -0.006] |
| mid|4h|gapdy_30 | 0.033 | 0.036 | [0.028, 0.043] |
| mid|4h|gapdp_60 | -0.009 | -0.024 | [-0.033, -0.011] |
| mid|4h|gapdy_60 | 0.046 | 0.044 | [0.034, 0.053] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -10396.2 [-12363.4, -9548.7] | 0.095 |
| fv | -2585.2 | -1057.0 [-2523.8, 2442.5] | 0.140 |
