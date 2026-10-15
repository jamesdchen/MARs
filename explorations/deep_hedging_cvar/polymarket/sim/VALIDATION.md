# Generator validation: sim vs replay through the C env (train days)

`sim/validate_env.py --reps 20 --chunk 5 --fill-model 1 --latency 1` ran in 9 min. params.json sha256 `1b893e1dfb49b2ab`, mm.h sha256 `9189f83adbb3f42a`.

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
| sym_touch | -2957.6 | -3010.6 [-3311.9, -2791.2] | -3.010 | [-2.634, -1.645] | 0.104 | 0.085 | FAIL |
| fair_value | -796.6 | -13.6 [-439.5, 291.9] | 0.000 | [0.000, 0.000] | 0.198 | 0.105 | FAIL |
| no_quote | 0.0 | 0.0 | 0 | 0 | | | PASS |

P&L by contract length (total $):

| policy | len | real | sim median [5%, 95%] |
|---|---|---|---|
| sym_touch | 5m | -1911.5 | -2065.1 [-2293.7, -1785.4] |
| sym_touch | 15m | -952.6 | -957.5 [-1162.0, -829.6] |
| sym_touch | 4h | -93.5 | -28.7 [-97.8, 39.7] |
| fair_value | 5m | -553.0 | -110.1 [-444.9, 129.4] |
| fair_value | 15m | -336.2 | -73.4 [-203.4, 152.7] |
| fair_value | 4h | 92.7 | 117.7 [57.3, 181.5] |

## G3: Log attribution per market ($, means over markets) and markout curve

| policy | component | real | sim median [5%, 95%] | inside |
|---|---|---|---|---|
| sym_touch | fills | 86.701 | 95.299 [89.453, 100.478] | False |
| sym_touch | fill_vol | 164.380 | 198.966 [187.290, 218.321] | False |
| sym_touch | spread_capture | 0.675 | 1.516 [1.382, 1.666] | False |
| sym_touch | markout_5 | -4.763 | -4.660 [-4.865, -4.357] | True |
| sym_touch | markout_30 | -7.265 | -8.460 [-8.963, -7.815] | False |
| sym_touch | markout_60 | -7.982 | -8.824 [-9.406, -8.149] | False |
| sym_touch | inventory_pnl | 0.037 | -0.218 [-0.774, 0.230] | True |
| sym_touch | rebates | 0.407 | 0.447 [0.416, 0.491] | False |
| sym_touch | markout 5s (c/share) | -2.898 | -2.341 [-2.423, -2.123] | False |
| sym_touch | markout 30s (c/share) | -4.420 | -4.171 [-4.519, -3.876] | True |
| sym_touch | markout 60s (c/share) | -4.856 | -4.466 [-4.886, -4.062] | True |
| fair_value | fills | 49.930 | 42.374 [39.564, 44.172] | False |
| fair_value | fill_vol | 85.714 | 90.259 [86.588, 97.884] | False |
| fair_value | spread_capture | 1.354 | 1.635 [1.540, 1.770] | False |
| fair_value | markout_5 | -2.611 | -1.763 [-1.925, -1.652] | False |
| fair_value | markout_30 | -3.815 | -3.034 [-3.362, -2.783] | False |
| fair_value | markout_60 | -4.047 | -2.855 [-3.602, -2.474] | False |
| fair_value | inventory_pnl | 0.629 | 1.199 [0.111, 1.680] | True |
| fair_value | rebates | 0.216 | 0.196 [0.187, 0.213] | False |
| fair_value | markout 5s (c/share) | -3.046 | -1.955 [-2.059, -1.823] | False |
| fair_value | markout 30s (c/share) | -4.451 | -3.326 [-3.729, -3.039] | False |
| fair_value | markout 60s (c/share) | -4.721 | -3.115 [-4.005, -2.678] | False |

## G4: stylised facts

| check | result |
|---|---|
| spread_median_by_tau | FAIL (n_fail=2, n=19) |
| onesided_by_tau | PASS (n_fail=0, n=19) |
| trade_rate_by_tau | FAIL (n_fail=1, n=19) |
| ks_spread|5m | PASS (D=0.040) |
| ks_spread|15m | PASS (D=0.014) |
| ks_spread|4h | PASS (D=0.044) |
| ks_log_top_sz|5m | PASS (D=0.028) |
| ks_log_top_sz|15m | PASS (D=0.012) |
| ks_log_top_sz|4h | PASS (D=0.005) |
| ks_log_top5|5m | PASS (D=0.034) |
| ks_log_top5|15m | PASS (D=0.031) |
| ks_log_top5|4h | PASS (D=0.050) |
| ks_trades_per_market|5m | FAIL (D=0.155) |
| ks_trades_per_market|15m | FAIL (D=0.153) |
| ks_trades_per_market|4h | PASS (D=0.124) |
| pbuy_gap_curve | FAIL (n_fail=2, n=29) |
| sign_acf1|5m | PASS (real=0.232, sim=0.224) |
| sign_acf1|15m | PASS (real=0.298, sim=0.327) |
| sign_acf1|4h | PASS (real=0.262, sim=0.258) |

### Book and trade rate by length and time-to-close bucket (sim = median over replications)

tau bucket edges (s): [0, 10, 30, 60, 120, 300, 900, 3600, 14401]

| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |
|---|---|---|---|---|---|---|---|---|---|
| 5m | 0 | 0.913/0.907 | 0.024/0.008 | 0.010/0.009 | 0.56/0.78 | 50/50 | 250/141 | 0.32/0.26 | 0.388/0.385 |
| 5m | 1 | 0.845/0.840 | 0.027/0.020 | 0.020/0.010 | 0.47/0.65 | 50/50 | 255/177 | 0.29/0.24 | 0.597/0.654 |
| 5m | 2 | 0.608/0.604 | 0.088/0.059 | 0.020/0.010 | 0.47/0.52 | 50/50 | 260/250 | 0.29/0.24 | 1.424/1.394 |
| 5m | 3 | 0.221/0.223 | 0.103/0.104 | 0.020/0.020 | 0.44/0.47 | 50/50 | 300/280 | 0.32/0.27 | 1.434/1.369 |
| 5m | 4 | 0.013/0.012 | 0.111/0.130 | 0.020/0.020 | 0.45/0.45 | 50/50 | 322/316 | 0.32/0.26 | 1.652/1.538 |
| 15m | 0 | 0.899/0.875 | 0.067/0.011 | 0.010/0.009 | 0.61/0.80 | 50/50 | 185/185 | 0.30/0.22 | 0.028/0.018 |
| 15m | 1 | 0.834/0.815 | 0.085/0.019 | 0.010/0.009 | 0.58/0.75 | 50/50 | 250/208 | 0.25/0.18 | 0.071/0.063 |
| 15m | 2 | 0.700/0.693 | 0.085/0.038 | 0.020/0.010 | 0.46/0.54 | 50/50 | 250/250 | 0.23/0.18 | 0.221/0.176 |
| 15m | 3 | 0.450/0.443 | 0.056/0.057 | 0.019/0.020 | 0.48/0.46 | 50/50 | 317/299 | 0.26/0.19 | 0.311/0.275 |
| 15m | 4 | 0.175/0.169 | 0.067/0.079 | 0.020/0.020 | 0.43/0.43 | 50/50 | 427/379 | 0.29/0.21 | 0.381/0.340 |
| 15m | 5 | 0.006/0.007 | 0.092/0.100 | 0.020/0.020 | 0.42/0.42 | 50/50 | 400/390 | 0.29/0.21 | 0.359/0.327 |
| 4h | 0 | 0.700/0.676 | 0.081/0.027 | 0.008/0.009 | 0.75/0.80 | 50/75 | 265/238 | 0.07/0.07 | 0.065/0.055 |
| 4h | 1 | 0.672/0.637 | 0.081/0.027 | 0.009/0.009 | 0.77/0.77 | 55/61 | 315/240 | 0.11/0.11 | 0.030/0.030 |
| 4h | 2 | 0.532/0.522 | 0.081/0.054 | 0.009/0.009 | 0.66/0.71 | 100/56 | 315/213 | 0.07/0.07 | 0.048/0.046 |
| 4h | 3 | 0.447/0.421 | 0.084/0.054 | 0.009/0.009 | 0.61/0.68 | 103/55 | 294/247 | 0.09/0.10 | 0.034/0.043 |
| 4h | 4 | 0.203/0.188 | 0.095/0.072 | 0.009/0.009 | 0.63/0.64 | 55/54 | 316/260 | 0.06/0.06 | 0.023/0.022 |
| 4h | 5 | 0.093/0.088 | 0.084/0.082 | 0.009/0.009 | 0.58/0.59 | 53/53 | 320/256 | 0.09/0.09 | 0.022/0.022 |
| 4h | 6 | 0.047/0.052 | 0.081/0.081 | 0.011/0.011 | 0.49/0.49 | 50/52 | 338/302 | 0.10/0.10 | 0.015/0.015 |
| 4h | 7 | 0.000/0.000 | 0.081/0.081 | 0.020/0.020 | 0.39/0.38 | 50/50 | 400/400 | 0.18/0.18 | 0.011/0.012 |

### P(taker buys Up | gap = p_model - mid at block time - 3 s)

gap bucket edges: [-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0]

| len / bucket | n real | real | sim |
|---|---|---|---|
| 5m|0 | 13823 | 0.398 | 0.415 |
| 5m|1 | 12459 | 0.474 | 0.478 |
| 5m|2 | 12977 | 0.458 | 0.481 |
| 5m|3 | 6481 | 0.457 | 0.503 |
| 5m|4 | 7216 | 0.513 | 0.603 |
| 5m|5 | 8577 | 0.504 | 0.404 |
| 5m|6 | 6910 | 0.538 | 0.492 |
| 5m|7 | 13334 | 0.536 | 0.523 |
| 5m|8 | 11837 | 0.541 | 0.532 |
| 5m|9 | 13657 | 0.568 | 0.569 |
| 15m|0 | 2616 | 0.426 | 0.444 |
| 15m|1 | 4776 | 0.460 | 0.463 |
| 15m|2 | 7510 | 0.489 | 0.479 |
| 15m|3 | 3528 | 0.460 | 0.472 |
| 15m|4 | 3913 | 0.496 | 0.537 |
| 15m|5 | 4166 | 0.528 | 0.492 |
| 15m|6 | 3528 | 0.556 | 0.538 |
| 15m|7 | 7364 | 0.565 | 0.547 |
| 15m|8 | 5177 | 0.583 | 0.572 |
| 15m|9 | 2607 | 0.600 | 0.598 |
| 4h|1 | 455 | 0.433 | 0.441 |
| 4h|2 | 545 | 0.424 | 0.423 |
| 4h|3 | 363 | 0.424 | 0.411 |
| 4h|4 | 532 | 0.451 | 0.446 |
| 4h|5 | 745 | 0.541 | 0.533 |
| 4h|6 | 573 | 0.635 | 0.611 |
| 4h|7 | 1301 | 0.587 | 0.579 |
| 4h|8 | 1147 | 0.616 | 0.602 |
| 4h|9 | 655 | 0.594 | 0.599 |

### Other statistics (reported, not gated; sim median and 5-95% band)

| stat | real | sim median | sim [5%, 95%] |
|---|---|---|---|
| align_at_or_through|5m|k0 | 0.544 | 0.602 | [0.588, 0.616] |
| align_through|5m|k0 | 0.446 | 0.464 | [0.450, 0.485] |
| align_at_or_through|15m|k0 | 0.561 | 0.610 | [0.595, 0.637] |
| align_through|15m|k0 | 0.407 | 0.418 | [0.398, 0.442] |
| align_at_or_through|4h|k0 | 0.498 | 0.694 | [0.670, 0.718] |
| align_through|4h|k0 | 0.206 | 0.433 | [0.411, 0.458] |
| align_at_or_through|5m|k1 | 0.561 | 0.630 | [0.616, 0.644] |
| align_through|5m|k1 | 0.453 | 0.471 | [0.456, 0.494] |
| align_at_or_through|15m|k1 | 0.587 | 0.639 | [0.624, 0.664] |
| align_through|15m|k1 | 0.424 | 0.432 | [0.410, 0.453] |
| align_at_or_through|4h|k1 | 0.537 | 0.705 | [0.682, 0.728] |
| align_through|4h|k1 | 0.225 | 0.430 | [0.405, 0.453] |
| align_at_or_through|5m|k2 | 0.590 | 0.657 | [0.644, 0.671] |
| align_through|5m|k2 | 0.469 | 0.477 | [0.462, 0.501] |
| align_at_or_through|15m|k2 | 0.621 | 0.664 | [0.653, 0.685] |
| align_through|15m|k2 | 0.453 | 0.440 | [0.419, 0.461] |
| align_at_or_through|4h|k2 | 0.577 | 0.718 | [0.696, 0.739] |
| align_through|4h|k2 | 0.247 | 0.426 | [0.399, 0.453] |
| align_at_or_through|5m|k3 | 0.626 | 0.679 | [0.668, 0.693] |
| align_through|5m|k3 | 0.511 | 0.475 | [0.463, 0.501] |
| align_at_or_through|15m|k3 | 0.665 | 0.681 | [0.673, 0.701] |
| align_through|15m|k3 | 0.507 | 0.447 | [0.429, 0.465] |
| align_at_or_through|4h|k3 | 0.628 | 0.731 | [0.715, 0.751] |
| align_through|4h|k3 | 0.287 | 0.419 | [0.392, 0.444] |
| align_at_or_through|5m|k4 | 0.643 | 0.653 | [0.643, 0.669] |
| align_through|5m|k4 | 0.558 | 0.495 | [0.483, 0.521] |
| align_at_or_through|15m|k4 | 0.682 | 0.663 | [0.651, 0.682] |
| align_through|15m|k4 | 0.557 | 0.471 | [0.454, 0.485] |
| align_at_or_through|4h|k4 | 0.641 | 0.723 | [0.707, 0.739] |
| align_through|4h|k4 | 0.317 | 0.426 | [0.399, 0.449] |
| align_at_or_through|5m|k6 | 0.624 | 0.593 | [0.584, 0.611] |
| align_through|5m|k6 | 0.572 | 0.524 | [0.513, 0.546] |
| align_at_or_through|15m|k6 | 0.659 | 0.611 | [0.598, 0.632] |
| align_through|15m|k6 | 0.567 | 0.503 | [0.491, 0.522] |
| align_at_or_through|4h|k6 | 0.630 | 0.676 | [0.659, 0.696] |
| align_through|4h|k6 | 0.344 | 0.453 | [0.428, 0.472] |
| sign_acf|5m|1 | 0.232 | 0.224 | [0.214, 0.236] |
| sign_acf|5m|2 | 0.196 | 0.213 | [0.200, 0.229] |
| sign_acf|5m|5 | 0.166 | 0.190 | [0.179, 0.207] |
| sign_acf|15m|1 | 0.298 | 0.327 | [0.305, 0.343] |
| sign_acf|15m|2 | 0.206 | 0.280 | [0.260, 0.301] |
| sign_acf|15m|5 | 0.123 | 0.201 | [0.174, 0.224] |
| sign_acf|4h|1 | 0.262 | 0.258 | [0.240, 0.266] |
| sign_acf|4h|2 | 0.222 | 0.116 | [0.082, 0.133] |
| sign_acf|4h|5 | 0.138 | 0.030 | [0.019, 0.056] |
| markout|5m|0 | 2.975 | 1.779 | [1.600, 2.038] |
| markout|15m|0 | 1.983 | 1.342 | [1.228, 1.601] |
| markout|4h|0 | 1.127 | 1.558 | [1.422, 1.694] |
| markout|5m|5 | 1.611 | 0.272 | [0.168, 0.638] |
| markout|15m|5 | 0.634 | 0.407 | [0.248, 0.678] |
| markout|4h|5 | 0.476 | 1.509 | [1.399, 1.642] |
| markout|5m|30 | 0.527 | -1.550 | [-1.963, -1.314] |
| markout|15m|30 | 0.004 | -1.175 | [-1.342, -0.943] |
| markout|4h|30 | 0.017 | 1.426 | [1.293, 1.532] |
| markout|5m|60 | 0.111 | -1.937 | [-2.339, -1.512] |
| markout|15m|60 | 0.117 | -1.094 | [-1.403, -0.887] |
| markout|4h|60 | -0.087 | 1.394 | [1.214, 1.568] |
| taker_size_med|5m | 5.880 | 5.416 | [5.141, 5.945] |
| taker_size_p90|5m | 40.159 | 38.988 | [36.422, 40.510] |
| levels_per_tx|5m | 1.179 | 1.162 | [1.157, 1.168] |
| buy_share|5m | 0.498 | 0.504 | [0.499, 0.511] |
| taker_size_med|15m | 5.000 | 5.074 | [5.000, 5.914] |
| taker_size_p90|15m | 38.000 | 39.937 | [36.138, 43.947] |
| levels_per_tx|15m | 1.130 | 1.173 | [1.168, 1.181] |
| buy_share|15m | 0.519 | 0.516 | [0.509, 0.520] |
| taker_size_med|4h | 20.000 | 17.119 | [16.046, 19.286] |
| taker_size_p90|4h | 100.000 | 100.000 | [99.158, 100.000] |
| levels_per_tx|4h | 1.136 | 1.333 | [1.315, 1.353] |
| buy_share|4h | 0.541 | 0.530 | [0.510, 0.539] |
| mid|5m|slope_1 | 0.210 | 0.222 | [0.211, 0.231] |
| mid|5m|slope_5 | 0.505 | 0.466 | [0.455, 0.483] |
| mid|5m|slope_30 | 0.747 | 0.751 | [0.742, 0.767] |
| mid|5m|slope_120 | 0.838 | 0.813 | [0.797, 0.844] |
| mid|5m|level_slope | 0.839 | 0.843 | [0.825, 0.860] |
| mid|5m|sd_gap | 0.562 | 0.554 | [0.539, 0.568] |
| mid|5m|acf_5 | 0.759 | 0.750 | [0.736, 0.762] |
| mid|5m|acf_30 | 0.469 | 0.496 | [0.473, 0.531] |
| mid|5m|acf_120 | 0.189 | 0.157 | [0.122, 0.197] |
| mid|5m|p_change | 0.263 | 0.232 | [0.230, 0.235] |
| mid|5m|gapdp_30 | -0.037 | -0.058 | [-0.118, -0.004] |
| mid|5m|gapdy_30 | 0.391 | 0.404 | [0.337, 0.442] |
| mid|5m|gapdp_60 | -0.110 | -0.135 | [-0.257, -0.038] |
| mid|5m|gapdy_60 | 0.420 | 0.476 | [0.367, 0.564] |
| mid|15m|slope_1 | 0.173 | 0.180 | [0.171, 0.189] |
| mid|15m|slope_5 | 0.446 | 0.405 | [0.394, 0.421] |
| mid|15m|slope_30 | 0.746 | 0.743 | [0.733, 0.754] |
| mid|15m|slope_120 | 0.843 | 0.834 | [0.825, 0.847] |
| mid|15m|slope_300 | 0.864 | 0.852 | [0.842, 0.867] |
| mid|15m|level_slope | 0.871 | 0.860 | [0.850, 0.871] |
| mid|15m|sd_gap | 0.388 | 0.393 | [0.382, 0.401] |
| mid|15m|acf_5 | 0.814 | 0.813 | [0.795, 0.824] |
| mid|15m|acf_30 | 0.634 | 0.658 | [0.632, 0.677] |
| mid|15m|acf_120 | 0.434 | 0.452 | [0.420, 0.496] |
| mid|15m|p_change | 0.199 | 0.178 | [0.176, 0.179] |
| mid|15m|gapdp_30 | -0.027 | -0.030 | [-0.055, 0.017] |
| mid|15m|gapdy_30 | 0.292 | 0.301 | [0.265, 0.322] |
| mid|15m|gapdp_60 | -0.034 | -0.034 | [-0.110, 0.018] |
| mid|15m|gapdy_60 | 0.333 | 0.336 | [0.280, 0.384] |
| mid|4h|slope_1 | 0.125 | 0.103 | [0.092, 0.114] |
| mid|4h|slope_5 | 0.362 | 0.259 | [0.241, 0.272] |
| mid|4h|slope_30 | 0.595 | 0.559 | [0.547, 0.570] |
| mid|4h|slope_120 | 0.683 | 0.672 | [0.657, 0.682] |
| mid|4h|slope_300 | 0.721 | 0.709 | [0.695, 0.725] |
| mid|4h|slope_900 | 0.754 | 0.783 | [0.762, 0.806] |
| mid|4h|level_slope | 0.910 | 0.896 | [0.873, 0.924] |
| mid|4h|sd_gap | 0.436 | 0.454 | [0.428, 0.474] |
| mid|4h|acf_5 | 0.986 | 0.980 | [0.975, 0.982] |
| mid|4h|acf_30 | 0.954 | 0.949 | [0.939, 0.953] |
| mid|4h|acf_120 | 0.873 | 0.884 | [0.868, 0.893] |
| mid|4h|p_change | 0.083 | 0.136 | [0.135, 0.137] |
| mid|4h|gapdp_30 | -0.002 | -0.012 | [-0.017, -0.008] |
| mid|4h|gapdy_30 | 0.033 | 0.035 | [0.032, 0.043] |
| mid|4h|gapdp_60 | -0.009 | -0.024 | [-0.034, -0.015] |
| mid|4h|gapdy_60 | 0.046 | 0.042 | [0.036, 0.050] |

### Numpy fill check (secondary cross-check, validate.py rules)

| rule | real total | sim total median [5%, 95%] | KS hourly |
|---|---|---|---|
| touch | -10327.9 | -9607.3 [-11448.2, -8218.3] | 0.102 |
| fv | -2585.2 | -787.0 [-3102.5, 490.8] | 0.155 |
