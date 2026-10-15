# Calibrated synthetic-market generator (`sim/`)

This generator produces synthetic Polymarket crypto up/down markets for **pretraining and stress
tests** of the market-making policy, following `../eval/SIMULATOR.md`. It never produces a number of
record (EVALUATION.md).

It writes markets in exactly the on-disk format the C replay env reads:
* `ticks.bin`, `markets.bin`, `meta.json` and `splits.json`, as in `data/build`;
* `fills.parquet`, in the schema of `data/raw/trades.parquet`;
* `feat.bin`, `book.bin`, `fills.bin`, `fill_index.bin` and `feat_meta.json`, produced by
  `data/features.py` **unmodified** (see `features_sim.py`).

So `mm.h` replays synthetic markets unchanged: pass `data_dir=data/build_sim`.

**Status.** The consistency gate of SIMULATOR.md §5 **fails** (`VALIDATION.md`: C env, central fills,
L = 1 s, 20 replications on the train schedule; it also has the before/after table of the informed-taker
fix). Per that spec, use the sim only as an initialiser for pretraining, followed by replay fine-tuning,
and quote no sim result.

| gate | result after the informed-taker fix |
|---|---|
| G1 | sym_touch: the per-market median is inside the band and KS is 0.065, but the total overshoots (-3564 vs -2958) and the hourly KS is 0.18. fair_value: +425 [45, 829] vs real -797, hourly KS 0.25. |
| G2 | fails: the sim still ranks fair_value above no_quote |
| G3 | 4/22 components inside the band (was 1). sym_touch spread capture is 2.6x real. |
| G4 | the book passes (spread/depth/trades-per-market KS, one-sided 19/19), and so does sign ACF 3/3. It fails on 1 spread-median cell, 3 trade-rate cells and 2 P(buy\|gap) cells. |

## Quick start

```bash
source ../../.venv/bin/activate        # numpy, pandas, pyarrow, scipy (scipy was pip-installed for this)
cd explorations/deep_hedging_cvar/polymarket/sim
python fetch_binance.py --coins btc eth sol xrp --start 2026-07-23 --end 2026-08-21   # ~30 days, cached in cache/
python fit.py                          # train-only fit -> params.json + params_book.npz (~10 min)
python generate.py --n-markets 2000 --coins btc eth sol xrp --seed 1 --randomize      # -> ../data/build_sim/
python calibrate_env.py --grid refill && python calibrate_env.py --apply --tag refill   # C-env calibration of the flow
python fit.py --parts rate             # re-match mean trade counts after flow changes
python validate_env.py --reps 20 --chunk 5   # gate through the C env -> VALIDATION.md, validation_env.json
```

### `generate.py` options

| option | meaning |
|---|---|
| `--n-markets N` | number of markets (`slice`: rounded to whole 4 h slices) |
| `--lengths 5m 15m 4h` | contract lengths |
| `--length-weights` | random schedule only; the default is the train mix 248/146/37 |
| `--coins` | any of btc eth sol xrp bnb doge zec that are cached (`fetch_binance.py`). HYPE has no spot 1 s klines. |
| `--start/--end` | UTC days of cached Binance history to sample from. They must form one contiguous segment. |
| `--schedule random` | random coin, length and grid-aligned open time |
| `--schedule slice` | random UTC-aligned 4 h slices, with every market of every coin x length live in them. This gives real cross-coin correlation and concurrency. |
| `--schedule train --reps R` | the 431 train markets on their own Binance paths, R replications. Used by the gate. |
| `--seed` | everything is seeded |
| `--randomize / --no-randomize` | per-market domain randomisation (table below) |
| `--features / --no-features` | run `data/features.py` on the output (default on) |
| `--out` | output dir (default `../data/build_sim`) |

Every synthetic market:
* is split 0, so the env's train split plays it;
* carries flag bit 16 (`SYNTHETIC`);
* has a slug `sim<s seed>-<coin>-updown-<len>-<open>-<k>`, or `sim<seed>-r<rep>-<real slug>` for the train schedule.

`sim_markets.parquet` holds each market's draws: segment, strike, winner and every randomisation knob.

Throughput and size.
* 500 random markets (296 x 5m, 159 x 15m, 45 x 4h) with randomisation (the current `data/build_sim/`):
  - 0.88 M rows and 0.74 M fill legs;
  - 375 MB in total: ticks 141 MB, feat 155 MB, book 56 MB, fills.parquet 27 MB;
  - 53 s single-threaded: engine 20 s, data/features.py 25 s.
* The engine runs B same-length markets in lockstep. A 4h batch is 14,401 sequential numpy steps, so
  the cost is dominated by 4h markets: about 45 k rows/s for the engine.
* The C env replays the build like data/build (`load.verify` passes; baselines run through
  `runner.Runner` with `runner.DATA_DIR` set to it).
* `fit.py` takes about 10 min single-threaded:
  - book 10 s;
  - mid simulated moments 3.5 min;
  - flow Hawkes 1.5-3.5 min;
  - informed/impact grid 5 min;
  - rate correction 1 min.
* The gate takes 9 min for 20 replications.

## Leakage controls

* **Fitting.** Every parameter is fitted on the train split only: the 431 markets that open on
  2026-08-21..22, their ticks rows, their `trades.parquet` legs and their `markets.parquet` strikes.
  - The loaders in `common.py` assert split 0 and Binance < 2026-08-23. They go through
    `fair_value.load_underlying`, which asserts the same.
* **Binance history for generation** (`fetch_binance.py`).
  - Days 2026-08-22..25 (val and test plus a 1-day embargo either side) are never fetched or loaded.
    `load_closes` asserts this.
  - The default cache is 2026-07-23..08-21: 30 days x BTC/ETH/SOL/XRP, 80 MB.
* **Validation.** The gate is run on train only. The val day (G5) was not touched: a val evaluation
  needs a ledger entry (EVALUATION.md T4).

## Model (all in the Up frame; parameters in `params.json`, fitted by `fit.py`)

Row t is unix second t, as in `data/README.md`. Within second t, events happen in this order:
1. the regime, tick grid and update event;
2. taker prints against the actual book;
3. the captured snapshot is written to row t;
4. prints are stamped at block time t + d.

| component | model | fitted on train (value) |
|---|---|---|
| Underlying | Real Binance 1 s closes, causal (row t sees kline t-1) | -- |
| Strike, resolution | K = Binance TWAP60 at open. `price_to_beat` = K exp(coin basis). Up iff TWAP60_binance(end) exp(eps) >= K, with eps = 0 in the base sim and `sigma_basis` under randomisation. | Open basis per coin from -1.9 to -2.7 bp (HYPE -8.3). Drift sd 1.9 bp. Binance outcome agrees 415/417. |
| Fair value | `fair_value.model_at` imported. Student-t nu = 4, EWMA half-life 300 s on 30 s returns, closing-TWAP adjustment. | (study settings) |
| Mid target (logit) | s_fast logit p + (s_slow - s_fast) EWMA(logit p) + b(len, tau) + u + v. u is AR(1) disagreement; v is the impact of prints. | Simulated moments (horizon slopes 1-900 s, level slope, gap sd and ACF, P(change)), see below |
| Sticky book | The book re-aligns to the target only at update events, with probability pi(len, tau) per second. This is the captured snapshot. | pi ~ 0.27 (5m), 0.20 (15m), 0.06-0.18 (4h) |
| Tick | 0.01, then 0.001 for good once \|mid - 0.5\| > 0.48 | 215/431 markets switched |
| Regime | Normal, hollow (stub quotes 0.01-0.02 / 0.98-0.99) or one-sided (the extreme side empty). P(one-sided \| len, tau, ext(p)) and P(hollow \| len, coin group, tau). Persistence through per-market Gaussian-copula thresholds. | Hollow is a ZEC effect (69-100%, at most 3% elsewhere). Copula half-lives were fitted to the observed P(same state). |
| Spread | Empirical categorical by (len, coin group btc/eth/alt/zec, tau, ext(p), tick grid), backed off to coarser cells | Real spread support in mils |
| Depth | The whole top-5 bid/ask size vector is resampled from the real snapshots of the same (len, coin group, tau) cell, with total/top-5 depth ratio. The previous vector is kept with probability 1 - (1 - acf1)/pi. Level gaps are empirical. | Library of 3000 vectors per cell (`params_book.npz`) |
| Taker arrivals | Linear Hawkes: lambda = A_m exp(cell(len, tau) + b_gap \|p - mid\| + b_z \|z5\| + b_O O) + eta H_t, with H an exponential kernel. A_m ~ lognormal(len, coin). No flow in hollow books. | eta = 0.43, tau = 5 s (LR 17,239 vs Poisson). b_gap 1.37, b_z 0.28, b_O -2.3. Market sd 0.90. Simulated mean-rate correction per length. |
| Direction | Noise takers follow the fitted P(buy \| gap bucket), plus a copy of the previous sign with probability rho. **Look-ahead informed takers**: each arrival is informed with probability 1 - exp(-kappa_I edge), where edge = (mid target h s ahead, Binance-driven part only) minus the actual touch, net of the taker fee, in cents. An informed arrival takes that side. The older "staleness" informed share w is set to 0. | P(buy) 0.40 -> 0.57/0.60/0.59 by gap. h = 5 s; kappa_I 0.5/3/3 per cent; rho 0/0.10/0.25 (5m/15m/4h). From the C-env calibration (`calibrate_env.py`). |
| Sizes, sweep | Taker size from empirical quantiles by (len, coin group). It sweeps the **actual** book: the snapshot shape translated to the current mid target. A level a print exhausts is re-posted at the same price with probability r_refill (fresh library size), else removed. Each level gives geometric(mean 1.35) maker legs. | Median 10 (BTC 5m), 0.02 (alt 5m), 15-20 (4h); r_refill = 0.8 |
| Impact | Each print adds kappa sign sqrt(min(size, 500)/10) to an accumulator (half-life 300 s). The target absorbs it with a 15 s half-life. | kappa 0.01/0.01/0.02 (half the earlier simulated-moment values, from the C-env calibration) |
| Legs, tokens | The taker leg is on Up with probability 0.73 (buy) / 0.30 (sell). Maker legs are on Up with probability 0.13 (buy) / 0.89 (sell). Down legs are priced 1 - p. | Train legs |
| Block time | Stamp = match + d, d in {3, 4} with probability 1/2 each. Prints after the end appear only in `fills.parquet`, as in the real data. | COVERAGE.md. A fitted delay mixture could not reproduce the real alignment, see VALIDATION.md. |
| Fees, rewards | taker 0.07 p(1-p), maker rebate 20%, min order size 5. Rewards (min size, max spread) are drawn from the train joint distribution by length. | markets.parquet |

Mid simulated-moment fit (5m / 15m / 4h):
* s_fast 0.86 / 0.88 / 0.72;
* s_slow 0.89 / 0.75 / 1.05;
* slow half-life 3500 / 845 / 1667 s;
* u half-life 38 / 116 / 887 s;
* sigma_u 0.075 / 0.028 / 0.016.

At these values the displayed mid reproduces the real slopes of Δmid on Δp:
* 1 s: 0.21 / 0.17 / 0.13;
* 30 s: 0.75 / 0.75 / 0.60;
* levels: 0.84 / 0.87 / 0.91.

The level slope corresponds to the 73-85% same-second pass-through of REPORT §3 measured on snapshot pairs.

Four features beyond SIMULATOR.md came out of the validation. All are documented in VALIDATION.md.
* **Look-ahead informed takers** (see Direction above). Real takers adversely select fair-value quotes
  much harder than takers who only see the 1 s Binance kline.
* **Touch resilience** (refill 0.8). BTC/ETH books absorb about 8 prints/s without the touch moving.
* **Captured vs actual book.** Real snapshots are sparse: book_age p90 12-31 s, and only about 30%
  of seconds show a change even with 1.4 prints/s on 5m. Prints therefore land far from the captured
  touch. Prints execute against the actual book (the captured shape shifted to the current target),
  while ticks.bin shows the snapshot from the last update. Without this, prints line up with the
  snapshot far better than real prints do, and makers are picked off far less.
* **Price impact with a 15 s rise.** Real maker markouts of all prints decay from +3.0c to +0.1c over
  60 s (5m). The fair-value gap alone cannot produce that, consistent with REPORT §3 ("the gap barely
  predicts the next mid move").

Not implemented (SIMULATOR.md items left for later):
* competitor re-pricing in reaction to our quotes;
* the 8-coin joint *portfolio* env;
* crossed or stale capture artefacts as a stress mode. The sim has `crossed_raw = book_from_down = 0` and `book_valid = 1`.
* hour-of-day flow effects (only 2 train days);
* fitting flow on months of on-chain fills (only the 2 train days of fills are used);
* stratified vol-regime sampling.

## Domain randomisation (`--randomize`, per market; `params.json` "randomization")

| knob | range |
|---|---|
| extra mid lag | 0-3 s (integer) |
| fast pass-through s_fast | x0.7-1.15 (capped so s_fast <= 1) |
| slow-component half-life | x0.5-2 (log-uniform) |
| disagreement sigma_u | x0.5-2 (log) |
| book update (refresh) rate | x0.5-2 (log) |
| taker intensity | x0.5-2 (log) |
| taker size | x0.5-2 (log) |
| informed share w | 0.10-0.60 |
| Hawkes branching ratio | 0-0.8, with the baseline rescaled so the mean rate is unchanged |
| print impact kappa | x0.5-2 (log) |
| depth | x0.3-3 (log) |
| spread | x0.5-2 (log), rounded to the tick, at least 1 tick |
| P(one-sided), P(hollow) | +-0.5 on the logit |
| taker fee rate | x0.5-1.5; fee-free with probability 0.1 |
| maker rebate share | x0.5-1.5 |
| settlement basis sd | 0-2 x 1.9 bp |

## Files

| file | role |
|---|---|
| `common.py` | format constants imported from `data/build.py`; the pricer from `analysis/fair_value`; train-only loaders; bucket definitions |
| `fetch_binance.py` | caches Binance 1 s closes (excludes 08-22..25); `load_closes` returns contiguous segments |
| `fit.py` | train-only fit: `--parts settlement book mid flow informed rate fees`. `--pre-trades START END` also pools on-chain pre-period trades (onchain/, days <= 2026-08-20) into the book-free flow parameters: sizes, tokens, legs, sweep depth. |
| `params.json`, `params_book.npz` | fitted parameters and the empirical size library |
| `engine.py` | the vectorised batch engine, running B same-length markets in lockstep per second; fill legs |
| `generate.py` | schedules, pricing, writes the build, calls `features_sim.py` |
| `features_sim.py` | runs `data/features.py` unmodified on a sim build by repointing its module globals |
| `metrics.py` | statistics shared by fit and validation (book, trades, alignment, markouts, mid moments, numpy fill check) |
| `validate_env.py` | **the gate**: no_quote / sym_touch / fair_value (baselines DEFAULTS) through the C env, real vs sim train. Writes `VALIDATION.md` and `validation_env.json`. |
| `validate.py` | fast numpy-only variant. Its `evaluate` provides G4. Writes to `cache/`. |
| `calibrate_env.py` | Calibrates look-ahead informed takers, refill, impact and copy probability **through the C env** on one train-schedule replication per candidate. The loss covers the baselines' markout curves, fill volume, capture and totals, print alignment, sign ACF and trade counts. `--grid NAME`, then `--apply`; log in `cache/calib_env.jsonl`. |
| `cache/` | git-ignored Binance cache, logs and gate scratch |
