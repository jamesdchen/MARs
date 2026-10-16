# Market-making replay env v3 (`mm3.h`): pre-game sports and election reward farming

**Build** (once per machine and after every change to `mm3.h` / `binding3.c`; the `.so` is git-ignored):
`../../.venv/bin/python v3env/setup.py build_ext --inplace` (from `polymarket/`; works from any directory).

v3 of the replay env for protocol v3 (DECISIONS.md D11): track S (pre-game sports, listing to
kickoff) and track E (US-midterm reward farming). It is `../mm.h` (v2, the frozen crypto env) with
seven extensions (the seventh is the D16 recording-gap rule, section 11a). **With every new option at its default and no v3 side files in the data directory,
mm3 is v2 bit for bit**: same observations 0..49, rewards, terminals, Log and per-episode P&L
(section 12a). `../ENV.md` documents everything inherited: time and causality, match delay, the
three fill models, quote construction, accounting, the T5 flags and the v2 observation.

`../mm.h`, `../binding.c`, `../env.py`, `../reference.py` and `../test_env.py` are untouched (mm.h
sha256 `9189f83a…` stays frozen for the crypto v2 test). Every v3 name is prefixed (`m3_`, `M3_`,
`Mm3`), so both cores can be included in one translation unit, and the PufferLib 3.0 extension is
called `binding3`, so `env.py` and `env3.py` import side by side in one process.

| file | role |
|---|---|
| `mm3.h` | the core: pure C, no PufferLib includes, compiles as C99 and C++17 |
| `binding3.c`, `setup.py` | PufferLib 3.0 native binding, extension `binding3` (`python v3env/setup.py build_ext --inplace`; `env3.py` warns when the `.so` is older than its sources) |
| `env3.py` | `MmEnv3(pufferlib.PufferEnv)` (alias `MmEnv`), MultiDiscrete `[7, 7, 3]`, Box `(obs_size,)` |
| `reference3.py` | readable numpy reference of the same model (tests only) |
| `adapt3.py` | makes `build_sports.py` (sports-1, sports-2 and sports-3) / `build_elections.py` outputs loadable (writes `markets3.bin` with the D16 gap links, `x3.bin`, `meta3.json` with the gap summary) |
| `fixtures3.py` | synthetic sports (sports-2, its --no-legacy-settle and --bucket-segments variants, sports-1, and the sports-3 gap build with its continuous twin) / election builds (plain and with gaps) in the builders' schemas, and the reward unit market |
| `test_env3.py` | tests (section 12); `harness3.c` (ctypes harness), `cmp_harness.c` (v2 and v3 side by side) |
| `bench3.py` | throughput against v2 (section 13) |
| `puffer5/polymarket_mm3.h`, `puffer5/test_wrapper3.c`, `PUFFER5_V3.md` | PufferLib 5.0 wrapper, its test and port notes |

## 1. What v3 adds

| # | extension | default (= v2) |
|---|---|---|
| 1 | Episode length from the data: an episode is a market's `n_rows - 1` rows, any length. Length bit 8 and `dt_other` for lengths other than 300 / 900 / 14400 rows. Optional variable cadence `dt_far` / `far_s`. | no market has another length in crypto data |
| 2 | Float settlement per market (`markets3.bin` `settle_mode`, `settle_price`): 0 winner_up, 1 a stored price (sports: kickoff mid), 2 the mark at the last row (unresolved elections). `flat_c` flattening cost; `pull_s` quotes forced off before the event; the close-inventory shaping (`close_inv_coef`, `close_ramp`, `close_ramp_s`) moved into the core. | mode 0, all off |
| 3 | Liquidity-reward hook with per-market or per-row max spread v, min size, daily rate and competitor Q (`markets3.bin`, `x3.bin`); the 1.5 c / 50 constants survive only as the legacy values for data without v3 side files. | `liq_rewards = 0` |
| 4 | `size_mode 1`: size heads = `size_rel{0,1,2}` (1 / 2 / 4) x max(reward min size, min_order_size). | `size_mode 0`: 1 / 2 / 5 x min_order_size |
| 5 | `obs_size > 50` appends 28 slots (time to event, category, reward features). Obs 0..49 are v2's. | `obs_size = 50` |
| 6 | Fee and rebate coefficients per category (`fee_rate_<cat>`, `rebate_rate_<cat>`) and per market (`markets3.bin`). Fill models, latency, match delay and the rebate formula are unchanged. | `-1`: markets.bin |
| 7 | D16 recording gaps (section 11a): gap links and flags in `markets3.bin`; a piece with a continuation settles at its last pre-gap mid without `flat_c`; `gap_chain = 1` carries inventory and cash into the next piece (one continuous episode, quotes forced off across the gap); obs 78 `gap_recent`; obs 51 and the close ramp of a gap-cut piece use the time to the event. | no links or flags: nothing changes; `gap_chain = 0` |

## 2. Data

mm3 reads the five v2 files (`markets.bin`, `feat.bin` [N, 44], `book.bin`, `fills.bin`,
`fill_index.bin`; layouts in `../ENV.md` section 1) and two optional side files in the same
directory, written by `adapt3.py`:

### `markets3.bin`: 64 B per market (`Mm3Ext`), same order as `markets.bin`

| offset | field | type | meaning |
|---|---|---|---|
| 0 | `settle_price` | double | settle_mode 1: value of one Up share at the end of the episode, in [0, 1] |
| 8 | `event_ts` | int64 | unix s of the event the episode runs to: kickoff (sports), window end (elections), market end (crypto). <= 0: `end_ts` |
| 16 | `settle_mode` | int32 | 0 winner_up, 1 settle_price, 2 `feat.mark` at the last row |
| 20 | `category` | int32 | 0 crypto, 1 sports, 2 elections, 3 other |
| 24 | `row_s` | float | seconds per row (the builder's grid step) |
| 28 | `rw_v_c` | float | reward max spread v, cents from the adjusted mid (0 = no program) |
| 32 | `rw_min_size` | float | reward min size, shares per order |
| 36 | `rw_rate_day` | float | reward pool, USD per day (observation; the paid pool is `liq_pi`) |
| 40 | `fee_rate`, `fee_exp`, `rebate_rate` | float x 3 | per-market overrides; < 0: markets.bin |
| 52 | `flags3` | int32 | 1 LEGACY (synthesised for v2 data); D16: 2 GAP_CUT, 4 AFTER_GAP, 8 EVAL_EXCL (last-hour gap: not scored on val / test), 16 DAY_INELIGIBLE (train only) |
| 56 | `gap_next` | int32 | D16: index + 1 of the piece after the gap, 0 = none (was padding, 0 in every earlier file) |
| 60 | `gap_prev` | int32 | D16: index + 1 of the piece before the gap, 0 = none |

### `x3.bin`: float32 [N, 8], per row (optional)

`rw_v_c`, `rw_min_size`, `rw_rate_day`, `rw_in_program`, `liq_mid`, `liq_comp`, `liq_pi`, pad. When
present, these replace the per-market program of `markets3.bin` and feat.bin's columns 38–40
(`liq_mid`, `liq_comp`, `liq_pi`) everywhere: hook, sizes and observations. This lets an election
program change from hour to hour. For sports-2 builds it is a copy of the builder's full-ladder
`sports_x3.bin`; for sports-1 builds `adapt3.py --rewards` computes it from book.bin's top 5 levels.

### Without side files (v2 data)

mm3 synthesises per-market records: `settle_mode` 0 (2 if `winner_up < 0`), `event_ts = end_ts`,
crypto, `row_s` 1, `rw_v_c` 1.5, `rw_min_size` 50 (the constants `feat.bin`'s `liq_comp` was
computed with), rate 0, fees from markets.bin, flag LEGACY (the AUG_2026 pool is assumed for every
market, so the reward slots read "in program"). Crypto `markets.bin` itself mixes 0/50 and
0/1.5/4.5 in its rewards fields, so those fields are not used for legacy data.

Validation at open: sizes match `N` / `n_markets`, `settle_mode` in 0..2, `category` in 0..3,
`row_s > 0`, `settle_price` in [0, 1] for mode 1, and gap links in range, not to itself and mutual. Wide feat layouts (G = 88) are refused, so
elections must be built with the default split layout.

## 3. Episode

- **Length.** `n_rows - 1` rows from the market's own record, for the start draw, the dt class, the
  inventory-penalty denominator and the `ep_left` observation. Every builder so far writes
  `n_rows = contract_len_s + 1`, so for crypto this is v2's value.
- **Length bit and dt.** 300 -> bit 1 (`dt_5m`), 900 -> bit 2 (`dt_15m`), 14400 -> bit 4 (`dt_4h`),
  any other length -> bit 8 (`dt_other`, default 5 rows). `len_mask` defaults to 15. `dt > 0`
  overrides all four.
- **Variable cadence.** With `dt_far > 0`, a decision whose row is more than `far_s` seconds
  (default 7200) before the event covers `dt_far` rows instead. This is the sports spec's "30 s far
  from kickoff, 2 s inside 2 h", for example `dt_other=2, dt_far=30`.
- **Category filter.** `cat_mask` (bit per category, default 15) joins `split` and `len_mask` in
  the market lists.
- **Start.** As v2: a random row in the first `start_frac` of the episode, or `fixed_start`, or
  `cycle_stride` / `fixed_market`.
- **Time to event.** `to_event(t) = (event_ts - open_ts) - (t - r0) * row_s` seconds. It drives
  `pull_s`, `dt_far`, `close_ramp_s` and observation slot 50.
- **Quote pull.** `pull_s > 0` turns both quotes off (cancel) for every decision with
  `to_event(t) <= pull_s`. The cancel reaches the exchange after the latency, as any decision does.
  With the event at the last row, the book is flat of our quotes at kickoff.
- **Terminal and settlement.** At the last row the inventory is valued at `settle`:

  | settle_mode | settle | used by |
  |---|---|---|
  | 0 | `winner_up` (0/1) | crypto (v2) |
  | 1 | `markets3.settle_price` | sports: the last pre-kickoff mark (build_sports.py's `SETTLE_PX` value) |
  | 2 | `feat.mark[last row]` | elections (unresolved, `winner_up = -1`); also any episode cut at an artificial boundary |

  The `settle_mode` parameter (>= 0) overrides the data. `winner_up < 0` with mode 0 falls back to
  mode 2, and legacy data with mode 1 also falls back to mode 2. For modes 1 and 2, `flat_c`
  (cents per share) charges a flattening cost `|inv| * flat_c / 100` at the terminal: that is the
  crossing cost of the sports spec, half spread plus slippage. The cost is in the P&L, the
  reward and the Log (`flat_cost`). Markouts past the end of the episode (Log only) use `settle`.
  A piece cut at a recording gap that has a continuation (`gap_next`) is never flattened: it
  settles at its last pre-gap mid (D16, section 11a).

## 4. Reward and shaping

`W = cash + rebates + rewards + inv * mark` (at the terminal: `inv * settle - |inv| * flat`).
Reward modes 0, 1 and 2 are as in v2. Then:

- **Close-inventory shaping** (moved from `train_ppo.py` `CloseInventoryPenalty`, same formula):
  `r -= close_inv_coef * |inv| / Q * max(0, 1 - left)` on every step. `inv` is the inventory at the
  decision row, and `left` is the fraction of the episode left divided by `close_ramp`, or, with
  `close_ramp_s > 0`, the seconds to the event divided by `close_ramp_s` (the ramp on time to
  kickoff of the sports spec). It is in reward units, not in the P&L, and it is logged as
  `close_penalty`. It is applied before the [-1, 1] clip.
- **Identity (mode 0, no clipping):** `sum(r) / scale + inv_penalty_sum + close_penalty / scale =
  P&L`. Tested to 2e-4 USD. Mode 2: `... = fill credits + rebates + rewards`.

## 5. Liquidity-reward hook (`liq_rewards = 1`)

Polymarket's scorer (`analysis/rewards/REPORT.md`, `reward_terms.py`; also what
`build_elections.py` implements), evaluated at match row `tau` on each sample row (`liq_pi > 0`)
whose book is scorable (`liq_mid >= 0`):

- For each of our live orders with remaining size >= `min_size` and age >= `liq_moas` match rows:
  `S = ((v - d) / v)^2 * size`, where `d = |price - liq_mid|` in cents and S = 0 when `d >= v`.
- Two-sided rule. Inside [0.10, 0.90]: `Q_min = max(min(Q_bid, Q_ask), max(Q_bid, Q_ask) / 3)`.
  Outside: `Q_min = min(Q_bid, Q_ask)`.
- Paid: `liq += liq_pi * Q_min / (Q_min + liq_comp)`. `liq_pi` is the USD per sample: daily rate /
  1440 on one sample second per UTC minute.

| input | source, in order of precedence |
|---|---|
| v (cents) | `liq_v_c > 0`, then `x3.rw_v_c[tau]`, then `markets3.rw_v_c` (legacy 1.5) |
| min size | `liq_min_size > 0`, then `x3.rw_min_size[tau]`, then `markets3.rw_min_size` (legacy 50) |
| adjusted mid, competitor Q_min, USD per sample | `x3.liq_mid/liq_comp/liq_pi[tau]`, else feat.bin columns 38/39/40 |

Approximations, all inherited from the builders' precomputation:
- the replay book does not react to us;
- our orders move neither the adjusted mid nor the competitors' Q;
- competitors use the 'hi' bound (two-sided as possible);
- sports-2 and elections builds score the full recorded ladder; only the sports-1 fallback
  (`adapt3.py --rewards`) scores the top 5 levels of `book.bin`;
- the in-game multiplier b is 1.

Log: `liq_rewards`, `rw_samples` (sample rows while any quote was live), `rw_scored`, `rw_share`
(the sum of our share at scored samples). With legacy crypto data the hook is exactly v2's.

## 6. Order sizes

| size_mode | head k size |
|---|---|
| 0 (v2) | `size_mult * min_order_size * {1, 2, 5}[k]` |
| 1 | `size_mult * max(rw_min_size[tau], min_order_size) * size_rel{k}` (defaults 1, 2, 4) |

The min size is read at the order's match row, so it follows per-row programs. The inventory cap Q
still truncates (or pulls, below `min_order_size`) a new order. Reward min sizes run from 20 to 1000
shares, so set Q to at least 4–8x the largest min size in the market set (open issue 7).

## 7. Observation: `obs_size` floats (50 = v2; 80 recommended)

Obs 0..49 are v2's (`../ENV.md` section 6). Obs 49 is the constant 1. For `obs_size > 50`, slots
50.. hold the first `obs_size - 50` of these 29; any slot beyond 78 is 0. Every input is at row
<= t.

| obs | name | definition |
|---|---|---|
| 50 | ev_log | log1p(seconds to event) / log1p(86400): kickoff clock (sports), window clock (elections) |
| 51 | ep_left | rows to the episode end / episode rows. A GAP_CUT piece (D16): the time to the event / its value at the piece start, so the slot does not announce the gap |
| 52–54 | cat_crypto, cat_sports, cat_elections | category one-hot |
| 55 | rw_active | the program is active at t (x3 `rw_in_program`, else per-market rate > 0, legacy 1). **Slots 55–77 are 0 when liq_rewards = 0 or no program** |
| 56 | rw_rate | log1p(daily rate USD) / 5 |
| 57 | rw_v | v / 5 (cents) |
| 58 | rw_min_rel | log1p(min size / min_order_size) / 3 |
| 59 | rw_scorable | liq_mid >= 0 (two-sided valid book) |
| 60 | rw_two_sided_only | scorable and the adjusted mid is outside [0.10, 0.90]: single-sided quotes earn 0 |
| 61 | rw_comp | log1p(competitor Q_min) / 10 |
| 62–64 | rw_size_ok0..2 | size head k (current size_mode) >= min size |
| 65 | rw_cur_share | our current share Q_min / (Q_min + Q_comp) from the live orders (min-size and age rules) |
| 66–71 | rw_bid_share_a1..a6 | per price head a: `S_a M / (S_a M + Q_comp)`, where S_a = ((v - d_a)/v)^2 at the head's bid price (other side off) and M = max(min size, min_order_size), the smallest eligible order. One-sided: ignores the two-sided rule (slot 60) |
| 72–77 | rw_ask_share_a1..a6 | the same for the ask heads |
| 78 | gap_recent | D16: on a piece that starts after a recording gap, 1 − min(rows since its start, 120) / 120 (1 at the first post-gap row, 0 from row 120 on); 0 on every other piece. Independent of liq_rewards (section 11a) |

## 8. Fees and rebates

The rebate per fill is v2's: `rebate_rate * fee_rate * q * (p(1-p))^fee_exp`. The coefficients
resolve per market, with the first match winning:
1. `fee_rate` / `rebate_rate` (>= 0, all markets);
2. `fee_rate_<crypto|sports|elections>` / `rebate_rate_<...>` (>= 0, that category);
3. `markets3.bin`;
4. `markets.bin`.

The builders already copy each market's gamma `feeSchedule` into markets.bin:

| schedule | fee rate | rebate rate | where it applies |
|---|---|---|---|
| `sports_fees_v3` | 0.05 | 0.15 | most sports |
| `sports_fees_nfl_cfb_oct26` | 0.03 | 0.15 | NFL/CFB moneylines |
| `zero_fees` | 0 | 0 | NFL/CFB spreads, totals and props |
| politics | 0.04 | 0.25 | elections |

So the per-category parameters are for sensitivity runs (`analysis/sports/out/docs_notes.md`,
`analysis/v3_sports_rules/ENV_SPEC.md`).

## 9. Log

v2's 42 fields in v2's order (`pnl_trading` now includes the flattening cost; for a piece that
continues a D16 gap chain, `pnl` and `pnl_trading` are its increment over the wealth carried in),
then
`close_penalty`, `flat_cost`, `settle_px`, `rw_samples`, `rw_scored`, `rw_share` and
`episode_rows`, then the six D15 netting / flatten diagnostics below, then `n` (56 floats).

**D15 netting and flatten drag** (DECISIONS.md D15; `analysis/v3_sports_rules/RECONCILE.md` section 6
finds that pre-game makers net only 10–25% of their fills before kickoff and that the flatten drag
exceeds the edge). They are Log-only: nothing else reads them, so observations, rewards and P&L are
unchanged (section 12a).

| field | per episode |
|---|---|
| `net_gross_sh` | gross filled shares, buys + sells (accumulated in double; = `fill_vol` up to float32 rounding) |
| `net_notional` | filled notional, sum of price x shares (USD) |
| `net_end_inv` | \|inventory\| at the episode's end, the settlement row: kickoff for sports (= `final_abs_inv`) |
| `net_share` | 1 − `net_end_inv` / `net_gross_sh` for an episode with fills, else 0 |
| `net_eps` | 1 for an episode with fills, else 0 |
| `net_flat_drag` | flattening cost charged at settlement, `flat_c` / 100 x \|end inv\| USD (= `flat_cost`; already in `pnl`) |

With `gap_chain = 1` (D16) a piece whose inventory carries into the next piece logs `net_end_inv`
0 and no `net_share` / `net_eps` (its inventory has not ended), and the chain's last piece logs its
\|end inv\| with `net_share` taken against the gross filled shares of the whole chain, so the
pooled netted share is per chain.

`env3.netting(log)` forms the ratios (the Log is averaged or summed over episodes, and ratios are
unaffected): `netted_share` = 1 − Σ\|end inv\| / Σ gross (RECONCILE's pooled definition),
`netted_share_ep` = `net_share` / `net_eps` (the mean over episodes with fills), `flat_drag_usd`
and `flat_drag_bps` = drag / notional x 1e4. `MmEnv3.step` adds them to every info log;
`train3/evaluator3.py` reports them per market, group, category and bucket.

## 10. New parameters (after v2's 27, in `M3_PARAM_NAMES` order)

| name | default | meaning |
|---|---|---|
| len_mask | **15** (v2: 7) | + bit 8: lengths other than 300 / 900 / 14400 rows |
| dt_other | 5 | decision interval (rows) for other lengths |
| dt_far, far_s | 0, 7200 | > 0: rows per decision while more than far_s s remain to the event |
| cat_mask | 15 | 1 crypto, 2 sports, 4 elections, 8 other |
| obs_size | 50 | 50..128; 80 = all 29 extras + 1 pad |
| size_mode, size_rel0/1/2 | 0, 1, 2, 4 | section 6 |
| liq_v_c, liq_min_size | -1, -1 | > 0: override the reward max spread / min size |
| settle_mode | -1 | >= 0: override markets3 settle_mode |
| flat_c | 0 | flattening cost, cents per share, at mode 1 / 2 settlements |
| pull_s | 0 | > 0: quotes off within pull_s s of the event |
| close_inv_coef, close_ramp, close_ramp_s | 0, 0.2, 0 | close-inventory shaping (section 4) |
| fee_rate_crypto, rebate_rate_crypto, fee_rate_sports, rebate_rate_sports, fee_rate_elections, rebate_rate_elections | -1 | per-category fee / rebate (section 8) |
| gap_chain | 0 | D16: 1 = a piece with a continuation in the same market list hands its inventory and cash to it (section 11a); train3's eval env sets 1 |

## 11. Builder outputs and the schema mapping

`python adapt3.py BUILD_DIR` detects the builder from `meta.json`. It writes `markets3.bin`,
`x3.bin` (when there are per-row rewards) and `meta3.json` into the build directory, and touches
nothing else.

**Sports, sports-2 and sports-3** (`analysis/v3_sports/build_sports.py`, `build_version` "sports-2"; "sports-3" adds the D16 gap fields of section 11a: `meta.json`
carries `segments`, one record per markets.bin row, and the build writes `sports_x3.bin`)

| mm3 input | from |
|---|---|
| episode | each market's `n_rows`: `--seg 0` = one episode per game to kickoff (length bit 8), or a bucket piece (`--bucket-segments`), or 300 / 900 / 14400 |
| settle_mode / settle_price | 1 / `segments[i].settle_price` (the episode's last-row mark). Builds made with `--no-legacy-settle` (no flag 16, `winner_up` −1) load the same way. When the legacy copy is present, it must equal the meta value, or adapt3 refuses the build |
| event_ts | `segments[i].kickoff_ts`, for every episode. Bucket pieces and episodes cut before kickoff (`--cut-before-kickoff-s`) end earlier, but their time to event (slot 50, `pull_s`, `dt_far`, `close_ramp_s`) is the time to kickoff |
| row_s | `segments[i].row_s` |
| category | 1 |
| rw_v_c / rw_min_size | `markets.rewards_max_spread` / `rewards_min_size` (the program at the last row) |
| rw_rate_day | `segments[i].rw.rate_day_last` |
| x3.bin | a byte copy of `sports_x3.bin`: the full recorded ladder, from the daily `/rewards/markets/current` snapshots. It is never recomputed, and `--rewards` is ignored with a warning |
| TRADES_TRUNCATED | markets.bin flag 1024 (bit 32 is unused in sports-2 and is not read); listed in `meta3.json` `trades_truncated_markets` |
| fees | markets.bin (gamma `feeSchedule`) |

**Sports, sports-1** (legacy; also a build without `meta.json` `segments`)

| mm3 input | from |
|---|---|
| episode | each segment's `n_rows` |
| settle_mode / settle_price | 1 / `markets.log_strike_vs_ul_open` (flag `SETTLE_PX` 16 required: the last pre-kickoff mark) |
| event_ts | `open_ts + sports_feat.seconds_to_kickoff[first row]`: the kickoff, also for segments that end before it |
| row_s | `meta.sports.dt_s` |
| rw_v_c / rw_min_size | `markets.rewards_max_spread` / `rewards_min_size` (gamma) |
| rw_rate_day | `--rewards rates.json` {slug: USD/day}, else 0 |
| x3.bin | with `--rewards` (the fallback for builds without `sports_x3.bin`): from `book.bin` top 5 levels: the size-cutoff adjusted mid, competitors' 'hi' Q_min, one seeded sample second per UTC minute, `liq_pi = rate / 1440` |
| TRADES_TRUNCATED | markets.bin flag 32 |

**Elections** (`analysis/v3_elections/build_elections.py`, split feat layout)

| mm3 input | from |
|---|---|
| episode | the window (`window_s + 1` rows; 86400 gets length bit 8 and dt_other) |
| settle_mode | 2 (mark at the last row; `winner_up = -1`); `settle_price` = that mark, for information |
| event_ts | `end_ts` (window end) |
| row_s / category | 1 / 2 |
| rw_* (per market) | feat_x at the first row |
| x3.bin | per row: feat_x `rw_max_spread_c`, `rw_min_size`, `rw_daily_rate`, `rw_in_program`; feat.bin `liq_mid`, `liq_comp`, `liq_pi`. Streamed through a memmap (an 18-day build has 10^8 rows; x3 adds 32 B/row to its 580) |

**Crypto** (`data/build`): nothing needed. `--kind crypto` writes the legacy records explicitly.

**Proposals for the builders** (their folders were not edited). S1–S3 landed in build_sports sports-2
(2026-10-15) and adapt3.py reads them; they are kept here for the record.
- **S1 (sports): `--seg 0` = one episode per game window.** mm3 accepts any length, so the
  "6 x 4 h segments" tiling and its artificial mid-window settlements are no longer needed.
  Combine it with `dt_far` for the far-from-kickoff cadence. `SEG_LENS` and the `len_*` one-hot
  assertion in `train/envs.py` MarketSetEnv are the only blockers, and the latter is the trainer's.
- **S2 (sports): write the reward columns.** From the full recorded ladder (as the elections
  builder does), and with a daily CLOB `/rewards/markets/current` snapshot for `rate_day`. Put
  them in feat.bin `liq_mid/liq_comp/liq_pi`, or as an `x3.bin` directly. Today `adapt3.py
  --rewards` approximates from the top 5 levels.
- **S3 (sports): store `kickoff_ts` per segment in meta.json**, so the adapter does not have to
  read it from `sports_feat.bin` row 0. Also stop overloading markets flag bit 16 and
  `log_strike_vs_ul_open`: bit 16 means SETTLE_PX in sports and ELECTION in elections, and bit 32
  is TRADES_TRUNCATED in one and UNRESOLVED in the other. `markets3.bin` is the explicit
  replacement.
- **E1 (elections): window length.** Keep 86400 (it works now). For training throughput consider
  4 h windows (bit 4, `dt_4h`), or `dt_other` 5–10. Settlement at the window end is a mark, not a
  resolution (open issue 4).
- **E2 (elections): add `x_log_days_to_end` and the event start to x3** (pad column) if
  time-to-resolution should be observed. Slot 50 measures time to the window end.

## 11a. D16 recording gaps (sports-3, elections)

DECISIONS.md D16: a recording gap longer than 15 s splits an episode, and no row inside it is
synthesised. The inventory is marked at the last pre-gap mid and carried into the next segment
as a forced quote pull. A day with more than 30 min of gaps is train only. A sports game with a
gap in its last pre-kickoff hour is excluded from val / test for the last-hour buckets. The
builders cut the episodes and write the gap fields; adapt3 carries them into mm3, and
`train3/evaluator3.py` applies the eval rules (train3/README.md, "D16").

**Where the links live: in `markets3.bin`, not a side file.** The two int32 words at offset 56 that
were padding are now `gap_next` / `gap_prev` (index + 1 of the linked piece, 0 = none), and flags3
gets four bits (section 2). Reasons:
- mm3 has to know the continuation inside the step: the chain hop happens at the terminal row,
  in C;
- the record is already per market, in market order;
- every earlier `markets3.bin` and the synthesised v2 records have zeros there, which means "no
  gap". So mm3 stays v2 bit for bit with the options off (section 12a), and older adapted builds
  load unchanged.

mm3 checks at open that the links are in range and mutual. `meta3.json` `gaps` holds what only
train3 reads:
- the per-day summary `days` {day: n_gaps, total_gap_s, eligible, ...};
- `days_ineligible`, `chains` (market indices, head first) and `n_links`;
- the flagged pieces and `eval_excluded_buckets` {market: [build bucket names]}.

| D16 field | sports-3 (`build_sports.py`) | elections (`build_elections.py`) |
|---|---|---|
| links | `segments[i].gap.next_sid` / `prev_sid`, by sid. The market order is (split, end_ts, sid), so linked pieces need not be adjacent | `meta.json episodes[i].prev_segment_id` / `next_segment_id` when present, else `markets_meta.parquet` (same columns), else the builder's own rule: consecutive episodes of a market, `end_reason` gap -> `start_reason` gap |
| GAP_CUT / AFTER_GAP | markets.bin flags 16384 / 32768 (or `cut_reason` / `start_reason` 'gap') | `end_reason` / `start_reason` 'gap' |
| EVAL_EXCL | flag 65536 / `excluded_from_eval_last_hour`, with `eval_excluded_buckets` | none (no last-hour rule) |
| DAY_INELIGIBLE | any UTC day of [open_ts, end_ts] with `meta.gaps.days[d].eligible` false | `not episodes[i].day_eligible`, or that UTC day's recorder-wide row of `gaps_per_day.csv` is not eligible |
| per-day summary | `meta.gaps.days` (n_gaps, total_gap_s, eligible, all_logged); `days_not_all_logged` lists days whose gaps are not all in GAPS.txt (reported, not used: the builder's `eligible` decides) | `gaps_per_day.csv`, scope recorder (n_gaps, total_gap_s, eligible, ...), plus the per-day count of ineligible market-days |
| event_ts | the kickoff, unchanged | a linked piece's event is the end of its chain's last piece (the window episode the gaps cut), so obs 50 and `pull_s` do not announce the gap |
| settlement | the piece's `settle_price` = its last-row mark = the last pre-gap mid (the builder's) | mode 2, the mark at the piece's last row |

adapt3 refuses links that are not mutual, that do not join a GAP_CUT piece to an AFTER_GAP piece
of the same market and split forward in time, and a `prev_sid` that disagrees with `next_sid`.

**Inventory across a gap.**
- **Settlement.** A piece with a continuation settles at its last pre-gap mid without `flat_c`. A
  GAP_CUT piece without a continuation (the builder dropped its post-gap piece as too short)
  settles at `settle_price` and pays `flat_c`, like any other episode end.
- **`gap_chain = 1`.** At the terminal of a piece whose `gap_next` is in the same market list (same
  split, `len_mask`, `cat_mask`), mm3 logs the piece, then starts `gap_next` at row 0 with:
  - inventory = the carried inventory;
  - cash = cash + rebates + rewards, so the next piece's rebate and reward fields count from 0;
  - wealth = the piece's terminal wealth (`w0`).

  Live orders and the pending FIFO are dropped: the gap is a forced quote pull. The step returns
  terminal = 1, because each piece is its own Log record, together with the observation of the
  next piece's first row. The handle then plays the chain to its end.
- **Increments.** A continued piece's Log `pnl` and `pnl_trading` are increments over `w0`. The
  price move across the gap on the carried inventory lands in the continuation's first step. So
  the pieces' P&Ls sum to one continuous episode over the same rows, in which:
  - the quotes are forced off at the last pre-gap row;
  - the decision grid restarts at the first post-gap row;
  - the minute windows of the participation cap follow the real clock (a gap is > 15 s; each
    piece restarts its minute window);
  - settlement and `flat_c` happen only at the chain's end.

  Section 12g tests this against an independent continuous-episode reference: 3e-13 USD in
  double, 1.3e-8 relative on the float32 Log.
- **Netting.** A continued piece logs `net_end_inv` 0 and no `net_share` / `net_eps`. The chain's
  last piece logs its |end inventory| and its netted share against the whole chain's gross
  (section 9).
- **`gap_chain = 0`, the default, used for training.** Each piece is independent: it settles at its
  last pre-gap mid without `flat_c` (the mark-to-mid settle), and the post-gap piece starts flat.
  Chaining in training is possible (`gap_chain=1` in `--env3-kw`), but the terminal at the hop
  means PPO's value does not bootstrap across a gap. Eval chains by default (train3 `make_eval`).

**No lookahead to the gap.** A gap-cut piece's row count would reveal that a gap is coming. So for
GAP_CUT pieces, obs 51 (`ep_left`) and the fraction-mode close ramp (`close_ramp`) are computed
from the time to the event over the time from the piece start to the event (kickoff; elections:
the chain's end), not from the rows to the piece end. `pull_s`, `dt_far`, `close_ramp_s` and obs 50
already use the event. The builders' own feature `tau_frac` (obs 7) still counts rows to the
segment end in sports-3, which is a builder change (below).

**Lagged features after a gap: a flag, obs 78 `gap_recent`.** For up to 120 rows after a gap, the
builders' lagged and rolling features (flow / mret / dgap windows <= 120 rows) look back over the
forward-filled pre-gap book. mm3 adds a flag rather than masking. On a piece that starts after a
gap the flag is 1 − min(rows since its start, 120) / 120; elsewhere it is 0. Why a flag:
- masking needs to know, per builder, which of the 33 observed feat columns are lagged and over
  how many rows, and it would change the builders' feature semantics in the env;
- the flag leaves the data untouched and lets the policy discount those rows.

The slot was a zero pad of obs 80. With `obs_size` 50 (options off) nothing changes, and on data
without AFTER_GAP pieces it is 0, so every earlier configuration is unchanged.

**Builder changes still useful** (the builder folders were not edited):
- **w1 (sports):**
  - sports-3's `tau_frac` (feat col 7, observed) counts rows to the segment end. For a GAP_CUT
    segment this announces the gap. It should count to the kickoff, or to the end of the gap
    chain. mm3 fixes only the slots it computes itself (obs 51, close ramp).
  - `meta.gaps.days[d].eligible` ignores `all_logged`, although the rule text in meta asks for
    both. Either fold `all_logged` into `eligible` or drop it from the rule text. adapt3 uses
    `eligible` and reports `days_not_all_logged`.
- **w3 (elections):**
  - Put `prev_segment_id` / `next_segment_id` / `gap_before_s` and `window_end_ts` into
    `meta.json episodes`, so adapt3 needs neither the parquet nor the link rule, and the event can
    be the window end rather than the chain end.
  - Put the recorder-wide per-day summary of `gaps_per_day.csv` into `meta.json gaps.days`.
  - `seconds_to_close` / `tau_frac` count to the window end, which is fine. The per-window
    `log_tau` is also unaffected.

## 12. Tests (`python test_env3.py`, 68 s; `--quick`, `--no-builders`, `--json OUT`)

Results of the committed version (2026-10-16 ~03:00 UTC, full run with the D16 changes, 303 checks,
all ok; the v2 comparison is unchanged: the same 9,079,000 steps and 16,218 episodes, bit-identical,
P&L difference 0.0 USD):

**(a) v2 bit-exactness** (`cmp_harness.c`: `../mm.h` and `mm3.h` in one translation unit).

Runs:
- 19 configs: 9 fill-model x latency pairs, mark modes 1/2 with penalty, the probe, tick_rule 1
  with dt 3, delay 2 with no cap and 35% rebate, pessimistic delay 4 L 2, liquidity rewards with
  size x10, optimistic size 0.5 with Q 20, credit + liquidity with mark 2, and central with
  obs_size 80;
- each config on every val market of data/build (247; 4h markets only in the 3000-step runs) and
  40 train markets;
- random sticky actions.

In total, **9,079,000 steps and 16,218 episodes**. Observations 0..49, rewards, terminals,
inventory / cash / rebates / rewards / wealth / penalty / order state and the Log are
**bit-identical**. Each episode's P&L, computed in double at its terminal step, differs by
**0.0 USD** (tolerance 1e-9; sum of |P&L| compared: 187,739 USD).

**(b) C == reference3** on five sets:
- the synthetic sports build (sports-2 schema): 6 games of 1201–3001 rows, kickoff at the last row, 7 configs;
- its `--bucket-segments` variant (16 pieces of 201–1501 rows, ending 1800 / 300 / 0 s before kickoff):
  obs 80, liquidity rewards, `pull_s` 600, `flat_c`, close shaping;
- the synthetic elections build: 5 one-hour windows, programs changing mid-window, 5 configs;
- crypto markets with v3 options on: 3 configs;
- the real output of `analysis/v3_sports/build_sports.py --selftest` (now sports-3) through `adapt3.py`:
  2 configs;
- the D16 gap fixtures (section 12g): the sports-3 gap build under 3 configs and the elections gap
  build under 2, each on its train and its val markets. These configs cover chains
  (`gap_chain` 1, with the auto-reset episode being the chained piece, its carried state and its
  whole Log compared) and independent pieces (`gap_chain` 0), obs 80 (slots 51 and 78), liquidity
  rewards, `flat_c`, `pull_s`, both close ramps, `dt_far` and pess L2.

Configs cover every new option:
- obs 64/66/80;
- liq with size_mode 0/1, relative sizes, moas 30, liq_v_c / liq_min_size overrides;
- pull_s, flat_c, close shaping (fraction and seconds ramp), settle_mode override, dt_far;
- per-category fees;
- reward_mode 2;
- pessimistic / optimistic models, latency 0–2.

Every step compares: reward (1e-7), terminal, every observation slot (1e-6, all were 0.0), the
30-value state (1e-9 rel), the Log including the six `net_*` fields (1e-4 rel; observed ≤ 6e-8)
and the auto-reset episode. Telescoping with the
penalty, flattening cost and close shaping holds to 2e-4 USD. Lookahead holds on sports, elections
and crypto (obs 80, rewards on): perturbing feat / book / fills / x3 after t changes nothing up
to t, and does change things after.

**(c) Settlement** (sports fixture, join-heavy policy, `pull_s` 120, Q 1000):
- every episode ends at kickoff: `open + (n_rows - 1) * row_s == event_ts == end_ts`;
- `settle_price == log_strike (SETTLE_PX) == mark at the last row`;
- the C Log `settle_px` is the kickoff mark;
- inventories of 133–269 shares are held into kickoff and marked there;
- `flat_c = 0.5` lowers the P&L by exactly `|inv| * 0.005` (fills unchanged);
- `settle_mode 0` (winner_up) differs by `|inv| * |winner - mid|`, while mode 2 equals mode 1;
- flipping `winner_up` changes nothing (the game result is never read);
- after the pull no fill is matched and no quote is live at kickoff;
- a never-quoting policy makes 0 under every settle mode;
- elections: `winner_up = -1`, settled at the last mark.

**(d) Reward hook.** Static book 0.49 / 0.51 and a single sample row. v = 3 c, min size 100,
Q_comp = 100, 14.4 USD/day (0.01 per sample, stored as float32).

| case | hand | reference | C (float32 Log) |
|---|---|---|---|
| bid 0.49 + ask 0.52, 100 each: Q = 44.44 / 11.11, Q_min = 14.81 | 0.0012903226 | 0.0012903226 | 0.0012903225 |
| bid 0.50 alone: Q_min = 100 / 3 | 0.0025 | 0.0025 | 0.0025 |
| 5-share orders (size_mode 0), or orders younger than moas | 0 | 0 | 0 |
| no competitor | 0.01 | 0.01 | 0.01 |

The reward observations (current share, head-3 bid share, the out-of-range head 6, size
eligibility, active / scorable / two-sided-only) equal the hand values. On 2000 random books,
reference3's Q_min equals `reward_terms.side_scores` + `q_min` (max difference 0).

**(e) adapt3 on the build_sports schemas** (`fixtures3.py`):
- the same six games built as sports-2, as sports-2 `--no-legacy-settle` and as sports-1 (with
  `--rewards`) adapt to byte-identical `markets3.bin` and `x3.bin`;
- sports-2 `x3.bin` is a byte copy of `sports_x3.bin`. With `sports_x3.bin` perturbed and
  `--rewards` given (999 USD/day for every slug), `x3.bin` follows the perturbed file, and
  `rw_rate_day` stays `segments[i].rw.rate_day_last`;
- TRADES_TRUNCATED is read at 1024 in sports-2 and at 32 in sports-1. A stray bit 32 in sports-2
  is ignored;
- with `--no-legacy-settle` the build has no flag 16, `winner_up` −1 and `log_strike` 0; it settles
  at `segments.settle_price` = the last-row mark, mode 1;
- adapt3 refuses a legacy copy that disagrees with meta, and a sports-1 build without flag 16;
- bucket pieces: `event_ts` = `kickoff_ts`, every piece pure, ending 1800 / 300 / 0 s before
  kickoff and settling at its own last-row mark; obs 50 at row 0 decodes to the seconds to kickoff
  (rel 3e-7).

**(f) D15 netting fields** (sports fixture, join-heavy random policy, Q 1000, `flat_c` 0.5):
- `net_end_inv` == `final_abs_inv`, and `net_gross_sh` == `fill_vol` == buy + sell (rel 1e-7);
- `net_flat_drag` == `flat_cost` == |end inv| x 0.005;
- `net_share` == 1 − end / gross per episode, and it lies in [0, 1];
- `net_eps` marks the episodes with fills, and notional / gross is a price;
- `env3.netting` gives the pooled share and the drag in bps;
- a never-quoting policy gives 0 everywhere.
- The binding's `vec_log` carries every Log field, the net_* fields included.

**(g) D16 recording gaps** (fixtures `sports3_gaps`, `sports3_nogap`, `elections_gaps`):
- **adapt3, sports-3.**
  - `gap_next` / `gap_prev` equal the segments' `next_sid` / `prev_sid`, including in builder
    market order.
  - flags3 GAP_CUT / AFTER_GAP / EVAL_EXCL equal markets.bin flags 16384 / 32768 / 65536.
  - DAY_INELIGIBLE is set exactly on the game of the 40-minute day.
  - One GAP_CUT piece has no continuation.
  - `eval_excluded_buckets` and the per-day counts and seconds are carried into meta3.json.
  - Builds without gap fields keep zeros.
  - mm3 refuses links that are not mutual, and adapt3 refuses a `prev_sid` that disagrees.
- **adapt3, elections.**
  - Links from `markets_meta.parquet`, from the builder's rule (parquet deleted) and from explicit
    `meta.json episodes` fields give byte-identical `markets3.bin`.
  - A chain's `event_ts` is its last piece's end.
  - DAY_INELIGIBLE follows `day_eligible` and the `gaps_per_day.csv` day.
- **Chained P&L == one continuous episode.**
  - Setup: game 0's two pieces (2,000 and 3,881 rows, with a 120 s gap) are played as a chain
    (C, `gap_chain` 1). They are compared with `test_env3.ContinuousGap3`, a reference that plays
    the same rows back to back as one market (`sports3_nogap`). In that reference the quotes are
    forced off at the gap, the decision grid restarts after it, and the minutes follow the real
    clock; there is no carry or Log split. Four configs: central; obs 80 + liq + size_mode 1 +
    `flat_c` + `pull_s` + `close_ramp_s` + `dt_far`; pess L2 with `flat_c` 1; opt L0 with the
    participation cap. Inventory of 56–700 shares crosses the gap.
  - The rewards are identical step by step (max difference 0).
  - The P&L agrees: the reference chain (double) is within 3e-13 USD, and the C Logs summed
    (float32) within 1.3e-8 relative.
  - The pre-gap piece pays no flattening cost, and the chain's end pays |inv| x `flat_c`.
  - Netting: the pre-gap piece logs no end inventory, and the end's netted share uses the chain's
    gross.
  - With `gap_chain` 0, the pre-gap piece settles at its last pre-gap mid without a flattening
    cost, and the post-gap piece starts flat.
  - A GAP_CUT piece without a continuation pays |inv| x `flat_c`.
  - obs 78 equals 1 − rows / 120 after a gap and 0 elsewhere.
  - obs 51 of a GAP_CUT piece is the time-to-kickoff fraction.
- **train3 evaluator3, in subprocesses.**
  - Sports: 6 val pieces are played and 2 are scored (game 0's chain, one handle, `chain_pos` 0 /
    1). Excluded: 3 last-hour pieces and 1 piece on an ineligible day.
  - The scored P&L equals the chain's pieces summed, and every piece's time buckets add up to its
    P&L.
  - A last-hour-excluded piece keeps its `1_6h` steps in `tk/*` but not its excluded buckets.
  - The result carries `gaps` (per-day counts and seconds, eligibility, evaluated days).
  - With `gap_chain` 0, every piece is its own episode.
  - Elections: 2 chains of 2 pieces are scored, with the gaps from `gaps_per_day.csv`.
  - split3 records `_gaps_per_day`, and the ineligible day (test by D11) is dropped as train only,
    for both tracks.
- **Binding.** The PufferLib 3.0 binding equals the core on the sports-3 gap build with
  `gap_chain` 1.

**Also tested:**
- parameter tables: mm3.h == Config3; the first 27 are v2's, and only `len_mask` changed default;
- the Log's first 42 fields are v2's;
- the split guard;
- the PufferLib 3.0 `binding3` == core, bit for bit, on crypto (obs 80, rewards) and elections
  (pull, flat), in subprocesses;
- the 5.0 wrapper (`puffer5/test_wrapper3.c`) == core under OpenMP (run before the D15 fields; its
  Log name table now has 55 names, and a compile against mm3.h confirms the layout check. The 5.0
  sources are not on this machine, so it was not re-run).

## 13. Throughput (`python bench3.py`)

Single thread, 4096 envs, crypto train split, random actions, central, L 1. Best of 3 on the
shared 4-core VM, with 2–4 PPO trainings running (load about 2). Interleaved v2 / v3 rows:

| case | wall M steps/s | CPU M steps/s |
|---|---|---|
| core v2 (`mm.h`, `mm_harness.c`) | 3.84 | 3.89 |
| **core v3, options off** | **3.89** | **3.94** |
| core v3, obs 80 | 3.23 | 3.25 |
| core v3, obs 80 + liq + size_mode 1 | 2.06 | 2.09 |
| binding v2 (`env.py`) | 3.80 | 3.83 |
| binding v3 (`env3.py`), options off | 3.84 | 3.85 |
| binding v3, obs 80 + liq + size_mode 1 | 2.09 | 2.11 |

With options off v3 costs nothing: 1.01x v2 (re-measured after the D16 changes, 2026-10-16:
`bench3.py --skip-binding`, 4.15 / 4.19 M steps/s, v3 / v2 = 1.011). The 28 (now 29) extra slots cost about 17%, mostly the 12
per-head share computations. The reward hook costs a further 35%, because with it on every row is
matched one by one: v2's quiet-row skipping is off, as in v2's own liquidity mode. PPO stays
policy-bound, as `../ENV.md` section 11 shows.

## 14. Open issues

1. **Sports reward pools** are in sports-2 builds (`sports_x3.bin`, from daily
   `/rewards/markets/current` snapshots since 2026-10-15). Older games infer their program from the
   earliest snapshot (flag RW_LATER_SNAPSHOT). Only sports-1 builds need `adapt3.py --rewards`, which
   scores the top 5 levels.
2. **Taker delay.** Sports takers are delayed about 1 s (`secondsDelay`), and our cancels can beat
   them (ENV_SPEC section 6). The fill models are unchanged by instruction, so this edge is not
   modelled, which is conservative for adverse selection.
3. **Grids with row_s > 1** (sports `--dt 6`). Latency, match delay, the part_cap minute, the
   markouts and the T5 last30/last10 flags count rows, not seconds. mm3 uses `row_s` only for
   time to event. Build at dt 1 and use `dt_far` for cadence.
4. **Elections end at a window boundary**, not a resolution. Settling at the last mark is a
   modelling choice. `flat_c` and the close shaping are the levers. Daily windows also cut a
   program day into independent episodes.
5. **Kickoff accuracy and truncated trade history** are the sports builder's open problems
   (README_build.md). With `pull_s` >= a few minutes, a late `gameStartTime` matters less.
6. **The trainer adapter.** `train3/envs3.py` V3Adapter is the v3 adapter (it hashes `v3env/`,
   groups by category and evaluates with its own MarketSetEnv3, which has no length one-hot assert,
   so `--seg 0` builds work). `train/envs.py` RealAdapter can also drive env3:
   `ad = envs.get_adapter('real'); ad.mod = env3; ad.obs_size = env3.OBS_SIZE`, with
   `MM3_OBS_SIZE=80` in the environment (tested for make_train). Its `identity()` still hashes v2's
   files, its MarketSetEnv asserts the crypto length one-hot, and its groups are 5m / 15m / 4h. A
   v3 adapter should hash `v3env/` and group by category.
7. **Q vs reward min sizes** (20–1000 shares). There is no per-market Q, and reward_scale = 1/Q.
   A `q_rel` option (Q = q_rel x min size per market) would need a per-market reward scale.
8. **Own orders move neither the adjusted mid nor competitor Q**, and only one order per side is
   allowed (no laddering for reward farming).
9. The per-head share observation (66–77) ignores the two-sided rule (slot 60 flags it) and
   assumes the smallest eligible size.
10. **D16 gap chains in training.** Training plays each gap piece on its own (`gap_chain` 0,
    settled at the pre-gap mid without `flat_c`). The policy is never charged for inventory held
    into a gap, while eval carries that inventory across the gap. `gap_chain=1` trains on chains,
    but each hop is a terminal for PPO, so the value of the carried inventory is not bootstrapped
    across the gap. The builders' `tau_frac` (obs 7) still announces a sports-3 gap cut (section
    11a, builder changes).
