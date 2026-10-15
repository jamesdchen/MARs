# Market-making replay env (`mm.h`)

A single-agent market-making environment that replays Polymarket crypto up/down markets (5m / 15m /
4h, 8 coins, 2026-08-21..24). One episode is one market. The agent posts a bid and an ask for the
Up token. Up and Down share one book, and everything is in the Up frame. The agent's quotes are
filled by the recorded taker trades under one of three fill models.

* `mm.h`: the core, pure C. No PufferLib includes. It compiles as C99 and C++17 (PUFFER5.md).
* `binding.c` + `setup.py`: the PufferLib 3.0 native binding (`env_binding.h`).
  Build with `python setup.py build_ext --inplace`.
* `env.py`: `MmEnv(pufferlib.PufferEnv)`, with a MultiDiscrete `[7, 7, 3]` action space and a Box obs `(50,)`.
* `reference.py`: a readable Python/numpy implementation of the same model, used only by the tests.
* `test_env.py`: checks C against the reference, reward telescoping, lookahead, the split guard
  and binding parity.
* `bench.py`: throughput.
* `mm_harness.c`: ctypes harness used by `test_env.py` and `bench.py`.
* `puffer5/polymarket_mm.h`: the PufferLib 5.0 wrapper, maintained by another agent.
* Data: `data/build.py` writes ticks/markets. `data/features.py` writes `feat.bin`, `book.bin`,
  `fills.bin`, `fill_index.bin` and `feat_meta.json`. Run it with a Python that has scipy, e.g.
  `/usr/bin/python3`, because it imports the fair-value pricer. It writes atomically
  (temp file + `os.replace`), so running processes that have the old files mapped are safe.

## 1. Time, data and causality

* Row r of a market is unix second r in `[open, end]`, so a market has `contract_len_s + 1` rows.
* Row t carries:
  - the book: the latest snapshot `<= t`;
  - the trades whose **block time** is in `(t-1, t]`;
  - the Binance kline closed by t.

  This follows `data/README.md`.
* **Decision.** The decision at row t sees only `feat.bin[t]`, `book.bin[t]` and agent state
  built from rows `<= t`. The market's outcome, `winner_up`, is read only for the terminal
  settlement and, in the Log only, for markouts past the end.
* **Latency.** `latency = L` s. The quote decided at t reaches the exchange at t + L. A cancel or
  replace takes effect then too. Until then the previous quote rests.
* **Match delay.** Trade timestamps are Polygon block times, about 3-4 s after the off-chain match.
  A trade printed in row u is matched against our quotes as they stood at the **match row
  tau = u - D**, with `D = ceil(d + 0.5)`. This is the mid-second convention: a print in (u-1, u]
  matched d seconds earlier happened around u - 0.5 - d.
  - d defaults per fill model as in EVALUATION.md T1: optimistic 0 s (D = 1), central 3.5 s
    (D = 4), pessimistic 5 s (D = 6).
  - `match_delay` overrides d.
  - So a decision at t can first be filled by the prints of row t + L + D. The core keeps the
    order state in match time. A FIFO of pending decisions activates them when tau reaches
    their activation row.
* **Fills are booked** (inventory, cash, reward) at the print row u, when the data shows them.
  That is conservative: in reality a maker learns of its fill at the match.
* Prints after the market end are not in `fills.bin` and never fill.
* **Lookahead test.** In `test_env.py`, perturbing every data row after t leaves every
  observation, reward and causal state up to t unchanged.

### Files (written by `data/features.py`, layouts in `data/build/feat_meta.json`)

| file | layout | content |
|---|---|---|
| `feat.bin` | float32 `[N, 44]` | market-only features. Columns 0-32 are observed as obs 0-32. Then the aux columns: `p_model` (33), `mark` (34), `delta_usd_raw` (35), `sigma` (36), `sqrt_v` (37), `liq_mid`, `liq_comp`, `liq_pi` (38-40), `bid_minus_p_c` and `ask_minus_p_c` (41-42, observed as obs 47-48), pad (43) |
| `book.bin` | 64 B/row | top-5 bid/ask prices as int16 **mils** (0 = no bid, 1000 = no ask), sizes float32, `flags` (bit0 valid, bit1 crossed_raw, bit2 book_age > 10 s, bit3 from_down, bit4 ladder same as the previous row), `tick` (mils), `tick_alt`, `same_run` |
| `fills.bin` | 20 B/leg | every **maker** leg (`takerOnly=false`) with block time in [open, end]: `row`, `side` (+1 taker bought Up: hits asks; -1 taker sold Up: hits bids), `first` (first leg of its tx), `px` (mils), `price`, `size`, `tx_size`. Sorted by row, then tx, then sweep order (best price for the taker first) |
| `fill_index.bin` | int32 `[N+1]` | the legs of row r are `fills[idx[r]:idx[r+1]]` |

Integer mils make every price comparison exact. All book and maker prices lie on the 0.001
grid: 994 maker legs carried data-api rounding of at most 4e-5 and are rounded.

**Fair value** (`p_model`) is the pricer of `analysis/fair_value/fair_value.py`, imported, not
copied. It uses the settings that study fixed on train:
* Student-t with nu = 4;
* EWMA vol with half-life 300 s on overlapping 30 s Binance log returns, vol multiplier 1;
* strike K = Binance TWAP60 at open;
* closing-TWAP adjustment for tau < 60 s.

At tau = 0, p_model is the Binance-TWAP outcome (known at the end). That value is used only as
the mark fallback, and the terminal step pays the true settlement.

**Tick.** `tick` is 0.01 until the first row whose top-5 book, or any maker fill up to that
row, shows a sub-cent price. From there it is 0.001 for the rest of the market (sticky).
- 366 markets switch.
- The switch comes a median 115 s after the two-sided mid first leaves [0.04, 0.96].
- 284 markets whose mid went extreme never show a sub-cent price.

So the coordinator's rule "0.001 when p < 0.04 or p > 0.96" is not what the books show
instantly. That rule is available as `tick_rule = 1`: fine once the mid leaves [0.04, 0.96]
or a sub-cent price appears, sticky.

## 2. Episode

* **Market.** Each episode draws a market uniformly from the split's list, filtered by
  `len_mask`. `cycle_stride > 0` cycles deterministically instead: env with seed s plays markets
  s, s + stride, ... of the list. `fixed_market` pins one global market index.
* **Start.** A uniform random second in the first `start_frac` = 20% of the market (or
  `fixed_start`). The episode runs to the last row.
  - Episode lengths and start phases differ, so terminals do not line up with PufferLib
    segment starts (PUFFER5 rule 7).
  - Do not depend on `horizon`.
* **Decision interval dt.** 1 s for 5m and 15m, 5 s for 4h (`dt_5m`, `dt_15m`, `dt_4h`; `dt > 0`
  overrides all three).
  - Steps per episode: about 240-300 (5m), 720-900 (15m), 2300-2880 (4h).
  - With uniform market sampling, the share of steps is about 24% 5m, 42% 15m, 34% 4h.
  - The last step of an episode may cover fewer than dt seconds.
* **Terminal.** At the last row, inventory settles at `winner_up`: Up pays 1. A negative
  inventory is a short Up, which is equivalent to holding Down at 1 - p. That step's reward
  uses the settlement in place of the mark. `terminal = 1`. The env then starts the next
  episode and writes its first observation in the same step (auto-reset).
* **Split guard.** `split` is 0 train, 1 val, 2 test. Split 2 is refused unless `allow_test=1`
  **and** the environment variable `MM_FINAL=1`. When both are set, the env prints a loud
  banner to stderr once per process, and env.py prints one too.

## 3. Actions: MultiDiscrete `{7, 7, 3}` (every head >= 2 options, every action legal)

| head | values |
|---|---|
| bid level | 0 off; 1 improve 2 ticks; 2 improve 1; 3 join best bid; 4 one tick behind; 5 two behind; 6 four behind |
| ask level | the same, relative to the best ask; improving means a lower ask |
| size | 0, 1, 2: 1x, 2x, 5x `min_order_size` (5 shares) x `size_mult` |

Quote construction at decision row t, in integer mils:
1. The base is the best bid (or best ask) of `book[t]`. If that side is empty, the base is
   p_model rounded to the tick: floor for the bid, ceil for the ask.
2. Post-only: a bid at or above the best ask is pulled back to one tick below the ask, and the
   ask symmetrically.
3. The price is kept inside [tick, 1 - tick]; outside that the side is off.
4. If our bid >= our ask, the ask moves to bid + tick. The bid has priority.

When the order reaches the exchange (match row t + L):
* It is **rejected** if it would cross the then-current opposite best (post-only). This is
  counted in `rejects`.
* An order **at the same price as our live order is kept**, with its queue position and
  remaining size. The size head is ignored until the order is filled or moved.
* The **inventory cap** Q (default 100 shares) truncates a new order to Q - inventory (bid) or
  Q + inventory (ask). If truncation leaves less than min_order_size, the order is pulled.
  Each fill is also clamped to the room left, so |inventory| <= Q always holds.
* One order per side. Fees: makers pay 0. Our quotes never take liquidity, so no taker fee arises.

## 4. Fill models (EVALUATION.md T1; one code path, three parameter sets)

For a resting bid at b, a leg is a maker leg of a taker *sell* in row u, judged against our
order as it stood at tau = u - D. Asks are symmetric.
* A leg at a price **better** than b (higher) is not ours.
* A leg **at** b first consumes the queue ahead of us.
* A leg **through** b (lower) means the taker swept our level.

| | optimistic | central | pessimistic |
|---|---|---|---|
| match delay d (D) | 0 s (1) | 3.5 s (4) | 5 s (6) |
| queue on joining | 0 (front) | displayed size at our price in book[tau] (0 when improving; 0 if beyond the 5 shown levels) | same as central |
| at-price legs | fill us fully | consume the queue, the rest fills us | consume the queue, never fill us |
| through legs | fill us fully | fill us fully, queue := 0 (level swept) | first consume the shown size still ahead of us, the rest fills us |
| cancels ahead | n/a | pro rata: after prints, `q -= cancelled * q / (shown_prev - traded_at)`, and `q <= shown` | none |
| size | leg size | leg size | 50% of the leg, at most 20% of the taker tx in total |
| participation cap | none | our fills <= 25% of all taker volume so far in the UTC minute (`part_cap`) | same as central |
| book crossed/forward-filled at tau | fills allowed | no fills | no fills |
| book older than 10 s at tau | allowed | allowed | no fills |

* Every fill is also capped by our remaining size and the inventory room.
* **No market impact.** The replay does not react to our quotes:
  - takers who hit us would partly have hit someone else;
  - other makers do not re-price around us;
  - improving the touch gets no extra flow.
* **Data finding on through-prints.** About 80-90% of our fill volume comes from legs priced
  through the pre-trade touch. Measured with the central model, L = 1, on 40 train markets, for
  quotes from join to 4 behind. Only 5-10% comes from prints at our own price, because at the
  touch the displayed queue usually absorbs at-price prints. So the central model's "through
  fills us fully" rule drives most fills. Read results with this in mind, and check the
  pessimistic model.

## 5. Accounting and reward

* **Per fill** of q shares at price p (signed sq = +q bought, -q sold):
  - `cash -= sq * p`;
  - `inventory += sq`;
  - rebate `rebate_rate * fee_rate * q * (p(1-p))^fee_exponent`, credited at the fill.

  `fee_rate` (0.07) and `rebate_rate` (0.20) come from markets.bin. Override them with
  `fee_rate` / `rebate_rate` >= 0; 0.35 is the rebate sensitivity case.
* **Liquidity rewards**: a hook, **off by default** (`liq_rewards = 0`), following the rewards
  study (`analysis/rewards/reward_terms.py`, AUG_2026 preset):
  - one sample second per UTC minute, drawn in `features.py` with a fixed seed;
  - on that second, `pi_m * Q_min / (Q_min + Q_comp)`;
  - `Q_comp` is precomputed from the replay book: the top 5 levels, the 'hi' competitor bound,
    with v = 1.5 c and min size 50;
  - our `Q_min` uses `S = ((v - d)/v)^2 * size` for our orders with size >= 50 and age >=
    `liq_moas` s, the two-sided rule inside [0.10, 0.90], and c = 3;
  - approximations: the adjusted mid ignores our own orders, and only the top 5 levels are used.

  With the default sizes (5-25 shares), our orders never qualify. Use `size_mult >= 2` to see
  rewards.
* **Wealth.** `W = cash + rebates + rewards + inventory * mark`. The mark depends on `mark_mode`:
  - 0: book mid, with p_model when the book is one-sided or invalid;
  - 1: p_model;
  - 2: the average of the two.

  At the terminal step, the settlement replaces the mark.
* **Reward per step:**
  `r = reward_scale * (W_t' - W_t - inv_penalty * inv^2 * mark(1-mark) * (t' - t) / contract_len_s)`.
  - `reward_scale` defaults to 1/Q, so |r| <= 1 except in pathological cases.
  - r is clipped to [-1, 1], and `reward_clip_frac` logs the fraction of clipped steps.
  - The rewards telescope: `sum(r) / reward_scale + inv_penalty_sum = final P&L` for every
    mark_mode (tested).
  - The inventory penalty is training shaping only. It is in the reward, but not in `pnl`.
* `reward_mode = 1` is a learnability probe: r = 1 if both heads chose "join" (3), else 0.

## 6. Observation: 50 float32, O(1), clipped to [-5, 5], last = 1

Market block, `feat.bin` columns 0-32 copied as is:

| obs | name | definition / scale |
|---|---|---|
| 0 | p_logit | logit(p_model) / 3, p clipped to [1e-4, 1-1e-4] |
| 1 | mark_logit | logit(book mark) / 3 |
| 2 | gap_c | (p_model - mark) in cents / 5 |
| 3 | dgap5_c | 5 s change of the gap, cents / 2 |
| 4 | z | ln(F/K)/sqrt(V) / 2 (+-5 at tau = 0) |
| 5 | sdv_bp | log1p(sqrt(V) in bp) / 3 |
| 6 | sigma_1m_bp | log1p(sigma * sqrt(60) in bp) / 3 |
| 7 | tau_frac | seconds_to_close / contract_len_s |
| 8 | log_tau | log1p(seconds_to_close) / log1p(14400) |
| 9 | in_twap | 1 inside the closing 60 s TWAP window |
| 10 | delta_usd | log1p(dp/dlnS, USD of underlying per Up share) / 5 |
| 11 | spread_ticks | ln(spread / tick) / 2 if two-sided, else 0 |
| 12 | bid_present | a best bid exists |
| 13 | ask_present | a best ask exists |
| 14 | imb_l1 | (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0) |
| 15 | bid_sz0 | log1p(best bid size) / 5 |
| 16 | ask_sz0 | log1p(best ask size) / 5 |
| 17-19 | flow5, flow30, flow120 | signed taker flow (buy - sell)/(buy + sell + 5) over 5/30/120 s (block-time rows <= t) |
| 20 | mret5_c | mark change over 5 s, cents / 2 |
| 21-22 | mret30_c, mret120_c | mark change over 30/120 s, cents / 5 |
| 23-25 | ulz5, ulz30, ulz120 | Binance log return over 5/30/120 s / (sigma sqrt(k)) |
| 26-28 | len_5m, len_15m, len_4h | contract-length one-hot |
| 29 | tick_fine | 1 once the tick is 0.001 (data rule) |
| 30 | book_age | log1p(book age s) / 4 |
| 31 | crossed_raw | latest raw snapshot crossed (book forward-filled) |
| 32 | mark_is_fv | mark fell back to p_model (one-sided or no book) |

Agent block (computed by the core at decision row t; the order state is the match-time state):

| obs | name | definition / scale |
|---|---|---|
| 33 | inventory | inventory / Q |
| 34 | wealth | clip(W / (0.05 Q), +-5), W = cash + rebates + rewards + inventory * mark (mark per mark_mode) |
| 35 | inv_delta | clip(inventory * delta_usd_raw / (200 Q), +-5) |
| 36 | bid_live | our bid rests at the exchange (match time) |
| 37 | ask_live | our ask rests |
| 38 | bid_dist | live: clip((bid - book mark) in cents / 2, +-5); else 0 |
| 39 | ask_dist | the same for the ask |
| 40 | bid_queue | live: clip(queue_ahead / shown size at our level, 0, 1) (shown size as tracked at the last match row); else 0 |
| 41 | ask_queue | the same for the ask |
| 42 | bid_rebate | live: clip(rebate_rate * fee_rate * p(1-p) / 0.0035, 0, 5) at our bid price; 1 = 0.2 * 0.07 * 0.25 |
| 43 | ask_rebate | the same at our ask price |
| 44 | since_fill | log1p(min(seconds since our last fill, 3600)) / log(3601) |
| 45 | pending | the latest decision's prices differ from the live orders (still in flight: latency + match delay) |
| 46 | n_fills | log1p(fills this episode) / 5 |
| 47 | bid_minus_p_c | (best bid - p_model) in cents / 5; 0 if no bid (feat col 41) |
| 48 | ask_minus_p_c | (best ask - p_model) in cents / 5; 0 if no ask (feat col 42) |
| 49 | one | constant 1 (PufferNet has no biases) |

None of these features uses data after row t, the outcome or the final price.

## 7. Log (floats; PufferLib sums them over envs and divides by n)

| field | meaning |
|---|---|
| score, pnl | episode P&L in USD: cash + rebates + rewards + inventory * settlement |
| perf | 1 if pnl > 0 |
| episode_return, episode_length | sum of rewards, number of decisions |
| pnl_trading | cash + inventory * settlement (no rebates or rewards) |
| spread_capture | sum of sq * (mid at tau - p) |
| markout_5, markout_30, markout_60 | sum of sq * (mid(tau + h) - mid(tau)), with the settlement past the end |
| inventory_pnl | pnl_trading - spread_capture - markout_60 |
| rebates, liq_rewards | USD |
| inv_penalty | USD of shaping penalty (not in pnl) |
| fills, fill_vol, buy_vol, sell_vol | counts and shares |
| max_abs_inv, final_abs_inv | inventory extremes |
| uptime_two_sided, uptime_any | fraction of match seconds with both quotes / any quote live |
| reward_clip_frac | fraction of clipped steps |
| rejects | post-only rejects at arrival |
| vol_X, pnl_X | for X in crossed, stale, onesided, last30, last10, inside2, extreme, outside, inspread: fill volume, and spread capture + 60 s markout, of fills with that flag. Flags are judged on book[tau] |
| n | episodes |

The `mid` used for attribution is always the book mark (`feat.mark`), regardless of mark_mode.

Flag definitions (EVALUATION.md T5):
* `crossed`: crossed_raw.
* `stale`: book_age > 10 s.
* `onesided`: either side empty.
* `last30` / `last10`: end - tau < 30 / < 10.
* `inside2`: our quote more than 2 ticks inside the touch.
* `extreme`: our quote <= 0.02 or >= 0.98.
* `outside`: the print is outside [best bid, best ask], which includes every sweep through the touch.
* `inspread`: the print is on the wrong side of the hit touch, i.e. a sell above the best bid
  or a buy below the best ask. The snapshot cannot explain such a print.

`outside` measured on 40 train markets (central, L = 1), from join to 4 behind:
* the book at tau flags 83-93% of fill volume;
* the book at the print row u would flag 67-74%;
* `inspread` is about 4%.

So `outside` mostly measures sweeps, not an alignment artifact. The fill logic and the flags
use the match-row book (tau = u - D), which is the pre-trade book. Under the literal T5 rule
("> 25% of P&L from one flag → invalid") this flag needs a protocol decision.

`env.py` adds `share_X` (volume share) and `pnl_share_X` (P&L share) to its info dicts.

## 8. Parameters (`MM_PARAM_NAMES`, defaults `MM_DEFAULTS`; doubles in this order)

| name | default | meaning |
|---|---|---|
| split | 0 | 0 train, 1 val, 2 test (guarded) |
| allow_test | 0 | must be 1 for split 2, and MM_FINAL=1 must be set too |
| fill_model | 1 | 0 optimistic, 1 central, 2 pessimistic |
| latency | 1 | quote placement/cancel delay L in s (0..5) |
| match_delay | -1 | d in s; < 0: model default 0 / 3.5 / 5 |
| dt | 0 | > 0: decision interval for all lengths |
| dt_5m, dt_15m, dt_4h | 1, 1, 5 | decision interval per contract length |
| Q | 100 | inventory cap (shares) |
| inv_penalty | 0.001 | lambda of the inventory-variance penalty (reward only) |
| reward_scale | 0 | <= 0: 1/Q |
| size_mult | 1 | order size multiplier (size sweep x0.5 / x1 / x2) |
| part_cap | -1 | < 0: 0 (off) optimistic, 0.25 otherwise; 0 disables |
| fee_rate, rebate_rate | -1 | < 0: markets.bin (0.07, 0.20) |
| liq_rewards | 0 | 1 turns the liquidity-reward hook on |
| liq_moas | 1 | min order age (s) for rewards |
| mark_mode | 0 | 0 mid (p_model when one-sided), 1 p_model, 2 blend |
| reward_mode | 0 | 1 = join-probe reward |
| tick_rule | 0 | 0 data-observed tick, 1 also price rule |
| len_mask | 7 | 1 5m, 2 15m, 4 4h |
| fixed_market | -1 | global market index (tests and debugging) |
| fixed_start | -1 | start offset in seconds |
| cycle_stride | 0 | > 0: deterministic market cycling (evaluation) |
| start_frac | 0.2 | random start within this fraction of the market |

The binding also takes `data_dir` (str), and env_binding passes `seed`.

**Config registry.** Envs with identical parameters share one interned config. A process can hold
at most `MM_MAX_CFG = 256` distinct configs. So per-env `fixed_market` over many markets is not
possible in one process. Evaluators use `cycle_stride` (env i plays markets i, i + stride, ...).

## 9. PufferLib 5.0 rules (PUFFER5.md)

| rule | how |
|---|---|
| 1 | pure C core, no PufferLib includes; compiles as C99 and C++17 (`g++ -std=c++17 -fsyntax-only` in development); casts on malloc/mmap |
| 2 | `mm_step(Mm*, const int* act, float* obs, float* rew, float* term, Log*)`; `mm_reset(Mm*, float* obs)` |
| 3 | float32 obs, clipped to [-5, 5], relative quantities (logits, cents, ticks, log sizes), constant 1 last |
| 4 | `MM_ACT_SIZES {7, 7, 3}`; out-of-range actions are clamped |
| 6 | dense reward written every step, scaled by 1/Q, clipped, clip fraction logged |
| 7 | auto-reset inside step; true terminal at settlement; random start offset in the first 20% |
| 8 | five flat files mmapped `PROT_READ, MAP_PRIVATE` once behind `MM_DATA_READY`; `close` never unmaps; envs hold cursors only (Mm about 0.55 kB) |
| 9 | per-env splitmix64 RNG seeded from the seed kwarg; no mutable globals in step (the config registry and log tables are written only by init / data open) |
| 10 | `mm_init` does not touch buffers; `mm_reset` writes the first obs; repeated init is safe |
| 11 | Log of floats with score, perf, n |

Floating point: build with `-ffp-contract=off` (setup.py does) so that FMA contraction does not
change results between builds. The binding and the harness then match bit for bit.

## 10. Tests (`python test_env.py`, about 2 min)

* **C vs reference:**
  - 22 envs: 14 train and 6 val markets of 5m/15m, plus 2 4h. Start offsets are random and the
    actions are sticky random, biased to the touch.
  - Configs: the 9 fill-model x latency pairs, plus mark modes 1/2 with penalty, the probe,
    tick_rule 1 with dt 3, an explicit delay with no cap and 35% rebate, pessimistic with
    delay 4, liquidity rewards on, optimistic with size 0.5 and Q 20, and fee/rebate 0 with
    dt_15m 2.
  - Every step is compared: reward (1e-7), terminal flag, observation (1e-6), state (inventory,
    cash, rebates, rewards, wealth, penalty, order prices/remaining/queue, attribution; 1e-9
    relative) and the first-episode Log (1e-4 relative). The auto-reset episode is compared
    too.
* Reward telescoping for mark_mode 0/1/2, and the P&L identity.
* Lookahead: perturbing the data after the cut changes nothing before it (and does change things after).
* Split guard: refused without allow_test, refused without MM_FINAL, opens with a banner when
  both are set.
* PufferLib 3.0 binding == core, bit for bit, on the same seeds and actions.

## 11. Throughput (`python bench.py`)

Measured on this 4-core VM (Xeon, 2.1 GHz, 2 MB L2/core) while other agents' jobs ran (load 2-12).
Single thread, 4096 envs, random actions, central model, latency 1. The step design:
* every market-only quantity is precomputed;
* integer-mil prices make comparisons exact;
* the data is mmapped once and shared;
* no allocation, no libm transcendental calls in the step (log1p via lookup tables);
* quiet seconds (no legs, no activation, unchanged ladder) are skipped in bulk, exactly;
* an Mm of 552 B per env.

| measurement | steps/s |
|---|---|
| env only, PufferLib 3.0 binding, default dt (1/1/5 s; 4h steps cover 5 s) | 3.8-4.3 M (3.9-4.8 M per CPU-second) |
| same, dt = 1 for every length | 4.7-4.9 M |
| bare core (ctypes harness, no Python) | 3.7-4.4 M; 15m only 6.5 M; 5m only 4.8-5.5 M; 4h only 4.6 M |
| bare core, constant "join" actions | 4.8-5.0 M (dt = 1: 5.4 M) |
| bare core, all 4096 envs on one cache-resident market | 9.5-13 M |
| PPO end to end (PuffeRL, MLP 128, horizon 64, minibatch 32768, 1 update epoch), 1 torch thread | 154 k (rollout alone 359 k) |
| same, 4 torch threads (machine load about 2) | 312 k (rollout alone 571 k) |

* **PPO is bound by the policy, not the env.** One vectorized step of 4096 envs costs about
  1 ms in C. The MLP forward costs about 4 ms and PufferLib's multi-discrete sampling about
  11 ms on 1 thread.
* **Oversubscription.** More torch threads than idle cores collapses throughput by 10-50x. On a
  shared machine, use `--threads 1`.
* **Where the env's time goes.**
  - Cross-env software prefetch helped about 10% on 15m only (`mm_prefetch_state` /
    `mm_prefetch_data` / `mm_prefetch_legs`, used by `mm_harness.c` and available to 5.0
    wrappers). A huge-page copy of the data did not help.
  - The remaining cost is about 550 instructions per step: the step itself, the 17-feature
    agent block and quote construction. It also includes data-dependent branches across
    markets of different lengths.

## 12. Data issues found

* **Through-prints.** Most of our fill volume comes from prints through the pre-trade touch
  (section 4), and the T5 `outside` flag marks 83-93% of fill volume (section 7). The alignment
  is correct: with the print-row book the share would be 67-74%.
* **Tick.** Sub-cent prices appear only in 366 markets, a median 115 s after the mid first
  leaves [0.04, 0.96]. 284 markets go extreme but never show a sub-cent price (section 1).
* **Maker legs.**
  - The maker legs of every taker tx sum to the taker size within 0.0094 shares (17 txs differ
    by more than 1e-3 shares, all float32 rounding on tiny trades).
  - 994 maker prices are off the 0.001 grid by at most 4e-5 (data-api rounding) and are rounded.
  - Every maker leg's aggressor sign matches its taker leg.
* **Leg counts.** 412,792 maker legs fall in the market windows. 20,424 fall before the open
  and 1,192 after the end; both sets are dropped.
* **Volume check.** Per-row volume from the legs matches `ticks.bin` buy_vol + sell_vol within
  0.01 shares.
* **Liquidity rewards.** Orders of 5-25 shares never reach the 50-share minimum, so rewards
  need `size_mult >= 2`. Rewards are off by default.

## 13. Limitations and assumptions

* **Replay.**
  - No market impact or competitor response (section 4).
  - The queue position is estimated from L2 (no L3).
  - Through-prints fill us fully in the central model.
  - Displayed sizes come only from the top 5 levels; beyond them the queue is assumed 0.
  - Same-price re-quotes keep queue and size.
  - One order per side.
* **Timing.**
  - Prints are 1-s block times with an assumed constant match delay.
  - Book snapshots are sparse (book_age p90 12-31 s); forward-filled crossed books block fills
    in the central and pessimistic models.
* **Fair value.** It uses Binance, while settlement uses the Chainlink 60 s TWAP (basis about
  2 bp). HYPE uses the Binance perp.
* **Tick.** The tick rule is data-observed and conservative (section 1).
* **Rewards.** The liquidity rewards are a hook with assumed pools; they are off by default.
* **Attribution.** Markouts in the Log look past the decision row by design. They never feed
  the observation or the reward.
* **Single agent per market.** There is no portfolio or cross-market inventory and no hedge action.
