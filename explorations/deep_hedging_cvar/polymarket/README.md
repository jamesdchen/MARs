# Polymarket market making with RL (PufferLib)

An RL market maker for Polymarket's crypto up/down binary markets (5m, 15m and 4h contracts, 8
coins). It is trained with PPO in a fast C environment that replays real 1-second order books and
trades. The work reuses the CVaR and PufferLib infrastructure from the deep-hedging explorations
next to this folder.

## Result (test day, run once under a pre-registered protocol)

**No demonstrated edge.** The selected RL policy makes markets and is not just a directional bet.
It loses less than most classical quoters that are forced to trade. It does **not** beat not
quoting, and it only makes money under the optimistic fill model. Full report:
[`final/RESULTS.md`](final/RESULTS.md).

Test day 2026-08-24 (127 markets). Central fill model, 1 s quote latency, 20% maker rebate,
liquidity rewards off. Five training seeds.

| | test P&L, total | per market |
|---|---:|---:|
| **RL** (PPO trained on fill markouts, R7 `c02_h120`), seed mean | **−$14** | −$0.11 |
| not quoting | $0 | $0 |
| pre-registered comparator (fair-value quoter, best on val) | −$42 | −$0.33 |
| best baseline forced to trade (Avellaneda–Stoikov in logit) | +$2 | +$0.02 |
| fair-value quoter forced to trade | −$127 | −$1.00 |
| symmetric quote at the touch, forced to trade | −$202 | −$1.59 |

- **Primary endpoint:** RL minus the comparator is +$28, 95% CI [−$259, +$294], so the verdict is
  **FAIL**. The concentration check (T7) also fails: without its top 5 markets, RL makes −$194.
- **What RL did learn:**
  - It makes markets: 21 fills per market, two-sided 18% of the time.
  - It is not directional (T6) and does not exploit replay artifacts (T5).
  - It beat three of the four forced-to-trade baselines, though every one of those CIs includes
    zero.
- **Where the money goes:** spread capture +$69, adverse selection −$151, inventory +$104,
  resolution −$49, rebates +$13. Adverse selection, meaning being picked off before Binance-driven
  price moves, outweighs the spread on every method tested.
- **Sensitivity:**
  - Fill model at 1 s latency: optimistic +$245, central −$14, pessimistic −$70.
  - Seeds: −$148 to +$63.
  - By contract length: positive on 4h markets (+$118), negative on 5m and 15m.

## v2 result (Jun 15)

**FAIL again, and worse.** Protocol v2 kept improving on validation (Aug 23–24, 374 markets) in
the local sweep and five parallel workers. It then froze one method and ran it once on an
untouched test set: every BTC 5-minute market of 2026-06-15, 288 markets from Marketlens. Full
report: [`final/RESULTS_V2.md`](final/RESULTS_V2.md). Selection: [`final/v2/SELECTION.md`](final/v2/SELECTION.md).

**Selected method: `v2s1/c00`.**
- A reward-mode-2 MLP trained on 60 s fill markouts. It quotes every contract length; 5 seeds.
- It had the best val v2 seed mean among methods that count as market making (D1.3) and have at
  least 3 seeds, judged with the rule frozen in the test plan (DECISIONS.md D8, D13).
- **It already lost on val:** −$35 against no-quote's $0. **No D1.3-passing method made money on
  val v2.** Every profitable method was directional, mostly through 4h resolution P&L.

| Jun 15, central fills, 1 s latency | P&L, 288 markets | per market |
|---|---:|---:|
| **RL `v2s1/c00`**, seed mean of 5 | **−$926** | −$3.21 |
| not quoting | $0 | $0 |
| pre-registered comparator (fair-value quoter) | −$431 | −$1.49 |
| best forced-to-trade baseline (fair-value, active) | −$101 | −$0.35 |

- **Primary endpoint:** RL minus the comparator is −$495, 95% CI [−$1,610, +$551], so the verdict
  is **FAIL**. The pessimistic-fill, 2 s latency and concentration criteria fail too.
- **RL still makes markets.** It fills 26 times per market, is not directional and does not
  exploit replay artifacts. Adverse selection (−$1,735) swamps spread capture (+$472).
- **Fill model at 1 s:** optimistic +$521, central −$926, pessimistic −$1,128. Every seed loses
  (−$654 to −$1,252).
- **Fills on bookless seconds (D10, reported only).** Fills in the seconds before a market has a
  book cost RL $444. Without them RL is still −$481.
- **Caveats (fixed before the run).** The test is BTC 5m only, from a different data source
  (Marketlens), two months before training. So it tests robustness to a shift in source and
  period, not the full portfolio.

## How we got here

1. **Data.**
   - PolyOrderBooks L2 snapshots: 805 markets, Aug 21–24 2026.
   - Taker and maker fills from the data-api, which matches on-chain records exactly.
   - Binance 1 s prices.
   - Days split: train Aug 21–22, val Aug 23, test Aug 24.
2. **Environment.** A C core that ports to PufferLib 5.0, checked bit-for-bit against a numpy
   reference. It has queue-aware fills (optimistic, central and pessimistic), 0–2 s latency, and a
   digital-option fair value from Binance.
3. **Baselines** were tuned on val with the same trial budget as RL: none of them has a robust edge.
4. **PPO.**
   - A 64-trial sweep plus improvement rounds: behaviour-cloning warm start, an off-quote prior,
     fill-markout reward, mark choice, longer runs and a recurrent policy.
   - 1,489 val evaluations in total. All of them are logged and counted for multiple testing.
   - The overnight log is in `train/OVERNIGHT.md`.
5. **Honesty notes.**
   - **Amendment D6** (`eval/DECISIONS.md`) was made *after* seeing R1/R2 val results. It let the
     fill-markout round run after a stopping rule had cut it off. The test split stayed untouched
     until the plan was frozen (sha256 `75915990…`).
   - **R1's only profitable RL policies** (+$0.45–0.58/market on val) traded fewer than 5 times
     per market. They were excluded as not market making, by a rule fixed before training.
   - **Container restarts** at about 11:38 and 12:38 UTC killed training mid-round. The rounds
     resumed from their run directories. R6 (recurrent policy) was stopped by the stopping rule
     before it finished.
   - **The simulator** (`sim/`) failed its validation gate, so sim pretraining (R5) never ran.

## What would change the answer

- **More data.** Four days is very little: the test day had lower volatility and fewer trades
  than training. `recorder/` (live L2 recorder) and `onchain/` (months of fills) are ready.
- **Rewards earned before the window.** In August, liquidity rewards were paid mostly in the
  ~24 h *before* each market's trading window, and the dataset doesn't cover that period
  (`analysis/rewards/`).
- **A GPU run on PufferLib 5.0** (`puffer5/colab.ipynb`). It allows far more training per config;
  on CPU, PPO was policy-bound at about 160k steps/s per process.
- **Live shadow quoting.** Before any real money, measure real fills and queue position against
  the replay's fill models.

## Map

| folder | contents |
|---|---|
| `data/` | data pipeline, splits, test-set hash |
| `mm.h`, `binding.c`, `env.py`, `reference.py`, `test_env.py`, `ENV.md` | C replay env, numpy reference, equivalence tests |
| `baselines/` | classical quoters and their tuning |
| `train/` | PPO, sweeps, rounds, ledger, overnight log |
| `final/` | pre-registered test plan, runner and report |
| `eval/` | protocol (`EVALUATION.md`), amendments (`DECISIONS.md`), statistics |
| `analysis/` | fair-value/hedging study and rewards/rebates study |
| `sim/` | calibrated synthetic generator (gate fails) |
| `puffer5/` | PufferLib 5.0 port and Colab notebook |
| `recorder/`, `onchain/` | data collection tools |
| `research/` | literature review and on-chain notes |
