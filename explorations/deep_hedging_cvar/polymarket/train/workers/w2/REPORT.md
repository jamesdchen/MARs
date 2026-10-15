# w2: per-contract-length specialists (crypto v2), stopped early by the D11 pivot

Worker w2, 2026-10-15. Everything here is **val v2** (`data/build_v2`, split 1, 374 markets: 223 5m,
119 15m, 32 4h), central fills, 1 s latency, greedy policy. Training is on split 0 with `len_mask`
(bit 1 = 5m, 2 = 15m, 4 = 4h). Every run starts from `train/selected.json` (reward_mode 2), 25M steps.
No test data was touched. Ledger: `trials.jsonl` (112 rows: one all-markets row per run from
`train_ppo.py` plus one derived length-filter row per run from `analyze_w2.py`).

**Status: incomplete.** The coordinator's PIVOT (DECISIONS.md D11, ~20:20 UTC) stopped new crypto runs.
Done: all 40 seed-1 runs (4 masks x 10 configs) and 16 seed-2 runs (4h x 10, 15m c00–c05). Not run:
seed 3, the 5-seed reruns, the 4h close-inventory configs c10–c13 and the 50M steps check
(4 of those were started by a scripting error, killed within a minute, and never evaluated).

## TL;DR

1. **No specialist makes markets on val v2.** Every specialist with positive own-length val P&L fails D1.3,
   because T6 > 0.5: its profit is inventory P&L, not spread.
2. **4h specialists show val profit: +$100 to +$170 on the 32 4h markets for most configs.** The
   decomposition shows directional inventory holding, not market making. For example, 4h_c01 seed 2:
   - spread capture: +$28;
   - 60 s markout: −$58 (adverse selection);
   - inventory P&L: +$237;
   - rebates: +$10.

   T6 is 1.1–1.7. The P&L is concentrated: the top 3 of 32 markets give $111–154 of each run's
   total, and without them 4 of 6 top-config runs are below $20. With 32 markets this is
   settlement and drift noise as much as skill.
3. **15m and 5m: nothing beats $0** on own-length val. The best are 15m_c05, −$3 ± 98 (2 seeds), and
   5m_c08, −$62 (1 seed). Every 5m/15m specialist that passes D1.3 loses money.
4. **Portfolio of specialists, by P&L only as assigned:** 5m off, 15m off, 4h = 4h_c01.
   - Combined val v2: **+$169 ± 47** (2 seeds: +$122, +$216). All of it comes from 4h.
   - Selection bias: this is the maximum over 10 configs, picked on the same val.
   - Over 374 markets it is 1.6–2.1 fills/market, T6 > 1, so **D1.3 fails**.
   - **With D1.3 required, the portfolio is all off: $0.**
5. **Recommended candidate: none for market making.** If the coordinator wants a 4h-only directional
   reference, use 4h_c03. It has the most stable seeds (+$131, +$168) and the most fills (51/4h market),
   but it is still directional (T6 1.4).

## Table (own-length val v2 P&L, seed mean ± SE; D1.3 = fraction of seeds whose own-length view passes)

Columns:
- "own-length" is P&L on the specialist's own lengths. The other lengths are set to no-quote
  (`evaluator.filter_records`).
- "per mkt", "fills/mkt" and "T6" are over its own-length markets only.
- The 5m, 15m and 4h columns are the same checkpoint run on every market. Off-mask columns show how a
  specialist generalises; they are not used.
- D1.3 counts T6 as passed when P&L ≤ 0 (evaluator convention), so losing runs "pass" trivially.

| run | seeds | own-length val P&L $ | per mkt | fills/mkt | T6 | D1.3 | 5m | 15m | 4h |
| 4h_c01 | 2 | +169.03 ± 47.34 | +5.28 ± 1.48 | 21.4 | 1.20 | 0.00 | -696.89 ± 73.78 | -282.32 ± 91.18 | +169.03 ± 47.34 |
| 4h_c03 | 2 | +149.42 ± 18.81 | +4.67 ± 0.59 | 51.5 | 1.40 | 0.00 | -1012.02 ± 56.87 | -695.11 ± 1.40 | +149.42 ± 18.81 |
| 4h_c00 | 2 | +125.30 ± 10.19 | +3.92 ± 0.32 | 25.6 | 1.11 | 0.00 | -463.04 ± 52.99 | -224.00 ± 52.61 | +125.30 ± 10.19 |
| 4h_c04 | 2 | +111.47 ± 6.65 | +3.48 ± 0.21 | 12.6 | 1.16 | 0.00 | -446.28 ± 23.88 | -183.99 ± 38.85 | +111.47 ± 6.65 |
| 4h_c07 | 2 | +104.02 ± 5.08 | +3.25 ± 0.16 | 22.2 | 1.38 | 0.00 | -848.61 ± 38.94 | -361.26 ± 16.18 | +104.02 ± 5.08 |
| 4h_c02 | 2 | +99.35 ± 12.62 | +3.10 ± 0.39 | 20.1 | 1.46 | 0.00 | -904.60 ± 208.27 | -355.86 ± 6.16 | +99.35 ± 12.62 |
| 4h_c09 | 2 | +89.11 ± 32.86 | +2.78 ± 1.03 | 37.2 | 1.67 | 0.00 | -888.42 ± 304.79 | -589.83 ± 0.32 | +89.11 ± 32.86 |
| 4h_c08 | 2 | +81.84 ± 133.06 | +2.56 ± 4.16 | 26.0 | 0.54 | 0.50 | -925.57 ± 143.26 | -302.78 ± 41.64 | +81.84 ± 133.06 |
| 4h_c05 | 2 | +16.94 ± 164.11 | +0.53 ± 5.13 | 47.3 | 0.66 | 0.50 | -1341.64 ± 86.82 | -790.73 ± 117.70 | +16.94 ± 164.11 |
| 15m_c06 | 1 | +4.36 | +0.04 | 36.3 | 33.27 | 0.00 | -438.78 | +4.36 | -34.75 |
| 4h15m_c02 | 1 | -2.05 | -0.01 | 23.6 | 0.00 | 1.00 | -336.68 | -123.91 | +121.87 |
| 15m_c05 | 2 | -3.45 ± 97.52 | -0.03 ± 0.82 | 19.8 | 0.96 | 0.50 | -281.46 ± 79.53 | -3.45 ± 97.52 | +33.65 ± 19.94 |
| 4h_c06 | 2 | -13.16 ± 159.03 | -0.41 ± 4.97 | 30.0 | 0.52 | 0.50 | -741.98 ± 200.29 | -369.19 ± 28.82 | -13.16 ± 159.03 |
| 4h15m_c04 | 1 | -52.13 | -0.35 | 12.4 | 0.00 | 1.00 | -181.51 | -10.97 | -41.16 |
| 5m_c08 | 1 | -62.18 | -0.28 | 12.9 | 0.00 | 1.00 | -62.18 | +21.37 | -32.97 |
| 15m_c08 | 1 | -67.26 | -0.57 | 22.7 | 0.00 | 1.00 | -296.83 | -67.26 | -60.22 |
| 5m_c00 | 1 | -69.83 | -0.31 | 7.5 | 0.00 | 1.00 | -69.83 | -21.14 | -34.06 |
| 4h15m_c03 | 1 | -74.27 | -0.49 | 44.8 | 0.00 | 1.00 | -976.15 | -236.85 | +162.58 |
| 5m_c02 | 1 | -75.03 | -0.34 | 14.1 | 0.00 | 1.00 | -75.03 | -28.61 | -8.70 |
| 15m_c02 | 2 | -84.56 ± 40.26 | -0.71 ± 0.34 | 16.5 | 0.00 | 1.00 | -323.12 ± 81.64 | -84.56 ± 40.26 | -52.58 ± 61.63 |
| 5m_c07 | 1 | -85.90 | -0.39 | 16.5 | 0.00 | 1.00 | -85.90 | +37.01 | -26.95 |
| 4h15m_c09 | 1 | -88.58 | -0.59 | 39.2 | 0.00 | 1.00 | -734.76 | -241.42 | +152.84 |
| 4h15m_c06 | 1 | -103.04 | -0.68 | 35.8 | 0.00 | 1.00 | -914.82 | -268.11 | +165.07 |
| 5m_c03 | 1 | -107.37 | -0.48 | 17.5 | 0.00 | 1.00 | -107.37 | -24.74 | +0.56 |
| 15m_c00 | 2 | -108.54 ± 79.37 | -0.91 ± 0.67 | 34.8 | 0.00 | 1.00 | -487.13 ± 183.49 | -108.54 ± 79.37 | -40.26 ± 29.22 |
| 5m_c06 | 1 | -109.91 | -0.49 | 10.0 | 0.00 | 1.00 | -109.91 | -40.49 | +12.15 |
| 5m_c09 | 1 | -122.54 | -0.55 | 18.7 | 0.00 | 1.00 | -122.54 | -114.20 | +19.95 |
| 15m_c01 | 2 | -128.90 ± 49.97 | -1.08 ± 0.42 | 21.9 | 0.00 | 1.00 | -251.10 ± 79.63 | -128.90 ± 49.97 | +19.92 ± 56.84 |
| 5m_c01 | 1 | -175.81 | -0.79 | 15.0 | 0.00 | 1.00 | -175.81 | -71.68 | -15.16 |
| 15m_c04 | 2 | -186.63 ± 25.97 | -1.57 ± 0.22 | 36.7 | 0.00 | 1.00 | -622.32 ± 211.52 | -186.63 ± 25.97 | -85.11 ± 108.46 |
| 15m_c07 | 1 | -191.15 | -1.61 | 24.2 | 0.00 | 1.00 | -499.15 | -191.15 | -56.60 |
| 5m_c05 | 1 | -196.90 | -0.88 | 16.4 | 0.00 | 1.00 | -196.90 | -75.19 | -27.36 |
| 5m_c04 | 1 | -199.76 | -0.90 | 16.2 | 0.00 | 1.00 | -199.76 | +34.78 | -44.80 |
| 4h15m_c01 | 1 | -288.23 | -1.91 | 25.7 | 0.00 | 1.00 | -559.42 | -247.64 | -40.59 |
| 4h15m_c05 | 1 | -297.08 | -1.97 | 42.2 | 0.00 | 1.00 | -931.57 | -296.95 | -0.14 |
| 4h15m_c07 | 1 | -298.80 | -1.98 | 43.5 | 0.00 | 1.00 | -612.23 | -391.81 | +93.01 |
| 4h15m_c08 | 1 | -314.08 | -2.08 | 37.0 | 0.00 | 1.00 | -567.36 | -397.17 | +83.09 |
| 4h15m_c00 | 1 | -430.12 | -2.85 | 49.8 | 0.00 | 1.00 | -884.88 | -433.10 | +2.99 |
| 15m_c03 | 2 | -471.35 ± 185.41 | -3.96 ± 1.56 | 56.7 | 0.00 | 1.00 | -929.40 ± 107.20 | -471.35 ± 185.41 | +82.65 ± 24.27 |
| 15m_c09 | 1 | -581.50 | -4.89 | 68.9 | 0.00 | 1.00 | -618.48 | -581.50 | +3.44 |

Configs (deltas from `selected.json`: lr 6.8e-4, ent 4.0e-4, inv_penalty 5.2e-5, markout_h 120, hidden 64):
c00 base; c01 markout_h 60; c02 markout_h 300; c03 hidden 128; c04 lr 3e-4; c05 lr 1.5e-3;
c06 ent 3e-3; c07 inv_penalty 1e-3; c08 inv_penalty 1e-2; c09 markout_h 300 + hidden 128 + inv_penalty 1e-3.

## Steps check for 4h

A 4h episode is about 2300–2880 steps, so 25M steps is about 9.6k 4h episodes (~100 per train 4h market).
That is ~3x the 4h experience of the mixed-length selected run, which spent ~34% of its steps on 4h.
The planned 50M check was cancelled by the pivot. The two seeds per config agree in sign for 4h, but not in size.

## What worked, what didn't

- **Worked:** nothing that passes D1.3.
- **Didn't work:**
  - 5m and 15m specialists. Restricting the training distribution does not fix adverse selection:
    60 s markouts outweigh spread capture in every run.
  - Inventory penalties large enough to stop holding (c08: inv_penalty 1e-2). In seed 1 the 4h profit
    vanishes (−$51, inventory P&L ≈ 0).
- **Pattern:** in 4h markets, takers who hit us are informed at 60 s, but the price later reverts or
  settles in our favour often enough on these 32 markets.

## Files

- `sweep_w2.py`: runner. It includes the planned c10–c13 configs and `check4h` (not run).
- `analyze_w2.py`: filters, aggregates and builds the portfolio, writing `summary.json`.
- `ckpt/`: final checkpoints and `config.json` for the top 3 configs by own-length val P&L
  (4h_c01, 4h_c03, 4h_c00), seeds 1–2.
- `trials.jsonl`: the val ledger.
- `runs.jsonl`: run summaries from `train_ppo.py`.
