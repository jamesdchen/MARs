# w3: architecture and PPO optimisation (crypto v2), stopped by the D11 pivot

**Status: incomplete by design.** At 20:18 UTC the coordinator relayed the user's pivot (DECISIONS.md D11)
and asked for no new crypto runs. The planned sweep had 32 configs x 3 seeds at 25M steps, then the top 3 x 5 seeds
(`sweep_w3.py list`). It was stopped after the 12 recurrent runs it had started; 9 of them finished and were logged. The MLP-size
group (b) and the PPO-knob group (c) never ran, nor did the top-3 x 5-seed stage. **Nothing here is a selection
candidate:** every config is far below no-quote ($0) and below the v1 best (R7 `c02_h120`, -$0.25/market on v1 val).

## Setup

- Hashes verified: `mm.h` 9189f83a..., the build_v2 markets file and the four data/build binaries all match PROTOCOL.md.
  `test_env.py` printed ok.
- Base config: `train/selected.json` (reward_mode 2, markout_h 120, mark_mode 1, inv_penalty 5.2e-5,
  lr 6.8e-4, ent 4.0e-4, gamma 0.9988, lambda 0.979, 4096 envs, horizon 64, mb 8192, 1 epoch), 25M steps.
  Policy replaced by the recurrent PufferNet replica (`--policy puffernet`, R6).
- Train on build_v2 split 0. Val v2 = build_v2 split 1, **374 markets** (checked in every ledger row).
  Central fills, 1 s latency.
- Ledger: `trials.jsonl`, 18 val evaluations (9 runs x {central_1s, pess_1s}). Final evaluations only;
  no periodic val evaluations were made.

## Results (val v2, central 1 s, final checkpoint; seed mean +- SE over 3 seeds)

| config | n | P&L/mkt (mean ± SE) | min / max seed | total | fills/mkt | T6 | T5 | D1.3 (agg; seeds) | 5m | 15m | 4h | pess | ent h0/h1/h2 | KL | EV | gap (train−val) | SPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pnet_L1_H256 | 3 | -1.356 ± 0.985 | -3.32 / -0.31 | -507 | 33.9 | — | — | pass; 3/3 | -1.62 | -1.19 | -0.12 | -1.49 | 1.61/1.00/0.98 | 0.0002 | 0.39 | -1.84 | 25k |
| pnet_L2_H128 | 3 | -1.485 ± 0.523 | -2.53 / -0.93 | -555 | 28.6 | — | — | pass; 3/3 | -1.87 | -1.59 | 1.63 | -1.21 | 1.47/1.46/0.90 | 0.0002 | 0.37 | -0.51 | 31k |
| pnet_L2_H256 | 3 | -2.130 ± 0.489 | -2.74 / -1.16 | -797 | 37.4 | — | — | pass; 3/3 | -2.60 | -1.96 | 0.49 | -1.59 | 1.43/1.42/0.88 | 0.0002 | 0.42 | -0.92 | 15k |

Column notes:
- **D1.3** is computed on the seed-mean flags (fills >= 5, T6 <= 0.5, T5 <= 0.25). T6 and T5 are undefined
  ("—") because P&L and fill P&L are negative; under D1.3 that counts as passing those two rules. All seeds
  pass D1.3 on activity, but they pass by losing money, not by making markets profitably.
- **5m / 15m / 4h** are P&L per market for each contract length.
- **ent h0/h1/h2** is the mean policy entropy per head over the last 20% of epochs. The bid and ask price
  heads have 7 actions (max entropy ln 7 = 1.95); the size head has 3 (max ln 3 = 1.10).
- **KL** is the approx KL per update and **EV** the explained variance, both averaged over the same last 20%.
- **gap** is train-split P&L per market minus val P&L per market.

Per seed, P&L per market and fills per market:

| config | s0 | s1 | s2 |
|---|---|---|---|
| pnet_L1_H256 | -0.31 (25) | -0.44 (14) | -3.32 (63) |
| pnet_L2_H128 | -0.93 (15) | -0.99 (43) | -2.53 (28) |
| pnet_L2_H256 | -1.16 (16) | -2.49 (54) | -2.74 (42) |

## Training-stability diagnostics

- **The price heads stay close to uniform.** Bid-head entropy is 1.43–1.61 out of 1.95, and the ask head's is
  1.00–1.46. Only the size head has collapsed (0.88–0.98 of 1.10). This is the same v1 symptom: with the selected
  config's small entropy bonus and an approx KL of about **2e-4 per update** (clip 0.2), each update barely moves the
  policy. Over 25M steps the recurrent policy does not get far from its initialisation.
- **Explained variance 0.37–0.42.** The critic explains well under half of the markout-reward return, so the
  advantages stay noisy.
- **Seed variance is large.** Within each config the worst seed loses 3–10x what the best one does, and losses
  grow with fill count: the seeds that quote most (40–60 fills per market) lose the most. This repeats the v1
  adverse-selection finding.
- **Train-val gap is negative** (train P&L per market is *worse* than val by 0.5–1.8). This is not overfitting.
  Train days are simply harder, or the policy is equally bad everywhere.
- **Bigger recurrent networks did worse.** L2/H256 is worse than L2/H128 and costs half the throughput
  (15k vs 31k SPS, against about 140k for the MLP). L1/H256 has the best single seed (-0.31) but the worst
  one too (-3.32).
- **4h markets come out positive in 6 of 9 seeds** (up to +6.8 per market), while 5m and 15m lose in every seed.
  There are only a few 4h markets in val, so this is noise-prone. It is a pointer for the per-length
  specialist line (w2), not a result.

## What worked, what didn't, recommendation

- **Didn't work:** the recurrent PufferNet replica at these sizes, with the selected PPO settings and 25M steps.
  All three configs lose 1.4–2.1 dollars per market.
- **Untested (pivot):** MLP width and depth, separate trunks, update_epochs, minibatch, horizon, lambda,
  num_envs, clip, grad norm and value-function settings. The driver `sweep_w3.py` still defines them
  (`configs()`) and is resumable if crypto work restarts. Given the tiny approx KL, I would test
  update_epochs 2–4 and a larger entropy coefficient or learning rate first.
- **Recommended candidate from w3:** none. No w3 config beats no-quote or the v1 best.

## Files

- `sweep_w3.py`: config definitions and a resumable 4-way runner (it calls the shared `train/train_ppo.py`
  unchanged), plus `follow`, which keeps the cores busy with reseeds.
- `analyze_w3.py`: per-config seed means, per-length P&L and diagnostics (it writes the table above).
- `trials.jsonl` / `runs.jsonl`: the ledger and run index.
- `ckpt/<config>_s<seed>.pt` + `_config.json`: all 9 checkpoints of the 3 configs that ran.

After the pivot, w3 works on D11 track E (election recording): see `analysis/v3_elections/`.
