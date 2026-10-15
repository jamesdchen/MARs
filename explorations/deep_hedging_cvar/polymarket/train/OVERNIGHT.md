# Overnight log: PPO improvement rounds (val split only)

The rules for these runs are in `eval/DECISIONS.md` D3–D5. Every number
below is on the val split. The test split stays untouched until the finish
step.

## The bar to beat (val, central fills, 1 s latency)

| comparator | $ / market | source |
|---|---|---|
| no quote, also the best tuned unconstrained baseline | ≈ 0 | `baselines/eval_val.json` |
| best baseline forced to keep trading (as_logit) | −0.30 | `baselines/tuned_active.json` |

## Main pipeline (finished about 09:37 UTC)

This is the 64-trial sweep at 25M steps per trial. The config was picked by
the mean of 3 seeds among D1.3-passing configs, then trained on 5 seeds at
100M steps each.

- **Sweep.** Every single-seed trial lost money, from −$2.7 to −$5.9 per
  market. The price heads stayed close to uniform (entropy 1.6–1.94 out of a
  maximum of 1.95).
- **Selected config.** MLP 64×2 with `mark_mode=1` (fair-value mark),
  `inv_penalty=0.0037`, `lr=3.9e-4`, `ent_coef=3.7e-4`, `gamma=0.9988`.
  Its 3-seed sweep mean was −$149 in total, about −$0.60 per market.
- **5 final seeds at 100M steps.**

  | | central fills | pessimistic fills |
  |---|---|---|
  | P&L per market, mean (seed range) | **−$0.99** (−1.68 to −0.35) | −$0.54 |

  The central-fill P&L breaks down as:
  - spread capture +$64 in total;
  - 60 s markout −$327;
  - rebates +$17;
  - inventory ≈ 0.

  The median market's P&L is $0, because most markets get no fills. Two-sided
  uptime is 3%, and maximum |inventory| is about 99, at the 100-share cap.
- **Reading.** The policy quotes one side at a time, so it piles up inventory
  until it reaches the cap, and those fills get picked off. It does worse than
  both bars. Rounds R1 (behaviour-cloning warm start), R2 (off-prior) and R7
  (fill-markout reward) target this failure.

## Rounds

(Appended as each round finishes.)
