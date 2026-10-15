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

### R1: BC warm start (finished ~10:15 UTC). Not improved under D1.3.
- 16 configs × 3 seeds. Half imitate the tuned fair_value quoter
  (`baselines/tuned.json`), half the active one (`tuned_active.json`). After
  imitation, PPO trains with a decaying KL penalty to the BC policy.
- **The `tuned` teacher gives the only profitable RL policies so far**:
  - seed means of +$110 to +$144 total on val, about +$0.45–0.58 per market;
  - small SEs, for example 140.6 ± 7.8;
  - but only 4.0–4.7 fills per market. That is below D1.3's threshold of 5,
    so these policies can't be selected.
  - They reproduce the low-activity, directional teacher.
- The best D1.3-passing config is `c13_tuned_active`: −$147.7 ± 4.2 total.
  That matches the main config.

### R2: off-prior init (finished ~10:35 UTC). Not improved.
- **A strong prior towards "off"** gives about 0 fills, i.e. it learns not to
  quote: +$6 ± 6 total.
- **A weak prior** (0.5, lr 2×): −$69 ± 109 total, 10 fills per market. This
  is the best D1.3-passing result, but within noise of the previous best.

### Stopping rule and D6 (10:50 UTC)
- R1 and R2 both failed to improve, so `auto` stopped (D4 streak rule). That
  happened before R7 ran.
- D6, recorded after seeing these results, resumes with R7, then R3, R4, R5
  (only if the sim gate passes), R6, then finalize, with the 15:30 deadline
  unchanged.
- The finalize step that had started was stopped.

### Sim gate (10:44 UTC): still fails, so R5 is skipped
- **G2 rank order now passes**: no_quote > fair_value > sym_touch, as in the
  replay. Two fixes got it there:
  - print impact removed;
  - bursty, calibrated trade arrivals.

  sym_touch's total P&L and its hourly KS also fall inside the band now.
- **Still failing:** fair_value quotes are adversely selected less than in the
  replay. Its 5 s markout is −1.9c in the sim vs −3.05c real, and its total is
  −54 vs −797. G3 has 5 of 22 components inside the band.
- **On-chain pre-period trades** (Jul 21 – Aug 20, all 31 days converted) made
  the gate worse, so the final params don't use them.
- **Note on the gate's design (from the sim agent):** G1 and G3 together are 26
  separate checks against 90% bands, so a perfect sim would pass all of them
  only about 7% of the time. The current sim also fails looser rules, so the
  gate stays as specified, and R5 does not run.

### Container restarts (11:38 and 12:38 UTC)
- The cloud container is reclaimed when the session sits idle, and that kills
  detached training. R7 lost about an hour as a result: only 8 of its 16
  configs finished.
- **Fix (12:40):** the chain was relaunched; rounds resume from their run
  dirs. A harness-tracked background task now waits on the chain to keep the
  session alive, and it is re-armed at every hourly check-in.
