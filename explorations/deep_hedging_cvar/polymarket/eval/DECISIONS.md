# Protocol decisions made during development

Each entry was recorded before any test-split run, together with the reason
it was needed. Every entry extends `EVALUATION.md` and supersedes it where
the two differ.

## D1. Comparators, given that no baseline has a robust edge on val (2026-10-15)

**What we saw.** We tuned four baselines on val (64 TPE trials each; central
fills, 1 s latency). Every one that actually trades loses money: spread
capture is smaller than the 60 s markout. The few positive tuned results are:
- within noise;
- concentrated in a handful of markets;
- directional under T6;
- not robust to the fill model or latency.

So the tuner drives every baseline towards not quoting at all
(`baselines/eval_val.json`).

**Decision.**
1. **Primary comparator.** The best tuned baseline, with no-quote included as
   a candidate. Concretely, the RL policy has to beat about $0 per market.
   The primary endpoint in EVALUATION.md is unchanged.
2. **Secondary comparators.** Baselines that are required to keep trading:
   - at least 5 fills per market on average;
   - two-sided uptime of at least 0.25;
   - tuned with the same 64-trial budget on the frozen env.

   These answer a different question: does RL quote better than the classical
   rules when both are forced to make markets? They are reported alongside the
   primary result, not instead of it.
3. **When the RL result counts as market making.** RL counts as "making
   markets" only if it passes every one of these:
   - at least 5 fills per market;
   - the T6 directional share is at most 50%;
   - the T5 artifact share is at most 25%.

   If it fails any of them, we report what the policy actually does (for
   example "learns to mostly not quote"), whatever its P&L.
4. **Re-tuning on the frozen env.** (Frozen 2026-10-15: `mm.h` sha256
   `746dd7dea250eee035a8cb08a7be479eacf2d44ac1b7251503bc0d0e468566a9`; build
   hashes are in ENV.md.) The baseline tuning above ran on an env
   version from before the freeze. Every baseline will be re-tuned on the
   frozen env and data hashes before the final comparison, and all earlier
   trials stay in the multiple-testing count.

## D2. T5 artifact flags: a sweep through the touch is normal, not an artifact (2026-10-15)

**What we saw.** The env sets a flag on any fill whose trade printed outside
the snapshot's `[bid, ask]`. The env owner checked this on 40 train markets
(central fills, latency 1). The flag compares each trade against the book at
the match row, which is the pre-trade book, as intended. The flag covers 83%
of fill volume for quotes that join the touch, and 93% for quotes 4 ticks
behind it.

The reason is that most of our fill volume comes from takers sweeping through
the touch. A print exactly at our price is mostly absorbed by the displayed
queue ahead of us, which leaves only 5-10% of our volume. These fills are how
a resting order normally gets filled, so they are not an artifact of the
replay. If EVALUATION.md T5 is read literally ("more than 25% of P&L from one
flag makes the result invalid"), almost every policy would be invalid.

**Decision.** T5's 25% P&L rule applies to the flags that point to replay
artifacts:
- `inspread`: a print on the wrong side of the touch, which the snapshot
  cannot explain;
- crossed or stale book;
- one-sided book;
- the last 30 s before close.

The through-the-touch flag is still logged and reported, but it is not an
exploit detector.

## D3. Training run settings (recorded 2026-10-15, before the main sweep)

- **Sweep.** A fresh sweep named `main` with 64 trials, the same budget as
  each baseline, at 25M steps per trial. Sweep `s1` was stopped by the user,
  so it is aborted. Its two finished trials still count towards N.
- **Selection.** The config with the best mean val score over 3 seeds, among
  the top 4 sweep configs (`--reseed-top 4`).
  - Only configs that pass D1.3 can be selected (`--require-mm`): at least 5
    fills per market, and a T6 directional share of at most 0.5.
  - There is no CVaR floor. The env enforces the inventory cap Q = 100 shares.
- **Final seeds.** Seeds 100–104 at 100M steps each.
- **Conditions.** Central fills, 1 s latency, maker rebate 20%, liquidity
  rewards off. Every other parameter is an env default from ENV.md.
- **If PPO underperforms** (task "diagnose and improve"), every follow-up run
  goes into the same val ledger. The improvements to try, in order:
  1. fair-value mark;
  2. warm start from imitating the fair-value baseline, with a decaying KL
     penalty;
  3. an "off" prior at initialisation;
  4. longer runs.

  The test split stays untouched until a single final configuration is
  frozen.
