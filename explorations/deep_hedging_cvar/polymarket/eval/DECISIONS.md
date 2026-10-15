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
4. **Re-tuning on the frozen env.** The baseline tuning above ran on an env
   version from before the freeze. Every baseline will be re-tuned on the
   frozen env and data hashes before the final comparison, and all earlier
   trials stay in the multiple-testing count.
