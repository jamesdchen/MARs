# Pivot assessment (DECISIONS.md D9)

Source: `analysis/census/REPORT.md`, worker w7. It covers on-chain fills of every Polymarket market
for Oct 8–14 2026: 25.7M fill legs across 13 categories. Edge is measured per $ of maker notional.

## Ranking

| rank | category | edge before rewards | with rewards | notes |
|---|---|---|---|---|
| 1 | **Pre-game sports**, pulling quotes at kickoff | **+64 bps [1, 130]** at +30 min | – | the only large category whose CI is above 0 before rewards; positive 5 of 7 days; top-20 makers +236 bps; $14.9M/day maker volume; moderate competition |
| 2 | **Elections**, reward farming (US midterms Nov 3) | −8 bps [−51, 18] | **+133 bps [94, 165]** | lowest adverse selection of any category; largest reward pool (~$30k/day); the edge depends on the reward program |
| – | crypto up/down (our v1/v2 market) | +30 bps [−6, 69] on average | – | continuous quoters (our agent's profile) −23 bps; reproduces our finding |
| avoid | in-play sports, weather, economics, tech, culture, finance-prices | adverse selection > spread | – | – |

## What a v3 would need

- **Pre-game sports:** 3–4 weeks of L2 books. PolyOrderBooks has a sports archive behind an API key,
  or our recorder can collect it from now on.
- **Elections:** L2 at 1-minute reward resolution. No vendor archive has it, so recording has to
  start now (Oct 16 → Nov 4).
- The C env is data-source agnostic. Porting it means building ticks/feat/fills/markets binaries
  for the new category, plus category-specific features: kickoff time and, for elections, the
  reward score.
- v3 must be pre-registered before any tuning, with a date-split train/val and an untouched test
  set (D9).

Still pending: politics (w6) and sports (w8) deep-dives.
