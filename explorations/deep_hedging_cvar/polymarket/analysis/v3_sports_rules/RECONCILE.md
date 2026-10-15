# Reconciling w2's pre-game number with the census (w7)

Worker w2, 2026-10-15 ~22:30 UTC. Coordinator request:
1. Run w2's at-touch measure on the census window.
2. Diff the definitions and run each window under both.
3. Is the gap a regime effect or a definitional one?
4. Fold in w7's out-of-sample replication (census REPORT.md §6) and D14.

This is analysis, not rule selection. Oct 8–14 is read here only to compare definitions.

**Files:**
- `reconcile_legs.py` → `cache/legs_*.parquet`: census-definition pre-game maker legs, plus the extra columns
  needed to switch definitions.
- `reconcile.py` → `out/reconcile.csv` and `out/reconcile.txt`: the definition ladder below.
- `out/reconcile_ticks_sep.txt`, `out/reconcile_sizeweight_sep.txt`.
- `select_windows.py` → `out/select_windows.csv` and `.txt`: §4.

Data: on-chain fills for Sep 24 – Oct 14 (all markets), plus gamma. CIs are event-bootstrap 95%.

## TL;DR

1. **The gap is definitional, not a regime effect.**
   - My pipeline reproduces the census on its own week: **+66 bps [2, 135]** on Oct 8–14 (census +64 [1, 130]).
   - My at-touch measure on the same week is **−52 [−95, −1]**.
   - The difference is one step:
     - my measure re-prices every maker leg at the micro-price touch;
     - it counts the volume of sweeps that went through the touch as if it had filled at the touch.
   - In reality that volume filled deeper makers, at better prices. Those legs are the most profitable in the
     data (+69 bps at +1 tick, +116 at +2, at 30 min). My measure turns them into the most toxic.
   - The same step explains the gap in every week: −26 vs +44 on Sep 24 – Oct 7, −52 vs +62 on Oct 8–14.
   - **So my REPORT.md numbers are biased against quoting.** That covers the −26/−33 bps pool and every rule
     result in `rules.py`, whose fill model has the same flaw. They are superseded by §3–4 here.
2. **The census's week-to-week instability is in-play noise in its 30 min mark.**
   - **The noise:** the census marks pre-game legs at t + 30 min even after kickoff. Legs in the last
     30 min before kickoff then carry in-game price moves.
   - **Census definition (C0):** +52 [1, 95] on Sep 24–30, −26 [−168, 83] on Oct 1–7, +66 [2, 135] on Oct 8–14.
     Pooled Sep 24 – Oct 7 is +14 [−51, 75], which matches w7's out-of-sample +13 [−35, 55].
   - **Same legs, mark capped at kickoff (C1):** +50 [28, 73], +49 [32, 67], +71 [34, 116]. That is stable
     and positive in all three weeks.
   - **To kickoff (C3), with the micro-price in [0.05, 0.95]:** +22 [−12, 47], +41 [18, 66], +17 [−23, 62].
     Positive point estimates in every week, but CIs above 0 in only one.
3. **The pre-game maker pool is modestly positive to kickoff, before rewards and flatten costs.**
   - **The whole pool (C3):** +31 bps [12, 50] on Sep 24 – Oct 7.
   - **Legs at exactly the touch** (76% of notional): +20 [1, 37] to kickoff. By week: +6, +35, +16.
4. **The time-to-kickoff structure: the "last hour ≈ 0" in w7/D14 is the in-play mark artifact.**
   - With the mark capped at kickoff, the last 30 min is the strongest bucket in every week:
     - 0–5 min: +88 at 30 min (C1);
     - 0–5 min at the touch, to kickoff: +79, +70, +65 bps;
     - 5–30 min: +57.
   - Uncapped (census), the same buckets are +71 [−221, 358] and −95 [−386, 106]: pure noise.
   - Caveat: in the last minutes, a capped mark leaves little time for adverse selection to show, and
     flattening at kickoff costs about half a spread (see §5). The last-30-min edge is real only if inventory
     can be netted or carried flat. **D14's T−1h cut is not supported by the capped data. It drops the bucket
     that is most positive on a pre-kickoff mark.**
5. **No window or league structure survives selection on the corrected estimator** (§4).
   - The rule chosen on Sep 24 – Oct 2 is to quote 1 tick behind the touch, T−24 h → kickoff. It gives
     +97 [34, 155] on selection and +83 [24, 166] on check, but −25 [−79, 29] on Oct 8–14.
   - The candidate rank correlation is 0.39 between selection and check, and −0.36 between selection and
     Oct 8–14.
   - The D14 window (T−24 h → T−1 h) at the touch: −7 [−52, 28] on selection, +22 [−23, 76] on check,
     −57 [−106, −3] on Oct 8–14.
   - The most stable candidate is the plainest one: the touch over the whole pre-game window gives
     +13 [−14, 33], +31 [1, 60], +16 [−35, 79].

## 1. Definitions diffed

| item | census (w7, `build_legs.py` / `census.py`) | w2 at-touch pool (`build_events.py` / `touch_edge.py`) |
|---|---|---|
| unit | each maker leg | each taker event (tx × condition × direction) |
| price | the leg's own fill price q | the touch level: ceil/floor of the micro-price on the 1¢ grid |
| volume | the leg's size | all volume at or beyond the touch, at the touch price |
| frame | the leg's own token | outcome 0 |
| micro-price | latest taker buy/sell mean, last block before | same (same code) |
| pre-game | t < gameStartTime (else eventStartTime) | same |
| 30 min mark | mid(t + 30 min), in-play if past kickoff; settlement if closed | capped at the last pre-kickoff mid |
| price filter | none | micro-price in [0.05, 0.95] |
| weighting | q × size | cost of the token bought |
| CI | Poisson bootstrap over gamma events | same |

## 2. The ladder

One switch per rung, in bps of notional (point estimate, then [95% CI]).

| rung | Sep 24–30 | Oct 1–7 | Sep 24–Oct 7 | Oct 8–14 (census week) |
|---|---|---|---|---|
| C0 census definition | +52 [1, 95] | −26 [−168, 83] | +14 [−51, 75] | **+66 [2, 135]** |
| C1 + 30 min mark capped at kickoff | +50 [28, 73] | +49 [32, 67] | +50 [35, 64] | +71 [34, 116] |
| C2 + price filter [0.05, 0.95] | +35 [13, 56] | +45 [29, 63] | +40 [26, 53] | +58 [26, 99] |
| C3 C2, to kickoff | +22 [−12, 47] | +41 [18, 66] | +31 [12, 50] | +17 [−23, 62] |
| C4 C2, w2 weighting | +35 | +45 | +40 | +58 |
| C5 C2, every leg re-priced at the touch | −10 [−32, 12] | −1 [−22, 17] | −5 [−20, 10] | +17 [−15, 61] |
| C6 C5, only legs at or beyond the touch (= w2 pool, 30 min) | −30 [−55, −9] | −22 [−45, −2] | **−26 [−42, −11]** | −4 [−40, 44] |
| C7 C6, to kickoff (= w2 pool, to kickoff) | −43 [−78, −15] | −22 [−49, 3] | **−33 [−55, −15]** | **−52 [−95, −1]** |
| C8 C2, legs at or beyond the touch, own price | +39 [17, 60] | +49 [32, 69] | +44 [29, 59] | +62 [30, 111] |

- C6 and C7 reproduce the w2 REPORT.md numbers exactly: −26 and −33 bps on Sep 24 – Oct 7.
- C8 versus C6 is the same legs, own price versus touch price: **+70 bps of difference, in every week**.
- C4 shows the weighting is irrelevant.

## 3. Where the gap comes from: legs by distance from the touch

Own price; Sep 24 – Oct 7; micro-price in [0.05, 0.95].

| level | share of notional | spread bps | 30 min (capped) | to kickoff |
|---|---|---|---|---|
| inside the touch | 6% | −199 | −39 [−74, −5] | −35 [−78, 3] |
| **touch** | **76%** | 94 | **+29 [17, 40]** | **+20 [1, 37]** |
| +1 tick | 14% | 276 | +69 [37, 107] | +81 [34, 124] |
| +2 ticks | 3% | 501 | +116 [45, 202] | +54 [−49, 165] |
| +3 or more | 2% | 838+ | +478 [371, 591] | +369 [184, 541] |

**Touch, to kickoff, by week:** +6 [−23, 28], +35 [11, 61], +16 [−28, 79].

**What a small quoter's fill model should be.**
- A quoter with random queue position at level L is filled in proportion to the volume that actually traded at
  L. On a sweep it is filled in full, as is every maker at L. On a partial event it is filled pro rata.
- Its per-$ edge is therefore the own-price edge of the legs at L.
- `rules.py` instead gave min(S, volume at or beyond L) per event. That overweights small events, which are
  toxic. In the bucket tables, at-touch fills from large takers were favourable and small ones adverse
  (`out/reconcile_sizeweight_sep.txt`).
- With the volume-proportional fill model, quoting one tick behind the touch is the best level. It is not the
  disaster `rules.py` reported.
- The +3 or more bucket is not a strategy. It is mostly stale micro-prices, where the "spread" is the price
  move since the last print.

## 4. Window and league selection with the corrected estimator

`select_windows.py`. Estimator: own-price legs at the quote level, to kickoff.

- **Candidates:** 3 levels × 4 starts (∞, 24, 6, 2 h) × 4 pulls (0, 5, 15, 60 min) × 8 leagues.
- **Selection:** on Sep 24 – Oct 2 (at least $2M notional).
- **Evaluation:** on Oct 3–7 (check) and on Oct 8–14 (census week).

| candidate | Sep 24 – Oct 2 (sel) | Oct 3–7 (check) | Oct 8–14 |
|---|---|---|---|
| **selected, pooled leagues:** +1 tick, T−24 h → kickoff | +97 [34, 155] | +83 [24, 166] | −25 [−79, 29] |
| best overall: +1 tick, NFL, T−24 h → T−1 h | +198 [17, 351] | −13 [−102, 107] | −35 [−165, 100] |
| **D14:** touch, T−24 h → T−1 h, all | −7 [−52, 28] | +22 [−23, 76] | −57 [−106, −3] |
| D14: touch or +1, T−24 h → T−1 h, all | +8 [−38, 41] | +32 [−20, 90] | −62 [−123, −15] |
| touch, whole window, all | +13 [−14, 33] | +31 [1, 60] | +16 [−35, 79] |
| touch or +1, whole window, all | +25 [−1, 46] | +37 [10, 69] | +10 [−27, 53] |

- Rank correlation of candidates: 0.39 between selection and check, −0.36 between selection and Oct 8–14.
- League picks don't carry over. MLB is the exception: +76 [51, 112] at the touch on Oct 8–14, but only +9
  on selection.

**On the coordinator's question about windows beyond 6 h.**
- Yes. The `rules.py` grid included A = ∞ and A = 24 h with T up to 60 min, so T−24 h → T−1 h was tested.
- But that grid's fill model is flawed (§3), so those results don't count.
- Re-running with the corrected estimator, restricted to T−24 h → T−1 h, gives the D14 rows above. They are
  not positive out of sample, and negative on the census week.

## 5. What this means for Track S and D14

- **Pre-game maker economics are positive but thin.**
  - About +20–30 bps of notional to kickoff before rewards: about 0.1–0.15¢ per share at p ≈ 0.5.
  - CIs reach 0 in some weeks.
  - That is before the cost of flattening net inventory at kickoff. The pool's inventory is not ours; for a
    quoter, flattening costs about half a spread, 0.5¢/share ≈ 100 bps, on whatever does not net out.
  - **So the edge survives only for a quoter whose fills net out within the pre-game window, or one that is
    allowed to stop quoting one-sided.** The replay env must model this (ENV_SPEC.md §1, `flat_c`).
- **D14 (T−1 h) should be revisited.**
  - Its evidence for "last hour ≈ 0" uses marks that cross kickoff.
  - With marks capped at kickoff, the last 30 min is the best bucket in all three weeks.
  - The real constraint on the last hour is the flatten cost and the kickoff wipe, not adverse selection.
    Suggested D14 amendment: keep quoting to about T−5 min, with an inventory penalty that ramps over the last
    30 min (ENV_SPEC.md §5). Let the replay decide the pull time on val.
- **For the replay:** queue-aware fills at the touch and one tick behind are the decision-relevant levels.
  Their per-$ edge is the own-price pool edge in §3 (+20 to +80 bps to kickoff, depending on level). Capacity
  depends on queue position, which only the L2 replay can measure.
- **My REPORT.md (rules) is superseded.** "No simple rule makes money" came from a fill model that credited
  sweep volume at the touch price. Its other parts stand: the anchor survey, ENV_SPEC.md, and the per-day
  instability of 30 min numbers.

## 6. Follow-up (D15): how much pre-game inventory nets out before kickoff, and the flatten drag

`netting.py` → `out/netting.csv` and `.txt`.

**Unit:** one maker wallet × one condition, in the outcome-0 frame, so trades on both tokens net as a merge
would.
- netted share = 1 − Σ|net at kickoff| / Σ gross shares;
- drag = 0.5¢ × Σ|net| / notional (D15: half a spread);
- edge = own price, to kickoff (C3).

Only conditions whose kickoff falls in the window are included. The CI is on the edge after drag (event
bootstrap).

| group | Sep 24 – Oct 7: netted | drag bps | edge bps | after drag [CI] | Oct 8–14: netted | drag | edge | after drag [CI] |
|---|---|---|---|---|---|---|---|---|
| **all** | **19%** | **83** | +31 | **−52 [−74, −34]** | **14%** | 86 | +17 | **−69 [−104, −23]** |
| soccer | 24% | 80 | +53 | −27 [−61, 7] | 15% | 84 | −12 | −96 [−137, −58] |
| NFL | 21% | 79 | +12 | −67 [−84, −50] | 9% | 94 | −3 | −97 [−143, −49] |
| esports | 10% | 90 | +31 | −59 [−142, 6] | 10% | 90 | +53 | −37 [−105, 5] |
| CFB | 14% | 86 | −7 | −93 [−117, −72] | 9% | 93 | +6 | −87 [−133, −46] |
| MLB | 11% | 93 | +20 | −73 [−113, −36] | 7% | 94 | +85 | −9 [−37, 34] |
| tennis | 25% | 77 | +100 | +23 [−53, 94] | 22% | 79 | −125 | −204 [−362, −44] |
| NHL | 6% | 94 | −25 | −119 [−243, −43] | 11% | 88 | −0 | −88 [−121, −56] |

Same quantities by the size of the wallet's position in one condition (gross shares):

| gross shares | Sep 24 – Oct 7: netted | drag bps | edge bps | after drag [CI] | Oct 8–14: netted | drag | edge | after drag [CI] |
|---|---|---|---|---|---|---|---|---|
| < 50 | 14% | 86 | +50 | −37 [−51, −25] | 11% | 88 | +64 | −24 [−50, 5] |
| 50–250 | 21% | 78 | +97 | +19 [5, 33] | 18% | 80 | +30 | −50 [−61, −39] |
| 250–1k | 22% | 77 | +27 | −50 [−63, −38] | 20% | 78 | +12 | −67 [−81, −53] |
| 1k–5k | 21% | 80 | +4 | −76 [−96, −59] | 19% | 81 | −7 | −88 [−105, −68] |
| 5k–25k | 23% | 79 | +7 | −72 [−106, −42] | 17% | 82 | −4 | −86 [−114, −59] |
| > 25k | 16% | 87 | +50 | −38 [−65, −8] | 8% | 93 | +40 | −53 [−130, 57] |

By wallet tier (total pre-game notional), Sep 24 – Oct 7 then Oct 8–14:

| tier | netted | edge after drag, Sep 24 – Oct 7 | edge after drag, Oct 8–14 |
|---|---|---|---|
| top 20 | 18–19% | −35 | −58 |
| 21–200 | 10–20% | −61 | −80 |
| 201–2000 | 12–15% | −77 | −73 |
| rest | 9–10% | +59 [−25, 143] | +5 |

**Reading.**
- **Passive pre-game makers barely net their inventory before kickoff.** Only 10–25% of filled shares net out
  per wallet and condition, in every league, size and tier. Most makers carry their position into the game.
  That is consistent with the census finding that the edge to resolution is ±700 bps of noise.
- **At half a spread, the flatten drag is 80–90 bps. It is larger than the whole pre-game edge** (+17 to +31
  bps to kickoff), so the pool is negative after drag in both periods:
  - −52 bps [−74, −34] on Sep 24 – Oct 7;
  - −69 [−104, −23] on Oct 8–14.
  - No league or size bucket is positive after drag in both periods.
- **The pool's netting is a lower bound for a quoter that controls inventory.** Netting is measured per
  condition. Cross-condition hedges inside one game are not counted:
  - soccer 1X2 is three neg-risk conditions;
  - moneyline against spread.
  - Makers that skew or pull the side that adds to inventory would net more. The pool is mostly makers who
    don't try.
- **Break-even.** With a +20 to +30 bps pre-game edge and a 100 bps flatten cost per un-netted share, a quoter
  must net about 70–80% of its fills before kickoff. Unconditionally, the pool nets 15–20%.

**Implications for the replay (ENV_SPEC.md, D15).**
1. Inventory control is the core problem of the sports env, not an add-on. The close-inventory penalty
   ramping over the last 30 min, and one-sided pulls, must be in the action space and the reward.
2. Report netted share and drag as standard Log fields next to the D1.3 detectors.
3. A game-level book (ENV_SPEC.md §7) may be needed to reach the 70–80% netting. Cross-condition netting
   within a game is the next measurement. It needs a neg-risk outcome map per event, from gamma `negRisk`
   events, which this analysis did not build.
4. Alternative to flattening: carrying inventory into the game is ±700 bps of noise per week of data (census),
   and D11 rules it out.
