# Pivot status (DECISIONS.md D9, D11): crypto v2 → pre-game sports + election reward farming

Status page, updated 2026-10-15 ~21:30 UTC from the worker reports. All numbers are copied from the
cited reports. "In progress" means the report itself says so.

## 1. Why we pivoted

**Census** (`analysis/census/REPORT.md`, w7): every on-chain fill on Polymarket, Oct 8–14 2026, 25.72M fill
legs. Edge is per $ of maker notional, with an event-bootstrap 95% CI.

| category | before rewards | with rewards | notes |
|---|---|---|---|
| **pre-game sports** | **+64 bps [1, 130]** at +30 min (**not replicated** on Sep 24–Oct 7, §3) | – | the only large category with CI > 0 before rewards; positive 5 of 7 days; top-20 makers +236 bps; $14.9M/day maker volume; edge to resolution +109 [−627, 808], a coin flip, so quotes must come off at kickoff |
| **elections** | −8 bps [−51, 18] at +2 h | **+133 bps [94, 165]** at +2 h; +122 [74, 156] at +1 d | lowest adverse selection (0.5–0.6¢/share); largest reward pool (~$30k/day); the edge is the reward program's money |
| **crypto up/down** (our v1/v2 market) | +30 bps [−6, 69] at +2 h | +31 [−6, 69] | top-20 continuous quoters (our agent's profile) **−23 bps**: adverse selection 1.35¢ vs spread 1.02¢ |
| in-play sports | −40 bps [−134, 51] at +2 h | – | non-top makers −123 bps |
| weather / economics / tech / culture / finance-prices | negative, CIs below 0 before rewards | – | avoid |

**Politics deep-dive** (`analysis/politics/REPORT.md`, w6): 60 politics/economics/geopolitics markets, Sep 1–Oct 14,
597k maker legs, same code run on crypto fills (CIs clustered by market × day).
- Politics maker fill, net at 5 min incl. rebate: **+0.23¢/share = +54 bps [0.18, 0.28¢]**; same at 2 h (+55 bps).
- Crypto up/down on the same yardstick: **+2 bps [−0.18, 0.20¢]** (Aug 15–20), −63 bps [−1.10, 0.46¢] (train days).
- Revenue split for all makers in the 60 markets per day: spread net of 5-min adverse selection $10.7k, rebates $6.4k,
  current reward pools $11.3k (38 : 23 : 40).
- Politics touch is one tick wide 100% of the time and changes price 0–1.3 times/hour (crypto 160–550), so a replay
  must model the queue. Main risk: jumps (largest 10-min move 48.5¢).

**Sports deep-dive** (`analysis/sports/REPORT.md`, w8): 235 sports markets, Oct 2–14, ¢/maker share, SE clustered by
market. Section 4 (live books) is still a placeholder in the report.
- **Pre-game, all makers:** +0.25¢ at 1 min (SE 0.03), +0.03 at 5 min (0.08), −0.27 at 30 min (0.45), rebate 0.13.
- **Crypto up/down on the same yardstick** (Oct 14, 08–14 UTC): +0.69¢ at 30 min (0.26). By this measure the average
  sports maker does *worse* than crypto; w8 notes this does not contradict our replay, because incumbents with speed
  and queue priority earn what a new entrant cannot.
- **In-play, kept separate from the census pre-game figure (they measure different things):** two-sided wallets earn
  **+1.48¢/share to resolution (SE 0.59)** plus a 0.10¢ rebate on 96M shares; one-sided resting wallets lose −3.36¢
  (SE 1.24). **This conflicts with the census's and D11's "avoid in-play", and it is unconfirmed:** a single estimate,
  to be rechecked after tonight's games. The track decision (pre-game only) is unchanged.
- Pre-game, w8 finds the binding constraint is queue position (NFL: ~2.5M shares within 1¢ vs ~120k maker shares
  filled per hour), and the last 30 min before the start is −0.9¢.

## 2. Crypto v2 epilogue

D11 (~20:25 UTC) stopped new crypto runs. The selection (`final/v2/SELECTION.md`, reading C of D13) found
**no D1.3-passing method with positive val v2 P&L**: 148 methods eligible, 0 with a positive seed mean. Every
profitable method fails D1.3 through T6 (inventory/resolution P&L, mostly on the 32 4h markets). Comparator
`fair_value` scores +101.1 on val v2 (374 markets).

| line | best result on val v2 (central fills, 1 s) | D1.3 |
|---|---|---|
| w1 risk-aware objectives (15 configs) | t04: −$0.33 ± 0.33/market; v1 replica t00 −$0.49 ± 0.28; only positive config t12 (+$0.11, 0.9 fills/mkt) | t04 pass, t12 fail |
| w2 per-length specialists | 4h_c01: +$169 ± 47 on the 32 4h markets (2 seeds), T6 1.2 (directional); 5m/15m nothing beats $0 | fail; portfolio with D1.3 = all off, $0 |
| w3 recurrent PufferNet | −$1.36 to −$2.13/market; approx KL ~2e-4/update, price heads near uniform | pass by losing |
| w4 toxicity quoter + BC/PPO | quoter −$0.059 ± 0.018/market (13.0 fills); best RL c2 +$0.030 ± 0.007 but 0/3 seeds pass D1.3. 60 s fill P&L > 0 in 1 of 475 evaluations | quoter passes 3/3 under w4's reading, fails C (positive-component share 0.58) |
| local sweep v2s1 (32 configs × 3 seeds) | **winner `v2s1/c00`: −$35.0 ± 98.1** (3 seeds), 17.5 fills/mkt; with seeds 3–4 the 5-seed mean is −111.0 ± 83.3 | pass (C) |
| D12 Binance sub-second features | nothing passes the pre-set rule; takers react 2–3.5 s before the print, after our 1 s-latency decision | not adopted |

**In progress:** the v2 freeze and the one-time Jun 15 test (`final/v2/`; the full dry run was at 99/168 runs when
this page was written). A FAIL on the primary endpoint is the expected outcome.

## 3. Track S: pre-game sports (w1 data, w2 rules, w8 context)

> **Warning: the Track S premise is unconfirmed and may be a one-week artifact.** w2
> (`analysis/v3_sports_rules/REPORT.md`) tested the two weeks before the census (Sep 24–Oct 7, on-chain, 9.9k games):
> - at-touch maker flow, valued like the census: **−26 bps [−43, −11] at 30 min, −33 bps [−55, −14] to kickoff**,
>   positive at 30 min on only 2 of 14 days, against the census's **+64 bps [1, 130]** on Oct 8–14;
> - **0 of 7,680 pre-game rules** (2,880 in pass 1, 4,800 in pass 2) are positive before rewards on the selection
>   period; the best pass-2 rule loses −1,740 bps [−3,291, −480] on the check period;
> - only the last 5 min before kickoff (+35 bps [12, 58]) and NHL at 30 min (+32 [16, 48]) are positive, and both
>   turn negative with realistic quote size and flatten cost (joining in the last 5–10 min: −57 to −166 bps).
>
> w8 also finds the average pre-game maker at −0.27¢ at 30 min (SE 0.45). w2 recommends going ahead with sports RL only
> if the L2 replay shows queue-front fills beat the at-touch pool. Track E rests on rewards and is not affected.
> The track decision stays with the user.

**Recording** (`analysis/v3_sports/`): polyrec running since 2026-10-15 20:22 UTC in w1's container. About 290 pre-game
moneylines (top leagues) with kickoff in the next 48 h, rotated every 5 min; book rows until kickoff + 5 min.
Compact daily files committed hourly (~40 MB/day); Oct 15 so far: 303 markets, 19,964 book rows (partial day).
`kickoff_watch.py` logs the sports websocket and gamma status (312 games in `kickoff/kickoffs.csv`).

**Data rule (recorded ~21:15 UTC, before any v3 result):**
- The episode is cut at the **scheduled `gameStartTime`**. The `clearBookOnStart` wipe fires at the scheduled time,
  not the actual start (val-flyr-mibrf wiped at 21:00:00.1, went live 21:08:15, 496 s late).
- The observed start only labels games and excludes early starts; it never moves the cut later.
- Esports (`endDate` = start + 6 h) use a separate recorder source shifted by −6 h since 21:14 UTC.

**Backfill:** `backfill_pob.py` is ready and **waits for `POLYORDERBOOKS_API_KEY`** (planned range e.g.
2026-09-16 → 10-14; trades come keyless from data-api). `build_sports.py` turns recorder/backfill data into env binaries.

**Baseline** (`baseline_pregame.py`, 9 fixed rules, nothing tuned): only a mechanics test so far (20 open markets,
~33 min of book each; not results). Fills only in the 2 NBA games; optimistic vs central fills differ 20–60× (the queue).
Real-game results: to be filled after ~00:30 UTC Oct 16.

**Other pre-game findings:**
- Fills away from the touch are toxic: books are one tick wide and ~100k shares deep, so a quote 1–3¢ away fills only
  on sweeps, which lose 300–2,000 bps within 60 s. Pulling quotes when the micro-price is stale (> 30 s) is the most
  useful filter in every top rule. Per-league selection overfits (e.g. "other" +$167 → −$3,436 on check).
- **Fair-value anchor** (w2, `fair_value_anchor.md`): Pinnacle, Smarkets, DraftKings, FanDuel and Kalshi agree with the
  Polymarket mid within ~0.5¢. Use an external price as a guard rail and feature, not as the quote centre. Lead/lag is
  still unmeasured.
- Env spec for sports (`ENV_SPEC.md`): queue tracking is required; takers wait ~1 s (`secondsDelay`), makers cancel freely.

## 4. Track E: election reward farming (w3; w6 context)

**Recording** (`analysis/v3_elections/`): the top 300 US-midterm markets by reward rate (≥ $20/day) =
**$26.4k/day of the $36.6k/day midterm pool** (2,892 rewarded midterm markets). Continuous since 20:24:09 UTC Oct 15
(24 markets with future event dates only since 20:46). Hourly reward snapshots of all 2,892 markets; hourly autopush of
the compacted parquet. Restarts at 21:00–21:07 for memory (RSS cap now 6 GB, ~2 restarts/day of ~5 s expected).
**Oct 16 is the first full recorded day.**

**Reward economics:**
- Census: +133 bps [94, 165] with rewards at +2 h; the reward proxy is conservative (actual top-20 payouts 2.0× the proxy).
- w6 live books: a min-size quote at the touch earns 0.02–1.1% of the pool; per $1k of capital a median of ≈ $0.05/h
  (0.12%/day), $1.8–2.3/h in the best (Fed October) markets.
- w3 baseline (`baseline_farm.py`, **interim WIP, 0.63 h window 20:21–20:59 UTC, 300 markets**): quoting at ½ the
  max spread at 1× min size earns $474.9/day of rewards on $18.6k capital, net $432.7/day with 1 fill; at ¼ the max
  spread, $549.3/day of rewards, net $495.6/day. This per-capital yield is far above w6's median estimate; it needs a
  full day of data before it is used.
- Risks: program changes (rates are snapshots), election-night jumps (out of scope per D11), UMA delays.

**Builder** (`build_elections.py`, `BUILD.md`): ready and smoke-tested. One episode = one market × one UTC-day window;
the reward hook reads per-market, per-hour program columns (checked 500/500 against `reward_terms.py`); settlement is
the last mark (markets unresolved). Size: ~50 MB per market-day, ~14 GB/day for all 276 markets (27 GB free).
`analysis/v3_elections_rules/` does not exist yet.

## 5. v3 env and trainer

**Ready** (`v3env/ENV3.md`, `mm3.h`): v2 plus episode length from the data, kickoff/last-mark settlement with
`flat_c`, `pull_s` quote pull, per-market/per-row reward hook, reward-relative order sizes, 28 extra observation
slots, per-category fees. With options off it is **v2 bit for bit** (9.08M steps, 16,218 episodes, P&L difference
0.0 USD); 169 checks pass; C equals the numpy reference on synthetic sports, elections and the real sports
selftest build. Throughput 3.89M steps/s options off, 2.06M with obs 80 + rewards. PufferLib 3.0 binding and a 5.0
wrapper exist; `adapt3.py` adapts both builders' outputs.

**Open:**
- No v3 trainer adapter: `train/envs.py` RealAdapter can drive env3 with an override, but it hashes v2 files, asserts
  the crypto length one-hot and groups by 5m/15m/4h. A v3 adapter should hash `v3env/` and group by category.
- Sports reward pools are in no build (proposal S2); the sports taker delay is not modelled (conservative).
- Elections end at a window boundary, not a resolution; one order per side (no laddering); no per-market Q against
  reward min sizes of 20–1000 shares.

## 6. Next steps and blockers

1. **PolyOrderBooks API key (user).** It unblocks the track S backfill; without it, sports data is only what we record.
2. **Recording continuity.** Both recorders run in ephemeral containers (w1 sports, w3 elections) and die with them.
   Multi-week recording (w6: ≥ 4 weeks, Oct 16 → Nov 13; elections data before Nov 3 cannot be re-collected) needs a
   VPS (`recorder/README.md`, "Deploy").
3. **Pre-register the splits once the data exists** (D11): sports by game date (oldest ~60% train, ~20% val, newest
   ~20% test, later days a forward test), fixed once the backfill range is known and before any evaluation;
   elections by recorded day, with the last days before Nov 3 as the untouched test.
4. Open checks: the real pre-game baseline run (after ~00:30 UTC Oct 16); w8's in-play two-sided result after tonight's
   games; Smarkets–Polymarket lead/lag; a full-day rerun of the election farming baseline; the pre-game edge
   on the next unseen week (w2), before more Track S building.
