# v3 status: crypto closed, elections and pre-game sports open (DECISIONS.md D11, D13–D15)

Status page, updated 2026-10-16 ~01:00 UTC after every worker report was in. Every number is copied from the cited
report; nothing here is a test result for v3. Paths are relative to
`explorations/deep_hedging_cvar/polymarket/`.

## TL;DR

- **Crypto is closed. Both one-time tests failed:** v1 (Aug 24) +27.65 [−259.07, +293.92] against fair_value, and
  v2 (Jun 15) −$495 [−1,610, +551], where RL lost $926 against $0 for not quoting. The cause is structural: the 60 s
  fill P&L was positive in 1 of 493 w4 evaluations.
- **Elections (Track E) are the strongest lead, and the income is reward money.** w7's contested-market curve gives
  $36 per $1k per day at $10k, $22 at $25k and $10 at $100k. w3's replay gives 2.6%/day for the spec quoter but only
  **0.57%/day** for a quoter that never improves the touch. The defensible range is therefore 0.57–2.6%/day.
- **The gap between those two figures is untested fills.** 79% of the spec's rewards come from quotes that improve the
  touch in near-empty books, where nobody trades. The fills and adverse selection those quotes would attract are
  unobserved.
- **Pre-game sports (Track S) has a real but thin edge**, about +20–30 bps to kickoff before rewards. Pre-game makers
  net only 14–19% of their inventory before kickoff, and flattening the rest costs ~85 bps, more than the edge.
  After that drag the maker pool is −52 and −69 bps.
- **Which pre-kickoff window is best is UNRESOLVED.** w7 finds the last 30 min the best bucket, w8 finds the last hour
  −130 bps, and w1 finds quoting up to kickoff the worst choice on one evening. On w1's 16 real games, no fixed
  quoter beats not quoting.
- **In-play two-sided makers earn +1.48¢/share (SE 0.59)**, but they need a fast score feed. Polymarket's public feed
  lags the book by a median of 74 s. In-play stays out of scope (D11).
- **The infrastructure is ready:** env mm3, train3, both recorders and both builders. Recording started Oct 15
  ~20:20 UTC. **No v3 split can be fixed until about 3 full days exist**, and the sports backfill is blocked on the
  API key's environment variable name.
- **Decisions for you (§6):** fix the key variable name, approve w4 and w5, an always-on recording server until Nov 3,
  whether to prioritise elections, and pre-registering the splits.

## 1. Crypto: closed

| test | primary endpoint (RL − fair_value, central, 1 s) | verdict | source |
|---|---|---|---|
| v1, Aug 24, 127 markets | +27.65 [−259.07, +293.92] | FAIL (CI; T7) | `final/RESULTS.md` |
| v2, Jun 15, 288 BTC 5m markets | **−495.16 [−1,609.81, +550.93]** | **FAIL** (5 of 7 criteria) | `final/RESULTS_V2.md` |

- **v2:** RL `v2s1/c00` (5 seeds) lost **−$925.68, −$3.21/market**; fair_value −$430.52; no-quote $0. RL was
  significantly worse than not quoting (−925.68 [−1,626.15, −221.80]); only optimistic fills were positive (+$521).
  Expected: on val v2 the method was −$35 (3 seeds) and −$111 (5 seeds).
- **The structural reason (w4).** Write P&L = F + inv, where F is spread capture + 60 s markout + rebates. If F < 0
  and P&L > 0, inventory exceeds the whole P&L and T6 calls the method directional. Across w4's ledger **F > 0 in 1 of
  493 val evaluations** (inventory positive in 474), and **0 of 493 both pass D1.3 and make money**. The census
  agrees: the top-20 continuous crypto quoters lose −23 bps (Oct 8–14) and −12 bps (out of sample).
- **Workers w1–w5** (`train/workers/*/REPORT.md`, all done; w1–w3 stopped early by D11): w1 no config beats
  no-quote; w2's 4h profits are directional (T6 1.1–1.7); w3 −$1.36 to −$2.13/market; w4's quoter −$0.059 ± 0.018
  (D1.3 3/3), its PPO c2 +$0.030 ± 0.007 but directional; w5's best D1.3-passing config −$0.233 ± 0.022.
  No further crypto work is planned.

## 2. Elections, Track E: the strongest lead

**w7's capital curve** (`analysis/census/REPORT.md` §7): live books of the top 300 midterm markets (pool
$25,960/day), 6 polls 22:10–23:00 UTC Oct 15, one min-size bid and ask about 3¢ from the mid, the conservative
("hi") competitor bound, capital allocated greedily.

| capital | $/day per $1k, contested markets [95% CI] | %/day | all 300 markets (incl. 6 empty bands) |
|---|---|---|---|
| $1k | 125.8 [79.9, 191.6] | 12.6 | 283.8 [155.5, 437.4] |
| $10k | 36.2 [29.4, 45.9] | 3.6 | 52.2 [38.6, 67.6] |
| $25k | 22.4 [18.9, 27.2] | 2.2 | 28.8 [23.1, 35.3] |
| $100k | 10.1 [8.6, 11.7] | 1.0 | 11.7 [9.9, 13.6] |
| $250k | 6.0 [5.2, 6.8] | 0.6 | 6.6 [5.7, 7.5] |

- **w7's planning number:** $22–36 per $1k per day at $10–25k, before $0.2–2 of fill costs, which is about 2–3.5%/day.
  The yield is steep in capital because every market's pool is fixed. This reconciles w6's 0.12%/day (w6 assumed a
  very large book or stiffer competition).
- **Out-of-sample census:** +101 bps [63, 130] with rewards (D14). Without rewards it is about 0 to slightly positive.

**w3's measured range** (`analysis/v3_elections/BASELINE.md`, Update section). This is a replay on the recorded
books and real trades over 3.36 h, 20:21–23:43 UTC, for all 300 markets:

| variant | net $/h | net %/h of capital [market CI] | ×24 |
|---|---|---|---|
| spec, ½ max spread, 1× min size | 20.5 | 0.109 [0.059, 0.169] | **2.6%/day** |
| same, rewards only, no fills | 22.2 | 0.119 | 2.9%/day |
| ¼ max spread | 18.3 | 0.090 [0.035, 0.152] | 2.2%/day (65 fills) |
| **spec, join/behind only (never improves the touch)** | 4.5 | 0.024 [0.006, 0.042] | **0.57%/day** |
| ¼ max spread, join/behind only | 6.2 | 0.031 [0.003, 0.058] | 0.74%/day |

**Why the two figures differ: rewards from quotes that improve the touch in empty books.**
- About 79% of the spec's rewards (58.9 of $74.6) come from quotes inside the touch, mostly in near-empty books such
  as `closer-governor-race-*` (0.13 / 0.88). In the 1.91 h run those 20 markets had **2 taker transactions in
  total**, none at or through our quote.
- **So their fills are unobserved.** A tight quote in a 75¢-wide book would attract flow history cannot price,
  informed flow included. Where flow exists, adverse selection shows: the ¼-spread variant lost ~$28 of markout over
  65 fills.
- **The join/behind-only row is the defensible floor**, and the spec figure is the optimistic case. w7's $10–25k
  points (2.2–3.6%/day) agree with w3's spec figure. Whether w7's curve leans on the same inside-touch quotes is not
  measured.

**Risks:**
- **The Nov 3 jump.** A two-sided min-size quote can be swept for a 20–50¢ loss per share. Quotes come off before
  the first returns, and election night is out of scope (D11).
- **Competition.** A second farmer halves the inside-touch rewards, and the 6 empty-band markets would draw
  competitors within days. Use the contested curve.
- **Pool drift.** The selected pool fell from $26,412/day to $25,753/day in one hour (−2.5%). The `closer-*` and
  `*-on-october-23` programs end with their markets, and the program can change or stop.
- **Thin evidence and model limits:** evening-only (US) windows (six snapshots for w7, 3.36 h for w3); the env has
  one order per side, and our orders move neither the adjusted mid nor competitor Q (ENV3 §14).

## 3. Sports, Track S: the edge is real but thin

**The reconciliation (w2, `analysis/v3_sports_rules/RECONCILE.md`; D15).**
- **w2's earlier −26 bps was a valuation error.** It priced sweep volume at the touch, a ~70 bps bias. With census
  definitions, w2's pipeline gives +66 [2, 135] against the census's +64 [1, 130].
- **With marks capped at kickoff** (w7 §6d), the pre-game edge is stable: **+53, +50, +49 and +71 bps** by week
  (Sep 17 → Oct 14), +49 [32, 64] out of sample and +57 [34, 79] pooled.
- **To kickoff the edge is thin:** +31 [12, 50] for the whole pool (Sep 24–Oct 7), **+20 [1, 37] at the touch** and
  +81 [34, 124] one tick behind.
- **Selections do not transfer.** Rank correlation is 0.39 from selection to check and −0.36 to Oct 8–14.

**Netting and flatten drag (w2 §6).**
- Only **19% (Sep 24–Oct 7) and 14% (Oct 8–14)** of pre-game maker inventory nets out before kickoff. At half a
  spread the drag is **83–86 bps**, more than the +17 to +31 bps edge, so the pool is negative after drag:
  **−52 [−74, −34]** and **−69 [−104, −23]**. No league or size bucket is positive after drag in both periods.
- **Break-even needs about 70–80% netting.** Inventory control is the core problem of this env: pulls, skew, the close
  ramp, and possibly game-level netting across moneyline, spread and 1X2.

**UNRESOLVED: where before kickoff to quote.** The three sources disagree:
- **w7 (census §6d, capped, pooled):** the last 30 min is **+69 [2, 128]**, +62 to +69 in every week, and the best
  bucket. 30–60 min is the weakest at +32 [−6, 91].
- **w8 (`analysis/sports/REPORT.md` §2.2, D14 table, +30 min mark):** the last hour, 54% of pre-game maker volume,
  is **−130 [−463, +178]**; 6–24 h is 0 where w7 has +41/+43. The table does not say whether its mark is capped at
  kickoff.
- **w1 (one evening, 16 real games, mostly NHL):** quoting to kickoff (pull 0 or 300 s) is **worst** and pulling
  1 h early is best. w1 itself calls this one noisy data point, not evidence against D15.
- **How it gets settled.** D15 already makes `pull_s` ∈ {0, 300, 900, 1800, 3600} a val-selected hyperparameter, so the
  replay on recorded data decides. Nobody should pick a window from these figures.

**In-play (w8, final at commit 4261d56).**
- **Two-sided wallets** earn **+1.48¢/share to resolution (SE 0.59)** plus a 0.10¢ rebate, on 96M shares; one-sided
  wallets lose −3.36¢ (SE 1.24). Tonight's recheck on true mids: +0.83¢ against −1.04¢ at 5 min; the resolution
  figure is unconfirmed (6–8 markets resolved, SE 2–4¢).
- **It needs a fast score feed.** Polymarket's public score websocket lags the book by a median of **74 s** (21 of 21
  events), Smarkets by 42 s. A commercial feed or a book-reactive pull rule is required; in-play stays out of scope
  (D11).

**w1's real-game baseline** (`analysis/v3_sports/README.md`, 00:41 and 00:50 UTC).
- 16 kicked-off games (9 NHL, 3 CFB, 1 NBA, 3 markets of one Argentine soccer game), 2.6–3.6 h of book each.
- **No fixed quoter beats no-quote.** Every variant with ≥ 5 fills per market loses $5.5–7 in total after the
  half-spread flatten; spread capture 0.50¢/share against a −0.4 to −0.5¢ markout at 30 s.
- **The evaluator3 comparators (sym_touch, mid_value, as_mid) agree:** none makes markets (< 5 fills per market) and
  none is positive at any pull before 3600 s. The pull-3600 positives (+$0.93 to +1.75) come from ~1 fill per market
  and vanish under pessimistic fills.

**Kickoff (w1).**
- **Starts.** No game started early. Every confidently timed start came after the schedule, by +5 s to +496 s.
- **The `clearBookOnStart` wipe is not universal.** It came at the scheduled time for nba-tor-nyk (+0.4 s),
  nba-okc-hou (−8.8 s) and cfb-gas-old (−7.7 s); **no wipe came within +10 min** for 7 NHL games, nfl-sea-den and 2 CFB
  games; cs2-fly-offcho was wiped 602 s *before* its scheduled time and an MLB NRFI market 1,117 s after.
- w8's books show 60–87% depth drops within 1–2 s of the start on NHL, NBA, CFB and NFL; the two detectors are not
  reconciled. **So the episode is cut at the scheduled `gameStartTime`**; the observed start only labels games and
  excludes early starts.

**Queues (w8 §4, §6).**
- Pre-game books are one tick wide 100% of the time. NFL holds **3.0M shares within 1¢ on the moneyline** (1.8M on
  the spread) against ~120k maker shares filled per hour, so a new quote at the touch would wait hours. NBA turns over
  far faster: 2–6k shares within 1¢ against 23k per hour.
- w8: the binding constraint pre-game is queue position, not information.

## 4. Infrastructure: ready

- **`v3env/` (mm3.h, ENV3.md):** v2 plus data-length episodes, kickoff/last-mark settlement with `flat_c`, `pull_s`,
  a per-row reward hook, reward-relative sizes, obs 80, per-category fees and the D15 netting fields. Options off, it
  is **v2 bit for bit** (9,079,000 steps, 16,218 episodes); 200 test checks all ok; 3.89M steps/s (2.06M with obs 80
  and rewards).
- **`train3/`:** `V3Adapter` (hashes v3env, groups by category, refuses the test split and jun15); `evaluator3.py`
  (v3 comparators, `--pull-grid`, D13 pooling, netting, time buckets); `train_ppo3.py`; `split3.py` (sha256
  registry); `smoke3.py` (mechanics pass).
- **Recorders:** polyrec for sports (`analysis/v3_sports/`) and elections (`analysis/v3_elections/`, supervised,
  hourly autopush), the kickoff watcher and detector, hourly reward snapshots. **Builders:** `build_sports.py`
  (sports-2, `--seg 0`, D15 bucket segments) and `build_elections.py` feed `adapt3.py`; `backfill_pob.py` is ready.

## 5. Data status

- **Recording since Oct 15 ~20:20 UTC.** Elections: **300 markets**, continuous since 20:24:09 (24 only since
  20:46); **Oct 16 is the first full day**. Sports: pre-game books since 20:22, **~700 markets** on the 7-day horizon
  since 22:19; compact daily files committed hourly.
- **Incident, Oct 16 00:44.** w1's container restarted during an evaluator3 run (exit 137). The sports recorder was
  down from 00:44:13 to 00:45:58 UTC and kickoff_watch from 00:42:36 to 00:46 (`kickoff/snapshot/RECORDER_GAPS.txt`).
  Both recorders still live in ephemeral worker containers and die with them.
- **The PolyOrderBooks backfill is blocked on the environment variable name.** `backfill_pob.py` reads
  `POLYORDERBOOKS_API_KEY`; this session's environment has no such variable, and the only Polymarket-related name
  present is `POLYMARKET_API`. Planned range: e.g. 2026-09-16 → 10-14.
- **Free L2 archives (w8 §5):** Hugging Face `Joseph3222/polymarket-orderbook` (every market, Feb 22–Aug 10 2026,
  ~1.2 TB) and `Houroux/polymarket-l2-history` (reward-eligible markets, Jul–Oct 2026, ~310 GB, still updated, with a
  `sports_ws` score folder). Unchecked: whether Houroux covers rewarded election markets, which would give Track E
  history before Oct 15. Paid: Marketlens Pro, $39/month, tick-level L2 since Mar 2026.

## 6. Decisions for you

1. **Fix the key variable name.** Export the PolyOrderBooks key as `POLYORDERBOOKS_API_KEY`, or tell us the name you
   used. That unblocks the sports backfill: about 4 weeks of pre-game L2 instead of what we record from now on.
2. **Approve w4 and w5 in their own sessions.** Both are waiting on you there.
3. **An always-on server for multi-week recording until Nov 3** (`recorder/README.md`, "Deploy"). Election books from
   before Nov 3 cannot be collected again, and the 00:44 restart shows how fragile the worker containers are.
4. **Whether to prioritise elections.** It is the stronger lead (0.57–2.6%/day measured; $22–36 per $1k per day at
   $10–25k). Sports is thin: the edge does not survive the flatten drag without 70–80% netting, and no fixed quoter
   beats no-quote yet.
5. **Pre-register the splits once 3 or more full recorded days exist**, with `split3.py`, before any model is
   evaluated on those dates: elections by recorded day (the last days before Nov 3 are the untouched test); sports
   by game date, about 60/20/20, once the backfill range is known. Then: evaluator3 comparators over the D15 pull
   grid, then training; each track's test runs once (D11).
