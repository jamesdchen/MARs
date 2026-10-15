# v3 track S: pre-game sports data pipeline (DECISIONS.md D11; worker w1)

| piece | file | status |
|---|---|---|
| Live recorder | `recorder_pregame.toml` (polyrec) | Running since 2026-10-15 20:22 UTC in w1's container. ~290 pre-game moneylines (top leagues by census volume) with kickoff in the next 48 h, rotating new games in every 5 min. Book rows run until kickoff + 5 min; trades and markets are finalized after kickoff. Output goes to `rec_out/` (git-ignored). |
| Compact daily files | `export_daily.py` → `rec_daily/<day>/` (committed hourly) | Outcome-0 book only (outcome 1 is its mirror), top 10 levels, zstd 19. Also trades, markets and coverage. About 40 MB/day. |
| Env binaries | `build_sports.py`, `README_build.md`, `env_sports/` | Raw data (recorder / backfill) becomes ticks/feat/fills/book/markets in the `data/build` schema. Episodes end at kickoff, and inventory settles at the last pre-kickoff mid. That settlement needs the patched env copy in `env_sports/` (fractional settlement, 3 lines). |
| Historical backfill | `backfill_pob.py`, `README_backfill.md`, `pob_openapi.json` | Ready for `POLYORDERBOOKS_API_KEY`. Run `--probe` first, then e.g. `--start 2026-09-16 --end 2026-10-14 --mirror --min-volume 50000`. It is resumable and writes `raw_pob/` in the recorder layout. Trades come from data-api, keyless (`--stage meta` works now). |

Pipeline: `rec_out/` and/or `raw_pob/` → `python build_sports.py --rec rec_out --rec raw_pob --out build --splits splits.json --verify`.
D11 fixes the splits by game date once the backfill range is known, and before any evaluation.

**The recorder dies with w1's container.** For days of unattended recording, deploy polyrec with
this config on a VPS (`recorder/README.md` "Deploy").
