# PufferLib 5.0 port of the v3 core (`mm3.h`)

This file covers only what differs from v2. `../PUFFER5.md` holds the 12 rules and the evidence for
them (pin `6ffa5b1`; `ea7f989` has the same env API). The wrapper is `puffer5/polymarket_mm3.h`,
patterned on `../puffer5/polymarket_mm.h`, and `puffer5/test_wrapper3.c` checks it. It was
compiled and run against a PufferLib `ea7f989` checkout, with a stub `raylib.h` because the env
does not render.

## The 12 rules: how mm3 meets each one

| rule | v3 |
|---|---|
| 1 pure C core, C99 + C++17 | `mm3.h` has no PufferLib includes. Every name is prefixed (`m3_`, `M3_`, `Mm3`), so it can sit next to `mm.h`. `g++ -std=c++17 -fsyntax-only` is clean, and so is the wrapper test as C++17. Casts on malloc / mmap / calloc. |
| 2 ints and floats API | `m3_step(Mm3*, const int* act, float* obs, float* rew, float* term, Mm3Log*)`, `m3_reset(Mm3*, float*)` |
| 3 obs float32, O(1), last-ish constant 1 | Obs 49 stays the constant 1, and the extra slots are in [0, 5]. **`OBS_SIZE` is compile-time in 5.0**: the wrapper uses `MM3_OBS_SIZE` (default `M3_OBS_FULL` = 80, a multiple of 8) and exits if the ini's `obs_size` differs. For the v2 observation, build with `-DMM3_OBS_SIZE=50` and `obs_size = 50`. |
| 4 `ACT_SIZES` compile-time brace list | `M3_ACT_SIZES {7, 7, 3}`, unchanged. Every action is legal: clamped, and `pull_s` turns quotes off inside the core. |
| 5 Muon alignment | unchanged (discrete heads; hidden % 8) |
| 6 dense reward in [-1, 1] | As v2. The close-inventory shaping is now inside the core and applied before the clip, so the trainer-side wrapper (`CloseInventoryPenalty`) is not needed in 5.0. |
| 7 auto-reset, true terminals | As v2. Settlement at kickoff mid / last mark (`settle_mode` 1 / 2) makes every episode end a true terminal. Episode lengths now vary with the data (1-day elections, whole pre-game windows), and the random start in the first 20% keeps terminals off segment row 0. |
| 8 one flat read-only data set, mmapped once | Seven files from one directory: v2's five plus the optional `markets3.bin` and `x3.bin`. All are mapped `PROT_READ, MAP_PRIVATE` behind `M3_DATA_READY`, and `puf_close` never unmaps. For legacy data, the per-market `Mm3Ext` records are synthesised once at open and are read-only afterwards. |
| 9 no mutable globals in step, per-env RNG | Step reads only the data, the interned config and the market lists. The lists are built by `m3_config` during init, which is serial in 5.0 (`puf_init` loop). Seed is `(seed << 32) + env index`. |
| 10 init does not touch buffers | As v2. `m3_reset` writes the first observation. Repeated init is safe: same directory, same interned configs and lists. |
| 11 Log of floats with n | `Mm3Log`: 49 floats, then `n`. The wrapper does `#define Mm3Log Log` before including the core, because the trainer reads `log.n` and `sizeof(Log)` itself. |
| 12 `--float` for parity | Builds need `-ffp-contract=off` for bit parity with the 3.0 binding, as in v2. |

## What to do in a PufferLib checkout

1. Copy `puffer5/polymarket_mm3.h` and `mm3.h` to `PufferLib/ocean/polymarket_mm3/`. In the
   wrapper, change `#include "../mm3.h"` to `#include "mm3.h"`.
2. Write `config/polymarket_mm3.ini`, with `[env]` holding every name of `M3_PARAM_NAMES`. The
   wrapper reads them with `dict_get`, which exits on a missing key, so none can drift. Add
   `data_dir`, `seed` and `log_episodes`, and set `obs_size` equal to the compiled `MM3_OBS_SIZE`.
   v2's `../puffer5/make_ini.py` can be pointed at `mm3.h`: it parses `MM_PARAM_NAMES` /
   `MM_DEFAULTS`, so the regex prefixes need `M3_`. Its `[eval_env]` block (split 1,
   `fixed_start = 0`, `cycle_stride`, `log_episodes = 1`) carries over unchanged.
3. Run `./build.sh polymarket_mm3 --float`, adding `-DMM3_OBS_SIZE=50` for a v2-observation build.
4. Data: run `adapt3.py` on the build first. `data_dir` is the build directory and holds the seven
   files.

## Test (`puffer5/test_wrapper3.c`)

```
cc -std=gnu11 -O2 -fopenmp -ffp-contract=off -I$PUFFERLIB/src -I$RAYLIB_INCLUDE test_wrapper3.c -o t3 -lm
./t3 ../../data/build                                             # options off, OBS_SIZE 80
./t3 ../../data/build liq_rewards=1 size_mode=1 Q=300 close_inv_coef=0.02 pull_s=20 settle_mode=2 flat_c=0.5
g++ -std=c++17 -x c++ -fsyntax-only -fopenmp -I$PUFFERLIB/src -I$RAYLIB_INCLUDE test_wrapper3.c
```

The test builds 64 envs the way the trainer does: calloc, `rng = index`, every `puf_init`, then the
buffers bound, then `puf_reset`. It steps them 1500 times with float actions under OpenMP and runs
`m3_step` serially on the same actions with seed `(1 << 32) + i`. It checks:
- observations (all 80), rewards, terminals and the Logs are bit-identical;
- `puf_log` puts every name at its offset;
- split = 2 is refused without `MM_FINAL`.

Result (2026-10-15, both command lines): 159 episode ends, 0 mismatching arrays, 49 / 49 names, and
the split guard refused.

## Not ported / open

- **No 5.0 `make_ini3.py` or sweep config yet.** Step 2 above describes the edits to v2's
  generator.
- **`obs_size` must match at compile time**, so one 5.0 binary serves one observation width.
- **Category-balanced evaluation.** v2's `[eval_env]` weights validation markets equally. With
  mixed categories, use `cat_mask` to evaluate one category per run.
