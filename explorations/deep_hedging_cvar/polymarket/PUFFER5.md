# PufferLib 5.0 portability: rules for the Polymarket env core

Pin: PufferLib 5.0 at `6ffa5b1` (2026-09-13), the same pin as `bates/puffer5/make_notebook.py:9`.
The latest `origin/5.0` is `ea7f989` (2026-10-14). It is 3 commits ahead and changes only cuBLAS
handles and workspaces, which are now per stream (`src/algo.cu`, `src/pufferl.cu`).
`pufferenv.h`, `ini.h`, `puffercpu.c`, `build.sh` and `config/default.ini` are identical. The
Muon offset bug and the reward clamp below are still present. Bumping the pin is safe, and the env
API is the same. Line numbers refer to `6ffa5b1`. "3.0" means the `pufferlib-3.0.0` sdist that
`../setup.sh` installs.

## Rules for the env core

1. **One pure-C core (`pm.h`) with no PufferLib includes.** It must compile both as C (3.0 setuptools)
   and as C++17: nvcc compiles `src/pufferl.cu`, which does `#include ENV_HEADER` (pufferl.cu:65,
   build.sh:505-519). So cast `malloc` results, and avoid `restrict`, VLAs, `_Generic` and
   out-of-order designated initializers.
2. **The core API takes ints and floats, never PufferLib types.** For example:
   `pm_step(Pm*, const int* act, float* obs, float* rew, float* term, Log*)`.
   - The 5.0 wrapper casts `agents[0].actions[i]` (float) to int and writes `terminals` as float.
   - The 3.0 wrapper reads int32 actions and writes `unsigned char` terminals.
3. **Observations: float32, O(1) scale, last feature a constant 1.0.** PufferNet has no biases at
   all and the trainer does no observation normalization.
   - The default 5.0 build stores rollout observations, rewards and values as bf16 (8-bit mantissa).
     So encode relative quantities (ticks from mid, log sizes, centred time-to-resolution), never
     absolute levels that differ in the 3rd digit, and clip to about [-5, 5].
   - `OBS_SIZE` is free. A multiple of 8 is nice for GEMM K-alignment but not required.
4. **Actions: discrete or multi-discrete. `ACT_SIZES` is a compile-time brace list, e.g. `{7, 7, 3}`.**
   - Every head needs at least 2 options: size 1 means "continuous", and mixing the two is rejected.
   - `NUM_ATNS` is the number of heads.
   - Do not rely on action masks: 3.0's trainer ignores them. Make every action legal (clamp, or
     treat as a no-op inside the core). The 5.0 wrapper may still write `action_mask` as an extra.
5. **Muon alignment.** Discrete heads have no `logstd` tensor, so every parameter tensor is a
   multiple of `hidden_size`. The only requirement is `hidden_size % 8 == 0` (bf16; `% 4` with
   `--float`). Every pow2 value in the sweeps qualifies. `OBS_SIZE` and `ACT_SIZES` are
   unconstrained. "Obs count multiple of 4" was never the real constraint: only a continuous
   `logstd` with `NUM_ATNS` not a multiple of 4 or 8 triggers the bug.
6. **Rewards: dense, per step (e.g. mark-to-market PnL change), scaled so |r| <= 1 almost always.**
   Both 3.0 and 5.0 clamp rewards to [-1, 1]. `bates/README.md` (item 5) is wrong that 5.0 does
   not clip. Write the reward and the terminal on every step: 5.0 zeroes them before each step and
   3.0 does not.
7. **Episodes: the env auto-resets inside step.**
   - On the last step, write the final reward, `terminal=1` and the new episode's first observation
     together.
   - 5.0 has no truncation flag, and 3.0 stores only terminals. So every episode end is a terminal
     with no bootstrap. Make it a true terminal: flatten or mark inventory to market at the end of
     the window, or end at market resolution.
   - Episodes longer than `horizon` are fine.
   - Randomize episode length or start offset so episodes desynchronize. The reward that lands at
     index 0 of a segment is never used, so with in-sync episodes whose length is a multiple of
     `horizon`, the terminal reward is lost every time.
   - Never depend on `horizon`: `default.ini` sweeps it over 32-512.
8. **Data: one flat, read-only binary file.**
   - It holds fixed-size records with precomputed per-snapshot features and an index of
     market/episode offsets.
   - The core's `pm_data_open(path)` mmaps it (`PROT_READ`, `MAP_PRIVATE`) into a static global,
     once, guarded by a static flag.
   - Each env keeps only cursors into it. `close` never unmaps it.
   - The path comes from the ini string `[env] data_path` (5.0) or a Python kwarg (3.0).
   - Keep `Env` small: 5.0 callocs `total_agents` Env structs before it counts agents.
9. **Thread safety and RNG.**
   - `step` runs concurrently on different envs (OpenMP worker threads in both trainers' C paths),
     so there must be no mutable globals and no `rand()`.
   - Use a per-env RNG seeded from `(seed, env index)`. 5.0 sets `env->rng` to the env index, and
     you add `[env] seed`. 3.0 passes `kwargs["seed"] = i + seed*num_envs`.
10. **Init and reset.** `init` must not touch the agent buffers: 5.0 binds them only after every
    `puf_init`. Write the first observation in `reset`. `init` may run several times per process
    (eval probe, then a fresh eval trainer), hence the static data guard.
11. **Log: a flat struct of floats with an `n` field (episodes finished).** Include `score`, which
    is the sweep metric by default, and `perf`. Accumulate at episode end. Both trainers sum Logs
    over envs and divide by `n`.
12. **Build 5.0 with `./build.sh pm --float`** for numerical parity with 3.0, i.e. fp32
    observations, rewards and values in the rollout, as bates did.

## Evidence

### a. Env API (`src/pufferenv.h`, `src/pufferl.cu`, `ocean/template/template.h`)
- **Agent struct.** `Agent {obs_t* observations; float* actions; float* rewards; float* terminals;
  unsigned char* action_mask; int policy;}` (pufferenv.h:24-31).
- **Entry points:** `puf_init(Env*, Dict*)`, `puf_reset`, `puf_step`, `puf_render`, `puf_close`,
  `puf_log(Log*, Dict*)` (pufferenv.h:44-50).
- **Env struct.** The env defines `struct Log` and `struct Env`, typedef'd in pufferenv.h:21-22.
  `Env` must have `Log log; Agent agents[N]; int tag, boundary_reached, num_agents; unsigned int
  rng;`. The trainer writes these fields (pufferl.cu:1003, 1085-1086; template.h:20-30).
- **`obs_t`.** The env header must `typedef obs_t` before including pufferenv.h; build.sh:460-462
  greps for it. It is float (63 envs), `unsigned char` (26), bf16 or precision_t (5).
  - Obs to rollout: a D2D copy if `sizeof(obs_t) == sizeof(precision_t)`, otherwise a cast
    (pufferl.cu:804-811).
  - `precision_t` is bf16 unless built with `--float` (pufferl.cu:41-49, build.sh:42).
- **Sizes.** `OBS_SIZE` is per-agent floats. `NUM_ATNS` is the number of heads, i.e. the stride of
  `actions`. `ACT_SIZES` gives the per-head sizes, with 1 meaning continuous (pufferl.cu:1852-1872,
  1023-1032).
- **Env creation.** `Env* envs = calloc(total_agents, sizeof(Env))`, then for each env:
  `envs[i].rng = i; puf_init(&envs[i], ek)`. This is serial, and runs until enough agents exist
  (pufferl.cu:1000-1007). Buffer pointers are bound afterwards (1069-1084).
- **Stepping.** Per step: memset rewards and terminals, then `#pragma omp parallel for` over
  `puf_step` (1180-1185). `puf_reset` runs only at (re)start (1205-1214); auto-reset is the env's
  job (template.h:47-58).
- **Logging.** The Log is summed as floats when `n != 0` (922-938), divided by `n`, passed to
  `puf_log`, and keys get the prefix `env/` (1363-1381).
- **Re-initialization.** `eval_make` calls `puf_init`/`puf_close` on a probe Env (2915-2918) and
  builds a new trainer, which runs `puf_init` on every env again (2948).

### b. Action spaces
- **Allowed spaces.** Discrete and multi-discrete are supported. Mixed continuous/discrete is
  asserted out (pufferl.cu:1866-1867). The decoder width is the sum of `ACT_SIZES` plus 1 for the
  value (1914).
- **Sampling.** `sample_logits` (pufferl.cu:592-700) computes a per-head logsumexp with masking,
  then samples by inverse CDF. It stores `actions[aidx] = (float)sampled` into a float32 buffer
  ("Float32 preserves large categorical IDs", 678-682). So the env reads exact integers in float.
- **CPU reference.** `_multidiscrete` in puffercpu.c:181-252 (at most 32 heads, puffercpu.c:433).
- **Compile-time constant.** `ACT_SIZES` is read in a `constexpr` function (algo.cu:1277-1287), so
  it must be a compile-time constant.
- **3.0 for comparison.** Discrete/MultiDiscrete actions are an int32 buffer (pufferlib.py:31-34).
  The trainer ignores masks ("not yet handling masks", pufferl.py:277).
- **Muon bug.**
  - The allocator rounds every tensor start up to 16 bytes (pufferl.cu:241, 263).
  - `muon_step` walks grads with `offset += numel` and no padding (algo.cu:1158-1163).
  - Parameter tensors: encoder `(H, OBS)` (algo.cu:583-587); decoder `(sum+1, H)` plus `logstd
    (1, NUM_ATNS)` only if continuous (636-644); MinGRU `(3H, H)` per layer (741-747).
  - With discrete actions every tensor's size is k·H, so H%8==0 (bf16) or H%4==0 (float) is enough.

### c. Config and data loading
- **Every ini value is also kept as a string.** `DictItem` has `str`, `value` and `values`
  (ini.h:11-17), and `puf_ini_set` stores the raw string, then parses a number or list (ini.h:343-362).
- **`puf_init` gets the raw `[env]` Dict** (pufferl.cu:1818, 1895), so
  `dict_find(kwargs, "data_path")->str` works. `dict_get_str` exits if the key is missing
  (ini.h:107-114).
- **Path syntax.** Quotes are stripped (ini.h:492). `#` or `;` outside quotes starts a comment
  (ini.h:190-199). A CLI override `--env.data_path=...` only replaces a key that already exists
  (ini.h:405-409). Relative paths resolve against the PufferLib root, as ocean's `resources/...` do.
- **Precedent: boxoban.**
  - `puf_init` reads an optional string `map_bin` (`map_item->str`, boxoban.h:456-472).
  - The first `init` loads maps behind `static int boxoban_maps_ready` (boxoban.h:129-139).
  - The maps go into a global `MAP_BASE` via `mmap(PROT_READ, MAP_PRIVATE)`
    (boxoban_maps.h:43, 533-558).
- **Other precedents.**
  - drive: a `MY_VEC_INIT` hook (drive.h:68; pufferl.cu:996-999) with per-env files under
    `resources/drive` (drive.h:1661-1675). Not needed here.
  - Compile-time `-D` defines need build.sh edits per env (build.sh:141-208), so prefer the ini
    string.
- **3.0.** `my_init` gets the PyObject kwargs (env_binding.h:438), so `PyUnicode_AsUTF8` on
  `data_path` feeds the same core loader. `unpack()` handles numbers only (env_binding.h:615-641).
  Buffers are bound before `my_init` (417-420), and the per-env seed is `i + seed*num_envs`
  (423-430).

### d. PufferNet (`algo.cu`, CPU reference `puffercpu.c:530-595`)
- **Encoder.** `x = W_enc obs`: no bias, no activation (algo.cu:557-566). Kaiming-uniform init,
  bound `sqrt(2)/sqrt(OBS)` (170-187, 574-581).
- **MinGRU, `num_layers` layers**, each with an unbiased `W (3H, H)` (741-747). The gate
  (algo.cu:200-226, scan 278-297):
  - `z = σ(g)`
  - `h̃ = a >= 0 ? a + 0.5 : σ(a)`
  - `h = lerp(h, h̃, z)`
  - `out = σ(p)·h + (1-σ(p))·x` (highway)
- **Decoder.** One unbiased `W (sum(ACT_SIZES)+1, H)`, where the last column is the value
  (617-625, 636-644). Init gain is 1 for the decoder and for MinGRU (627-634, 730-739).
- **Config keys:** `[policy] hidden_size`, `num_layers` (pufferl.cu:1785-1786); defaults 128 and 4
  (default.ini).
- **Recurrent state.**
  - It is zeroed when this step's terminal flag is set, before the obs is encoded (pufferl.cu:743-760,
    852-853; train scan algo.cu:280-282).
  - It carries across segments unless `reset_every_horizon=1` (default.ini;
    pufferl.cu:855-860, 1491-1499).
  - Gradients are truncated at segment boundaries.
- **Implication for observations.** No bias means an all-zero input gives zero logits and value,
  hence the constant 1.0 feature, which the highway carries to the decoder as an effective bias.
  No normalization and bf16 storage mean O(1), offset-free features.

### e. Segments, advantages, terminals
- **Horizon asserts** (pufferl.cu:3413-3420):
  - `horizon % ADV_VEC_WIDTH == 0`, i.e. 8 for bf16 and 4 for float;
  - `minibatch_size % horizon == 0`;
  - `minibatch_size <= horizon*total_agents`;
  - `total_agents % (minibatch_size/horizon) == 0`.
- **Rollout layout.** Rollout row `t` holds `(o_t, r_t, d_t)` from the previous step plus the action
  sampled at `t`.
- **Advantage.** `puff_advantage` (algo.cu:1698-1747) runs per segment of `horizon` steps with
  `delta_i = r[i+1] + γ V[i+1](1-d[i+1]) - V[i]`.
  - Its advantage is 0 at `i = H-1`, and `r[0]` is never read.
  - So the transition from action `H-1` (its reward lands in the next segment's row 0) is dropped
    once per segment. That is 1/H of the signal with dense rewards and desynchronized episodes, but
    100% of a terminal reward that always lands on row 0.
  - The bates hedge.h used 30 dates plus 2 pad steps in a 32-step segment for exactly this
    (hedge.h:11-16).
  - 3.0 is identical: `puff_advantage_row` loops `t = horizon-2 .. 0` on `rewards[t+1]`
    (extensions/pufferlib.cpp:28-41).
- **Truncations.** 3.0 stores only `d` (pufferl.py:231 "TODO: Handle truncations", 275), and 5.0
  has no truncation buffer. Both bootstrap through `V[i+1]` unless the terminal flag is set.

### f. 3.0 vs 5.0: nothing forces divergent cores
These differences belong in the wrappers:
- obs pointer type: `obs_t` vs the numpy dtype;
- action dtype: float vs int32;
- terminal dtype: float vs uint8;
- kwargs: Dict and `.str` vs PyDict;
- seeding: `env->rng` index plus `[env] seed` vs `kwargs["seed"]`;
- buffer binding: after init (5.0) vs before (3.0);
- masks: 5.0 only.

These are shared constraints the core must already respect:
- the [-1, 1] reward clamp (3.0 pufferl.py:255; 5.0 pufferl.cu:1487-1489);
- no truncations;
- auto-reset;
- same-segment GAE;
- a float Log with `n`.

Hazards that matter only in 5.0, which the rules above already cover:
- C++ compilation;
- bf16 storage;
- concurrent `puf_step` within and across buffers;
- the calloc of `total_agents` Envs;
- repeated `puf_init`.

Test pattern to reuse: a 5.0 wrapper test like `bates/puffer5/test_wrapper.c`, compiled with
`-I PufferLib/src -I raylib/include` on CPU, checks that it steps bit-for-bit like the core.
`./build.sh pm --cpu` (puffercpu.c eval binary) also compile-checks the header without CUDA.
