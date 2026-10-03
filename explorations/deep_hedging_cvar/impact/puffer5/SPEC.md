# PufferLib 5.0 port: interface contract

PufferLib 5.0 (branch `5.0` of github.com/PufferAI/PufferLib, commit
`6ffa5b1`) trains with a native CUDA trainer (`src/pufferl.cu`) that compiles
the env in from a C header. This folder ports the impact + fixed cost
hedging env to that format. The model is defined by `../market.py`
(`ImpactConfig`, `simulate` with `gate_mode='hard'`); the Cython env
`../cy_hedging.pyx` is an existing, tested implementation of the same
dynamics. Everything below must hold exactly.

## Files

| file | role |
|---|---|
| `deep_hedging.h` | the env, copied to `PufferLib/ocean/deep_hedging/deep_hedging.h` |
| `deep_hedging.ini` | config, copied to `PufferLib/config/deep_hedging.ini` |
| `test_c_env.py` (+ C harness) | checks the C env against `market.simulate` |
| `puffernet.py` | loads a 5.0 `.bin` checkpoint into PyTorch and runs its policy |
| `test_puffernet.py` (+ C harness) | checks `puffernet.py` against `src/puffercpu.c` |

## Observation (`typedef float obs_t; OBS_SIZE 8`)

Index 0–6 are exactly `market.make_obs(k, s, delta, wealth, w, impact, cfg)`:
`[k / n_steps, log(S/K) / (sigma sqrt T), delta, wealth / scale, w / scale,
(w + wealth) / scale, impact / kappa (0 if kappa == 0)]`, with
`wealth = cash + delta * S` and `S = F + impact`. Index 7 is the constant
`1.0` (PufferNet has no biases).

## Action (`NUM_ATNS 2`, `ACT_SIZES {1, 1}`, continuous)

`[target, signal]`. Trade iff `signal > 0`; then `target` is clamped to
`[target_low, target_high] = [-0.5, 1.5]` and the position moves to it.
The trainer does not clip continuous actions; the env must.

## Dynamics

Identical to `market.simulate` with the hard gate, state in double:
trade cost `q S + kappa q^2 / 2 + cost |q| S + fixed_cost`, then
`impact += kappa q`, then `F *= exp(drift + vol z)`, `impact *= decay`.
Cash starts at the Black-Scholes premium of the call at `s0` (compute it
in C with `erf`). Expiry: payoff `(S_n - K)^+` on the quoted price, then
liquidation with the same costs (`fixed_cost` only if `delta != 0`),
`L = payoff - cash_after_liquidation`.

Normals: the env's own generator (e.g. splitmix64-seeded xoshiro256+ or
PCG32, with Box–Muller), seeded from `env->rng` (the trainer sets it to the
env index) mixed with the `seed` kwarg. Draw `z` when it is used.
For the equivalence test the harness must be able to inject `z`; keep the
core step a function that takes `z` as an argument.

## Episode layout (32 env steps; `train.horizon = 32`)

5.0's advantage kernel works within a horizon segment and needs
`horizon % 4 == 0`, so an episode is exactly 32 env steps and the
segments line up with episodes:

| step `k` on entry | action | after the step |
|---|---|---|
| 0 … 28 | hedge | `k + 1`, obs of date `k + 1`, shaped reward, `terminal = 0` |
| 29 | hedge | `k = 30`, expiry obs (time fraction 1), last shaped reward (or the terminal reward if unshaped), **`terminal = 1`** |
| 30 | ignored | `k = 31`, same expiry obs, reward 0, `terminal = 0` |
| 31 | ignored | new episode: update w, reset, obs of date 0, reward 0, **`terminal = 1`** |

`terminal = 1` on the expiry step stops bootstrapping from the expiry
value; `terminal = 1` on the reset step zeroes the MinGRU state before date
0. `puf_reset` itself starts an episode at `k = 0` (used once at start).

## Reward

`shaping = 1` (default): every hedge step pays `Phi(s_{k+1}) - Phi(s_k)`,
with `Phi` exactly as `CyHedging.potential` in `../cy_hedging.pyx`
(`-(option - close_out - w)^+ / scale`, Black-Scholes option value before
expiry, payoff at expiry). `shaping = 0`: only `-(L - w)^+ / scale` at
expiry. Rewards are not clipped by the 5.0 trainer.

## Threshold w

Each env keeps its own `w` (price units), initialized to `w_init * scale`,
and after every episode applies the Robbins–Monro VaR step of Bardou,
Frikha & Pagès (2009):
`w += w_eta * scale * (1{L > w} / (1 - alpha) - 1)`.

## Log (`struct Log`, floats only, `n` last)

Sum over finished episodes: `score` (= -L), `perf` (= -L / scale),
`loss` (L), `excess` ((L - w)^+), `w`, `trades` (trades in the episode),
`episode_return`, `episode_length` (32), then `n`.

## `[env]` kwargs (`dict_get` returns double)

`s0 strike sigma mu maturity cost fixed_cost kappa half_life alpha
target_low target_high w_init w_eta shaping seed`, defaults equal to
`ImpactConfig()` and `w_init = 0.3`, `w_eta = 0.01`. `n_steps` is fixed at
30 by the episode layout.

## Policy weights (`.bin`, flat fp32)

As documented in `src/puffercpu.c` (`make_puffernet`, `forward_puffernet`):
encoder `(H, 8)`, decoder `(2 + 1, H)` (last row is the value), logstd
`(2,)`, then `L` MinGRU projections `(3H, H)`; each block padded to a
multiple of 8 floats; no biases. Deterministic action = decoder output 0–1.
