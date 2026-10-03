# Part 3: a short option book under Bates dynamics

Parts 1 and 2 hedged one call under geometric Brownian motion with a
single hedging instrument. Here the market and the baselines are both
harder.

**Market** (`market.py`):
- **Dynamics:** Bates (1996), i.e. Heston stochastic variance with leverage
  (ρ = −0.7, vol of vol 0.6) plus lognormal crash jumps (rate 2 a year, mean
  −6%). Simulated with full-truncation Euler on 4 substeps a day.
- **Book:** short one 100-strike call and one 95-strike put, 30 days, hedged
  daily, priced at inception with the Lewis Fourier formula.
- **Two instruments:**
  - the stock, with transient price impact, proportional cost and a fixed fee;
  - a variance swap on daily log-returns, traded at its model value, with a
    cost for each contract and the fee.
- **Crowd:** other traders' orders feed the same transient impact as ours,
  as in the jointly aggregated impact of Neuman & Voss (2023). Two kinds:
  - 8 dealers, short 3 copies of our book between them. Each re-hedges to
    the book's BS delta when the price after our trade leaves its no-trade
    band (half-widths from 0.02 to 0.3 shares for each book).
  - random orders: a Poisson number of 0.3-share orders, with mean
    0.5 + |r|, where r is the last return in daily standard deviations.
    Each goes the way of r with probability 0.7.

  The crowd reacts through thresholds and counts, so on a single path the
  loss jumps when our trade moves a dealer out of its band or changes the
  order count. The last return is added to the observation (13 features).
- **Decisions:** 4 actions, a target and a trade signal for each instrument.
  An instrument trades to its target only when its signal is positive.
- **Objective:** CVaR at 95% of the loss, written in its
  Rockafellar–Uryasev form.

**One C env** (`hedge.h`), with no Cython:
- **Two wrappers:** PufferLib 3.0's native binding (`binding.c`, CPU) and
  PufferLib 5.0's env API (`puffer5/deep_hedging.h`, CUDA trainer).
- **`test_c.py`:**
  - the core matches `market.py` on identical noise and actions, with the
    crowd off, at its default and strong: losses to 4e-13, observations to
    6e-8, rewards to float32;
  - its own generator reproduces the Fourier option prices and the variance
    swap strike within 1.5 standard errors;
  - the 3.0 binding steps bit-for-bit like the core.
- **`puffer5/test_wrapper.c`:** the 5.0 wrapper also steps bit-for-bit like
  the core.

## Baselines

All tuning uses validation paths (seed 999). Every method is scored once on
the same 200k test paths (seed 12345), with deterministic actions.

| baseline | what it is | tuning |
|---|---|---|
| BS delta | book delta at the initial vol, stock only, daily | none |
| Bates delta (plain, minimum-variance) | model delta from Fourier greeks (`greeks.py`), stock only, daily | none |
| Bates delta-vega | model delta in stock, plus `y* = (dB/dv)/(dV/dv)` variance swaps, daily | none |
| no-trade band | band around the delta-vega targets with gamma/vanna/volga-based widths: Whalley–Wilmott (proportional) and Altarovici–Muhle-Karbe–Soner (fixed cost) terms, partial adjustment, target multipliers | 8 constants, Optuna TPE, 32 trials |
| pathwise deep hedging, straight-through | backprop through `market.simulate`; exact forward pass, sigmoid gradient for trade decisions | none (GPU sweep in the notebook) |
| pathwise deep hedging, hybrid | pathwise gradient for targets plus score-function gradient for sampled trade decisions (Schulman et al. 2015), leave-one-out baseline over 4 samples (Kool et al. 2019); deterministic at test | none (GPU sweep in the notebook) |

The hybrid estimator was checked against an exact enumeration of all
decision sequences on a 2-date book. Its error stays within Monte Carlo
error; without the score term the error is 33 times larger
(`test_pathwise.py`).

## Results on CPU, with the crowd

Training ran on 4 CPU threads in this container. PPO is PufferLib 3.0 with
the C env, untuned apart from an entropy bonus of 0.01 (Adam, 200M steps).

| strategy | val CVaR95 | test mean | std | VaR95 | **test CVaR95** | stock trades | swap trades | train time (s) | sim steps |
|---|---|---|---|---|---|---|---|---|---|
| no hedge | 10.63 | 0.06 | 3.89 | 7.17 | 10.71 | 0 | 0 | – | – |
| BS delta | 13.16 | 1.83 | 3.21 | 6.98 | 13.22 | 30 | 0 | – | – |
| Bates delta | 13.62 | 1.80 | 3.35 | 7.29 | 13.70 | 30 | 0 | – | – |
| Bates min-variance delta | 12.54 | 1.77 | 3.02 | 6.44 | 12.56 | 30 | 0 | – | – |
| Bates delta-vega | 6.43 | 2.58 | 2.06 | 3.80 | 6.34 | 30 | 30 | – | – |
| band, theory constants | 5.51 | 0.95 | 2.14 | 4.32 | 5.47 | 3.3 | 1.6 | – | – |
| **tuned no-trade band** | 3.45 | 1.01 | 1.94 | 2.97 | **3.45** | 6.6 | 1.0 | 170 (tuning) | 96M |
| PPO, PufferLib 3.0 + C env, entropy 0.01 | 4.01 | 1.32 | 4.58 | 3.66 | 4.02 | 18.0 | 1.0 | 1684 | 200M |
| pathwise, straight-through | 2.88 | 1.47 | 1.99 | 2.67 | 2.87 | 23.6 | 1.0 | 582 | 246M |
| **pathwise, hybrid** | 2.82 | 1.58 | 1.99 | 2.64 | **2.81** | 27.4 | 1.0 | 565 | 246M |

Without the crowd (`results/no_crowd/cpu/summary.md`, run before the crowd
was added, 12 observations):

| strategy | test CVaR95 |
|---|---|
| Bates delta-vega | 5.99 |
| tuned no-trade band | 3.04 |
| PPO, no entropy bonus, rewards unscaled | 8.34 |
| pathwise, straight-through | 2.93 |
| pathwise, hybrid | 2.76 |

### Does the crowd break the pathwise gradient?

On a path the loss is discontinuous in our actions, but its expectation
over paths is smooth: the noise has a density, so the chance that a dealer
crosses its band or that the order count steps changes smoothly with our
trades. Autograd sees only the slope between the jumps and misses what the
jumps add to the gradient of the expectation (the pathwise method needs the
loss to be continuous in the parameters; Glasserman 2004, §7.2).
`crowd_gradient.py` measures how much it misses. Policy: θ × the BS book
delta, rebalanced every date (no trade decisions), derivative at θ = 1.
It compares autograd with central finite differences on the same 200k
paths, which do see the jumps; the gap is paired on the paths.

| crowd | derivative | autograd | finite diff, h = 0.02 | gap |
|---|---|---|---|---|
| none | dCVaR/dθ | 8.771 | 8.770 | −0.001 ± 0.005 |
| default | dE[L]/dθ | 1.532 | 1.533 | 0.002 ± 0.000 |
| default | dCVaR/dθ | 8.894 | 8.900 | 0.006 ± 0.007 |
| strong (12 books, 16 dealers, bands from 0.001, 3 + 2\|r\| orders, follow 0.9) | dE[L]/dθ | 2.936 | 2.954 | 0.018 ± 0.002 |
| strong | dCVaR/dθ | 11.441 | 11.475 | 0.034 ± 0.017 |

Full table: `results/crowd_gradient.md`. The gap is real but below 1%. A
hedger's trades are small: they move the price by about a tenth of a daily
standard deviation, so they rarely decide whether the crowd trades. Raising
impact until they do makes the crowd's own feedback unstable (with 4 times
the impact and the strong crowd, the dealers' gamma times impact exceeds 1
and prices run away).

### What we see

1. **Delta hedging alone is worse than not hedging.** The book is short a
   put, so its delta hedge is long stock. A crash jump hits the put and the
   hedge together. Only the variance swap, whose realized variance jumps in
   a crash, offsets that risk: daily delta-vega roughly halves CVaR.
2. **Costs matter as much as the model.** The tuned band holds a static
   variance swap position and rebalances stock about 7 times. That cuts
   CVaR from 6.3 to 3.4.
3. **The crowd hurts the hand-built strategies** (the band 3.04 → 3.45,
   delta-vega 5.99 → 6.34) **and barely moves pathwise deep hedging**
   (hybrid 2.76 → 2.81, straight-through 2.93 → 2.87). Its gradient misses
   under 1% (above), so it still trains well and still beats the tuned
   band. Both pathwise variants trade the stock on most dates in small
   amounts and hold the swap.
4. **PPO needs an entropy bonus.** Without one, PPO stops trading after
   date 0, with rewards scaled (point 5) or not (`results/no_crowd` logs).
   The trade signal is Gaussian and random trades pay the fee, so PPO
   pushes the signal down until it never explores rebalancing again (8.34;
   entropy fell to −3.3). An entropy bonus of 0.01
   keeps it rebalancing on about half the dates, and it reaches 4.02, behind
   the tuned band. It holds 3 variance swaps on every path, the position
   limit, where the band and pathwise hold about 1.7. Its training
   objective stopped improving after about 8 minutes
   (`results/cpu/training.png`).
5. **Rewards are scaled by 0.1.** PufferLib 3.0 clips rewards to [−1, 1].
   Unscaled, 12% of the reward signal's total size sat beyond the clip
   (crash days), so PPO saw crash losses flattened. Scaling leaves the
   optimal policy unchanged. PufferLib 5.0 does not clip; it gets the same
   scale.

![positions](results/cpu/positions.png)
![losses](results/cpu/loss_hist.png)
![training](results/cpu/training.png)

## GPU and sweeps

`puffer5/colab.ipynb`, generated by `puffer5/make_notebook.py`, runs on a
Colab GPU:
- builds PufferLib 5.0 (pinned commit) with `hedge.h` compiled in;
- runs the three equivalence tests;
- trains PPO for 200M and 1B steps;
- sweeps PPO with PufferLib's Protein (`puffer5/sweep.ini`), with the top
  runs re-scored on validation (`puffer5/select_sweep.py`);
- tunes the band and sweeps hybrid pathwise with the same number of trials;
- scores everything with `evaluate.py`.

The 5.0 config (`puffer5/make_ini.py`) sets the entropy bonus to 0.01; the
sweep searches 1e-5 to 0.03. The 5.0 env uses 13 observations and 4
actions, so no PufferNet tensor needs padding. That sidesteps the misalignment in PufferLib 5.0's Muon step
documented in part 2.

## Reproduce (CPU)

```bash
source ../.venv/bin/activate && pip install optuna   # ../setup.sh makes the venv
python setup.py build_ext --inplace     # C env for PufferLib 3.0
python test_c.py                        # core vs market.py, binding vs core
python test_greeks.py && python test_pathwise.py
python crowd_gradient.py > results/crowd_gradient.md
python train_ppo.py --ent-coef 0.01     # results/ppo.pt
python tune_baselines.py --trials 32 --device cpu --out results/baselines.json
for g in ste hybrid; do python train_pathwise.py --gate $g --iters 1000 --batch 8192 \
  --out results/pathwise_$g.pt; done
python evaluate.py --device cpu --out results/cpu --runs "baselines|baselines|results/baselines.json" \
  "ppo3|PPO, PufferLib 3.0 + C env, entropy 0.01|results/ppo.pt" \
  "pathwise|pathwise, ste gate|results/pathwise_ste.pt" \
  "pathwise|pathwise, hybrid gate|results/pathwise_hybrid.pt"
```

## References

- D. Bates, *Jumps and stochastic volatility: exchange rate processes
  implicit in Deutsche mark options*, Rev. Financial Studies 9, 1996.
- S. Heston, *A closed-form solution for options with stochastic
  volatility*, Rev. Financial Studies 6, 1993.
- A. Lewis, *A simple option formula for general jump-diffusion and other
  exponential Lévy processes*, 2001.
- R. Lord, R. Koekkoek, D. van Dijk, *A comparison of biased simulation
  schemes for stochastic volatility models*, Quantitative Finance 10, 2010.
- A. E. Whalley, P. Wilmott, *An asymptotic analysis of an optimal hedging model for option
  pricing with transaction costs*, Math. Finance 7, 1997.
- A. Altarovici, J. Muhle-Karbe, H. M. Soner, *Asymptotics for fixed
  transaction costs*, Finance and Stochastics 19, 2015.
- E. Neuman, M. Voss, *Trading with the crowd*, Math. Finance 33, 2023.
- P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2004.
- N. Gârleanu, L. Pedersen, *Dynamic trading with predictable returns and
  transaction costs*, J. Finance 68, 2013.
- J. Schulman, N. Heess, T. Weber, P. Abbeel, *Gradient estimation using
  stochastic computation graphs*, NeurIPS 2015.
- W. Kool, H. van Hoof, M. Welling, *Buy 4 REINFORCE samples, get a
  baseline for free!*, 2019.
- Plus the references of parts 1 and 2.
