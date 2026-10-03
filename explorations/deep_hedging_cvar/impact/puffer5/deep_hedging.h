/* Deep hedging: CVaR hedging of a short call with transient price impact
 * and fixed costs, as a PufferLib 5.0 env (interface in SPEC.md).
 *
 * The model is ../market.py (simulate with the hard gate) and the arithmetic
 * mirrors the Cython env ../cy_hedging.pyx, with the state in double. The
 * fundamental price F follows geometric Brownian motion, our trades leave a
 * transient displacement on the quoted price S = F + impact (Obizhaeva &
 * Wang 2013), and every trade pays proportional and fixed costs. The action
 * is (target position, trade signal): trade to the clamped target iff the
 * signal is positive. The loss is L = payoff - final cash after liquidation.
 *
 * CVaR enters through the Rockafellar-Uryasev threshold w: the terminal
 * reward is -(L - w)^+ / scale, optionally spread over the dates by
 * potential-based shaping (Ng, Harada & Russell 1999), and each env moves w
 * toward VaR_alpha(L) after every episode by the Robbins-Monro step of
 * Bardou, Frikha & Pages (2009).
 *
 * An episode is 32 env steps so that train.horizon = 32 segments line up
 * with episodes: 30 hedging dates, then a step that keeps the expiry
 * observation and a step that starts the next episode.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {1, 1}
#define OBS_SIZE 8
#define NUM_ATNS 2
#define N_STEPS 30
#define EPISODE_STEPS 32

// score = -ru, the Rockafellar-Uryasev objective w + (L - w)^+ / (1 - alpha)
// at the env's w: unlike CVaR it is an average, so it can be logged as a sum,
// and it bounds CVaR from above. It is the metric of the hyperparameter sweep.
struct Log {
    float score;
    float perf;
    float ru;
    float loss;
    float excess;
    float w;
    float trades;
    float episode_return;
    float episode_length;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    uint64_t rng_state[4];
    int shaping;
    double s0, strike, sigma, maturity, cost, fixed_cost, kappa, decay, alpha;
    double target_low, target_high, w_eta, premium, scale, drift, vol;
    int k;
    int trades;
    double f, impact, cash, delta, w, phi, loss, episode_return;
};
typedef Env DeepHedging;

double bs_call(DeepHedging* env, double s, double tau) {
    double sd = env->sigma * sqrt(tau);
    double d1 = (log(s / env->strike) + 0.5 * sd * sd) / sd;
    return s * 0.5 * (1.0 + erf(d1 * 0.7071067811865476))
        - env->strike * 0.5 * (1.0 + erf((d1 - sd) * 0.7071067811865476));
}

// Cash after selling the whole position at quoted price s.
double close_out(DeepHedging* env, double s) {
    double d = env->delta;
    double fee = d != 0.0 ? env->fixed_cost : 0.0;
    return env->cash + d * s - 0.5 * env->kappa * d * d - env->cost * fabs(d) * s
        - fee;
}

// -(option - close_out - w)^+ / scale, with the Black-Scholes value of the
// option before expiry and its payoff at expiry.
double potential(DeepHedging* env, int date) {
    double s = env->f + env->impact;
    double option = s > env->strike ? s - env->strike : 0.0;
    if (date < N_STEPS) {
        option = bs_call(env, s, env->maturity * (1.0 - (double)date / N_STEPS));
    }
    double excess = option - close_out(env, s) - env->w;
    return excess > 0.0 ? -excess / env->scale : 0.0;
}

// market.make_obs, then a constant 1 (PufferNet has no biases).
void write_obs(DeepHedging* env, int date) {
    obs_t* obs = env->agents[0].observations;
    double s = env->f + env->impact;
    double wealth = env->cash + env->delta * s;
    obs[0] = (double)date / N_STEPS;
    obs[1] = log(s / env->strike) / (env->sigma * sqrt(env->maturity));
    obs[2] = env->delta;
    obs[3] = wealth / env->scale;
    obs[4] = env->w / env->scale;
    obs[5] = (env->w + wealth) / env->scale;
    obs[6] = env->kappa > 0.0 ? env->impact / env->kappa : 0.0;
    obs[7] = 1.0f;
}

void start_episode(DeepHedging* env) {
    env->k = 0;
    env->f = env->s0;
    env->impact = 0.0;
    env->cash = env->premium;
    env->delta = 0.0;
    env->trades = 0;
    env->episode_return = 0.0;
    env->phi = potential(env, 0);
    write_obs(env, 0);
}

void puf_reset(DeepHedging* env) {
    start_episode(env);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
}

// One env step. z is the fundamental noise of date k and is only read on
// hedging dates (k < N_STEPS).
void hedge_step(DeepHedging* env, double z) {
    Agent* agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
    if (env->k == EPISODE_STEPS - 1) {
        double excess = env->loss > env->w ? env->loss - env->w : 0.0;
        double ru = env->w + excess / (1.0 - env->alpha);
        env->log.score -= ru;
        env->log.perf -= ru / env->scale;
        env->log.ru += ru;
        env->log.loss += env->loss;
        env->log.excess += excess;
        env->log.w += env->w;
        env->log.trades += env->trades;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += EPISODE_STEPS;
        env->log.n += 1.0f;
        env->w += env->w_eta * env->scale
            * ((env->loss > env->w) / (1.0 - env->alpha) - 1.0);
        start_episode(env);
        agent->terminals[0] = 1.0f;
        return;
    }
    if (env->k == N_STEPS) {
        env->k += 1;
        write_obs(env, N_STEPS);
        return;
    }

    float* act = agent->actions;
    double s = env->f + env->impact;
    if (act[1] > 0.0f) {
        double target = act[0];
        if (target < env->target_low) {
            target = env->target_low;
        } else if (target > env->target_high) {
            target = env->target_high;
        }
        double q = target - env->delta;
        env->cash -= q * s + 0.5 * env->kappa * q * q + env->cost * fabs(q) * s
            + env->fixed_cost;
        env->impact += env->kappa * q;
        env->delta += q;
        env->trades += 1;
    }
    env->f *= exp(env->drift + env->vol * z);
    env->impact *= env->decay;
    env->k += 1;

    if (env->k == N_STEPS) {
        s = env->f + env->impact;
        env->loss = (s > env->strike ? s - env->strike : 0.0) - close_out(env, s);
        agent->terminals[0] = 1.0f;
    }
    double reward = 0.0;
    if (env->shaping) {
        double phi = potential(env, env->k);
        reward = phi - env->phi;
        env->phi = phi;
    } else if (env->k == N_STEPS) {
        double excess = env->loss - env->w;
        reward = excess > 0.0 ? -excess / env->scale : 0.0;
    }
    agent->rewards[0] = reward;
    env->episode_return += reward;
    write_obs(env, env->k);
}

// xoshiro256+ (Blackman & Vigna 2018); the top 53 bits as a uniform on (0, 1].
double rand_uniform(DeepHedging* env) {
    uint64_t* s = env->rng_state;
    uint64_t out = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return ((out >> 11) + 1) * 0x1.0p-53;
}

void puf_step(DeepHedging* env) {
    double z = 0.0;
    if (env->k < N_STEPS) {
        // Box-Muller
        double r = sqrt(-2.0 * log(rand_uniform(env)));
        z = r * cos(2.0 * M_PI * rand_uniform(env));
    }
    hedge_step(env, z);
}

void puf_render(DeepHedging* env) {
}

void puf_close(DeepHedging* env) {
}

void puf_init(DeepHedging* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->s0 = dict_get(kwargs, "s0");
    env->strike = dict_get(kwargs, "strike");
    env->sigma = dict_get(kwargs, "sigma");
    env->maturity = dict_get(kwargs, "maturity");
    env->cost = dict_get(kwargs, "cost");
    env->fixed_cost = dict_get(kwargs, "fixed_cost");
    env->kappa = dict_get(kwargs, "kappa");
    env->decay = pow(0.5, 1.0 / dict_get(kwargs, "half_life"));
    env->alpha = dict_get(kwargs, "alpha");
    env->target_low = dict_get(kwargs, "target_low");
    env->target_high = dict_get(kwargs, "target_high");
    env->w_eta = dict_get(kwargs, "w_eta");
    env->shaping = dict_get(kwargs, "shaping") != 0.0;
    env->scale = env->sigma * sqrt(env->maturity) * env->s0;
    double dt = env->maturity / N_STEPS;
    env->drift = (dict_get(kwargs, "mu") - 0.5 * env->sigma * env->sigma) * dt;
    env->vol = env->sigma * sqrt(dt);
    env->premium = bs_call(env, env->s0, env->maturity);
    env->w = dict_get(kwargs, "w_init") * env->scale;

    // splitmix64 (Steele, Lea & Flood 2014) seeds the generator from the
    // seed kwarg and env->rng, which the trainer sets to the env index.
    uint64_t x = ((uint64_t)dict_get(kwargs, "seed") << 32) ^ env->rng;
    for (int i = 0; i < 4; i++) {
        x += 0x9E3779B97F4A7C15ull;
        uint64_t y = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        y = (y ^ (y >> 27)) * 0x94D049BB133111EBull;
        env->rng_state[i] = y ^ (y >> 31);
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "ru", log->ru);
    dict_set(out, "loss", log->loss);
    dict_set(out, "excess", log->excess);
    dict_set(out, "w", log->w);
    dict_set(out, "trades", log->trades);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
