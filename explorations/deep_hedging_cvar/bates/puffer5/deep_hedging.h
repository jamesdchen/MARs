/* PufferLib 5.0 env: ../hedge.h behind 5.0's env API (src/pufferenv.h).
 *
 * Copy this file and ../hedge.h to PufferLib/ocean/deep_hedging/ and
 * deep_hedging.ini to PufferLib/config/. All dynamics, the 32-step episode
 * layout (train.horizon = 32), shaping, w and the Log live in hedge.h.
 * Four continuous actions and 13 observations make every PufferNet tensor a
 * multiple of 4 floats, so 5.0's allocator adds no padding (its muon_step
 * walks the gradient buffer with unpadded offsets).
 */

#include <stdlib.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include "hedge.h"

#define ACT_SIZES {1, 1, 1, 1}
#define OBS_SIZE HEDGE_OBS
#define NUM_ATNS HEDGE_ACT

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    HedgeParams params;
    Hedge h;
};
typedef Env DeepHedging;

void puf_reset(DeepHedging* env) {
    hedge_start(&env->h, &env->params);
    hedge_write_obs(&env->h, &env->params, env->agents[0].observations);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
}

void puf_step(DeepHedging* env) {
    double noise[HEDGE_NOISE];
    Agent* a = &env->agents[0];
    if (env->h.k < HEDGE_DATES) {
        hedge_draw_noise(&env->h, &env->params, noise);
    }
    hedge_step(&env->h, &env->params, a->actions, noise, a->observations, a->rewards,
        a->terminals, &env->log);
}

void puf_render(DeepHedging* env) {
}

void puf_close(DeepHedging* env) {
}

// The trainer sets env->rng to the env index; mixed with the seed kwarg.
void puf_init(DeepHedging* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    double* fields = (double*)&env->params;
    for (int i = 0; i < HEDGE_NUM_PARAMS; i++) {
        fields[i] = dict_get(kwargs, HEDGE_PARAM_NAMES[i]);
    }
    hedge_params_finish(&env->params);
    uint64_t seed = (uint64_t)dict_get(kwargs, "seed");
    hedge_init(&env->h, &env->params, (seed << 32) + env->rng);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "ru", log->ru);
    dict_set(out, "loss", log->loss);
    dict_set(out, "excess", log->excess);
    dict_set(out, "w", log->w);
    dict_set(out, "trades_stock", log->trades_stock);
    dict_set(out, "trades_swap", log->trades_swap);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
