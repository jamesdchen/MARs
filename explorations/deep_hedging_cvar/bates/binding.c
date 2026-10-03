/* PufferLib 3.0 binding of hedge.h: PufferLib's own native-env interface
 * (pufferlib/ocean/env_binding.h), which steps every agent in C and writes
 * straight into the trainer's numpy buffers. Built by setup.py. */

#include "hedge.h"

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    HedgeParams params;
    Hedge h;
} Bates;

void c_reset(Bates* env) {
    hedge_start(&env->h, &env->params);
    hedge_write_obs(&env->h, &env->params, env->observations);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

void c_step(Bates* env) {
    double noise[HEDGE_NOISE];
    if (env->h.k < HEDGE_DATES) {
        hedge_draw_noise(&env->h, &env->params, noise);
    }
    float terminal;
    hedge_step(&env->h, &env->params, env->actions, noise, env->observations,
        env->rewards, &terminal, &env->log);
    env->terminals[0] = terminal != 0.0f;
}

void c_render(Bates* env) {
}

void c_close(Bates* env) {
}

#define Env Bates
#include "env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    double* fields = (double*)&env->params;
    for (int i = 0; i < HEDGE_NUM_PARAMS; i++) {
        fields[i] = unpack(kwargs, (char*)HEDGE_PARAM_NAMES[i]);
        if (PyErr_Occurred()) {
            return -1;
        }
    }
    hedge_params_finish(&env->params);
    if (env->params.subs < 1 || env->params.subs > HEDGE_MAX_SUB) {
        PyErr_SetString(PyExc_ValueError, "n_sub must be in [1, HEDGE_MAX_SUB]");
        return -1;
    }
    // env_binding sets kwargs["seed"] to this env's seed (index + seed * num_envs).
    hedge_init(&env->h, &env->params, (uint64_t)unpack(kwargs, "seed"));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "ru", log->ru);
    assign_to_dict(dict, "loss", log->loss);
    assign_to_dict(dict, "excess", log->excess);
    assign_to_dict(dict, "w", log->w);
    assign_to_dict(dict, "trades_stock", log->trades_stock);
    assign_to_dict(dict, "trades_swap", log->trades_swap);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
