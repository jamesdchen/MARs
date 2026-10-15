/* PufferLib 3.0 binding of mm.h: PufferLib's native-env interface (pufferlib/ocean/env_binding.h),
 * which steps every env in C and writes straight into the trainer's numpy buffers. Built by
 * setup.py. Keyword arguments: data_dir (str) plus every name in MM_PARAM_NAMES (env.py fills
 * the defaults); env_binding adds seed = env index + seed * num_envs. */

#include "mm.h"

typedef struct {
    Log log;
    float* observations;
    int* actions;              // MultiDiscrete: int32 [3] per env
    float* rewards;
    unsigned char* terminals;
    Mm mm;
} MmEnv;

void c_reset(MmEnv* env) {
    mm_reset(&env->mm, env->observations);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

void c_step(MmEnv* env) {
    float terminal;
    mm_step(&env->mm, env->actions, env->observations, env->rewards, &terminal, &env->log);
    env->terminals[0] = terminal != 0.0f;
}

void c_render(MmEnv* env) {
}

void c_close(MmEnv* env) {
    // The mmapped data is shared by every env of the process and never unmapped.
}

#define Env MmEnv
#include "env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    char err[512];
    PyObject* dir = PyDict_GetItemString(kwargs, "data_dir");
    if (dir == NULL || !PyUnicode_Check(dir)) {
        PyErr_SetString(PyExc_TypeError, "mm: keyword data_dir (str) is required");
        return -1;
    }
    if (mm_data_open(PyUnicode_AsUTF8(dir), err, sizeof(err))) {
        PyErr_SetString(PyExc_RuntimeError, err);
        return -1;
    }
    MmParams p;
    double* fields = (double*)&p;
    for (int i = 0; i < MM_NUM_PARAMS; i++) {
        fields[i] = unpack(kwargs, (char*)MM_PARAM_NAMES[i]);
        if (PyErr_Occurred()) {
            return -1;
        }
    }
    uint64_t seed = (uint64_t)unpack(kwargs, "seed");
    if (mm_init(&env->mm, &p, seed, err, sizeof(err))) {
        PyErr_SetString(PyExc_ValueError, err);
        return -1;
    }
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "pnl", log->pnl);
    assign_to_dict(dict, "pnl_trading", log->pnl_trading);
    assign_to_dict(dict, "spread_capture", log->spread_capture);
    assign_to_dict(dict, "markout_5", log->markout_5);
    assign_to_dict(dict, "markout_30", log->markout_30);
    assign_to_dict(dict, "markout_60", log->markout_60);
    assign_to_dict(dict, "inventory_pnl", log->inventory_pnl);
    assign_to_dict(dict, "rebates", log->rebates);
    assign_to_dict(dict, "liq_rewards", log->liq_rewards);
    assign_to_dict(dict, "inv_penalty", log->inv_penalty);
    assign_to_dict(dict, "fills", log->fills);
    assign_to_dict(dict, "fill_vol", log->fill_vol);
    assign_to_dict(dict, "buy_vol", log->buy_vol);
    assign_to_dict(dict, "sell_vol", log->sell_vol);
    assign_to_dict(dict, "max_abs_inv", log->max_abs_inv);
    assign_to_dict(dict, "final_abs_inv", log->final_abs_inv);
    assign_to_dict(dict, "uptime_two_sided", log->uptime_two_sided);
    assign_to_dict(dict, "uptime_any", log->uptime_any);
    assign_to_dict(dict, "reward_clip_frac", log->reward_clip_frac);
    assign_to_dict(dict, "rejects", log->rejects);
    assign_to_dict(dict, "vol_crossed", log->vol_crossed);
    assign_to_dict(dict, "vol_stale", log->vol_stale);
    assign_to_dict(dict, "vol_onesided", log->vol_onesided);
    assign_to_dict(dict, "vol_last30", log->vol_last30);
    assign_to_dict(dict, "vol_last10", log->vol_last10);
    assign_to_dict(dict, "vol_inside2", log->vol_inside2);
    assign_to_dict(dict, "vol_extreme", log->vol_extreme);
    assign_to_dict(dict, "vol_outside", log->vol_outside);
    assign_to_dict(dict, "vol_inspread", log->vol_inspread);
    assign_to_dict(dict, "pnl_crossed", log->pnl_crossed);
    assign_to_dict(dict, "pnl_stale", log->pnl_stale);
    assign_to_dict(dict, "pnl_onesided", log->pnl_onesided);
    assign_to_dict(dict, "pnl_last30", log->pnl_last30);
    assign_to_dict(dict, "pnl_last10", log->pnl_last10);
    assign_to_dict(dict, "pnl_inside2", log->pnl_inside2);
    assign_to_dict(dict, "pnl_extreme", log->pnl_extreme);
    assign_to_dict(dict, "pnl_outside", log->pnl_outside);
    assign_to_dict(dict, "pnl_inspread", log->pnl_inspread);
    return 0;
}
