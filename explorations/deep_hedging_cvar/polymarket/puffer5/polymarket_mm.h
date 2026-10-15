/* PufferLib 5.0 env: ../mm.h (Polymarket market-making replay core) behind 5.0's env API
 * (src/pufferenv.h).
 *
 * Copy this file and ../mm.h to PufferLib/ocean/polymarket_mm/ and polymarket_mm.ini (from
 * make_ini.py) to PufferLib/config/, then `./build.sh polymarket_mm [--float]`. The core owns
 * the MDP: dynamics, rewards, auto-reset, the Log. This wrapper only
 *   - reads [env]: data_dir (string; absolute or relative to the PufferLib root), every name in
 *     MM_PARAM_NAMES, seed, and log_episodes;
 *   - maps the data once per process (mm_data_open's static guard; puf_init runs again for the
 *     eval trainer and must not remap);
 *   - seeds the core with (seed << 32) + env index (5.0 sets env->rng to the index), so with
 *     cycle_stride > 0 env i starts on market i mod n of its split;
 *   - converts the three float actions (exact integers from 5.0's sampler) to int;
 *   - with log_episodes = k > 0, adds only the first k finished episodes of each env to the Log
 *     (later ones go to a scratch Log). The final evaluation uses k = 1 and
 *     eval_episodes = eval_agents, so the trainer's eval loop, which stops at a count of
 *     finished episodes, waits for every env's first market instead of counting the short
 *     markets that finish first (make_ini.py).
 * No mutable globals: 5.0 steps envs concurrently (OpenMP). Errors exit(1) with a message, as
 * dict_get does; in a sweep that marks the trial failed.
 */

#include <stddef.h>
#include <stdlib.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include "mm.h"

#define ACT_SIZES MM_ACT_SIZES
#define OBS_SIZE MM_OBS
#define NUM_ATNS MM_NUM_ATNS

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    int log_episodes;   // 0: log every episode; k > 0: only this env's first k
    int episodes;       // finished episodes
    Mm mm;
};
typedef Env PolymarketMM;

// Log fields in struct order, all floats, n last (the trainer sets env/n itself).
// test_wrapper.c checks every name against offsetof.
static const char* PMM_LOG_NAMES[] = {
    "score", "perf", "episode_return", "episode_length", "pnl", "pnl_trading", "spread_capture",
    "markout_5", "markout_30", "markout_60", "inventory_pnl", "rebates", "liq_rewards",
    "inv_penalty", "fills", "fill_vol", "buy_vol", "sell_vol", "max_abs_inv", "final_abs_inv",
    "uptime_two_sided", "uptime_any", "reward_clip_frac", "rejects",
    "vol_crossed", "vol_stale", "vol_onesided", "vol_last30", "vol_last10", "vol_inside2",
    "vol_extreme", "vol_outside", "vol_inspread",
    "pnl_crossed", "pnl_stale", "pnl_onesided", "pnl_last30", "pnl_last10", "pnl_inside2",
    "pnl_extreme", "pnl_outside", "pnl_inspread",
};
#define PMM_LOG_N (int)(sizeof(PMM_LOG_NAMES) / sizeof(PMM_LOG_NAMES[0]))
// Exploit flags (EVALUATION.md T5), in the order of MM_FL_*: Log.vol_<flag> (fill volume) and
// Log.pnl_<flag> (spread capture + 60 s markout of those fills) are MM_NFLAGS floats each.
static const char* PMM_FLAG_NAMES[] = {
    "crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside", "inspread"};
#define PMM_NFLAG_NAMES (int)(sizeof(PMM_FLAG_NAMES) / sizeof(PMM_FLAG_NAMES[0]))
#define PMM_VOL0 (int)(offsetof(Log, vol_crossed) / sizeof(float))
#define PMM_PNL0 (int)(offsetof(Log, pnl_crossed) / sizeof(float))
// Fail to compile if mm.h's Log or flags change shape.
typedef char pmm_log_layout_check[sizeof(Log) == (PMM_LOG_N + 1) * sizeof(float)
    && offsetof(Log, n) == PMM_LOG_N * sizeof(float) ? 1 : -1];
typedef char pmm_flag_check[MM_NFLAGS == PMM_NFLAG_NAMES
    && offsetof(Log, vol_inspread) == (PMM_VOL0 + MM_NFLAGS - 1) * sizeof(float)
    && PMM_PNL0 == PMM_VOL0 + MM_NFLAGS
    && offsetof(Log, pnl_inspread) == (PMM_PNL0 + MM_NFLAGS - 1) * sizeof(float) ? 1 : -1];

static void pmm_fail(const char* msg) {
    fprintf(stderr, "polymarket_mm: %s\n", msg);
    exit(1);
}

void puf_init(PolymarketMM* env, Dict* kwargs) {
    char err[512];
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    DictItem* dir = dict_find(kwargs, "data_dir");
    if (!dir || !dir->str || !dir->str[0]) {
        pmm_fail("[env] data_dir (the directory of data/build) is required");
    }
    if (mm_data_open(dir->str, err, sizeof(err))) {
        pmm_fail(err);
    }
    MmParams p;
    double* fields = (double*)&p;
    for (int i = 0; i < MM_NUM_PARAMS; i++) {
        fields[i] = dict_get(kwargs, MM_PARAM_NAMES[i]);
    }
    env->log_episodes = (int)dict_get(kwargs, "log_episodes");
    env->episodes = 0;
    uint64_t seed = ((uint64_t)dict_get(kwargs, "seed") << 32) + env->rng;
    if (mm_init(&env->mm, &p, seed, err, sizeof(err))) {
        pmm_fail(err);
    }
}

void puf_reset(PolymarketMM* env) {
    mm_reset(&env->mm, env->agents[0].observations);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
}

void puf_step(PolymarketMM* env) {
    Agent* a = &env->agents[0];
    int act[MM_NUM_ATNS];
    for (int i = 0; i < MM_NUM_ATNS; i++) {
        act[i] = (int)a->actions[i];
    }
    Log scratch;
    Log* log = &env->log;
    if (env->log_episodes > 0 && env->episodes >= env->log_episodes) {
        memset(&scratch, 0, sizeof(scratch));
        log = &scratch;
    }
    mm_step(&env->mm, act, a->observations, a->rewards, a->terminals, log);
    if (a->terminals[0] != 0.0f) {
        env->episodes++;
    }
}

void puf_render(PolymarketMM* env) {
}

// The mapped data is shared by every env of the process and never unmapped.
void puf_close(PolymarketMM* env) {
}

// The trainer passes the Log summed over envs and divided by n.
void puf_log(Log* log, Dict* out) {
    const float* f = (const float*)log;
    for (int i = 0; i < PMM_LOG_N; i++) {
        dict_set(out, PMM_LOG_NAMES[i], f[i]);
    }
    // As env.py: share of fill volume, and of fill P&L (spread capture + 60 s markout), per
    // exploit flag; the P&L share is NaN when that total is not positive.
    float vol = log->fill_vol, fill_pnl = log->spread_capture + log->markout_60;
    for (int i = 0; i < MM_NFLAGS; i++) {
        char key[64];
        snprintf(key, sizeof(key), "share_%s", PMM_FLAG_NAMES[i]);
        dict_set(out, key, vol > 0.0f ? f[PMM_VOL0 + i] / vol : 0.0f);
        snprintf(key, sizeof(key), "pnl_share_%s", PMM_FLAG_NAMES[i]);
        dict_set(out, key, fill_pnl > 0.0f ? f[PMM_PNL0 + i] / fill_pnl : NAN);
    }
}
