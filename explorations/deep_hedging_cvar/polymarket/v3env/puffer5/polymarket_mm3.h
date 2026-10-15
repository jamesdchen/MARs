/* PufferLib 5.0 env: ../mm3.h (Polymarket market-making replay core v3) behind 5.0's env API
 * (src/pufferenv.h). Same design as ../../puffer5/polymarket_mm.h (the v2 wrapper); PUFFER5_V3.md
 * has the port notes.
 *
 * Copy this file and ../mm3.h to PufferLib/ocean/polymarket_mm3/ (the include below becomes
 * "mm3.h"), write an ini whose [env] has every name of M3_PARAM_NAMES plus data_dir, seed and
 * log_episodes, then `./build.sh polymarket_mm3 --float`.
 *   - OBS_SIZE is a compile-time constant in 5.0. It is MM3_OBS_SIZE (default M3_OBS_FULL = 80:
 *     v2's 50 slots, the 28 v3 slots, 2 zero pads); build with -DMM3_OBS_SIZE=50 for the v2
 *     observation. puf_init exits unless the ini's obs_size equals it.
 *   - The data is mapped once per process (m3_data_open's static guard); puf_init runs again for
 *     the eval trainer and must not remap.
 *   - The core is seeded with (seed << 32) + env index (5.0 sets env->rng to the index).
 *   - Actions arrive as floats holding exact integers and are converted to int.
 *   - log_episodes = k > 0 logs only each env's first k episodes (final evaluation, as v2).
 * No mutable globals: 5.0 steps envs concurrently (OpenMP). Errors exit(1) with a message. */

#include <stddef.h>
#include <stdlib.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#define Mm3Log Log          // the core's Log struct under the name 5.0's trainer requires (it reads log.n)
#include "../mm3.h"

#ifndef MM3_OBS_SIZE
#define MM3_OBS_SIZE M3_OBS_FULL
#endif
#define ACT_SIZES M3_ACT_SIZES
#define OBS_SIZE MM3_OBS_SIZE
#define NUM_ATNS M3_NUM_ATNS

typedef char pmm3_obs_check[(MM3_OBS_SIZE >= M3_OBS_V2 && MM3_OBS_SIZE <= M3_OBS_MAX) ? 1 : -1];

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    int log_episodes;
    int episodes;
    Mm3 mm;
};
typedef Env PolymarketMM3;

// Log fields in struct order, all floats, n last (the trainer sets env/n itself).
static const char* PMM3_LOG_NAMES[] = {
    "score", "perf", "episode_return", "episode_length", "pnl", "pnl_trading",
    "spread_capture", "markout_5", "markout_30", "markout_60", "inventory_pnl", "rebates",
    "liq_rewards", "inv_penalty", "fills", "fill_vol", "buy_vol", "sell_vol",
    "max_abs_inv", "final_abs_inv", "uptime_two_sided", "uptime_any", "reward_clip_frac", "rejects",
    "vol_crossed", "vol_stale", "vol_onesided", "vol_last30", "vol_last10", "vol_inside2",
    "vol_extreme", "vol_outside", "vol_inspread", "pnl_crossed", "pnl_stale", "pnl_onesided",
    "pnl_last30", "pnl_last10", "pnl_inside2", "pnl_extreme", "pnl_outside", "pnl_inspread",
    "close_penalty", "flat_cost", "settle_px", "rw_samples", "rw_scored", "rw_share",
    "episode_rows",
};
#define PMM3_LOG_N (int)(sizeof(PMM3_LOG_NAMES) / sizeof(PMM3_LOG_NAMES[0]))
typedef char pmm3_log_layout_check[sizeof(Mm3Log) == (PMM3_LOG_N + 1) * sizeof(float)
    && offsetof(Mm3Log, n) == PMM3_LOG_N * sizeof(float) ? 1 : -1];
static const char* PMM3_FLAG_NAMES[] = {
    "crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside", "inspread"};
#define PMM3_VOL0 (int)(offsetof(Mm3Log, vol_crossed) / sizeof(float))
#define PMM3_PNL0 (int)(offsetof(Mm3Log, pnl_crossed) / sizeof(float))

static void pmm3_fail(const char* msg) {
    fprintf(stderr, "polymarket_mm3: %s\n", msg);
    exit(1);
}

void puf_init(PolymarketMM3* env, Dict* kwargs) {
    char err[512];
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    DictItem* dir = dict_find(kwargs, "data_dir");
    if (!dir || !dir->str || !dir->str[0]) {
        pmm3_fail("[env] data_dir (a build directory) is required");
    }
    if (m3_data_open(dir->str, err, sizeof(err))) {
        pmm3_fail(err);
    }
    Mm3Params p;
    double* fields = (double*)&p;
    for (int i = 0; i < M3_NUM_PARAMS; i++) {
        fields[i] = dict_get(kwargs, M3_PARAM_NAMES[i]);
    }
    if ((int)p.obs_size != OBS_SIZE) {
        snprintf(err, sizeof(err), "[env] obs_size %d differs from the compiled OBS_SIZE %d (build with "
            "-DMM3_OBS_SIZE=%d or change the ini)", (int)p.obs_size, OBS_SIZE, (int)p.obs_size);
        pmm3_fail(err);
    }
    env->log_episodes = (int)dict_get(kwargs, "log_episodes");
    env->episodes = 0;
    uint64_t seed = ((uint64_t)dict_get(kwargs, "seed") << 32) + env->rng;
    if (m3_init(&env->mm, &p, seed, err, sizeof(err))) {
        pmm3_fail(err);
    }
}

void puf_reset(PolymarketMM3* env) {
    m3_reset(&env->mm, env->agents[0].observations);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
}

void puf_step(PolymarketMM3* env) {
    Agent* a = &env->agents[0];
    int act[M3_NUM_ATNS];
    for (int i = 0; i < M3_NUM_ATNS; i++) {
        act[i] = (int)a->actions[i];
    }
    Mm3Log scratch;
    Mm3Log* log = &env->log;
    if (env->log_episodes > 0 && env->episodes >= env->log_episodes) {
        memset(&scratch, 0, sizeof(scratch));
        log = &scratch;
    }
    m3_step(&env->mm, act, a->observations, a->rewards, a->terminals, log);
    if (a->terminals[0] != 0.0f) {
        env->episodes++;
    }
}

void puf_render(PolymarketMM3* env) {
}

void puf_close(PolymarketMM3* env) {
}

void puf_log(Log* log, Dict* out) {
    const float* f = (const float*)log;
    for (int i = 0; i < PMM3_LOG_N; i++) {
        dict_set(out, PMM3_LOG_NAMES[i], f[i]);
    }
    float vol = log->fill_vol, fill_pnl = log->spread_capture + log->markout_60;
    for (int i = 0; i < M3_NFLAGS; i++) {
        char key[64];
        snprintf(key, sizeof(key), "share_%s", PMM3_FLAG_NAMES[i]);
        dict_set(out, key, vol > 0.0f ? f[PMM3_VOL0 + i] / vol : 0.0f);
        snprintf(key, sizeof(key), "pnl_share_%s", PMM3_FLAG_NAMES[i]);
        dict_set(out, key, fill_pnl > 0.0f ? f[PMM3_PNL0 + i] / fill_pnl : NAN);
    }
}
