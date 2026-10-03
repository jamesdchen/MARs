/* Checks that the PufferLib 5.0 wrapper (deep_hedging.h) steps exactly like
 * the core it wraps: N envs built as the trainer builds them (env.rng = env
 * index, kwargs from deep_hedging.ini), random actions, T steps, compared
 * with ../hedge.h called directly with the same seeds.
 *
 *     cc -O2 -I$PUFFERLIB/src -I$PUFFERLIB/raylib-5.5_linux_amd64/include -I.. \
 *        test_wrapper.c -o test_wrapper -lm && ./test_wrapper deep_hedging.ini
 */

#include <stdio.h>
#include "deep_hedging.h"

#define N 256
#define T (3 * HEDGE_EPISODE)

int main(int argc, char** argv) {
    Ini ini = {0};
    puf_ini_load_file(&ini, argc > 1 ? argv[1] : "deep_hedging.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    static float obs[N][HEDGE_OBS], act[N][HEDGE_ACT], rew[N], term[N];
    static DeepHedging envs[N];
    static Hedge core[N];
    static Log logs[N];
    HedgeParams p = {0};
    double* fields = (double*)&p;
    for (int i = 0; i < HEDGE_NUM_PARAMS; i++) {
        fields[i] = dict_get(kwargs, HEDGE_PARAM_NAMES[i]);
    }
    hedge_params_finish(&p);
    uint64_t seed = (uint64_t)dict_get(kwargs, "seed");
    for (int i = 0; i < N; i++) {
        envs[i].rng = i;
        envs[i].agents[0] = (Agent){.observations = obs[i], .actions = act[i],
            .rewards = &rew[i], .terminals = &term[i]};
        puf_init(&envs[i], kwargs);
        puf_reset(&envs[i]);
        hedge_init(&core[i], &p, (seed << 32) + i);
    }
    unsigned int r = 7;
    double err = 0.0;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < N; i++) {
            for (int a = 0; a < HEDGE_ACT; a++) {
                act[i][a] = 4.0f * rand_r(&r) / RAND_MAX - 2.0f;
            }
            float o[HEDGE_OBS], rw, tm;
            double noise[HEDGE_NOISE];
            if (core[i].k < HEDGE_DATES) {
                hedge_draw_noise(&core[i], &p, noise);
            }
            hedge_step(&core[i], &p, act[i], noise, o, &rw, &tm, &logs[i]);
            puf_step(&envs[i]);
            for (int j = 0; j < HEDGE_OBS; j++) {
                err = fmax(err, fabs(o[j] - obs[i][j]));
            }
            err = fmax(err, fmax(fabs(rw - rew[i]), fabs(tm - term[i])));
        }
    }
    double n = 0;
    for (int i = 0; i < N; i++) {
        n += envs[i].log.n;
        err = fmax(err, fabs(envs[i].log.score - logs[i].score));
    }
    printf("5.0 wrapper vs core: %d envs, %d steps, %g episodes logged, max abs diff %g\n",
        N, T, n, err);
    if (err != 0.0 || n != (double)T / HEDGE_EPISODE * N) {
        printf("FAIL\n");
        return 1;
    }
    printf("ok\n");
    return 0;
}
