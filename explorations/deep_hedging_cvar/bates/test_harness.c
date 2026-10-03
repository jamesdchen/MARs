/* ctypes harness for test_c.py: runs N agents for T env steps of hedge.h
 * on given actions (T x N x 4 float32) and, if noise is not NULL, on
 * injected noise (N x episodes x 30 x n_sub x 4 float64); otherwise on the
 * agents' own generator. Records obs ((T + 1) x N x 12), rewards and
 * terminals (T x N), the state [k, w, phi, loss] ((T + 1) x N x 4) and the
 * Log of each agent (N x 11 floats). */

#include <stdlib.h>
#include "hedge.h"

static void record(const Hedge* h, double* st) {
    st[0] = h->k;
    st[1] = h->w;
    st[2] = h->phi;
    st[3] = h->loss;
}

int hedge_run(const double* values, int n, int T, const float* actions, const double* noise,
        uint64_t seed, float* obs, float* rewards, float* terminals, double* state, float* logs) {
    HedgeParams p;
    double* fields = (double*)&p;
    for (int i = 0; i < HEDGE_NUM_PARAMS; i++) {
        fields[i] = values[i];
    }
    hedge_params_finish(&p);
    if (p.subs < 1 || p.subs > HEDGE_MAX_SUB) {
        return 1;
    }
    int episodes = (T + HEDGE_EPISODE - 1) / HEDGE_EPISODE;
    size_t block = (size_t)HEDGE_DATES * p.subs * 4;
    Hedge* hs = calloc(n, sizeof(Hedge));
    Log* lg = calloc(n, sizeof(Log));
    double buf[HEDGE_NOISE];
    for (int i = 0; i < n; i++) {
        hedge_init(&hs[i], &p, seed + i);
        hedge_write_obs(&hs[i], &p, obs + (size_t)i * HEDGE_OBS);
        record(&hs[i], state + (size_t)i * 4);
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            Hedge* h = &hs[i];
            const double* z = buf;
            if (h->k < HEDGE_DATES) {
                if (noise) {
                    z = noise + ((size_t)i * episodes + t / HEDGE_EPISODE) * block
                        + (size_t)h->k * p.subs * 4;
                } else {
                    hedge_draw_noise(h, &p, buf);
                }
            }
            size_t o = (size_t)(t + 1) * n + i;
            hedge_step(h, &p, actions + ((size_t)t * n + i) * HEDGE_ACT, z,
                obs + o * HEDGE_OBS, rewards + (size_t)t * n + i, terminals + (size_t)t * n + i,
                &lg[i]);
            record(h, state + o * 4);
        }
    }
    memcpy(logs, lg, (size_t)n * sizeof(Log));
    free(hs);
    free(lg);
    return 0;
}
