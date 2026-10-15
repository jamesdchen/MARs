/* ctypes harness of mm3.h for test_env3.py and bench3.py (no Python per step).
 *
 * m3_h_run: n envs, T steps on given actions (T x n x 3 int32). Env i starts on global market
 * markets[i] at offset starts[i] (or a random episode if markets[i] < 0) and auto-resets on its own
 * RNG. Records observations ((T + 1) x n x obs_size), rewards and terminals (T x n), M3_NSTATE
 * doubles of state after every step and each env's Log right after its first terminal step.
 * m3_h_release: test-only, unmaps the data so the next m3_h_open can load another directory.
 * m3_h_bench: steps per second of the bare core on random actions (single thread). */

#include <time.h>
#include "mm3.h"

#define M3_NSTATE 30

int m3_h_open(const char* dir, char* err, int n) {
    return m3_data_open(dir, err, n);
}

void m3_h_release(void) {
    m3_data_release();
}

int m3_h_sizeof_env(void) {
    return (int)sizeof(Mm3);
}

int m3_h_sizeof_log(void) {
    return (int)sizeof(Mm3Log);
}

int m3_h_num_params(void) {
    return M3_NUM_PARAMS;
}

static void m3_record(const Mm3* m, double* s) {
    s[0] = m->t;
    s[1] = m->market;
    s[2] = m->s0;
    s[3] = m->inv;
    s[4] = m->cash;
    s[5] = m->rebates;
    s[6] = m->liq;
    s[7] = m->wealth;
    s[8] = m->penalty;
    s[9] = m->n_fills;
    s[10] = m->fill_vol;
    s[11] = m->ord[0].live;
    s[12] = m->ord[0].px;
    s[13] = m->ord[0].rem;
    s[14] = m->ord[0].q;
    s[15] = m->ord[1].live;
    s[16] = m->ord[1].px;
    s[17] = m->ord[1].rem;
    s[18] = m->ord[1].q;
    s[19] = m->spread;
    s[20] = m->mo5;
    s[21] = m->mo30;
    s[22] = m->mo60;
    s[23] = m->rejects;
    s[24] = m->settle;
    s[25] = m->close_pen;
    s[26] = m->flat_cost;
    s[27] = m->rw_samples;
    s[28] = m->rw_scored;
    s[29] = m->rw_share;
}

int m3_h_run(const double* params, int n, int T, const int* markets, const int* starts,
        const int* actions, uint64_t seed, float* obs, float* rew, float* term, double* state,
        float* logs, char* err, int errlen) {
    const Mm3Params* P = (const Mm3Params*)params;
    int W = (int)P->obs_size;
    Mm3* ms = (Mm3*)calloc(n, sizeof(Mm3));
    Mm3Log* lg = (Mm3Log*)calloc(n, sizeof(Mm3Log));
    Mm3Log* first = (Mm3Log*)calloc(n, sizeof(Mm3Log));
    char* done = (char*)calloc(n, 1);
    int rc = 0;
    for (int i = 0; i < n; i++) {
        if (m3_init(&ms[i], P, seed + i, err, errlen)) {
            rc = 1;
            goto out;
        }
        if (markets[i] >= 0) {
            m3_begin(&ms[i], markets[i], starts[i]);
        } else {
            m3_new_episode(&ms[i]);
        }
        m3_write_obs(&ms[i], obs + (size_t)i * W);
        m3_record(&ms[i], state + (size_t)i * M3_NSTATE);
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            size_t k = (size_t)t * n + i;
            size_t o = (size_t)(t + 1) * n + i;
            m3_step(&ms[i], actions + k * M3_NUM_ATNS, obs + o * W, rew + k, term + k, &lg[i]);
            m3_record(&ms[i], state + o * M3_NSTATE);
            if (!done[i] && term[k] != 0.0f) {
                first[i] = lg[i];
                done[i] = 1;
            }
        }
    }
    memcpy(logs, first, (size_t)n * sizeof(Mm3Log));
out:
    free(ms);
    free(lg);
    free(first);
    free(done);
    return rc;
}

#ifndef M3_AHEAD
#define M3_AHEAD 4
#endif
double m3_h_bench(const double* params, int n, int T, uint64_t seed, double* log_out, double* cpu_sps) {
    const char* pl = getenv("MM_PIPELINE");
    int pipeline = pl == NULL || atoi(pl) != 0;
    char err[512];
    const Mm3Params* P = (const Mm3Params*)params;
    int W = (int)P->obs_size;
    Mm3* ms = (Mm3*)calloc(n, sizeof(Mm3));
    Mm3Log* lg = (Mm3Log*)calloc(n, sizeof(Mm3Log));
    float* obs = (float*)calloc((size_t)n * W, sizeof(float));
    float* rew = (float*)calloc(n, sizeof(float));
    float* term = (float*)calloc(n, sizeof(float));
    int na = 64;
    int* act = (int*)malloc((size_t)na * n * M3_NUM_ATNS * sizeof(int));
    uint64_t x = seed + 1;
    for (size_t j = 0; j < (size_t)na * n; j++) {
        for (int h = 0; h < M3_NUM_ATNS; h++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            act[j * M3_NUM_ATNS + h] = (int)(x % (h < 2 ? 7 : 3));
        }
    }
    for (int i = 0; i < n; i++) {
        if (m3_init(&ms[i], P, seed + i, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err);
            return -1.0;
        }
        m3_reset(&ms[i], obs + (size_t)i * W);
    }
    struct timespec a, b, ca, cb;
    clock_gettime(CLOCK_MONOTONIC, &a);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ca);
    for (int t = 0; t < T; t++) {
        const int* at = act + (size_t)(t % na) * n * M3_NUM_ATNS;
        for (int i = 0; i < n; i++) {
            if (pipeline) {
                if (i + 3 * M3_AHEAD < n) m3_prefetch_state(&ms[i + 3 * M3_AHEAD]);
                if (i + 2 * M3_AHEAD < n) m3_prefetch_data(&ms[i + 2 * M3_AHEAD]);
                if (i + M3_AHEAD < n) m3_prefetch_legs(&ms[i + M3_AHEAD]);
            }
            m3_step(&ms[i], at + (size_t)i * M3_NUM_ATNS, obs + (size_t)i * W, rew + i, term + i, &lg[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cb);
    double secs = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
    double csecs = (cb.tv_sec - ca.tv_sec) + 1e-9 * (cb.tv_nsec - ca.tv_nsec);
    if (cpu_sps) *cpu_sps = (double)n * T / csecs;
    Mm3Log tot;
    memset(&tot, 0, sizeof(tot));
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < sizeof(Mm3Log) / sizeof(float); j++) ((float*)&tot)[j] += ((float*)&lg[i])[j];
    }
    for (size_t j = 0; j < sizeof(Mm3Log) / sizeof(float); j++) log_out[j] = ((float*)&tot)[j];
    free(ms); free(lg); free(obs); free(rew); free(term); free(act);
    return (double)n * T / secs;
}
