/* ctypes harness for test_env.py and bench.py (no Python per step).
 *
 * mm_h_run: n envs, T steps of mm.h on given actions (T x n x 3 int32). Env i starts on global
 * market markets[i] at start offset starts[i] (or a random episode if markets[i] < 0) and then
 * auto-resets on its own RNG. Records observations ((T + 1) x n x MM_OBS), rewards and terminals
 * (T x n), MM_NSTATE doubles of state after every step ((T + 1) x n) and each env's Log as it
 * stood right after its first terminal step (all zeros if it never finished an episode).
 * mm_h_bench: steps per second of the bare core on random actions (single thread). */

#include <time.h>
#include "mm.h"

#define MM_NSTATE 24

int mm_h_open(const char* dir, char* err, int n) {
    return mm_data_open(dir, err, n);
}

int mm_h_sizeof_env(void) {
    return (int)sizeof(Mm);
}

int mm_h_sizeof_log(void) {
    return (int)sizeof(Log);
}

static void record(const Mm* m, double* s) {
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
}

int mm_h_run(const double* params, int n, int T, const int* markets, const int* starts,
        const int* actions, uint64_t seed, float* obs, float* rew, float* term, double* state,
        float* logs, char* err, int errlen) {
    Mm* ms = (Mm*)calloc(n, sizeof(Mm));
    Log* lg = (Log*)calloc(n, sizeof(Log));
    Log* first = (Log*)calloc(n, sizeof(Log));
    char* done = (char*)calloc(n, 1);
    for (int i = 0; i < n; i++) {
        if (mm_init(&ms[i], (const MmParams*)params, seed + i, err, errlen)) {
            free(ms);
            free(lg);
            free(first);
            free(done);
            return 1;
        }
        if (markets[i] >= 0) {
            mm_begin(&ms[i], markets[i], starts[i]);
        } else {
            mm_new_episode(&ms[i]);
        }
        mm_write_obs(&ms[i], obs + (size_t)i * MM_OBS);
        record(&ms[i], state + (size_t)i * MM_NSTATE);
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            size_t k = (size_t)t * n + i;
            size_t o = (size_t)(t + 1) * n + i;
            mm_step(&ms[i], actions + k * MM_NUM_ATNS, obs + o * MM_OBS, rew + k, term + k, &lg[i]);
            record(&ms[i], state + o * MM_NSTATE);
            if (!done[i] && term[k] != 0.0f) {
                first[i] = lg[i];
                done[i] = 1;
            }
        }
    }
    memcpy(logs, first, (size_t)n * sizeof(Log));
    free(ms);
    free(lg);
    free(first);
    free(done);
    return 0;
}

// Bare-core throughput: n envs stepped round-robin for T steps on random actions. Returns steps
// per wall-clock second; *cpu_sps gets steps per second of this thread's CPU time (robust to
// other load on the machine). log_out receives the Log summed over envs.
#ifndef MM_AHEAD
#define MM_AHEAD 4
#endif
double mm_h_bench(const double* params, int n, int T, uint64_t seed, double* log_out, double* cpu_sps) {
    const char* pl = getenv("MM_PIPELINE");
    int pipeline = pl == NULL || atoi(pl) != 0;
    char err[512];
    Mm* ms = (Mm*)calloc(n, sizeof(Mm));
    Log* lg = (Log*)calloc(n, sizeof(Log));
    float* obs = (float*)calloc((size_t)n * MM_OBS, sizeof(float));
    float* rew = (float*)calloc(n, sizeof(float));
    float* term = (float*)calloc(n, sizeof(float));
    int na = 64;
    int* act = (int*)malloc((size_t)na * n * MM_NUM_ATNS * sizeof(int));
    uint64_t x = seed + 1;
    for (size_t j = 0; j < (size_t)na * n; j++) {
        for (int h = 0; h < MM_NUM_ATNS; h++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            act[j * MM_NUM_ATNS + h] = (int)(x % (h < 2 ? 7 : 3));
        }
    }
    for (int i = 0; i < n; i++) {
        if (mm_init(&ms[i], (const MmParams*)params, seed + i, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err);
            return -1.0;
        }
        mm_reset(&ms[i], obs + (size_t)i * MM_OBS);
    }
    struct timespec a, b, ca, cb;
    clock_gettime(CLOCK_MONOTONIC, &a);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ca);
    for (int t = 0; t < T; t++) {
        const int* at = act + (size_t)(t % na) * n * MM_NUM_ATNS;
        for (int i = 0; i < n; i++) {
            if (pipeline) {                 // the same software pipelining as binding.c
                if (i + 3 * MM_AHEAD < n) mm_prefetch_state(&ms[i + 3 * MM_AHEAD]);
                if (i + 2 * MM_AHEAD < n) mm_prefetch_data(&ms[i + 2 * MM_AHEAD]);
                if (i + MM_AHEAD < n) mm_prefetch_legs(&ms[i + MM_AHEAD]);
            }
            mm_step(&ms[i], at + (size_t)i * MM_NUM_ATNS, obs + (size_t)i * MM_OBS, rew + i, term + i, &lg[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cb);
    double secs = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
    double csecs = (cb.tv_sec - ca.tv_sec) + 1e-9 * (cb.tv_nsec - ca.tv_nsec);
    if (cpu_sps) *cpu_sps = (double)n * T / csecs;
    Log tot;
    memset(&tot, 0, sizeof(tot));
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < sizeof(Log) / sizeof(float); j++) ((float*)&tot)[j] += ((float*)&lg[i])[j];
    }
    for (size_t j = 0; j < sizeof(Log) / sizeof(float); j++) log_out[j] = ((float*)&tot)[j];
    free(ms); free(lg); free(obs); free(rew); free(term); free(act);
    return (double)n * T / secs;
}
