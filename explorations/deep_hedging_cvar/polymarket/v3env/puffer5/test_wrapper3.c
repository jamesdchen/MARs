/* Checks that the PufferLib 5.0 wrapper (polymarket_mm3.h) steps exactly like the core (../mm3.h),
 * on CPU, without CUDA:
 *
 *     cc -std=gnu11 -O2 -fopenmp -I$PUFFERLIB/src -I$RAYLIB_INCLUDE test_wrapper3.c -o t3 -lm
 *     ./t3 DATA_DIR [key=value ...]          # e.g. ./t3 ../../data/build liq_rewards=1 size_mode=1 Q=300
 *
 * (it also compiles as C++17: g++ -std=c++17 -x c++ ..., as nvcc compiles the env into
 * src/pufferl.cu). [env] is every name of M3_PARAM_NAMES at its default, then the key=value
 * overrides, plus data_dir, seed = 1 and log_episodes = 0; obs_size is set to the compiled
 * OBS_SIZE.
 * 1. N envs built as the trainer builds them (calloc, env.rng = index, puf_init on every env, then
 *    the buffers bound, then puf_reset), stepped T times with random float actions concurrently
 *    (OpenMP, as 5.0 runs puf_step), against m3_step called serially with seed (1 << 32) + i and
 *    the same actions as ints: observations, rewards, terminals and the Logs bit-identical.
 * 2. puf_log: every name sits at its field's offset; share_<flag> keys are added.
 * 3. split = 2 is refused (puf_init exits 1) with allow_test = 1 but no MM_FINAL. */

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>
#include "polymarket_mm3.h"

#define N 64
#define T 1500

static const int SIZES[M3_NUM_ATNS] = M3_ACT_SIZES;

static int split_refused(Dict* kw) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        unsetenv("MM_FINAL");
        if (!freopen("/dev/null", "w", stderr)) _exit(3);
        Dict d;
        memset(&d, 0, sizeof(d));
        dict_copy(&d, kw);
        puf_ini_set(&d, "split", "2");
        puf_ini_set(&d, "allow_test", "1");
        PolymarketMM3 env;
        memset(&env, 0, sizeof(env));
        puf_init(&env, &d);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s DATA_DIR [key=value ...]\n", argv[0]);
        return 2;
    }
    Dict kw;
    memset(&kw, 0, sizeof(kw));
    char buf[64];
    for (int i = 0; i < M3_NUM_PARAMS; i++) {
        snprintf(buf, sizeof(buf), "%.17g", M3_DEFAULTS[i]);
        puf_ini_set(&kw, M3_PARAM_NAMES[i], buf);
    }
    snprintf(buf, sizeof(buf), "%d", OBS_SIZE);
    puf_ini_set(&kw, "obs_size", buf);
    puf_ini_set(&kw, "data_dir", argv[1]);
    puf_ini_set(&kw, "seed", "1");
    puf_ini_set(&kw, "log_episodes", "0");
    for (int a = 2; a < argc; a++) {
        char key[64];
        const char* eq = strchr(argv[a], '=');
        if (!eq || eq - argv[a] >= (int)sizeof(key)) return 2;
        memcpy(key, argv[a], eq - argv[a]);
        key[eq - argv[a]] = 0;
        if (!dict_find(&kw, key)) {
            fprintf(stderr, "unknown key %s\n", key);
            return 2;
        }
        puf_ini_set(&kw, key, eq + 1);
    }
    float* wobs = (float*)calloc((size_t)N * OBS_SIZE, sizeof(float));
    float* wact = (float*)calloc((size_t)N * M3_NUM_ATNS, sizeof(float));
    float* wrew = (float*)calloc(N, sizeof(float));
    float* wterm = (float*)calloc(N, sizeof(float));
    float* cobs = (float*)calloc((size_t)N * OBS_SIZE, sizeof(float));
    float* crew = (float*)calloc(N, sizeof(float));
    float* cterm = (float*)calloc(N, sizeof(float));
    int* act = (int*)calloc((size_t)N * M3_NUM_ATNS, sizeof(int));
    PolymarketMM3* envs = (PolymarketMM3*)calloc(N, sizeof(PolymarketMM3));
    Mm3* cores = (Mm3*)calloc(N, sizeof(Mm3));
    Mm3Log* clogs = (Mm3Log*)calloc(N, sizeof(Mm3Log));
    for (int i = 0; i < N; i++) {
        envs[i].rng = i;
        puf_init(&envs[i], &kw);
    }
    for (int i = 0; i < N; i++) {
        Agent* a = &envs[i].agents[0];
        a->observations = wobs + (size_t)i * OBS_SIZE;
        a->actions = wact + (size_t)i * M3_NUM_ATNS;
        a->rewards = wrew + i;
        a->terminals = wterm + i;
        puf_reset(&envs[i]);
    }
    Mm3Params p;
    double* fields = (double*)&p;
    for (int i = 0; i < M3_NUM_PARAMS; i++) fields[i] = dict_get(&kw, M3_PARAM_NAMES[i]);
    char err[512];
    for (int i = 0; i < N; i++) {
        if (m3_init(&cores[i], &p, ((uint64_t)1 << 32) + (uint64_t)i, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err);
            return 1;
        }
        m3_reset(&cores[i], cobs + (size_t)i * OBS_SIZE);
    }
    long bad = memcmp(wobs, cobs, (size_t)N * OBS_SIZE * sizeof(float)) != 0, terms = 0;
    unsigned int rs = 7;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < N * M3_NUM_ATNS; i++) {
            act[i] = (int)(rand_r(&rs) % (unsigned)SIZES[i % M3_NUM_ATNS]);
            wact[i] = (float)act[i];
        }
        #pragma omp parallel for schedule(static, 4)
        for (int i = 0; i < N; i++) {
            puf_step(&envs[i]);
        }
        for (int i = 0; i < N; i++) {
            m3_step(&cores[i], act + i * M3_NUM_ATNS, cobs + (size_t)i * OBS_SIZE, crew + i, cterm + i, &clogs[i]);
            terms += cterm[i] != 0.0f;
        }
        bad += memcmp(wobs, cobs, (size_t)N * OBS_SIZE * sizeof(float)) != 0;
        bad += memcmp(wrew, crew, (size_t)N * sizeof(float)) != 0;
        bad += memcmp(wterm, cterm, (size_t)N * sizeof(float)) != 0;
    }
    double logged = 0;
    for (int i = 0; i < N; i++) {
        bad += memcmp(&envs[i].log, &clogs[i], sizeof(Mm3Log)) != 0;
        logged += envs[i].log.n;
    }
    int nf = 0;
    for (int i = 0; i < PMM3_LOG_N; i++) {
        const float* f = (const float*)&envs[0].log;
        Dict out;
        memset(&out, 0, sizeof(out));
        puf_log(&envs[0].log, &out);
        nf += dict_get(&out, PMM3_LOG_NAMES[i]) == (double)f[i] && out.size == PMM3_LOG_N + 2 * M3_NFLAGS;
        dict_clear(&out);
    }
    int refused = split_refused(&kw);
    printf("polymarket_mm3 (5.0 wrapper, OBS_SIZE %d): %d envs x %d steps (OpenMP), %ld episode ends, %g logged; "
        "mismatching arrays %ld; puf_log names at their offsets %d/%d; split=2 refused without MM_FINAL: %s\n",
        OBS_SIZE, N, T, terms, logged, bad, nf, PMM3_LOG_N, refused ? "yes" : "NO");
    int fail = bad != 0 || terms == 0 || logged != (double)terms || nf != PMM3_LOG_N || !refused;
    printf(fail ? "FAIL\n" : "ok\n");
    return fail;
}
