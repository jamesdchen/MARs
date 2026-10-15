/* Checks that the PufferLib 5.0 wrapper (polymarket_mm.h) steps exactly like the core it wraps
 * (../mm.h), on CPU, without CUDA:
 *
 *     cc -O2 -fopenmp -I$PUFFERLIB/src -I$PUFFERLIB/raylib-5.5_linux_amd64/include -I.. \
 *        test_wrapper.c -o test_wrapper -lm && ./test_wrapper polymarket_mm.ini [data_dir]
 *
 * (also compiles as C++17: g++ -std=c++17 -x c++ ..., as nvcc compiles the env into
 * src/pufferl.cu). data_dir overrides the ini's [env] data_dir, which is relative to the
 * PufferLib root.
 *
 * 1. Training config ([env]): N envs built as the trainer builds them (calloc, env.rng = index,
 *    puf_init on every env, then the buffers bound, then puf_reset), stepped T times with random
 *    actions given as floats, concurrently (OpenMP, as 5.0 runs puf_step), against mm.h called
 *    directly and serially with seed (seed << 32) + i and the same actions as ints.
 *    Observations, rewards and terminals at every step and the final Logs are bit-identical.
 * 2. Final-evaluation config ([env] with the ini's [eval_env] applied, as patch_pufferlib.py
 *    does): env i starts on validation market i mod n_val from the open, so each validation
 *    market is played k times by k * n_val envs; each env logs exactly one episode, and that
 *    Log equals the core's Log at its first terminal.
 * 3. split = 2 is refused (puf_init exits 1) without MM_FINAL=1, and without allow_test = 1.
 * 4. puf_log: every name of PMM_LOG_NAMES sits at its field's offset in mm.h's Log (checked
 *    with offsetof), and the summed-then-averaged Log (as vec_log) comes out under those names
 *    plus share_<flag> and pnl_share_<flag>.
 */

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>
#include "polymarket_mm.h"

#define N 128
#define T 2000
#define K_EVAL 2
#define T_EVAL 2900   // > 2880 decisions of a 4h market at dt = 5

static const int SIZES[MM_NUM_ATNS] = MM_ACT_SIZES;

typedef struct {
    float* obs;
    float* act;
    float* rew;
    float* term;
} Bufs;

static Bufs bufs_alloc(int n) {
    Bufs b;
    b.obs = (float*)calloc((size_t)n * MM_OBS, sizeof(float));
    b.act = (float*)calloc((size_t)n * MM_NUM_ATNS, sizeof(float));
    b.rew = (float*)calloc(n, sizeof(float));
    b.term = (float*)calloc(n, sizeof(float));
    return b;
}

// As the trainer: calloc, rng = index, puf_init for all, then bind buffers, then puf_reset.
static PolymarketMM* make_envs(int n, Dict* kw, Bufs b) {
    PolymarketMM* envs = (PolymarketMM*)calloc(n, sizeof(PolymarketMM));
    for (int i = 0; i < n; i++) {
        envs[i].rng = i;
        puf_init(&envs[i], kw);
    }
    for (int i = 0; i < n; i++) {
        Agent* a = &envs[i].agents[0];
        a->observations = b.obs + (size_t)i * MM_OBS;
        a->actions = b.act + (size_t)i * MM_NUM_ATNS;
        a->rewards = b.rew + i;
        a->terminals = b.term + i;
    }
    for (int i = 0; i < n; i++) {
        puf_reset(&envs[i]);
    }
    return envs;
}

static Mm* make_cores(int n, Dict* kw, float* obs) {
    MmParams p;
    double* fields = (double*)&p;
    for (int i = 0; i < MM_NUM_PARAMS; i++) {
        fields[i] = dict_get(kw, MM_PARAM_NAMES[i]);
    }
    uint64_t seed = (uint64_t)dict_get(kw, "seed");
    Mm* cores = (Mm*)calloc(n, sizeof(Mm));
    char err[512];
    for (int i = 0; i < n; i++) {
        if (mm_init(&cores[i], &p, (seed << 32) + (uint64_t)i, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err);
            exit(1);
        }
        mm_reset(&cores[i], obs + (size_t)i * MM_OBS);
    }
    return cores;
}

#define FIELD(x) {#x, offsetof(Log, x)}
static const struct { const char* name; size_t off; } LOG_FIELDS[] = {
    FIELD(score), FIELD(perf), FIELD(episode_return), FIELD(episode_length), FIELD(pnl),
    FIELD(pnl_trading), FIELD(spread_capture), FIELD(markout_5), FIELD(markout_30),
    FIELD(markout_60), FIELD(inventory_pnl), FIELD(rebates), FIELD(liq_rewards),
    FIELD(inv_penalty), FIELD(fills), FIELD(fill_vol), FIELD(buy_vol), FIELD(sell_vol),
    FIELD(max_abs_inv), FIELD(final_abs_inv), FIELD(uptime_two_sided), FIELD(uptime_any),
    FIELD(reward_clip_frac), FIELD(rejects), FIELD(vol_crossed), FIELD(vol_stale),
    FIELD(vol_onesided), FIELD(vol_last30), FIELD(vol_last10), FIELD(vol_inside2),
    FIELD(vol_extreme), FIELD(vol_outside), FIELD(vol_inspread), FIELD(pnl_crossed),
    FIELD(pnl_stale), FIELD(pnl_onesided), FIELD(pnl_last30), FIELD(pnl_last10),
    FIELD(pnl_inside2), FIELD(pnl_extreme), FIELD(pnl_outside), FIELD(pnl_inspread),
};

// Names vs offsets, and puf_log on the Logs summed and divided by n as vec_log does.
static int check_log(PolymarketMM* envs, int n) {
    int bad = 0, nf = (int)(sizeof(LOG_FIELDS) / sizeof(LOG_FIELDS[0]));
    bad += nf != PMM_LOG_N;
    for (int i = 0; i < nf && i < PMM_LOG_N; i++) {
        bad += LOG_FIELDS[i].off != i * sizeof(float) || strcmp(LOG_FIELDS[i].name, PMM_LOG_NAMES[i]);
    }
    Log agg;
    memset(&agg, 0, sizeof(agg));
    float* acc = (float*)&agg;
    for (int e = 0; e < n; e++) {
        const float* f = (const float*)&envs[e].log;
        for (int j = 0; j < (int)(sizeof(Log) / sizeof(float)); j++) {
            acc[j] += f[j];
        }
    }
    float cnt = agg.n;
    for (int j = 0; j < (int)(sizeof(Log) / sizeof(float)); j++) {
        acc[j] /= cnt;
    }
    Dict out = {0};
    puf_log(&agg, &out);
    bad += out.size != PMM_LOG_N + 2 * MM_NFLAGS;
    bad += dict_get(&out, "score") != agg.score || dict_get(&out, "pnl_inspread") != agg.pnl_inspread;
    double share = dict_get(&out, "share_inspread");
    bad += agg.fill_vol > 0 && fabs(share - agg.vol_inspread / agg.fill_vol) > 1e-6;
    printf("puf_log: %d Log names at their offsets, %d keys; mean over %g episodes: score %.3f, "
        "fill volume %.1f, share_inspread %.3f%s\n", nf, out.size, cnt, dict_get(&out, "score"),
        agg.fill_vol, share, bad ? " MISMATCH" : "");
    dict_clear(&out);
    return bad;
}

static int same(const void* a, const void* b, size_t n) {
    return memcmp(a, b, n) == 0;
}

// Steps wrapper and core T times on the same random actions; returns mismatching values.
static long run(int n, int steps, PolymarketMM* envs, Bufs w, Mm* cores, Bufs c, Log* core_logs,
        Log* first_logs, unsigned int* rs, long* terminals) {
    long bad = 0;
    int* act = (int*)calloc((size_t)n * MM_NUM_ATNS, sizeof(int));
    int* seen = (int*)calloc(n, sizeof(int));
    bad += !same(w.obs, c.obs, (size_t)n * MM_OBS * sizeof(float));
    for (int t = 0; t < steps; t++) {
        for (int i = 0; i < n; i++) {
            for (int h = 0; h < MM_NUM_ATNS; h++) {
                int a = (int)(rand_r(rs) % (unsigned)SIZES[h]);
                act[i * MM_NUM_ATNS + h] = a;
                w.act[i * MM_NUM_ATNS + h] = (float)a;
            }
        }
        #pragma omp parallel for schedule(static, 8)
        for (int i = 0; i < n; i++) {
            puf_step(&envs[i]);
        }
        for (int i = 0; i < n; i++) {
            mm_step(&cores[i], act + i * MM_NUM_ATNS, c.obs + (size_t)i * MM_OBS, c.rew + i,
                c.term + i, &core_logs[i]);
            if (c.term[i] != 0.0f) {
                (*terminals)++;
                if (!seen[i] && first_logs) {
                    first_logs[i] = core_logs[i];
                }
                seen[i] = 1;
            }
        }
        bad += !same(w.obs, c.obs, (size_t)n * MM_OBS * sizeof(float));
        bad += !same(w.rew, c.rew, (size_t)n * sizeof(float));
        bad += !same(w.term, c.term, (size_t)n * sizeof(float));
    }
    free(act);
    free(seen);
    return bad;
}

// [eval_env] items applied to a copy of [env], as the patched trainer does.
static void apply_eval_env(Ini* ini, Dict* out) {
    dict_copy(out, puf_ini_section(ini, "env", 0));
    Dict* over = puf_ini_section(ini, "eval_env", 0);
    for (int i = 0; i < over->size; i++) {
        if (!dict_find(out, over->items[i].key)) {
            fprintf(stderr, "[eval_env] %s is not an [env] key\n", over->items[i].key);
            exit(1);
        }
        puf_ini_set(out, over->items[i].key, over->items[i].str);
    }
}

// puf_init with split = 2 in a child process: returns its exit status.
static int test_split_init(Dict* kw, const char* allow_test, const char* mm_final) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        if (mm_final) {
            setenv("MM_FINAL", mm_final, 1);
        } else {
            unsetenv("MM_FINAL");
        }
        if (!freopen("/dev/null", "w", stderr)) {
            _exit(3);
        }
        Dict d;
        dict_copy(&d, kw);
        puf_ini_set(&d, "split", "2");
        puf_ini_set(&d, "allow_test", allow_test);
        PolymarketMM env;
        memset(&env, 0, sizeof(env));
        puf_init(&env, &d);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main(int argc, char** argv) {
    Ini ini = {0};
    puf_ini_load_file(&ini, argc > 1 ? argv[1] : "polymarket_mm.ini");
    Dict* kw = puf_ini_section(&ini, "env", 0);
    if (argc > 2) {
        puf_ini_set(kw, "data_dir", argv[2]);
    }
    unsigned int rs = 7;
    int fail = 0;

    // 1. training config
    Bufs w = bufs_alloc(N), c = bufs_alloc(N);
    PolymarketMM* envs = make_envs(N, kw, w);
    Mm* cores = make_cores(N, kw, c.obs);
    Log* core_logs = (Log*)calloc(N, sizeof(Log));
    long terms = 0;
    long bad = run(N, T, envs, w, cores, c, core_logs, NULL, &rs, &terms);
    double n_logged = 0;
    for (int i = 0; i < N; i++) {
        bad += !same(&envs[i].log, &core_logs[i], sizeof(Log));
        n_logged += envs[i].log.n;
    }
    printf("train config: %d envs x %d steps (OpenMP), %ld episode ends, %g logged; "
        "mismatching obs/reward/terminal/Log arrays: %ld\n", N, T, terms, n_logged, bad);
    fail |= bad != 0 || n_logged != (double)terms || terms == 0;
    fail |= check_log(envs, N) != 0;

    // 2. final-evaluation config
    Dict ek;
    apply_eval_env(&ini, &ek);
    PolymarketMM probe;
    memset(&probe, 0, sizeof(probe));
    puf_init(&probe, &ek);
    const int* list = probe.mm.cfg->list;
    int n_val = probe.mm.cfg->n_list;
    if (n_val <= 0 || n_val > 100000) {
        fprintf(stderr, "no validation markets\n");
        return 1;
    }
    int n = K_EVAL * n_val;
    Bufs w2 = bufs_alloc(n), c2 = bufs_alloc(n);
    PolymarketMM* eenvs = make_envs(n, &ek, w2);
    Mm* ecores = make_cores(n, &ek, c2.obs);
    long cover_bad = 0;
    int* count = (int*)calloc(MM_DATA.n_markets, sizeof(int));
    for (int i = 0; i < n; i++) {
        const MmMarket* M = &MM_DATA.markets[eenvs[i].mm.market];
        cover_bad += eenvs[i].mm.market != list[i % n_val] || M->split != 1
            || eenvs[i].mm.s0 != (int)M->row_offset;
        count[eenvs[i].mm.market]++;
    }
    for (int j = 0; j < n_val; j++) {
        cover_bad += count[list[j]] != K_EVAL;
    }
    Log* ecore_logs = (Log*)calloc(n, sizeof(Log));
    Log* first = (Log*)calloc(n, sizeof(Log));
    long eterms = 0;
    long ebad = run(n, T_EVAL, eenvs, w2, ecores, c2, ecore_logs, first, &rs, &eterms);
    long once_bad = 0;
    double score = 0;
    for (int i = 0; i < n; i++) {
        once_bad += eenvs[i].log.n != 1.0f || !same(&eenvs[i].log, &first[i], sizeof(Log));
        score += eenvs[i].log.score;
    }
    printf("eval config ([eval_env]): %d validation markets x %d envs, from the open; "
        "start-market mismatches %ld; %ld episode ends, one logged per env (mismatches %ld); "
        "step mismatches %ld; mean P&L per market under random actions %.3f\n",
        n_val, K_EVAL, cover_bad, eterms, once_bad, ebad, score / n);
    fail |= cover_bad != 0 || once_bad != 0 || ebad != 0 || eterms <= n;

    // 3. test-split guard
    int s1 = test_split_init(kw, "1", NULL);
    int s2 = test_split_init(kw, "0", "1");
    printf("split = 2: refused with allow_test=1 and no MM_FINAL (exit %d), refused with "
        "MM_FINAL=1 and allow_test=0 (exit %d)\n", s1, s2);
    fail |= s1 != 1 || s2 != 1;

    printf(fail ? "FAIL\n" : "ok\n");
    return fail;
}
