/* Test harness for deep_hedging.h, driven by test_c_env.py.
 *
 *     c_env_harness DIR N T INJECT key=value ...
 *
 * Builds N envs the way the trainer does (env.rng = env index, one agent
 * each, buffers in shared arrays), applies the key=value pairs as the [env]
 * kwargs, calls puf_reset and runs T steps on the actions in DIR/actions.bin
 * (float32, T x N x 2). With INJECT = 1 the steps go through hedge_step with
 * the noise in DIR/z.bin (float64, N x episodes x 30) and NAN on the steps
 * that must not use it; with INJECT = 0 they go through puf_step and the
 * env's own generator. Writes, for the reset and after every step,
 * obs.bin (float32, (T + 1) x N x 8) and state.bin (float64, (T + 1) x N x
 * [k, w, phi, loss, premium]), and for every step rewards.bin and
 * terminals.bin (float32, T x N), then log.bin (the Log of each env).
 */

#include <stdio.h>
#include "deep_hedging.h"

#define STATE_SIZE 5

FILE* open_file(const char* dir, const char* name, const char* mode) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* fp = fopen(path, mode);
    if (!fp) {
        perror(path);
        exit(1);
    }
    return fp;
}

void read_file(const char* dir, const char* name, void* data, size_t size) {
    FILE* fp = open_file(dir, name, "rb");
    if (fread(data, 1, size, fp) != size) {
        fprintf(stderr, "short read: %s\n", name);
        exit(1);
    }
    fclose(fp);
}

void write_file(const char* dir, const char* name, void* data, size_t size) {
    FILE* fp = open_file(dir, name, "wb");
    fwrite(data, 1, size, fp);
    fclose(fp);
}

void record(Env* envs, int n, double* state) {
    for (int i = 0; i < n; i++) {
        double* row = state + (size_t)i * STATE_SIZE;
        row[0] = envs[i].k;
        row[1] = envs[i].w;
        row[2] = envs[i].phi;
        row[3] = envs[i].loss;
        row[4] = envs[i].premium;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s DIR N T INJECT key=value ...\n", argv[0]);
        return 1;
    }
    const char* dir = argv[1];
    int n = atoi(argv[2]);
    int T = atoi(argv[3]);
    int inject = atoi(argv[4]);
    int episodes = (T + EPISODE_STEPS - 1) / EPISODE_STEPS;

    Dict kwargs = {0};
    for (int i = 5; i < argc; i++) {
        char key[PUF_DICT_MAX_KEY];
        char* eq = strchr(argv[i], '=');
        if (!eq || eq - argv[i] >= PUF_DICT_MAX_KEY) {
            fprintf(stderr, "bad kwarg: %s\n", argv[i]);
            return 1;
        }
        snprintf(key, sizeof(key), "%.*s", (int)(eq - argv[i]), argv[i]);
        dict_set(&kwargs, key, strtod(eq + 1, NULL));
    }

    float* actions_all = calloc((size_t)T * n * NUM_ATNS, sizeof(float));
    read_file(dir, "actions.bin", actions_all, (size_t)T * n * NUM_ATNS * sizeof(float));
    double* z = NULL;
    if (inject) {
        z = calloc((size_t)n * episodes * N_STEPS, sizeof(double));
        read_file(dir, "z.bin", z, (size_t)n * episodes * N_STEPS * sizeof(double));
    }
    obs_t* obs = calloc((size_t)(T + 1) * n * OBS_SIZE, sizeof(obs_t));
    double* state = calloc((size_t)(T + 1) * n * STATE_SIZE, sizeof(double));
    float* rewards = calloc((size_t)T * n, sizeof(float));
    float* terminals = calloc((size_t)T * n, sizeof(float));
    float* actions = calloc((size_t)n * NUM_ATNS, sizeof(float));
    float* reward_buf = calloc(n, sizeof(float));
    float* terminal_buf = calloc(n, sizeof(float));
    obs_t* obs_buf = calloc((size_t)n * OBS_SIZE, sizeof(obs_t));

    Env* envs = calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        envs[i].rng = i;
        puf_init(&envs[i], &kwargs);
        envs[i].agents[0].observations = obs_buf + (size_t)i * OBS_SIZE;
        envs[i].agents[0].actions = actions + (size_t)i * NUM_ATNS;
        envs[i].agents[0].rewards = reward_buf + i;
        envs[i].agents[0].terminals = terminal_buf + i;
        puf_reset(&envs[i]);
    }
    memcpy(obs, obs_buf, (size_t)n * OBS_SIZE * sizeof(obs_t));
    record(envs, n, state);

    for (int t = 0; t < T; t++) {
        memcpy(actions, actions_all + (size_t)t * n * NUM_ATNS,
            (size_t)n * NUM_ATNS * sizeof(float));
        // The trainer zeroes these before every step.
        memset(reward_buf, 0, n * sizeof(float));
        memset(terminal_buf, 0, n * sizeof(float));
        for (int i = 0; i < n; i++) {
            if (!inject) {
                puf_step(&envs[i]);
                continue;
            }
            int k = envs[i].k;
            int e = t / EPISODE_STEPS;
            hedge_step(&envs[i], k < N_STEPS ? z[((size_t)i * episodes + e) * N_STEPS + k] : NAN);
        }
        memcpy(obs + (size_t)(t + 1) * n * OBS_SIZE, obs_buf,
            (size_t)n * OBS_SIZE * sizeof(obs_t));
        memcpy(rewards + (size_t)t * n, reward_buf, n * sizeof(float));
        memcpy(terminals + (size_t)t * n, terminal_buf, n * sizeof(float));
        record(envs, n, state + (size_t)(t + 1) * n * STATE_SIZE);
    }

    Log* logs = calloc(n, sizeof(Log));
    for (int i = 0; i < n; i++) {
        logs[i] = envs[i].log;
        puf_close(&envs[i]);
    }
    write_file(dir, "obs.bin", obs, (size_t)(T + 1) * n * OBS_SIZE * sizeof(obs_t));
    write_file(dir, "state.bin", state, (size_t)(T + 1) * n * STATE_SIZE * sizeof(double));
    write_file(dir, "rewards.bin", rewards, (size_t)T * n * sizeof(float));
    write_file(dir, "terminals.bin", terminals, (size_t)T * n * sizeof(float));
    write_file(dir, "log.bin", logs, (size_t)n * sizeof(Log));
    dict_clear(&kwargs);
    return 0;
}
