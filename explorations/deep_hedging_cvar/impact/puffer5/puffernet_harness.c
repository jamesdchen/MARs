// Runs PufferLib 5.0's C PufferNet (src/puffercpu.c, compiled as a library,
// i.e. without PUFFERCPU_EVAL_MAIN) on inputs written by test_puffernet.py.
//
//   puffernet_harness weights.bin obs.f32 terminals.f32 out.f32
//       B T obs_size hidden layers num_actions
//
// obs.f32 is (T, B, obs_size) and terminals.f32 is (T, B), float32. Each
// step calls forward_puffernet, which zeroes the state of terminal rows and
// runs the net. out.f32 receives the decoder outputs (T, B, num_actions + 1),
// whose last column is the value, then the actions (T, B, num_actions).
//
// Build: cc -O2 -I<PufferLib>/src puffernet_harness.c -o puffernet_harness -lm
#include "puffercpu.c"

static float* read_f32(const char* path, size_t n) {
    float* buf = malloc(n * sizeof(float));
    FILE* f = fopen(path, "rb");
    if (!f || fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "cannot read %zu floats from %s\n", n, path);
        exit(1);
    }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc != 11) {
        fprintf(stderr, "usage: %s weights.bin obs.f32 terminals.f32 out.f32 "
            "B T obs_size hidden layers num_actions\n", argv[0]);
        return 2;
    }
    int B = atoi(argv[5]), T = atoi(argv[6]), obs_size = atoi(argv[7]);
    int hidden = atoi(argv[8]), layers = atoi(argv[9]), A = atoi(argv[10]);

    Weights* weights = load_weights(argv[1]);
    if (!weights) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    int file_floats = weights->size - 7;
    int act_sizes[64];
    for (int i = 0; i < A; i++) {
        act_sizes[i] = 1;
    }
    PufferNet* net = make_puffernet(weights, B, obs_size, hidden, layers, act_sizes, A);
    int need = weights->idx;
    printf("file_floats=%d need=%d\n", file_floats, need);
    // Same check as the PUFFERCPU_EVAL_MAIN loader (puffernet_weight_count).
    if (!(need - file_floats <= 7 && file_floats <= need)) {
        fprintf(stderr, "weight count mismatch\n");
        return 1;
    }

    float* obs = read_f32(argv[2], (size_t)T * B * obs_size);
    float* term = read_f32(argv[3], (size_t)T * B);
    float* dec = malloc((size_t)T * B * (A + 1) * sizeof(float));
    float* act = malloc((size_t)T * B * A * sizeof(float));
    for (int t = 0; t < T; t++) {
        forward_puffernet(net, obs + (size_t)t * B * obs_size,
            act + (size_t)t * B * A, NULL, term + (size_t)t * B);
        memcpy(dec + (size_t)t * B * (A + 1), net->decoder->output,
            (size_t)B * (A + 1) * sizeof(float));
    }

    FILE* out = fopen(argv[4], "wb");
    if (!out) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        return 1;
    }
    fwrite(dec, sizeof(float), (size_t)T * B * (A + 1), out);
    fwrite(act, sizeof(float), (size_t)T * B * A, out);
    fclose(out);

    free_puffernet(net);
    free(weights);
    free(obs);
    free(term);
    free(dec);
    free(act);
    return 0;
}
