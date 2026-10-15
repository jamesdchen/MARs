// Runs PufferLib 5.0's C PufferNet (src/puffercpu.c, as a library, without
// PUFFERCPU_EVAL_MAIN) with discrete heads on inputs from test_puffernet_mm.py.
//
//   puffernet_mm_harness weights.bin obs.f32 terminals.f32 out.f32
//       B T obs_size hidden layers act_size...
//
// obs.f32 is (T, B, obs_size), terminals.f32 (T, B), float32. Each step calls
// forward_puffernet (zeroes the state of terminal rows, runs the net; its
// sampled actions are discarded), then takes the argmax of each head with
// _multidiscrete(hard=1). out.f32 receives the decoder outputs
// (T, B, sum(act_sizes) + 1), last column the value, then the argmax actions
// (T, B, heads). Prints the offset of each tensor in the file (encoder,
// decoder, MinGRU layers), the file length and the length make_puffernet reads.
//
// Build: cc -O2 -I<PufferLib>/src puffernet_mm_harness.c -o puffernet_mm_harness -lm
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
    if (argc < 11) {
        fprintf(stderr, "usage: %s weights.bin obs.f32 terminals.f32 out.f32 "
            "B T obs_size hidden layers act_size...\n", argv[0]);
        return 2;
    }
    int B = atoi(argv[5]), T = atoi(argv[6]), obs_size = atoi(argv[7]);
    int hidden = atoi(argv[8]), layers = atoi(argv[9]);
    int heads = argc - 10;
    int act_sizes[32];
    int atn_sum = 0;
    if (heads > 32) {
        fprintf(stderr, "at most 32 heads\n");
        return 2;
    }
    for (int i = 0; i < heads; i++) {
        act_sizes[i] = atoi(argv[10 + i]);
        if (act_sizes[i] < 2) {
            fprintf(stderr, "discrete heads need size >= 2\n");
            return 2;
        }
        atn_sum += act_sizes[i];
    }

    Weights* weights = load_weights(argv[1]);
    if (!weights) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    int file_floats = weights->size - 7;
    PufferNet* net = make_puffernet(weights, B, obs_size, hidden, layers, act_sizes, heads);
    int need = weights->idx;
    float* base = weights->data;
    printf("offsets=%td,%td", net->encoder->weights - base, net->decoder->weights - base);
    for (int l = 0; l < layers; l++) {
        printf(",%td", net->mingru->proj[l]->weights - base);
    }
    printf(" file_floats=%d need=%d\n", file_floats, need);
    if (!(need - file_floats <= 7 && file_floats <= need)) {
        fprintf(stderr, "weight count mismatch\n");
        return 1;
    }

    int D = atn_sum + 1;
    float* obs = read_f32(argv[2], (size_t)T * B * obs_size);
    float* term = read_f32(argv[3], (size_t)T * B);
    float* dec = malloc((size_t)T * B * D * sizeof(float));
    float* act = malloc((size_t)T * B * heads * sizeof(float));
    float* sampled = malloc((size_t)B * heads * sizeof(float));
    for (int t = 0; t < T; t++) {
        forward_puffernet(net, obs + (size_t)t * B * obs_size, sampled, NULL,
            term + (size_t)t * B);
        memcpy(dec + (size_t)t * B * D, net->decoder->output, (size_t)B * D * sizeof(float));
        _multidiscrete(net->decoder->output, act + (size_t)t * B * heads, B, act_sizes,
            heads, 1, NULL);
    }

    FILE* out = fopen(argv[4], "wb");
    if (!out) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        return 1;
    }
    fwrite(dec, sizeof(float), (size_t)T * B * D, out);
    fwrite(act, sizeof(float), (size_t)T * B * heads, out);
    fclose(out);

    free_puffernet(net);
    free(weights);
    free(obs);
    free(term);
    free(dec);
    free(act);
    free(sampled);
    return 0;
}
