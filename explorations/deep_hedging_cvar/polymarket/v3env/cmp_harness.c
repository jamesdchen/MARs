/* v2 (../mm.h, frozen) and v3 (mm3.h) in one translation unit, stepped side by side on the same
 * parameters, seeds, markets, offsets and actions. Used by test_env3.py (bit-exactness with the
 * v3 options off). Nothing here changes either core.
 *
 * cmp_run compares after every step: observations 0..49 (bitwise), rewards (bitwise), terminal
 * flags, the double state (inventory, cash, rebates, rewards, wealth, penalty, order state) and
 * the Log (v2's 42 float fields, bitwise). At every terminal it also computes both episode P&Ls in
 * double (cash + rebates + rewards + inventory * settlement), by running the terminal step on a
 * copy of each env up to the wealth computation, and stores them in pnl2 / pnl3. */

#include "../mm.h"
#include "mm3.h"

int cmp_open(const char* dir, char* err, int n) {
    if (mm_data_open(dir, err, n)) return -1;
    return m3_data_open(dir, err, n);
}

// The terminal step of v2's mm_step up to the wealth at settlement, on a copy (no side effects).
static double v2_terminal_pnl(Mm m, const int* act) {
    int t = m.t, bid, ask;
    mm_targets(&m, t, act, &bid, &ask);
    int size = act[2] < 0 ? 0 : (act[2] > 2 ? 2 : act[2]);
    MmPend* e = &m.pend[(m.ph + m.pn) & (MM_PEND - 1)];
    e->row = t + m.cfg->L;
    e->bid = (int16_t)(bid | (size << 11));
    e->ask = (int16_t)ask;
    m.pn++;
    int tn = t + m.dt < m.r_end ? t + m.dt : m.r_end;
    mm_rows(&m, t, tn);
    return m.cash + m.rebates + m.liq + m.inv * (double)m.winner;
}

// The same for m3_step (pull_s, dt_far and close shaping are off in the comparison).
static double v3_terminal_pnl(Mm3 m, const int* act) {
    int t = m.t, bid, ask;
    m3_targets(&m, t, act, &bid, &ask);
    int size = act[2] < 0 ? 0 : (act[2] > 2 ? 2 : act[2]);
    Mm3Pend* e = &m.pend[(m.ph + m.pn) & (M3_PEND - 1)];
    e->row = t + m.cfg->L;
    e->bid = (int16_t)(bid | (size << 11));
    e->ask = (int16_t)ask;
    m.pn++;
    int tn = t + m.dt < m.r_end ? t + m.dt : m.r_end;
    m3_rows(&m, t, tn);
    double w = m.cash + m.rebates + m.liq + m.inv * m.settle;
    if (m.flat > 0.0) w -= fabs(m.inv) * m.flat;
    return w;
}

static double rdiff(double a, double b) {
    double d = fabs(a - b);
    return d != d ? 1e300 : d;
}

// stats: [0] max |obs diff| (0..49), [1] obs floats not bit-identical, [2] max |reward diff|,
// [3] terminal mismatches, [4] max |Log diff| (42 fields), [5] Log floats not bit-identical,
// [6] max |state diff|, [7] steps, [8] terminals, [9] max |pnl2 - pnl3|.
int cmp_run(const double* p3, int n, int T, const int* markets, const int* starts, const int* actions,
        uint64_t seed, double* stats, double* pnl2, double* pnl3, int max_eps, int* n_eps, char* err, int errlen) {
    const Mm3Params* P3 = (const Mm3Params*)p3;
    MmParams P2;
    memcpy(&P2, p3, sizeof(MmParams));        // v3's first 27 parameters are v2's
    P2.len_mask = (double)((int)P2.len_mask & 7);   // v3's bit 8 (other lengths) does not exist in v2
    int W = (int)P3->obs_size;
    Mm* a = (Mm*)calloc(n, sizeof(Mm));
    Mm3* b = (Mm3*)calloc(n, sizeof(Mm3));
    Log* la = (Log*)calloc(n, sizeof(Log));
    Mm3Log* lb = (Mm3Log*)calloc(n, sizeof(Mm3Log));
    float* oa = (float*)calloc(MM_OBS, sizeof(float));
    float* ob = (float*)calloc(W, sizeof(float));
    int rc = 0;
    for (int k = 0; k < 10; k++) stats[k] = 0.0;
    *n_eps = 0;
    for (int i = 0; i < n; i++) {
        if (mm_init(&a[i], &P2, seed + i, err, errlen) || m3_init(&b[i], P3, seed + i, err, errlen)) {
            rc = 1;
            goto out;
        }
        if (markets[i] >= 0) {
            mm_begin(&a[i], markets[i], starts[i]);
            m3_begin(&b[i], markets[i], starts[i]);
        } else {
            mm_new_episode(&a[i]);
            m3_new_episode(&b[i]);
        }
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            const int* act = actions + ((size_t)t * n + i) * 3;
            Mm* x = &a[i];
            Mm3* y = &b[i];
            if (x->t + x->dt >= x->r_end && *n_eps < max_eps) {       // terminal step: P&L in double
                pnl2[*n_eps] = v2_terminal_pnl(*x, act);
                pnl3[*n_eps] = v3_terminal_pnl(*y, act);
                double d = rdiff(pnl2[*n_eps], pnl3[*n_eps]);
                if (d > stats[9]) stats[9] = d;
                (*n_eps)++;
            }
            float ra, rb, ta, tb;
            mm_step(x, act, oa, &ra, &ta, &la[i]);
            m3_step(y, act, ob, &rb, &tb, &lb[i]);
            for (int k = 0; k < MM_OBS; k++) {
                double d = rdiff(oa[k], ob[k]);
                if (d > stats[0]) stats[0] = d;
                if (memcmp(&oa[k], &ob[k], 4) != 0) stats[1] += 1;
            }
            double dr = rdiff(ra, rb);
            if (dr > stats[2]) stats[2] = dr;
            if (memcmp(&ra, &rb, 4) != 0 && dr == 0.0) stats[2] = stats[2] > 1e-300 ? stats[2] : 1e-300;
            if (ta != tb) stats[3] += 1;
            const float* fa = (const float*)&la[i];
            const float* fb = (const float*)&lb[i];
            for (int k = 0; k < 42; k++) {
                double d = rdiff(fa[k], fb[k]);
                if (d > stats[4]) stats[4] = d;
                if (memcmp(&fa[k], &fb[k], 4) != 0) stats[5] += 1;
            }
            if (fa[42] != fb[49]) stats[5] += 1;     // n
            double sa[16] = {x->inv, x->cash, x->rebates, x->liq, x->wealth, x->penalty, x->ord[0].rem, x->ord[0].q,
                             x->ord[1].rem, x->ord[1].q, (double)x->ord[0].px, (double)x->ord[1].px,
                             (double)x->ord[0].live, (double)x->ord[1].live, (double)x->t, (double)x->market};
            double sb[16] = {y->inv, y->cash, y->rebates, y->liq, y->wealth, y->penalty, y->ord[0].rem, y->ord[0].q,
                             y->ord[1].rem, y->ord[1].q, (double)y->ord[0].px, (double)y->ord[1].px,
                             (double)y->ord[0].live, (double)y->ord[1].live, (double)y->t, (double)y->market};
            for (int k = 0; k < 16; k++) {
                double d = rdiff(sa[k], sb[k]);
                if (d > stats[6]) stats[6] = d;
            }
            stats[7] += 1;
            stats[8] += ta != 0.0f;
        }
    }
out:
    free(a); free(b); free(la); free(lb); free(oa); free(ob);
    return rc;
}
