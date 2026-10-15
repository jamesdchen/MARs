/* mm.h: Polymarket crypto up/down market-making replay env, pure C core.
 *
 * One agent quotes a bid and an ask in one market per episode (Up-token frame; Up and Down
 * share one book). The core has no PufferLib includes. It compiles as C99 and as C++17, as
 * PUFFER5.md asks. The PufferLib 3.0 binding is binding.c, and test_env.py checks this core
 * against the numpy reference in reference.py. ENV.md documents the model.
 *
 * Data: all market-only work is offline (data/features.py). mm_data_open mmaps five files
 * from one directory once, behind a static guard, and every env shares them:
 *   markets.bin (88 B/market), feat.bin (float32 [N, 44]), book.bin (64 B/row: top-5 ladder in
 *   integer mils), fills.bin (20 B per maker leg, sorted by row) and fill_index.bin (int32 [N+1]).
 * A step reads a few cache lines of these and allocates nothing.
 *
 * Time. Row r = unix second r of a market. A decision at row t sees row t. Its quote becomes
 * live after the latency, at t + L. Trades printed in row u (block time u) carry a lag: they
 * matched off-chain about d seconds earlier, so they are matched against our quotes as they
 * stood at match row tau = u - D, with D = ceil(d + 0.5) (the mid-second convention). The
 * order book state is kept in match time: pending decisions activate when tau reaches t + L.
 */

#ifndef MM_H
#define MM_H

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MM_OBS 48
#define MM_NUM_ATNS 3
#define MM_ACT_SIZES {7, 7, 3}
#define MM_G 44            // feat.bin columns
#define MM_FEAT_OBS 33     // feat columns 0..32 are observed
#define MM_F_P 33          // p_model (fair value of Up)
#define MM_F_MARK 34       // book mid, p_model fallback when one-sided
#define MM_F_DUSD 35       // dp/dlnS, USD of underlying per Up share
#define MM_F_LIQ_MID 38
#define MM_F_LIQ_COMP 39
#define MM_F_LIQ_PI 40
#define MM_PEND 16         // pending-decision FIFO (needs latency + D + 2 <= 16)
#define MM_LIQ_V 1.5       // liquidity-reward max spread (cents) baked into liq_comp
#define MM_LIQ_MIN 50.0    // liquidity-reward min size baked into liq_comp
#define MM_STALE_S 10      // book.bin STALE flag threshold (documentation)

enum { MM_OPT = 0, MM_CENTRAL = 1, MM_PESS = 2 };
enum { MM_BOOK_VALID = 1, MM_BOOK_CROSSED = 2, MM_BOOK_STALE = 4, MM_BOOK_SAME = 16 };
// Fill-flag slots of the exploit detectors (EVALUATION.md T5): volume filled while ...
enum { MM_FL_CROSSED, MM_FL_STALE, MM_FL_ONESIDED, MM_FL_LAST30, MM_FL_LAST10, MM_FL_INSIDE2,
       MM_FL_EXTREME, MM_FL_OUTSIDE, MM_NFLAGS };

typedef struct {
    int64_t row_offset, open_ts, end_ts;
    int32_t n_rows, contract_len_s, coin_id, winner_up, split, flags;
    float tick_size, taker_fee_rate, fee_exponent, min_order_size;
    float rewards_min_size, rewards_max_spread, log_strike_vs_ul_open, maker_rebate_rate;
    int32_t market_idx;
    float tick_size_final;
} MmMarket;                                      // 88 bytes, data/README.md

typedef struct {
    int16_t bid_px[5];                           // mils; 0 = empty
    int16_t ask_px[5];                           // mils; 1000 = empty
    float bid_sz[5];
    float ask_sz[5];
    uint8_t flags;                               // MM_BOOK_*
    uint8_t tick;                                // tick in mils (10 or 1), data-observed rule
    uint8_t tick_alt;                            // tick under tick_rule 1 (price rule)
    uint8_t pad;
} MmBook;                                        // 64 bytes

typedef struct {
    int32_t row;                                 // global row = block timestamp
    int8_t side;                                 // +1 taker bought Up (hits asks), -1 sold (hits bids)
    uint8_t first;                               // first leg of a taker tx
    int16_t px;                                  // maker resting price, mils, Up frame
    float price, size, tx_size;
} MmFill;                                        // 20 bytes

// Floats only, n last: PufferLib sums Logs over envs and divides by n.
typedef struct Log Log;
struct Log {
    float score;              // episode P&L in USD incl. rebates and rewards (sweep metric)
    float perf;               // 1 if the episode P&L > 0
    float episode_return;
    float episode_length;     // decisions
    float pnl;                // = score
    float pnl_trading;        // cash + inventory * settlement (no rebates / rewards)
    float spread_capture;     // sum sq * (mid at match - price)
    float markout_5;          // sum sq * (mid(+5 s) - mid), settlement past the end
    float markout_30;
    float markout_60;
    float inventory_pnl;      // pnl_trading - spread_capture - markout_60
    float rebates;
    float liq_rewards;
    float inv_penalty;        // USD of shaping penalty (not in pnl)
    float fills;
    float fill_vol;
    float buy_vol;
    float sell_vol;
    float max_abs_inv;
    float final_abs_inv;
    float uptime_two_sided;   // fraction of match seconds with both quotes live
    float uptime_any;
    float vol_crossed;        // fill volume while the book was crossed / forward-filled
    float vol_stale;          // ... book older than 10 s
    float vol_onesided;
    float vol_last30;         // ... in the last 30 s of the window
    float vol_last10;
    float vol_inside2;        // ... with our quote > 2 ticks inside the touch
    float vol_extreme;        // ... with our quote <= 0.02 or >= 0.98
    float vol_outside;        // ... with the print outside the snapshot's [bid, ask]
    float reward_clip_frac;
    float rejects;            // quotes rejected at arrival (post-only would cross)
    float n;
};

// Parameters: doubles in this order (binding kwargs, harness arrays). ENV.md has the table.
typedef struct {
    double split, allow_test, fill_model, latency, match_delay, dt, dt_5m, dt_15m, dt_4h;
    double Q, inv_penalty, reward_scale, size_mult, part_cap, fee_rate, rebate_rate;
    double liq_rewards, liq_moas, mark_mode, reward_mode, tick_rule, len_mask;
    double fixed_market, fixed_start, cycle_stride, start_frac;
} MmParams;

static const char* MM_PARAM_NAMES[] = {
    "split", "allow_test", "fill_model", "latency", "match_delay", "dt", "dt_5m", "dt_15m", "dt_4h",
    "Q", "inv_penalty", "reward_scale", "size_mult", "part_cap", "fee_rate", "rebate_rate",
    "liq_rewards", "liq_moas", "mark_mode", "reward_mode", "tick_rule", "len_mask",
    "fixed_market", "fixed_start", "cycle_stride", "start_frac",
};
#define MM_NUM_PARAMS (int)(sizeof(MM_PARAM_NAMES) / sizeof(MM_PARAM_NAMES[0]))

static const double MM_DEFAULTS[] = {
    0, 0, 1, 1, -1, 0, 1, 1, 5,
    100, 0.001, 0, 1, -1, -1, -1,
    0, 1, 0, 0, 0, 7,
    -1, -1, 0, 0.2,
};

// ------------------------------------------------------------------ shared data

typedef struct {
    const MmMarket* markets;
    const float* feat;
    const MmBook* book;
    const MmFill* fills;
    const int32_t* fill_off;
    int64_t N, n_fills;
    int n_markets;
    int* lists[4][8];       // market indices per split and length mask (bit 1 5m, 2 15m, 4 4h)
    int n_list[4][8];
    char dir[1024];
} MmData;

static MmData MM_DATA;
static int MM_DATA_READY = 0;

// log1p lookup tables for the observation (filled by mm_data_open).
#define MM_SINCE_MAX 3600
#define MM_NFILL_TAB 1024
static float MM_LOG_SINCE[MM_SINCE_MAX + 1];
static float MM_LOG_FILLS[MM_NFILL_TAB + 1];

static int mm_map(const char* dir, const char* name, size_t* size, const void** out, char* err, int n) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(err, n, "mm: cannot open %s (run data/build.py and data/features.py)", path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        snprintf(err, n, "mm: cannot stat %s", path);
        return -1;
    }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        snprintf(err, n, "mm: mmap failed for %s", path);
        return -1;
    }
    *size = (size_t)st.st_size;
    *out = p;
    return 0;
}

static inline int mm_len_bit(int len) {
    return len == 300 ? 1 : (len == 900 ? 2 : (len == 14400 ? 4 : 0));
}

// Maps the data once per process (static guard); later calls only check the directory.
static int mm_data_open(const char* dir, char* err, int n) {
    if (MM_DATA_READY) {
        if (strcmp(dir, MM_DATA.dir) != 0) {
            snprintf(err, n, "mm: data already loaded from %s, not %s", MM_DATA.dir, dir);
            return -1;
        }
        return 0;
    }
    MmData d;
    memset(&d, 0, sizeof(d));
    size_t s_mk, s_ft, s_bk, s_fl, s_fo;
    const void *p_mk, *p_ft, *p_bk, *p_fl, *p_fo;
    if (mm_map(dir, "markets.bin", &s_mk, &p_mk, err, n) || mm_map(dir, "feat.bin", &s_ft, &p_ft, err, n)
            || mm_map(dir, "book.bin", &s_bk, &p_bk, err, n) || mm_map(dir, "fills.bin", &s_fl, &p_fl, err, n)
            || mm_map(dir, "fill_index.bin", &s_fo, &p_fo, err, n)) {
        return -1;
    }
    if (s_mk % sizeof(MmMarket) != 0) {
        snprintf(err, n, "mm: markets.bin size %zu is not a multiple of %zu", s_mk, sizeof(MmMarket));
        return -1;
    }
    d.markets = (const MmMarket*)p_mk;
    d.n_markets = (int)(s_mk / sizeof(MmMarket));
    const MmMarket* last = &d.markets[d.n_markets - 1];
    d.N = last->row_offset + last->n_rows;
    if (s_ft != (size_t)d.N * MM_G * sizeof(float)) {
        snprintf(err, n, "mm: feat.bin has %zu bytes, expected N=%lld x G=%d floats (rerun data/features.py)",
            s_ft, (long long)d.N, MM_G);
        return -1;
    }
    if (s_bk != (size_t)d.N * sizeof(MmBook) || s_fo != (size_t)(d.N + 1) * sizeof(int32_t)
            || s_fl % sizeof(MmFill) != 0) {
        snprintf(err, n, "mm: book.bin / fill_index.bin / fills.bin sizes do not match N=%lld", (long long)d.N);
        return -1;
    }
    d.feat = (const float*)p_ft;
    d.book = (const MmBook*)p_bk;
    d.fills = (const MmFill*)p_fl;
    d.fill_off = (const int32_t*)p_fo;
    d.n_fills = (int64_t)(s_fl / sizeof(MmFill));
    if (d.fill_off[0] != 0 || d.fill_off[d.N] != d.n_fills) {
        snprintf(err, n, "mm: fill_index.bin does not cover fills.bin");
        return -1;
    }
    for (int s = 0; s < 4; s++) {
        for (int mask = 1; mask < 8; mask++) {
            int c = 0;
            for (int i = 0; i < d.n_markets; i++) {
                c += d.markets[i].split == s && (mm_len_bit(d.markets[i].contract_len_s) & mask);
            }
            d.lists[s][mask] = (int*)malloc((size_t)(c > 0 ? c : 1) * sizeof(int));
            d.n_list[s][mask] = 0;
            for (int i = 0; i < d.n_markets; i++) {
                if (d.markets[i].split == s && (mm_len_bit(d.markets[i].contract_len_s) & mask)) {
                    d.lists[s][mask][d.n_list[s][mask]++] = i;
                }
            }
        }
    }
    snprintf(d.dir, sizeof(d.dir), "%s", dir);
    for (int i = 0; i <= MM_SINCE_MAX; i++) MM_LOG_SINCE[i] = (float)(log1p((double)i) / log(MM_SINCE_MAX + 1.0));
    for (int i = 0; i <= MM_NFILL_TAB; i++) MM_LOG_FILLS[i] = (float)(log1p((double)i) / 5.0);
    MM_DATA = d;
    MM_DATA_READY = 1;
    return 0;
}

// ------------------------------------------------------------------ env state

typedef struct {
    double rem;      // shares left
    double q;        // estimated queue ahead of us (shares)
    double disp;     // displayed size at our level at the last match row (-1 unknown)
    int px;          // mils
    int live;
    int born;        // match row the order became live (for the reward min-age rule)
    int pad;
} MmOrder;           // 32 bytes

typedef struct {
    int32_t row;     // activation (match) row = decision row + latency
    int16_t bid;     // mils (0 = off) | size head << 11
    int16_t ask;     // mils, 0 = off
} MmPend;            // 8 bytes

typedef struct {
    int split, model, L, D, dt_len[3], mark_mode, reward_mode, tick_rule, liq_on, cycle_stride;
    int fixed_market, fixed_start, n_list;
    const int* list;
    double Q, inv_penalty, reward_scale, size_mult, part_cap, fee_rate, rebate_rate, liq_moas,
        start_frac;
} MmCfg;

// Per-env state, about 0.5 kB; the fields every step touches come first.
typedef struct {
    const MmCfg* cfg;                  // interned, shared by envs with identical parameters
    int t, dt, r0, r_end, next_min, ph, pn, winner;
    int last_fill_row, market, s0, open_mod;
    double inv, cash, rebates, liq, wealth, penalty;
    double minute_vol, minute_fill, tx_fill;
    MmOrder ord[2];                    // 0 bid, 1 ask
    float rows, rows_two, rows_any, steps, clips, ret, n_fills, rejects;
    MmPend pend[MM_PEND];
    double len_s, min_size, fee_rate, rebate_rate, fee_exp, reb_obs;
    double spread, mo5, mo30, mo60;
    float fill_vol, buy_vol, sell_vol, max_inv;
    float vol_flag[MM_NFLAGS];
    uint64_t rng, cycle_pos;
} Mm;

// Config registry: written only by init (serial in PufferLib 3.0 and 5.0), read by step.
#define MM_MAX_CFG 256
static MmCfg MM_CFGS[MM_MAX_CFG];
static int MM_NCFG = 0;

static const MmCfg* mm_intern(const MmCfg* c) {
    for (int i = 0; i < MM_NCFG; i++) {
        if (memcmp(&MM_CFGS[i], c, sizeof(MmCfg)) == 0) return &MM_CFGS[i];
    }
    if (MM_NCFG == MM_MAX_CFG) return NULL;
    memcpy(&MM_CFGS[MM_NCFG], c, sizeof(MmCfg));
    return &MM_CFGS[MM_NCFG++];
}


static const int MM_OFFS[7] = {0, 2, 1, 0, -1, -2, -4};   // ticks better than the touch
static const double MM_SIZES[3] = {1.0, 2.0, 5.0};        // x min_order_size

// splitmix64 (Steele, Lea & Flood 2014)
static inline uint64_t mm_rand(Mm* m) {
    uint64_t z = (m->rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline const float* mm_feat(int64_t row) {
    return MM_DATA.feat + row * MM_G;
}

static inline double mm_clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Mark used for the reward and the wealth observation (mark_mode 0 mid / 1 fair value / 2 blend).
static inline double mm_mark(const Mm* m, int row) {
    const float* f = mm_feat(row);
    double mid = f[MM_F_MARK], p = f[MM_F_P];
    return m->cfg->mark_mode == 0 ? mid : (m->cfg->mark_mode == 1 ? p : 0.5 * (mid + p));
}

// Displayed size at our price on side (0 bid, 1 ask): 0 for an empty level inside the shown
// ladder, -1 if the price lies beyond the 5 shown levels (unknown).
static inline double mm_level(const MmBook* B, int side, int px) {
    if (side == 0) {
        for (int k = 0; k < 5; k++) {
            int p = B->bid_px[k];
            if (p == px) return B->bid_sz[k];
            if (p < px) return 0.0;
        }
    } else {
        for (int k = 0; k < 5; k++) {
            int p = B->ask_px[k];
            if (p == px) return B->ask_sz[k];
            if (p > px) return 0.0;
        }
    }
    return -1.0;
}

static inline int mm_tick(const Mm* m, const MmBook* B) {
    return m->cfg->tick_rule ? B->tick_alt : B->tick;
}

// ------------------------------------------------------------------ config

static int mm_config(MmCfg* c, const MmParams* p, char* err, int n) {
    memset(c, 0, sizeof(*c));
    c->split = (int)p->split;
    c->model = (int)p->fill_model;
    c->L = (int)p->latency;
    if (c->split < 0 || c->split > 2) {
        snprintf(err, n, "mm: split must be 0 (train), 1 (val) or 2 (test)");
        return -1;
    }
    if (c->split == 2) {
        const char* f = getenv("MM_FINAL");
        if (p->allow_test != 1.0 || f == NULL || strcmp(f, "1") != 0) {
            snprintf(err, n, "mm: split=2 (TEST) refused: needs allow_test=1 AND env MM_FINAL=1 "
                "(EVALUATION.md section 5, run once)");
            return -1;
        }
        static int warned = 0;    // init is serial in both trainers
        if (!warned) {
            fprintf(stderr, "\n**************************************************************\n"
                "*** mm: TEST SPLIT (Aug 24) IN USE: allow_test=1, MM_FINAL=1 ***\n"
                "*** This must be the single pre-registered test run.       ***\n"
                "**************************************************************\n\n");
            warned = 1;
        }
    }
    if (c->model < MM_OPT || c->model > MM_PESS || c->L < 0 || c->L > 5) {
        snprintf(err, n, "mm: fill_model must be 0/1/2 and latency in 0..5");
        return -1;
    }
    static const double delay[3] = {0.0, 3.5, 5.0};       // EVALUATION.md T1
    double d = p->match_delay < 0 ? delay[c->model] : p->match_delay;
    c->D = (int)ceil(d + 0.5);
    if (c->L + c->D + 2 > MM_PEND) {
        snprintf(err, n, "mm: latency + match delay too large for the pending FIFO");
        return -1;
    }
    int dts[3] = {(int)p->dt_5m, (int)p->dt_15m, (int)p->dt_4h};
    for (int i = 0; i < 3; i++) {
        c->dt_len[i] = p->dt > 0 ? (int)p->dt : dts[i];
        if (c->dt_len[i] < 1) {
            snprintf(err, n, "mm: dt must be >= 1");
            return -1;
        }
    }
    c->Q = p->Q;
    c->inv_penalty = p->inv_penalty;
    c->reward_scale = p->reward_scale > 0 ? p->reward_scale : 1.0 / p->Q;
    c->size_mult = p->size_mult;
    c->part_cap = p->part_cap >= 0 ? p->part_cap : (c->model == MM_OPT ? 0.0 : 0.25);
    c->fee_rate = p->fee_rate;
    c->rebate_rate = p->rebate_rate;
    c->liq_on = p->liq_rewards != 0.0;
    c->liq_moas = p->liq_moas;
    c->mark_mode = (int)p->mark_mode;
    c->reward_mode = (int)p->reward_mode;
    c->tick_rule = (int)p->tick_rule;
    c->fixed_market = (int)p->fixed_market;
    c->fixed_start = (int)p->fixed_start;
    c->cycle_stride = (int)p->cycle_stride;
    c->start_frac = p->start_frac;
    int mask = (int)p->len_mask;
    if (c->Q <= 0 || c->size_mult <= 0 || mask < 1 || mask > 7 || c->mark_mode < 0 || c->mark_mode > 2
            || c->reward_mode < 0 || c->reward_mode > 1 || c->tick_rule < 0 || c->tick_rule > 1
            || c->start_frac < 0 || c->start_frac >= 1) {
        snprintf(err, n, "mm: bad Q / size_mult / len_mask / mark_mode / reward_mode / tick_rule / start_frac");
        return -1;
    }
    c->list = MM_DATA.lists[c->split][mask];
    c->n_list = MM_DATA.n_list[c->split][mask];
    if (c->n_list == 0) {
        snprintf(err, n, "mm: no market in split %d with len_mask %d", c->split, mask);
        return -1;
    }
    if (c->fixed_market >= 0 && (c->fixed_market >= MM_DATA.n_markets
            || MM_DATA.markets[c->fixed_market].split != c->split)) {
        snprintf(err, n, "mm: fixed_market %d is not a market of split %d", c->fixed_market, c->split);
        return -1;
    }
    return 0;
}

// ------------------------------------------------------------------ episodes

// Starts an episode on global market index mk at start offset off (seconds after open).
static inline void mm_begin(Mm* m, int mk, int off) {
    const MmMarket* M = &MM_DATA.markets[mk];
    m->market = mk;
    m->r0 = (int)M->row_offset;
    m->r_end = m->r0 + M->n_rows - 1;
    if (off > M->n_rows - 2) off = M->n_rows - 2;
    if (off < 0) off = 0;
    m->s0 = m->t = m->r0 + off;
    int len = M->contract_len_s;
    m->dt = m->cfg->dt_len[len == 300 ? 0 : (len == 900 ? 1 : 2)];
    m->open_mod = (int)(M->open_ts % 60);
    m->winner = M->winner_up;
    m->len_s = len;
    m->min_size = M->min_order_size;
    m->fee_rate = m->cfg->fee_rate >= 0 ? m->cfg->fee_rate : M->taker_fee_rate;
    m->rebate_rate = m->cfg->rebate_rate >= 0 ? m->cfg->rebate_rate : M->maker_rebate_rate;
    m->fee_exp = M->fee_exponent;
    m->reb_obs = m->rebate_rate * m->fee_rate * (1.0 / (0.2 * 0.07 * 0.25));   // 1 at the default rates, p = 0.5
    m->next_min = 0;                 // the first matched row starts a minute
    m->last_fill_row = m->t - 1000000;
    m->ph = m->pn = 0;
    memset(m->ord, 0, sizeof(m->ord));
    m->inv = m->cash = m->rebates = m->liq = m->wealth = m->penalty = 0.0;
    m->spread = m->mo5 = m->mo30 = m->mo60 = 0.0;
    m->minute_vol = m->minute_fill = m->tx_fill = 0.0;
    m->n_fills = m->fill_vol = m->buy_vol = m->sell_vol = m->max_inv = 0.0f;
    memset(m->vol_flag, 0, sizeof(m->vol_flag));
    m->rows = m->rows_two = m->rows_any = m->steps = m->clips = m->rejects = m->ret = 0.0f;
}

static inline void mm_new_episode(Mm* m) {
    int mk;
    if (m->cfg->fixed_market >= 0) {
        mk = m->cfg->fixed_market;
    } else if (m->cfg->cycle_stride > 0) {
        mk = m->cfg->list[m->cycle_pos % (uint64_t)m->cfg->n_list];
        m->cycle_pos += (uint64_t)m->cfg->cycle_stride;
    } else {
        mk = m->cfg->list[(mm_rand(m) >> 11) % (uint64_t)m->cfg->n_list];
    }
    int len = MM_DATA.markets[mk].contract_len_s;
    int off;
    if (m->cfg->fixed_start >= 0) {
        off = m->cfg->fixed_start;
    } else {
        int span = (int)(m->cfg->start_frac * len) + 1;     // a random second in the first 20%
        off = (int)((mm_rand(m) >> 11) % (uint64_t)span);
    }
    mm_begin(m, mk, off);
}

// Data must be open. Does not touch agent buffers (PUFFER5 rule 10).
static int mm_init(Mm* m, const MmParams* p, uint64_t seed, char* err, int n) {
    memset(m, 0, sizeof(*m));
    if (!MM_DATA_READY) {
        snprintf(err, n, "mm: data not open");
        return -1;
    }
    MmCfg c;
    if (mm_config(&c, p, err, n)) return -1;
    m->cfg = mm_intern(&c);
    if (m->cfg == NULL) {
        snprintf(err, n, "mm: more than %d distinct configs in one process", MM_MAX_CFG);
        return -1;
    }
    m->rng = seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    m->cycle_pos = seed;
    mm_rand(m);
    return 0;
}

// ------------------------------------------------------------------ quoting and fills

// Target prices at decision row t (mils, 0 = off). Relative to the touch, or to the fair value
// rounded to the tick when that side of the book is empty; post-only against the opposite
// touch; inside [tick, 1 - tick]; our own bid stays below our ask.
static inline void mm_targets(const Mm* m, int t, const int* act, int* bid, int* ask) {
    const MmBook* B = &MM_DATA.book[t];
    int tick = mm_tick(m, B);
    int bb = B->bid_px[0], ba = B->ask_px[0];
    double fvm = (double)mm_feat(t)[MM_F_P] * 1000.0;
    int a0 = act[0], a1 = act[1];
    a0 = a0 < 0 ? 0 : (a0 > 6 ? 6 : a0);
    a1 = a1 < 0 ? 0 : (a1 > 6 ? 6 : a1);
    int b = 0, a = 0;
    if (a0) {
        int base = bb > 0 ? bb : (int)floor(fvm / tick) * tick;
        b = base + MM_OFFS[a0] * tick;
        if (ba < 1000 && b >= ba) b = ba - tick;
        if (b < tick) b = 0;
    }
    if (a1) {
        int base = ba < 1000 ? ba : (int)ceil(fvm / tick) * tick;
        a = base - MM_OFFS[a1] * tick;
        if (bb > 0 && a <= bb) a = bb + tick;
        if (a > 1000 - tick) a = 0;
    }
    if (b && a && b >= a) {
        a = b + tick;
        if (a > 1000 - tick) a = 0;
    }
    *bid = b;
    *ask = a;
}

// A pending decision reaches the exchange at match row tau.
static inline void mm_activate(Mm* m, int side, int px, int size_idx, int tau) {
    MmOrder* o = &m->ord[side];
    if (px == 0) {                                   // cancel
        o->live = 0;
        return;
    }
    if (o->live && o->px == px) return;              // unchanged price: keep queue position
    const MmBook* B = &MM_DATA.book[tau];
    int cross = side == 0 ? (B->ask_px[0] < 1000 && px >= B->ask_px[0])
                          : (B->bid_px[0] > 0 && px <= B->bid_px[0]);
    if (cross) {                                     // post-only reject at arrival
        o->live = 0;
        m->rejects += 1.0f;
        return;
    }
    double size = m->cfg->size_mult * m->min_size * MM_SIZES[size_idx];
    double room = side == 0 ? m->cfg->Q - m->inv : m->cfg->Q + m->inv;
    if (size > room) {                               // inventory cap: truncate or pull
        size = room;
        if (size < m->min_size) {
            o->live = 0;
            return;
        }
    }
    double d = mm_level(B, side, px);
    o->px = px;
    o->rem = size;
    o->q = (m->cfg->model == MM_OPT || d < 0.0) ? 0.0 : d;
    o->disp = d;
    o->live = 1;
    o->born = tau;
}

static inline double mm_mark_at(const Mm* m, int row) {
    return row <= m->r_end ? (double)mm_feat(row)[MM_F_MARK] : (double)m->winner;
}

static inline void mm_book_fill(Mm* m, int side, double f, int tau, int u, const MmBook* B, int fpx) {
    MmOrder* o = &m->ord[side];
    double sq = side == 0 ? f : -f;
    double p = o->px / 1000.0;
    m->inv += sq;
    m->cash -= sq * p;
    double pq = p * (1.0 - p);
    m->rebates += m->rebate_rate * m->fee_rate * f * (m->fee_exp == 1.0 ? pq : pow(pq, m->fee_exp));
    o->rem -= f;
    if (o->rem <= 1e-9) o->live = 0;
    m->tx_fill += f;
    m->minute_fill += f;
    m->n_fills += 1.0f;
    m->fill_vol += (float)f;
    if (side == 0) m->buy_vol += (float)f; else m->sell_vol += (float)f;
    double ai = fabs(m->inv);
    if (ai > m->max_inv) m->max_inv = (float)ai;
    m->last_fill_row = u;
    double mid0 = mm_feat(tau)[MM_F_MARK];
    m->spread += sq * (mid0 - p);
    m->mo5 += sq * (mm_mark_at(m, tau + 5) - mid0);
    m->mo30 += sq * (mm_mark_at(m, tau + 30) - mid0);
    m->mo60 += sq * (mm_mark_at(m, tau + 60) - mid0);
    int tick = mm_tick(m, B), bb = B->bid_px[0], ba = B->ask_px[0];
    float ff = (float)f;
    m->vol_flag[MM_FL_CROSSED] += (B->flags & MM_BOOK_CROSSED) ? ff : 0.0f;
    m->vol_flag[MM_FL_STALE] += (B->flags & MM_BOOK_STALE) ? ff : 0.0f;
    m->vol_flag[MM_FL_ONESIDED] += (bb == 0 || ba == 1000) ? ff : 0.0f;
    m->vol_flag[MM_FL_LAST30] += (m->r_end - tau < 30) ? ff : 0.0f;
    m->vol_flag[MM_FL_LAST10] += (m->r_end - tau < 10) ? ff : 0.0f;
    int inside = side == 0 ? (bb > 0 && o->px > bb + 2 * tick) : (ba < 1000 && o->px < ba - 2 * tick);
    m->vol_flag[MM_FL_INSIDE2] += inside ? ff : 0.0f;
    m->vol_flag[MM_FL_EXTREME] += (o->px <= 20 || o->px >= 980) ? ff : 0.0f;
    m->vol_flag[MM_FL_OUTSIDE] += (fpx < bb || fpx > ba) ? ff : 0.0f;
}

// Liquidity-reward hook (rewards study, AUG_2026 preset): our side only; the competitors' Q
// and the sample seconds are precomputed in feat.bin.
static inline void mm_liquidity(Mm* m, int tau) {
    const float* F = mm_feat(tau);
    double pi = F[MM_F_LIQ_PI], mid = F[MM_F_LIQ_MID];
    if (pi <= 0.0 || mid < 0.0) return;
    double qs[2] = {0.0, 0.0};
    for (int s = 0; s < 2; s++) {
        const MmOrder* o = &m->ord[s];
        if (!o->live || o->rem < MM_LIQ_MIN || tau - o->born < m->cfg->liq_moas) continue;
        double dist = fabs(o->px / 1000.0 - mid) * 100.0;
        if (dist < MM_LIQ_V) qs[s] = (MM_LIQ_V - dist) / MM_LIQ_V * ((MM_LIQ_V - dist) / MM_LIQ_V) * o->rem;
    }
    double lo = qs[0] < qs[1] ? qs[0] : qs[1], hi = qs[0] < qs[1] ? qs[1] : qs[0];
    double qmin = (mid >= 0.10 && mid <= 0.90) ? (lo > hi / 3.0 ? lo : hi / 3.0) : lo;
    if (qmin > 0.0) m->liq += pi * qmin / (qmin + F[MM_F_LIQ_COMP]);
}

// Match row tau = u - D against the trades printed in row u.
static inline void mm_row(Mm* m, int u) {
    const MmCfg* c = m->cfg;
    int tau = u - c->D;
    while (m->pn > 0 && m->pend[m->ph].row <= tau) {
        const MmPend* e = &m->pend[m->ph];
        mm_activate(m, 0, e->bid & 2047, e->bid >> 11, tau);
        mm_activate(m, 1, e->ask, e->bid >> 11, tau);
        m->ph = (m->ph + 1) & (MM_PEND - 1);
        m->pn--;
    }
    if (u >= m->next_min) {                          // a new UTC minute (participation cap)
        m->minute_vol = 0.0;
        m->minute_fill = 0.0;
        m->next_min = u + 60 - (m->open_mod + u - m->r0) % 60;
    }
    int live0 = m->ord[0].live, live1 = m->ord[1].live;
    m->rows += 1.0f;
    m->rows_two += (float)(live0 & live1);
    m->rows_any += (float)(live0 | live1);
    int lo = MM_DATA.fill_off[u], hi = MM_DATA.fill_off[u + 1];
    if (!(live0 | live1)) {
        for (int j = lo; j < hi; j++) m->minute_vol += MM_DATA.fills[j].size;
        return;
    }
    const MmBook* B = &MM_DATA.book[tau];
    if (c->liq_on) mm_liquidity(m, tau);
    double traded[2] = {0.0, 0.0};
    int allowed = 1;
    if (lo < hi && c->model != MM_OPT) {
        allowed = !(B->flags & MM_BOOK_CROSSED) && !(c->model == MM_PESS && (B->flags & MM_BOOK_STALE));
    }
    for (int j = lo; j < hi; j++) {
        const MmFill* f = &MM_DATA.fills[j];
        double sz = f->size;
        m->minute_vol += sz;
        if (f->first) m->tx_fill = 0.0;
        int side = f->side < 0 ? 0 : 1;              // a taker sell hits our bid
        MmOrder* o = &m->ord[side];
        if (!o->live) continue;
        int at = f->px == o->px;
        int through = side == 0 ? f->px < o->px : f->px > o->px;
        if (!(at | through)) continue;               // a better level absorbs this leg
        double avail;
        if (c->model == MM_OPT) {
            avail = sz;
        } else if (at) {
            traded[side] += sz;
            double eat = o->q < sz ? o->q : sz;
            o->q -= eat;
            avail = c->model == MM_CENTRAL ? sz - eat : 0.0;
        } else if (c->model == MM_CENTRAL) {
            avail = sz;
            o->q = 0.0;                              // the level was swept
        } else {
            double eat = o->q < sz ? o->q : sz;      // pessimistic: shown size at our level first
            o->q -= eat;
            avail = sz - eat;
        }
        double fill = c->model == MM_PESS ? 0.5 * avail : avail;
        if (c->model == MM_PESS) {
            double cap = 0.2 * f->tx_size - m->tx_fill;
            if (fill > cap) fill = cap;
        }
        if (c->part_cap > 0.0) {
            double cap = c->part_cap * m->minute_vol - m->minute_fill;
            if (fill > cap) fill = cap;
        }
        if (fill > o->rem) fill = o->rem;
        double room = side == 0 ? c->Q - m->inv : c->Q + m->inv;
        if (fill > room) fill = room;
        if (!allowed) fill = 0.0;
        if (fill > 1e-12) mm_book_fill(m, side, fill, tau, u, B, f->px);
    }
    // Cancels ahead of us, pro rata. Skipping rows whose ladder equals the previous row's is
    // exact: then the shown size equals disp and the update below is a no-op.
    if (c->model == MM_CENTRAL && !(B->flags & MM_BOOK_SAME)) {
        for (int s = 0; s < 2; s++) {
            MmOrder* o = &m->ord[s];
            if (!o->live || o->q <= 0.0) continue;
            double d = mm_level(B, s, o->px);
            if (d >= 0.0 && o->disp >= 0.0) {
                double others = o->disp - traded[s];
                double cancel = others - d;
                if (cancel > 0.0 && others > 1e-12) o->q -= cancel * o->q / others;
            }
            if (d >= 0.0 && o->q > d) o->q = d;
            if (o->q < 0.0) o->q = 0.0;
            o->disp = d;
        }
    }
}

// ------------------------------------------------------------------ observation

static inline void mm_write_obs(const Mm* m, float* obs) {
    int t = m->t;
    const float* F = mm_feat(t);
    memcpy(obs, F, MM_FEAT_OBS * sizeof(float));
    double Q = m->cfg->Q, mark = mm_mark(m, t), midb = F[MM_F_MARK];
    double wealth = m->cash + m->rebates + m->liq + m->inv * mark;
    float* o = obs + MM_FEAT_OBS;
    o[0] = (float)(m->inv / Q);
    o[1] = (float)mm_clip(wealth / (0.05 * Q), -5.0, 5.0);
    o[2] = (float)mm_clip(m->inv * F[MM_F_DUSD] / (200.0 * Q), -5.0, 5.0);
    for (int s = 0; s < 2; s++) {
        const MmOrder* q = &m->ord[s];
        double p = q->px / 1000.0;
        // queue ahead / shown size at our level, both in match time (disp: last match row
        // where it was tracked; queue estimates are updated there)
        o[3 + s] = (float)q->live;
        o[5 + s] = q->live ? (float)mm_clip((p - midb) * 100.0 / 2.0, -5.0, 5.0) : 0.0f;
        o[7 + s] = (q->live && q->disp > 0.0) ? (float)mm_clip(q->q / q->disp, 0.0, 1.0) : 0.0f;
        o[9 + s] = q->live ? (float)mm_clip(m->reb_obs * p * (1.0 - p), 0.0, 5.0) : 0.0f;
    }
    int since = t - m->last_fill_row;
    o[11] = MM_LOG_SINCE[since < MM_SINCE_MAX ? since : MM_SINCE_MAX];
    int pending = 0;
    if (m->pn > 0) {
        const MmPend* e = &m->pend[(m->ph + m->pn - 1) & (MM_PEND - 1)];
        pending = (e->bid & 2047) != (m->ord[0].live ? m->ord[0].px : 0)
            || e->ask != (m->ord[1].live ? m->ord[1].px : 0);
    }
    o[12] = (float)pending;
    int nf = (int)m->n_fills;
    o[13] = nf <= MM_NFILL_TAB ? MM_LOG_FILLS[nf] : (float)(log1p((double)nf) / 5.0);
    o[14] = 1.0f;
}

static inline void mm_reset(Mm* m, float* obs) {
    mm_new_episode(m);
    mm_write_obs(m, obs);
}

static inline void mm_prefetch(const Mm* m) {
#if defined(__GNUC__) || defined(__clang__)
    int t = m->t, dt = m->dt;
    int tn = t + dt <= m->r_end ? t + dt : m->r_end;
    int tau = t + 1 - m->cfg->D;
    tau = tau > 0 ? tau : 0;
    for (int r = tau; r <= tn - m->cfg->D; r++) __builtin_prefetch(&MM_DATA.book[r]);
    __builtin_prefetch(&MM_DATA.fill_off[t + 1]);
    const char* f = (const char*)mm_feat(tn);
    __builtin_prefetch(f);
    __builtin_prefetch(f + 64);
    __builtin_prefetch(f + 128);
    __builtin_prefetch(&MM_DATA.book[tn]);
#endif
}

static inline void mm_log_episode(Mm* m, Log* log, double pnl) {
    double trading = m->cash + m->inv * m->winner;
    log->score += (float)pnl;
    log->perf += pnl > 0.0 ? 1.0f : 0.0f;
    log->episode_return += m->ret;
    log->episode_length += m->steps;
    log->pnl += (float)pnl;
    log->pnl_trading += (float)trading;
    log->spread_capture += (float)m->spread;
    log->markout_5 += (float)m->mo5;
    log->markout_30 += (float)m->mo30;
    log->markout_60 += (float)m->mo60;
    log->inventory_pnl += (float)(trading - m->spread - m->mo60);
    log->rebates += (float)m->rebates;
    log->liq_rewards += (float)m->liq;
    log->inv_penalty += (float)m->penalty;
    log->fills += m->n_fills;
    log->fill_vol += m->fill_vol;
    log->buy_vol += m->buy_vol;
    log->sell_vol += m->sell_vol;
    log->max_abs_inv += m->max_inv;
    log->final_abs_inv += (float)fabs(m->inv);
    log->uptime_two_sided += m->rows > 0 ? m->rows_two / m->rows : 0.0f;
    log->uptime_any += m->rows > 0 ? m->rows_any / m->rows : 0.0f;
    log->vol_crossed += m->vol_flag[MM_FL_CROSSED];
    log->vol_stale += m->vol_flag[MM_FL_STALE];
    log->vol_onesided += m->vol_flag[MM_FL_ONESIDED];
    log->vol_last30 += m->vol_flag[MM_FL_LAST30];
    log->vol_last10 += m->vol_flag[MM_FL_LAST10];
    log->vol_inside2 += m->vol_flag[MM_FL_INSIDE2];
    log->vol_extreme += m->vol_flag[MM_FL_EXTREME];
    log->vol_outside += m->vol_flag[MM_FL_OUTSIDE];
    log->reward_clip_frac += m->steps > 0 ? m->clips / m->steps : 0.0f;
    log->rejects += m->rejects;
    log->n += 1.0f;
}

// One decision. act = {bid level 0..6, ask level 0..6, size 0..2}. Writes the next observation,
// the reward and the terminal flag every step; on the last row of the market it settles, logs,
// starts the next episode and writes that episode's first observation (auto-reset).
static inline void mm_step(Mm* m, const int* act, float* obs, float* rew, float* term, Log* log) {
    int t = m->t;
    int bid, ask;
    mm_targets(m, t, act, &bid, &ask);
    int size = act[2] < 0 ? 0 : (act[2] > 2 ? 2 : act[2]);
    MmPend* e = &m->pend[(m->ph + m->pn) & (MM_PEND - 1)];
    e->row = t + m->cfg->L;
    e->bid = (int16_t)(bid | (size << 11));
    e->ask = (int16_t)ask;
    m->pn++;
    int tn = t + m->dt < m->r_end ? t + m->dt : m->r_end;
    for (int u = t + 1; u <= tn; u++) mm_row(m, u);
    m->t = tn;
    int done = tn == m->r_end;
    double mark = done ? (double)m->winner : mm_mark(m, tn);
    double w = m->cash + m->rebates + m->liq + m->inv * mark;
    double pen = m->cfg->inv_penalty * m->inv * m->inv * mark * (1.0 - mark) * (tn - t) / m->len_s;
    double r = m->cfg->reward_mode == 0 ? (w - m->wealth - pen) * m->cfg->reward_scale
                                       : (act[0] == 3 && act[1] == 3 ? 1.0 : 0.0);
    m->wealth = w;
    m->penalty += pen;
    float rf = (float)r;
    if (rf > 1.0f || rf < -1.0f) {
        m->clips += 1.0f;
        rf = rf > 1.0f ? 1.0f : -1.0f;
    }
    *rew = rf;
    m->ret += rf;
    m->steps += 1.0f;
    if (done) {
        mm_log_episode(m, log, w);
        mm_new_episode(m);
        *term = 1.0f;
    } else {
        *term = 0.0f;
    }
    mm_write_obs(m, obs);
    mm_prefetch(m);
}

#endif
