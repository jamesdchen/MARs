/* mm3.h: Polymarket market-making replay env v3, pure C core (DECISIONS.md D11: pre-game sports
 * and election reward farming, plus the crypto up/down data of v2).
 *
 * This is mm.h (v2, sha256 9189f83a...) with every name prefixed m3_ / M3_ / Mm3 so both headers
 * can be included in one translation unit (cmp_harness.c does, to check bit-exactness), plus six
 * extensions. With every new option at its default and no v3 side files in the data directory,
 * mm3 steps bit for bit like mm.h (observations 0..49, rewards, terminals, Log). ENV3.md documents
 * the options; ../ENV.md documents everything inherited.
 *
 *  1. Episode length from the data: a market's episode has n_rows - 1 rows, whatever that is.
 *     Lengths other than 300 / 900 / 14400 get length bit 8 and their own dt (dt_other).
 *     Optional variable dt: dt_far rows per decision while more than far_s seconds remain to the
 *     market's event (kickoff / window end).
 *  2. Settlement at a float price, per market (markets3.bin): settle_mode 0 winner_up (v2),
 *     1 settle_price (sports: the kickoff mid), 2 the book mark at the last row (unresolved
 *     elections). flat_c (cents/share) charges a flattening cost at settlement for modes 1/2.
 *     close_inv_coef / close_ramp / close_ramp_s move the trainer's close-inventory shaping
 *     (train_ppo.py CloseInventoryPenalty) into the core. pull_s forces quotes off near the event.
 *  3. Liquidity-reward hook with per-market (markets3.bin) or per-row (x3.bin) max spread v,
 *     min size, daily rate and competitor Q. Polymarket's quadratic score ((v - d)/v)^2 * size,
 *     two-sided Q_min rule inside [0.10, 0.90] (c = 3), Q_min / (Q_min + Q_comp) share of the
 *     per-sample pool. The v2 constants (1.5 c, 50 shares) remain only as the legacy values for
 *     data without v3 side files (crypto feat.bin liq_comp was computed with them).
 *  4. size_mode 1: size heads are size_rel{0,1,2} x max(rewards min size, min_order_size).
 *  5. obs_size > 50 appends M3_NX extra slots (time to event, category, reward features); the
 *     first 50 are v2's.
 *  6. Fee / rebate coefficients per category (fee_rate_<cat>, rebate_rate_<cat>), and per market
 *     in markets3.bin; the fill models, latency, match delay and rebate formula are unchanged.
 *
 * Data: the five v2 files (markets.bin, feat.bin, book.bin, fills.bin, fill_index.bin) plus two
 * optional v3 side files written by adapt3.py: markets3.bin (64 B per market, Mm3Market) and
 * x3.bin (float32 [N, 8], per-row reward program and liquidity columns).
 */

#ifndef MM3_H
#define MM3_H

#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define M3_OBS_V2 50       // v2 observation (obs 0..49, obs 49 = 1)
#define M3_NX 28           // extra slots written at obs 50.. when obs_size > 50
#define M3_OBS_MAX 128     // largest obs_size accepted (slots past 50 + M3_NX are 0)
#define M3_OBS_FULL 80     // recommended width: 50 + 28 extras + 2 zero pads (multiple of 8)
#define M3_NUM_ATNS 3
#define M3_ACT_SIZES {7, 7, 3}
#define M3_G 44            // feat.bin columns
#define M3_FEAT_OBS 33
#define M3_F_P 33
#define M3_F_MARK 34
#define M3_F_DUSD 35
#define M3_F_LIQ_MID 38
#define M3_F_LIQ_COMP 39
#define M3_F_LIQ_PI 40
#define M3_F_BIDP 41
#define M3_F_ASKP 42
#define M3_PEND 16
#define M3_LEGACY_V 1.5    // v2 reward max spread (cents): only for data without markets3.bin
#define M3_LEGACY_MIN 50.0 // v2 reward min size: only for data without markets3.bin
#define M3_BAND_LO 0.10    // Polymarket: single-sided quotes score (divided by c) inside the band
#define M3_BAND_HI 0.90
#define M3_REWARD_C 3.0
#define M3_X 8             // x3.bin columns

enum { M3_OPT = 0, M3_CENTRAL = 1, M3_PESS = 2 };
enum { M3_BOOK_VALID = 1, M3_BOOK_CROSSED = 2, M3_BOOK_STALE = 4, M3_BOOK_SAME = 16 };
enum { M3_FL_CROSSED, M3_FL_STALE, M3_FL_ONESIDED, M3_FL_LAST30, M3_FL_LAST10, M3_FL_INSIDE2,
       M3_FL_EXTREME, M3_FL_OUTSIDE, M3_FL_INSPREAD, M3_NFLAGS };
enum { M3_SETTLE_WINNER = 0, M3_SETTLE_PRICE = 1, M3_SETTLE_LAST_MARK = 2 };
enum { M3_CAT_CRYPTO = 0, M3_CAT_SPORTS = 1, M3_CAT_ELECTIONS = 2, M3_CAT_OTHER = 3 };
// x3.bin columns (float32 [N, M3_X]); when present they replace the per-market reward program and
// feat.bin's liq_mid / liq_comp / liq_pi.
enum { M3_X_V = 0, M3_X_MIN, M3_X_RATE, M3_X_PROG, M3_X_LIQ_MID, M3_X_LIQ_COMP, M3_X_LIQ_PI, M3_X_PAD };
#define M3_FLAG3_LEGACY 1  // Mm3Market.flags3: synthesised for v2 data (no markets3.bin)

typedef struct {
    int64_t row_offset, open_ts, end_ts;
    int32_t n_rows, contract_len_s, coin_id, winner_up, split, flags;
    float tick_size, taker_fee_rate, fee_exponent, min_order_size;
    float rewards_min_size, rewards_max_spread, log_strike_vs_ul_open, maker_rebate_rate;
    int32_t market_idx;
    float tick_size_final;
} Mm3Market;                                     // 88 bytes, markets.bin (data/README.md)

typedef struct {
    double settle_price;     // settle_mode 1: settlement value of one Up share
    int64_t event_ts;        // unix s of the event the episode runs to (kickoff, window end)
    int32_t settle_mode;     // M3_SETTLE_*
    int32_t category;        // M3_CAT_*
    float row_s;             // seconds per row (builders' grid step dt_s)
    float rw_v_c;            // reward max spread v (cents from the adjusted mid); 0 = no program
    float rw_min_size;       // reward min size (shares per order)
    float rw_rate_day;       // reward pool USD/day (observation; the paid pool is liq_pi)
    float fee_rate, fee_exp, rebate_rate;    // < 0: markets.bin
    int32_t flags3;
    int32_t pad[2];
} Mm3Ext;                                        // 64 bytes, markets3.bin

typedef struct {
    int16_t bid_px[5];
    int16_t ask_px[5];
    float bid_sz[5];
    float ask_sz[5];
    uint8_t flags;
    uint8_t tick;
    uint8_t tick_alt;
    uint8_t same_run;
} Mm3Book;                                       // 64 bytes

typedef struct {
    int32_t row;
    int8_t side;
    uint8_t first;
    int16_t px;
    float price, size, tx_size;
} Mm3Fill;                                       // 20 bytes

typedef char m3_layout_check[(sizeof(Mm3Market) == 88 && sizeof(Mm3Ext) == 64 && sizeof(Mm3Book) == 64
    && sizeof(Mm3Fill) == 20) ? 1 : -1];

// Floats only, n last. The first 42 fields are v2's Log, in v2's order.
typedef struct Mm3Log Mm3Log;
struct Mm3Log {
    float score;
    float perf;
    float episode_return;
    float episode_length;
    float pnl;
    float pnl_trading;        // cash + inventory * settlement - flattening cost (no rebates / rewards)
    float spread_capture;
    float markout_5;
    float markout_30;
    float markout_60;
    float inventory_pnl;
    float rebates;
    float liq_rewards;
    float inv_penalty;
    float fills;
    float fill_vol;
    float buy_vol;
    float sell_vol;
    float max_abs_inv;
    float final_abs_inv;
    float uptime_two_sided;
    float uptime_any;
    float reward_clip_frac;
    float rejects;
    float vol_crossed;
    float vol_stale;
    float vol_onesided;
    float vol_last30;
    float vol_last10;
    float vol_inside2;
    float vol_extreme;
    float vol_outside;
    float vol_inspread;
    float pnl_crossed;
    float pnl_stale;
    float pnl_onesided;
    float pnl_last30;
    float pnl_last10;
    float pnl_inside2;
    float pnl_extreme;
    float pnl_outside;
    float pnl_inspread;
    // v3 additions
    float close_penalty;      // sum of the close-inventory shaping (reward units; not in pnl)
    float flat_cost;          // USD charged for flattening at settlement (in pnl)
    float settle_px;          // settlement value used
    float rw_samples;         // reward sample rows seen while any quote was live
    float rw_scored;          // ... at which our orders scored
    float rw_share;           // sum over scored samples of our share Q_min / (Q_min + Q_comp)
    float episode_rows;       // rows of the episode (start to end)
    // D15 netting / flatten diagnostics (RECONCILE.md section 6); Log only, nothing else reads them
    float net_gross_sh;       // gross filled shares (buys + sells), summed in double
    float net_notional;       // filled notional, sum of price x shares (USD)
    float net_end_inv;        // |inventory| at the episode's end (the settlement row: kickoff for sports)
    float net_share;          // 1 - net_end_inv / net_gross_sh, for an episode with fills (else 0)
    float net_eps;            // 1 for an episode with fills (net_share / net_eps = mean per episode)
    float net_flat_drag;      // flattening cost charged at settlement, USD (= flat_c / 100 x net_end_inv; in pnl)
    float n;
};

typedef char m3_log_layout_check[(sizeof(Mm3Log) == 56 * sizeof(float)
    && offsetof(Mm3Log, pnl_crossed) - offsetof(Mm3Log, vol_crossed) == M3_NFLAGS * sizeof(float)
    && offsetof(Mm3Log, close_penalty) == 42 * sizeof(float)) ? 1 : -1];

// Parameters: doubles in this order. The first 27 are v2's MM_PARAM_NAMES, unchanged.
typedef struct {
    double split, allow_test, fill_model, latency, match_delay, dt, dt_5m, dt_15m, dt_4h;
    double Q, inv_penalty, reward_scale, size_mult, part_cap, fee_rate, rebate_rate;
    double liq_rewards, liq_moas, mark_mode, reward_mode, tick_rule, len_mask;
    double fixed_market, fixed_start, cycle_stride, start_frac, markout_h;
    // v3
    double dt_other, dt_far, far_s, cat_mask, obs_size;
    double size_mode, size_rel0, size_rel1, size_rel2;
    double liq_v_c, liq_min_size;
    double settle_mode, flat_c, pull_s, close_inv_coef, close_ramp, close_ramp_s;
    double fee_rate_crypto, rebate_rate_crypto, fee_rate_sports, rebate_rate_sports,
        fee_rate_elections, rebate_rate_elections;
} Mm3Params;

static const char* M3_PARAM_NAMES[] = {
    "split", "allow_test", "fill_model", "latency", "match_delay", "dt", "dt_5m", "dt_15m", "dt_4h",
    "Q", "inv_penalty", "reward_scale", "size_mult", "part_cap", "fee_rate", "rebate_rate",
    "liq_rewards", "liq_moas", "mark_mode", "reward_mode", "tick_rule", "len_mask",
    "fixed_market", "fixed_start", "cycle_stride", "start_frac", "markout_h",
    "dt_other", "dt_far", "far_s", "cat_mask", "obs_size",
    "size_mode", "size_rel0", "size_rel1", "size_rel2",
    "liq_v_c", "liq_min_size",
    "settle_mode", "flat_c", "pull_s", "close_inv_coef", "close_ramp", "close_ramp_s",
    "fee_rate_crypto", "rebate_rate_crypto", "fee_rate_sports", "rebate_rate_sports",
    "fee_rate_elections", "rebate_rate_elections",
};
#define M3_NUM_PARAMS (int)(sizeof(M3_PARAM_NAMES) / sizeof(M3_PARAM_NAMES[0]))

static const double M3_DEFAULTS[] = {
    0, 0, 1, 1, -1, 0, 1, 1, 5,
    100, 0.001, 0, 1, -1, -1, -1,
    0, 1, 0, 0, 0, 15,
    -1, -1, 0, 0.2, 60,
    5, 0, 7200, 15, 50,
    0, 1, 2, 4,
    -1, -1,
    -1, 0, 0, 0, 0.2, 0,
    -1, -1, -1, -1,
    -1, -1,
};

typedef char m3_param_check[(sizeof(Mm3Params) == sizeof(M3_DEFAULTS)
    && sizeof(M3_DEFAULTS) / sizeof(double) == sizeof(M3_PARAM_NAMES) / sizeof(char*)) ? 1 : -1];

// ------------------------------------------------------------------ shared data

typedef struct {
    const Mm3Market* markets;
    const Mm3Ext* ext;           // markets3.bin, or synthesised legacy records (owned)
    const float* feat;
    const float* x3;             // x3.bin or NULL
    const Mm3Book* book;
    const Mm3Fill* fills;
    const int32_t* fill_off;
    int64_t N, n_fills;
    int n_markets, has_ext, ext_owned;
    void* maps[7];
    size_t map_size[7];
    char dir[1024];
} Mm3Data;

static Mm3Data M3_DATA;
static int M3_DATA_READY = 0;

#define M3_SINCE_MAX 3600
#define M3_NFILL_TAB 1024
static float M3_LOG_SINCE[M3_SINCE_MAX + 1];
static float M3_LOG_FILLS[M3_NFILL_TAB + 1];

static int m3_map(const char* dir, const char* name, size_t* size, const void** out, char* err, int n,
        int optional) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (optional) {
            *out = NULL;
            *size = 0;
            return 0;
        }
        snprintf(err, n, "mm3: cannot open %s", path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        snprintf(err, n, "mm3: cannot stat %s (or it is empty)", path);
        return -1;
    }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        snprintf(err, n, "mm3: mmap failed for %s", path);
        return -1;
    }
    *size = (size_t)st.st_size;
    *out = p;
    return 0;
}

// Length bit of an episode of len rows: 1 (300), 2 (900), 4 (14400), 8 (any other length).
static inline int m3_len_bit(int len) {
    return len == 300 ? 1 : (len == 900 ? 2 : (len == 14400 ? 4 : 8));
}

static inline int m3_len_class(int len) {
    return len == 300 ? 0 : (len == 900 ? 1 : (len == 14400 ? 2 : 3));
}

static void m3_unmap_all(Mm3Data* d) {
    for (int i = 0; i < 7; i++) {
        if (d->maps[i]) munmap(d->maps[i], d->map_size[i]);
        d->maps[i] = NULL;
    }
    if (d->ext_owned) free((void*)d->ext);
    d->ext = NULL;
    d->ext_owned = 0;
}

// Maps the data once per process (static guard); later calls only check the directory.
static int m3_data_open(const char* dir, char* err, int n) {
    if (M3_DATA_READY) {
        if (strcmp(dir, M3_DATA.dir) != 0) {
            snprintf(err, n, "mm3: data already loaded from %s, not %s", M3_DATA.dir, dir);
            return -1;
        }
        return 0;
    }
    Mm3Data d;
    memset(&d, 0, sizeof(d));
    static const char* names[7] = {"markets.bin", "feat.bin", "book.bin", "fills.bin", "fill_index.bin",
                                   "markets3.bin", "x3.bin"};
    const void* p[7];
    for (int i = 0; i < 7; i++) {
        if (m3_map(dir, names[i], &d.map_size[i], &p[i], err, n, i >= 5)) {
            m3_unmap_all(&d);
            return -1;
        }
        d.maps[i] = (void*)p[i];
    }
    size_t s_mk = d.map_size[0], s_ft = d.map_size[1], s_bk = d.map_size[2], s_fl = d.map_size[3];
    size_t s_fo = d.map_size[4], s_ex = d.map_size[5], s_x3 = d.map_size[6];
    if (s_mk % sizeof(Mm3Market) != 0) {
        snprintf(err, n, "mm3: markets.bin size %zu is not a multiple of %zu", s_mk, sizeof(Mm3Market));
        m3_unmap_all(&d);
        return -1;
    }
    d.markets = (const Mm3Market*)p[0];
    d.n_markets = (int)(s_mk / sizeof(Mm3Market));
    const Mm3Market* last = &d.markets[d.n_markets - 1];
    d.N = last->row_offset + last->n_rows;
    if (s_ft != (size_t)d.N * M3_G * sizeof(float)) {
        snprintf(err, n, "mm3: feat.bin has %zu bytes, expected N=%lld x G=%d floats (wide layouts are not "
            "supported: build with the split layout)", s_ft, (long long)d.N, M3_G);
        m3_unmap_all(&d);
        return -1;
    }
    if (s_bk != (size_t)d.N * sizeof(Mm3Book) || s_fo != (size_t)(d.N + 1) * sizeof(int32_t)
            || s_fl % sizeof(Mm3Fill) != 0) {
        snprintf(err, n, "mm3: book.bin / fill_index.bin / fills.bin sizes do not match N=%lld", (long long)d.N);
        m3_unmap_all(&d);
        return -1;
    }
    if (p[5] != NULL && s_ex != (size_t)d.n_markets * sizeof(Mm3Ext)) {
        snprintf(err, n, "mm3: markets3.bin has %zu bytes, expected %d x %zu (rerun adapt3.py)", s_ex,
            d.n_markets, sizeof(Mm3Ext));
        m3_unmap_all(&d);
        return -1;
    }
    if (p[6] != NULL && s_x3 != (size_t)d.N * M3_X * sizeof(float)) {
        snprintf(err, n, "mm3: x3.bin has %zu bytes, expected N=%lld x %d floats (rerun adapt3.py)", s_x3,
            (long long)d.N, M3_X);
        m3_unmap_all(&d);
        return -1;
    }
    d.feat = (const float*)p[1];
    d.book = (const Mm3Book*)p[2];
    d.fills = (const Mm3Fill*)p[3];
    d.fill_off = (const int32_t*)p[4];
    d.x3 = (const float*)p[6];
    d.n_fills = (int64_t)(s_fl / sizeof(Mm3Fill));
    if (d.fill_off[0] != 0 || d.fill_off[d.N] != d.n_fills) {
        snprintf(err, n, "mm3: fill_index.bin does not cover fills.bin");
        m3_unmap_all(&d);
        return -1;
    }
    if (p[5] != NULL) {
        d.ext = (const Mm3Ext*)p[5];
        d.has_ext = 1;
        for (int i = 0; i < d.n_markets; i++) {
            const Mm3Ext* e = &d.ext[i];
            if (e->settle_mode < 0 || e->settle_mode > 2 || e->category < 0 || e->category > 3 || !(e->row_s > 0.0f)
                    || (e->settle_mode == M3_SETTLE_PRICE && !(e->settle_price >= 0.0 && e->settle_price <= 1.0))) {
                snprintf(err, n, "mm3: markets3.bin record %d is invalid (settle_mode %d, category %d, row_s %g, "
                    "settle_price %g)", i, e->settle_mode, e->category, (double)e->row_s, e->settle_price);
                m3_unmap_all(&d);
                return -1;
            }
        }
    } else {                     // v2 data: synthesise the legacy per-market records
        Mm3Ext* e = (Mm3Ext*)calloc((size_t)d.n_markets, sizeof(Mm3Ext));
        for (int i = 0; i < d.n_markets; i++) {
            const Mm3Market* M = &d.markets[i];
            e[i].settle_price = M->winner_up >= 0 ? (double)M->winner_up : 0.0;
            e[i].event_ts = M->end_ts;
            e[i].settle_mode = M->winner_up >= 0 ? M3_SETTLE_WINNER : M3_SETTLE_LAST_MARK;
            e[i].category = M3_CAT_CRYPTO;
            e[i].row_s = 1.0f;
            e[i].rw_v_c = (float)M3_LEGACY_V;
            e[i].rw_min_size = (float)M3_LEGACY_MIN;
            e[i].rw_rate_day = 0.0f;
            e[i].fee_rate = e[i].fee_exp = e[i].rebate_rate = -1.0f;
            e[i].flags3 = M3_FLAG3_LEGACY;
        }
        d.ext = e;
        d.ext_owned = 1;
    }
    snprintf(d.dir, sizeof(d.dir), "%s", dir);
    for (int i = 0; i <= M3_SINCE_MAX; i++) M3_LOG_SINCE[i] = (float)(log1p((double)i) / log(M3_SINCE_MAX + 1.0));
    for (int i = 0; i <= M3_NFILL_TAB; i++) M3_LOG_FILLS[i] = (float)(log1p((double)i) / 5.0);
    M3_DATA = d;
    M3_DATA_READY = 1;
    return 0;
}

// ------------------------------------------------------------------ env state

typedef struct {
    double rem;
    double q;
    double disp;
    int px;
    int live;
    int born;
    int pad;
} Mm3Order;

typedef struct {
    int32_t row;
    int16_t bid;     // mils | size head << 11
    int16_t ask;
} Mm3Pend;

typedef struct {
    int split, model, L, D, dt_len[4], mark_mode, reward_mode, tick_rule, liq_on, cycle_stride;
    int fixed_market, fixed_start, n_list, markout_h;
    int dt_far, obs_size, size_mode, settle_mode, pad;
    const int* list;
    double Q, inv_penalty, reward_scale, size_mult, part_cap, fee_rate, rebate_rate, liq_moas,
        start_frac;
    double far_s, size_rel[3], liq_v_c, liq_min_size, flat_c, pull_s, close_inv_coef, close_ramp,
        close_ramp_s, fee_cat[4], reb_cat[4];
} Mm3Cfg;

typedef struct {
    const Mm3Cfg* cfg;
    int t, dt, r0, r_end, next_min, ph, pn, len;
    int last_fill_row, market, s0, open_mod;
    double inv, cash, rebates, liq, wealth, penalty;
    double minute_vol, minute_fill, tx_fill, credit;
    Mm3Order ord[2];
    float rows, rows_two, rows_any, steps, clips, ret, n_fills, rejects;
    Mm3Pend pend[M3_PEND];
    double len_s, min_size, fee_rate, rebate_rate, fee_exp, reb_obs;
    double spread, mo5, mo30, mo60;
    float fill_vol, buy_vol, sell_vol, max_inv;
    float vol_flag[M3_NFLAGS];
    float pnl_flag[M3_NFLAGS];
    uint64_t rng, cycle_pos;
    // v3 episode constants and accumulators
    double settle, flat, row_s, ev_off;     // ev_off = event_ts - open_ts (s)
    double rw_v, rw_min, rw_rate;
    double close_pen, flat_cost, rw_share;
    double gross, notional;                 // D15 diagnostics: filled shares and notional (Log only)
    float rw_samples, rw_scored;
    int category, legacy;
} Mm3;

#define M3_MAX_CFG 256
static Mm3Cfg M3_CFGS[M3_MAX_CFG];
static int M3_NCFG = 0;

// Market lists per (split, len_mask, cat_mask), built by init (serial) and shared.
#define M3_MAX_LISTS 128
typedef struct {
    int split, len_mask, cat_mask, n;
    int* idx;
} Mm3List;
static Mm3List M3_LISTS[M3_MAX_LISTS];
static int M3_NLISTS = 0;

static const Mm3Cfg* m3_intern(const Mm3Cfg* c) {
    for (int i = 0; i < M3_NCFG; i++) {
        if (memcmp(&M3_CFGS[i], c, sizeof(Mm3Cfg)) == 0) return &M3_CFGS[i];
    }
    if (M3_NCFG == M3_MAX_CFG) return NULL;
    memcpy(&M3_CFGS[M3_NCFG], c, sizeof(Mm3Cfg));
    return &M3_CFGS[M3_NCFG++];
}

static const Mm3List* m3_list(int split, int len_mask, int cat_mask) {
    for (int i = 0; i < M3_NLISTS; i++) {
        const Mm3List* l = &M3_LISTS[i];
        if (l->split == split && l->len_mask == len_mask && l->cat_mask == cat_mask) return l;
    }
    if (M3_NLISTS == M3_MAX_LISTS) return NULL;
    Mm3List* l = &M3_LISTS[M3_NLISTS];
    l->split = split;
    l->len_mask = len_mask;
    l->cat_mask = cat_mask;
    l->idx = (int*)malloc((size_t)(M3_DATA.n_markets > 0 ? M3_DATA.n_markets : 1) * sizeof(int));
    l->n = 0;
    for (int i = 0; i < M3_DATA.n_markets; i++) {
        const Mm3Market* M = &M3_DATA.markets[i];
        if (M->split == split && (m3_len_bit(M->n_rows - 1) & len_mask)
                && ((1 << M3_DATA.ext[i].category) & cat_mask)) {
            l->idx[l->n++] = i;
        }
    }
    M3_NLISTS++;
    return l;
}

// Test-only: unmap the data and drop configs and lists, so a process can open another directory.
// Every Mm3 of the process becomes invalid. The bindings never call it.
static void m3_data_release(void) {
    if (!M3_DATA_READY) return;
    for (int i = 0; i < M3_NLISTS; i++) free(M3_LISTS[i].idx);
    M3_NLISTS = 0;
    M3_NCFG = 0;
    m3_unmap_all(&M3_DATA);
    memset(&M3_DATA, 0, sizeof(M3_DATA));
    M3_DATA_READY = 0;
}

static const int M3_OFFS[7] = {0, 2, 1, 0, -1, -2, -4};
static const double M3_SIZES[3] = {1.0, 2.0, 5.0};

static inline uint64_t m3_rand(Mm3* m) {
    uint64_t z = (m->rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline const float* m3_feat(int64_t row) {
    return M3_DATA.feat + row * M3_G;
}

static inline double m3_clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline double m3_mark(const Mm3* m, int row) {
    const float* f = m3_feat(row);
    double mid = f[M3_F_MARK], p = f[M3_F_P];
    return m->cfg->mark_mode == 0 ? mid : (m->cfg->mark_mode == 1 ? p : 0.5 * (mid + p));
}

static inline double m3_level(const Mm3Book* B, int side, int px) {
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

static inline int m3_tick(const Mm3* m, const Mm3Book* B) {
    return m->cfg->tick_rule ? B->tick_alt : B->tick;
}

// Seconds from row t to the market's event (kickoff / window end).
static inline double m3_to_event(const Mm3* m, int t) {
    return m->ev_off - (double)(t - m->r0) * m->row_s;
}

// Per-row liquidity inputs: x3.bin when present, else feat.bin (liq_*) and the per-market program.
static inline double m3_liq_mid(int row) {
    return M3_DATA.x3 ? (double)M3_DATA.x3[(int64_t)row * M3_X + M3_X_LIQ_MID] : (double)m3_feat(row)[M3_F_LIQ_MID];
}
static inline double m3_liq_comp(int row) {
    return M3_DATA.x3 ? (double)M3_DATA.x3[(int64_t)row * M3_X + M3_X_LIQ_COMP] : (double)m3_feat(row)[M3_F_LIQ_COMP];
}
static inline double m3_liq_pi(int row) {
    return M3_DATA.x3 ? (double)M3_DATA.x3[(int64_t)row * M3_X + M3_X_LIQ_PI] : (double)m3_feat(row)[M3_F_LIQ_PI];
}
static inline double m3_rw_v(const Mm3* m, int row) {
    if (m->cfg->liq_v_c > 0.0) return m->cfg->liq_v_c;
    return M3_DATA.x3 ? (double)M3_DATA.x3[(int64_t)row * M3_X + M3_X_V] : m->rw_v;
}
static inline double m3_rw_min(const Mm3* m, int row) {
    if (m->cfg->liq_min_size > 0.0) return m->cfg->liq_min_size;
    return M3_DATA.x3 ? (double)M3_DATA.x3[(int64_t)row * M3_X + M3_X_MIN] : m->rw_min;
}
static inline double m3_rw_rate(const Mm3* m, int row) {
    return M3_DATA.x3 ? (double)M3_DATA.x3[(int64_t)row * M3_X + M3_X_RATE] : m->rw_rate;
}
static inline int m3_in_program(const Mm3* m, int row) {
    if (M3_DATA.x3) return M3_DATA.x3[(int64_t)row * M3_X + M3_X_PROG] > 0.0f;
    return m->legacy || m->rw_rate > 0.0;
}

// ------------------------------------------------------------------ config

static int m3_config(Mm3Cfg* c, const Mm3Params* p, char* err, int n) {
    memset(c, 0, sizeof(*c));
    c->split = (int)p->split;
    c->model = (int)p->fill_model;
    c->L = (int)p->latency;
    if (c->split < 0 || c->split > 2) {
        snprintf(err, n, "mm3: split must be 0 (train), 1 (val) or 2 (test)");
        return -1;
    }
    if (c->split == 2) {
        const char* f = getenv("MM_FINAL");
        if (p->allow_test != 1.0 || f == NULL || strcmp(f, "1") != 0) {
            snprintf(err, n, "mm3: split=2 (TEST) refused: needs allow_test=1 AND env MM_FINAL=1 "
                "(EVALUATION.md section 5, run once)");
            return -1;
        }
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "\n**************************************************************\n"
                "*** mm3: TEST SPLIT IN USE (data: %s): allow_test=1, MM_FINAL=1 ***\n"
                "*** This must be the single pre-registered test run.       ***\n"
                "**************************************************************\n\n", M3_DATA.dir);
            warned = 1;
        }
    }
    if (c->model < M3_OPT || c->model > M3_PESS || c->L < 0 || c->L > 5) {
        snprintf(err, n, "mm3: fill_model must be 0/1/2 and latency in 0..5");
        return -1;
    }
    static const double delay[3] = {0.0, 3.5, 5.0};
    double d = p->match_delay < 0 ? delay[c->model] : p->match_delay;
    c->D = (int)ceil(d + 0.5);
    if (c->L + c->D + 2 > M3_PEND) {
        snprintf(err, n, "mm3: latency + match delay too large for the pending FIFO");
        return -1;
    }
    int dts[4] = {(int)p->dt_5m, (int)p->dt_15m, (int)p->dt_4h, (int)p->dt_other};
    for (int i = 0; i < 4; i++) {
        c->dt_len[i] = p->dt > 0 ? (int)p->dt : dts[i];
        if (c->dt_len[i] < 1) {
            snprintf(err, n, "mm3: dt must be >= 1");
            return -1;
        }
    }
    c->Q = p->Q;
    c->inv_penalty = p->inv_penalty;
    c->reward_scale = p->reward_scale > 0 ? p->reward_scale : 1.0 / p->Q;
    c->size_mult = p->size_mult;
    c->part_cap = p->part_cap >= 0 ? p->part_cap : (c->model == M3_OPT ? 0.0 : 0.25);
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
    c->markout_h = (int)p->markout_h;
    c->dt_far = (int)p->dt_far;
    c->far_s = p->far_s;
    c->obs_size = (int)p->obs_size;
    c->size_mode = (int)p->size_mode;
    c->size_rel[0] = p->size_rel0;
    c->size_rel[1] = p->size_rel1;
    c->size_rel[2] = p->size_rel2;
    c->liq_v_c = p->liq_v_c;
    c->liq_min_size = p->liq_min_size;
    c->settle_mode = (int)p->settle_mode;
    c->flat_c = p->flat_c;
    c->pull_s = p->pull_s;
    c->close_inv_coef = p->close_inv_coef;
    c->close_ramp = p->close_ramp;
    c->close_ramp_s = p->close_ramp_s;
    c->fee_cat[0] = p->fee_rate_crypto;
    c->fee_cat[1] = p->fee_rate_sports;
    c->fee_cat[2] = p->fee_rate_elections;
    c->fee_cat[3] = -1.0;
    c->reb_cat[0] = p->rebate_rate_crypto;
    c->reb_cat[1] = p->rebate_rate_sports;
    c->reb_cat[2] = p->rebate_rate_elections;
    c->reb_cat[3] = -1.0;
    int mask = (int)p->len_mask, cmask = (int)p->cat_mask;
    if (c->Q <= 0 || c->size_mult <= 0 || mask < 1 || mask > 15 || cmask < 1 || cmask > 15 || c->mark_mode < 0
            || c->mark_mode > 2 || c->reward_mode < 0 || c->reward_mode > 2 || c->tick_rule < 0 || c->tick_rule > 1
            || c->start_frac < 0 || c->start_frac >= 1 || c->markout_h < 0) {
        snprintf(err, n, "mm3: bad Q / size_mult / len_mask / cat_mask / mark_mode / reward_mode / tick_rule / "
            "start_frac / markout_h");
        return -1;
    }
    if (c->dt_far < 0 || c->obs_size < M3_OBS_V2 || c->obs_size > M3_OBS_MAX || c->size_mode < 0 || c->size_mode > 1
            || !(c->size_rel[0] > 0 && c->size_rel[1] > 0 && c->size_rel[2] > 0) || c->settle_mode < -1
            || c->settle_mode > 2 || c->flat_c < 0 || c->pull_s < 0 || c->close_inv_coef < 0
            || (c->close_inv_coef > 0 && c->close_ramp_s <= 0 && c->close_ramp <= 0)) {
        snprintf(err, n, "mm3: bad dt_far / obs_size (50..%d) / size_mode / size_rel* / settle_mode / flat_c / pull_s"
            " / close_inv_coef / close_ramp", M3_OBS_MAX);
        return -1;
    }
    const Mm3List* l = m3_list(c->split, mask, cmask);
    if (l == NULL) {
        snprintf(err, n, "mm3: more than %d distinct (split, len_mask, cat_mask) lists", M3_MAX_LISTS);
        return -1;
    }
    c->list = l->idx;
    c->n_list = l->n;
    if (c->n_list == 0) {
        snprintf(err, n, "mm3: no market in split %d with len_mask %d and cat_mask %d", c->split, mask, cmask);
        return -1;
    }
    if (c->fixed_market >= 0 && (c->fixed_market >= M3_DATA.n_markets
            || M3_DATA.markets[c->fixed_market].split != c->split)) {
        snprintf(err, n, "mm3: fixed_market %d is not a market of split %d", c->fixed_market, c->split);
        return -1;
    }
    return 0;
}

// ------------------------------------------------------------------ episodes

static inline void m3_begin(Mm3* m, int mk, int off) {
    const Mm3Market* M = &M3_DATA.markets[mk];
    const Mm3Ext* E = &M3_DATA.ext[mk];
    const Mm3Cfg* c = m->cfg;
    m->market = mk;
    m->r0 = (int)M->row_offset;
    m->r_end = m->r0 + M->n_rows - 1;
    if (off > M->n_rows - 2) off = M->n_rows - 2;
    if (off < 0) off = 0;
    m->s0 = m->t = m->r0 + off;
    int len = M->n_rows - 1;                 // = contract_len_s for every builder so far
    m->len = len;
    m->dt = c->dt_len[m3_len_class(len)];
    m->open_mod = (int)(M->open_ts % 60);
    m->len_s = len;
    m->min_size = M->min_order_size;
    int cat = E->category;
    double fr = M->taker_fee_rate, rr = M->maker_rebate_rate;
    if (E->fee_rate >= 0.0f) fr = E->fee_rate;
    if (E->rebate_rate >= 0.0f) rr = E->rebate_rate;
    if (c->fee_cat[cat] >= 0) fr = c->fee_cat[cat];
    if (c->reb_cat[cat] >= 0) rr = c->reb_cat[cat];
    m->fee_rate = c->fee_rate >= 0 ? c->fee_rate : fr;
    m->rebate_rate = c->rebate_rate >= 0 ? c->rebate_rate : rr;
    m->fee_exp = E->fee_exp >= 0.0f ? E->fee_exp : M->fee_exponent;
    m->reb_obs = m->rebate_rate * m->fee_rate * (1.0 / (0.2 * 0.07 * 0.25));
    // settlement value of one Up share at the last row
    int mode = c->settle_mode >= 0 ? c->settle_mode : E->settle_mode;
    if (mode == M3_SETTLE_WINNER && M->winner_up < 0) mode = M3_SETTLE_LAST_MARK;   // unresolved
    if (mode == M3_SETTLE_PRICE && (E->flags3 & M3_FLAG3_LEGACY)) mode = M3_SETTLE_LAST_MARK;  // no price stored
    m->settle = mode == M3_SETTLE_WINNER ? (double)M->winner_up
              : (mode == M3_SETTLE_PRICE ? E->settle_price : (double)m3_feat(m->r_end)[M3_F_MARK]);
    m->flat = mode == M3_SETTLE_WINNER ? 0.0 : c->flat_c / 100.0;
    m->row_s = E->row_s;
    m->ev_off = (double)((E->event_ts > 0 ? E->event_ts : M->end_ts) - M->open_ts);
    m->rw_v = E->rw_v_c;
    m->rw_min = E->rw_min_size;
    m->rw_rate = E->rw_rate_day;
    m->category = cat;
    m->legacy = (E->flags3 & M3_FLAG3_LEGACY) != 0;
    m->next_min = 0;
    m->last_fill_row = m->t - 1000000;
    m->ph = m->pn = 0;
    memset(m->ord, 0, sizeof(m->ord));
    m->inv = m->cash = m->rebates = m->liq = m->wealth = m->penalty = 0.0;
    m->spread = m->mo5 = m->mo30 = m->mo60 = 0.0;
    m->minute_vol = m->minute_fill = m->tx_fill = m->credit = 0.0;
    m->n_fills = m->fill_vol = m->buy_vol = m->sell_vol = m->max_inv = 0.0f;
    memset(m->vol_flag, 0, sizeof(m->vol_flag));
    memset(m->pnl_flag, 0, sizeof(m->pnl_flag));
    m->rows = m->rows_two = m->rows_any = m->steps = m->clips = m->rejects = m->ret = 0.0f;
    m->close_pen = m->flat_cost = m->rw_share = 0.0;
    m->gross = m->notional = 0.0;
    m->rw_samples = m->rw_scored = 0.0f;
}

static inline void m3_new_episode(Mm3* m) {
    int mk;
    if (m->cfg->fixed_market >= 0) {
        mk = m->cfg->fixed_market;
    } else if (m->cfg->cycle_stride > 0) {
        mk = m->cfg->list[m->cycle_pos % (uint64_t)m->cfg->n_list];
        m->cycle_pos += (uint64_t)m->cfg->cycle_stride;
    } else {
        mk = m->cfg->list[(m3_rand(m) >> 11) % (uint64_t)m->cfg->n_list];
    }
    int len = M3_DATA.markets[mk].n_rows - 1;
    int off;
    if (m->cfg->fixed_start >= 0) {
        off = m->cfg->fixed_start;
    } else {
        int span = (int)(m->cfg->start_frac * len) + 1;
        off = (int)((m3_rand(m) >> 11) % (uint64_t)span);
    }
    m3_begin(m, mk, off);
}

static int m3_init(Mm3* m, const Mm3Params* p, uint64_t seed, char* err, int n) {
    memset(m, 0, sizeof(*m));
    if (!M3_DATA_READY) {
        snprintf(err, n, "mm3: data not open");
        return -1;
    }
    Mm3Cfg c;
    if (m3_config(&c, p, err, n)) return -1;
    m->cfg = m3_intern(&c);
    if (m->cfg == NULL) {
        snprintf(err, n, "mm3: more than %d distinct configs in one process", M3_MAX_CFG);
        return -1;
    }
    m->rng = seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    m->cycle_pos = seed;
    m3_rand(m);
    return 0;
}

// ------------------------------------------------------------------ quoting and fills

static inline void m3_targets(const Mm3* m, int t, const int* act, int* bid, int* ask) {
    const Mm3Book* B = &M3_DATA.book[t];
    int tick = m3_tick(m, B);
    int bb = B->bid_px[0], ba = B->ask_px[0];
    double fvm = (double)m3_feat(t)[M3_F_P] * 1000.0;
    int a0 = act[0], a1 = act[1];
    a0 = a0 < 0 ? 0 : (a0 > 6 ? 6 : a0);
    a1 = a1 < 0 ? 0 : (a1 > 6 ? 6 : a1);
    int b = 0, a = 0;
    if (a0) {
        int base = bb > 0 ? bb : (int)floor(fvm / tick) * tick;
        b = base + M3_OFFS[a0] * tick;
        if (ba < 1000 && b >= ba) b = ba - tick;
        if (b < tick) b = 0;
    }
    if (a1) {
        int base = ba < 1000 ? ba : (int)ceil(fvm / tick) * tick;
        a = base - M3_OFFS[a1] * tick;
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

// Order size of size head k at match row tau.
static inline double m3_order_size(const Mm3* m, int k, int tau) {
    if (m->cfg->size_mode == 0) return m->cfg->size_mult * m->min_size * M3_SIZES[k];
    double base = m3_rw_min(m, tau);
    if (!(base > m->min_size)) base = m->min_size;
    return m->cfg->size_mult * base * m->cfg->size_rel[k];
}

static inline void m3_activate(Mm3* m, int side, int px, int size_idx, int tau) {
    Mm3Order* o = &m->ord[side];
    if (px == 0) {
        o->live = 0;
        return;
    }
    if (o->live && o->px == px) return;
    const Mm3Book* B = &M3_DATA.book[tau];
    int cross = side == 0 ? (B->ask_px[0] < 1000 && px >= B->ask_px[0])
                          : (B->bid_px[0] > 0 && px <= B->bid_px[0]);
    if (cross) {
        o->live = 0;
        m->rejects += 1.0f;
        return;
    }
    double size = m3_order_size(m, size_idx, tau);
    double room = side == 0 ? m->cfg->Q - m->inv : m->cfg->Q + m->inv;
    if (size > room) {
        size = room;
        if (size < m->min_size) {
            o->live = 0;
            return;
        }
    }
    double d = m3_level(B, side, px);
    o->px = px;
    o->rem = size;
    o->q = (m->cfg->model == M3_OPT || d < 0.0) ? 0.0 : d;
    o->disp = d;
    o->live = 1;
    o->born = tau;
}

// Book mark for attribution; the settlement value past the end of the episode.
static inline double m3_mark_at(const Mm3* m, int row) {
    return row <= m->r_end ? (double)m3_feat(row)[M3_F_MARK] : m->settle;
}

static inline void m3_book_fill(Mm3* m, int side, double f, int tau, int u, const Mm3Book* B, int fpx) {
    Mm3Order* o = &m->ord[side];
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
    m->gross += f;
    m->notional += f * p;
    if (side == 0) m->buy_vol += (float)f; else m->sell_vol += (float)f;
    double ai = fabs(m->inv);
    if (ai > m->max_inv) m->max_inv = (float)ai;
    m->last_fill_row = u;
    double mid0 = m3_feat(tau)[M3_F_MARK];
    double mid60 = m3_mark_at(m, tau + 60);
    m->spread += sq * (mid0 - p);
    m->mo5 += sq * (m3_mark_at(m, tau + 5) - mid0);
    m->mo30 += sq * (m3_mark_at(m, tau + 30) - mid0);
    m->mo60 += sq * (mid60 - mid0);
    if (m->cfg->reward_mode == 2) {
        int rh = tau + m->cfg->markout_h <= m->r_end ? tau + m->cfg->markout_h : m->r_end;
        m->credit += sq * ((double)m3_feat(rh)[M3_F_P] - p);
    }
    int tick = m3_tick(m, B), bb = B->bid_px[0], ba = B->ask_px[0];
    int inside = side == 0 ? (bb > 0 && o->px > bb + 2 * tick) : (ba < 1000 && o->px < ba - 2 * tick);
    int hit[M3_NFLAGS];
    hit[M3_FL_CROSSED] = (B->flags & M3_BOOK_CROSSED) != 0;
    hit[M3_FL_STALE] = (B->flags & M3_BOOK_STALE) != 0;
    hit[M3_FL_ONESIDED] = bb == 0 || ba == 1000;
    hit[M3_FL_LAST30] = m->r_end - tau < 30;
    hit[M3_FL_LAST10] = m->r_end - tau < 10;
    hit[M3_FL_INSIDE2] = inside;
    hit[M3_FL_EXTREME] = o->px <= 20 || o->px >= 980;
    hit[M3_FL_OUTSIDE] = fpx < bb || fpx > ba;
    hit[M3_FL_INSPREAD] = side == 0 ? fpx > bb : fpx < ba;
    float ff = (float)f, pf = (float)(sq * (mid60 - p));
    for (int k = 0; k < M3_NFLAGS; k++) {
        m->vol_flag[k] += hit[k] ? ff : 0.0f;
        m->pnl_flag[k] += hit[k] ? pf : 0.0f;
    }
}

// Polymarket's per-share score of a quote d cents from the adjusted mid, max spread v.
static inline double m3_score(double d, double v) {
    return d < v ? (v - d) / v * ((v - d) / v) : 0.0;
}

// Our Q_min at row `row` from the live orders (score uses `mid` and v at that row). The min-size
// rule applies to the remaining size, the min-age rule to the match row the order went live.
static inline double m3_qmin(const Mm3* m, int row, double mid, double v, double minsz) {
    double qs[2] = {0.0, 0.0};
    for (int s = 0; s < 2; s++) {
        const Mm3Order* o = &m->ord[s];
        if (!o->live || o->rem < minsz || row - o->born < m->cfg->liq_moas) continue;
        double dist = fabs(o->px / 1000.0 - mid) * 100.0;
        if (dist < v) qs[s] = (v - dist) / v * ((v - dist) / v) * o->rem;
    }
    double lo = qs[0] < qs[1] ? qs[0] : qs[1], hi = qs[0] < qs[1] ? qs[1] : qs[0];
    return (mid >= M3_BAND_LO && mid <= M3_BAND_HI) ? (lo > hi / M3_REWARD_C ? lo : hi / M3_REWARD_C) : lo;
}

// Liquidity-reward hook at match row tau: on a sample row (liq_pi > 0) with a scorable book
// (liq_mid >= 0), pay pi * Q_min / (Q_min + Q_comp).
static inline void m3_liquidity(Mm3* m, int tau) {
    double pi = m3_liq_pi(tau), mid = m3_liq_mid(tau);
    if (pi <= 0.0 || mid < 0.0) return;
    double v = m3_rw_v(m, tau);
    if (!(v > 0.0)) return;
    m->rw_samples += 1.0f;
    double qmin = m3_qmin(m, tau, mid, v, m3_rw_min(m, tau));
    if (qmin > 0.0) {
        double comp = m3_liq_comp(tau);
        m->liq += pi * qmin / (qmin + comp);
        m->rw_scored += 1.0f;
        m->rw_share += qmin / (qmin + comp);
    }
}

static inline void m3_row(Mm3* m, int u) {
    const Mm3Cfg* c = m->cfg;
    int tau = u - c->D;
    while (m->pn > 0 && m->pend[m->ph].row <= tau) {
        const Mm3Pend* e = &m->pend[m->ph];
        m3_activate(m, 0, e->bid & 2047, e->bid >> 11, tau);
        m3_activate(m, 1, e->ask, e->bid >> 11, tau);
        m->ph = (m->ph + 1) & (M3_PEND - 1);
        m->pn--;
    }
    if (u >= m->next_min) {
        m->minute_vol = 0.0;
        m->minute_fill = 0.0;
        m->next_min = u + 60 - (m->open_mod + u - m->r0) % 60;
    }
    int live0 = m->ord[0].live, live1 = m->ord[1].live;
    m->rows += 1.0f;
    m->rows_two += (float)(live0 & live1);
    m->rows_any += (float)(live0 | live1);
    int lo = M3_DATA.fill_off[u], hi = M3_DATA.fill_off[u + 1];
    if (!(live0 | live1)) {
        for (int j = lo; j < hi; j++) m->minute_vol += M3_DATA.fills[j].size;
        return;
    }
    const Mm3Book* B = &M3_DATA.book[tau];
    if (c->liq_on) m3_liquidity(m, tau);
    double traded[2] = {0.0, 0.0};
    int allowed = 1;
    if (lo < hi && c->model != M3_OPT) {
        allowed = !(B->flags & M3_BOOK_CROSSED) && !(c->model == M3_PESS && (B->flags & M3_BOOK_STALE));
    }
    for (int j = lo; j < hi; j++) {
        const Mm3Fill* f = &M3_DATA.fills[j];
        double sz = f->size;
        m->minute_vol += sz;
        if (f->first) m->tx_fill = 0.0;
        int side = f->side < 0 ? 0 : 1;
        Mm3Order* o = &m->ord[side];
        if (!o->live) continue;
        int at = f->px == o->px;
        int through = side == 0 ? f->px < o->px : f->px > o->px;
        if (!(at | through)) continue;
        double avail;
        if (c->model == M3_OPT) {
            avail = sz;
        } else if (at) {
            traded[side] += sz;
            double eat = o->q < sz ? o->q : sz;
            o->q -= eat;
            avail = c->model == M3_CENTRAL ? sz - eat : 0.0;
        } else if (c->model == M3_CENTRAL) {
            avail = sz;
            o->q = 0.0;
        } else {
            double eat = o->q < sz ? o->q : sz;
            o->q -= eat;
            avail = sz - eat;
        }
        double fill = c->model == M3_PESS ? 0.5 * avail : avail;
        if (c->model == M3_PESS) {
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
        if (fill > 1e-12) m3_book_fill(m, side, fill, tau, u, B, f->px);
    }
    if (c->model == M3_CENTRAL && !(B->flags & M3_BOOK_SAME)) {
        for (int s = 0; s < 2; s++) {
            Mm3Order* o = &m->ord[s];
            if (!o->live || o->q <= 0.0) continue;
            double d = m3_level(B, s, o->px);
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

static inline int m3_quiet(const Mm3* m, int a, int b) {
    const Mm3Cfg* c = m->cfg;
    if (M3_DATA.fill_off[a] != M3_DATA.fill_off[b + 1] || c->liq_on) return 0;
    if (c->model == M3_CENTRAL && ((m->ord[0].live && m->ord[0].q > 0.0) || (m->ord[1].live && m->ord[1].q > 0.0))
            && M3_DATA.book[a - c->D].same_run < b - a + 1) {
        return 0;
    }
    return 1;
}

static inline void m3_skip(Mm3* m, int a, int b) {
    float k = (float)(b - a + 1);
    int live0 = m->ord[0].live, live1 = m->ord[1].live;
    m->rows += k;
    m->rows_two += k * (float)(live0 & live1);
    m->rows_any += k * (float)(live0 | live1);
    if (b >= m->next_min) {
        m->minute_vol = 0.0;
        m->minute_fill = 0.0;
        m->next_min = b + 60 - (m->open_mod + b - m->r0) % 60;
    }
}

static inline void m3_rows(Mm3* m, int t, int tn) {
    int u = t + 1;
    while (u <= tn) {
        int stop = tn;
        if (m->pn > 0) {
            int au = m->pend[m->ph].row + m->cfg->D;
            if (au - 1 < stop) stop = au - 1;
        }
        if (stop >= u && m3_quiet(m, u, stop)) {
            m3_skip(m, u, stop);
            u = stop + 1;
            if (u > tn) break;
        }
        m3_row(m, u);
        u++;
    }
}

// ------------------------------------------------------------------ observation

// Price (mils, 0 = no quote) of price head a on one side at decision row t, with the other side off
// (m3_targets without the bid < ask rule).
static inline int m3_head_px(const Mm3Book* B, int tick, double fvm, int side, int a) {
    int bb = B->bid_px[0], ba = B->ask_px[0];
    if (side == 0) {
        int base = bb > 0 ? bb : (int)floor(fvm / tick) * tick;
        int b = base + M3_OFFS[a] * tick;
        if (ba < 1000 && b >= ba) b = ba - tick;
        return b < tick ? 0 : b;
    }
    int base = ba < 1000 ? ba : (int)ceil(fvm / tick) * tick;
    int x = base - M3_OFFS[a] * tick;
    if (bb > 0 && x <= bb) x = bb + tick;
    return x > 1000 - tick ? 0 : x;
}

// Extra observation slots (obs 50.. ; ENV3.md section 5). Every input is at row <= t.
static inline void m3_write_extra(const Mm3* m, float* e) {
    const Mm3Cfg* c = m->cfg;
    int t = m->t;
    double sec = m3_to_event(m, t);
    e[0] = (float)m3_clip(log1p(sec > 0.0 ? sec : 0.0) / log1p(86400.0), 0.0, 5.0);
    e[1] = (float)((double)(m->r_end - t) / (double)m->len);
    e[2] = (float)(m->category == M3_CAT_CRYPTO);
    e[3] = (float)(m->category == M3_CAT_SPORTS);
    e[4] = (float)(m->category == M3_CAT_ELECTIONS);
    for (int k = 5; k < M3_NX; k++) e[k] = 0.0f;
    if (!c->liq_on || !m3_in_program(m, t)) return;
    double v = m3_rw_v(m, t), minsz = m3_rw_min(m, t), mid = m3_liq_mid(t), comp = m3_liq_comp(t);
    double rate = m3_rw_rate(m, t);
    e[5] = 1.0f;
    e[6] = (float)m3_clip(log1p(rate > 0.0 ? rate : 0.0) / 5.0, 0.0, 5.0);
    e[7] = (float)m3_clip(v / 5.0, 0.0, 5.0);
    e[8] = (float)m3_clip(log1p(minsz > 0.0 ? minsz / m->min_size : 0.0) / 3.0, 0.0, 5.0);
    int scorable = mid >= 0.0 && v > 0.0;
    e[9] = (float)scorable;
    e[10] = (float)(scorable && (mid < M3_BAND_LO || mid > M3_BAND_HI));
    e[11] = (float)m3_clip(log1p(comp > 0.0 ? comp : 0.0) / 10.0, 0.0, 5.0);
    for (int k = 0; k < 3; k++) e[12 + k] = (float)(m3_order_size(m, k, t) >= minsz - 1e-9);
    if (!scorable) return;
    double q = m3_qmin(m, t, mid, v, minsz);
    e[15] = q > 0.0 ? (float)(q / (q + comp)) : 0.0f;
    const Mm3Book* B = &M3_DATA.book[t];
    int tick = m3_tick(m, B);
    double fvm = (double)m3_feat(t)[M3_F_P] * 1000.0;
    double M = minsz > m->min_size ? minsz : m->min_size;     // the smallest eligible size
    for (int s = 0; s < 2; s++) {
        for (int a = 1; a <= 6; a++) {
            int px = m3_head_px(B, tick, fvm, s, a);
            double sc = px ? m3_score(fabs(px / 1000.0 - mid) * 100.0, v) * M : 0.0;
            e[16 + 6 * s + a - 1] = sc > 0.0 ? (float)(sc / (sc + comp)) : 0.0f;
        }
    }
}

static inline void m3_write_obs(const Mm3* m, float* obs) {
    int t = m->t;
    const float* F = m3_feat(t);
    memcpy(obs, F, M3_FEAT_OBS * sizeof(float));
    double Q = m->cfg->Q, mark = m3_mark(m, t), midb = F[M3_F_MARK];
    double wealth = m->cash + m->rebates + m->liq + m->inv * mark;
    float* o = obs + M3_FEAT_OBS;
    o[0] = (float)(m->inv / Q);
    o[1] = (float)m3_clip(wealth / (0.05 * Q), -5.0, 5.0);
    o[2] = (float)m3_clip(m->inv * F[M3_F_DUSD] / (200.0 * Q), -5.0, 5.0);
    for (int s = 0; s < 2; s++) {
        const Mm3Order* q = &m->ord[s];
        double p = q->px / 1000.0;
        o[3 + s] = (float)q->live;
        o[5 + s] = q->live ? (float)m3_clip((p - midb) * 100.0 / 2.0, -5.0, 5.0) : 0.0f;
        o[7 + s] = (q->live && q->disp > 0.0) ? (float)m3_clip(q->q / q->disp, 0.0, 1.0) : 0.0f;
        o[9 + s] = q->live ? (float)m3_clip(m->reb_obs * p * (1.0 - p), 0.0, 5.0) : 0.0f;
    }
    int since = t - m->last_fill_row;
    o[11] = M3_LOG_SINCE[since < M3_SINCE_MAX ? since : M3_SINCE_MAX];
    int pending = 0;
    if (m->pn > 0) {
        const Mm3Pend* e = &m->pend[(m->ph + m->pn - 1) & (M3_PEND - 1)];
        pending = (e->bid & 2047) != (m->ord[0].live ? m->ord[0].px : 0)
            || e->ask != (m->ord[1].live ? m->ord[1].px : 0);
    }
    o[12] = (float)pending;
    int nf = (int)m->n_fills;
    o[13] = nf <= M3_NFILL_TAB ? M3_LOG_FILLS[nf] : (float)(log1p((double)nf) / 5.0);
    o[14] = F[M3_F_BIDP];
    o[15] = F[M3_F_ASKP];
    o[16] = 1.0f;
    int w = m->cfg->obs_size;
    if (w > M3_OBS_V2) {
        float ex[M3_NX];
        m3_write_extra(m, ex);
        int k = w - M3_OBS_V2 < M3_NX ? w - M3_OBS_V2 : M3_NX;
        memcpy(obs + M3_OBS_V2, ex, (size_t)k * sizeof(float));
        for (int i = M3_OBS_V2 + k; i < w; i++) obs[i] = 0.0f;
    }
}

static inline void m3_reset(Mm3* m, float* obs) {
    m3_new_episode(m);
    m3_write_obs(m, obs);
}

#if defined(__GNUC__) || defined(__clang__)
#define M3_PREFETCH(p) __builtin_prefetch(p)
#else
#define M3_PREFETCH(p) ((void)(p))
#endif

static inline void m3_prefetch_state(const Mm3* m) {
    const char* p = (const char*)m;
    for (size_t b = 0; b < sizeof(Mm3); b += 64) M3_PREFETCH(p + b);
}

static inline void m3_prefetch_data(const Mm3* m) {
    if (m->cfg == NULL) return;
    int t = m->t;
    int tn = t + m->dt <= m->r_end ? t + m->dt : m->r_end;
    int tau = t + 1 - m->cfg->D;
    tau = tau > 0 ? tau : 0;
    M3_PREFETCH(&M3_DATA.book[t]);
    for (int r = tau; r <= tn - m->cfg->D; r++) M3_PREFETCH(&M3_DATA.book[r]);
    M3_PREFETCH(&M3_DATA.fill_off[t + 1]);
    M3_PREFETCH(&M3_DATA.fill_off[tn + 1]);
    const char* f = (const char*)m3_feat(tn);
    M3_PREFETCH(f);
    M3_PREFETCH(f + 64);
    M3_PREFETCH(f + 128);
    M3_PREFETCH(f + M3_G * sizeof(float) - 1);
    M3_PREFETCH((const char*)m3_feat(t) + M3_F_P * sizeof(float));
}

static inline void m3_prefetch_legs(const Mm3* m) {
    if (m->cfg == NULL) return;
    int t = m->t;
    int tn = t + m->dt <= m->r_end ? t + m->dt : m->r_end;
    int lo = M3_DATA.fill_off[t + 1], hi = M3_DATA.fill_off[tn + 1];
    if (lo < hi) {
        M3_PREFETCH(&M3_DATA.fills[lo]);
        M3_PREFETCH(&M3_DATA.fills[hi - 1]);
    }
}

static inline void m3_log_episode(Mm3* m, Mm3Log* log, double pnl) {
    double trading = m->cash + m->inv * m->settle - m->flat_cost;
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
    log->reward_clip_frac += m->steps > 0 ? m->clips / m->steps : 0.0f;
    log->rejects += m->rejects;
    float* fvol = &log->vol_crossed;
    float* fpnl = &log->pnl_crossed;
    for (int k = 0; k < M3_NFLAGS; k++) {
        fvol[k] += m->vol_flag[k];
        fpnl[k] += m->pnl_flag[k];
    }
    log->close_penalty += (float)m->close_pen;
    log->flat_cost += (float)m->flat_cost;
    log->settle_px += (float)m->settle;
    log->rw_samples += m->rw_samples;
    log->rw_scored += m->rw_scored;
    log->rw_share += (float)m->rw_share;
    log->episode_rows += (float)(m->r_end - m->s0);
    double end_inv = fabs(m->inv);
    log->net_gross_sh += (float)m->gross;
    log->net_notional += (float)m->notional;
    log->net_end_inv += (float)end_inv;
    if (m->gross > 0.0) {
        double ns = 1.0 - end_inv / m->gross;
        log->net_share += (float)(ns > 0.0 ? ns : 0.0);
        log->net_eps += 1.0f;
    }
    log->net_flat_drag += (float)m->flat_cost;
    log->n += 1.0f;
}

// One decision. act = {bid level 0..6, ask level 0..6, size 0..2}. Writes the next observation,
// the reward and the terminal flag every step; at the episode's last row it settles, logs, starts
// the next episode and writes its first observation (auto-reset).
static inline void m3_step(Mm3* m, const int* act, float* obs, float* rew, float* term, Mm3Log* log) {
    const Mm3Cfg* c = m->cfg;
    int t = m->t;
    int bid, ask;
    m3_targets(m, t, act, &bid, &ask);
    double cpen = 0.0;
    if (c->pull_s > 0.0 || c->dt_far > 0 || c->close_inv_coef > 0.0) {
        double sec = m3_to_event(m, t);
        if (c->pull_s > 0.0 && sec <= c->pull_s) bid = ask = 0;      // quotes forced off near the event
        if (c->close_inv_coef > 0.0) {          // close-inventory shaping, on the state the action sees
            double left = c->close_ramp_s > 0.0 ? sec / c->close_ramp_s
                                                : (double)(m->r_end - t) / (double)m->len / c->close_ramp;
            if (left < 1.0) cpen = c->close_inv_coef * fabs(m->inv) / c->Q * (1.0 - (left > 0.0 ? left : 0.0));
        }
    }
    int size = act[2] < 0 ? 0 : (act[2] > 2 ? 2 : act[2]);
    Mm3Pend* e = &m->pend[(m->ph + m->pn) & (M3_PEND - 1)];
    e->row = t + c->L;
    e->bid = (int16_t)(bid | (size << 11));
    e->ask = (int16_t)ask;
    m->pn++;
    int dt = m->dt;
    if (c->dt_far > 0 && m3_to_event(m, t) > c->far_s) dt = c->dt_far;
    int tn = t + dt < m->r_end ? t + dt : m->r_end;
    double c0 = m->rebates + m->liq;
    m->credit = 0.0;
    m3_rows(m, t, tn);
    m->t = tn;
    int done = tn == m->r_end;
    double mark = done ? m->settle : m3_mark(m, tn);
    double w = m->cash + m->rebates + m->liq + m->inv * mark;
    if (done && m->flat > 0.0) {
        m->flat_cost = fabs(m->inv) * m->flat;
        w -= m->flat_cost;
    }
    double pen = c->inv_penalty * m->inv * m->inv * mark * (1.0 - mark) * (tn - t) / m->len_s;
    double r;
    if (c->reward_mode == 0) {
        r = (w - m->wealth - pen) * c->reward_scale;
    } else if (c->reward_mode == 1) {
        r = act[0] == 3 && act[1] == 3 ? 1.0 : 0.0;
    } else {
        r = (m->credit + ((m->rebates + m->liq) - c0) - pen) * c->reward_scale;
    }
    if (cpen > 0.0) {
        r -= cpen;
        m->close_pen += cpen;
    }
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
        m3_log_episode(m, log, w);
        m3_new_episode(m);
        *term = 1.0f;
    } else {
        *term = 0.0f;
    }
    m3_write_obs(m, obs);
}

#endif
