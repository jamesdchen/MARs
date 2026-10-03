/* hedge.h: the part-3 hedging env in C, with no PufferLib dependency.
 *
 * Implements market.py (simulate with hard gates) for one agent: Bates
 * dynamics on n_sub full-truncation Euler substeps between hedging dates, a
 * stock with transient price impact, proportional cost and a fixed fee, a
 * variance swap traded at its model value with a cost per contract and the
 * fee, and a short call + put book. State is double; observations float.
 * The same core runs under PufferLib 3.0 (binding.c) and PufferLib 5.0
 * (puffer5/deep_hedging.h); test_c.py checks it against market.py.
 *
 * Episode: 32 env steps so that 32-step PufferLib segments line up with
 * episodes (5.0's advantage kernel needs a multiple of 4; 3.0's also never
 * looks across segments): 30 hedging dates, then a step that keeps the
 * expiry observation and a step that starts the next episode. The expiry
 * step and the reset step raise the terminal flag: the first stops
 * bootstrapping from the expiry value, the second clears recurrent state.
 *
 * Reward: with shaping, Phi(s_{k+1}) - Phi(s_k) on every hedging date, with
 * Phi = -(Lhat - w)^+ / scale (Ng, Harada & Russell 1999); Phi at expiry is
 * the true terminal reward -(L - w)^+ / scale, so the rewards telescope to
 * it. Without shaping, only the terminal reward. Every reward is multiplied
 * by reward_scale (0.1 in the configs): PufferLib 3.0 clips rewards to
 * [-1, 1], and unscaled crash-day rewards reach about 8, which would cut off
 * the tail that CVaR is about. A positive constant factor leaves the optimal
 * policy unchanged. w is the
 * Rockafellar-Uryasev threshold; each agent moves it toward VaR_alpha(L)
 * after every episode by the Robbins-Monro step of Bardou, Frikha & Pages
 * (2009). The Log's score is -(w + (L - w)^+ / (1 - alpha)), an average
 * (unlike CVaR) that bounds CVaR from above: the hyperparameter sweep metric.
 */

#ifndef HEDGE_H
#define HEDGE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define HEDGE_DATES 30
#define HEDGE_EPISODE 32
#define HEDGE_OBS 13
#define HEDGE_ACT 4
#define HEDGE_MAX_SUB 16
#define HEDGE_MAX_DEALERS 16
#define HEDGE_ARRIVAL_MAX 8
#define HEDGE_CROWD_ROWS 3
// Draws for one hedging interval: 4 for each substep, then the crowd's rows.
#define HEDGE_NOISE ((HEDGE_MAX_SUB + HEDGE_CROWD_ROWS) * 4)

// Floats only, n last: PufferLib sums these over agents and divides by n.
typedef struct Log Log;
struct Log {
    float score;
    float perf;
    float ru;
    float loss;
    float excess;
    float w;
    float trades_stock;
    float trades_swap;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    double s0, strike_call, strike_put, maturity, n_sub;
    double v0, kappa_v, theta, xi, rho, lam, mu_j, sig_j;
    double cost, fixed_cost, kappa, half_life, vs_cost, alpha;
    double delta_low, delta_high, vs_low, vs_high;
    double premium, n_vs, k_var, w_init, w_eta, shaping, reward_scale;
    double crowd_books, dealers, dealer_band_low, dealer_band_high;
    double arrival_base, arrival_move, arrival_size, arrival_follow;
    // Derived by hedge_params_finish.
    double dt_sub, sigma0, scale, decay, jump_mean, jump_var, rho_bar, ret_sd;
    double bands[HEDGE_MAX_DEALERS];
    int subs, n_dealers;
} HedgeParams;

// Keyword names in the order of the fields above (before the derived ones).
static const char* HEDGE_PARAM_NAMES[] = {
    "s0", "strike_call", "strike_put", "maturity", "n_sub",
    "v0", "kappa_v", "theta", "xi", "rho", "lam", "mu_j", "sig_j",
    "cost", "fixed_cost", "kappa", "half_life", "vs_cost", "alpha",
    "delta_low", "delta_high", "vs_low", "vs_high",
    "premium", "n_vs", "k_var", "w_init", "w_eta", "shaping", "reward_scale",
    "crowd_books", "dealers", "dealer_band_low", "dealer_band_high",
    "arrival_base", "arrival_move", "arrival_size", "arrival_follow",
};
#define HEDGE_NUM_PARAMS (int)(sizeof(HEDGE_PARAM_NAMES) / sizeof(HEDGE_PARAM_NAMES[0]))

static inline void hedge_params_finish(HedgeParams* p) {
    p->subs = (int)p->n_sub;
    p->dt_sub = p->maturity / HEDGE_DATES / p->subs;
    p->sigma0 = sqrt(p->v0);
    p->scale = p->sigma0 * sqrt(p->maturity) * p->s0;
    p->decay = pow(0.5, 1.0 / p->half_life);
    p->jump_mean = exp(p->mu_j + 0.5 * p->sig_j * p->sig_j) - 1.0;
    p->jump_var = p->mu_j * p->mu_j + p->sig_j * p->sig_j;
    p->rho_bar = sqrt(1.0 - p->rho * p->rho);
    p->ret_sd = p->sigma0 * sqrt(p->maturity / HEDGE_DATES);
    p->n_dealers = (int)p->dealers;
    int last = p->n_dealers > 1 ? p->n_dealers - 1 : 1;
    for (int i = 0; i < p->n_dealers; i++) {
        p->bands[i] = p->dealer_band_low
            * pow(p->dealer_band_high / p->dealer_band_low, (double)i / last);
    }
}

typedef struct {
    uint64_t rng[4];
    int k;
    double x, v, impact, cash, delta, y, realized, swap, w, phi, loss, ep_return;
    double s_prev;                          // quoted price before trading, last date
    double dealer_pos[HEDGE_MAX_DEALERS];   // each dealer's stock hedge, for each book
    int trades_stock, trades_swap;
} Hedge;

// xoshiro256+ (Blackman & Vigna 2018), seeded by splitmix64 (Steele, Lea &
// Flood 2014); uniforms on (0, 1].
static inline void hedge_seed(Hedge* h, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) {
        x += 0x9E3779B97F4A7C15ull;
        uint64_t z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        h->rng[i] = z ^ (z >> 31);
    }
}

static inline double hedge_uniform(Hedge* h) {
    uint64_t* s = h->rng;
    uint64_t out = s[0] + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return ((out >> 11) + 1) * 0x1.0p-53;
}

// Noise for one hedging interval, laid out like market_noise: for each
// substep z1, z2 (normal), u (uniform), zj (normal), Box-Muller pairs; then
// HEDGE_CROWD_ROWS rows of uniforms on [0, 1) for the crowd's orders.
static inline void hedge_draw_noise(Hedge* h, const HedgeParams* p, double* noise) {
    for (int j = 0; j < p->subs; j++) {
        double r = sqrt(-2.0 * log(hedge_uniform(h)));
        double a = 6.283185307179586 * hedge_uniform(h);
        double r2 = sqrt(-2.0 * log(hedge_uniform(h)));
        noise[4 * j] = r * cos(a);
        noise[4 * j + 1] = r * sin(a);
        noise[4 * j + 2] = 1.0 - hedge_uniform(h);
        noise[4 * j + 3] = r2 * cos(6.283185307179586 * hedge_uniform(h));
    }
    for (int j = 4 * p->subs; j < 4 * (p->subs + HEDGE_CROWD_ROWS); j++) {
        noise[j] = 1.0 - hedge_uniform(h);
    }
}

// E[integrated variance over the next tau | v]: diffusion plus jumps.
static inline double hedge_expected_variance(const HedgeParams* p, double v, double tau) {
    double vp = v > 0.0 ? v : 0.0;
    return p->theta * tau + (vp - p->theta) * -expm1(-p->kappa_v * tau) / p->kappa_v
        + p->lam * p->jump_var * tau;
}

static inline double hedge_tau(int date, const HedgeParams* p) {
    return p->maturity * (1.0 - (double)date / HEDGE_DATES);
}

static inline double hedge_swap_value(const HedgeParams* p, double realized, double v, int date) {
    double rest = date < HEDGE_DATES ? hedge_expected_variance(p, v, hedge_tau(date, p)) : 0.0;
    return p->n_vs * ((realized + rest) / p->maturity - p->k_var);
}

static inline double hedge_cdf(double x) {
    return 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

static inline double hedge_payoff(const HedgeParams* p, double s) {
    return (s > p->strike_call ? s - p->strike_call : 0.0)
        + (p->strike_put > s ? p->strike_put - s : 0.0);
}

// Short book at BS with the expected average remaining variance.
static inline double hedge_bs_book(const HedgeParams* p, double s, double var_avg, double tau) {
    double sd = sqrt((var_avg > 1e-12 ? var_avg : 1e-12) * tau);
    double d1 = log(s / p->strike_call) / sd + 0.5 * sd;
    double call = s * hedge_cdf(d1) - p->strike_call * hedge_cdf(d1 - sd);
    d1 = log(s / p->strike_put) / sd + 0.5 * sd;
    double put = s * hedge_cdf(d1) - p->strike_put * hedge_cdf(d1 - sd) - s + p->strike_put;
    return call + put;
}

// Delta of the book at BS with average variance var_avg over tau.
static inline double hedge_bs_book_delta(const HedgeParams* p, double s, double var_avg,
        double tau) {
    double sd = sqrt((var_avg > 1e-12 ? var_avg : 1e-12) * tau);
    double d_call = hedge_cdf(log(s / p->strike_call) / sd + 0.5 * sd);
    double d_put = hedge_cdf(log(s / p->strike_put) / sd + 0.5 * sd);
    return d_call + d_put - 1.0;
}

// The crowd's shares on the current date after our trade (market.crowd_flow):
// dealers whose band the quoted price s_after leaves trade back to the book
// delta; then a Poisson number of random orders, mostly following the last
// quoted return. u holds the date's crowd uniforms.
static inline double hedge_crowd_flow(Hedge* h, const HedgeParams* p, double s_after,
        double s, const double* u) {
    double tau = hedge_tau(h->k, p);
    double target = hedge_bs_book_delta(p, s_after,
        hedge_expected_variance(p, h->v, tau) / tau, tau);
    double each = p->crowd_books / p->dealers;
    double sum = 0.0;
    for (int i = 0; i < p->n_dealers; i++) {
        double out = fabs(h->dealer_pos[i] - target) > p->bands[i] ? 1.0 : 0.0;
        sum += out * (target - h->dealer_pos[i]);
        h->dealer_pos[i] = h->dealer_pos[i] + out * (target - h->dealer_pos[i]);
    }
    double flow = each * sum;
    double r = log(s / h->s_prev) / p->ret_sd;
    double mean = p->arrival_base + p->arrival_move * fabs(r);
    double pk = exp(-mean);
    double cdf = pk;
    int count = 0;
    for (int j = 1; j <= HEDGE_ARRIVAL_MAX; j++) {
        count += u[0] > cdf;
        pk = pk * mean / j;
        cdf = cdf + pk;
    }
    double up = r > 0.0 ? 1.0 : (r < 0.0 ? -1.0 : 1.0);
    double orders = 0.0;
    for (int j = 1; j <= count; j++) {
        int follow = r != 0.0 ? u[j] < p->arrival_follow : u[j] < 0.5;
        orders += 2.0 * follow - 1.0;
    }
    return flow + p->arrival_size * up * orders;
}

// Cash after selling the stock position at quoted price s.
static inline double hedge_close_out(const HedgeParams* p, double cash, double d, double s) {
    double fee = d != 0.0 ? p->fixed_cost : 0.0;
    return cash + d * s - 0.5 * p->kappa * (d * d) - p->cost * fabs(d) * s - fee;
}

static inline double hedge_potential(const Hedge* h, const HedgeParams* p) {
    double s = exp(h->x) + h->impact;
    double book;
    if (h->k < HEDGE_DATES) {
        double tau = hedge_tau(h->k, p);
        book = hedge_bs_book(p, s, hedge_expected_variance(p, h->v, tau) / tau, tau);
    } else {
        book = hedge_payoff(p, s);
    }
    double excess = book - hedge_close_out(p, h->cash + h->y * h->swap, h->delta, s) - h->w;
    return excess > 0.0 ? -excess / p->scale : 0.0;
}

// market.make_obs at the current date (the expiry date on the padding step).
static inline void hedge_write_obs(const Hedge* h, const HedgeParams* p, float* obs) {
    int date = h->k < HEDGE_DATES ? h->k : HEDGE_DATES;
    double s = exp(h->x) + h->impact;
    double wealth = h->cash + h->delta * s + h->y * h->swap;
    double vp = h->v > 0.0 ? h->v : 0.0;
    obs[0] = (double)date / HEDGE_DATES;
    obs[1] = log(s / p->strike_call) / (p->sigma0 * sqrt(p->maturity));
    obs[2] = sqrt(vp) / p->sigma0;
    obs[3] = h->delta;
    obs[4] = h->y;
    obs[5] = wealth / p->scale;
    obs[6] = h->w / p->scale;
    obs[7] = (h->w + wealth) / p->scale;
    obs[8] = p->kappa > 0.0 ? h->impact / p->kappa : 0.0;
    obs[9] = h->swap / p->scale;
    obs[10] = h->realized / (p->theta * p->maturity);
    obs[11] = log(s / h->s_prev) / p->ret_sd;
    obs[12] = 1.0f;
}

static inline void hedge_start(Hedge* h, const HedgeParams* p) {
    h->k = 0;
    h->x = log(p->s0);
    h->v = p->v0;
    h->impact = 0.0;
    h->cash = p->premium;
    h->delta = 0.0;
    h->y = 0.0;
    h->realized = 0.0;
    h->swap = 0.0;
    h->trades_stock = 0;
    h->trades_swap = 0;
    h->ep_return = 0.0;
    h->s_prev = exp(h->x) + h->impact;
    double start = hedge_bs_book_delta(p, h->s_prev,
        hedge_expected_variance(p, h->v, p->maturity) / p->maturity, p->maturity);
    for (int i = 0; i < p->n_dealers; i++) {
        h->dealer_pos[i] = start;
    }
    h->phi = hedge_potential(h, p);
}

static inline void hedge_init(Hedge* h, const HedgeParams* p, uint64_t seed) {
    hedge_seed(h, seed);
    h->w = p->w_init * p->scale;
    hedge_start(h, p);
}

static inline double hedge_clamp(double a, double lo, double hi) {
    return a < lo ? lo : (a > hi ? hi : a);
}

// One env step. noise holds the interval's draws and is only read on
// hedging dates (k < HEDGE_DATES). Writes obs, *reward and *terminal.
static inline void hedge_step(Hedge* h, const HedgeParams* p, const float* act,
        const double* noise, float* obs, float* reward, float* terminal, Log* log) {
    *reward = 0.0f;
    *terminal = 0.0f;
    if (h->k == HEDGE_EPISODE - 1) {
        double excess = h->loss > h->w ? h->loss - h->w : 0.0;
        double ru = h->w + excess / (1.0 - p->alpha);
        log->score -= ru;
        log->perf -= ru / p->scale;
        log->ru += ru;
        log->loss += h->loss;
        log->excess += excess;
        log->w += h->w;
        log->trades_stock += h->trades_stock;
        log->trades_swap += h->trades_swap;
        log->episode_return += h->ep_return;
        log->episode_length += HEDGE_EPISODE;
        log->n += 1.0f;
        h->w += p->w_eta * p->scale * ((h->loss > h->w) / (1.0 - p->alpha) - 1.0);
        hedge_start(h, p);
        hedge_write_obs(h, p, obs);
        *terminal = 1.0f;
        return;
    }
    if (h->k == HEDGE_DATES) {
        h->k += 1;
        hedge_write_obs(h, p, obs);
        return;
    }

    double s = exp(h->x) + h->impact;
    double gs = act[1] > 0.0f ? 1.0 : 0.0;
    double gy = act[3] > 0.0f ? 1.0 : 0.0;
    double q = gs * (hedge_clamp(act[0], p->delta_low, p->delta_high) - h->delta);
    double pv = gy * (hedge_clamp(act[2], p->vs_low, p->vs_high) - h->y);
    h->cash = h->cash - q * s - 0.5 * p->kappa * (q * q) - p->cost * fabs(q) * s
        - p->fixed_cost * gs - pv * h->swap - p->vs_cost * fabs(pv) - p->fixed_cost * gy;
    h->impact += p->kappa * q;
    h->delta += q;
    h->y += pv;
    h->trades_stock += (int)gs;
    h->trades_swap += (int)gy;
    h->impact += p->kappa * hedge_crowd_flow(h, p, exp(h->x) + h->impact, s,
        noise + 4 * p->subs);
    h->s_prev = s;

    double x0 = h->x;
    double dt = p->dt_sub;
    for (int j = 0; j < p->subs; j++) {
        const double* z = noise + 4 * j;
        double vp = h->v > 0.0 ? h->v : 0.0;
        double sq = sqrt(vp * dt);
        double jump = z[2] < p->lam * dt ? p->mu_j + p->sig_j * z[3] : 0.0;
        h->x = h->x + (-p->lam * p->jump_mean - 0.5 * vp) * dt + sq * z[0] + jump;
        h->v = h->v + p->kappa_v * (p->theta - vp) * dt
            + p->xi * sq * (p->rho * z[0] + p->rho_bar * z[1]);
    }
    double r = h->x - x0;
    h->realized += r * r;
    h->impact *= p->decay;
    h->k += 1;
    h->swap = hedge_swap_value(p, h->realized, h->v, h->k);

    if (h->k == HEDGE_DATES) {
        s = exp(h->x) + h->impact;
        h->loss = hedge_payoff(p, s) - hedge_close_out(p, h->cash + h->y * h->swap, h->delta, s);
        *terminal = 1.0f;
    }
    double rew = 0.0;
    if (p->shaping != 0.0) {
        double phi = hedge_potential(h, p);
        rew = phi - h->phi;
        h->phi = phi;
    } else if (h->k == HEDGE_DATES) {
        double excess = h->loss - h->w;
        rew = excess > 0.0 ? -excess / p->scale : 0.0;
    }
    rew *= p->reward_scale;
    *reward = rew;
    h->ep_return += rew;
    hedge_write_obs(h, p, obs);
}

#endif
