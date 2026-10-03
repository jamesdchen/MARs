# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled step for the impact + fixed cost hedging env.

Implements the dynamics of market.py (see there for the model) on all
agents in one C loop, reading actions from and writing observations and
rewards into PufferLib's shared numpy buffers. The Python wrapper (env.py)
owns every array and handles episode starts, logging and the threshold w.
"""

from libc.math cimport exp, log, sqrt, erf, fabs


cdef inline double norm_cdf(double x) noexcept nogil:
    return 0.5 * (1.0 + erf(x * 0.7071067811865476))


cdef class CyHedging:
    cdef public int k
    cdef int n, n_steps
    cdef bint shaping
    cdef double s0, strike, sigma, maturity, cost, fixed_cost, kappa, decay
    cdef double premium, scale, drift, vol, target_low, target_high
    cdef float[:, ::1] obs
    cdef float[::1] rewards
    cdef double[:, ::1] z
    cdef double[::1] w, loss, f, impact, cash, delta, phi, trades

    def __init__(self, dict cfg, float[:, ::1] obs, float[::1] rewards,
                 double[:, ::1] z, double[::1] w, double[::1] loss,
                 double[::1] f, double[::1] impact, double[::1] cash,
                 double[::1] delta, double[::1] phi, double[::1] trades,
                 bint shaping):
        self.n = obs.shape[0]
        self.n_steps = cfg['n_steps']
        self.s0 = cfg['s0']
        self.strike = cfg['strike']
        self.sigma = cfg['sigma']
        self.maturity = cfg['maturity']
        self.cost = cfg['cost']
        self.fixed_cost = cfg['fixed_cost']
        self.kappa = cfg['kappa']
        self.decay = 0.5 ** (1.0 / cfg['half_life'])
        self.premium = cfg['premium']
        self.scale = self.sigma * sqrt(self.maturity) * self.s0
        dt = self.maturity / self.n_steps
        self.drift = (cfg['mu'] - 0.5 * self.sigma * self.sigma) * dt
        self.vol = self.sigma * sqrt(dt)
        self.target_low = cfg['target_low']
        self.target_high = cfg['target_high']
        self.obs, self.rewards, self.z, self.w, self.loss = obs, rewards, z, w, loss
        self.f, self.impact, self.cash, self.delta, self.phi = f, impact, cash, delta, phi
        self.trades = trades
        self.shaping = shaping
        self.k = 0

    cdef double close_out(self, int i, double s) noexcept nogil:
        cdef double d = self.delta[i]
        cdef double fee = self.fixed_cost if d != 0.0 else 0.0
        return (self.cash[i] + d * s - 0.5 * self.kappa * d * d
                - self.cost * fabs(d) * s - fee)

    cdef double potential(self, int i) noexcept nogil:
        """-(Lhat - w)^+ / scale with Lhat the Black-Scholes close-out loss."""
        cdef double s = self.f[i] + self.impact[i]
        cdef double option, tau, sd, d1
        if self.k < self.n_steps:
            tau = self.maturity * (1.0 - <double>self.k / self.n_steps)
            sd = self.sigma * sqrt(tau)
            d1 = (log(s / self.strike) + 0.5 * sd * sd) / sd
            option = s * norm_cdf(d1) - self.strike * norm_cdf(d1 - sd)
        else:
            option = s - self.strike if s > self.strike else 0.0
        cdef double excess = option - self.close_out(i, s) - self.w[i]
        return -excess / self.scale if excess > 0.0 else 0.0

    cdef void write_obs(self, int i) noexcept nogil:
        cdef double s = self.f[i] + self.impact[i]
        cdef double wealth = self.cash[i] + self.delta[i] * s
        self.obs[i, 0] = <double>self.k / self.n_steps
        self.obs[i, 1] = log(s / self.strike) / (self.sigma * sqrt(self.maturity))
        self.obs[i, 2] = self.delta[i]
        self.obs[i, 3] = wealth / self.scale
        self.obs[i, 4] = self.w[i] / self.scale
        self.obs[i, 5] = (self.w[i] + wealth) / self.scale
        self.obs[i, 6] = self.impact[i] / self.kappa if self.kappa > 0.0 else 0.0

    def reset_episode(self):
        cdef int i
        self.k = 0
        with nogil:
            for i in range(self.n):
                self.f[i] = self.s0
                self.impact[i] = 0.0
                self.cash[i] = self.premium
                self.delta[i] = 0.0
                self.trades[i] = 0.0
                self.phi[i] = self.potential(i)
                self.rewards[i] = 0.0
                self.write_obs(i)

    def step(self, float[:, ::1] actions):
        """One hedging date for every agent. Requires k < n_steps."""
        cdef int i
        cdef int k = self.k
        cdef double s, target, q, phi, excess
        cdef bint last = k + 1 == self.n_steps
        with nogil:
            for i in range(self.n):
                s = self.f[i] + self.impact[i]
                if actions[i, 1] > 0.0:
                    target = actions[i, 0]
                    if target < self.target_low:
                        target = self.target_low
                    elif target > self.target_high:
                        target = self.target_high
                    q = target - self.delta[i]
                    self.cash[i] -= (q * s + 0.5 * self.kappa * q * q
                                     + self.cost * fabs(q) * s + self.fixed_cost)
                    self.impact[i] += self.kappa * q
                    self.delta[i] += q
                    self.trades[i] += 1.0
                self.f[i] *= exp(self.drift + self.vol * self.z[i, k])
                self.impact[i] *= self.decay
            self.k = k + 1
            for i in range(self.n):
                if last:
                    s = self.f[i] + self.impact[i]
                    self.loss[i] = ((s - self.strike if s > self.strike else 0.0)
                                    - self.close_out(i, s))
                if self.shaping:
                    phi = self.potential(i)
                    self.rewards[i] = phi - self.phi[i]
                    self.phi[i] = phi
                elif last:
                    excess = self.loss[i] - self.w[i]
                    self.rewards[i] = -excess / self.scale if excess > 0.0 else 0.0
                else:
                    self.rewards[i] = 0.0
                self.write_obs(i)
