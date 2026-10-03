"""Check that the Cython env (and its torch twin) match the reference model.

Runs both on the same fundamental noise and the same actions (random
targets, some outside the action bounds, and random trade signals) and
compares observations at every date, terminal losses, and the shaped
rewards, which must telescope to the terminal reward.

    python test_env.py
"""

import numpy as np
import torch

from market import ImpactConfig, simulate, make_obs
from env import ImpactHedgingEnv

torch.set_default_dtype(torch.float64)


def run(cfg, backend, n=5000, seed=0):
    env = ImpactHedgingEnv(cfg, num_agents=n, adapt_w=False, w_jitter=0.3,
                           backend=backend, seed=seed)
    env.reset()
    z, w = torch.from_numpy(env.z.copy()), torch.from_numpy(env.w.copy())
    phi0 = env.state['phi'].copy()
    rng = np.random.default_rng(seed + 1)
    actions = np.stack([rng.uniform(-0.7, 1.7, (cfg.n_steps, n)),
                        rng.normal(0, 1, (cfg.n_steps, n))], axis=-1).astype(np.float32)
    env_obs, reward_sum = [], np.zeros(n)
    for k in range(cfg.n_steps):
        env_obs.append(env.observations.copy())
        env.step(actions[k])
        reward_sum += env.rewards
    env_loss = env.loss.copy()

    ref_obs = []
    def replay(obs):
        ref_obs.append(obs.clone())
        return torch.from_numpy(actions[len(ref_obs) - 1]).double()
    ref_loss = simulate(z, replay, w, cfg)

    obs_err = max(np.abs(a - b.numpy()).max() for a, b in zip(env_obs, ref_obs))
    loss_err = np.abs(env_loss - ref_loss.numpy()).max()
    terminal = -np.maximum(env_loss - env.w, 0) / cfg.scale
    tele_err = np.abs(reward_sum + phi0 - terminal).max()
    return obs_err, loss_err, tele_err


if __name__ == '__main__':
    for backend, cfg in [(b, c) for b in ['cython', 'torch'] for c in [
            ImpactConfig(), ImpactConfig(kappa=0.0, fixed_cost=0.0),
            ImpactConfig(kappa=2.0, fixed_cost=0.1, half_life=3.0)]]:
        obs_err, loss_err, tele_err = run(cfg, backend)
        print(f'{backend} kappa={cfg.kappa} f={cfg.fixed_cost}: max obs err {obs_err:.2e}, '
              f'max loss err {loss_err:.2e}, telescoping err {tele_err:.2e}')
        # Observations are float32; losses are float64 on both sides.
        assert obs_err < 1e-4 and loss_err < 1e-9 and tele_err < 1e-5
    print('ok')
