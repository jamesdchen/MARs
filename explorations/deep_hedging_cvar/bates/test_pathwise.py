"""Quick CPU checks for the pathwise baseline (policy.py, train_pathwise.py).

1. The policy keeps the PufferLib contract: forward_eval returns (Normal,
   value), the Normal's mean is mean_action, and the value is 0 on the
   post-expiry observation.
2. Every gate mode trains a few iterations on the full book; each checkpoint
   reloads into HedgePolicy, and market.simulate on the monitoring paths
   reproduces the logged exact-model CVaR, w and trade counts.
3. The hybrid estimator is unbiased. On a 2-date book (4 binary decisions on
   each path, 16 sequences) the expected RU objective given the market
   noise, E[mean_i c_i | noise], is computed exactly by enumerating the
   decision sequences, and autograd differentiates it. The hybrid surrogate's
   gradient, averaged over many sampled decisions on the same noise (common
   random numbers), must match it within Monte Carlo error, for every
   network weight and for w. Dropping the score term must not (a check that
   the test can detect bias).

    python test_pathwise.py
"""

import itertools
import math
import os
import tempfile

import torch

from market import BatesConfig, OBS_DIM, ACT_DIM, market_noise, simulate, cvar
from policy import HedgePolicy
from train_pathwise import (train, SampledGates, hybrid_surrogate, monitor_noise,
                            VAL_SEED, TEST_SEED)


def test_policy_contract():
    torch.manual_seed(0)
    pol = HedgePolicy(hidden=16)
    obs = torch.randn(6, OBS_DIM)
    obs[:, 0] = torch.tensor([0.0, 0.5, 29 / 30, 1.0, 1.0, 0.2])
    dist, value = pol.forward_eval(obs, None)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (6, ACT_DIM) and value.shape == (6, 1)
    assert torch.equal(dist.mean, pol.mean_action(obs))
    assert torch.allclose(dist.mean[0], torch.tensor([0.3, 0.25, 0.0, 0.25]))
    assert pol.logstd.shape == (1, ACT_DIM)
    assert torch.allclose(dist.stddev[0], pol.logstd.exp()[0])
    assert (value[3:5] == 0).all() and (value[[0, 1, 2, 5]] != 0).all()
    print('policy contract: ok')


def test_train_and_reload():
    cfg = BatesConfig()
    a = f'CVaR{cfg.alpha:g}'
    for gate in ['hard', 'ste', 'sigmoid', 'hybrid']:
        ckpt = train(gate, iters=3, batch=128, hidden=16, seed=3, log=None)
        assert set(ckpt) == {'actor', 'w', 'cfg', 'gate', 'history', 'wall_time', 'device',
                             'steps', 'hidden', 'args'}
        last = ckpt['history'][-1]
        assert last['iter'] == 2 and all(math.isfinite(v) for v in last.values())
        with tempfile.TemporaryDirectory() as d:
            torch.save(ckpt, os.path.join(d, 'ckpt.pt'))
            ck = torch.load(os.path.join(d, 'ckpt.pt'), weights_only=False)
        pol = HedgePolicy(hidden=ck['hidden'])
        pol.actor.load_state_dict(ck['actor'])
        ccfg = BatesConfig(**ck['cfg'])
        with torch.no_grad():
            loss, rec = simulate(monitor_noise(ck['args']['seed'], ccfg), pol.mean_action,
                                 ck['w'], ccfg, record=True)
        err = abs(cvar(loss, ccfg.alpha) - last[a])
        trades = rec['trade_s'].float().sum(1).mean().item(), \
            rec['trade_y'].float().sum(1).mean().item()
        print(f'{gate}: logged exact CVaR {last[a]:.5f}, reproduced to {err:.1e}; '
              f'trades {trades[0]:.2f} / {trades[1]:.2f}')
        assert err < 1e-4 and ck['w'] == last['w']
        assert abs(trades[0] - last['trades_stock']) < 1e-6
        assert abs(trades[1] - last['trades_swap']) < 1e-6
    for seed in (VAL_SEED, TEST_SEED):
        try:
            train('ste', iters=1, batch=8, seed=seed, log=None)
        except ValueError:
            continue
        raise AssertionError(f'seed {seed} was accepted')


def exact_expected_objective(noise, policy, w, cfg, temp):
    """E[mean_i c_i | noise] for decisions g ~ Bernoulli(sigmoid(signal / temp)),
    summing over every decision sequence its probability times its cost."""
    total = 0.0
    for seq in itertools.product([0.0, 1.0], repeat=2 * cfg.n_steps):
        prob = [1.0]

        def gates(act, k):
            out = []
            for j, col in enumerate((1, 3)):
                p = torch.sigmoid(act[:, col] / temp)
                g = seq[2 * k + j]
                prob[0] = prob[0] * (p if g else 1 - p)
                out.append(torch.full_like(p, g))
            return tuple(out)

        loss = simulate(noise, policy.mean_action, w, cfg, gates=gates)
        c = w + (loss - w).clamp_min(0) / (1 - cfg.alpha)
        total = total + (prob[0] * c).mean()
    return total


def flat_grad(value, params, retain_graph=False):
    grads = torch.autograd.grad(value, params, retain_graph=retain_graph)
    return torch.cat([g.reshape(-1) for g in grads])


def test_hybrid_unbiased(n_paths=16, reps=4096, groups=40, temp=0.5):
    cfg = BatesConfig(n_steps=2, n_sub=2, alpha=0.8)
    torch.manual_seed(1)
    policy = HedgePolicy(hidden=8)
    with torch.no_grad():
        policy.actor[-1].weight.normal_(0, 0.5)
        policy.actor[-1].bias.copy_(torch.tensor([0.3, 0.0, 0.5, 0.0]))
    w = torch.nn.Parameter(torch.tensor(0.5))
    params = list(policy.actor.parameters()) + [w]
    gen = torch.Generator().manual_seed(2)
    noise = market_noise(n_paths, cfg, gen)
    exact = flat_grad(exact_expected_objective(noise, policy, w, cfg, temp), params)

    rep = noise.repeat(reps, 1, 1, 1)
    hybrid, pathwise = [], []
    for _ in range(groups):
        gates = SampledGates(torch.rand(rep.shape[0], cfg.n_steps, 2, generator=gen), temp)
        loss = simulate(rep, policy.mean_action, w, cfg, gates=gates)
        surrogate, obj = hybrid_surrogate(loss, gates.logp, w, cfg.alpha)
        hybrid.append(flat_grad(surrogate, params, retain_graph=True))
        pathwise.append(flat_grad(obj, params))
    hybrid, pathwise = torch.stack(hybrid), torch.stack(pathwise)

    est, se = hybrid.mean(0), hybrid.std(0) / math.sqrt(groups)
    z = (est - exact) / se
    rel = ((est - exact).norm() / exact.norm()).item()
    rel_pw = ((pathwise.mean(0) - exact).norm() / exact.norm()).item()
    rel_se = (se.norm() / exact.norm()).item()
    print(f'hybrid vs exact gradient ({exact.numel()} components, '
          f'{groups} x {rep.shape[0]} sampled paths): relative error {rel:.4f} '
          f'(Monte Carlo error {rel_se:.4f}), mean z^2 {z.pow(2).mean():.2f}, '
          f'max |z| {z.abs().max():.2f}, dw exact {exact[-1]:.4f} est {est[-1]:.4f}')
    print(f'pathwise term alone: relative error {rel_pw:.4f}')
    assert (se > 0).all()
    assert z.pow(2).mean() < 1.5 and z.abs().max() < 5 and rel < 3 * rel_se
    assert rel_pw > 10 * rel_se


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
    test_policy_contract()
    test_train_and_reload()
    test_hybrid_unbiased()
    print('ok')
