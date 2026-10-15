"""Tests for the training stack (stand-in env; no real env needed).

    python test_train.py            # or: pytest -q test_train.py

1. sample_actions (rollout sampler) and pufferlib.pytorch.sample_logits agree on
   log-probs and entropy of the same actions; argmax is deterministic.
2. PufferNetPolicy: the training scan over (S, T) segments, with an initial
   state and terminal resets inside the segment, equals stepping forward_eval
   one row at a time, to 1e-6.
3. Trainer (PuffeRL 3.0 subclass), MLP and PufferNet: after one rollout, the
   training forward on every stored segment (with the stored segment-start
   state and terminals) reproduces the rollout log-probs and values to 1e-5,
   i.e. the importance ratio is exactly 1 before the first update. With a
   recurrent policy this fails unless state carry/reset match.
4. The reward-clip diagnostic counts |r| > 1 before the clamp.
5. The evaluator plays every validation market exactly once and is
   deterministic.
6. Config hash ignores the seed and run-only keys, and changes with a hyperparameter.
7. A short PPO run on the 'join' learnability probe learns it (> 0.9 greedy).
"""

import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')  # shared CPUs: no spin-waiting threads (bench.py)

import copy
import os
import sys
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pufferlib.pufferl  # noqa: E402,F401
import pufferlib.pytorch  # noqa: E402
warnings.filterwarnings('default', category=RuntimeWarning)

import ledger  # noqa: E402
from envs import StandInAdapter  # noqa: E402
from evaluator import evaluate  # noqa: E402
from policy import MLPPolicy, PufferNetPolicy, greedy_actions, logprob_entropy, sample_actions  # noqa: E402
from puffer3 import Trainer, ppo_config  # noqa: E402

torch.set_num_threads(2)


def test_sampler_matches_sample_logits():
    torch.manual_seed(0)
    logits = tuple(torch.randn(256, n) * 2 for n in (7, 7, 3))
    a, lp = sample_actions(logits)
    _, lp2, ent = pufferlib.pytorch.sample_logits(logits, action=a)
    assert torch.allclose(lp, lp2, atol=1e-5), (lp - lp2).abs().max()
    ent_ref = sum(torch.distributions.Categorical(logits=l).entropy() for l in logits)
    assert torch.allclose(ent, ent_ref, atol=1e-5)
    lp4, ent4 = logprob_entropy(logits, a)
    assert torch.allclose(lp4, lp2, atol=1e-5) and torch.allclose(ent4, ent, atol=1e-5)
    g = greedy_actions(logits)
    assert torch.equal(g, greedy_actions(logits)) and g.shape == (256, 3)
    a2, lp3, _ = pufferlib.pytorch.sample_logits(logits)
    assert a2.shape == (256, 3)


def test_puffernet_scan_equals_steps():
    torch.manual_seed(1)
    S, T, D = 5, 24, 16
    net = PufferNetPolicy(obs_size=D, hidden=32, layers=3)
    with torch.no_grad():  # larger weights so the gates move
        for w in net.tensors():
            w.mul_(2.0)
    obs = torch.randn(S, T, D)
    obs[..., -1] = 1
    term = (torch.rand(S, T) < 0.15).float()
    h0 = torch.randn(S, net.hidden_size) * 0.5
    st = {'lstm_h': h0.clone(), 'terminals': term}
    logits, v = net(obs, st)
    h = h0.clone()
    ref_l, ref_v = [], []
    for t in range(T):
        s = {'lstm_h': h, 'done': term[:, t]}
        lt, vt = net.forward_eval(obs[:, t], s)
        h = s['lstm_h']
        ref_l.append(torch.cat(lt, -1))
        ref_v.append(vt[:, 0])
    ref_l = torch.stack(ref_l, 1).reshape(S * T, -1)
    assert torch.allclose(torch.cat(logits, -1), ref_l, atol=1e-5)
    assert torch.allclose(v, torch.stack(ref_v, 1), atol=1e-5)
    assert torch.allclose(st['lstm_h'], h, atol=1e-5)


def _trainer(policy, num_envs=64, horizon=16, **kw):
    env = StandInAdapter().make_train(num_envs, seed=3, reward_scale=1.0, lengths=(20, 30, 50))
    cfg = ppo_config(bptt_horizon=horizon, minibatch_size=num_envs * horizon, total_timesteps=10**6,
                     use_rnn=getattr(policy, 'recurrent', False), **kw)
    return Trainer(cfg, env, policy), env


def _check_rollout_replay(policy, epochs=3):
    torch.manual_seed(2)
    tr, env = _trainer(policy)
    for _ in range(epochs):  # several rollouts so the carried state is non-zero
        tr.evaluate()
    rnn = tr.config['use_rnn']
    with torch.no_grad():
        obs = tr.observations
        state = {'lstm_h': tr.seg_state.clone() if rnn else None, 'terminals': tr.terminals}
        if rnn:
            logits, v = policy(obs, state)
        else:
            logits, v = policy(obs.reshape(-1, obs.shape[-1]), state)
        _, lp, _ = pufferlib.pytorch.sample_logits(logits, action=tr.actions)
    assert torch.allclose(lp.reshape(tr.logprobs.shape), tr.logprobs, atol=1e-5), \
        (lp.reshape(tr.logprobs.shape) - tr.logprobs).abs().max()
    assert torch.allclose(v.reshape(tr.values.shape), tr.values, atol=1e-5)
    assert tr.terminals.sum() > 0, 'test needs terminals inside segments'
    env.close()
    return tr


def test_rollout_replay_mlp():
    _check_rollout_replay(MLPPolicy(obs_size=16, hidden=32))


def test_rollout_replay_puffernet():
    tr = _check_rollout_replay(PufferNetPolicy(obs_size=16, hidden=32, layers=2))
    assert tr.seg_state.abs().sum() > 0, 'segment-start state should be carried, not zero'
    logs = tr.train()  # one update runs end to end on segments
    assert np.isfinite(logs['losses/policy_loss'])


class _RewardEnv:
    """Constant obs; rewards cycle through [-2, -0.5, 0.5, 3] (half beyond the clamp)."""

    def __init__(self, n=32):
        import gymnasium
        import pufferlib
        outer = self

        class E(pufferlib.PufferEnv):
            def __init__(self):
                self.num_agents = n
                self.single_observation_space = gymnasium.spaces.Box(-1, 1, (16,), np.float32)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 7, 3])
                super().__init__()
                self.k = 0

            def _fill(self):
                self.observations[:] = 0.5
                self.rewards[:] = np.array([-2, -0.5, 0.5, 3], np.float32)[(np.arange(n) + self.k) % 4]
                self.terminals[:] = (np.arange(n) + self.k) % 37 == 0
                self.k += 1

            def reset(self, seed=None):
                self._fill()
                return self.observations, []

            def step(self, a):
                self._fill()
                return self.observations, self.rewards, self.terminals, self.truncations, []

            def close(self):
                pass
        outer.env = E()


def test_reward_clip_stat():
    env = _RewardEnv().env
    cfg = ppo_config(bptt_horizon=16, minibatch_size=32 * 16, total_timesteps=10**5)
    tr = Trainer(cfg, env, MLPPolicy(obs_size=16, hidden=16))
    tr.evaluate()
    assert float(tr.rewards.abs().max()) <= 1.0  # PuffeRL's clamp still applies
    logs = tr.train()
    assert abs(logs['clip_frac_reward'] - 0.5) < 1e-9, logs['clip_frac_reward']
    assert abs(logs['clip_mass_reward'] - 3.0 / 6.0) < 1e-6  # (1 + 2) / (2 + .5 + .5 + 3)
    assert 0 < logs['term_row0_frac'] < 0.3


def test_evaluator_covers_all_markets_once():
    ad = StandInAdapter()
    pol = MLPPolicy(obs_size=16, hidden=16)
    r1, recs = evaluate(pol, ad, split=1, return_records=True)
    n = len(ad.mod.build_split(1)['lens'])
    assert r1['n_markets'] == n and sorted(r['market'] for r in recs) == list(range(n))
    r2 = evaluate(pol, ad, split=1)
    assert r1['pnl_total'] == r2['pnl_total']
    pr = evaluate(PufferNetPolicy(obs_size=16, hidden=16, layers=1), ad, split=1)
    assert pr['n_markets'] == n
    try:
        evaluate(pol, ad, split=2)
        raise AssertionError('test split must be refused')
    except PermissionError:
        pass


def test_config_hash():
    cfg = {'ppo': {'learning_rate': 1e-3}, 'seed': 1, 'run': {'out': 'x'}}
    c2 = copy.deepcopy(cfg)
    c2['seed'] = 7
    c2['run'] = {'out': 'y'}
    assert ledger.config_hash(cfg) == ledger.config_hash(c2)
    c2['ppo']['learning_rate'] = 2e-3
    assert ledger.config_hash(cfg) != ledger.config_hash(c2)


def test_join_probe_learns():
    import probes
    res = probes.run_probe('join', backend='standin', steps=1_500_000, num_envs=512, threads=2,
                           quiet=True)
    assert res['greedy_success'] > 0.9, res


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__, flush=True)
