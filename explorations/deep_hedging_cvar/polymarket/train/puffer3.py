"""PuffeRL (PufferLib 3.0) with the few changes this project needs.

`Trainer` subclasses `pufferlib.pufferl.PuffeRL` and keeps its PPO exactly:
same rollout buffers, same [-1, 1] reward clamp, same per-segment
puff_advantage (V-trace clipped GAE, recomputed with the current ratios at every
minibatch), same clipped policy/value losses, Adam/Muon, cosine LR annealing.
`evaluate()` and `train()` are copies of 3.0's with these changes:

1. Recurrent policies (`use_rnn`): the per-agent state is carried across
   epochs and reset only on terminals (inside the policy, via `state['done']`),
   the state before each segment's row 0 is stored, and `train()` passes it and
   the segment's terminal flags to `policy.forward`. Stock 3.0 instead zeroes
   the state at every `evaluate()` and never resets it inside a training
   segment. The new behaviour is 5.0's default (`reset_every_horizon = 0`);
   `reset_every_horizon=True` in the config restores zeroing at every epoch.
2. Diagnostics computed in the rollout: the fraction of raw rewards at or
   beyond +-1 (`clip_frac_reward`, before the clamp; the env clips at exactly
   +-1 itself, so "at" counts), the share of the clamped
   reward mass (`clip_mass_reward`), and the share of terminals that land on a
   segment's row 0 (`term_row0_frac`; puff_advantage never reads row 0's reward,
   so this should be about 1/horizon, not 1).
3. `train()` returns a log dict every epoch: losses of this epoch, means of the
   env's Log infos, the diagnostics, and steps/s of the rollout, of the update
   and overall. No dashboard, no utilization thread, no 3.0 checkpointing
   (train_ppo.py saves its own).
4. Diagnostics per epoch: entropy of each head (training minibatches),
   action histograms per head in the rollout, mapped to categories by
   `action_labels` (e.g. off/improve/join/behind), and raw reward mean/std per
   group (e.g. contract length) when `group_fn` is given: a callable returning
   (B,) int labels of the market each agent is in, read before the action is
   sent so that the reward of that action is credited to its own market even
   when the env auto-resets on the step.
5. Optional kickstarting (Schmitt et al. 2018): with `teacher` (a frozen
   policy, e.g. the behaviour-cloned warm start) and `kl_teacher_coef` > 0,
   the loss adds coef * KL(teacher || policy) summed over the heads, with coef
   decaying linearly to 0 over the first `kl_teacher_decay` fraction of the
   epochs. A recurrent teacher runs each segment from a zero state, with the
   segment's terminal resets.
6. Optional fast sampler for the rollout (`fast_sample`): per-head
   `torch.multinomial` on softmax, the same distribution and log-prob convention
   as `pufferlib.pytorch.sample_logits` (which pads heads to (3, B, 7) with -inf
   first). Training scores multi-discrete actions with `policy.logprob_entropy`,
   which returns sample_logits' values per head without the padding (3x
   faster; test_train.py checks they agree).
"""

import time
from collections import defaultdict

import numpy as np
import torch

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.pytorch

from policy import logprob_entropy, sample_actions


def ppo_config(**overrides):
    """PuffeRL 3.0 train config with this project's defaults (Adam, plain PPO)."""
    cfg = dict(
        env='polymarket_mm', seed=0, torch_deterministic=True, device='cpu', cpu_offload=False,
        use_rnn=False, compile=False, compile_mode='default', compile_fullgraph=False,
        precision='float32', optimizer='adam', learning_rate=3e-4, anneal_lr=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, total_timesteps=10_000_000,
        batch_size='auto', bptt_horizon=64, minibatch_size=16384, max_minibatch_size=1 << 30,
        update_epochs=2, gamma=0.995, gae_lambda=0.95, clip_coef=0.2, vf_coef=0.5,
        vf_clip_coef=10.0, ent_coef=0.01, max_grad_norm=0.5, vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0, prio_alpha=0.0, prio_beta0=1.0, data_dir='runs',
        checkpoint_interval=1 << 30, reset_every_horizon=False, fast_sample=True,
        kl_teacher_coef=0.0, kl_teacher_decay=0.5, compile_eval=False,
    )
    cfg.update(overrides)
    return cfg


class Trainer(pufferl.PuffeRL):
    def __init__(self, config, vecenv, policy, logger=None, teacher=None, action_labels=None,
                 group_fn=None, group_names=None):
        if config['use_rnn'] and not hasattr(vecenv, 'agents_per_batch'):
            vecenv.agents_per_batch = vecenv.num_agents  # PufferEnv spells it agent_per_batch
        super().__init__(config, vecenv, policy, logger)
        self.utilization.stop()
        if config['use_rnn']:
            self.seg_state = torch.zeros(self.segments, policy.hidden_size, device=config['device'])
        self.nvec = tuple(int(n) for n in vecenv.single_action_space.nvec)
        self.action_labels = action_labels
        self.group_fn, self.group_names = group_fn, group_names
        self.teacher = teacher
        self.prev_groups = None
        if teacher is not None:
            teacher.requires_grad_(False)
            teacher.eval()
        self._reset_diag()
        self.t_eval = self.t_train = 0.0
        self._fwd_eval = self.uncompiled_policy.forward_eval
        if config.get('compile_eval'):
            # 3.0's `compile` wraps only policy.__call__ (the training forward);
            # this compiles the rollout forward too.
            self._fwd_eval = torch.compile(self.uncompiled_policy.forward_eval, mode=config['compile_mode'])

    def _reset_diag(self):
        self.diag = defaultdict(float)
        self.act_hist = [np.zeros(n) for n in self.nvec]
        G = len(self.group_names) if self.group_names else 0
        self.grp = np.zeros((3, G))  # count, sum, sum of squares of raw rewards

    def kl_coef(self):
        c = self.config.get('kl_teacher_coef', 0.0)
        if self.teacher is None or c <= 0:
            return 0.0
        frac = self.epoch / max(1, self.total_epochs)
        decay = self.config.get('kl_teacher_decay', 0.5)
        return c * max(0.0, 1.0 - frac / decay) if decay > 0 else c

    def print_dashboard(self, *a, **k):
        pass

    def save_checkpoint(self):
        return None

    @torch.no_grad()
    def _act(self, o, state):
        logits, value = self._fwd_eval(o, state)
        if self.config.get('fast_sample', True) and not isinstance(logits, torch.distributions.Normal):
            action, logprob = sample_actions(logits)
        else:
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
        return action, logprob, value

    def evaluate(self):
        t0 = time.perf_counter()
        config = self.config
        device = config['device']
        rnn = config['use_rnn']
        if rnn and config.get('reset_every_horizon', False):
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        diag = self.diag
        self.full_rows = 0
        while self.full_rows < self.segments:
            o, r, d, t, info, env_id, mask = self.vecenv.recv()
            env_id = slice(env_id[0], env_id[-1] + 1)
            self.global_step += int(mask.sum())

            o_np = o
            o = torch.as_tensor(o).to(device)
            r = torch.as_tensor(r).to(device)
            d = torch.as_tensor(d).to(device)
            l = self.ep_lengths[env_id.start].item()
            batch_rows = slice(self.ep_indices[env_id.start].item(),
                               1 + self.ep_indices[env_id.stop - 1].item())

            state = dict(reward=r, done=d, env_id=env_id, mask=mask)
            if rnn:
                h = self.lstm_h[env_id.start]
                if l == 0:
                    self.seg_state[batch_rows] = h
                state['lstm_h'] = h
                state['lstm_c'] = self.lstm_c[env_id.start]

            action, logprob, value = self._act(o, state)

            if self.prev_groups is not None and self.grp.shape[1]:
                g, rr = self.prev_groups, r.double().cpu().numpy()
                G = self.grp.shape[1]
                self.grp[0] += np.bincount(g, minlength=G)[:G]
                self.grp[1] += np.bincount(g, weights=rr, minlength=G)[:G]
                self.grp[2] += np.bincount(g, weights=rr * rr, minlength=G)[:G]

            with torch.no_grad():
                for h, n in enumerate(self.nvec):
                    self.act_hist[h] += torch.bincount(action[:, h].long(), minlength=n)[:n].numpy()
                absr = r.abs()
                diag['n_rew'] += r.numel()
                diag['n_clip'] += float((absr >= 1).sum())
                diag['mass'] += float(absr.sum())
                diag['mass_clipped'] += float((absr - 1).clamp(min=0).sum())
                nterm = float(d.sum())
                diag['n_term'] += nterm
                if l == 0:
                    diag['n_term_row0'] += nterm
                r = torch.clamp(r, -1, 1)

                if rnn:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                self.observations[batch_rows, l] = o
                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                self.ep_lengths[env_id] += 1
                if l + 1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=device).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()

            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            if self.group_fn is not None:
                self.prev_groups = np.asarray(self.group_fn(o_np), dtype=np.int64)
            self.vecenv.send(action)

        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        self.t_eval = time.perf_counter() - t0
        return self.stats

    def train(self):
        t0 = time.perf_counter()
        losses = defaultdict(float)
        config = self.config
        device = config['device']
        rnn = config['use_rnn']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = pufferl.compute_puff_advantage(
                self.values, self.rewards, self.terminals, self.ratio, advantages,
                config['gamma'], config['gae_lambda'], config['vtrace_rho_clip'],
                config['vtrace_c_clip'])

            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv ** a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            if not rnn:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
            if rnn:
                state['lstm_h'] = self.seg_state[idx]
                state['terminals'] = mb_terminals

            logits, newvalue = self.policy(mb_obs, state)
            if isinstance(logits, (tuple, list)):  # MultiDiscrete: same values as sample_logits
                newlogprob, entropy = logprob_entropy(logits, mb_actions)
            else:
                _, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)
            with torch.no_grad():
                for h, lg in enumerate(logits):
                    lp = torch.log_softmax(lg, -1)
                    losses[f'entropy_head{h}'] += float(-(lp.exp() * lp).sum(-1).mean()) / self.total_minibatches

            kl_coef = self.kl_coef()
            kl_teacher = None
            if kl_coef > 0:
                with torch.no_grad():
                    tstate = dict(lstm_h=None, lstm_c=None)
                    if rnn:
                        tstate['terminals'] = mb_terminals
                    t_logits, _ = self.teacher(mb_obs, tstate)
                kl_teacher = 0.0
                for lg, tl in zip(logits, t_logits):
                    tlp = torch.log_softmax(tl, -1)
                    kl_teacher = kl_teacher + (tlp.exp() * (tlp - torch.log_softmax(lg, -1))).sum(-1).mean()

            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + config['vf_coef'] * v_loss - config['ent_coef'] * entropy_loss
            if kl_teacher is not None:
                loss = loss + kl_coef * kl_teacher
                losses['kl_teacher'] += kl_teacher.item() / self.total_minibatches

            self.values[idx] = newvalue.detach().float()

            n = self.total_minibatches
            losses['policy_loss'] += pg_loss.item() / n
            losses['value_loss'] += v_loss.item() / n
            losses['entropy'] += entropy_loss.item() / n
            losses['old_approx_kl'] += old_approx_kl.item() / n
            losses['approx_kl'] += approx_kl.item() / n
            losses['clipfrac'] += clipfrac.item() / n
            losses['importance'] += ratio.mean().item() / n

            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = float('nan') if var_y == 0 else float(1 - (y_true - y_pred).var() / var_y)
        losses['explained_variance'] = explained_var
        losses['kl_teacher_coef'] = self.kl_coef()
        self.epoch += 1
        self.t_train = time.perf_counter() - t0
        return self._logs(losses)

    def _logs(self, losses):
        batch = self.config['batch_size']
        d = self.diag
        env = {}
        for k, v in self.stats.items():
            try:
                env[k] = float(np.mean(v))
            except (TypeError, ValueError):
                pass
        logs = {
            'epoch': self.epoch, 'step': self.global_step, 'uptime': self.uptime,
            'lr': self.optimizer.param_groups[0]['lr'],
            'sps': batch / max(self.t_eval + self.t_train, 1e-9),
            'sps_rollout': batch / max(self.t_eval, 1e-9),
            'sps_update': batch / max(self.t_train, 1e-9),
            'clip_frac_reward': d['n_clip'] / max(d['n_rew'], 1),
            'clip_mass_reward': d['mass_clipped'] / max(d['mass'], 1e-12),
            'term_row0_frac': d['n_term_row0'] / d['n_term'] if d['n_term'] else float('nan'),
            'terminals': d['n_term'],
            **{f'losses/{k}': v for k, v in losses.items()},
            **{f'env/{k}': v for k, v in env.items()},
            **action_histogram(self.act_hist, self.action_labels, prefix='act/'),
        }
        if self.grp.shape[1]:
            for i, name in enumerate(self.group_names):
                n, s1, s2 = self.grp[:, i]
                if n:
                    logs[f'rew/{name}/mean'] = s1 / n
                    logs[f'rew/{name}/std'] = float(np.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0)))
        self.stats = defaultdict(list)
        self._reset_diag()
        return logs


HEAD_NAMES = ('bid', 'ask', 'size')


def action_histogram(counts, labels=None, prefix='act/'):
    """Fractions of actions per head, grouped by category label when given
    (e.g. bid indices 1, 2 -> 'improve'). counts: list of per-head count arrays."""
    out = {}
    for h, c in enumerate(counts):
        c = np.asarray(c, dtype=np.float64)
        tot = c.sum()
        if not tot:
            continue
        name = HEAD_NAMES[h] if h < len(HEAD_NAMES) else f'head{h}'
        labs = labels[h] if labels is not None else [str(i) for i in range(len(c))]
        agg = {}
        for i, lab in enumerate(labs):
            agg[lab] = agg.get(lab, 0.0) + c[i] / tot
        for lab, v in agg.items():
            out[f'{prefix}{name}/{lab}'] = v
    return out
