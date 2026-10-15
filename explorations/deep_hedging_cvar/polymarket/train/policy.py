"""Actor-critic policies for the Polymarket market-making env, in PufferLib 3.0's
policy contract.

Contract (pufferlib/pufferl.py, pufferlib/pytorch.py of 3.0):
- `forward_eval(obs, state) -> (logits, value)` is the rollout call. For the
  MultiDiscrete([7, 7, 3]) action space `logits` is a tuple of three (B, n_i)
  tensors: `sample_logits` pads them to (3, B, 7) with -inf, samples one action
  per head and sums the per-head log-probabilities and entropies. `value` is (B, 1).
- `forward(obs, state)` is the training call on a minibatch. Feed-forward
  policies get flat (B, obs) observations; recurrent ones get (S, T, obs)
  segments and return flat (S*T, n_i) logits and a (S, T) value.
- `is_continuous = False`; `hidden_size` sizes the recurrent state that PuffeRL
  allocates per agent (`lstm_h`) when `use_rnn` is on.

Two architectures:

MLPPolicy (default): tanh MLP, 2 x 128 by default. `shared=True` puts the
three categorical heads and the value head on one trunk; `shared=False` gives
the actor and the critic their own trunks (CleanRL's PPO layout). CleanRL
initialization: orthogonal, gain sqrt(2) in the trunk, 0.01 on the policy
logits (near-uniform initial policy), 1 on the value.

PufferNetPolicy: PufferLib 5.0's default network (src/algo.cu; CPU reference
src/puffercpu.c), so 3.0 and 5.0 train the same architecture:

    x = W_enc obs                                    (H, obs), no bias, no activation
    for each of `layers` MinGRU layers, with state s_l:
        a, g, p = split(W_l x)                       W_l is (3H, H), no bias
        h_tilde = a + 0.5 if a >= 0 else sigmoid(a)
        s_l     = s_l + sigmoid(g) * (h_tilde - s_l)          (lerp)
        x       = sigmoid(p) * s_l + (1 - sigmoid(p)) * x     (highway)
    out = W_dec x                                    (sum(nvec) + 1, H), no bias
    logits = out[:, :sum(nvec)] split by head; value = out[:, -1]

Init as puf_kaiming_init: uniform(+-gain/sqrt(fan_in)), gain sqrt(2) for the
encoder and 1 for the MinGRU and decoder matrices. No biases anywhere, so the
env's constant-1 last observation is the only offset (PUFFER5.md rule 3).
The state of a row is zeroed when that row's terminal flag is set, before the
observation is encoded: the 3.0 env raises the flag together with the first
observation of the next episode, as 5.0's does. The recurrent state is
(B, layers * H), flattened so that stock PuffeRL can hold it in `lstm_h`
(it ignores `lstm_c`). `hidden_size` is therefore layers * H; `width` is H.

Stock PuffeRL 3.0 starts every training segment from a zero state and never
resets inside a segment. `train/puffer3.py:Trainer` passes the rollout state
at each segment's first row and the terminal flags, so training sees exactly
the rollout's recurrence (5.0's default `reset_every_horizon = 0`).
"""

import math

import numpy as np
import torch
import torch.nn as nn

NVEC = (7, 7, 3)


def _space_dims(env=None, obs_size=None, nvec=None):
    if env is not None:
        obs_size = int(np.prod(env.single_observation_space.shape))
        nvec = tuple(int(n) for n in env.single_action_space.nvec)
    return int(obs_size), tuple(nvec or NVEC)


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


_ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}


def _trunk(inp, hidden, layers, act):
    mods, d = [], inp
    for _ in range(layers):
        mods += [_ortho(nn.Linear(d, hidden), math.sqrt(2)), _ACTS[act]()]
        d = hidden
    return nn.Sequential(*mods)


class MLPPolicy(nn.Module):
    """Feed-forward actor-critic with three categorical heads and a value head."""

    recurrent = False

    def __init__(self, env=None, obs_size=None, nvec=None, hidden=128, layers=2,
                 act='tanh', shared=False):
        super().__init__()
        self.obs_size, self.nvec = _space_dims(env, obs_size, nvec)
        self.hidden_size = hidden
        self.is_continuous = False
        self.shared = bool(shared)
        self.actor_trunk = _trunk(self.obs_size, hidden, layers, act)
        self.critic_trunk = None if self.shared else _trunk(self.obs_size, hidden, layers, act)
        self.actor_head = _ortho(nn.Linear(hidden, sum(self.nvec)), 0.01)
        self.value_head = _ortho(nn.Linear(hidden, 1), 1.0)

    def forward_eval(self, obs, state=None):
        x = obs.float()
        h = self.actor_trunk(x)
        logits = self.actor_head(h).split(self.nvec, dim=-1)
        v = self.value_head(h if self.shared else self.critic_trunk(x))
        return logits, v

    def forward(self, obs, state=None):
        return self.forward_eval(obs.reshape(-1, self.obs_size), state)


class PufferNetPolicy(nn.Module):
    """PufferLib 5.0's PufferNet (encoder, MinGRU layers with highway, fused decoder)."""

    recurrent = True

    def __init__(self, env=None, obs_size=None, nvec=None, hidden=128, layers=1):
        super().__init__()
        self.obs_size, self.nvec = _space_dims(env, obs_size, nvec)
        self.width, self.layers = hidden, layers
        self.hidden_size = layers * hidden  # per-agent recurrent state size
        self.is_continuous = False
        self.n_logits = sum(self.nvec)
        self.encoder = nn.Linear(self.obs_size, hidden, bias=False)
        self.mingru = nn.ModuleList(nn.Linear(hidden, 3 * hidden, bias=False)
                                    for _ in range(layers))
        self.decoder = nn.Linear(hidden, self.n_logits + 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gains = [(self.encoder, math.sqrt(2)), (self.decoder, 1.0)]
        gains += [(m, 1.0) for m in self.mingru]
        with torch.no_grad():
            for lin, gain in gains:
                bound = gain / math.sqrt(lin.in_features)
                lin.weight.uniform_(-bound, bound)

    def tensors(self):
        """Parameters in the 5.0 trainer's buffer order (encoder, decoder, MinGRU layers)."""
        return [self.encoder.weight, self.decoder.weight] + [m.weight for m in self.mingru]

    def save_puffer5_bin(self, path):
        """Write the weights as a PufferLib 5.0 checkpoint (.bin): fp32, the 5.0
        buffer order (tensors()), each tensor 16-byte aligned. Every tensor here is a
        multiple of H values, so with H % 8 == 0 there is no padding in either 5.0
        build and the file is the plain concatenation (puffer5/puffernet_mm.py loads it)."""
        if self.width % 8:
            raise ValueError('5.0 needs hidden % 8 == 0 (PUFFER5.md rule 5)')
        data = np.concatenate([t.detach().cpu().float().numpy().ravel() for t in self.tensors()])
        data.astype('<f4').tofile(path)
        return path

    def initial_state(self, batch, device=None):
        return torch.zeros(batch, self.hidden_size, device=device or self.encoder.weight.device)

    def _gates(self, proj, x):
        """h_tilde, z = sigmoid(g), sigmoid(p) of the three thirds of W x. Three
        matmuls on row blocks of the one (3H, H) weight (5.0's layout) give
        contiguous outputs; chunking one (.., 3H) output makes every elementwise
        op below strided and ~2x slower on CPU."""
        H = self.width
        W = proj.weight
        a = torch.nn.functional.linear(x, W[:H])
        g = torch.nn.functional.linear(x, W[H:2 * H])
        p = torch.nn.functional.linear(x, W[2 * H:])
        h_tilde = torch.where(a >= 0, a + 0.5, torch.sigmoid(a))
        return h_tilde, torch.sigmoid(g), torch.sigmoid(p)

    def _decode(self, x):
        out = self.decoder(x)
        return out[..., :self.n_logits].split(self.nvec, dim=-1), out[..., self.n_logits:]

    def step(self, obs, s, done=None):
        """One time step for B agents. s: (B, layers*H) or None; done: (B,) flags.
        Returns logits, value (B, 1) and the new state."""
        B = obs.shape[0]
        x = self.encoder(obs.float())
        s = x.new_zeros(B, self.layers, self.width) if s is None else s.view(B, self.layers, self.width)
        if done is not None:
            keep = 1.0 - torch.as_tensor(done, device=x.device).to(x.dtype).view(B, 1, 1)
            s = s * keep
        new = []
        for l, proj in enumerate(self.mingru):
            h_tilde, z, p = self._gates(proj, x)
            h = torch.lerp(s[:, l], h_tilde, z)
            x = p * h + (1 - p) * x
            new.append(h)
        logits, v = self._decode(x)
        return logits, v, torch.stack(new, 1).reshape(B, self.hidden_size)

    def forward_eval(self, obs, state=None):
        state = {} if state is None else state
        logits, v, s = self.step(obs, state.get('lstm_h'), state.get('done'))
        state['lstm_h'] = s
        return logits, v

    def forward(self, obs, state=None):
        """Training call on segments obs (S, T, obs) (or (B, obs) as one step).
        state['lstm_h'] (S, layers*H): the state before row 0, None for zeros;
        state['terminals'] (S, T): reset the state before row t where set."""
        state = {} if state is None else state
        if obs.dim() == 2:
            return self.forward_eval(obs, state)
        S, T, _ = obs.shape
        # time-major (T, S, .) so each step's slice is contiguous
        x = self.encoder(obs.float().transpose(0, 1))              # (T, S, H)
        s0 = state.get('lstm_h')
        s0 = x.new_zeros(S, self.layers, self.width) if s0 is None else s0.view(S, self.layers, self.width)
        term = state.get('terminals')
        keeps = None if term is None else (1.0 - term.to(x.dtype)).t().unsqueeze(-1).unbind(0)
        finals = []
        for l, proj in enumerate(self.mingru):
            h_tilde, z, p = self._gates(proj, x)                   # (T, S, H) each
            # unbind, not x[t]: a slice's backward writes a full (T, S, H)
            # zero tensor per step (quadratic in T); unbind's is one stack
            hts, zs = h_tilde.unbind(0), z.unbind(0)
            h, hs = s0[:, l], []
            for t in range(T):
                if keeps is not None:
                    h = h * keeps[t]
                h = torch.lerp(h, hts[t], zs[t])
                hs.append(h)
            x = torch.lerp(x, torch.stack(hs, 0), p)               # p*h + (1-p)*x
            finals.append(h)
        logits, v = self._decode(x.transpose(0, 1).reshape(S * T, self.width))
        state['lstm_h'] = torch.stack(finals, 1).reshape(S, self.hidden_size).detach()
        return logits, v.reshape(S, T)


POLICIES = {'mlp': MLPPolicy, 'puffernet': PufferNetPolicy}


def make_policy(kind='mlp', env=None, obs_size=None, nvec=None, **kw):
    """kind 'mlp' (kw: hidden, layers, act, shared) or 'puffernet' (kw: hidden, layers)."""
    cls = POLICIES[kind]
    if kind == 'puffernet':
        kw = {k: v for k, v in kw.items() if k in ('hidden', 'layers')}
    return cls(env=env, obs_size=obs_size, nvec=nvec, **kw)


def greedy_actions(logits):
    """Deterministic (argmax) action of each head: (B, num_heads) int32."""
    return torch.stack([l.argmax(-1) for l in logits], -1).int()


def sample_actions(logits, generator=None):
    """Sample each head; returns int32 actions (B, H) and the summed log-prob (B,),
    the same convention as pufferlib.pytorch.sample_logits."""
    acts, lps = [], []
    for l in logits:
        logp = torch.log_softmax(l.float(), -1)
        a = torch.multinomial(logp.exp(), 1, generator=generator)
        acts.append(a)
        lps.append(logp.gather(-1, a))
    return torch.cat(acts, -1).int(), torch.cat(lps, -1).sum(-1)


def logprob_entropy(logits, actions):
    """Summed log-prob of `actions` (B, H) (any leading shape, flattened) and summed
    entropy (B,) over the heads: the values pufferlib.pytorch.sample_logits returns
    for a MultiDiscrete, without its -inf padding to the widest head (3x faster)."""
    a = actions.reshape(-1, len(logits)).long()
    lp, ent = 0.0, 0.0
    for h, l in enumerate(logits):
        logp = torch.log_softmax(l.float(), -1)
        lp = lp + logp.gather(-1, a[:, h:h + 1]).squeeze(-1)
        ent = ent - (logp.exp() * logp).sum(-1)
    return lp, ent


def load_checkpoint(path, env=None, map_location='cpu'):
    """Rebuild a policy from a train_ppo.py checkpoint. Returns (policy, ckpt dict)."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    pk = dict(ck['policy_kwargs'])
    kind = pk.pop('kind')
    policy = make_policy(kind, env=env, obs_size=ck['obs_size'], nvec=tuple(ck['nvec']), **pk)
    policy.load_state_dict(ck['policy'])
    policy.eval()
    return policy, ck
