"""PyTorch port of PufferNet, the default policy of PufferLib 5.0.

Loads the flat fp32 .bin checkpoints written by the 5.0 trainer
(puf_save_weights in src/pufferl.cu) and runs the same forward pass as the
C reference in src/puffercpu.c (make_puffernet, forward_puffernet);
test_puffernet.py checks the two against each other.

Architecture, with no biases anywhere (the observation carries a constant
1.0 instead, as its last feature):

    x = W_enc obs                                   encoder (H, obs_size)
    for each MinGRU layer l, with state s_l:
        a, g, p = split(W_l x)                      W_l is (3H, H)
        h_tilde = a + 0.5 if a >= 0 else sigmoid(a)
        s_l     = s_l + sigmoid(g) (h_tilde - s_l)
        x       = sigmoid(p) s_l + (1 - sigmoid(p)) x     (highway)
    out = W_dec x                                   decoder (num_actions + 1, H)

out[:num_actions] is the Gaussian mean, i.e. the deterministic action, and
out[-1] the value; the standard deviation exp(logstd) does not depend on
the observation. Before each step the state of every row whose terminal
flag is set is zeroed (mingru_zero_term), so the 5.0 env, which raises the
flag together with the first observation of an episode, starts every
episode from a zero state. The state has shape (layers, B, H).

Weight file. The trainer keeps all parameters in one buffer, in this order:

    encoder (H, obs_size), decoder (num_actions + 1, H), logstd (num_actions,),
    then the MinGRU projection (3H, H) of each layer,

row-major, each tensor starting at a 16-byte boundary. The file is the fp32
master copy of that buffer, cut at the unpadded element count. 16 bytes
are 8 values in the default bfloat16 build, the layout src/puffercpu.c reads
(get_weights_aligned), but 4 in a build with `./build.sh --float`, which
the Colab notebook uses. The file length is the same for both, so `align`
must match the build; the default is the --float layout. With 2 actions
logstd would be padded and the last floats of the last MinGRU projection
cut from the file (load_bin reads them as zeros, as src/puffercpu.c does).
The hedging env has 13 observations and 4 actions, so in the --float build
every tensor is a multiple of 4 floats and nothing is padded or cut.
"""

import numpy as np
import torch
import torch.nn as nn

FLOAT_ALIGN = 4  # ./build.sh --float: 16 bytes of float32
BF16_ALIGN = 8   # default bfloat16 build; src/puffercpu.c


class PufferNet(nn.Module):
    """Encoder, MinGRU layers and decoder with a fused value, continuous actions."""

    def __init__(self, obs_size=13, hidden=64, layers=2, num_actions=4):
        super().__init__()
        self.obs_size, self.hidden = obs_size, hidden
        self.layers, self.num_actions = layers, num_actions
        self.encoder = nn.Linear(obs_size, hidden, bias=False)
        self.decoder = nn.Linear(hidden, num_actions + 1, bias=False)
        self.logstd = nn.Parameter(torch.zeros(num_actions))
        self.mingru = nn.ModuleList(
            nn.Linear(hidden, 3 * hidden, bias=False) for _ in range(layers))
        self.reset_parameters()

    def reset_parameters(self):
        """The 5.0 initialization (puf_kaiming_init in src/algo.cu): uniform on
        +-gain / sqrt(fan_in), gain sqrt(2) for the encoder and 1 for the
        other matrices, logstd 0."""
        gains = [(self.encoder, 2 ** 0.5), (self.decoder, 1.0)]
        gains += [(m, 1.0) for m in self.mingru]
        with torch.no_grad():
            for lin, gain in gains:
                bound = gain / lin.in_features ** 0.5
                lin.weight.uniform_(-bound, bound)
            self.logstd.zero_()

    def tensors(self):
        """Parameters in file order."""
        return [self.encoder.weight, self.decoder.weight, self.logstd] + [
            m.weight for m in self.mingru]

    def layout(self, align=FLOAT_ALIGN):
        """Offset and size of each tensor in the file, and the padded length."""
        blocks, n = [], 0
        for t in self.tensors():
            blocks.append((n, t.numel()))
            n = (n + t.numel() + align - 1) // align * align
        return blocks, n

    def load_bin(self, path, align=FLOAT_ALIGN):
        """Load a 5.0 checkpoint. Missing trailing floats read as zeros."""
        data = np.fromfile(path, dtype='<f4')
        blocks, need = self.layout(align)
        unpadded = sum(size for _, size in blocks)
        if not unpadded <= len(data) <= need:
            raise ValueError(
                f'{path}: {len(data)} floats, expected {unpadded} to {need} for '
                f'obs_size={self.obs_size}, hidden={self.hidden}, '
                f'layers={self.layers}, num_actions={self.num_actions}, align={align}')
        data = np.concatenate([data, np.zeros(need - len(data), np.float32)])
        with torch.no_grad():
            for t, (off, size) in zip(self.tensors(), blocks):
                t.copy_(torch.from_numpy(data[off:off + size].reshape(t.shape)))
        return self

    def save_bin(self, path, align=FLOAT_ALIGN, truncate=False):
        """Write the layout load_bin reads, with zero padding. truncate=True
        stops at the unpadded count, as puf_save_weights does (and drops the
        last floats of the last MinGRU projection)."""
        blocks, need = self.layout(align)
        data = np.zeros(need, np.float32)
        for t, (off, size) in zip(self.tensors(), blocks):
            data[off:off + size] = t.detach().cpu().float().numpy().ravel()
        if truncate:
            data = data[:sum(size for _, size in blocks)]
        data.astype('<f4').tofile(path)

    def initial_state(self, batch):
        return self.encoder.weight.new_zeros(self.layers, batch, self.hidden)

    def step(self, obs, state, terminals=None):
        """One time step for B agents (forward_puffernet).

        obs (B, obs_size); state (layers, B, hidden), None for zeros;
        terminals (B,): rows with terminals > 0.5 start from a zero state.
        Returns the action mean (B, num_actions), the value (B,) and the new
        state, in the dtype and on the device of the parameters.
        """
        w = self.encoder.weight
        obs = obs.to(device=w.device, dtype=w.dtype)
        if state is None:
            state = self.initial_state(obs.shape[0])
        if terminals is not None:
            reset = torch.as_tensor(terminals, device=w.device).to(w.dtype) > 0.5
            state = torch.where(reset[None, :, None], torch.zeros_like(state), state)
        x = self.encoder(obs)
        new_state = []
        for s, proj in zip(state, self.mingru):
            hidden, gate, highway = proj(x).chunk(3, dim=-1)
            h_tilde = torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
            h = s + torch.sigmoid(gate) * (h_tilde - s)
            p = torch.sigmoid(highway)
            x = p * h + (1 - p) * x
            new_state.append(h)
        out = self.decoder(x)
        return out[:, :self.num_actions], out[:, -1], torch.stack(new_state)

    def forward(self, obs, state, terminals=None):
        return self.step(obs, state, terminals)


# The hedging env (../hedge.h): stock target, stock signal, swap target, swap
# signal, after 13 observations whose last one is the constant 1.
ENV_OBS, ENV_ACTIONS = 13, 4


class PufferNetPolicy:
    """policy_fn for market.simulate: the deterministic actions of a 5.0 checkpoint.

    Takes market.make_obs observations (13 features, the last one constant)
    and runs the net in float32, the precision of the env's observations and of the C
    forward, on the device of the observations. Carries the MinGRU state
    from one call to the next and zeroes it on date 0 (obs[:, 0] == 0),
    where the 5.0 env raises its terminal flag, so consecutive simulate
    calls are independent. Returns the action means in the observations'
    dtype. `align` is that of the build that wrote the file (see above).
    """

    def __init__(self, path, hidden=64, layers=2, device='cpu', align=FLOAT_ALIGN,
                 num_actions=ENV_ACTIONS):
        self.net = PufferNet(ENV_OBS, hidden, layers, num_actions).load_bin(path, align)
        self.net.requires_grad_(False).to(device)
        self.state = None

    def reset(self):
        self.state = None

    def __call__(self, obs):
        net = self.net
        if obs.shape[-1] != net.obs_size:
            raise ValueError(f'expected {net.obs_size} features, got {obs.shape[-1]}')
        if net.encoder.weight.device != obs.device:
            net.to(obs.device)
            self.state = None
        n = obs.shape[0]
        first = obs[:, 0] == 0
        if self.state is not None and self.state.shape[1] != n:
            if not bool(first.all()):
                raise ValueError('batch size changed in the middle of an episode')
            self.state = None
        with torch.no_grad():
            mean, _, self.state = net.step(obs.float(), self.state, first)
        return mean.to(obs.dtype)
