"""PyTorch port of PufferNet, PufferLib 5.0's default policy, for the MM env.

Loads the flat fp32 .bin checkpoints the 5.0 trainer writes
(puf_save_weights in src/pufferl.cu) and runs the forward pass of the C
reference in src/puffercpu.c (make_puffernet, forward_puffernet), with the
env's multi-discrete action heads ACT_SIZES = (7, 7, 3): bid choice, ask
choice, size. test_puffernet_mm.py checks it against that C code. The logic
follows ../../bates/puffer5/puffernet.py (continuous heads there).

Architecture, no biases anywhere (the observation's last feature is a
constant 1.0 instead):

    x = W_enc obs                                   encoder (H, obs_size)
    for each MinGRU layer l, with state s_l:
        a, g, p = split(W_l x)                      W_l is (3H, H)
        h_tilde = a + 0.5 if a >= 0 else sigmoid(a)
        s_l     = s_l + sigmoid(g) (h_tilde - s_l)
        x       = sigmoid(p) s_l + (1 - sigmoid(p)) x     (highway)
    out = W_dec x                                   decoder (sum(ACT_SIZES) + 1, H)

out[:, :17] are the logits of the three heads, in order, and out[:, -1] the
value. The trainer samples each head from softmax(logits); hard=1 in
puffercpu.c takes the argmax (first maximum on ties, as torch.argmax).
Before each step the state of every row whose terminal flag is set is
zeroed: the 5.0 env raises the flag together with the first observation of
the next episode, so each episode starts from a zero state. The state has
shape (layers, B, H).

Weight file. All parameters live in one buffer, in this order:

    encoder (H, obs_size), decoder (sum(ACT_SIZES) + 1, H),
    then the MinGRU projection (3H, H) of each layer

(no logstd: the heads are discrete), row-major, each tensor starting at a
16-byte boundary: 8 values in the default bfloat16 build, 4 with
`./build.sh --float`. The file is the fp32 master copy cut at the unpadded
element count. Every tensor here is a multiple of H values, so with
H % 8 == 0 (all hidden sizes the sweep draws; PUFFER5.md rule 5) nothing is
padded, both builds give the same layout, and the file length fixes
obs_size (infer_obs_size). `align` matters only for other hidden sizes.
"""

import configparser

import numpy as np
import torch
import torch.nn as nn

ACT_SIZES = (7, 7, 3)
FLOAT_ALIGN = 4  # ./build.sh --float: 16 bytes of float32
BF16_ALIGN = 8   # default bfloat16 build; src/puffercpu.c reads this layout


class PufferNetMM(nn.Module):
    """Encoder, MinGRU layers and a decoder with fused value; discrete heads."""

    def __init__(self, obs_size, hidden=128, layers=4, act_sizes=ACT_SIZES):
        super().__init__()
        assert all(n >= 2 for n in act_sizes), 'a head of size 1 is continuous in 5.0'
        self.obs_size, self.hidden, self.layers = obs_size, hidden, layers
        self.act_sizes = tuple(int(n) for n in act_sizes)
        self.encoder = nn.Linear(obs_size, hidden, bias=False)
        self.decoder = nn.Linear(hidden, sum(self.act_sizes) + 1, bias=False)
        self.mingru = nn.ModuleList(
            nn.Linear(hidden, 3 * hidden, bias=False) for _ in range(layers))
        self.reset_parameters()

    def reset_parameters(self):
        """The 5.0 initialization (puf_kaiming_init in src/algo.cu): uniform on
        +-gain / sqrt(fan_in), gain sqrt(2) for the encoder, 1 otherwise."""
        gains = [(self.encoder, 2 ** 0.5), (self.decoder, 1.0)]
        gains += [(m, 1.0) for m in self.mingru]
        with torch.no_grad():
            for lin, gain in gains:
                bound = gain / lin.in_features ** 0.5
                lin.weight.uniform_(-bound, bound)

    def tensors(self):
        """Parameters in file order."""
        return [self.encoder.weight, self.decoder.weight] + [m.weight for m in self.mingru]

    def layout(self, align=FLOAT_ALIGN):
        """Offset and size of each tensor in the file, and the padded length."""
        blocks, n = [], 0
        for t in self.tensors():
            blocks.append((n, t.numel()))
            n = (n + t.numel() + align - 1) // align * align
        return blocks, n

    def load_bin(self, path, align=FLOAT_ALIGN):
        """Load a 5.0 checkpoint. Missing trailing floats read as zeros, as in
        src/puffercpu.c (only possible when H % 8 != 0)."""
        data = np.fromfile(path, dtype='<f4')
        blocks, need = self.layout(align)
        unpadded = sum(size for _, size in blocks)
        if not unpadded <= len(data) <= need:
            raise ValueError(
                f'{path}: {len(data)} floats, expected {unpadded} to {need} for '
                f'obs_size={self.obs_size}, hidden={self.hidden}, layers={self.layers}, '
                f'act_sizes={self.act_sizes}, align={align}')
        data = np.concatenate([data, np.zeros(need - len(data), np.float32)])
        with torch.no_grad():
            for t, (off, size) in zip(self.tensors(), blocks):
                t.copy_(torch.from_numpy(data[off:off + size].reshape(t.shape)))
        return self

    def save_bin(self, path, align=FLOAT_ALIGN, truncate=True):
        """Write the layout load_bin reads, zero padded. truncate=True (the
        default) stops at the unpadded count, as puf_save_weights does."""
        blocks, need = self.layout(align)
        data = np.zeros(need, np.float32)
        for t, (off, size) in zip(self.tensors(), blocks):
            data[off:off + size] = t.detach().cpu().float().numpy().ravel()
        if truncate:
            data = data[:sum(size for _, size in blocks)]
        data.astype('<f4').tofile(path)

    def initial_state(self, batch):
        return self.encoder.weight.new_zeros(self.layers, batch, self.hidden)

    def step(self, obs, state=None, terminals=None):
        """One time step for B agents (forward_puffernet).

        obs (B, obs_size); state (layers, B, H) or None for zeros; terminals
        (B,): rows with terminals > 0.5 start from a zero state. Returns the
        logits (B, sum(act_sizes)), the value (B,) and the new state, in the
        dtype and on the device of the parameters.
        """
        w = self.encoder.weight
        obs = torch.as_tensor(obs).to(device=w.device, dtype=w.dtype)
        if obs.shape[-1] != self.obs_size:
            raise ValueError(f'expected {self.obs_size} features, got {obs.shape[-1]}')
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
        return out[:, :-1], out[:, -1], torch.stack(new_state)

    def forward(self, obs, state=None, terminals=None):
        return self.step(obs, state, terminals)

    def split_logits(self, logits):
        return torch.split(logits, self.act_sizes, dim=-1)

    def act(self, logits, deterministic=True, generator=None):
        """Actions (B, heads), int64: argmax per head, or a softmax sample."""
        heads = []
        for lg in self.split_logits(logits):
            if deterministic:
                heads.append(lg.argmax(-1))
            else:
                probs = torch.softmax(lg.float(), -1)
                heads.append(torch.multinomial(probs, 1, generator=generator)[:, 0])
        return torch.stack(heads, -1)


def weight_count(obs_size, hidden, layers, act_sizes=ACT_SIZES):
    """Floats in a trainer checkpoint (the unpadded parameter count)."""
    return hidden * obs_size + (sum(act_sizes) + 1) * hidden + layers * 3 * hidden * hidden


def infer_obs_size(n_floats, hidden, layers, act_sizes=ACT_SIZES):
    """obs_size from a checkpoint length, for hidden % 8 == 0 (no padding)."""
    rest = n_floats - (sum(act_sizes) + 1) * hidden - layers * 3 * hidden * hidden
    if hidden % 8 or rest <= 0 or rest % hidden:
        raise ValueError(f'{n_floats} floats do not fit hidden={hidden}, layers={layers}, '
                         f'act_sizes={act_sizes}; pass obs_size')
    return rest // hidden


def read_run_ini(path):
    """hidden_size and num_layers from a 5.0 run log (logs/<env>/<run_id>.ini)."""
    ini = configparser.ConfigParser(strict=False, interpolation=None)
    with open(path) as f:
        ini.read_string(''.join(line for line in f if not line.startswith('#')))
    return int(float(ini['policy']['hidden_size'])), int(float(ini['policy']['num_layers']))


def load_checkpoint(path, hidden, layers, obs_size=None, align=FLOAT_ALIGN,
                    act_sizes=ACT_SIZES, device='cpu'):
    if obs_size is None:
        n = np.fromfile(path, dtype='<f4').size
        obs_size = infer_obs_size(n, hidden, layers, act_sizes)
    net = PufferNetMM(obs_size, hidden, layers, act_sizes).load_bin(path, align)
    return net.requires_grad_(False).to(device)


class PufferNetMMPolicy:
    """policy_fn for the Python evaluator: actions of a 5.0 checkpoint.

        policy = PufferNetMMPolicy('run.bin', hidden=128, layers=4)
        state = None
        actions, state = policy(obs, state, done)      # each step

    obs: (B, obs_size) float32, numpy or torch. state: None (zeros) or what
    the previous call returned. done: optional (B,) flags for rows whose
    previous step ended an episode (the 5.0 env's terminal on the first
    observation of the next one); their state is zeroed first. Returns int32
    actions (B, 3) as numpy (the core's const int*) and the new state.
    deterministic=True takes the argmax of each head; False samples like the
    trainer, from torch.Generator(seed). Runs in float32 on `device`.
    """

    def __init__(self, path, hidden=128, layers=4, obs_size=None, device='cpu',
                 align=FLOAT_ALIGN, deterministic=True, seed=0):
        self.net = load_checkpoint(path, hidden, layers, obs_size, align, device=device)
        self.device = device
        self.deterministic = deterministic
        self.generator = None if deterministic else torch.Generator(device).manual_seed(seed)

    @property
    def obs_size(self):
        return self.net.obs_size

    @torch.no_grad()
    def __call__(self, obs, state=None, done=None):
        obs = torch.as_tensor(np.asarray(obs, np.float32) if not torch.is_tensor(obs) else obs)
        if state is not None and state.shape[1] != obs.shape[0]:
            raise ValueError('batch size changed while carrying a state')
        logits, _, state = self.net.step(obs, state, done)
        act = self.net.act(logits, self.deterministic, self.generator)
        return act.cpu().numpy().astype(np.int32), state


def policy_fn(path, **kwargs):
    """Shorthand: PufferNetMMPolicy(path, **kwargs), a callable (obs, state, done=None)."""
    return PufferNetMMPolicy(path, **kwargs)


class EvaluatorPolicy(nn.Module):
    """A 5.0 checkpoint in the policy contract of ../train/evaluator.py (PufferLib
    3.0's): forward_eval(obs, state) -> (per-head logits tuple, value (B, 1)),
    with state a dict whose 'lstm_h' (here (layers, B, H)) the call updates and
    whose 'done' (B,) flags restart rows from a zero state; recurrent = True.
    The evaluator takes the argmax of each head (puffercpu's hard=1)."""

    recurrent = True
    is_continuous = False

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.nvec = net.act_sizes
        self.obs_size = net.obs_size
        self.hidden_size = net.layers * net.hidden

    @classmethod
    def from_bin(cls, path, hidden=128, layers=4, obs_size=None, align=FLOAT_ALIGN,
                 device='cpu'):
        return cls(load_checkpoint(path, hidden, layers, obs_size, align, device=device))

    def forward_eval(self, obs, state=None):
        state = {} if state is None else state
        logits, value, state['lstm_h'] = self.net.step(obs, state.get('lstm_h'), state.get('done'))
        return self.net.split_logits(logits), value[:, None]

    def forward(self, obs, state=None):
        return self.forward_eval(obs, state)
