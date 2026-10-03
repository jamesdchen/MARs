"""Check puffernet.py against PufferLib 5.0's C forward pass.

For hidden/layers 64/2, 32/1 and 128/3: writes random weights with
PufferNet.save_bin, compiles puffernet_harness.c against src/puffercpu.c,
runs forward_puffernet for T steps and B agents, with terminals in the
middle of the sequences, and compares the decoder outputs (action means and
value) with PufferNet.step in float32 and float64, for a full file with
junk in the padding and for a file cut to the length the trainer writes.
Also checks the file layout against the offsets puffercpu.c reads and
against the trainer's allocator in both builds, and runs PufferNetPolicy
inside market.simulate.

    python test_puffernet.py

PUFFERLIB_DIR is a checkout of PufferLib's 5.0 branch (commit 6ffa5b1;
default /tmp/claude-0/pl5, set it to /content/PufferLib on Colab); CC picks
the C compiler (default cc).
"""

import copy
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from puffernet import BF16_ALIGN, ENV_ACTIONS, FLOAT_ALIGN, PufferNet, PufferNetPolicy  # noqa: E402
from market import ImpactConfig, simulate, fundamental_noise  # noqa: E402

torch.set_num_threads(1)
PUFFERLIB_DIR = os.environ.get('PUFFERLIB_DIR', '/tmp/claude-0/pl5')
OBS, ACTIONS = 8, 2
CONFIGS = [(64, 2), (32, 1), (128, 3)]
B, T = 96, 48


def random_net(hidden, layers, seed, enc=2 ** 0.5, dec=1.0, logstd=1.0, mingru=2.0,
               actions=ACTIONS):
    """Gaussian weights with std gain / sqrt(fan_in) (logstd: std 1), large
    enough that the gates move and both branches of h_tilde are taken."""
    g = torch.Generator().manual_seed(seed)
    net = PufferNet(OBS, hidden, layers, actions)
    gains = [enc, dec, None] + [mingru] * layers
    with torch.no_grad():
        for t, gain in zip(net.tensors(), gains):
            std = logstd if gain is None else gain / t.shape[1] ** 0.5
            t.copy_(torch.randn(t.shape, generator=g) * std)
    return net


def make_inputs(seed):
    """Random observations (last feature 1.0) and terminals: half the agents
    follow the 32-step episode layout of the env (terminal on the expiry
    step and on the reset step) at random phases, half are Bernoulli(0.1)."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1, (T, B, OBS)).astype(np.float32)
    obs[..., -1] = 1.0
    half = B // 2
    k = (np.arange(T)[:, None] + rng.integers(0, 32, half)[None, :]) % 32
    term = np.zeros((T, B), np.float32)
    term[:, :half] = (k == 30) | (k == 0)
    term[:, half:] = rng.random((T, B - half)) < 0.1
    return obs, term


def trainer_layout(net, elem_bytes):
    """Element offsets of the parameters in the trainer's buffer and the
    length of its checkpoint: _alloc_register / alloc_create in
    src/pufferl.cu put every tensor at a 16-byte boundary, and
    puf_save_weights writes total_elems (the unpadded count) floats."""
    offsets, nbytes = [], 0
    for t in net.tensors():
        nbytes = (nbytes + 15) & ~15
        offsets.append(nbytes // elem_bytes)
        nbytes += t.numel() * elem_bytes
    return offsets, sum(t.numel() for t in net.tensors())


def build_harness(tmp):
    src = os.path.join(PUFFERLIB_DIR, 'src')
    if not os.path.exists(os.path.join(src, 'puffercpu.c')):
        sys.exit(f'{src}/puffercpu.c not found: set PUFFERLIB_DIR to a PufferLib 5.0 checkout')
    exe = os.path.join(tmp, 'puffernet_harness')
    subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-Wall', '-I' + src,
                    os.path.join(HERE, 'puffernet_harness.c'), '-o', exe, '-lm'], check=True)
    return exe


def run_c(exe, tmp, weights, obs, term, hidden, layers):
    paths = [os.path.join(tmp, name) for name in ('obs.f32', 'term.f32', 'out.f32')]
    obs.tofile(paths[0])
    term.tofile(paths[1])
    res = subprocess.run([exe, weights, *paths] + [str(v) for v in (B, T, OBS, hidden, layers, ACTIONS)],
                         check=True, capture_output=True, text=True)
    m = re.fullmatch(r'offsets=([\d,]+) file_floats=(\d+) need=(\d+)', res.stdout.strip())
    assert m, res.stdout
    info = {'offsets': [int(v) for v in m[1].split(',')], 'file_floats': int(m[2]), 'need': int(m[3])}
    out = np.fromfile(paths[2], np.float32)
    n = T * B * (ACTIONS + 1)
    return out[:n].reshape(T, B, ACTIONS + 1), out[n:].reshape(T, B, ACTIONS), info


def run_py(net, obs, term, dtype, use_terminals=True):
    net = copy.deepcopy(net).to(dtype)
    means, values, state = [], [], None
    with torch.no_grad():
        for t in range(T):
            m, v, state = net.step(torch.from_numpy(obs[t]), state,
                                   torch.from_numpy(term[t]) if use_terminals else None)
            means.append(m)
            values.append(v)
    return torch.stack(means).double().numpy(), torch.stack(values).double().numpy()


def gate_stats(net, obs, term):
    """Share of h_tilde inputs >= 0 and of gates in (0.1, 0.9), all layers."""
    combined = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: combined.append(out)) for m in net.mingru]
    with torch.no_grad():
        state = None
        for t in range(T):
            _, _, state = net.step(torch.from_numpy(obs[t]), state, torch.from_numpy(term[t]))
    for h in hooks:
        h.remove()
    hidden, gate, highway = torch.cat(combined).chunk(3, dim=-1)
    mid = lambda a: ((torch.sigmoid(a) > 0.1) & (torch.sigmoid(a) < 0.9)).double().mean().item()  # noqa: E731
    return (hidden >= 0).double().mean().item(), mid(gate), mid(highway)


def test_layout(tmp, hidden, layers):
    net = random_net(hidden, layers, 1)
    path = os.path.join(tmp, 'layout.bin')
    desc = []
    for align, elem_bytes, name in [(FLOAT_ALIGN, 4, '--float'), (BF16_ALIGN, 2, 'bf16')]:
        blocks, need = net.layout(align)
        offsets, file_len = trainer_layout(net, elem_bytes)
        assert [off for off, _ in blocks] == offsets
        assert sum(size for _, size in blocks) == file_len

        net.save_bin(path, align=align)
        assert os.path.getsize(path) == 4 * need
        loaded = PufferNet(OBS, hidden, layers, ACTIONS).load_bin(path, align=align)
        assert all(torch.equal(a, b) for a, b in zip(net.tensors(), loaded.tensors()))

        net.save_bin(path, align=align, truncate=True)
        assert os.path.getsize(path) == 4 * file_len
        loaded = PufferNet(OBS, hidden, layers, ACTIONS).load_bin(path, align=align)
        cut = need - file_len
        last, ref = loaded.mingru[-1].weight.flatten(), net.mingru[-1].weight.flatten()
        assert torch.equal(last[-cut:], torch.zeros(cut)) and torch.equal(last[:-cut], ref[:-cut])
        assert all(torch.equal(a, b) for a, b in zip(net.tensors()[:-1], loaded.tensors()[:-1]))
        desc.append(f'{name} offsets {offsets} padded {need}, last {cut} floats read as 0')
    print(f'layout, hidden={hidden} layers={layers}: {file_len} floats in a trainer file '
          f'(as the trainer allocator); {"; ".join(desc)}; round trips exact')


def test_against_c(exe, tmp, hidden, layers, seed):
    net = random_net(hidden, layers, seed)
    obs, term = make_inputs(seed)
    frac_pos, frac_gate, frac_hw = gate_stats(net, obs, term)
    print(f'C reference, hidden={hidden} layers={layers}: B={B}, T={T}, {int(term.sum())} terminals; '
          f'hidden >= 0: {frac_pos:.2f}, gate in (0.1, 0.9): {frac_gate:.2f}, '
          f'highway in (0.1, 0.9): {frac_hw:.2f}')
    assert 0.2 < frac_pos < 0.8 and frac_gate > 0.3 and frac_hw > 0.3

    full, cut = os.path.join(tmp, 'full.bin'), os.path.join(tmp, 'cut.bin')
    blocks, need = net.layout(BF16_ALIGN)
    net.save_bin(full, align=BF16_ALIGN)
    data = np.fromfile(full, np.float32)
    pad = np.ones(need, bool)
    for off, size in blocks:
        pad[off:off + size] = False
    data[pad] = 1e6
    data.tofile(full)
    net.save_bin(cut, align=BF16_ALIGN, truncate=True)

    for name, path in [('full file, junk padding', full), ('trainer-length file', cut)]:
        dec, act, info = run_c(exe, tmp, path, obs, term, hidden, layers)
        assert info['offsets'] == [off for off, _ in blocks] and info['need'] == need, info
        assert info['file_floats'] == os.path.getsize(path) // 4
        assert np.array_equal(act, dec[..., :ACTIONS])
        loaded = PufferNet(OBS, hidden, layers, ACTIONS).load_bin(path, align=BF16_ALIGN)
        errs = []
        for dtype in (torch.float32, torch.float64):
            mean, value = run_py(loaded, obs, term, dtype)
            err_mean = np.abs(mean - dec[..., :ACTIONS]).max()
            err_value = np.abs(value - dec[..., ACTIONS]).max()
            assert err_mean < 1e-4 and err_value < 1e-4, (err_mean, err_value)
            errs.append(f'{str(dtype)[6:]} mean {err_mean:.1e} value {err_value:.1e}')
        mean, _ = run_py(loaded, obs, term, torch.float32, use_terminals=False)
        err_noterm = np.abs(mean - dec[..., :ACTIONS]).max()
        assert err_noterm > 0.1
        print(f'  {name}: {info["file_floats"]} of {need} floats, offsets {info["offsets"]} as '
              f'make_puffernet, |out| max {np.abs(dec).max():.1f}; max abs err {", ".join(errs)}; '
              f'ignoring terminals instead {err_noterm:.1e}')


def recorded(fn):
    """fn and the list of actions it returns."""
    acts = []

    def g(obs):
        acts.append(fn(obs))
        return acts[-1]
    return g, acts


def test_policy(tmp):
    cfg, n, hidden, layers = ImpactConfig(), 300, 64, 2
    w = 0.3 * cfg.scale
    path = os.path.join(tmp, 'policy.bin')
    random_net(hidden, layers, 4, dec=0.3, actions=ENV_ACTIONS).save_bin(path, truncate=True)
    ref_net = PufferNet(OBS, hidden, layers, ENV_ACTIONS).load_bin(path)
    z32 = fundamental_noise(n, cfg, generator=torch.Generator().manual_seed(1))
    out = {}

    def reference(state):
        """The net run by hand from `state` at date 0 (None: zeros)."""
        box = [state]

        def fn(obs):
            x = torch.cat([obs.float(), torch.ones(obs.shape[0], 1)], dim=-1)
            m, _, box[0] = ref_net.step(x, box[0])
            return m[:, :2].to(obs.dtype)
        return fn

    def run(z, fn, record=False):
        fn, acts = recorded(fn)
        with torch.no_grad():
            res = simulate(z, fn, w, cfg, record=record)
        return res, torch.stack(acts)

    for z in (z32, z32.double()):
        policy = PufferNetPolicy(path, hidden=hidden, layers=layers)
        (loss, rec), acts = run(z, policy, record=True)
        carried = policy.state.clone()
        loss2, acts2 = run(z, policy)
        loss_ref, acts_ref = run(z, reference(None))
        loss_carry, acts_carry = run(z, reference(carried))
        loss_sub, _ = run(z[:200], policy)
        out[z.dtype] = loss, acts

        trade = rec['trade'].double().mean().item()
        carry_diff = (acts_carry[0] - acts[0]).abs().max().item()
        assert loss.shape == (n,) and loss.dtype == z.dtype and torch.isfinite(loss).all()
        assert acts.shape == (cfg.n_steps, n, ACTIONS) and acts.dtype == z.dtype
        assert torch.equal(acts, acts2) and torch.equal(loss, loss2)
        assert torch.equal(acts, acts_ref) and torch.equal(loss, loss_ref)
        assert carried.abs().max() > 0 and carry_diff > 0.01
        assert torch.allclose(loss_sub, loss[:200], atol=1e-5)
        assert 0.05 < trade < 0.95
        print(f'policy, {str(z.dtype)[6:]} paths: {n} paths, mean loss {loss.mean():.4f}, '
              f'signal > 0 on {trade:.0%} of dates, actions in {str(z.dtype)[6:]}; second call '
              f'identical; equal to the net run from a zero state at date 0 (carrying the state '
              f'instead moves date-0 actions by up to {carry_diff:.2f}); then 200 paths: same losses')

    _, wrong = run(z32, PufferNetPolicy(path, hidden=hidden, layers=layers, align=BF16_ALIGN))
    wrong_diff = (wrong - out[torch.float32][1]).abs().max().item()
    assert wrong_diff > 0.01
    print(f'policy, file read with the bf16 alignment instead of --float: actions move by up to '
          f'{wrong_diff:.2f}')

    if torch.cuda.is_available():
        with torch.no_grad():
            obs0 = torch.zeros(n, 7)
            cpu = PufferNetPolicy(path, hidden=hidden, layers=layers)(obs0)
            gpu = PufferNetPolicy(path, hidden=hidden, layers=layers, device='cuda')(obs0.cuda())
            follow = PufferNetPolicy(path, hidden=hidden, layers=layers)
            loss_cuda = simulate(z32.double().cuda(), follow, w, cfg)
        assert gpu.device.type == 'cuda' and torch.allclose(gpu.cpu(), cpu, atol=1e-4)
        assert loss_cuda.device.type == 'cuda' and loss_cuda.dtype == torch.float64
        assert torch.isfinite(loss_cuda).all()
        print(f'policy on cuda: date-0 actions match cpu (max diff {(gpu.cpu() - cpu).abs().max():.1e}); '
              f'a cpu policy follows cuda observations; median |loss diff| vs cpu '
              f'{(loss_cuda.cpu() - out[torch.float64][0]).abs().median():.1e}')
    else:
        print('policy on cuda: no GPU, not run')


def git_rev():
    try:
        return subprocess.run(['git', '-C', PUFFERLIB_DIR, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown commit'


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        for hidden, layers in CONFIGS:
            test_layout(tmp, hidden, layers)
        exe = build_harness(tmp)
        print(f'C reference: {PUFFERLIB_DIR}/src/puffercpu.c ({git_rev()}), library build')
        for i, (hidden, layers) in enumerate(CONFIGS):
            test_against_c(exe, tmp, hidden, layers, seed=i)
        test_policy(tmp)
    print('ok')
