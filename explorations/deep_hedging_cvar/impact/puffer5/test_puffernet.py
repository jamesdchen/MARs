"""Check puffernet.py against PufferLib 5.0's C forward pass.

Writes random weights with PufferNet.save_bin, compiles puffernet_harness.c
against src/puffercpu.c, runs forward_puffernet for T steps and B agents,
with terminals in the middle of the sequences, and compares the decoder
outputs (action means and value) with PufferNet.step in float32 and
float64. It does this for a full file with junk in the padding and for a
file cut to the length the trainer writes. Also checks the file layout, and
runs PufferNetPolicy inside market.simulate.

    python test_puffernet.py

PUFFERLIB_DIR is a checkout of PufferLib's 5.0 branch (commit 6ffa5b1,
default /tmp/claude-0/pl5); CC picks the C compiler (default cc).
"""

import copy
import os
import subprocess
import sys
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from puffernet import PufferNet, PufferNetPolicy  # noqa: E402
from market import ImpactConfig, simulate, fundamental_noise  # noqa: E402

torch.set_num_threads(1)
PUFFERLIB_DIR = os.environ.get('PUFFERLIB_DIR', '/tmp/claude-0/pl5')
OBS, HIDDEN, LAYERS, ACTIONS = 8, 64, 2, 2
B, T = 96, 48


def random_net(seed, enc=0.5, dec=0.3, logstd=1.0, mingru=0.25):
    """Gaussian weights, large enough that the gates move and both branches
    of h_tilde are taken."""
    g = torch.Generator().manual_seed(seed)
    net = PufferNet(OBS, HIDDEN, LAYERS, ACTIONS)
    with torch.no_grad():
        for t, s in zip(net.tensors(), [enc, dec, logstd] + [mingru] * LAYERS):
            t.copy_(torch.randn(t.shape, generator=g) * s)
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


def build_harness(tmp):
    src = os.path.join(PUFFERLIB_DIR, 'src')
    if not os.path.exists(os.path.join(src, 'puffercpu.c')):
        sys.exit(f'{src}/puffercpu.c not found: set PUFFERLIB_DIR to a PufferLib 5.0 checkout')
    exe = os.path.join(tmp, 'puffernet_harness')
    subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-Wall', '-I' + src,
                    os.path.join(HERE, 'puffernet_harness.c'), '-o', exe, '-lm'], check=True)
    return exe


def run_c(exe, tmp, weights, obs, term):
    paths = [os.path.join(tmp, name) for name in ('obs.f32', 'term.f32', 'out.f32')]
    obs.tofile(paths[0])
    term.tofile(paths[1])
    res = subprocess.run([exe, weights, *paths] + [str(v) for v in (B, T, OBS, HIDDEN, LAYERS, ACTIONS)],
                         check=True, capture_output=True, text=True)
    out = np.fromfile(paths[2], np.float32)
    n = T * B * (ACTIONS + 1)
    return out[:n].reshape(T, B, ACTIONS + 1), out[n:].reshape(T, B, ACTIONS), res.stdout.strip()


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
    run_py(net, obs, term, torch.float32)
    for h in hooks:
        h.remove()
    hidden, gate, highway = torch.cat(combined).chunk(3, dim=-1)
    mid = lambda a: ((torch.sigmoid(a) > 0.1) & (torch.sigmoid(a) < 0.9)).double().mean().item()  # noqa: E731
    return (hidden >= 0).double().mean().item(), mid(gate), mid(highway)


def test_layout(tmp):
    net = random_net(1)
    blocks, need = net.layout()
    assert [off for off, _ in blocks] == [0, 512, 704, 712, 13000] and need == 25288
    unpadded = sum(size for _, size in blocks)
    assert unpadded == 25282
    assert [off for off, _ in net.layout(align=4)[0]] == [0, 512, 704, 708, 12996]

    path = os.path.join(tmp, 'layout.bin')
    for align in (8, 4):
        net.save_bin(path, align=align)
        loaded = PufferNet(OBS, HIDDEN, LAYERS, ACTIONS).load_bin(path, align=align)
        assert all(torch.equal(a, b) for a, b in zip(net.tensors(), loaded.tensors()))
    assert os.path.getsize(path) == 4 * 25284

    net.save_bin(path, truncate=True)
    assert os.path.getsize(path) == 4 * unpadded
    loaded = PufferNet(OBS, HIDDEN, LAYERS, ACTIONS).load_bin(path)
    last = loaded.mingru[-1].weight.flatten()
    assert torch.equal(last[-6:], torch.zeros(6))
    assert torch.equal(last[:-6], net.mingru[-1].weight.flatten()[:-6])
    print(f'layout: {unpadded} floats, {need} padded; offsets {[off for off, _ in blocks]}; '
          f'round trip exact; trainer-length file reads the last 6 floats as 0')


def test_against_c(tmp):
    exe = build_harness(tmp)
    net = random_net(0)
    obs, term = make_inputs(0)
    print(f'C reference: {PUFFERLIB_DIR}/src/puffercpu.c ({git_rev()}), B={B}, T={T}, '
          f'hidden={HIDDEN}, layers={LAYERS}, {int(term.sum())} terminals')
    frac_pos, frac_gate, frac_hw = gate_stats(net, obs, term)
    print(f'  hidden >= 0: {frac_pos:.2f}, gate in (0.1, 0.9): {frac_gate:.2f}, '
          f'highway in (0.1, 0.9): {frac_hw:.2f}')
    assert 0.2 < frac_pos < 0.8 and frac_gate > 0.3 and frac_hw > 0.3

    full, cut = os.path.join(tmp, 'full.bin'), os.path.join(tmp, 'cut.bin')
    net.save_bin(full)
    data = np.fromfile(full, np.float32)
    blocks, need = net.layout()
    pad = np.ones(need, bool)
    for off, size in blocks:
        pad[off:off + size] = False
    data[pad] = 1e6
    data.tofile(full)
    net.save_bin(cut, truncate=True)

    for name, path in [('full file, junk padding', full), ('trainer-length file', cut)]:
        dec, act, info = run_c(exe, tmp, path, obs, term)
        assert info == f'file_floats={os.path.getsize(path) // 4} need={need}', info
        assert np.array_equal(act, dec[..., :ACTIONS])
        loaded = PufferNet(OBS, HIDDEN, LAYERS, ACTIONS).load_bin(path)
        print(f'  {name}: {info}, |decoder out| max {np.abs(dec).max():.1f}')
        for dtype in (torch.float32, torch.float64):
            mean, value = run_py(loaded, obs, term, dtype)
            err_mean = np.abs(mean - dec[..., :ACTIONS]).max()
            err_value = np.abs(value - dec[..., ACTIONS]).max()
            print(f'    {str(dtype)[6:]}: max abs err mean {err_mean:.2e}, value {err_value:.2e}')
            assert err_mean < 1e-4 and err_value < 1e-4
        mean, value = run_py(loaded, obs, term, torch.float32, use_terminals=False)
        print(f'    ignoring terminals instead: max abs err {np.abs(mean - dec[..., :ACTIONS]).max():.2e}')
        assert np.abs(mean - dec[..., :ACTIONS]).max() > 0.1


def test_policy():
    cfg, n = ImpactConfig(), 300
    w = 0.3 * cfg.scale
    torch.manual_seed(2)
    for dtype in (torch.float32, torch.float64):
        net = PufferNet(OBS, HIDDEN, LAYERS, ACTIONS)
        torch.set_default_dtype(dtype)
        policy = PufferNetPolicy(net)
        z = fundamental_noise(n, cfg, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            loss1, rec = simulate(z, policy, w, cfg, record=True)
            state = policy.state.clone()
            loss2 = simulate(z, policy, w, cfg)

            ref = [None]
            def fresh(obs):
                x = torch.cat([obs, obs.new_ones(obs.shape[0], 1)], dim=-1)
                m, _, ref[0] = net.step(x, ref[0])
                return m
            loss_ref = simulate(z, fresh, w, cfg)
            ref[0] = state
            loss_carry = simulate(z, fresh, w, cfg)

        assert loss1.shape == (n,) and loss1.dtype == dtype and torch.isfinite(loss1).all()
        assert net.encoder.weight.dtype == dtype
        assert torch.equal(loss1, loss2) and torch.equal(loss1, loss_ref)
        assert state.abs().max() > 0 and not torch.equal(loss1, loss_carry)
        print(f'policy, {str(dtype)[6:]}: {n} paths, mean loss {loss1.mean():.4f}, '
              f'signal > 0 on {rec["trade"].double().mean():.0%} of dates; '
              f'second call identical, equal to a fresh state, '
              f'max |loss change| without the reset {(loss1 - loss_carry).abs().max():.2e}')

        if torch.cuda.is_available():
            loss_cuda = simulate(z.cuda(), PufferNetPolicy(net), w, cfg).cpu()
            assert torch.allclose(loss_cuda, loss1, atol=1e-4)
            print('  cuda matches cpu')
    torch.set_default_dtype(torch.float32)
    if not torch.cuda.is_available():
        print('policy on cuda: no GPU, not run')


def git_rev():
    try:
        return subprocess.run(['git', '-C', PUFFERLIB_DIR, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown commit'


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        test_layout(tmp)
        test_against_c(tmp)
    test_policy()
    print('ok')
