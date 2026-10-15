"""Check puffernet_mm.py against PufferLib 5.0's C PufferNet, discrete heads.

1. Layout: for several (obs_size, hidden, layers), the tensor offsets equal
   the trainer allocator's (16-byte aligned, both builds) and the file length
   its unpadded count; a randomly initialized net (5.0's init) written in the
   5.0 layout round-trips exactly, obs_size is recovered from the file length,
   and with hidden % 8 != 0 the cut tail reads as zeros, as in puffercpu.c.
2. C reference: compiles puffernet_mm_harness.c against src/puffercpu.c, runs
   forward_puffernet for T steps and B agents with terminals mid-sequence, and
   compares the decoder outputs (17 logits + value) with PufferNetMM.step in
   float32 and float64, and the per-head argmax (puffercpu's hard=1) with
   PufferNetMM.act, for a full file with junk in the padding and for a file
   cut at the trainer's length.
3. PufferNetMMPolicy: int32 actions in range, state carried across calls
   equals the net run by hand, `done` restarts rows from a zero state,
   sampling is reproducible and follows softmax(logits).

    python test_puffernet_mm.py

PUFFERLIB_DIR is a checkout of PufferLib's 5.0 branch (default: the clone in
this session's scratchpad; /content/PufferLib on Colab); CC picks the C compiler.
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
from puffernet_mm import (ACT_SIZES, BF16_ALIGN, FLOAT_ALIGN, PufferNetMM,  # noqa: E402
                          PufferNetMMPolicy, infer_obs_size, weight_count)

torch.set_num_threads(1)
PUFFERLIB_DIR = os.environ.get(
    'PUFFERLIB_DIR',
    '/tmp/claude-0/-home-user-MARs/bd289cf3-b2b3-5940-9bd3-6798f7649804/scratchpad/pufferlib5')
# the env's observation size (50), from the core
with open(os.path.join(os.path.dirname(HERE), 'mm.h')) as _f:
    OBS = int(re.search(r'#define MM_OBS (\d+)', _f.read())[1])
CONFIGS = [(OBS, 128, 4), (OBS, 64, 2), (OBS, 32, 1), (13, 20, 1)]
B, T = 64, 40
D = sum(ACT_SIZES) + 1


def random_net(obs, hidden, layers, seed, enc=2 ** 0.5, dec=1.0, mingru=2.0):
    """Gaussian weights with std gain / sqrt(fan_in), large enough that the
    gates move and both branches of h_tilde are taken."""
    g = torch.Generator().manual_seed(seed)
    net = PufferNetMM(obs, hidden, layers)
    with torch.no_grad():
        for t, gain in zip(net.tensors(), [enc, dec] + [mingru] * layers):
            t.copy_(torch.randn(t.shape, generator=g) * gain / t.shape[1] ** 0.5)
    return net


def make_inputs(obs_size, seed):
    """Observations N(0, 1) with the last feature 1.0; terminals: half the
    agents on episodes of 7-25 steps, half Bernoulli(0.1)."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1, (T, B, obs_size)).astype(np.float32)
    obs[..., -1] = 1.0
    term = np.zeros((T, B), np.float32)
    half = B // 2
    for b in range(half):
        t = rng.integers(0, 25)
        while t < T:
            term[t, b] = 1
            t += rng.integers(7, 26)
    term[:, half:] = rng.random((T, B - half)) < 0.1
    return obs, term


def trainer_layout(net, elem_bytes):
    """Element offsets in the trainer's parameter buffer (_alloc_register /
    alloc_create in src/pufferl.cu put each tensor at a 16-byte boundary) and
    the length puf_save_weights writes (the unpadded count)."""
    offsets, nbytes = [], 0
    for t in net.tensors():
        nbytes = (nbytes + 15) & ~15
        offsets.append(nbytes // elem_bytes)
        nbytes += t.numel() * elem_bytes
    return offsets, sum(t.numel() for t in net.tensors())


def test_layout(tmp, obs, hidden, layers):
    net = PufferNetMM(obs, hidden, layers)   # 5.0's own (Kaiming-uniform) init
    path = os.path.join(tmp, 'layout.bin')
    desc = []
    for align, elem_bytes, name in [(FLOAT_ALIGN, 4, '--float'), (BF16_ALIGN, 2, 'bf16')]:
        blocks, need = net.layout(align)
        offsets, file_len = trainer_layout(net, elem_bytes)
        assert [off for off, _ in blocks] == offsets, (blocks, offsets)
        assert file_len == weight_count(obs, hidden, layers)

        net.save_bin(path, align=align, truncate=False)
        assert os.path.getsize(path) == 4 * need
        loaded = PufferNetMM(obs, hidden, layers).load_bin(path, align=align)
        assert all(torch.equal(a, b) for a, b in zip(net.tensors(), loaded.tensors()))

        net.save_bin(path, align=align)   # trainer length
        assert os.path.getsize(path) == 4 * file_len
        loaded = PufferNetMM(obs, hidden, layers).load_bin(path, align=align)
        cut = need - file_len
        last, ref = loaded.mingru[-1].weight.flatten(), net.mingru[-1].weight.flatten()
        if cut:
            assert torch.equal(last[-cut:], torch.zeros(cut)) and torch.equal(last[:-cut], ref[:-cut])
        else:
            assert torch.equal(last, ref)
        assert all(torch.equal(a, b) for a, b in zip(net.tensors()[:-1], loaded.tensors()[:-1]))
        desc.append(f'{name} offsets {offsets}, last {cut} floats cut')
    if hidden % 8 == 0:
        assert infer_obs_size(file_len, hidden, layers) == obs
        assert net.layout(FLOAT_ALIGN) == net.layout(BF16_ALIGN)
        desc.append(f'obs_size {obs} recovered from the length; both builds identical')
    print(f'layout obs={obs} hidden={hidden} layers={layers}: {file_len} floats; '
          f'{"; ".join(desc)}; round trips exact')


def build_harness(tmp):
    src = os.path.join(PUFFERLIB_DIR, 'src')
    if not os.path.exists(os.path.join(src, 'puffercpu.c')):
        sys.exit(f'{src}/puffercpu.c not found: set PUFFERLIB_DIR to a PufferLib 5.0 checkout')
    exe = os.path.join(tmp, 'puffernet_mm_harness')
    subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-Wall', '-I' + src,
                    os.path.join(HERE, 'puffernet_mm_harness.c'), '-o', exe, '-lm'], check=True)
    return exe


def run_c(exe, tmp, weights, obs, term, hidden, layers):
    paths = [os.path.join(tmp, name) for name in ('obs.f32', 'term.f32', 'out.f32')]
    obs.tofile(paths[0])
    term.tofile(paths[1])
    args = [B, T, obs.shape[-1], hidden, layers, *ACT_SIZES]
    res = subprocess.run([exe, weights, *paths] + [str(v) for v in args],
                         check=True, capture_output=True, text=True)
    m = re.fullmatch(r'offsets=([\d,]+) file_floats=(\d+) need=(\d+)', res.stdout.strip())
    assert m, res.stdout
    info = {'offsets': [int(v) for v in m[1].split(',')], 'file_floats': int(m[2]),
            'need': int(m[3])}
    out = np.fromfile(paths[2], np.float32)
    n = T * B * D
    return out[:n].reshape(T, B, D), out[n:].reshape(T, B, len(ACT_SIZES)), info


def run_py(net, obs, term, dtype):
    net = copy.deepcopy(net).to(dtype)
    outs, acts, state = [], [], None
    with torch.no_grad():
        for t in range(T):
            logits, value, state = net.step(torch.from_numpy(obs[t]), state,
                                            torch.from_numpy(term[t]))
            outs.append(torch.cat([logits, value[:, None]], -1))
            acts.append(net.act(logits))
    return torch.stack(outs).double().numpy(), torch.stack(acts).numpy()


def gate_stats(net, obs, term):
    combined = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: combined.append(out))
             for m in net.mingru]
    with torch.no_grad():
        state = None
        for t in range(T):
            _, _, state = net.step(torch.from_numpy(obs[t]), state, torch.from_numpy(term[t]))
    for h in hooks:
        h.remove()
    hidden, gate, highway = torch.cat(combined).chunk(3, dim=-1)
    mid = lambda a: ((torch.sigmoid(a) > 0.1) & (torch.sigmoid(a) < 0.9)).double().mean().item()  # noqa: E731
    return (hidden >= 0).double().mean().item(), mid(gate), mid(highway)


def argmax_ties_ok(c_act, py_act, logits):
    """Argmax may differ only where the top two logits of a head are within 1e-4."""
    bad = c_act.astype(np.int64) != py_act
    if not bad.any():
        return 0
    start = np.cumsum((0,) + ACT_SIZES[:-1])
    for t, b, h in zip(*np.nonzero(bad)):
        lg = np.sort(logits[t, b, start[h]:start[h] + ACT_SIZES[h]])
        assert lg[-1] - lg[-2] < 1e-4, (t, b, h, lg[-2:])
    return int(bad.sum())


def test_against_c(exe, tmp, obs_size, hidden, layers, seed):
    net = random_net(obs_size, hidden, layers, seed)
    obs, term = make_inputs(obs_size, seed)
    frac_pos, frac_gate, frac_hw = gate_stats(net, obs, term)
    print(f'C reference obs={obs_size} hidden={hidden} layers={layers}: B={B} T={T}, '
          f'{int(term.sum())} terminals; hidden >= 0: {frac_pos:.2f}, gate in (0.1, 0.9): '
          f'{frac_gate:.2f}, highway in (0.1, 0.9): {frac_hw:.2f}')
    assert 0.2 < frac_pos < 0.8 and frac_gate > 0.3 and frac_hw > 0.3

    full, cut = os.path.join(tmp, 'full.bin'), os.path.join(tmp, 'cut.bin')
    blocks, need = net.layout(BF16_ALIGN)
    net.save_bin(full, align=BF16_ALIGN, truncate=False)
    data = np.fromfile(full, np.float32)
    pad = np.ones(need, bool)
    for off, size in blocks:
        pad[off:off + size] = False
    data[pad] = 1e6
    data.tofile(full)
    net.save_bin(cut, align=BF16_ALIGN)

    for name, path in [('full file, junk padding', full), ('trainer-length file', cut)]:
        dec, act, info = run_c(exe, tmp, path, obs, term, hidden, layers)
        assert info['offsets'] == [off for off, _ in blocks] and info['need'] == need, info
        assert info['file_floats'] == os.path.getsize(path) // 4
        loaded = PufferNetMM(obs_size, hidden, layers).load_bin(path, align=BF16_ALIGN)
        errs = []
        for dtype in (torch.float32, torch.float64):
            out, py_act = run_py(loaded, obs, term, dtype)
            err = np.abs(out - dec).max()
            assert err < 1e-4, err
            ties = argmax_ties_ok(act, py_act, out)
            errs.append(f'{str(dtype)[6:]} {err:.1e} ({ties} argmax ties)')
        print(f'  {name}: {info["file_floats"]} of {need} floats, offsets {info["offsets"]}, '
              f'|out| max {np.abs(dec).max():.1f}; max abs err {", ".join(errs)}; '
              f'actions per head used: {[len(np.unique(act[..., h])) for h in range(3)]}')


def test_policy(tmp):
    obs_size, hidden, layers, n = OBS, 64, 2, 256
    path = os.path.join(tmp, 'policy.bin')
    net = random_net(obs_size, hidden, layers, 7, dec=0.5)
    net.save_bin(path)
    obs, term = make_inputs(obs_size, 3)
    obs, term = obs[:, :n], term[:, :n]

    pol = PufferNetMMPolicy(path, hidden=hidden, layers=layers)   # obs_size from the file
    assert pol.obs_size == obs_size
    state, acts = None, []
    for t in range(T):
        a, state = pol(obs[t], state, term[t])
        assert a.dtype == np.int32 and a.shape == (B, 3)
        assert all(((a[:, h] >= 0) & (a[:, h] < s)).all() for h, s in enumerate(ACT_SIZES))
        acts.append(a)
    _, ref = run_py(net, obs, term, torch.float32)
    assert np.array_equal(np.stack(acts), ref)

    # done restarts from zeros: one row, state carried vs restarted
    s1 = None
    for t in range(5):
        _, s1 = pol(obs[t], s1)
    a_carry, _ = pol(obs[5], s1)
    a_done, _ = pol(obs[5], s1, np.ones(B, np.float32))
    a_fresh, _ = pol(obs[5], None)
    assert np.array_equal(a_done, a_fresh) and not np.array_equal(a_carry, a_fresh)
    try:
        pol(obs[0, :10], s1)
        raise AssertionError('batch change accepted')
    except ValueError:
        pass

    # sampling: reproducible, and the empirical head frequencies follow softmax
    reps = 20000
    x = np.repeat(obs[0, :1], reps, 0)
    s_a, _ = PufferNetMMPolicy(path, hidden=hidden, layers=layers, deterministic=False, seed=1)(x)
    s_b, _ = PufferNetMMPolicy(path, hidden=hidden, layers=layers, deterministic=False, seed=1)(x)
    assert np.array_equal(s_a, s_b)
    with torch.no_grad():
        logits, _, _ = net.step(torch.from_numpy(obs[0, :1]))
    worst = 0.0
    for h, lg in enumerate(net.split_logits(logits)):
        p = torch.softmax(lg[0].double(), -1).numpy()
        freq = np.bincount(s_a[:, h], minlength=ACT_SIZES[h]) / reps
        worst = max(worst, np.abs(freq - p).max())
    assert worst < 0.02, worst
    print(f'policy: {B} agents x {T} steps match the net run by hand; done restarts rows; '
          f'sampling reproducible, max |freq - softmax| {worst:.4f} over {reps} draws')


def git_rev():
    try:
        return subprocess.run(['git', '-C', PUFFERLIB_DIR, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown commit'


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in CONFIGS:
            test_layout(tmp, *cfg)
        exe = build_harness(tmp)
        print(f'C reference: {PUFFERLIB_DIR}/src/puffercpu.c ({git_rev()})')
        for i, cfg in enumerate(CONFIGS):
            test_against_c(exe, tmp, *cfg, seed=i)
        test_policy(tmp)
    print('ok')
