"""Unit tests for obs_spec.py and baselines.py on synthetic observations (no env needed).

    python test_baselines.py          # or: pytest test_baselines.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as bl  # noqa: E402
from obs_spec import SpecError, load_spec, spec_from_names  # noqa: E402

NAMES = ["seconds_to_close", "best_bid", "best_ask", "inventory", "contract_len_s",
         "p_model", "book_valid", "const_one"]
OFF, IMP2, IMP1, JOIN, B1, B2, B4 = range(7)


def make_obs(spec, **cols):
    """(B, n) float32 obs from raw columns (applies the spec's forward normalisation)."""
    B = max(np.size(v) for v in cols.values())
    obs = np.zeros((B, max(f["index"] for f in spec.features.values()) + 1), np.float32)
    for name, f in spec.features.items():
        if name not in cols:
            continue
        x = np.broadcast_to(np.asarray(cols[name], np.float64), (B,)).copy()
        if f.get("transform") == "log1p":
            x = np.log1p(x)
        sc, sh = float(f.get("scale", 1.0)), float(f.get("shift", 0.0))
        obs[:, f["index"]] = (x - sh) / sc
    return obs


def base_spec():
    return spec_from_names(NAMES)


def state_obs(spec=None, **kw):
    spec = spec or base_spec()
    d = dict(seconds_to_close=200.0, best_bid=0.48, best_ask=0.52, inventory=0.0,
             contract_len_s=900.0, p_model=0.50, book_valid=1.0, const_one=1.0)
    d.update(kw)
    return spec, make_obs(spec, **d)


# ----------------------------------------------------------------------------- spec loading
def test_spec_json_and_markdown_and_missing():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "feat_meta.json"
        p.write_text(json.dumps({"obs_size": 8, "features": [
            {"name": n, "index": i, **({"scale": 100.0} if n == "inventory" else {})}
            for i, n in enumerate(NAMES)]}))
        spec = load_spec(p)
        assert spec.find("q") == "inventory" and spec.features["inventory"]["scale"] == 100.0
        obs = make_obs(spec, seconds_to_close=10, best_bid=0.3, best_ask=0.33, inventory=-25,
                       contract_len_s=300, book_valid=1)
        st = spec.decode(obs)
        assert abs(st["q"][0] + 25) < 1e-4 and st["T"][0] == 300 and st["tau"][0] == 10
        md = Path(td) / "ENV.md"
        md.write_text("# env\n\n| idx | name | note |\n|---|---|---|\n"
                      + "".join(f"| {i} | `{n}` | x |\n" for i, n in enumerate(NAMES)))
        spec2 = load_spec(md)
        assert spec2.features["p_model"]["index"] == 5
        md.write_text("text\n```json\n" + json.dumps({"obs_index": {n: i for i, n in enumerate(NAMES)}})
                      + "\n```\n")
        assert load_spec(md).features["best_ask"]["index"] == 2
        try:
            load_spec(Path(td) / "nope.json")
            raise AssertionError("missing spec must raise")
        except SpecError as e:
            assert "no observation spec" in str(e)
    spec = spec_from_names(["best_bid", "best_ask", "seconds_to_close", "contract_len_s"])
    try:
        spec.decode(np.zeros((2, 4), np.float32))
        raise AssertionError("missing inventory must raise")
    except SpecError as e:
        assert "inventory" in str(e)
    spec = spec_from_names(["best_bid", "best_ask", "seconds_to_close", "contract_len_s", "inventory"])
    try:
        spec.decode(np.zeros((2, 5), np.float32), need_fair=True)
        raise AssertionError("missing p_model must raise")
    except SpecError as e:
        assert "p_model" in str(e)


def test_decode_derivations():
    spec = spec_from_names(["mid", "spread", "inventory", "tau_frac", "len_5m", "len_15m", "len_4h", "gap"],
                           inventory={"scale": 50.0, "shift": 0.0})
    obs = make_obs(spec, mid=[0.5, 0.975], spread=[0.02, 0.01], inventory=[10, -10],
                   tau_frac=[0.5, 0.1], len_5m=[1, 0], len_15m=[0, 0], len_4h=[0, 1], gap=[0.03, -0.01])
    st = spec.decode(obs, need_fair=True)
    np.testing.assert_allclose(st["bid"], [0.49, 0.97], atol=1e-6)
    np.testing.assert_allclose(st["ask"], [0.51, 0.98], atol=1e-6)
    np.testing.assert_allclose(st["T"], [300, 14400])
    np.testing.assert_allclose(st["tau"], [150, 1440], rtol=1e-6)
    np.testing.assert_allclose(st["tick"], [0.01, 0.001])          # fine tick above 0.96
    np.testing.assert_allclose(st["p_model"], [0.53, 0.965], atol=1e-6)
    np.testing.assert_allclose(st["q"], [10, -10], atol=1e-5)
    # empty-side sentinels (bid 0, ask 1) mark the side missing
    spec, obs = state_obs(best_bid=[0.0, 0.4], best_ask=[0.6, 1.0])
    st = spec.decode(obs)
    assert st["bid_ok"].tolist() == [False, True] and st["ask_ok"].tolist() == [True, False]


# ----------------------------------------------------------------------------- mapping
def test_price_to_choice():
    am = bl.ActionMap(base_spec())
    best = np.full(9, 0.40)
    other = np.full(9, 0.45)                          # 5-tick spread: improvement cap 4 ticks
    tick = np.full(9, 0.01)
    ok = np.ones(9, bool)
    want = np.array([0.42, 0.41, 0.40, 0.39, 0.38, 0.365, 0.36, 0.33, 0.414])
    got = bl.price_to_choice(am, want, best, other, tick, +1, ok)
    # 0.365 is 3.5 behind -> tie between 2 and 4 behind -> passive (4 behind); 0.33 = 7 behind -> off
    assert got.tolist() == [IMP2, IMP1, JOIN, B1, B2, B4, B4, OFF, IMP1], got
    # ask side is the mirror image
    got = bl.price_to_choice(am, 0.85 - want, 0.85 - best, 0.85 - other, tick, -1, ok)
    assert got.tolist() == [IMP2, IMP1, JOIN, B1, B2, B4, B4, OFF, IMP1], got
    # never lock or cross: with a 1-tick spread the best is join; 2-tick spread caps at improve 1
    g = bl.price_to_choice(am, np.array([0.99, 0.99]), np.array([0.40, 0.40]), np.array([0.41, 0.42]),
                           np.full(2, 0.01), +1, np.ones(2, bool))
    assert g.tolist() == [JOIN, IMP1]
    # NaN desired, side missing, inside_only
    g = bl.price_to_choice(am, np.array([np.nan, 0.40, 0.39, 0.40]), np.full(4, 0.40), np.full(4, 0.45),
                           np.full(4, 0.01), +1, np.array([True, False, True, True]), inside_only=True)
    assert g.tolist() == [OFF, OFF, OFF, JOIN]
    g = bl.price_to_choice(am, np.array([0.395, 0.3951]), np.full(2, 0.40), np.full(2, 0.45),
                           np.full(2, 0.01), +1, np.ones(2, bool), inside_only=True)
    assert g.tolist() == [JOIN, JOIN]                # half a tick behind: join, never behind
    # fine tick: 0.001 grid near 0
    g = bl.price_to_choice(am, np.array([0.021]), np.array([0.020]), np.array([0.030]),
                           np.array([0.001]), +1, np.array([True]))
    assert g.tolist() == [IMP1]


# ----------------------------------------------------------------------------- math
def test_norm_ppf_and_digital_var():
    ps = np.array([1e-6, 0.001, 0.02, 0.1, 0.3, 0.5, 0.77, 0.975, 0.9999])
    z = bl.norm_ppf(ps)
    back = 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in z]))
    np.testing.assert_allclose(back, ps, rtol=1e-7)
    # full horizon: Bernoulli variance; tiny horizon at m = .5: f / (2 pi)
    m = np.array([0.5, 0.2, 0.93])
    np.testing.assert_allclose(bl.digital_var(m, np.full(3, 300.0), 1e9), m * (1 - m), rtol=1e-6)
    v = bl.digital_var(np.array([0.5]), np.array([1000.0]), 0.96)       # f = 0.96/960 = 0.001
    np.testing.assert_allclose(v, math.asin(0.001) / (2 * math.pi), rtol=1e-9)
    # Monte Carlo check at m = 0.3, f = 0.4
    rng = np.random.default_rng(0)
    d = bl.norm_ppf(np.array([0.3]))[0]
    f = 0.4
    w = rng.standard_normal(2_000_000)
    p_next = 0.5 * (1 + np.vectorize(math.erf)((d + math.sqrt(f) * w[:200_000]) / math.sqrt(1 - f) / math.sqrt(2)))
    tau = 1000.0
    h = f * (tau - 40)                                   # V = tau - 40 outside the TWAP window
    np.testing.assert_allclose(bl.digital_var(np.array([0.3]), np.array([tau]), h), p_next.var(), rtol=0.02)
    # TWAP window: remaining variance continuous at 60 s and -> 0 at close
    r = bl.remaining_var_units(np.array([60.0 - 1e-9, 60.0, 0.0, 30.0]))
    np.testing.assert_allclose(r[:2], [20, 20], rtol=1e-8)
    assert r[2] == 0 and abs(r[3] - 30 ** 3 / 10800) < 1e-12


# ----------------------------------------------------------------------------- policies
def test_no_quote_and_shapes():
    spec = base_spec()
    rng = np.random.default_rng(1)
    B = 4096
    bid = rng.uniform(0.0, 0.95, B).round(2)
    ask = np.minimum(bid + rng.integers(1, 8, B) * 0.01, 1.0)
    obs = make_obs(spec, seconds_to_close=rng.integers(0, 900, B), best_bid=bid, best_ask=ask,
                   inventory=rng.normal(0, 40, B), contract_len_s=rng.choice([300, 900, 14400], B),
                   p_model=np.clip(0.5 * (bid + ask) + rng.normal(0, 0.05, B), 0.001, 0.999),
                   book_valid=rng.random(B) > 0.05, const_one=1)
    for name in bl.FACTORIES:
        fn = bl.make(name, spec)
        a = fn(obs)
        assert a.shape == (B, 3) and a.dtype == np.int32, name
        assert ((a[:, :2] >= 0) & (a[:, :2] < 7)).all() and ((a[:, 2] >= 0) & (a[:, 2] < 3)).all(), name
        np.testing.assert_array_equal(a, fn(obs.copy()))                  # deterministic
        perm = rng.permutation(B)
        np.testing.assert_array_equal(a[perm], fn(obs[perm]))             # row-wise only
        bad = ~(obs[:, 6] > 0.5)
        assert (a[bad, :2] == OFF).all(), name                            # invalid book -> off
    assert (bl.make("no_quote", spec)(obs)[:, :2] == OFF).all()


def test_sym_touch():
    spec, obs = state_obs(inventory=[0, 45, 50, -50, 5], best_ask=[0.52, 0.52, 0.52, 0.52, 1.0])
    a = bl.sym_touch(spec, {"size": 0, "q_cap": 10, "cap_mode": "pull"})(obs)   # cap 50 shares
    assert a[:, 0].tolist() == [JOIN, JOIN, OFF, JOIN, JOIN]
    assert a[:, 1].tolist() == [JOIN, JOIN, JOIN, OFF, OFF]                    # last: no ask side
    a = bl.sym_touch(spec, {"size": 1, "q_cap": 10, "cap_mode": "back"})(obs)  # 10-share orders
    assert a[:, 0].tolist() == [JOIN, B4, B4, JOIN, JOIN] and (a[:, 2] == 1).all()
    a = bl.sym_touch(spec, {"size": 0, "q_cap": 100, "cap_mode": "back", "q_max": 50})(obs)
    assert a[:, 0].tolist() == [JOIN, JOIN, OFF, JOIN, JOIN]                   # hard limit pulls


def test_as_logit():
    spec, obs = state_obs(inventory=[0, 100, -100], best_bid=0.48, best_ask=0.52, seconds_to_close=600)
    st = spec.decode(obs)
    p = {"gamma": 0.02, "k": 25.0, "h_risk": 30.0, "size": 0}
    a = bl.as_logit(spec, p)(obs)
    # flat: symmetric 2/k-ish logit spread around 0.5 -> ~0.5 +- 1c -> improve one tick each side
    assert a[0, 0] == a[0, 1] == IMP1, a[0]
    # long inventory skews down: bid less aggressive, ask more aggressive; short is the mirror
    assert a[1, 0] > a[0, 0] and a[1, 1] <= a[0, 1]
    assert a[2, 1] > a[0, 1] and a[2, 0] <= a[0, 0]
    # reservation shift equals q_lots * gamma * v_x in logit units
    m = np.array([0.5])
    vx = bl.digital_var(m, st["tau"][:1], 30.0) / 0.0625
    assert 0 < vx[0] < 4
    # extreme mid: logit variance explodes, quotes leave the ladder -> off
    spec, obs = state_obs(best_bid=0.010, best_ask=0.012, seconds_to_close=5, inventory=0)
    a = bl.as_logit(spec, {"gamma": 1.0, "k": 1.0, "h_risk": 1e4, "size": 0})(obs)
    assert a[0, 0] == OFF and a[0, 1] == OFF


def test_fair_value_spec_point():
    # gap 0, 2-tick spread, mult 1: h = max(1c, 1c) = 1c -> quotes at the touch (join)
    spec, obs = state_obs(best_bid=0.49, best_ask=0.51, p_model=0.50, contract_len_s=900)
    a = bl.fair_value(spec)(obs)
    assert a[0].tolist() == [JOIN, JOIN, 0]
    # model above mid: centre (p+mid)/2 = 0.52, h = 1c + 0.5*4c = 3c -> bid 0.49 (join), ask 0.55
    spec, obs = state_obs(best_bid=0.49, best_ask=0.51, p_model=0.54)
    a = bl.fair_value(spec)(obs)
    assert a[0, 0] == JOIN and a[0, 1] == B4
    # 5m uses the logit blend 0.6 logit p + 0.32 logit mid (shrinks to 0.5)
    st = dict(bid=np.array([0.69]), ask=np.array([0.71]), bid_ok=np.array([True]), ask_ok=np.array([True]),
              tick=np.array([0.01]), q=np.array([0.0]), min_size=np.array([5.0]), p_model=np.array([0.7]),
              T=np.array([300.0]), tau=np.array([200.0]))
    _, c5, _, _ = bl.fv_quotes(st, bl.DEFAULTS["fair_value"])
    np.testing.assert_allclose(c5, bl.sigmoid(0.92 * bl.logit(0.7)), rtol=1e-12)
    st["T"] = np.array([900.0])
    _, c15, _, _ = bl.fv_quotes(st, bl.DEFAULTS["fair_value"])
    np.testing.assert_allclose(c15, 0.7)
    # pulls: tau < 30 s, |gap| > 10c
    spec, obs = state_obs(seconds_to_close=[29, 31, 200], p_model=[0.5, 0.5, 0.62])
    a = bl.fair_value(spec)(obs)
    assert a[:, 0].tolist() == [OFF, JOIN, OFF] and a[:, 1].tolist() == [OFF, JOIN, OFF]
    # skew: long inventory lowers both quotes
    spec, obs = state_obs(best_bid=0.45, best_ask=0.55, p_model=0.5, inventory=[0, 50])
    a = bl.fair_value(spec, {"k_skew": 0.005})(obs)       # 10 lots * 0.5c = 5c lower
    assert a[0].tolist()[:2] == [JOIN, JOIN] and a[1, 1] == IMP2 and a[1, 0] == B4   # bid 5 behind


def test_fv_smart():
    # wide market, centre at mid, h = 0.5*spread*mult: mult 1 joins, mult 1.5 sits behind -> off
    spec, obs = state_obs(best_bid=0.45, best_ask=0.55, p_model=0.5)
    assert bl.fv_smart(spec)(obs)[0].tolist()[:2] == [JOIN, JOIN]
    assert bl.fv_smart(spec, {"spread_mult": 1.5})(obs)[0].tolist()[:2] == [OFF, OFF]
    assert bl.fair_value(spec, {"spread_mult": 1.5})(obs)[0].tolist()[:2] == [B2, B2]
    assert bl.fv_smart(spec, {"spread_mult": 0.8})(obs)[0].tolist()[:2] == [IMP1, IMP1]
    # adverse gap: model below mid -> bid pulled, ask kept; above -> mirror
    spec, obs = state_obs(best_bid=0.45, best_ask=0.55, p_model=[0.44, 0.56], seconds_to_close=500)
    a = bl.fv_smart(spec, {"g_adv": 0.03, "spread_mult": 0.5})(obs)
    assert a[0, 0] == OFF and a[0, 1] != OFF and a[1, 1] == OFF and a[1, 0] != OFF


def test_feat_meta_decode_on_real_rows():
    """Decode real feat.bin rows (train split only) and compare with the raw build columns."""
    build = Path(__file__).resolve().parent.parent / "data" / "build"
    if not (build / "feat.bin").exists():
        print("   (skipped: no feat.bin)")
        return
    meta = json.loads((build / "meta.json").read_text())
    fm = json.loads((build / "feat_meta.json").read_text())
    ticks = np.memmap(build / "ticks.bin", np.float32, "r", shape=(meta["N"], meta["F"]))
    feat = np.memmap(build / "feat.bin", np.float32, "r", shape=(fm["N"], fm["G"]))
    n_train = meta["splits"]["counts"]["train"]["rows"]
    rows = np.sort(np.random.default_rng(0).choice(n_train, 200_000, replace=False))
    n_obs = fm["n_obs_features"]
    obs = np.concatenate([feat[rows, :n_obs], np.zeros((rows.size, 1), np.float32)], axis=1)
    spec = load_spec(build / "feat_meta.json")
    spec.features["inventory"] = {"index": n_obs}
    st = spec.decode(obs, need_fair=True)
    c = meta["column_index"]
    bid, ask = ticks[rows, c["bid_px_0"]].astype(float), ticks[rows, c["ask_px_0"]].astype(float)
    valid = ticks[rows, c["book_valid"]] > 0.5
    two = valid & (bid > 0) & (ask < 1)
    assert (st["bid_ok"] == (valid & (bid > 0))).all() and (st["ask_ok"] == (valid & (ask < 1))).all()
    assert np.isfinite(st["bid"]).sum() == two.sum()
    np.testing.assert_allclose(st["bid"][two], bid[two], atol=1e-6)
    np.testing.assert_allclose(st["ask"][two], ask[two], atol=1e-6)
    np.testing.assert_array_equal(st["tau"], ticks[rows, c["seconds_to_close"]])
    np.testing.assert_allclose(st["p_model"], feat[rows, fm["column_index"]["p_model"]], atol=1.0001e-4)   # p clipped to [1e-4, 1-1e-4]
    lens = np.repeat(np.array([300.0, 900.0, 14400.0]), 1)
    assert set(np.unique(st["T"])) <= set(lens.tolist())
    print(f"   decoded {rows.size} train rows; two-sided {two.mean():.3f}")


def test_uncross_joint_quotes():
    spec = base_spec()
    am = bl.ActionMap(spec)
    # 4-tick spread .60/.64; both sides want .62 -> one side backs off to keep bid < ask
    spec_, obs = state_obs(best_bid=0.60, best_ask=0.64)
    st = spec.decode(obs)
    a = bl.to_actions(am, st, np.array([0.619]), np.array([0.6225]), 0)
    assert a[0, 0] == IMP2 and a[0, 1] == IMP1, a          # ask is less hurt by moving: .63
    a = bl.to_actions(am, st, np.array([0.6175]), np.array([0.621]), 0)
    assert a[0, 0] == IMP1 and a[0, 1] == IMP2, a          # bid backs off to .61
    # crossing desires on a 2-tick spread: at most one side improves
    spec_, obs = state_obs(best_bid=0.60, best_ask=0.62)
    st = spec.decode(obs)
    a = bl.to_actions(am, st, np.array([0.63]), np.array([0.59]), 0)
    pb = 0.60 + np.array([0, 2, 1, 0, -1, -2, -4])[a[0, 0]] * 0.01
    pa = 0.62 - np.array([0, 2, 1, 0, -1, -2, -4])[a[0, 1]] * 0.01
    assert pb < pa - 1e-9 and {a[0, 0], a[0, 1]} == {IMP1, JOIN}, a
    # inside_only: a side pushed behind the touch goes off
    spec_, obs = state_obs(best_bid=0.60, best_ask=0.61)
    st = spec.decode(obs)
    a = bl.to_actions(am, st, np.array([0.605]), np.array([0.605]), 0, inside_only=True)
    assert a[0].tolist()[:2] == [JOIN, JOIN]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
