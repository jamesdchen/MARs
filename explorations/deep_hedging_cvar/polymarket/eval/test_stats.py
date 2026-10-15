"""Tests for stats.py on synthetic data with known answers. Small and CPU-light.

Run either way (pytest optional; numpy is the only dependency):
  python3 explorations/deep_hedging_cvar/polymarket/eval/test_stats.py
  python3 -m pytest -q explorations/deep_hedging_cvar/polymarket/eval/test_stats.py
"""
import hashlib
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402


def close(a, b, tol=1e-9):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def ar1(n, phi, rng, mu=0.0, sigma=1.0):
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + mu


# ---------------------------------------------------------------- bootstrap

def _coverage(kind, block_len, phi=0.6, reps=150, n=400, n_boot=300):
    rng = np.random.default_rng(123)
    hit = 0
    for r in range(reps):
        x = ar1(n, phi, rng, mu=0.3)
        res = S.block_bootstrap(x, "mean", block_len=block_len, n_boot=n_boot,
                                kind=kind, alpha=0.10, seed=r)
        hit += res["lo"] <= 0.3 <= res["hi"]
    return hit / reps


def test_block_bootstrap_coverage_near_nominal_on_ar1():
    for kind in ["stationary", "moving", "circular"]:
        cov = _coverage(kind, block_len=None)  # Politis-White block length
        assert 0.80 <= cov <= 0.97, (kind, cov)  # nominal 0.90; slight undercoverage is normal


def test_iid_bootstrap_undercovers_ar1():
    # variance inflation (1+phi)/(1-phi)=4 -> iid CI is ~half as wide -> ~0.59 coverage
    cov = _coverage("iid", block_len=1)
    assert cov < 0.75, cov


def test_optimal_block_length_grows_with_dependence():
    rng = np.random.default_rng(0)
    b0 = S.optimal_block_length(rng.normal(size=2000))
    b5 = S.optimal_block_length(ar1(2000, 0.5, rng))
    b9 = S.optimal_block_length(ar1(2000, 0.9, rng))
    assert b0 < 3 and b0 < b5 < b9


def test_stationary_indices_are_valid_and_contiguous():
    rng = np.random.default_rng(1)
    idx = S._resample_indices(50, 8, 20, "stationary", rng)
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    steps = (np.diff(idx, axis=1) % 50 == 1).mean()  # ~1 - 1/8 continue the block
    assert 0.8 < steps < 0.95


def test_bootstrap_2d_rows_resampled_jointly():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 2))
    x[:, 1] = 2 * x[:, 0]
    res = S.block_bootstrap(x, lambda m: m[:, 1].sum() - 2 * m[:, 0].sum(), block_len=5, n_boot=50)
    assert abs(res["lo"]) < 1e-9 and abs(res["hi"]) < 1e-9


def test_time_bins_zero_fill():
    t0, s = S.time_bins([0, 10, 7300], [1.0, 2.0, 5.0], 3600, t_start=0, t_end=4 * 3600)
    np.testing.assert_allclose(s, [3.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(t0, [0, 3600, 7200, 10800])


# ---------------------------------------------------------------- paired difference

def _paired_case(seed, hours=96):
    rng = np.random.default_rng(seed)
    # 8 coins x hourly markets; a common crypto factor per hour hits both methods
    t = np.repeat(np.arange(hours) * 3600.0 + 1800, 8)
    common = np.repeat(ar1(hours, 0.5, rng, sigma=3.0), 8)
    a = common + 0.5 + rng.normal(0, 0.3, t.size)  # true edge of A over B: 0.5 per market
    b = common + rng.normal(0, 0.3, t.size)
    return t, a, b


def test_paired_difference_covers_truth_and_beats_unpaired():
    hours, hits = 96, 0
    for seed in range(30):
        t, a, b = _paired_case(seed, hours)
        res = S.paired_difference_ci(a, b, unit_time=t, bin_seconds=3600, t_start=0,
                                     t_end=hours * 3600, n_boot=200, seed=seed, min_block_len=2)
        hits += res["lo"] <= 0.5 <= res["hi"]
        assert res["lo"] > 0 and res["corr_ab"] > 0.9
    assert hits >= 24, hits  # nominal 95% of 30 = 28.5
    _, sa = S.time_bins(t, np.column_stack([a, np.ones_like(a)]), 3600, 0, hours * 3600)
    unpaired = S.block_bootstrap(sa, lambda m: m[:, 0].sum() / m[:, 1].sum(), n_boot=200)
    assert (res["hi"] - res["lo"]) < 0.3 * (unpaired["hi"] - unpaired["lo"])


def test_paired_difference_series_mode():
    rng = np.random.default_rng(8)
    a = ar1(200, 0.3, rng) + 1.0
    res = S.paired_difference_ci(a, a - 1.0, n_boot=100)
    assert abs(res["estimate"] - 1.0) < 1e-12 and res["se"] < 1e-12


# ---------------------------------------------------------------- metrics

def test_cvar_known_values():
    x = np.arange(1, 101, dtype=float)  # worst 5% = 1..5
    assert close(S.cvar(x, 0.95), 3.0)
    assert close(S.cvar([-10.0, 0.0, 0.0, 0.0], 0.75), -10.0)
    assert close(S.cvar([-10.0, 0.0, 0.0, 0.0], 0.5), -5.0)  # fractional handling
    assert close(S.cvar(np.arange(10.0), 0.85), (0 + 0.5 * 1) / 1.5)  # k = 1.5 obs


def test_sharpe():
    assert close(S.sharpe([1.0, 3.0]), 2.0 / np.sqrt(2.0))
    assert np.isnan(S.sharpe([1.0, 1.0]))


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(3)
    x = rng.normal(0.1, 1.0, 96)
    d1 = S.deflated_sharpe(x, 1, 0.01)["dsr"]
    d100 = S.deflated_sharpe(x, 100, 0.01)["dsr"]
    assert 0 < d100 < d1 < 1


# ---------------------------------------------------------------- attribution

def _synthetic_book(rng, n_markets=30, fills_per=40):
    mk = np.repeat(np.arange(n_markets), fills_per)
    n = mk.size
    side = rng.choice([-1, 1], n)
    mid = rng.uniform(0.1, 0.9, n)
    price = np.clip(mid - side * rng.uniform(0.0, 0.02, n), 0.01, 0.99)
    fills = dict(market=mk, side=side, qty=rng.uniform(1, 50, n), price=price, mid=mid,
                 mid_5=np.clip(mid + rng.normal(0, 0.01, n), 0, 1),
                 mid_30=np.clip(mid + rng.normal(0, 0.02, n), 0, 1),
                 mid_60=np.clip(mid + rng.normal(0, 0.03, n), 0, 1),
                 fee=rng.uniform(0, 0.05, n), rebate=rng.uniform(0, 0.02, n))
    fills["mid_60"][rng.random(n) < 0.05] = np.nan  # window closed before +60s
    markets = dict(market=np.arange(n_markets), settlement=rng.integers(0, 2, n_markets).astype(float),
                   reward=rng.uniform(0, 1, n_markets))
    return fills, markets


def test_attribution_components_sum_to_total_pnl():
    rng = np.random.default_rng(11)
    fills, markets = _synthetic_book(rng)
    res = S.attribute_pnl(fills, markets, horizon=60)
    tot = res["totals"]
    comp = sum(tot[k] for k in ["spread_capture", "adverse_selection", "inventory",
                                "fees", "rebates", "rewards"])
    # direct cash accounting, written independently here
    sq = fills["side"] * fills["qty"]
    inv = np.bincount(fills["market"], weights=sq, minlength=30)
    cash = (-sq * fills["price"] - fills["fee"] + fills["rebate"]).sum() + markets["reward"].sum()
    direct = cash + (inv * markets["settlement"]).sum()
    assert close(comp, direct)
    assert close(tot["total"], direct)
    assert res["identity_error"] < 1e-9
    np.testing.assert_allclose(res["inventory"], inv)
    # per-fill totals + rewards also reconcile
    assert close(res["per_fill"]["total"].sum() + markets["reward"].sum(), direct)
    # primary-horizon markout is the adverse-selection component
    assert close(res["markouts"][60], tot["adverse_selection"])


def test_attribution_hand_example():
    # buy 10 @ 0.48 (mid 0.50 -> 0.45 after 60s), sell 4 @ 0.47 (mid 0.46 -> 0.46);
    # market settles Up = 1. fee 0.1 on first fill, reward 0.3.
    fills = dict(market=np.array([0, 0]), side=np.array(["buy", "sell"]), qty=np.array([10.0, 4.0]),
                 price=np.array([0.48, 0.47]), mid=np.array([0.50, 0.46]),
                 mid_60=np.array([0.45, 0.46]), fee=np.array([0.1, 0.0]))
    markets = dict(market=np.array([0]), settlement=np.array([1.0]), reward=np.array([0.3]),
                   final_inventory=np.array([6.0]))
    t = S.attribute_pnl(fills, markets)["totals"]
    assert close(t["spread_capture"], 10 * 0.02 + (-4) * (0.46 - 0.47))  # 0.24
    assert close(t["adverse_selection"], 10 * -0.05 + 0.0)               # -0.50
    assert close(t["inventory"], 10 * 0.55 - 4 * 0.54)                   # 3.34
    assert close(t["fees"], -0.1) and close(t["rewards"], 0.3)
    assert close(t["total"], -4.8 + 1.88 - 0.1 + 0.3 + 6.0)              # 3.28


def test_attribution_nan_markout_uses_settlement_and_mid_close_split():
    fills = dict(market=np.array([5]), side=np.array([1]), qty=np.array([2.0]),
                 price=np.array([0.9]), mid=np.array([0.92]), mid_60=np.array([np.nan]))
    markets = dict(market=np.array([5, 6]), settlement=np.array([0.0, 1.0]),
                   reward=np.array([0.0, 0.7]), mid_close=np.array([0.8, 0.5]))
    r = S.attribute_pnl(fills, markets)
    assert close(r["totals"]["adverse_selection"], 2 * (0.0 - 0.92))
    assert close(r["totals"]["inventory"], 0.0)
    assert close(r["totals"]["total"], -1.8 + 0.7)
    pm = r["per_market"]
    np.testing.assert_allclose(pm["inventory_carry"] + pm["resolution"], pm["inventory"])
    assert list(r["market_ids"]) == [5, 6]


def test_attribution_flags_inventory_mismatch():
    fills = dict(market=np.array([1, 1]), side=np.array([1, 1]), qty=np.array([1.0, 2.0]),
                 price=np.array([0.5, 0.5]), mid=np.array([0.5, 0.5]), mid_60=np.array([0.5, 0.5]))
    markets = dict(market=np.array([1]), settlement=np.array([1.0]), final_inventory=np.array([2.0]))
    assert S.attribute_pnl(fills, markets)["inventory_mismatch"] == [1]


def test_down_frame_conversion_is_pnl_equivalent():
    # buying Down @ 0.3 (mid_down 0.32) == selling Up @ 0.7 (mid_up 0.68)
    down = dict(market=np.array([0]), side=np.array([1]), qty=np.array([10.0]),
                price=np.array([0.3]), mid=np.array([0.32]), mid_60=np.array([0.35]))
    up = S.to_up_frame(down, is_down=np.array([True]))
    assert up["side"][0] == -1 and close(up["price"][0], 0.7)
    assert close(up["mid_60"][0], 0.65)
    # Up wins -> Down worthless: P&L of the Down buy is -3.0
    r = S.attribute_pnl(up, dict(market=np.array([0]), settlement=np.array([1.0])))
    assert close(r["totals"]["total"], -3.0)


def test_flag_pnl_share():
    pnl = np.array([1.0, 2.0, 3.0, -1.0])
    out = S.flag_pnl_share(pnl, dict(crossed=np.array([0, 0, 1, 0], bool),
                                     last30s=np.array([0, 1, 1, 0], bool)))
    assert close(out["crossed"]["share_of_total"], 0.6)
    assert close(out["_any"]["pnl_excluding"], 0.0)


# ---------------------------------------------------------------- concentration

def test_concentration_known_values():
    c = S.concentration([10.0, 0.0, 0.0, 0.0, -5.0, 5.0], ks=(1, 2))
    assert c["total"] == 10.0 and c["gross_profit"] == 15.0
    assert close(c["top1_share_of_total"], 1.0)
    assert close(c["ex_top1"], 0.0)
    assert close(c["top2_share_of_gross_profit"], 1.0)
    assert c["n_for_half_profit"] == 1
    assert close(c["n_effective"], 1 / (0.5 ** 2 + 2 * 0.25 ** 2))
    flat = S.concentration(np.ones(100), ks=(5,))
    assert close(flat["top5_share_of_total"], 0.05) and close(flat["n_effective"], 100)


# ---------------------------------------------------------------- splits / hashing

def test_assign_split_purges_straddlers():
    day = 86400.0
    b = [0, day, 2 * day, 3 * day]
    start = np.array([100, day - 600, day + 10, 2 * day - 4 * 3600, 2 * day + 5])
    end = start + np.array([300, 900, 300, 4 * 3600, 300])
    lab = S.assign_split(start, end, b, ["train", "val", "test"])
    assert list(lab) == ["train", None, "val", "val", "test"]
    lab_e = S.assign_split(start, end, b, ["train", "val", "test"], embargo=60)
    assert lab_e[2] is None and lab_e[4] is None


def test_sha256_helpers():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"polymarket" * 1000)
        p = f.name
    assert S.sha256_file(p) == hashlib.sha256(b"polymarket" * 1000).hexdigest()
    os.unlink(p)
    a = np.arange(6, dtype=np.int64)
    assert S.sha256_array(a) == S.sha256_array(a.copy())
    assert S.sha256_array(a) != S.sha256_array(a.astype(np.int32))
    assert S.sha256_array(a) != S.sha256_array(a.reshape(2, 3))
    assert S.sha256_json({"b": 1, "a": [1, 2]}) == S.sha256_json({"a": [1, 2], "b": 1})


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
