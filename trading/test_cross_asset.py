"""Tests for the cross-asset modules: no look-ahead, recovery, guards."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import assets
import stats_tools
import strategies
import vol_models

PROJECT_DIR = Path(__file__).resolve().parent


def _simulated_garch(n, omega=0.05, alpha=0.08, beta=0.90, nu=6.0, seed=1):
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    x = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    scale = np.sqrt((nu - 2) / nu)
    for t in range(n):
        if t:
            h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]
        x[t] = np.sqrt(h[t]) * rng.standard_t(nu) * scale
    return x, h


def test_garch_recovers_parameters():
    x, _ = _simulated_garch(6000)
    m = vol_models.GarchT().fit(x)
    assert abs(m.alpha_ - 0.08) < 0.03
    assert abs(m.beta_ - 0.90) < 0.04
    assert 3.5 < m.nu_ < 12


def test_garch_forecast_uses_past_only():
    x, _ = _simulated_garch(1500, seed=2)
    m = vol_models.GarchT().fit(x[:1000])
    base = m.forecast(x)
    x2 = x.copy()
    x2[1200:] += 5.0
    changed = m.forecast(x2)
    assert np.allclose(base[:1201], changed[:1201])
    assert not np.allclose(base[1201:], changed[1201:])


def test_har_forecast_uses_past_only():
    rng = np.random.default_rng(3)
    proxy = rng.gamma(2.0, 0.5, size=800)
    m = vol_models.HAR().fit(proxy[:600])
    base = m.forecast(proxy)
    proxy2 = proxy.copy()
    proxy2[700:] *= 10
    changed = m.forecast(proxy2)
    assert np.allclose(base[:701], changed[:701], equal_nan=True)


def test_har_recovers_coefficients():
    rng = np.random.default_rng(4)
    n = 6000
    proxy = np.empty(n)
    proxy[:22] = 1.0
    for t in range(22, n):
        d, w, mo = proxy[t - 1], proxy[t - 5:t].mean(), proxy[t - 22:t].mean()
        proxy[t] = max(0.1 + 0.3 * d + 0.3 * w + 0.3 * mo + rng.normal(0, 0.2), 0.01)
    m = vol_models.HAR().fit(proxy)
    assert np.allclose(m.beta_[1:], [0.3, 0.3, 0.3], atol=0.08)


def test_ewma_is_lagged():
    r = np.zeros(50)
    r[20] = 10.0
    v = vol_models.ewma_variance(r)
    assert v[20] == v[19]
    assert v[21] > v[20]


def test_yearly_walk_forward_never_trains_on_scored_year():
    index = pd.bdate_range("2012-01-02", "2016-12-30")
    seen = []

    def step(train, through):
        seen.append((index[train].max(), index[through].max()))
        return np.zeros(int(through.sum()))

    vol_models.yearly_walk_forward(index, 2014, 2016, step)
    for train_max, through_max in seen:
        assert train_max.year == through_max.year - 1


def test_walk_forward_hmm_signal_uses_past_only():
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2010-01-04", periods=1600)
    r = pd.Series(rng.standard_t(5, size=len(index)), index=index)
    out, params = vol_models.walk_forward_hmm(r, first_year=2015, last_year=2015, n_starts=2)
    r2 = r.copy()
    late = index >= "2015-07-01"
    r2[late] = r2[late] * 4 + 3
    out2, _ = vol_models.walk_forward_hmm(r2, first_year=2015, last_year=2015, n_starts=2)
    before = (index.year == 2015) & ~late
    assert np.allclose(out.loc[before, "hmm_var"], out2.loc[before, "hmm_var"])
    # The first day after the change still uses the unchanged history.
    first_late = index[late][0]
    assert np.isclose(out.loc[first_late, "hmm_var"], out2.loc[first_late, "hmm_var"])
    assert params.loc[0, "year"] == 2015 and params.loc[0, "train_obs"] == int((index.year < 2015).sum())


def test_tsmom_sign_uses_past_only():
    r = np.zeros(600)
    r[300] = 50.0
    s = strategies.tsmom_sign(r, lookback=252)
    assert s[300] <= 0 or np.isnan(s[300]) or s[300] == 0
    assert s[301] == 1.0
    assert s[552] == 1.0 and s[553] == 0.0


def test_regime_rules():
    p = np.array([0.1, 0.6, np.nan, 0.5])
    assert np.array_equal(strategies.regime_long_binary(p)[:2], [1.0, 0.0])
    assert np.isnan(strategies.regime_long_binary(p)[2])
    assert strategies.regime_long_binary(p)[3] == 0.0
    assert np.allclose(strategies.regime_long_scaled(p)[[0, 1]], [0.9, 0.4])


def test_vol_managed_uses_past_only():
    r = np.ones(100)
    v = np.ones(100)
    pos = strategies.vol_managed(r, v)
    assert np.isnan(pos[0]) and np.allclose(pos[1:], 1.0)


def test_qlike_zero_at_truth_and_dm_null():
    f = np.full(500, 2.0)
    assert np.allclose(stats_tools.qlike(f, f), 0.0)
    rng = np.random.default_rng(6)
    a = rng.gamma(2, 1, 5000)
    b = rng.gamma(2, 1, 5000)
    res = stats_tools.diebold_mariano(a, b)
    assert res["p_value"] > 0.01


def test_holm():
    adj = stats_tools.holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_returns_and_units():
    r = assets.load_returns("sp500")
    assert r.index.is_monotonic_increasing and r.index[-1] <= pd.Timestamp(assets.DEV_END)
    assert 0.5 < r.std() < 2.0
    w = assets.load_returns("wti")
    assert len(w) == 3770


def test_no_script_reads_a_holdout():
    pattern = re.compile(r"holdout=True|_holdout_\d")
    for path in sorted(PROJECT_DIR.glob("*.py")):
        if path.name in {"03_download_assets.py", "test_cross_asset.py"}:
            continue
        text = path.read_text()
        assert not pattern.search(text), f"{path.name} names a holdout file"
    for path in sorted(PROJECT_DIR.glob("*.py")):
        if path.name.startswith("test_") or path.name == "03_download_assets.py":
            continue
        assert "frozen_file(" not in path.read_text() or path.name == "assets.py"
