"""Forecast losses, Diebold-Mariano, Holm correction and pooled tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def qlike(forecast_var, proxy):
    """Patton's QLIKE loss, zero when the forecast equals the proxy."""
    f = np.asarray(forecast_var, dtype=float)
    p = np.asarray(proxy, dtype=float)
    ratio = p / f
    return ratio - np.log(ratio) - 1.0


def squared_error(forecast_var, proxy):
    return (np.asarray(proxy, dtype=float) - np.asarray(forecast_var, dtype=float)) ** 2


def newey_west_variance(d, lags=None):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = d - d.mean()
    gamma0 = (d @ d) / n
    total = gamma0
    for k in range(1, lags + 1):
        gamma = (d[k:] @ d[:-k]) / n
        total += 2 * (1 - k / (lags + 1)) * gamma
    return total / n


def diebold_mariano(loss_a, loss_b, lags=None):
    """Test that ``a`` and ``b`` have equal expected loss.

    Negative statistic means ``a`` has the lower loss. Returns the mean loss
    difference, the statistic and the two-sided p-value.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if len(d) < 30:
        return {"mean_diff": np.nan, "stat": np.nan, "p_value": np.nan, "n": int(len(d))}
    var = newey_west_variance(d, lags)
    stat = d.mean() / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(stat))) if np.isfinite(stat) else np.nan
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p_value": float(p), "n": int(len(d))}


def holm(p_values):
    """Holm step-down adjusted p-values, NaN preserved."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[i])
        running = max(running, adjusted)
        out[i] = running
    return out


def expected_false_positives(n_tests, level=0.05):
    return n_tests * level
