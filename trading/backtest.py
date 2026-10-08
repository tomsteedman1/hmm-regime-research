"""Volatility-target backtest engine.

Pure functions only. Nothing here fits a model. The conventions:

- ``changes[t]`` is the price change over day ``t`` in dollars per barrel.
- ``position[t]`` is the position held over day ``t``. It must be decided
  from information available at the close of day ``t - 1``.
- ``pnl[t] = position[t] * changes[t] - cost_per_unit * |position[t] - position[t-1]|``.

Positions are multiples of one barrel per unit of target volatility, so the
P&L is in dollars and Sharpe ratios are scale-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Expected volatility signals
# ----------------------------------------------------------------------

def hmm_expected_vol(model, changes):
    """One-step-ahead predictive standard deviation under a fitted HMM.

    Element ``t`` uses ``changes[:t]`` only. It is the standard deviation of
    the predictive mixture for day ``t``, with state weights
    ``P(z_t | x_1..x_{t-1})``.
    """
    values = np.asarray(changes, dtype=float)
    weights = model.predictive_proba(values)
    means = model.mu
    variances = model.stdev ** 2
    mixture_mean = weights @ means
    second_moment = weights @ (variances + means ** 2)
    return np.sqrt(np.maximum(second_moment - mixture_mean ** 2, 1e-12))


def realised_expected_vol(changes, span=21):
    """Exponentially weighted realised volatility, lagged one day.

    Element ``t`` uses ``changes[:t]`` only. The first element is NaN because
    there is no history; callers start scoring after the warm-up.
    """
    series = pd.Series(np.asarray(changes, dtype=float))
    variance = (series ** 2).ewm(span=span, adjust=False).mean()
    return np.sqrt(variance.shift(1).to_numpy())


# ----------------------------------------------------------------------
# Positions, P&L and metrics
# ----------------------------------------------------------------------

def vol_target_positions(expected_vol, target_vol, max_leverage):
    """Position = target / expected volatility, capped, NaN where no signal."""
    expected = np.asarray(expected_vol, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        position = target_vol / expected
    position = np.minimum(position, max_leverage)
    position[~np.isfinite(expected)] = np.nan
    return position


def apply_roll_treatment(changes, roll_mask, treatment):
    """Return the P&L change series under one of the registered treatments.

    ``raw`` leaves the series alone. ``zeroed`` sets the change on roll days
    to zero. ``dropped`` returns NaN on roll days so that the metrics ignore
    those rows. The position series is never altered, so costs still count.
    """
    values = np.asarray(changes, dtype=float).copy()
    mask = np.asarray(roll_mask, dtype=bool)
    if treatment == "raw":
        return values
    if treatment == "zeroed":
        values[mask] = 0.0
        return values
    if treatment == "dropped":
        values[mask] = np.nan
        return values
    raise ValueError(f"unknown roll treatment: {treatment}")


def daily_pnl(changes, position, cost_per_unit):
    """Net daily P&L given positions decided the previous close."""
    values = np.asarray(changes, dtype=float)
    pos = np.asarray(position, dtype=float)
    if len(values) != len(pos):
        raise ValueError("changes and position must have the same length")
    previous = np.concatenate([[0.0], pos[:-1]])
    previous = np.where(np.isnan(previous), 0.0, previous)
    held = np.where(np.isnan(pos), 0.0, pos)
    turnover = np.abs(held - previous)
    gross = held * values
    net = gross - cost_per_unit * turnover
    # A NaN change (dropped roll day) removes the row from the statistics but
    # the cost of the trade made that day is still real; keep it on the row.
    return pd.DataFrame(
        {
            "position": held,
            "gross": gross,
            "turnover": turnover,
            "cost": cost_per_unit * turnover,
            "net": net,
        }
    )


def sharpe(net):
    values = np.asarray(net, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or values.std(ddof=1) == 0:
        return np.nan
    return values.mean() / values.std(ddof=1) * np.sqrt(TRADING_DAYS)


def max_drawdown(net):
    values = np.asarray(net, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    path = np.cumsum(values)
    peak = np.maximum.accumulate(np.concatenate([[0.0], path]))[1:]
    return float((path - peak).min())


def metrics(pnl_frame):
    """Summary statistics for one strategy on one treatment."""
    net = pnl_frame["net"].to_numpy()
    valid = np.isfinite(net)
    years = valid.sum() / TRADING_DAYS
    return {
        "days": int(valid.sum()),
        "sharpe": sharpe(net),
        "annual_return": float(net[valid].mean() * TRADING_DAYS),
        "annual_vol": float(net[valid].std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "daily_vol": float(net[valid].std(ddof=1)),
        "max_drawdown": max_drawdown(net),
        "annual_turnover": float(pnl_frame["turnover"].to_numpy()[valid].sum() / years),
        "total_cost": float(pnl_frame["cost"].to_numpy()[valid].sum()),
        "mean_position": float(pnl_frame["position"].to_numpy()[valid].mean()),
    }


# ----------------------------------------------------------------------
# Stationary block bootstrap for a Sharpe difference
# ----------------------------------------------------------------------

def stationary_bootstrap_indices(length, mean_block, rng):
    """Politis and Romano stationary bootstrap index sample."""
    p = 1.0 / mean_block
    restart = rng.random(length) < p
    restart[0] = True
    starts = rng.integers(length, size=length)
    positions = np.arange(length)
    # Index of the most recent restart at or before each position.
    last_restart = np.maximum.accumulate(np.where(restart, positions, 0))
    return (starts[last_restart] + positions - last_restart) % length


def sharpe_difference_bootstrap(net_a, net_b, n_draws=2000, mean_block=21, seed=20260902):
    """Bootstrap the Sharpe difference ``a - b`` on jointly resampled days.

    Returns the observed difference, the one-sided p-value for the null that
    ``a`` is no better than ``b``, and a 95% percentile interval.
    """
    a = np.asarray(net_a, dtype=float)
    b = np.asarray(net_b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    observed = sharpe(a) - sharpe(b)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for d in range(n_draws):
        idx = stationary_bootstrap_indices(len(a), mean_block, rng)
        draws[d] = sharpe(a[idx]) - sharpe(b[idx])
    p_value = float(np.mean(draws <= 0.0))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "observed": float(observed),
        "p_value": p_value,
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": n_draws,
        "mean_block": mean_block,
    }
