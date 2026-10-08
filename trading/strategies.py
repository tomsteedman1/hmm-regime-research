"""Position rules on top of the forecasts in ``vol_models``.

Every rule returns the position held over day ``t``, decided from signals
whose element ``t`` uses data to ``t - 1``. P&L and costs come from
``backtest.daily_pnl``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import vol_target_positions

TARGET_VOL = 1.0
MAX_LEVERAGE = 3.0
COSTS = {"base": 0.01, "double": 0.02}
CALM_THRESHOLD = 0.5
TSMOM_LOOKBACK = 252


def vol_target(expected_var, target=TARGET_VOL, cap=MAX_LEVERAGE):
    return vol_target_positions(np.sqrt(np.asarray(expected_var, dtype=float)), target, cap)


def regime_long_binary(p_high, threshold=CALM_THRESHOLD):
    """One unit long when the calm state is more likely than not, else flat."""
    p = np.asarray(p_high, dtype=float)
    pos = np.where(1 - p > threshold, 1.0, 0.0)
    pos[~np.isfinite(p)] = np.nan
    return pos


def regime_long_scaled(p_high):
    """Long by the one-step probability of the calm state."""
    p = np.asarray(p_high, dtype=float)
    return 1.0 - p


def vol_managed(returns, ewma_var, cap=MAX_LEVERAGE):
    """Moreira and Muir: position = trailing mean variance / current variance.

    The numerator is the expanding mean of squared returns to ``t - 1``, so
    the rule uses no future information. Capped at ``cap``.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    trailing = (r ** 2).expanding().mean().shift(1).to_numpy()
    var = np.asarray(ewma_var, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = trailing / var
    pos = np.minimum(pos, cap)
    pos[~np.isfinite(var) | ~np.isfinite(trailing)] = np.nan
    return pos


def tsmom_sign(returns, lookback=TSMOM_LOOKBACK):
    """Sign of the past ``lookback`` returns, using days to ``t - 1``."""
    r = pd.Series(np.asarray(returns, dtype=float))
    past = r.rolling(lookback).sum().shift(1).to_numpy()
    sign = np.sign(past)
    sign[~np.isfinite(past)] = np.nan
    return sign


def tsmom(returns, expected_var, p_high=None, cut=None, lookback=TSMOM_LOOKBACK,
          target=TARGET_VOL, cap=MAX_LEVERAGE):
    """Time-series momentum sized to target volatility.

    With ``cut`` set, the position is zero when ``p_high`` exceeds ``cut``.
    """
    sign = tsmom_sign(returns, lookback)
    size = vol_target(expected_var, target, cap)
    pos = sign * size
    if cut is not None:
        p = np.asarray(p_high, dtype=float)
        pos = np.where(p > cut, 0.0, pos)
        pos[~np.isfinite(p)] = np.nan
    return pos


def equal_risk_portfolio(net_by_asset: dict[str, pd.Series]) -> pd.Series:
    """Average of net P&L series that each target one unit of daily vol."""
    frame = pd.DataFrame(net_by_asset)
    return frame.mean(axis=1, skipna=True).where(frame.notna().sum(axis=1) > 0)
