"""Walk-forward regime signal: parameters refitted once a year on past data."""

from __future__ import annotations

import numpy as np
import pandas as pd

import research_link  # noqa: F401
from backtest import hmm_expected_vol


def walk_forward_expected_vol(changes, fitter, first_year, refit_month=1):
    """Expected volatility for each day using only earlier data.

    ``changes`` is a Series indexed by date. For each calendar year from
    ``first_year`` on, ``fitter(history)`` is called once with every change
    dated before 1 January of that year and must return a fitted model with
    ``predictive_proba``, ``mu`` and ``stdev``. The signal for a day in that
    year is the model's one-step-ahead predictive standard deviation, which
    uses changes up to the previous day only. Days before ``first_year`` get
    NaN. ``refit_month`` is kept at 1; the yearly cut is the registered rule.
    """
    if refit_month != 1:
        raise ValueError("only a January refit is registered")
    index = changes.index
    expected = pd.Series(np.nan, index=index, name="expected_vol")
    years = sorted({year for year in index.year if year >= first_year})
    for year in years:
        cut = pd.Timestamp(year=year, month=1, day=1)
        history = changes[index < cut]
        model = fitter(history.to_numpy())
        through_year = changes[index < pd.Timestamp(year=year + 1, month=1, day=1)]
        signal = hmm_expected_vol(model, through_year.to_numpy())
        in_year = through_year.index.year == year
        expected.loc[through_year.index[in_year]] = signal[in_year]
    return expected
