"""Shared access to the frozen WTI estimation sample and its subsamples.

The definitions live in ``series.py`` (series ``wti``). This module keeps the
names other scripts and the trading project import.
"""

from series import SERIES

WTI = SERIES["wti"]
FROZEN_DATA = WTI.frozen_file
HOLDOUT_DATA = (
    WTI.frozen_file.parent
    / "wti_CL=F_holdout_2025-01-02_to_2026-08-31_downloaded_2026-09-01.csv"
)
CROSS_ZERO_DATES = WTI.excluded_dates
TRAIN_END = WTI.train_end
TEST_START = WTI.test_start
EXPECTED_OBSERVATIONS = WTI.expected_observations


def load_prices(path=FROZEN_DATA):
    """Load the frozen daily WTI close series."""
    return WTI.load_prices(path)


def load_changes(path=FROZEN_DATA):
    """Load daily WTI price changes in dollars per barrel."""
    return WTI.load_observations(path)


def stability_samples(changes):
    """Return the four samples registered for gate G4."""
    return WTI.stability_samples(changes)


def train_test_split(changes):
    """Return the training and test samples registered for gate G5."""
    return WTI.train_test_split(changes)
