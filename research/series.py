"""Registered input series for the model-comparison pipeline.

A series names a frozen price file, the observation built from it, the
subsamples gate G4 uses and the train and test split gate G5 uses. The
pipeline scripts (``06_model_comparison.py``, ``07_selected_model_bootstrap.py``)
take a series name and read everything else from here, so a new market or a
new observation is one new entry, registered before it is fitted.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from roll_dates import cl_roll_days


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_ROOT = PROJECT_DIR / "outputs"


def price_change(price):
    """Daily change in price units."""
    return price.diff()


ASINH_SCALE = 10.0


def asinh_return(price):
    """100 [asinh(P_t / 10) - asinh(P_{t-1} / 10)]: defined below zero, log-like above $30."""
    return 100.0 * np.arcsinh(price / ASINH_SCALE).diff()


def asinh_log_jacobian(price):
    """log g'(P_t), with g'(P) = 100 / sqrt(P^2 + 10^2), the derivative of y_t in x_t."""
    return np.log(100.0 / np.sqrt(price**2 + ASINH_SCALE**2))


@dataclass(frozen=True)
class Series:
    name: str
    frozen_file: Path
    price_column: str
    observation: Callable[[pd.Series], pd.Series]
    observation_name: str
    expected_observations: int
    output_dir: Path
    price_title: str
    price_unit: str
    # Candidate models run by default, and the G5 reference among them.
    candidates: tuple = ("G2", "G3", "G4", "T2", "T3", "T4")
    reference: str = "G2"
    # Third G4 subsample: the full sample without these dates.
    excluded_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    excluded_name: str = ""
    # Suffix of the held-out log-likelihood without the excluded dates.
    excluded_key: str = ""
    early_end: str = "2016-12-31"
    late_start: str = "2017-01-01"
    train_end: str = "2018-12-31"
    test_start: str = "2019-01-01"
    # Record-only G4 subsamples (see the GATES.md change log): name to a
    # function that returns the dates to drop from the full sample. They are
    # reported beside G4 and do not enter the G4 pass or fail.
    record_exclusions: dict = field(default_factory=dict)
    # Log derivative of the observation with respect to the price change,
    # per observation, when the observation is not in price units.
    log_jacobian: Optional[Callable[[pd.Series], pd.Series]] = None

    def load_prices(self, path=None):
        """Load the frozen daily price series."""
        path = Path(path) if path is not None else self.frozen_file
        if not path.exists():
            raise FileNotFoundError(f"Frozen data file not found: {path}")
        data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        price = data[self.price_column].astype(float).sort_index()
        if price.index.has_duplicates:
            raise ValueError("The frozen price series contains duplicate dates.")
        return price

    def load_observations(self, path=None):
        """Load the observation series the HMM is fitted to."""
        values = self.observation(self.load_prices(path)).rename(self.observation_name).dropna()
        if len(values) != self.expected_observations:
            raise ValueError(
                f"Expected {self.expected_observations} observations, found {len(values)}."
            )
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("The observation series contains non-finite values.")
        return values

    def stability_samples(self, values):
        """Return the full sample and the subsamples registered for gate G4."""
        samples = {
            "full_2010_2024": values,
            "early_2010_2016": values.loc[: self.early_end],
            "late_2017_2024": values.loc[self.late_start :],
        }
        if self.excluded_name:
            samples[f"full_without_{self.excluded_name}"] = values.drop(self.excluded_dates)
        return samples

    def record_samples(self, values):
        """Return the record-only subsamples, each the full sample minus some dates."""
        return {
            f"full_without_{name}": values.drop(dates(values.index).intersection(values.index))
            for name, dates in self.record_exclusions.items()
        }

    def train_test_split(self, values):
        """Return the training and test samples registered for gate G5."""
        return values.loc[: self.train_end], values.loc[self.test_start :]


SERIES = {
    "wti": Series(
        name="wti",
        frozen_file=RAW_DIR / "wti_CL=F_2010-01-04_to_2024-12-30_downloaded_2026-08-11.csv",
        price_column="price",
        observation=price_change,
        observation_name="price_change_usd",
        expected_observations=3770,
        output_dir=OUTPUT_ROOT / "model_comparison",
        price_title="WTI price",
        price_unit="USD/bbl",
        excluded_dates=pd.to_datetime(["2020-04-20", "2020-04-21"]),
        excluded_name="cross_zero_days",
        excluded_key="cross_zero",
        record_exclusions={"roll_days": cl_roll_days},
    ),
    # Registered 8 October 2026 (GATES.md change log, DATA_DECISION.md addendum).
    "wti_asinh10": Series(
        name="wti_asinh10",
        frozen_file=RAW_DIR / "wti_CL=F_2010-01-04_to_2024-12-30_downloaded_2026-08-11.csv",
        price_column="price",
        observation=asinh_return,
        observation_name="asinh_return_10_x100",
        expected_observations=3770,
        output_dir=OUTPUT_ROOT / "scale_free",
        price_title="WTI price",
        price_unit="USD/bbl",
        excluded_dates=pd.to_datetime(["2020-04-20", "2020-04-21"]),
        excluded_name="cross_zero_days",
        excluded_key="cross_zero",
        record_exclusions={"roll_days": cl_roll_days},
        candidates=("A2G", "A2"),
        reference="A2G",
        log_jacobian=asinh_log_jacobian,
    ),
}


def get_series(name):
    if name not in SERIES:
        raise SystemExit(f"Unknown series {name!r}. Registered: {sorted(SERIES)}")
    return SERIES[name]
