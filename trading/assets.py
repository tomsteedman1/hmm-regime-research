"""Cross-asset universe, frozen-file loaders and return construction.

Every loader reads a frozen development file from ``data/raw``. No function
in this module opens a holdout file; the holdout paths are listed only so
that ``test_assets.py`` can assert nothing else touches them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import research_link  # noqa: F401
from wti_data import load_changes as load_wti_changes  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

START = "2010-01-01"
DEV_END = "2024-12-30"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-08-31"

# Chosen before any result was seen; see RESEARCH_PROGRAMME.md.
UNIVERSE = {
    "wti": {"ticker": "CL=F", "group": "energy", "name": "WTI crude", "kind": "future"},
    "brent": {"ticker": "BZ=F", "group": "energy", "name": "Brent crude", "kind": "future"},
    "natgas": {"ticker": "NG=F", "group": "energy", "name": "Henry Hub gas", "kind": "future"},
    "gold": {"ticker": "GC=F", "group": "metals", "name": "Gold", "kind": "future"},
    "silver": {"ticker": "SI=F", "group": "metals", "name": "Silver", "kind": "future"},
    "copper": {"ticker": "HG=F", "group": "metals", "name": "Copper", "kind": "future"},
    "corn": {"ticker": "ZC=F", "group": "grains", "name": "Corn", "kind": "future"},
    "soybeans": {"ticker": "ZS=F", "group": "grains", "name": "Soybeans", "kind": "future"},
    "wheat": {"ticker": "ZW=F", "group": "grains", "name": "Wheat", "kind": "future"},
    "sp500": {"ticker": "ES=F", "group": "equity", "name": "S&P 500 e-mini", "kind": "future"},
    "tnote": {"ticker": "ZN=F", "group": "rates", "name": "10-year T-note", "kind": "future"},
    "euro": {"ticker": "6E=F", "group": "fx", "name": "Euro", "kind": "future"},
    "yen": {"ticker": "6J=F", "group": "fx", "name": "Japanese yen", "kind": "future"},
    "spy": {"ticker": "SPY", "group": "etf", "name": "SPY", "kind": "etf"},
    "tlt": {"ticker": "TLT", "group": "etf", "name": "TLT", "kind": "etf"},
    "gld": {"ticker": "GLD", "group": "etf", "name": "GLD", "kind": "etf"},
    "uso": {"ticker": "USO", "group": "etf", "name": "USO", "kind": "etf"},
    "ung": {"ticker": "UNG", "group": "etf", "name": "UNG", "kind": "etf"},
    "vix": {"ticker": "^VIX", "group": "covariate", "name": "VIX", "kind": "covariate"},
    "wti_ohlc": {"ticker": "CL=F", "group": "covariate", "name": "WTI OHLC for the range proxy only", "kind": "covariate"},
}

FUTURES = [k for k, v in UNIVERSE.items() if v["kind"] == "future"]
ETFS = [k for k, v in UNIVERSE.items() if v["kind"] == "etf"]
TRADED = FUTURES + ETFS


def frozen_file(key: str, holdout: bool = False) -> Path:
    """Return the frozen development file for ``key`` (or its holdout path)."""
    tag = "_holdout_" if holdout else "_"
    pattern = re.compile(rf"^{re.escape(key)}_[^_]+{tag}\d{{4}}-\d{{2}}-\d{{2}}_to_.*\.csv$")
    matches = sorted(p for p in DATA_DIR.glob(f"{key}_*.csv") if pattern.match(p.name))
    if not holdout:
        matches = [p for p in matches if "_holdout_" not in p.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one frozen file for {key} holdout={holdout}, found {matches}")
    return matches[0]


def load_ohlc(key: str) -> pd.DataFrame:
    """Frozen development OHLC frame, indexed by date, close strictly positive checked."""
    path = frozen_file(key)
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{key}: duplicate dates")
    if frame.index[-1] > pd.Timestamp(DEV_END):
        raise ValueError(f"{key}: development file extends past {DEV_END}")
    return frame


def load_returns(key: str) -> pd.Series:
    """The HMM input and P&L series for one asset.

    Percent log returns for every asset except WTI, which keeps the research
    decision of dollar changes on the research frozen close. ETFs use the
    dividend-adjusted close so that the return is what a holder earned.
    """
    if key == "wti":
        return load_wti_changes().rename("wti")
    frame = load_ohlc(key)
    column = "adj_close" if UNIVERSE[key]["kind"] == "etf" else "close"
    price = frame[column].astype(float)
    if (price <= 0).any():
        raise ValueError(f"{key}: non-positive prices, log returns undefined")
    returns = 100.0 * np.log(price).diff().dropna()
    returns = returns[returns.index >= pd.Timestamp(START)]
    return returns.rename(key)


def load_range_variance(key: str) -> pd.Series:
    """Daily Garman-Klass variance in the same units as ``load_returns`` squared.

    For WTI the OHLC comes from the separate ``wti_ohlc`` file and the
    variance is in dollars squared, matching the dollar-change input.
    """
    frame = load_ohlc("wti_ohlc" if key == "wti" else key)
    o, h, l, c = (frame[col].astype(float) for col in ("open", "high", "low", "close"))
    if key == "wti":
        # Dollar units: Garman-Klass on price levels rather than logs.
        hl = h - l
        co = c - o
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    else:
        hl = np.log(h / l)
        co = np.log(c / o)
        gk = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2) * 100.0 ** 2
    gk = gk.where(gk > 0)
    return gk.rename(f"{key}_gk")


def load_vix() -> pd.Series:
    return load_ohlc("vix")["close"].astype(float).rename("vix")
