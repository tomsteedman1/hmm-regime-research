"""Shared helpers for the E2 to E6 experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import backtest  # noqa: E402
from assets import ETFS, FUTURES, TRADED, UNIVERSE  # noqa: E402
from stats_tools import holm  # noqa: E402
from strategies import COSTS, equal_risk_portfolio  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
SIGNAL_DIR = PROJECT_DIR / "outputs" / "signals"
FIRST_YEAR = 2015
LAST_YEAR = 2024
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_BLOCK = 21
COLOURS = {"hmm": "#2F5D7C", "ewma": "#B4232C", "garch": "#3B8A5A", "har": "#B8860B", "constant": "#7C858E", "other": "#6A4C93"}


def load_signals(key: str) -> pd.DataFrame:
    frame = pd.read_csv(SIGNAL_DIR / f"{key}_signals.csv", parse_dates=["date"]).set_index("date")
    return frame


def scored_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.year >= FIRST_YEAR) & (index.year <= LAST_YEAR)


def pnl_table(series, positions: dict[str, np.ndarray], cost: float, scored: np.ndarray, index=None) -> dict[str, pd.DataFrame]:
    """Net P&L frames for several position rules on one asset, scored rows only."""
    out = {}
    for name, pos in positions.items():
        pos = np.where(scored, pos, np.nan)
        frame = backtest.daily_pnl(series, pos, cost)
        if index is not None:
            frame.index = index
        out[name] = frame[scored]
    return out


def sharpe_test(net_a, net_b):
    return backtest.sharpe_difference_bootstrap(net_a, net_b, n_draws=BOOTSTRAP_DRAWS, mean_block=BOOTSTRAP_BLOCK)


def summarise_tests(rows: list[dict], group_col: str = "comparison") -> pd.DataFrame:
    """Add Holm-adjusted p-values within each comparison and the null count."""
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, idx in table.groupby(group_col).groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_value"].to_numpy())
    return table


def count_line(table: pd.DataFrame, comparison: str, level: float = 0.05) -> dict:
    sub = table[table["comparison"] == comparison]
    return {
        "comparison": comparison,
        "assets": int(len(sub)),
        "positive": int((sub["observed"] > 0).sum()),
        "below_5pct": int((sub["p_value"] < level).sum()),
        "expected_under_null": float(len(sub) * level),
        "holm_below_5pct": int((sub["p_holm"] < level).sum()),
    }


def portfolio_net(store: dict[tuple, pd.DataFrame], strategy: str, keys, treatment_key=None) -> pd.Series:
    nets = {}
    for key in keys:
        frame = store[(key, strategy)] if treatment_key is None else store[(key, strategy, treatment_key)]
        nets[key] = frame["net"]
    return equal_risk_portfolio(nets)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float))


def group_of(key: str) -> str:
    return UNIVERSE[key]["group"]


def label(key: str) -> str:
    return UNIVERSE[key]["name"]


__all__ = [
    "ETFS", "FUTURES", "TRADED", "COSTS", "COLOURS", "PROJECT_DIR", "SIGNAL_DIR",
    "load_signals", "scored_mask", "pnl_table", "sharpe_test", "summarise_tests",
    "count_line", "portfolio_net", "write_json", "group_of", "label", "backtest",
]
