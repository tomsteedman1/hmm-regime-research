"""Walk-forward volatility-target backtest of the T2 regime signal.

Runs only after ``TRADING_GATES.md`` is registered. Scores 2015 to 2024 on
the research frozen sample; never touches the sealed holdout.
"""

import json
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import research_link  # noqa: F401
from multi_start import fit_multiple_starts_general  # noqa: E402
from regime_hmm import RegimeHMM  # noqa: E402
from wti_data import load_changes  # noqa: E402

import backtest  # noqa: E402
from walk_forward import walk_forward_expected_vol  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "walk_forward"
GATES_FILE = PROJECT_DIR / "TRADING_GATES.md"

# Protocol constants. They mirror TRADING_GATES.md and must not drift from it.
FIRST_YEAR = 2015
LAST_YEAR = 2024
N_STARTS = 12
BASE_SEED = 20260811
TARGET_VOL = 1.00
MAX_LEVERAGE = 3.0
COSTS = {"base": 0.01, "double": 0.02}
TREATMENTS = ("raw", "zeroed", "dropped")
REALISED_SPAN = 21
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_BLOCK = 21
TG1_P_VALUE = 0.05
TG3_BAND = 0.20


def gates_are_registered():
    text = GATES_FILE.read_text()
    if "Status: DRAFT" in text:
        return False
    return re.search(r"^Registered: \d{1,2} \w+ \d{4}", text, re.MULTILINE) is not None


def make_t2(init_strategy, random_state):
    return RegimeHMM(
        n_states=2,
        emission="student_t",
        init_strategy=init_strategy,
        random_state=random_state,
    )


def fitter(history):
    model, records, _ = fit_multiple_starts_general(history, make_t2, N_STARTS, BASE_SEED)
    return model


def roll_mask(changes):
    sys.path.insert(0, str(PROJECT_DIR))
    from importlib import import_module

    audit = import_module("01_roll_audit")
    label, _ = audit.flag_roll_days(changes)
    return (label == "day_after_expiry").to_numpy()


def main():
    if not gates_are_registered():
        raise SystemExit(
            "TRADING_GATES.md is still a draft. Register the gates before running."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    changes = load_changes()
    values = changes.to_numpy()
    index = changes.index
    rolls = roll_mask(changes)
    scored = (index.year >= FIRST_YEAR) & (index.year <= LAST_YEAR)

    print("Fitting the walk-forward regime signal.")
    hmm_vol = walk_forward_expected_vol(changes, fitter, FIRST_YEAR).to_numpy()
    realised_vol = backtest.realised_expected_vol(values, span=REALISED_SPAN)

    positions = {
        "hmm": backtest.vol_target_positions(hmm_vol, TARGET_VOL, MAX_LEVERAGE),
        "b2_realised": backtest.vol_target_positions(realised_vol, TARGET_VOL, MAX_LEVERAGE),
        "b1_constant": np.ones(len(values)),
    }
    for name in positions:
        positions[name] = np.where(scored, positions[name], np.nan)

    signals = pd.DataFrame(
        {
            "change": values,
            "roll_day": rolls,
            "hmm_expected_vol": hmm_vol,
            "realised_expected_vol": realised_vol,
            **{f"position_{k}": v for k, v in positions.items()},
        },
        index=index,
    )
    signals.to_csv(OUTPUT_DIR / "daily_signals.csv")

    rows = []
    tests = []
    pnl_store = {}
    for treatment in TREATMENTS:
        series = backtest.apply_roll_treatment(values, rolls, treatment)
        for cost_name, cost in COSTS.items():
            frames = {
                name: backtest.daily_pnl(series, pos, cost) for name, pos in positions.items()
            }
            for name, frame in frames.items():
                frame = frame[scored]
                pnl_store[(name, treatment, cost_name)] = frame
                rows.append(
                    {"strategy": name, "treatment": treatment, "cost": cost_name, **backtest.metrics(frame)}
                )
            boot = backtest.sharpe_difference_bootstrap(
                pnl_store[("hmm", treatment, cost_name)]["net"],
                pnl_store[("b2_realised", treatment, cost_name)]["net"],
                n_draws=BOOTSTRAP_DRAWS,
                mean_block=BOOTSTRAP_BLOCK,
            )
            tests.append({"treatment": treatment, "cost": cost_name, **boot})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUTPUT_DIR / "sharpe_difference_tests.csv", index=False)

    def passes(cost_name):
        sub = tests[tests["cost"] == cost_name]
        return bool((sub["observed"] > 0).all() and (sub["p_value"] < TG1_P_VALUE).all())

    hmm_raw_base = metrics.query("strategy == 'hmm' and treatment == 'raw' and cost == 'base'").iloc[0]
    gates = {
        "TG1": passes("base"),
        "TG2": passes("double"),
        "TG3": bool(abs(hmm_raw_base["daily_vol"] - TARGET_VOL) / TARGET_VOL <= TG3_BAND),
        "TG4": "sealed",
    }

    for treatment in TREATMENTS:
        pnl_store[("hmm", treatment, "base")].assign(date=index[scored]).to_csv(
            OUTPUT_DIR / f"pnl_hmm_{treatment}_base.csv", index=False
        )
        pnl_store[("b2_realised", treatment, "base")].assign(date=index[scored]).to_csv(
            OUTPUT_DIR / f"pnl_b2_{treatment}_base.csv", index=False
        )

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    dates = index[scored]
    for name, colour in [("hmm", "#2F5D7C"), ("b2_realised", "#B4232C"), ("b1_constant", "#7C858E")]:
        axes[0].plot(dates, pnl_store[(name, "raw", "base")]["net"].cumsum(), label=name, color=colour)
    axes[0].set_title("Cumulative net P&L, raw treatment, $0.01 costs")
    axes[0].legend()
    axes[1].plot(dates, hmm_vol[scored], label="HMM expected vol", color="#2F5D7C")
    axes[1].plot(dates, realised_vol[scored], label="EWMA realised vol", color="#B4232C", alpha=0.7)
    axes[1].set_title("Expected daily volatility, USD per barrel")
    axes[1].legend()
    axes[2].plot(dates, positions["hmm"][scored], label="HMM position", color="#2F5D7C")
    axes[2].plot(dates, positions["b2_realised"][scored], label="B2 position", color="#B4232C", alpha=0.7)
    axes[2].set_title("Position, units")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_walk_forward.png", dpi=160)

    summary = {
        "first_year": FIRST_YEAR,
        "last_year": LAST_YEAR,
        "scored_days": int(scored.sum()),
        "gates": gates,
        "seconds": time.time() - started,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nMetrics")
    print(metrics.round(4).to_string(index=False))
    print("\nSharpe difference, HMM minus B2")
    print(tests.round(4).to_string(index=False))
    print("\nGates:", gates)
    print(f"\nOutputs written to {OUTPUT_DIR} in {summary['seconds']:.0f}s")


if __name__ == "__main__":
    main()
