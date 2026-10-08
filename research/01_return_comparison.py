"""Compare WTI transformations and select the next HMM observation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaussian_hmm import GaussianHMM2State


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DATA = PROJECT_DIR / "outputs" / "data_audit" / "wti_audit_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "return_comparison"
ASINH_SCALES = (10.0, 20.0, 40.0)


def load_price(path=AUDIT_DATA):
    """Load the audited WTI close series."""
    if not Path(path).exists():
        raise FileNotFoundError(
            "Run 01_data_audit.py before 01_return_comparison.py."
        )
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    price = data["price"].astype(float).sort_index()
    if price.index.has_duplicates:
        raise ValueError("The audited price series contains duplicate dates.")
    return price


def build_transformations(price):
    """Create each candidate HMM observation series."""
    previous_price = price.shift(1)
    valid_log = (price > 0) & (previous_price > 0)

    transformations = {
        "simple_return": price.pct_change(fill_method=None),
        "log_return_valid": pd.Series(np.nan, index=price.index),
        "price_change_usd": price.diff(),
    }
    transformations["log_return_valid"].loc[valid_log] = np.log(
        price.loc[valid_log] / previous_price.loc[valid_log]
    )
    for scale in ASINH_SCALES:
        name = f"asinh_return_{int(scale)}"
        transformations[name] = np.arcsinh(price / scale).diff()
    return pd.DataFrame(transformations)


def describe_transformations(transformations):
    """Create distribution statistics for each candidate."""
    records = []
    for name in transformations:
        values = transformations[name].dropna()
        records.append(
            {
                "transformation": name,
                "observations": len(values),
                "missing": int(transformations[name].isna().sum()),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "skewness": values.skew(),
                "excess_kurtosis": values.kurt(),
                "minimum": values.min(),
                "one_percent": values.quantile(0.01),
                "ninety_nine_percent": values.quantile(0.99),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(records)


def fit_hmms(transformations):
    """Fit the same diagnostic Gaussian HMM to each candidate."""
    model_records = []
    state_series = {}

    for name in transformations:
        values = transformations[name].dropna()
        model = GaussianHMM2State(n_iter=200, tol=1e-5).fit(values.to_numpy())
        states = model.decode(values.to_numpy())
        state_series[name] = pd.Series(states, index=values.index, name=name)
        model_records.append(
            {
                "transformation": name,
                "observations": len(values),
                "calm_mean": model.mu[0],
                "stress_mean": model.mu[1],
                "calm_sigma": model.sigma[0],
                "stress_sigma": model.sigma[1],
                "calm_self_transition": model.A[0, 0],
                "stress_self_transition": model.A[1, 1],
                "stress_occupancy": (states == 1).mean(),
                "iterations": model.n_iter_,
                "log_likelihood": model.log_likelihood_,
            }
        )
    return pd.DataFrame(model_records), state_series


def april_table(price, transformations, state_series):
    """Combine April 2020 values and diagnostic states."""
    table = pd.DataFrame({"price": price}).loc["2020-04-13":"2020-04-27"]
    for name in transformations:
        table[name] = transformations[name]
        table[f"{name}_state"] = state_series[name].reindex(table.index)
    return table


def save_distribution_chart(transformations):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for axis, name in zip(axes.flat, transformations.columns):
        values = transformations[name].dropna()
        lower, upper = values.quantile([0.005, 0.995])
        central = values[values.between(lower, upper)]
        axis.hist(central, bins=80, color="#4472c4", alpha=0.85)
        axis.set_title(f"{name}\ncentral 99% shown")
        axis.set_xlabel("Daily observation")
    fig.suptitle("WTI candidate transformation distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_transformation_distributions.png", dpi=160)
    plt.close(fig)


def save_regime_chart(state_series):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, (name, states) in zip(axes.flat, state_series.items()):
        axis.fill_between(
            states.index,
            0,
            states.to_numpy(),
            step="mid",
            color="#c00000",
            alpha=0.7,
        )
        axis.set_ylim(0, 1.05)
        axis.set_yticks([0, 1], labels=["Calm", "Stress"])
        axis.set_title(name)
    fig.suptitle("Diagnostic two-state Gaussian HMM paths", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_regime_comparison.png", dpi=160)
    plt.close(fig)


def validate_primary_input(transformations):
    primary = transformations["price_change_usd"]
    complete = primary.dropna()
    if len(complete) != 3770:
        raise AssertionError("The primary input must contain 3,770 observations.")
    if not np.isfinite(complete.to_numpy()).all():
        raise AssertionError("The primary input contains a non-finite value.")
    np.testing.assert_allclose(primary.loc["2020-04-20"], -55.900002, atol=1e-5)
    np.testing.assert_allclose(primary.loc["2020-04-21"], 47.640001, atol=1e-5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price = load_price()
    transformations = build_transformations(price)
    validate_primary_input(transformations)

    statistics = describe_transformations(transformations)
    models, state_series = fit_hmms(transformations)
    april = april_table(price, transformations, state_series)

    transformations.to_csv(
        OUTPUT_DIR / "transformation_data.csv",
        index_label="date",
    )
    statistics.to_csv(OUTPUT_DIR / "transformation_summary.csv", index=False)
    models.to_csv(OUTPUT_DIR / "hmm_comparison.csv", index=False)
    april.to_csv(OUTPUT_DIR / "april_2020_comparison.csv", index_label="date")
    save_distribution_chart(transformations)
    save_regime_chart(state_series)

    print("Transformation statistics")
    print(statistics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nDiagnostic HMM comparison")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nFiles saved in {OUTPUT_DIR}")
    print("Selected HMM observation: daily WTI price change in USD per barrel.")


if __name__ == "__main__":
    main()
