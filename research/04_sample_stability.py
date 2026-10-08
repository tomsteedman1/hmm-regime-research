"""Measure two-state HMM sensitivity to the fitted sample."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_start import fit_multiple_starts


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DATA = PROJECT_DIR / "outputs" / "data_audit" / "wti_audit_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sample_stability"
N_STARTS = 8
BASE_SEED = 20260811


def load_samples(path=AUDIT_DATA):
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    changes = data["price"].astype(float).sort_index().diff().dropna()
    crossing_dates = pd.to_datetime(["2020-04-20", "2020-04-21"])
    return {
        "full_2010_2024": changes,
        "early_2010_2016": changes.loc[:"2016-12-31"],
        "late_2017_2024": changes.loc["2017-01-01":],
        "full_without_cross_zero_days": changes.drop(crossing_dates),
    }


def fit_samples(samples):
    fitted = {}
    for sample_number, (name, values) in enumerate(samples.items()):
        model, records, paths = fit_multiple_starts(
            values.to_numpy(),
            n_starts=N_STARTS,
            base_seed=BASE_SEED + sample_number * 100,
            n_iter=300,
            tol=1e-7,
        )
        fitted[name] = {
            "model": model,
            "records": records,
            "path": pd.Series(model.decode(values.to_numpy()), index=values.index),
        }
    return fitted


def result_table(samples, fitted):
    full_path = fitted["full_2010_2024"]["path"]
    records = []
    for name, values in samples.items():
        result = fitted[name]
        model = result["model"]
        path = result["path"]
        common_dates = full_path.index.intersection(path.index)
        agreement = (
            full_path.loc[common_dates].to_numpy()
            == path.loc[common_dates].to_numpy()
        ).mean()
        start_agreement = min(
            record["agreement_with_best"] for record in result["records"]
        )
        records.append(
            {
                "sample": name,
                "start_date": values.index.min(),
                "end_date": values.index.max(),
                "observations": len(values),
                "low_mean": model.mu[0],
                "high_mean": model.mu[1],
                "low_sigma": model.sigma[0],
                "high_sigma": model.sigma[1],
                "low_self_transition": model.A[0, 0],
                "high_self_transition": model.A[1, 1],
                "low_expected_duration": 1 / (1 - model.A[0, 0]),
                "high_expected_duration": 1 / (1 - model.A[1, 1]),
                "high_state_occupancy": (path == 1).mean(),
                "agreement_with_full_on_overlap": agreement,
                "minimum_start_agreement": start_agreement,
            }
        )
    return pd.DataFrame(records)


def path_table(fitted):
    return pd.concat(
        {
            name: result["path"] for name, result in fitted.items()
        },
        axis=1,
    )


def save_parameter_chart(results):
    labels = results["sample"].str.replace("_", "\n")
    positions = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].bar(positions, results["high_sigma"], color="#c00000")
    axes[0, 0].set(title="High-state volatility", ylabel="USD per barrel")

    axes[0, 1].bar(
        positions,
        results["high_state_occupancy"] * 100,
        color="#c00000",
    )
    axes[0, 1].set(title="High-state occupancy", ylabel="Percent of dates")

    axes[1, 0].bar(
        positions,
        results["high_expected_duration"],
        color="#c00000",
    )
    axes[1, 0].set(title="Expected high-state duration", ylabel="Trading days")

    axes[1, 1].bar(
        positions,
        results["agreement_with_full_on_overlap"] * 100,
        color="#1f4e79",
    )
    axes[1, 1].set(
        title="State-path agreement with full fit",
        ylabel="Percent of shared dates",
        ylim=(0, 101),
    )

    for axis in axes.flat:
        axis.set_xticks(positions, labels=labels, rotation=0)
    fig.suptitle("WTI HMM sample-period sensitivity")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_parameter_stability.png", dpi=160)
    plt.close(fig)


def save_path_chart(paths):
    fig, axes = plt.subplots(len(paths.columns), 1, figsize=(14, 9), sharex=True)
    for axis, name in zip(axes, paths.columns):
        series = paths[name].dropna()
        axis.fill_between(
            series.index,
            0,
            series.to_numpy(),
            step="mid",
            color="#c00000",
            alpha=0.7,
        )
        axis.set_ylim(0, 1.05)
        axis.set_yticks([0, 1], labels=["Low", "High"])
        axis.set_title(name.replace("_", " "))
    axes[-1].set_xlabel("Date")
    fig.suptitle("Viterbi paths from different fitted samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_sample_state_paths.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    fitted = fit_samples(samples)
    results = result_table(samples, fitted)
    paths = path_table(fitted)

    results.to_csv(OUTPUT_DIR / "sample_results.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "sample_state_paths.csv", index_label="date")
    save_parameter_chart(results)
    save_path_chart(paths)

    print("Sample-period stability results")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nEach sample used {N_STARTS} initialisations.")
    print(f"Files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
