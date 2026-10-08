"""Adequacy diagnostics for the validated two-state Gaussian WTI model.

Everything here is computed on the fit recorded on 11 August 2026. The script
answers four questions before any richer model is fitted:

1. Does the Markov structure matter? Compare with an independent two-component
   Gaussian mixture on the same data.
2. Are the one-step-ahead predictive distributions adequate? PIT uniformity,
   serial dependence of normal-score residuals and their squares.
3. How uncertain are the parameters? Parametric bootstrap standard errors.
4. Do realised episode durations match the geometric expectation, and do
   high-volatility dates cluster in the contract-roll calendar window?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from diagnostics import (
    autocorrelation,
    bootstrap_summary,
    duration_table,
    parametric_bootstrap,
    pit_diagnostics,
)
from regime_hmm import RegimeHMM
from wti_data import load_changes, load_prices


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "gaussian_diagnostics"
SAVED_PARAMETERS = (
    PROJECT_DIR / "outputs" / "two_state_validation" / "model_parameters.csv"
)
N_BOOTSTRAP = 200
LAGS = 20


def fit_reference(change):
    """Refit the two-state Gaussian model and confirm it matches the record."""
    model = RegimeHMM(n_states=2, n_iter=300, tol=1e-7).fit(change.to_numpy())
    saved = pd.read_csv(SAVED_PARAMETERS)
    np.testing.assert_allclose(
        model.mu, saved["mean_usd_per_barrel"].to_numpy(), atol=1e-8
    )
    np.testing.assert_allclose(
        model.sigma, saved["sigma_usd_per_barrel"].to_numpy(), atol=1e-8
    )
    np.testing.assert_allclose(
        np.diag(model.A), saved["self_transition"].to_numpy(), atol=1e-8
    )
    return model


def mixture_comparison(model, change):
    """Compare the HMM with an independent Gaussian mixture."""
    values = change.to_numpy()
    best_mixture = None
    for start in range(4):
        candidate = RegimeHMM(
            n_states=2,
            markov=False,
            n_iter=500,
            tol=1e-7,
            init_strategy="deterministic" if start == 0 else "random",
            random_state=None if start == 0 else 900 + start,
        ).fit(values)
        if best_mixture is None or candidate.score(values) > best_mixture.score(values):
            best_mixture = candidate
    rows = []
    for name, fitted in (("hmm", model), ("independent_mixture", best_mixture)):
        rows.append(
            {
                "model": name,
                "parameters": fitted.n_parameters,
                "log_likelihood": fitted.score(values),
                "aic": fitted.aic(values),
                "bic": fitted.bic(values),
                "sigma_low": fitted.sigma[0],
                "sigma_high": fitted.sigma[1],
                "weight_or_stationary_high": (
                    fitted.pi[1]
                    if not fitted.markov
                    else stationary_distribution(fitted.A)[1]
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["likelihood_ratio_statistic"] = 2 * (
        table.loc[0, "log_likelihood"] - table.loc[1, "log_likelihood"]
    )
    return table, best_mixture


def stationary_distribution(transition):
    values, vectors = np.linalg.eig(transition.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return vector / vector.sum()


def roll_window_table(daily):
    """Test whether high-volatility dates cluster in the calendar roll window."""
    in_window = daily.index.day.to_series(index=daily.index).between(14, 23)
    high = daily["viterbi_state"].eq(1)
    contingency = pd.crosstab(in_window, high)
    statistic, p_value, _, _ = chi2_contingency(contingency.to_numpy())
    return pd.DataFrame(
        [
            {
                "share_of_all_dates_in_roll_window": in_window.mean(),
                "share_of_high_state_dates_in_roll_window": in_window[high].mean(),
                "share_of_low_state_dates_in_roll_window": in_window[~high].mean(),
                "chi_square": statistic,
                "p_value": p_value,
            }
        ]
    )


def filtered_versus_smoothed(model, change):
    """Measure how far real-time classification lags the hindsight path."""
    values = change.to_numpy()
    filtered = model.filter_proba(values)[:, 1]
    smoothed = model.predict_proba(values)[:, 1]
    viterbi = model.decode(values)
    filtered_state = (filtered > 0.5).astype(int)
    smoothed_state = (smoothed > 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "dates": len(values),
                "filtered_vs_viterbi_disagreement": (filtered_state != viterbi).mean(),
                "smoothed_vs_viterbi_disagreement": (smoothed_state != viterbi).mean(),
                "filtered_high_occupancy": filtered_state.mean(),
                "viterbi_high_occupancy": viterbi.mean(),
                "mean_abs_filtered_minus_smoothed": np.abs(filtered - smoothed).mean(),
            }
        ]
    )


def save_pit_chart(result, change):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Two-state Gaussian HMM: one-step-ahead predictive diagnostics")

    axes[0, 0].hist(result["pit"], bins=25, color="#4c72b0", edgecolor="white")
    axes[0, 0].axhline(len(result["pit"]) / 25, color="black", linestyle="--")
    axes[0, 0].set_title(
        f"PIT histogram (KS p = {result['ks_p_value']:.4f})"
    )
    axes[0, 0].set_xlabel("PIT value")

    residual = np.sort(result["residual"])
    theoretical = norm.ppf((np.arange(1, len(residual) + 1) - 0.5) / len(residual))
    axes[0, 1].scatter(theoretical, residual, s=6, color="#4c72b0")
    limits = [theoretical.min(), theoretical.max()]
    axes[0, 1].plot(limits, limits, color="black", linestyle="--")
    axes[0, 1].set_title("Normal-score residual QQ plot")
    axes[0, 1].set_xlabel("Theoretical quantile")
    axes[0, 1].set_ylabel("Residual quantile")

    lags = np.arange(1, LAGS + 1)
    band = 1.96 / np.sqrt(len(result["residual"]))
    axes[1, 0].bar(lags, autocorrelation(result["residual"], LAGS), color="#4c72b0")
    axes[1, 0].axhline(band, color="black", linestyle="--")
    axes[1, 0].axhline(-band, color="black", linestyle="--")
    axes[1, 0].set_title(
        f"Residual autocorrelation (Ljung-Box p = {result['ljung_box_p_value']:.4f})"
    )
    axes[1, 0].set_xlabel("Lag")

    axes[1, 1].bar(
        lags, autocorrelation(result["residual"] ** 2, LAGS), color="#c44e52"
    )
    axes[1, 1].axhline(band, color="black", linestyle="--")
    axes[1, 1].axhline(-band, color="black", linestyle="--")
    axes[1, 1].set_title(
        "Squared-residual autocorrelation "
        f"(Ljung-Box p = {result['ljung_box_squared_p_value']:.4f})"
    )
    axes[1, 1].set_xlabel("Lag")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_pit_diagnostics.png", dpi=160)
    plt.close(fig)


def save_bootstrap_chart(draws, model):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Parametric bootstrap, {len(draws)} draws")
    panels = [
        ("sigma_1", model.sigma[1], "High-state sigma (USD)"),
        ("self_transition_0", model.A[0, 0], "Low-state self-transition"),
        ("self_transition_1", model.A[1, 1], "High-state self-transition"),
    ]
    for axis, (column, estimate, title) in zip(axes, panels):
        axis.hist(draws[column], bins=30, color="#4c72b0", edgecolor="white")
        axis.axvline(estimate, color="black", linestyle="--", label="Estimate")
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_bootstrap_distributions.png", dpi=160)
    plt.close(fig)


def save_filter_chart(model, change, price):
    values = change.to_numpy()
    filtered = model.filter_proba(values)[:, 1]
    smoothed = model.predict_proba(values)[:, 1]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(change.index, price.reindex(change.index), color="black", linewidth=0.8)
    axes[0].set_ylabel("USD per barrel")
    axes[0].set_title("WTI price")
    axes[1].plot(change.index, smoothed, color="#c44e52", linewidth=0.8, label="Smoothed")
    axes[1].plot(change.index, filtered, color="#4c72b0", linewidth=0.8, alpha=0.8, label="Filtered")
    axes[1].set_ylabel("P(high volatility)")
    axes[1].set_title("Filtered (real-time) versus smoothed (hindsight) probabilities")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_filtered_vs_smoothed.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price = load_prices()
    change = load_changes()
    values = change.to_numpy()
    model = fit_reference(change)
    print("Reference two-state Gaussian fit reproduced.")

    result = pit_diagnostics(model, values, lags=LAGS)
    pit_table = pd.DataFrame(
        [{key: value for key, value in result.items() if key not in {"pit", "residual"}}]
    )
    pit_table.to_csv(OUTPUT_DIR / "pit_diagnostics.csv", index=False)
    pd.DataFrame(
        {"pit": result["pit"], "residual": result["residual"]}, index=change.index
    ).to_csv(OUTPUT_DIR / "pit_values.csv", index_label="date")
    save_pit_chart(result, change)
    print("\nPIT diagnostics")
    print(pit_table.T.to_string(header=False))

    comparison, mixture = mixture_comparison(model, change)
    comparison.to_csv(OUTPUT_DIR / "mixture_comparison.csv", index=False)
    mixture_result = pit_diagnostics(mixture, values, lags=LAGS)
    pd.DataFrame(
        [{key: value for key, value in mixture_result.items() if key not in {"pit", "residual"}}]
    ).to_csv(OUTPUT_DIR / "mixture_pit_diagnostics.csv", index=False)
    print("\nHMM versus independent mixture")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    states = model.decode(values)
    durations = duration_table(states, model.A)
    durations.to_csv(OUTPUT_DIR / "duration_comparison.csv", index=False)
    print("\nRealised versus expected durations")
    print(durations.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    daily = pd.DataFrame({"viterbi_state": states}, index=change.index)
    roll = roll_window_table(daily)
    roll.to_csv(OUTPUT_DIR / "roll_window_check.csv", index=False)
    print("\nRoll-window check")
    print(roll.T.to_string(header=False))

    lag_table = filtered_versus_smoothed(model, change)
    lag_table.to_csv(OUTPUT_DIR / "filtered_vs_smoothed.csv", index=False)
    save_filter_chart(model, change, price)
    print("\nFiltered versus smoothed")
    print(lag_table.T.to_string(header=False))

    print(f"\nRunning {N_BOOTSTRAP} parametric bootstrap draws.")
    draws = parametric_bootstrap(
        model,
        len(values),
        n_draws=N_BOOTSTRAP,
        seed=20260901,
        fit_kwargs={"n_iter": 300, "tol": 1e-7},
    )
    draws.to_csv(OUTPUT_DIR / "bootstrap_draws.csv", index=False)
    summary = bootstrap_summary(model, draws)
    summary.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False)
    save_bootstrap_chart(draws, model)
    print("\nBootstrap summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFiles saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
