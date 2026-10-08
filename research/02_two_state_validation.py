"""Fit and validate a two-state Gaussian HMM on daily WTI price changes."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaussian_hmm import GaussianHMM2State


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DATA = PROJECT_DIR / "outputs" / "data_audit" / "wti_audit_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "two_state_validation"
STATE_NAMES = {0: "Low volatility", 1: "High volatility"}
COLOURS = {0: "#2e8b57", 1: "#c00000"}


def load_observations(path=AUDIT_DATA):
    """Load audited prices and calculate daily dollar changes."""
    if not Path(path).exists():
        raise FileNotFoundError(
            "Run 01_data_audit.py before 02_two_state_validation.py."
        )
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    price = data["price"].astype(float).sort_index()
    change = price.diff().rename("price_change_usd").dropna()
    if len(change) != 3770 or not np.isfinite(change.to_numpy()).all():
        raise ValueError("The WTI price-change series failed validation.")
    return price, change


def fit_model(change):
    """Fit the selected model and validate its numerical output."""
    model = GaussianHMM2State(n_iter=300, tol=1e-7).fit(change.to_numpy())
    posterior = model.predict_proba(change.to_numpy())
    states = model.decode(change.to_numpy())

    likelihood_changes = np.diff(model.log_likelihood_history_)
    if len(likelihood_changes) and likelihood_changes.min() < -1e-7:
        raise AssertionError("Training log-likelihood decreased.")
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
    if model.sigma[0] > model.sigma[1]:
        raise AssertionError("State labels are not ordered by volatility.")
    return model, posterior, states


def parameter_table(model, posterior, states):
    """Create one row for each estimated state."""
    records = []
    for state in range(2):
        records.append(
            {
                "state": state,
                "label": STATE_NAMES[state],
                "mean_usd_per_barrel": model.mu[state],
                "sigma_usd_per_barrel": model.sigma[state],
                "initial_probability": model.pi[state],
                "self_transition": model.A[state, state],
                "expected_duration_days": 1 / (1 - model.A[state, state]),
                "viterbi_occupancy": (states == state).mean(),
                "mean_posterior_probability": posterior[:, state].mean(),
            }
        )
    return pd.DataFrame(records)


def daily_table(price, change, posterior, states):
    """Create the date-level model output."""
    return pd.DataFrame(
        {
            "price": price.reindex(change.index),
            "price_change_usd": change,
            "probability_low_volatility": posterior[:, 0],
            "probability_high_volatility": posterior[:, 1],
            "viterbi_state": states,
            "state_label": [STATE_NAMES[state] for state in states],
        },
        index=change.index,
    )


def episode_table(daily):
    """Summarise each uninterrupted Viterbi state episode."""
    group = daily["viterbi_state"].ne(daily["viterbi_state"].shift()).cumsum()
    records = []
    for _, episode in daily.groupby(group):
        state = int(episode["viterbi_state"].iloc[0])
        records.append(
            {
                "state": state,
                "state_label": STATE_NAMES[state],
                "start_date": episode.index.min(),
                "end_date": episode.index.max(),
                "trading_days": len(episode),
                "cumulative_price_change": episode["price_change_usd"].sum(),
                "mean_absolute_price_change": episode[
                    "price_change_usd"
                ].abs().mean(),
                "maximum_high_volatility_probability": episode[
                    "probability_high_volatility"
                ].max(),
            }
        )
    return pd.DataFrame(records)


def save_model_chart(daily):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("WTI two-state Gaussian HMM using daily price changes")

    axes[0].plot(daily.index, daily["price"], color="black", linewidth=0.8)
    for state in range(2):
        mask = daily["viterbi_state"].eq(state)
        axes[0].fill_between(
            daily.index,
            0,
            daily["price"],
            where=mask,
            color=COLOURS[state],
            alpha=0.22,
        )
    axes[0].set_ylabel("USD per barrel")
    axes[0].set_title("WTI price shaded by Viterbi state")
    axes[0].legend(
        handles=[
            mpatches.Patch(color=COLOURS[state], label=STATE_NAMES[state])
            for state in range(2)
        ]
    )

    colours = [COLOURS[state] for state in daily["viterbi_state"]]
    axes[1].bar(
        daily.index,
        daily["price_change_usd"],
        color=colours,
        width=1.4,
        alpha=0.7,
    )
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("USD per barrel")
    axes[1].set_title("Daily price changes")

    axes[2].plot(
        daily.index,
        daily["probability_high_volatility"],
        color=COLOURS[1],
        linewidth=0.8,
    )
    axes[2].axhline(0.5, color="black", linewidth=0.6, linestyle=":")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("Probability")
    axes[2].set_title("Smoothed probability of the high-volatility state")
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_two_state_wti.png", dpi=160)
    plt.close(fig)


def save_likelihood_chart(model):
    history = np.asarray(model.log_likelihood_history_)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(history) + 1), history, marker="o", markersize=2)
    ax.set(
        title="Baum-Welch training log-likelihood",
        xlabel="Iteration",
        ylabel="Log-likelihood",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_likelihood_convergence.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price, change = load_observations()
    model, posterior, states = fit_model(change)

    parameters = parameter_table(model, posterior, states)
    daily = daily_table(price, change, posterior, states)
    episodes = episode_table(daily)
    transitions = pd.DataFrame(
        model.A,
        index=["from_low_volatility", "from_high_volatility"],
        columns=["to_low_volatility", "to_high_volatility"],
    )

    parameters.to_csv(OUTPUT_DIR / "model_parameters.csv", index=False)
    transitions.to_csv(OUTPUT_DIR / "transition_matrix.csv")
    daily.to_csv(OUTPUT_DIR / "daily_state_output.csv", index_label="date")
    episodes.to_csv(OUTPUT_DIR / "state_episodes.csv", index=False)
    pd.DataFrame(
        {
            "iteration": np.arange(1, len(model.log_likelihood_history_) + 1),
            "log_likelihood": model.log_likelihood_history_,
        }
    ).to_csv(OUTPUT_DIR / "likelihood_history.csv", index=False)

    save_model_chart(daily)
    save_likelihood_chart(model)

    uncertain = (posterior.max(axis=1) < 0.8).mean()
    print("Two-state WTI model parameters")
    print(parameters.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTransition matrix")
    print(transitions.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nTraining iterations: {model.n_iter_}")
    print(f"Final fitted log-likelihood: {model.score(change.to_numpy()):.6f}")
    print(f"Dates with maximum posterior below 80%: {uncertain:.2%}")
    print(f"Files saved in {OUTPUT_DIR}")
    print("State labels describe estimated volatility. They do not name a cause.")


if __name__ == "__main__":
    main()
