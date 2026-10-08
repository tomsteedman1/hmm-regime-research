"""Test local-optimum sensitivity with repeated HMM initialisations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_start import fit_multiple_starts


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DATA = PROJECT_DIR / "outputs" / "data_audit" / "wti_audit_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "multi_start_validation"
N_STARTS = 12
BASE_SEED = 20260811


def load_price_changes(path=AUDIT_DATA):
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    changes = data["price"].astype(float).sort_index().diff().dropna()
    if len(changes) != 3770 or not np.isfinite(changes.to_numpy()).all():
        raise ValueError("The WTI price-change series failed validation.")
    return changes.rename("price_change_usd")


def build_consensus_table(changes, best_model, paths):
    paths_array = np.asarray(paths)
    best_path = best_model.decode(changes.to_numpy())
    high_state_share = (paths_array == 1).mean(axis=0)
    agreement_share = (paths_array == best_path).mean(axis=0)
    posterior = best_model.predict_proba(changes.to_numpy())
    return pd.DataFrame(
        {
            "price_change_usd": changes,
            "best_viterbi_state": best_path,
            "best_high_state_probability": posterior[:, 1],
            "starts_classifying_high": high_state_share,
            "starts_agreeing_with_best": agreement_share,
        },
        index=changes.index,
    )


def save_start_chart(results):
    selected = results["selected"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    likelihood_gap_scaled = results["log_likelihood_gap"] * 1e8
    axes[0, 0].scatter(results["start"], likelihood_gap_scaled, s=35)
    axes[0, 0].scatter(
        results.loc[selected, "start"],
        likelihood_gap_scaled.loc[selected],
        color="#c00000",
        marker="*",
        s=150,
        label="Selected",
    )
    axes[0, 0].set(
        title="Log-likelihood gap from selected fit",
        xlabel="Start",
        ylabel="Gap multiplied by 100 million",
    )
    axes[0, 0].legend()

    selected_sigma = results.loc[selected, "high_sigma"].iloc[0]
    sigma_difference_scaled = (results["high_sigma"] - selected_sigma) * 1e5
    axes[0, 1].scatter(results["start"], sigma_difference_scaled, s=35)
    axes[0, 1].set(
        title="High-state volatility difference",
        xlabel="Start",
        ylabel="Difference multiplied by 100,000",
    )

    axes[1, 0].scatter(results["start"], results["iterations"], s=35)
    axes[1, 0].set(
        title="Iterations to convergence",
        xlabel="Start",
        ylabel="Iterations",
    )

    axes[1, 1].scatter(
        results["start"],
        results["agreement_with_best"] * 100,
        s=35,
    )
    axes[1, 1].set(
        title="State-path agreement with selected fit",
        xlabel="Start",
        ylabel="Percent of dates",
        ylim=(0, 101),
    )

    fig.suptitle("WTI HMM multiple-initialisation results")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_start_comparison.png", dpi=160)
    plt.close(fig)


def save_consensus_chart(consensus):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(
        consensus.index,
        consensus["best_high_state_probability"],
        color="#c00000",
        linewidth=0.8,
    )
    axes[0].set(
        title="Selected fit: smoothed high-volatility probability",
        ylabel="Probability",
        ylim=(-0.02, 1.02),
    )

    axes[1].plot(
        consensus.index,
        consensus["starts_classifying_high"],
        color="#1f4e79",
        linewidth=0.8,
    )
    axes[1].set(
        title="Share of starts assigning the high-volatility Viterbi state",
        ylabel="Share of starts",
        xlabel="Date",
        ylim=(-0.02, 1.02),
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_state_consensus.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    changes = load_price_changes()
    best_model, records, paths = fit_multiple_starts(
        changes.to_numpy(),
        n_starts=N_STARTS,
        base_seed=BASE_SEED,
        n_iter=300,
        tol=1e-7,
    )
    results = pd.DataFrame(records)
    consensus = build_consensus_table(changes, best_model, paths)
    selected = results.loc[results["selected"]].iloc[0]

    results.to_csv(OUTPUT_DIR / "start_results.csv", index=False)
    consensus.to_csv(OUTPUT_DIR / "daily_state_consensus.csv", index_label="date")
    save_start_chart(results)
    save_consensus_chart(consensus)

    close_to_best = results["log_likelihood_gap"].le(0.01)
    strong_agreement = results["agreement_with_best"].ge(0.95)
    unanimous_dates = consensus["starts_agreeing_with_best"].eq(1).mean()

    print("Multiple-initialisation result")
    print(f"Starts fitted: {N_STARTS}")
    print(f"Selected start: {int(selected['start'])} ({selected['strategy']})")
    print(f"Selected log-likelihood: {selected['log_likelihood']:.6f}")
    print(f"Starts within 0.01 log-likelihood units: {int(close_to_best.sum())}")
    print(f"Starts with at least 95% path agreement: {int(strong_agreement.sum())}")
    print(f"Dates with unanimous state agreement: {unanimous_dates:.2%}")
    print("\nStart comparison")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nFiles saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
