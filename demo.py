"""Two-minute demo that needs no market data.

Simulates 4,000 days from a two-state Student-t HMM with parameters close to
the ones the research found on WTI, fits a fresh model to the simulated
changes with the from-scratch Baum-Welch in ``research/regime_hmm.py``, and
checks how well it recovers the true parameters and the hidden states.

    python demo.py            # prints the table and writes demo_regimes.png
    python demo.py --no-plot  # table only
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "research"))

from regime_hmm import RegimeHMM  # noqa: E402

LENGTH = 4000
SEED = 7
TRUE = {
    "pi": [0.5, 0.5],
    "A": [[0.992, 0.008], [0.018, 0.982]],
    "mu": [0.05, -0.03],
    "sigma": [1.0, 1.75],
    "nu": [14.0, 4.5],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-plot", action="store_true", help="skip the chart")
    args = parser.parse_args()

    truth = RegimeHMM(n_states=2, emission="student_t").set_parameters(**TRUE)
    changes, true_states = truth.simulate(LENGTH, random_state=SEED)

    model = RegimeHMM(n_states=2, emission="student_t", n_iter=500, tol=1e-7)
    model.fit(changes)
    decoded = model.decode(changes)
    agreement = (decoded == true_states).mean()

    rows = [
        ("Low-state scale", truth.sigma[0], model.sigma[0]),
        ("High-state scale", truth.sigma[1], model.sigma[1]),
        ("Low-state degrees of freedom", truth.nu[0], model.nu[0]),
        ("High-state degrees of freedom", truth.nu[1], model.nu[1]),
        ("Low-state stay probability", truth.A[0, 0], model.A[0, 0]),
        ("High-state stay probability", truth.A[1, 1], model.A[1, 1]),
    ]
    print(f"Simulated {LENGTH} days from a known two-state Student-t HMM, then fitted a fresh model.\n")
    print(f"{'Parameter':<32}{'True':>10}{'Fitted':>10}")
    for name, true_value, fitted in rows:
        print(f"{name:<32}{true_value:>10.3f}{fitted:>10.3f}")
    print(f"\nViterbi path matches the true hidden state on {agreement:.1%} of days.")
    print(f"Baum-Welch stopped after {len(model.log_likelihood_history_)} iterations.")

    if args.no_plot:
        return
    import matplotlib.pyplot as plt

    days = np.arange(LENGTH)
    probability = model.predict_proba(changes)[:, 1]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    top.plot(days, changes, color="#1a1a1a", lw=0.5)
    top.fill_between(days, changes.min(), changes.max(), where=true_states == 1,
                     color="#c0392b", alpha=0.15, lw=0, label="True high-volatility state")
    top.set_ylabel("Simulated daily change")
    top.legend(loc="upper right", fontsize=9)
    top.set_title("Demo: the fitted model recovers hidden regimes in simulated data",
                  loc="left", fontweight="bold")
    bottom.plot(days, probability, color="#c0392b", lw=0.8)
    bottom.set_ylabel("Fitted P(high)")
    bottom.set_xlabel("Day")
    bottom.set_ylim(0, 1)
    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
    out = Path("demo_regimes.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"Chart written to {out.resolve()}")


if __name__ == "__main__":
    main()
