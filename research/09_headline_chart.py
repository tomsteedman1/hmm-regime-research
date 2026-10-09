"""Headline chart for the repository README.

Fits the selected model (T2: two states, Student-t emissions) to the frozen
WTI sample from its deterministic start, checks the fit against the
registered result in ``outputs/model_comparison/T2_summary.json``, and draws
the price with the high-volatility regime shaded.

Output: ``outputs/headline/wti_regimes.png``.
"""

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_hmm import RegimeHMM
from series import OUTPUT_ROOT
from wti_data import load_changes, load_prices

OUTPUT_DIR = OUTPUT_ROOT / "headline"
REGISTERED = OUTPUT_ROOT / "model_comparison" / "T2_summary.json"
FIT_KWARGS = {"n_iter": 500, "tol": 1e-7}

LOW_COLOUR = "#2e8b57"
HIGH_COLOUR = "#c0392b"
PRICE_COLOUR = "#1a1a1a"

EVENTS = [
    ("2014-11-27", "OPEC holds output,\nprice collapse"),
    ("2020-04-20", "Front month\nbelow zero"),
    ("2022-02-24", "Russia invades\nUkraine"),
]


def spells(states, index):
    """Return (start, end) dates of each run of the high-volatility state."""
    runs = []
    start = None
    for i, state in enumerate(states):
        if state == 1 and start is None:
            start = index[i]
        if state == 0 and start is not None:
            runs.append((start, index[i]))
            start = None
    if start is not None:
        runs.append((start, index[-1]))
    return runs


def main():
    changes = load_changes()
    prices = load_prices()
    values = changes.to_numpy()

    model = RegimeHMM(n_states=2, emission="student_t", **FIT_KWARGS)
    model.fit(values)
    registered = json.loads(REGISTERED.read_text())
    gap = abs(model.score(values) - registered["log_likelihood"])
    if gap > 1e-3:
        raise SystemExit(f"Fit differs from the registered T2 result by {gap:.4f} log-likelihood")

    states = model.decode(values)
    high_probability = pd.Series(model.predict_proba(values)[:, 1], index=changes.index)
    stdev = model.stdev
    durations = 1.0 / (1.0 - np.diag(model.A))

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )
    for start, end in spells(states, changes.index):
        for axis in (top, bottom):
            axis.axvspan(start, end, color=HIGH_COLOUR, alpha=0.16, lw=0)

    top.plot(prices.index, prices.to_numpy(), color=PRICE_COLOUR, lw=0.9)
    top.set_ylabel("WTI front month, $/bbl")
    top.set_ylim(-45, 150)
    top.axhline(0, color="#888888", lw=0.6)
    for date, label in EVENTS:
        when = pd.Timestamp(date)
        top.annotate(
            label, xy=(when, 149), ha="center", va="top", fontsize=8.5, color="#333333",
        )
    top.set_title(
        "WTI crude 2010 to 2024: the volatility regimes found by a two-state Student-t HMM",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="white", ec="#bbbbbb"),
        plt.Rectangle((0, 0), 1, 1, color=HIGH_COLOUR, alpha=0.3, lw=0),
    ]
    labels = [
        f"Low volatility: st. dev. ${stdev[0]:.2f}/day, mean spell {durations[0]:.0f} days",
        f"High volatility: st. dev. ${stdev[1]:.2f}/day, mean spell {durations[1]:.0f} days",
    ]
    top.legend(handles, labels, loc="lower left", fontsize=9, frameon=True, framealpha=0.95)

    bottom.plot(high_probability.index, high_probability.to_numpy(), color=HIGH_COLOUR, lw=0.8)
    bottom.set_ylim(0, 1)
    bottom.set_yticks([0, 0.5, 1])
    bottom.set_ylabel("P(high vol)")
    bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bottom.set_xlim(changes.index[0], changes.index[-1])

    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    bottom.text(
        0.0, -0.42,
        "Shading: most likely state path (Viterbi). Lower panel: smoothed probability of the "
        "high-volatility state. Model fitted to daily dollar price changes.",
        fontsize=8, color="#555555", transform=bottom.transAxes,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "wti_regimes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"log-likelihood gap to registered T2: {gap:.2e}")
    print(f"high-volatility share of days: {states.mean():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
