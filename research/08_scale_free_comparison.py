"""Apply the G5 rule registered on 8 October 2026: A2 on asinh returns against T2.

Both models are scored in dollar-change units. A2's log-likelihoods carry the
change-of-variables correction written by ``06_model_comparison.py --series
wti_asinh10``. A2 replaces T2 only if it passes G3, G4 on the three registered
subsamples, the G5 occupancy rule and the G6 persistence test, and beats T2
on both corrected BIC and corrected held-out log-likelihood. A2G is a
control and cannot replace T2.

Run after ``06_model_comparison.py --series wti_asinh10``. Writes
``outputs/scale_free/comparison.csv``.
"""

import json

import pandas as pd

from series import get_series


WTI = get_series("wti")
ASINH = get_series("wti_asinh10")
G5_MIN_OCCUPANCY = 0.02


def load(series, name):
    with open(series.output_dir / f"{name}_summary.json") as handle:
        return json.load(handle)


def viterbi(series, name):
    daily = pd.read_csv(series.output_dir / f"{name}_daily.csv", parse_dates=["date"])
    return daily.set_index("date")["viterbi_state"]


def main():
    t2 = load(WTI, "T2")
    rows = [
        {
            "candidate": "T2",
            "observation": "price change",
            "dollar_log_likelihood": t2["log_likelihood"],
            "dollar_bic": t2["bic"],
            "dollar_heldout_log_likelihood": t2["heldout_log_likelihood"],
            "parameters": t2["parameters"],
            "g3_pass": t2["g3_pass"],
            "g4_pass": t2["g4_pass"],
            "g4_roll_record_pass": t2["g4_record_full_without_roll_days_pass"],
            "min_occupancy": t2["min_occupancy"],
            "g6_persistence_pass": t2["g6_persistence_pass"],
            "viterbi_agreement_with_t2": 1.0,
        }
    ]
    t2_path = viterbi(WTI, "T2")
    for name in ("A2", "A2G"):
        summary = load(ASINH, name)
        path = viterbi(ASINH, name)
        rows.append(
            {
                "candidate": name,
                "observation": "asinh return, scale $10",
                "dollar_log_likelihood": summary["corrected_log_likelihood"],
                "dollar_bic": summary["corrected_bic"],
                "dollar_heldout_log_likelihood": summary["corrected_heldout_log_likelihood"],
                "parameters": summary["parameters"],
                "g3_pass": summary["g3_pass"],
                "g4_pass": summary["g4_pass"],
                "g4_roll_record_pass": summary["g4_record_full_without_roll_days_pass"],
                "min_occupancy": summary["min_occupancy"],
                "g6_persistence_pass": summary["g6_persistence_pass"],
                "viterbi_agreement_with_t2": (path == t2_path.reindex(path.index)).mean(),
            }
        )
    table = pd.DataFrame(rows).set_index("candidate")
    a2, ref = table.loc["A2"], table.loc["T2"]
    gates = bool(
        a2["g3_pass"]
        and a2["g4_pass"]
        and a2["min_occupancy"] >= G5_MIN_OCCUPANCY
        and a2["g6_persistence_pass"]
    )
    better_bic = bool(a2["dollar_bic"] < ref["dollar_bic"])
    better_heldout = bool(a2["dollar_heldout_log_likelihood"] > ref["dollar_heldout_log_likelihood"])
    table["replaces_t2"] = ""
    table.loc["T2", "replaces_t2"] = "selected before 8 Oct 2026"
    table.loc["A2G", "replaces_t2"] = "control, cannot replace"
    table.loc["A2", "replaces_t2"] = str(gates and better_bic and better_heldout)
    table.to_csv(ASINH.output_dir / "comparison.csv")

    print(table.T.to_string())
    print(
        f"\nA2 gates {'pass' if gates else 'fail'}; corrected BIC "
        f"{'better' if better_bic else 'worse'} ({a2['dollar_bic'] - ref['dollar_bic']:+.2f}); "
        f"corrected held-out {'better' if better_heldout else 'worse'} "
        f"({a2['dollar_heldout_log_likelihood'] - ref['dollar_heldout_log_likelihood']:+.2f})."
    )
    print("A2 replaces T2." if gates and better_bic and better_heldout else "T2 stays selected.")


if __name__ == "__main__":
    main()
