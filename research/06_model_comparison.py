"""Stage 2 and 3: Gaussian versus Student-t emissions, two to four states.

Every candidate passes through the gates registered in GATES.md before any
result is read:

- G3 initialisation stability, 12 starts on the full sample
- G4 sample stability, three subsamples with 8 starts each
- G5 model selection, BIC and held-out conditional log-likelihood
- G6 adequacy, independent-mixture comparison and PIT diagnostics

Results are written after each candidate so a long run can be resumed.
Pass candidate names on the command line to run a subset, for example
``python 06_model_comparison.py G2 T2``. ``--series NAME`` picks a series
registered in ``series.py`` (default ``wti``); ``--output-dir DIR`` writes
somewhere other than the series' own folder, for example to check that a
rerun reproduces the saved outputs. ``--table-only`` fits nothing and
rebuilds the summary table and charts from saved results, after
candidates were run in parallel processes.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics import pit_diagnostics
from multi_start import fit_multiple_starts_general
from regime_hmm import RegimeHMM
from series import get_series


# Set by main() from the chosen series.
SERIES = get_series("wti")
OUTPUT_DIR = SERIES.output_dir
CANDIDATES = {
    "G2": (2, "gaussian"),
    "G3": (3, "gaussian"),
    "G4": (4, "gaussian"),
    "T2": (2, "student_t"),
    "T3": (3, "student_t"),
    "T4": (4, "student_t"),
    # On the asinh observation (series wti_asinh10).
    "A2G": (2, "gaussian"),
    "A2": (2, "student_t"),
}
FULL_STARTS = 12
SAMPLE_STARTS = 8
TRAIN_STARTS = 8
MIXTURE_STARTS = 4
BASE_SEED = 20260811
FIT_KWARGS = {"n_iter": 500, "tol": 1e-7}

# Gate thresholds, copied from GATES.md.
G3_MIN_STARTS_NEAR_BEST = 10
G3_LIKELIHOOD_WINDOW = 1.0
G3_MIN_AGREEMENT = 0.99
G4_MIN_AGREEMENT = 0.90
G4_MAX_OCCUPANCY_SHIFT = 0.05
G5_MIN_OCCUPANCY = 0.02
G6_SOFT_P_VALUE = 0.01

STATE_COLOURS = ["#2e8b57", "#e6a700", "#c00000", "#4b0082"]


def make_factory(n_states, emission, markov=True):
    def factory(init_strategy, random_state):
        return RegimeHMM(
            n_states=n_states,
            emission=emission,
            init_strategy=init_strategy,
            random_state=random_state,
            markov=markov,
            **FIT_KWARGS,
        )

    return factory


def state_columns(model):
    record = {}
    for state in range(model.n_states):
        record[f"mu_{state}"] = model.mu[state]
        record[f"sigma_{state}"] = model.sigma[state]
        record[f"stdev_{state}"] = model.stdev[state]
        record[f"self_transition_{state}"] = model.A[state, state]
        record[f"expected_duration_{state}"] = 1.0 / (1.0 - model.A[state, state])
        if model.nu is not None:
            record[f"nu_{state}"] = model.nu[state]
    return record


def per_observation_log_likelihood(model, values):
    probabilities = model._emission_probabilities(values)
    _, scales = model._forward(probabilities)
    return np.log(scales + 1e-300)


def fit_subsample(name, sample_name, sample, full_path):
    """Fit one G4 subsample and compare its Viterbi path with the full-sample path."""
    n_states, emission = CANDIDATES[name]
    top = n_states - 1
    sample_best, sample_records, _ = fit_multiple_starts_general(
        sample.to_numpy(),
        make_factory(n_states, emission),
        SAMPLE_STARTS,
        BASE_SEED,
    )
    sample_path = pd.Series(sample_best.decode(sample.to_numpy()), index=sample.index)
    common = sample_path.index.intersection(full_path.index)
    agreement = (sample_path.loc[common] == full_path.loc[common]).mean()
    shift = (sample_path.loc[common] == top).mean() - (full_path.loc[common] == top).mean()
    row = {
        "candidate": name,
        "sample": sample_name,
        "observations": len(sample),
        "log_likelihood": sample_best.score(sample.to_numpy()),
        "agreement_with_full": agreement,
        "top_state_occupancy_shift": shift,
        "starts_near_best": int(
            (pd.DataFrame(sample_records)["log_likelihood_gap"] <= G3_LIKELIHOOD_WINDOW).sum()
        ),
    }
    row.update(state_columns(sample_best))
    sample_path.rename(sample_name).to_csv(
        OUTPUT_DIR / f"{name}_path_{sample_name}.csv", index_label="date"
    )
    return row


def run_candidate(name, changes, price):
    n_states, emission = CANDIDATES[name]
    values = changes.to_numpy()
    started = time.time()
    summary = {"candidate": name, "n_states": n_states, "emission": emission}

    # Full-sample fit and G3.
    best, records, paths = fit_multiple_starts_general(
        values, make_factory(n_states, emission), FULL_STARTS, BASE_SEED
    )
    starts = pd.DataFrame(records)
    starts.to_csv(OUTPUT_DIR / f"{name}_starts.csv", index=False)
    near_best = starts["log_likelihood_gap"] <= G3_LIKELIHOOD_WINDOW
    summary.update(
        {
            "log_likelihood": best.score(values),
            "parameters": best.n_parameters,
            "aic": best.aic(values),
            "bic": best.bic(values),
            "iterations": best.n_iter_,
            "g3_starts_near_best": int(near_best.sum()),
            "g3_min_agreement_near_best": starts.loc[near_best, "agreement_with_best"].min(),
        }
    )
    summary["g3_pass"] = bool(
        summary["g3_starts_near_best"] >= G3_MIN_STARTS_NEAR_BEST
        and summary["g3_min_agreement_near_best"] >= G3_MIN_AGREEMENT
    )
    summary.update(state_columns(best))

    full_path = pd.Series(best.decode(values), index=changes.index)
    posterior = best.predict_proba(values)
    filtered = best.filter_proba(values)
    daily = pd.DataFrame({"viterbi_state": full_path}, index=changes.index)
    for state in range(n_states):
        daily[f"smoothed_{state}"] = posterior[:, state]
        daily[f"filtered_{state}"] = filtered[:, state]
    daily.to_csv(OUTPUT_DIR / f"{name}_daily.csv", index_label="date")
    occupancy = np.array([(full_path == state).mean() for state in range(n_states)])
    for state in range(n_states):
        summary[f"occupancy_{state}"] = occupancy[state]
    summary["min_occupancy"] = occupancy.min()

    # G4 sample stability.
    sample_rows = [
        fit_subsample(name, sample_name, sample, full_path)
        for sample_name, sample in SERIES.stability_samples(changes).items()
        if sample_name != "full_2010_2024"
    ]
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(OUTPUT_DIR / f"{name}_samples.csv", index=False)
    summary["g4_min_agreement"] = samples["agreement_with_full"].min()
    summary["g4_max_occupancy_shift"] = samples["top_state_occupancy_shift"].abs().max()
    summary["g4_pass"] = bool(
        summary["g4_min_agreement"] >= G4_MIN_AGREEMENT
        and summary["g4_max_occupancy_shift"] <= G4_MAX_OCCUPANCY_SHIFT
    )

    # Record-only G4 subsamples: reported, never part of the G4 pass.
    record_rows = [
        fit_subsample(name, sample_name, sample, full_path)
        for sample_name, sample in SERIES.record_samples(changes).items()
    ]
    if record_rows:
        pd.DataFrame(record_rows).to_csv(OUTPUT_DIR / f"{name}_samples_record.csv", index=False)
    for row in record_rows:
        key = f"g4_record_{row['sample']}"
        summary[f"{key}_agreement"] = row["agreement_with_full"]
        summary[f"{key}_occupancy_shift"] = row["top_state_occupancy_shift"]
        summary[f"{key}_pass"] = bool(
            row["agreement_with_full"] >= G4_MIN_AGREEMENT
            and abs(row["top_state_occupancy_shift"]) <= G4_MAX_OCCUPANCY_SHIFT
        )

    # G5 held-out conditional log-likelihood.
    train, test = SERIES.train_test_split(changes)
    train_best, _, _ = fit_multiple_starts_general(
        train.to_numpy(), make_factory(n_states, emission), TRAIN_STARTS, BASE_SEED
    )
    joint = per_observation_log_likelihood(train_best, values)
    test_mask = changes.index >= test.index[0]
    test_contributions = pd.Series(joint[test_mask], index=changes.index[test_mask])
    summary["train_log_likelihood"] = train_best.score(train.to_numpy())
    summary["heldout_log_likelihood"] = test_contributions.sum()
    if SERIES.excluded_key:
        summary[f"heldout_log_likelihood_excl_{SERIES.excluded_key}"] = test_contributions.drop(
            SERIES.excluded_dates.intersection(test_contributions.index)
        ).sum()
    summary["heldout_observations"] = int(test_mask.sum())

    # Change of variables to dollar-change units (GATES.md, 8 October 2026).
    if SERIES.log_jacobian is not None:
        jacobian = SERIES.log_jacobian(price).reindex(changes.index)
        summary["log_jacobian_sum"] = jacobian.sum()
        summary["corrected_log_likelihood"] = summary["log_likelihood"] + jacobian.sum()
        summary["corrected_bic"] = summary["bic"] - 2.0 * jacobian.sum()
        summary["corrected_heldout_log_likelihood"] = (
            summary["heldout_log_likelihood"] + jacobian[test_mask].sum()
        )
    test_contributions.rename("log_likelihood").to_csv(
        OUTPUT_DIR / f"{name}_heldout_contributions.csv", index_label="date"
    )

    # G6 adequacy.
    mixture, _, _ = fit_multiple_starts_general(
        values, make_factory(n_states, emission, markov=False), MIXTURE_STARTS, BASE_SEED
    )
    summary["mixture_log_likelihood"] = mixture.score(values)
    summary["mixture_bic"] = mixture.bic(values)
    summary["g6_persistence_pass"] = bool(summary["bic"] < summary["mixture_bic"])
    diagnostics = pit_diagnostics(best, values)
    for key in (
        "ks_p_value",
        "ljung_box_p_value",
        "ljung_box_squared_p_value",
        "residual_skewness",
        "residual_excess_kurtosis",
    ):
        summary[key] = diagnostics[key]
    summary["g6_uniformity_pass"] = bool(diagnostics["ks_p_value"] >= G6_SOFT_P_VALUE)
    summary["g6_dependence_pass"] = bool(
        diagnostics["ljung_box_p_value"] >= G6_SOFT_P_VALUE
        and diagnostics["ljung_box_squared_p_value"] >= G6_SOFT_P_VALUE
    )
    pd.DataFrame(
        {"pit": diagnostics["pit"], "residual": diagnostics["residual"]},
        index=changes.index,
    ).to_csv(OUTPUT_DIR / f"{name}_pit.csv", index_label="date")

    summary["seconds"] = time.time() - started
    with open(OUTPUT_DIR / f"{name}_summary.json", "w") as handle:
        json.dump({key: _jsonable(value) for key, value in summary.items()}, handle, indent=2)
    print(
        f"{name}: loglik {summary['log_likelihood']:.2f}, BIC {summary['bic']:.2f}, "
        f"held-out {summary['heldout_log_likelihood']:.2f}, "
        f"G3 {'pass' if summary['g3_pass'] else 'fail'}, "
        f"G4 {'pass' if summary['g4_pass'] else 'fail'}, "
        f"{summary['seconds']:.0f}s",
        flush=True,
    )
    return summary


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_summaries(names):
    rows = []
    for name in names:
        path = OUTPUT_DIR / f"{name}_summary.json"
        if path.exists():
            with open(path) as handle:
                rows.append(json.load(handle))
    return pd.DataFrame(rows)


def apply_g5(table):
    """Apply the G5 selection rule against the reference model."""
    if SERIES.reference not in set(table["candidate"]):
        table["g5_pass"] = "no reference"
        return table
    reference = table.set_index("candidate").loc[SERIES.reference]
    table["bic_better_than_reference"] = table["bic"] < reference["bic"]
    table["heldout_better_than_reference"] = (
        table["heldout_log_likelihood"] > reference["heldout_log_likelihood"]
    )
    table["occupancy_pass"] = table["min_occupancy"] >= G5_MIN_OCCUPANCY
    table["g5_pass"] = (
        table["bic_better_than_reference"]
        & table["heldout_better_than_reference"]
        & table["g3_pass"]
        & table["g4_pass"]
        & table["occupancy_pass"]
        & table["g6_persistence_pass"]
    )
    table["g5_pass"] = table["g5_pass"].astype(object)
    table.loc[table["candidate"] == SERIES.reference, "g5_pass"] = "reference"
    return table


def save_state_path_chart(names, changes, price):
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(14, 2.2 * (len(names) + 1)), sharex=True)
    axes[0].plot(price.index, price, color="black", linewidth=0.8)
    axes[0].set_ylabel(SERIES.price_unit)
    axes[0].set_title(SERIES.price_title)
    for axis, name in zip(axes[1:], names):
        path = OUTPUT_DIR / f"{name}_daily.csv"
        if not path.exists():
            continue
        daily = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        n_states = CANDIDATES[name][0]
        for state in range(n_states):
            mask = daily["viterbi_state"].eq(state)
            axis.fill_between(
                daily.index, 0, 1, where=mask, color=STATE_COLOURS[state], alpha=0.8, step="mid"
            )
        axis.set_yticks([])
        axis.set_title(f"{name}: {n_states}-state {CANDIDATES[name][1].replace('_', '-')}")
    handles = [
        mpatches.Patch(color=STATE_COLOURS[state], label=f"State {state} (scale rank {state})")
        for state in range(4)
    ]
    axes[1].legend(handles=handles, loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_state_paths.png", dpi=160)
    plt.close(fig)


def save_criteria_chart(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    order = table.sort_values("bic")
    axes[0].barh(order["candidate"], order["bic"], color="#4c72b0")
    axes[0].set_title("BIC on the full sample (lower is better)")
    axes[0].invert_yaxis()
    order = table.sort_values("heldout_log_likelihood", ascending=False)
    axes[1].barh(order["candidate"], order["heldout_log_likelihood"], color="#c44e52")
    axes[1].set_title("Held-out conditional log-likelihood, 2019 to 2024 (higher is better)")
    axes[1].invert_yaxis()
    for axis in axes:
        axis.margins(x=0.02)
        low = min(axis.get_xlim()[0], min(patch.get_width() for patch in axis.patches))
        axis.set_xlim(left=low)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_selection_criteria.png", dpi=160)
    plt.close(fig)


def save_pit_chart(names):
    present = [name for name in names if (OUTPUT_DIR / f"{name}_pit.csv").exists()]
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.4), squeeze=False)
    for axis, name in zip(axes[0], present):
        pit = pd.read_csv(OUTPUT_DIR / f"{name}_pit.csv")["pit"]
        axis.hist(pit, bins=20, color="#4c72b0", edgecolor="white")
        axis.axhline(len(pit) / 20, color="black", linestyle="--")
        axis.set_title(name)
        axis.set_yticks([])
    fig.suptitle("PIT histograms from filtered one-step-ahead predictions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_pit_histograms.png", dpi=160)
    plt.close(fig)


def parse_args(argv):
    series_name, output_dir, names = "wti", None, []
    arguments = iter(argv)
    for argument in arguments:
        if argument == "--series":
            series_name = next(arguments)
        elif argument == "--output-dir":
            output_dir = Path(next(arguments))
        elif argument == "--table-only":
            names.append(argument)
        else:
            names.append(argument)
    return series_name, output_dir, names


def main(argv):
    global SERIES, OUTPUT_DIR
    series_name, output_dir, names = parse_args(argv)
    SERIES = get_series(series_name)
    OUTPUT_DIR = output_dir or SERIES.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_only = "--table-only" in names
    names = [] if table_only else names or list(SERIES.candidates)
    unknown = [name for name in names if name not in SERIES.candidates]
    if unknown:
        raise SystemExit(f"Unknown candidates: {unknown}")
    price = SERIES.load_prices()
    changes = SERIES.load_observations()

    for name in names:
        run_candidate(name, changes, price)

    table = apply_g5(load_summaries(list(SERIES.candidates)))
    table.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    present = list(table["candidate"])
    save_state_path_chart(present, changes, price)
    save_criteria_chart(table)
    save_pit_chart(present)

    columns = [
        "candidate",
        "parameters",
        "log_likelihood",
        "bic",
        "heldout_log_likelihood",
        *([f"heldout_log_likelihood_excl_{SERIES.excluded_key}"] if SERIES.excluded_key else []),
        "min_occupancy",
        "g3_pass",
        "g4_min_agreement",
        "g4_max_occupancy_shift",
        "g4_pass",
        "g6_persistence_pass",
        "ks_p_value",
        "ljung_box_squared_p_value",
        "g5_pass",
    ]
    print("\nCandidate summary")
    print(table[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFiles saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
