"""Bootstrap standard errors for the model selected under gate G5.

The selected model is refitted on the full sample from the multi-start
procedure, then refitted on data simulated from itself. Degrees of freedom
are weakly identified, so this script reports their uncertainty explicitly.

Usage: ``python 07_selected_model_bootstrap.py [--series NAME] [--model NAME]
[--output-dir DIR]``. Defaults: series ``wti``, model ``T2``, the series'
own output folder.
"""

import importlib
import sys
from pathlib import Path

from diagnostics import bootstrap_summary, parametric_bootstrap
from multi_start import fit_multiple_starts_general
from regime_hmm import RegimeHMM
from series import get_series


CANDIDATES = importlib.import_module("06_model_comparison").CANDIDATES
N_STARTS = 12
N_DRAWS = 100
FIT_KWARGS = {"n_iter": 500, "tol": 1e-7}


def parse_args(argv):
    options = {"--series": "wti", "--model": "T2", "--output-dir": None}
    arguments = iter(argv)
    for argument in arguments:
        if argument not in options:
            raise SystemExit(f"Unknown argument {argument!r}")
        options[argument] = next(arguments)
    return options["--series"], options["--model"], options["--output-dir"]


def main(argv):
    series_name, model_name, output_dir = parse_args(argv)
    series = get_series(series_name)
    if model_name not in CANDIDATES:
        raise SystemExit(f"Unknown model {model_name!r}")
    n_states, emission = CANDIDATES[model_name]
    output_dir = Path(output_dir) if output_dir else series.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    values = series.load_observations().to_numpy()
    model, _, _ = fit_multiple_starts_general(
        values,
        lambda strategy, seed: RegimeHMM(
            n_states=n_states,
            emission=emission,
            init_strategy=strategy,
            random_state=seed,
            **FIT_KWARGS,
        ),
        n_starts=N_STARTS,
        base_seed=20260811,
    )
    print(f"Selected model log-likelihood {model.score(values):.4f}")
    print(f"Running {N_DRAWS} parametric bootstrap draws.")
    draws = parametric_bootstrap(
        model, len(values), n_draws=N_DRAWS, seed=20260902, fit_kwargs=FIT_KWARGS
    )
    draws.to_csv(output_dir / f"{model_name}_bootstrap_draws.csv", index=False)
    summary = bootstrap_summary(model, draws)
    summary.to_csv(output_dir / f"{model_name}_bootstrap_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
