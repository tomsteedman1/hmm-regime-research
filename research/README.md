# Research: volatility regimes in WTI crude

Can a hidden Markov model find stable volatility regimes in daily WTI
crude oil futures prices, 2010 to 2024? The models are written from
scratch with numpy and scipy. `hmmlearn` is used only as a cross-check
in one test.

## Method

- The observation is the daily dollar price change. Log returns are
  undefined on 20 April 2020, when the front-month price fell below zero
  (`DATA_DECISION.md`).
- The price data was frozen and hashed before any fit
  (`DATA_MANIFEST.md`). A 2025 to 2026 holdout is sealed and no script
  reads it.
- Pass and fail thresholds were registered in `GATES.md` before the
  results they judge. Changes are dated in its change log.
- Every result document records the failed checks as well as the passed
  ones.

## Selected model

Two-state Student-t HMM (T2). It is the only one of six candidates (two,
three and four states, Gaussian or Student-t emissions) that passes every
registered gate. Details in `MODEL_COMPARISON_RESULT.md`.

| State | Daily mean | Scale | Degrees of freedom | Implied st. dev. | Self-transition | Expected duration | Share of days |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low volatility | +$0.05 | $1.02 | 14.5 | $1.09 | 99.16% | 119 days | 72.1% |
| High volatility | -$0.03 | $1.74 | 4.2 | $2.39 | 98.15% | 54 days | 27.9% |

The two-state Gaussian high state (4.8% of days, 7.5-day duration) was an
outlier state built around April 2020. The Student-t model takes those
days into its tails and finds a broad, persistent volatile regime.

Later checks, each registered before it was run:

- Without the two days the price crossed zero, T2's state path changes on
  0.2% of dates.
- Without the 180 roll days, every candidate keeps at least 94% Viterbi
  agreement with its full fit. T2 keeps 96.2%.
- A scale-free alternative, the same model on asinh returns, fails the
  sample-stability gate and scores worse on both selection criteria once
  both models are scored in dollar units (`SCALE_FREE_RESULT.md`).
- The likelihood and decoding match `hmmlearn` 0.3.3 in a cross-check
  test.

Known limits: volatility clustering remains in the residuals, dollar
changes are not scale-free across price levels, and the low-state degrees
of freedom have a wide bootstrap interval (8 to 39).

## Stages

| Script | What it does |
|---|---|
| `01_data_audit.py` | Data quality and the April 2020 audit |
| `01_return_comparison.py` | Compares log returns, simple returns and dollar changes as inputs |
| `02_two_state_validation.py` | Two-state Gaussian model on dollar changes |
| `03_multi_start_validation.py` | Local optima and state-path stability across starting values |
| `04_sample_stability.py` | Sensitivity to the period and to April 2020 (gate failed for the Gaussian model) |
| `05_gaussian_diagnostics.py` | PIT adequacy, mixture baseline, bootstrap, durations |
| `06_model_comparison.py` | Two, three and four states, Gaussian and Student-t, against every gate |
| `07_selected_model_bootstrap.py` | Bootstrap standard errors for the selected model |
| `08_scale_free_comparison.py` | Asinh-return candidate against T2 in dollar units |

Scripts 06 and 07 read their input from `series.py`. A new market or
observation is one new entry there, then
`python 06_model_comparison.py --series NAME`.

## Code

- `regime_hmm.py`: K-state HMM with Gaussian or Student-t emissions.
  Scaled forward-backward, Baum-Welch, Viterbi, filtered and smoothed
  probabilities.
- `gaussian_hmm.py`: the first two-state Gaussian model, kept as the
  reference.
- `multi_start.py`, `diagnostics.py`: multi-start fitting, probability
  integral transform, bootstrap and state durations.
- `series.py`, `wti_data.py`, `roll_dates.py`: frozen data access,
  registered subsamples and the CL roll-day rule.

## Running

From the repository root, install the packages and download the data
(see the root `README.md`), then:

```bash
cd research
python 01_data_audit.py        # and so on through 08
python -m pytest
```

The `hmmlearn` cross-check test skips when `hmmlearn` is not installed.
