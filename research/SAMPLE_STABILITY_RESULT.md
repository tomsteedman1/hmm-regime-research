# Sample-Stability Result

Run date: 11 August 2026

## Method

The two-state Gaussian HMM was fitted to four WTI price-change samples:

- Full sample from 2010 to 2024
- Early sample from 2010 to 2016
- Late sample from 2017 to 2024
- Full sample with 20 and 21 April 2020 removed

Each sample used eight initialisations. Every set of starts converged to one
state path within its sample.

## Results

| Sample | High-state volatility | High-state occupancy | Expected duration | Agreement with full fit |
|---|---:|---:|---:|---:|
| Full 2010 to 2024 | $6.57 | 4.77% | 7.5 days | 100.00% |
| Early 2010 to 2016 | $2.15 | 19.84% | 12.1 days | 82.60% |
| Late 2017 to 2024 | $7.61 | 5.57% | 10.8 days | 98.76% |
| Full without cross-zero days | $2.55 | 22.96% | 17.7 days | 81.77% |

## Finding

The full-sample solution passes the initialisation test and fails the
sample-stability gate. The April 2020 observations have a large effect on the
Gaussian high-volatility state. Their inclusion creates a narrow state with
high conditional volatility. Their removal produces a broader state that
covers more persistent volatile periods.

The early and late fits also estimate different state properties. Dollar price
changes and the Gaussian emission assumption both contribute to this
sensitivity.

## Research consequence

The current Gaussian result remains a diagnostic model. Its state parameters
and historical classifications are unsuitable for final economic claims.

Robust emissions now take priority over state-count selection. The next model
will test Student-t emissions on the same daily price-change data. The same
full-sample, period and cross-zero sensitivity checks will then be repeated.

## Reproduction

Run:

```bash
python 04_sample_stability.py
```

Outputs are in `outputs/sample_stability`.
