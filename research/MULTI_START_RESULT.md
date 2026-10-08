# Multiple-Initialisation Result

Run date: 11 August 2026

## Method

The two-state Gaussian HMM was fitted 12 times to the WTI daily price-change
series. Start 0 used the original deterministic parameters. The remaining 11
fits used reproducible random initial probabilities, transition persistence,
means and standard deviations.

Each fitted model ordered its states by estimated volatility. This removed
label-switching differences before comparison.

## Result

All 12 fits converged to the same Viterbi state path. Every date received the
same state label across all starts.

The maximum fitted log-likelihood difference was approximately
`0.0000000254`. High-state volatility varied by about `0.0000308` dollars per
barrel. High-state occupancy was `4.7745%` in every fit.

The fits required between 26 and 42 Baum-Welch iterations. Different starting
points changed the route and speed of convergence. They produced the same
empirical solution within numerical precision.

Start 4 had the highest raw log-likelihood. Its advantage was approximately
`0.0000000028` log-likelihood units over several other starts. This difference
has no practical meaning at the precision used for the model results.

## Interpretation

The current two-state solution is stable across the tested initialisation
family and seeds. This evidence reduces concern about a local optimum for this
specific model, sample and observation definition.

This test covers 12 starting points. Wider initialisation rules, different
samples, other emission distributions and other state counts remain separate
stability questions.

## Reproduction

Run:

```bash
python 03_multi_start_validation.py
```

Detailed tables and charts are in `outputs/multi_start_validation`.
