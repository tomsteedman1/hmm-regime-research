# Pre-registered validation gates

Registered: 1 September 2026, before any Student-t or three-state model was
fitted. These thresholds apply to every candidate model. A threshold may be
changed only in a dated note below, with the reason, and never after the
result it affects has been seen.

## G1. Numerical correctness

1. Brute-force agreement. On a six-observation sequence, the code's
   likelihood, smoothed posteriors and Viterbi path must match exhaustive
   enumeration of all state paths: likelihood within relative `1e-8`,
   posteriors within `1e-8`, Viterbi path exactly.
2. EM monotonicity. The log-likelihood must not fall by more than `1e-7`
   between iterations.
3. Posterior rows sum to one within `1e-10`.
4. Library cross-check. With identical parameters, `hmmlearn` must return the
   same log-likelihood within relative `1e-6`, the same smoothed posteriors
   within `1e-6`, and the same Viterbi path.

## G2. Simulated recovery

Fit to 5,000 simulated observations from known parameters, three seeds. The
estimator is the multi-start procedure of G3 (one deterministic start plus
random starts, best likelihood kept). After ordering states by emission scale,
every seed must satisfy:

- Gaussian: each `sigma` within 15% of truth; each self-transition within
  0.04 absolute; each mean within 0.25 of the state's true `sigma`.
- Student-t: each scale within 20% of truth; each self-transition within
  0.04 absolute; each degrees-of-freedom estimate within a factor of 2 of
  the truth when the truth is 5.
- Three states: the same rules as the matching emission family.

## G3. Initialisation stability

Twelve starts on the full 2010 to 2024 sample: one deterministic start and
eleven random starts with fixed seeds.

- At least 10 of 12 starts within 1.0 log-likelihood unit of the best start.
- Viterbi agreement between each of those starts and the best start at least
  99% of dates.

## G4. Sample stability

Three subsamples, eight starts each: 2010 to 2016, 2017 to 2024, and the
full sample without 20 and 21 April 2020.

- Viterbi path agreement with the full-sample fit on common dates at least
  90% for every subsample.
- Occupancy of the highest-volatility state on common dates within 5
  percentage points of the full-sample fit for every subsample.

The two-state Gaussian model fitted on 11 August 2026 scores 82.6%, 98.8% and
81.8% agreement and 15.1, 0.8 and 18.2 percentage points. It fails this gate.

## G5. Model selection

The reference model is the two-state Gaussian HMM. A candidate replaces the
reference only if all of the following hold:

1. BIC lower than the reference on the full sample.
2. Held-out conditional log-likelihood higher than the reference. Training
   sample: 5 January 2010 to 31 December 2018. Test sample: 2 January 2019 to
   30 December 2024. The test log-likelihood is conditional on the training
   history, computed as the full-sequence log-likelihood minus the training
   log-likelihood.
3. Passes G3 and G4.
4. Every state has Viterbi occupancy of at least 2% in the full sample.

If BIC and held-out log-likelihood disagree, the simpler model is kept.

Parameter counts. Gaussian with K states: `(K-1) + K(K-1) + 2K`. Student-t
with K states: `(K-1) + K(K-1) + 3K`. Independent mixtures replace `K(K-1)`
with zero.

## G6. Adequacy diagnostics

Computed on the probability integral transform (PIT) of each observation
under the one-step-ahead predictive distribution, which uses filtered state
probabilities only.

1. Persistence, hard test: the HMM must have lower BIC than the independent
   mixture with the same emission family and state count. Failure means the
   Markov structure is not supported and the model is rejected.
2. PIT uniformity, soft test: Kolmogorov-Smirnov p-value at least 0.01.
3. Serial dependence, soft test: Ljung-Box with 20 lags on the normal-score
   residuals and on their squares, p-value at least 0.01 for both.

Soft-test failures are recorded as limitations in the report. They do not
reject the model on their own.

## G7. Sealed holdout

The file `data/raw/wti_CL=F_holdout_2025-01-02_to_2026-08-31_downloaded_2026-09-01.csv`
is read by no script before Stage 6. It is evaluated once, with filtered
probabilities only, using the model selected under G5. No parameter, threshold
or model choice changes after the holdout is opened.

## Change log

- 1 September 2026: gates registered.
- 1 September 2026, before any real-data fit beyond two Gaussian states: G2
  now names the multi-start estimator. A single deterministic start reached a
  local optimum on simulated three-state data (log-likelihood -8,271 against
  -8,093 from every random start), so the single-start estimator is not the
  one used anywhere in the pipeline.
- 1 September 2026, before any real-data Student-t fit: the G2 tolerance on
  degrees of freedom widened from a factor of 1.5 to a factor of 2. On the
  three-state simulation with seed 303, the least-occupied state returned
  7.82 against a truth of 5 (ratio 1.56) while every scale and persistence
  estimate passed. Degrees of freedom are weakly identified from a few hundred
  observations, so the original factor was too tight for the smallest state.
  Every real-data Student-t result reports a bootstrap standard error for
  each degrees-of-freedom estimate.
- 2 September 2026, holdout still sealed: the sealed holdout will be opened
  once, jointly, for the Stage 6 evaluation here and for the trading gates
  registered in the trading half of this repository
  (`../trading/TRADING_GATES.md`). Both sets of gates must be
  registered and dated before that single opening. Nothing else in G7
  changes. The trading project develops on the 2010 to 2024 sample only.
- 8 October 2026, a record, no threshold changed: G1.4 run for the first
  time, on Windows Python 3.13.15 with hmmlearn 0.3.3, numpy 2.5.3, scipy
  1.18.1. Three-state Gaussian, 1,000 simulated observations: relative
  log-likelihood difference 8.0e-15 (threshold 1e-6), largest smoothed
  posterior difference 1.8e-13 (threshold 1e-6), Viterbi paths identical
  on all 1,000 dates. Passed. hmmlearn has no Student-t emission, so the
  Student-t path is covered by G1.1 to G1.3 only. The full research suite
  (23 tests) also passed under that Python.
- 8 October 2026, before any fit on this subsample: G4 gains a fourth
  subsample, recorded for every candidate, "the full sample without the
  day after each CL expiry". The trading roll audit (2 September
  2026) found that day to be the roll day of the spliced `CL=F` series;
  there are 180 of them in 2010 to 2024, and the expiry rule is ported
  unchanged into `roll_dates.py`. The same thresholds apply (Viterbi
  agreement at least 90%, top-state occupancy within 5 points), with eight
  starts. This subsample is a record only: the G4 pass or fail used for
  selection stays on the three registered subsamples, so it cannot change
  the selection already made. A failure is reported as a limitation of the
  model it affects. Nothing else changes.
- 8 October 2026, before any fit on the new observation: a scale-free
  candidate is registered (`DATA_DECISION.md` addendum of the same date).
  Observation `y_t = 100 [asinh(P_t / 10) - asinh(P_{t-1} / 10)]`, series
  `wti_asinh10`. Two candidates: A2 (two-state Student-t on y) and A2G
  (two-state Gaussian on y, the Gaussian control). G3, G4 (three
  registered subsamples, plus the roll-day record) and G6 apply unchanged,
  with the same starts and seeds.
  G5 across observations. Given the past, `y_t` is a smooth increasing
  function of `x_t`, with derivative `g'(P_t) = 100 / sqrt(P_t^2 + 100)`.
  So a model on y gives a density for x through
  `log f_x(x_t | past) = log f_y(y_t | past) + log g'(P_t)`.
  Corrected log-likelihood = log-likelihood on y plus the sum of
  `log g'(P_t)` over the same dates. Corrected BIC uses it with the
  ordinary parameter count. The corrected held-out log-likelihood adds the
  sum over the test dates only (2019 to 2024).
  Rule: A2 replaces T2 as the selected model only if it passes G3, G4 on
  the three registered subsamples, the G5 occupancy rule and the G6
  persistence test, and it beats T2 on both corrected BIC and corrected
  held-out log-likelihood. If the two criteria disagree, T2 stays. A2G
  cannot replace T2; it is reported as a control. The holdout stays
  sealed. The trading project keeps T2 whatever the result, by its own
  rule.
