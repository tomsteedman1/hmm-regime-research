# Second exploratory programme: what is left to ask of a regime model

Written: 9 September 2026, before any experiment below was run and before
any result was seen. Companion to `RESEARCH_PROGRAMME.md`, whose rules
apply unchanged: development sample only (2010 to 2024, scored 2015 to
2024 walk-forward), no holdout opened, every result exploratory, per-asset
tables with Holm and one pooled test read as the evidence.

## Why a second programme

The first programme answered the trading question: the two-state
Student-t regime signal adds nothing tradable on its own. It left six
questions that follow from that answer rather than from the original one.
Each is chosen because it teaches something about regime models in
general, and each has a stated failure reading before it is run.

## Departures from the first programme, stated in advance

- E9 and E12 fit new models. The first programme fixed T2 on returns. E9
  fits a two-state Gaussian model on the log of a realised variance proxy,
  and E12 fits T2 on a cross-asset composite return. These are new
  questions, not a reselection of the model for the old question. The
  T2-on-returns signal in `outputs/signals` is never refitted or replaced.
- E11 varies the transaction cost, which the first programme fixed at one
  and two basis points. It does so to find the break-even cost, not to pick
  a cost at which the model wins.

## Experiments

### E7. Within-state volatility dynamics

The diagnosis in E2 was that the HMM forecast moves between two levels
and ignores day-to-day variation. Two hybrids let the level move within a
state, using the walk-forward one-step regime probability from
`outputs/signals` and past returns only:

- State-weighted EWMA: two exponentially weighted variances, one per
  state, each updated by the day's squared return with weight equal to
  that day's one-step probability of the state (span 21, no fitting).
  Forecast: probability-weighted mixture of the two.
- GARCH-X: the E2 GARCH(1,1)-t plus a term delta times the one-step
  high-state probability in the variance equation, fitted yearly by
  maximum likelihood. The training-period probabilities come from the
  same January model reconstructed from its saved parameters, so the
  regressor in training is what the model would have shown at the time.

Test: QLIKE on the squared-return proxy, Diebold-Mariano of each hybrid
against GARCH-t and against the plain HMM, per asset and pooled.

Failure reading: if neither hybrid beats GARCH-t pooled, the regime label
carries no variance information beyond what the GARCH recursion extracts
from the same returns.

### E8. Forecast combination and encompassing

Does the HMM add anything to a combination? Equal-weight average of the
variance forecasts (arithmetic and geometric) from EWMA, GARCH-t and
HAR-GK, with and without the HMM. Test: QLIKE, Diebold-Mariano of the
with-HMM combination against the without, per asset and pooled. A
forecast-encompassing regression (Mincer-Zarnowitz of the squared return
on the GARCH-t and HMM forecasts jointly, yearly refit) reports the HMM
weight out of sample.

Failure reading: if adding the HMM does not lower the pooled QLIKE, the
HMM is encompassed by the standard set.

### E9. Regime model on realised variance

The return-based HMM sees one noisy draw a day. A regime model on the log
of the daily Garman-Klass variance sees the level directly. Two-state
Gaussian HMM on log GK (squared return where GK is undefined), yearly
refit with the twelve-start estimator, one-step predictive mixture
forecast E[exp(y)] = sum_k p_k exp(mu_k + sigma_k^2 / 2). Test: QLIKE on
the squared-return proxy against HAR-GK (same input), GARCH-t and the
return HMM, per asset and pooled. Also reported: state durations against
the return HMM's, to see whether the two models find the same regimes.

Failure reading: if the RV regime model does not beat HAR-GK, regimes on
the level of volatility add nothing to a linear model of the same level.

### E10. Duration dependence

A hidden Markov chain has geometric state durations: the chance of
leaving a state does not depend on how long it has been occupied.
Descriptive test on the real-time state label (one-step high-state
probability above one half): empirical hazard by time-in-state bucket
(1 to 10, 11 to 30, 31 to 90, over 90 days) against the constant hazard
implied by the mean spell; a logistic regression of exit on the log of
time in state; pooled across assets with asset fixed effects.

Failure reading: a falling hazard means long spells are stickier than the
model believes, which argues for a semi-Markov model or a third state; a
flat hazard means the Markov assumption is fine and the model's slowness
is a feature of the data, not of the chain.

### E11. Break-even transaction cost

The HMM sizer turns over less than the EWMA sizer because it moves
between two levels. At some cost that thrift must matter. For each asset
and pooled, using the E3 vol-target positions: the ratio of HMM to EWMA
turnover, and the cost per unit of turnover at which the two net Sharpe
ratios are equal, found by bisection between zero and one hundred basis
points. Reported in basis points of notional, against the one basis point
base cost.

Failure reading: a break-even above ten basis points means the HMM only
wins where trading is impractically expensive.

### E12. Shared regime factor against the VIX

The research plan's Stage 5 is a multivariate model. The cheapest version:
T2 on a composite return, the equal-weight mean across the thirteen
futures of each day's return divided by its own trailing 252-day standard
deviation (lagged one day), yearly refit, twelve starts. E6 found the VIX
carries cross-asset regime information better than the S&P 500 regime.
Repeat the E6 regression with the composite regime probability in place
of the equity regime: own regime, plus composite regime, plus lagged log
VIX, plus both. QLIKE, Diebold-Mariano against the own-regime regression.

Failure reading: if the composite regime adds nothing beyond the VIX, a
multivariate regime model is not worth building for forecasting, and
Stage 5 becomes a descriptive exercise.

## Fixed choices

EWMA span 21, GARCH-t as in `vol_models.py`, HAR-GK as in
`vol_models.py`, twelve starts with the research seeds, yearly January
refit, QLIKE on the squared return as the headline loss, Newey-West
Diebold-Mariano, Holm within each comparison, one pooled test per
comparison on the cross-asset mean loss differential. Threshold one half
for the state label. Thresholds and spans are not moved after a result.

## What counts

Nothing here is traded and no holdout is opened. A pooled result below 5%
in the model's favour becomes a candidate for a dated gate entry in
`TRADING_GATES.md`; a candidate is not a result.
