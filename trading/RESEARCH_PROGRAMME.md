# Cross-asset research programme

Written: 2 September 2026, before any data beyond WTI was downloaded and
before any result in this programme was seen. This document is the plan.
The results documents record what happened, including every failure.

## Why widen the scope

The registered WTI test (`WALK_FORWARD_RESULT.md`) found that the two-state
Student-t regime signal does not size risk better than a 21-day
exponentially weighted realised volatility. One asset and one rule cannot
say whether that is a fact about the model or a fact about WTI. The
programme below asks the same family of questions across many assets, and
adds the questions the first test could not ask: whether the regime carries
a drift, whether it helps a directional rule, and whether the HMM is simply
a weak volatility forecaster next to standard alternatives.

The aim is to learn what regime models can and cannot do for trading, with
enough discipline that a positive result, if one appears, can later be put
through registered gates and the sealed holdout without having been tuned
to pass.

## Status of every result here

Everything in this programme is exploratory. It runs on the development
sample only (2010 to 2024, scored 2015 to 2024 walk-forward). It reports
p-values, but with 15 assets and six experiments a handful of nominal
"significant" results are expected by chance. Each experiment therefore
reports:

- the per-asset result and the count of assets below 5%, against the
  count expected under the null (about 0.75 of 15);
- a Holm-corrected view of the per-asset p-values;
- one pooled test on an equal-risk portfolio across assets, which is the
  only test read as evidence.

No holdout is opened. The 2025 to 2026 period for every asset is downloaded
into a separate sealed file, hashed, and never read by any script in this
programme. A result that looks worth trading gets a dated gate entry in
`TRADING_GATES.md` first, then one holdout run, jointly with the research
Stage 6 and the WTI TG4.

The selected model is not reselected. Every experiment uses the two-state
Student-t HMM (T2), fitted per asset with the research multi-start
estimator, refitted every January on data before that January, with
filtered and one-step-ahead probabilities only. Nothing is fitted on the
year it scores.

## Asset universe

Fifteen daily series from Yahoo Finance, 2010 to 2024, chosen before any
result was seen for coverage of asset classes rather than for any prior
about where the model works.

| Group | Ticker | Instrument | Notes |
|---|---|---|---|
| Energy | `CL=F` | WTI crude | Research frozen file, dollar changes |
| Energy | `BZ=F` | Brent crude | |
| Energy | `NG=F` | Henry Hub natural gas | Seasonal, large roll gaps |
| Metals | `GC=F` | Gold | |
| Metals | `SI=F` | Silver | |
| Metals | `HG=F` | Copper | |
| Grains | `ZC=F` | Corn | |
| Grains | `ZS=F` | Soybeans | |
| Grains | `ZW=F` | Wheat | |
| Equity | `ES=F` | S&P 500 e-mini | |
| Rates | `ZN=F` | 10-year Treasury note | |
| FX | `6E=F` | Euro | |
| FX | `6J=F` | Japanese yen | |
| Tradable check | `SPY`, `TLT`, `GLD`, `USO`, `UNG` | ETFs, adjusted close | Roll-free, dividend-adjusted, real costs |
| Covariate | `^VIX` | CBOE VIX | Never traded, used in E6 only |

The five ETFs are counted separately from the fifteen futures-style series.
They exist to check whether a futures result survives on a series a retail
account could actually hold, with the drag of ETF roll costs included.

## Input

Log returns in percent for every series with strictly positive prices. WTI
keeps the research decision: dollar changes from the frozen file, because
the price crossed zero in April 2020 and log returns are undefined there.
The HMM likelihood is equivariant to a change of units, and every trading
metric is scale-free, so the WTI series is comparable with the rest.

Yahoo's continuous futures are front-month splices. The WTI roll audit
found the spread contamination small and without drift, and all three roll
treatments agreed. No expiry calendar exists here for the other twelve
contracts, so their P&L series are raw splices and the ETF check is the
roll-free control. This is a recorded limitation.

## Experiments

### E1. Regime atlas

Does the WTI finding, two persistent volatility states with a heavy-tailed
high state and near-zero state means, hold elsewhere? For each asset: fit
T2 and its Gaussian counterpart G2 on the full development sample with
twelve starts; report state parameters, occupancy, expected durations, the
BIC preference, initialisation stability (share of starts within one
log-likelihood unit of the best, Viterbi agreement), and sample stability
(Viterbi agreement between the full fit and fits on 2010 to 2016 and 2017
to 2024). The research thresholds (10 of 12 starts, 99% and 90% agreement)
are reported against, not used to drop assets.

What to learn: which assets have regimes stable enough to build on, and
whether the "volatility not direction" conclusion is general.

### E2. Volatility forecasting contest

The HMM is a volatility model. Before asking whether it trades, ask whether
it forecasts. One-day-ahead variance forecasts from four models, each using
only past data:

- HMM: predictive mixture variance from T2, yearly refit.
- EWMA: 21-day span on squared returns, lagged one day, no fitting.
- GARCH(1,1) with Student-t innovations, maximum likelihood, yearly refit,
  own implementation in `vol_models.py`.
- HAR: Corsi's heterogeneous autoregression on the daily Garman-Klass
  range variance, with daily, weekly and monthly averages, yearly refit by
  least squares.

Losses: QLIKE and squared error against two realised proxies, the squared
return and the Garman-Klass variance. QLIKE is robust to noise in the proxy
(Patton 2011). Pairwise Diebold-Mariano tests with a Newey-West variance,
HMM against each rival, per asset and pooled across assets.

What to learn: where the HMM sits in the standard volatility toolkit, and
why the vol-target test was a draw.

### E3. Volatility targeting across assets

The registered WTI test, run on every asset with the same rule: target one
unit of daily volatility, cap three, costs of one and two basis points of
notional per unit of turnover (the percent-return analogue of $0.01 a
barrel), HMM expected volatility against EWMA. Sharpe difference by
stationary block bootstrap. Portfolio version: equal-risk average of the
per-asset net P&L series.

What to learn: whether the WTI null is general.

### E4. Regime drift

Does the calm state carry the risk premium? Two strategies against a
constant long: long one unit when the one-step-ahead probability of the
calm state exceeds one half, flat otherwise; and long scaled by that
probability. Benchmark beyond constant long: the volatility-managed
portfolio of Moreira and Muir (2017), which scales by the inverse of EWMA
variance and needs no regime model. Costs as E3. Portfolio pooled test.

What to learn: whether the regime label carries directional information
that a volatility scaler does not.

### E5. Time-series momentum with a regime overlay

Time-series momentum (Moskowitz, Ooi and Pedersen 2012): sign of the past
252-day return, scaled to constant volatility, held one day, refreshed
daily. Three variants: sized by EWMA volatility (the standard), sized by
HMM volatility, and sized by EWMA volatility but cut to zero when the
one-step probability of the high state exceeds one half (the regime
overlay, after Daniel and Moskowitz 2016 on momentum crashes). Per asset
and as an equal-risk portfolio. Costs as E3.

What to learn: whether a regime signal has more value as a filter on a
directional rule than as a sizer on its own.

### E6. Cross-asset regime information

Descriptive: co-occurrence of high-volatility states across assets, and
lead-lag correlation of one-step high-state probabilities. Test: for each
asset, a rolling regression of log realised variance on its own regime
probability, with and without the equity-market regime probability and the
lagged VIX; compare QLIKE out of sample by Diebold-Mariano.

What to learn: whether one market's regime tells you about another's, and
whether a public volatility index already carries that information.

## Rules for this programme

- Nothing is fitted on the year it scores. Every signal for day t uses
  data to day t-1.
- The T2 model, the multi-start estimator, the yearly refit and the
  probability type are fixed. No search over states, emissions, spans,
  thresholds or costs. Where a threshold such as one half appears it is
  stated here first and not moved.
- Every experiment writes its per-asset table and its pooled test to
  `outputs/`, and the results document records them whether or not they
  favour the model.
- No experiment reads a holdout file. `test_assets.py` fails if any script
  imports a holdout path.
- If a pooled result is positive at the 5% level after Holm correction
  across the six experiments, it becomes a candidate for a registered gate
  entry. It is not traded, and the holdout is not opened, on the strength
  of this programme alone.

## Reading list

- Hamilton (1989), regime switching in time series.
- Moskowitz, Ooi and Pedersen (2012), time series momentum.
- Moreira and Muir (2017), volatility-managed portfolios.
- Daniel and Moskowitz (2016), momentum crashes.
- Bollerslev (1986), GARCH; Corsi (2009), HAR-RV; Garman and Klass (1980),
  range estimators.
- Patton (2011), robust loss functions for volatility forecasts; Diebold
  and Mariano (1995), comparing forecasts.
- Politis and Romano (1994), stationary bootstrap; Harvey, Liu and Zhu
  (2016) and Bailey and Lopez de Prado (2014) on multiple testing and
  deflated Sharpe ratios.
