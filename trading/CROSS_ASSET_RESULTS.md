# Cross-asset results

Run date: 2 September 2026. Plan: `RESEARCH_PROGRAMME.md`, written and
committed before any download. Scored period: 2015 to 2024, walk-forward,
yearly refit, filtered and one-step probabilities only. Every number here
is exploratory. No holdout was opened. Outputs in `outputs/`.

## Verdict in one paragraph

The two-state Student-t regime model is a real and stable description of
volatility regimes in most of the eighteen series, and it adds nothing
tradable on top of standard tools. As a one-day variance forecaster it is
the weakest of four (it loses to GARCH on the pooled test, p = 0.016, and
is roughly level with a 21-day EWMA). As a sizer it neither beats EWMA in
volatility targeting (pooled futures difference -0.02 Sharpe, p = 0.61)
nor in time-series momentum (-0.06, p = 0.82). The calm state carries no
real-time drift premium: on predicted-calm days the mean return is not
higher than on predicted-volatile days in any asset, even though the
in-sample smoothed states show a strong split in equities and crude. Other
markets' regimes do carry information about an asset's variance, but the
VIX carries the same information better. Nothing here becomes a candidate
for a registered gate.

## E1. Regime atlas

`05_regime_atlas.py`, `outputs/regime_atlas/regime_atlas.csv`. Full
sample 2010 to 2024, twelve starts. The WTI fit reproduces the research
selection exactly.

| Asset | High-state occupancy | Calm duration (days) | Volatile duration | Stdev calm | Stdev volatile | nu calm | nu volatile | Starts within 1 LL | Early-half agreement | Late-half agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTI crude | 28% | 119 | 54 | 1.10 | 2.39 | 14.5 | 4.2 | 12 | 98% | 100% |
| Brent | 42% | 75 | 51 | 1.35 | 3.11 | 6.8 | 4.3 | 12 | 86% | 92% |
| Henry Hub | 46% | 120 | 105 | 2.12 | 4.63 | 13.9 | 6.5 | 12 | 81% | 98% |
| Gold | 34% | 153 | 97 | 0.80 | 1.29 | 4.4 | 4.6 | 12 | 80% | 73% |
| Silver | 38% | 232 | 130 | 1.53 | 2.62 | 3.5 | 4.1 | 12 | 86% | 70% |
| Copper | 50% | 107 | 102 | 1.08 | 1.71 | 6.4 | 8.2 | 12 | 95% | 98% |
| Corn | 37% | 117 | 70 | 1.21 | 2.30 | 7.6 | 4.6 | 12 | 85% | 100% |
| Soybeans | 32% | 103 | 50 | 1.02 | 1.88 | 12.5 | 5.0 | 12 | 93% | 93% |
| Wheat | 31% | 124 | 56 | 1.57 | 2.69 | 12.6 | 7.7 | 12 | 57% | 94% |
| S&P 500 | 37% | 45 | 31 | 0.58 | 1.55 | 5.9 | 5.2 | 12 | 98% | 100% |
| 10-year note | 41% | 231 | 144 | 0.26 | 0.45 | 8.0 | 11.0 | 12 | 89% | 93% |
| Euro | 39% | 358 | 213 | 0.40 | 0.68 | 8.8 | 12.9 | 12 | 95% | 100% |
| Yen | 45% | 86 | 78 | 0.39 | 0.75 | 7.1 | 5.4 | 12 | 100% | 99% |
| SPY | 34% | 52 | 32 | 0.59 | 1.55 | 6.1 | 5.5 | 12 | 100% | 100% |
| TLT | 38% | 114 | 77 | 0.70 | 1.25 | 39.4 | 9.8 | 11 | 96% | 91% |
| GLD | 52% | 109 | 106 | 0.70 | 1.20 | 5.4 | 5.3 | 12 | 83% | 86% |
| USO | 37% | 84 | 54 | 1.43 | 3.04 | 10.7 | 5.0 | 12 | 70% | 68% |
| UNG | 41% | 164 | 130 | 1.98 | 3.99 | 18.5 | 11.2 | 12 | 83% | 98% |

Units: percent per day except WTI in dollars per barrel. Agreement is the
share of dates on which a fit to one half of the sample gives the same
state as the full-sample fit.

What the atlas says:

- The Student-t model beats its Gaussian counterpart on BIC in all
  eighteen series. In gold the Gaussian model finds a different solution
  from ten of twelve starts; the Student-t finds one solution from all
  twelve. The research finding that a Gaussian HMM builds an outlier state
  is general.
- Initialisation stability is not an issue anywhere: at least eleven of
  twelve starts land within one log-likelihood unit of the best, with
  Viterbi agreement of 97% or better.
- Sample stability is where assets differ. Eight series pass the research
  90% bar on both halves (WTI, copper, soybeans, S&P 500, euro, yen, SPY
  and TLT; the 10-year note misses by one point on the early half). The
  three metals series and USO fail on both halves: their state paths
  depend on the fitted period. Brent, natural gas, corn, wheat, GLD and
  UNG fail on the early half only.
- The volatile state is broad, not an outlier bucket: 28% to 52% of days,
  with expected durations from a month (equities) to a year (euro).
- In-sample smoothed state means split by sign in equities (S&P 500:
  +0.12% a day calm, -0.03% volatile) and in crude, but not in rates, FX,
  copper or the grains, and gold has the higher mean in its volatile
  state. This split is examined in real time in E4 below.

## E2. Volatility forecasting contest

`06_vol_forecast_contest.py`, `outputs/vol_forecast_contest/`. Mean QLIKE
against the squared-return proxy, 2015 to 2024:

| Asset | HMM | EWMA | GARCH-t | HAR-GK | HMM rank of 4 |
|---|---:|---:|---:|---:|---:|
| WTI crude | 2.374 | 1.685 | 1.722 | 1.724 | 4 |
| Brent | 1.780 | 1.608 | 1.595 | 1.606 | 4 |
| Henry Hub | 1.484 | 1.414 | 1.412 | 1.403 | 4 |
| Gold | 1.610 | 1.637 | 1.583 | 1.723 | 2 |
| Silver | 1.757 | 1.754 | 1.693 | 2.059 | 3 |
| Copper | 1.470 | 1.500 | 1.459 | 1.995 | 2 |
| Corn | 1.420 | 1.400 | 1.379 | 1.378 | 4 |
| Soybeans | 1.383 | 1.388 | 1.377 | 1.399 | 2 |
| Wheat | 1.323 | 1.267 | 1.247 | 1.303 | 4 |
| S&P 500 | 1.726 | 1.651 | 1.568 | 1.554 | 4 |
| 10-year note | 1.271 | 1.265 | 1.240 | 1.219 | 4 |
| Euro | 1.391 | 1.399 | 1.365 | 1.364 | 3 |
| Yen | 1.496 | 1.508 | 1.487 | 1.478 | 3 |
| SPY | 1.708 | 1.624 | 1.555 | 1.635 | 4 |
| TLT | 1.305 | 1.259 | 1.249 | 1.667 | 3 |
| GLD | 1.493 | 1.501 | 1.470 | 1.882 | 2 |
| USO | 1.588 | 1.392 | 1.387 | 1.633 | 3 |
| UNG | 1.284 | 1.235 | 1.231 | 1.660 | 3 |

Pooled Diebold-Mariano on the cross-asset average loss differential
(positive means the HMM loses more):

| Proxy | Rival | Mean QLIKE difference | Statistic | p-value | Assets where HMM wins |
|---|---|---:|---:|---:|---:|
| Squared return | EWMA | +0.076 | 1.71 | 0.086 | 6 of 18 |
| Squared return | GARCH-t | +0.102 | 2.41 | 0.016 | 0 of 18 |
| Squared return | HAR-GK | -0.046 | -1.13 | 0.257 | 8 of 18 |
| Garman-Klass | EWMA | +0.104 | 7.36 | < 0.001 | 0 of 18 |
| Garman-Klass | GARCH-t | +0.067 | 5.09 | < 0.001 | 0 of 18 |
| Garman-Klass | HAR-GK | +0.210 | 16.35 | < 0.001 | 0 of 18 |

Per-asset counts at 5%, QLIKE, squared-return proxy: GARCH beats the HMM
in 8 of 18 (3 after Holm), the HMM beats GARCH in none. The HAR row is
not a fair fight on the range proxy (HAR is trained on it) and its win on
the squared-return proxy is a bias effect: the range variance is smaller
than the squared close-to-close return on average, so HAR under-forecasts
that proxy on every asset.

Mincer-Zarnowitz regression of the squared return on the forecast:

| Model | Slope range | R-squared range | S&P 500 R-squared |
|---|---|---|---:|
| HMM | 0.65 to 1.52 | 0.00 to 0.06 | 0.03 |
| EWMA | 0.50 to 0.89 | 0.02 to 0.24 | 0.22 |
| GARCH-t | 0.63 to 1.03 | 0.01 to 0.31 | 0.28 |

This is the diagnosis. The HMM forecast is roughly unbiased (slopes near
one) but nearly uninformative day to day (R-squared near zero): two
persistent states produce a forecast that moves slowly between two
levels, so it explains the level of volatility over months and almost
none of the variation from one day to the next. GARCH and even EWMA track
the day-to-day changes. The WTI vol-target draw in `WALK_FORWARD_RESULT.md`
was not bad luck; the signal is a lagging, two-level version of what EWMA
already gives.

Per-year fitted parameters are saved in `outputs/signals/*_hmm_params.csv`.
On WTI the volatile-state degrees of freedom fall from about 7 to about 4
once 2020 enters the training sample, and the calm-state scale drops from
$1.13 to $0.89 in 2018. The signal drifts with the sample, as the
research sample-stability gate warned.

## E3. Volatility targeting across assets

`07_cross_asset_vol_target.py`, `outputs/cross_asset_vol_target/`. Sharpe
after base costs, 2015 to 2024:

| Asset | Constant long | EWMA target | GARCH target | HMM target | HMM minus EWMA | p |
|---|---:|---:|---:|---:|---:|---:|
| WTI crude | 0.05 | -0.05 | -0.07 | 0.07 | +0.12 | 0.23 |
| Brent | 0.06 | -0.04 | -0.05 | 0.02 | +0.06 | 0.25 |
| Henry Hub | 0.05 | 0.24 | 0.21 | 0.11 | -0.13 | 0.96 |
| Gold | 0.54 | 0.56 | 0.55 | 0.56 | 0.00 | 0.51 |
| Silver | 0.22 | 0.13 | 0.14 | 0.19 | +0.07 | 0.15 |
| Copper | 0.16 | 0.19 | 0.17 | 0.13 | -0.06 | 0.92 |
| Corn | 0.05 | 0.20 | 0.19 | 0.10 | -0.10 | 0.94 |
| Soybeans | -0.02 | 0.02 | 0.02 | 0.03 | +0.01 | 0.40 |
| Wheat | -0.02 | 0.04 | 0.06 | 0.03 | -0.02 | 0.64 |
| S&P 500 | 0.60 | 0.65 | 0.67 | 0.62 | -0.03 | 0.59 |
| 10-year note | -0.28 | -0.28 | -0.28 | -0.29 | -0.01 | 0.60 |
| Euro | -0.19 | -0.22 | -0.25 | -0.18 | +0.04 | 0.17 |
| Yen | -0.29 | -0.38 | -0.35 | -0.33 | +0.05 | 0.22 |
| SPY | 0.70 | 0.80 | 0.82 | 0.73 | -0.07 | 0.70 |
| TLT | -0.07 | -0.01 | -0.03 | -0.02 | -0.01 | 0.57 |
| GLD | 0.53 | 0.53 | 0.55 | 0.53 | -0.01 | 0.57 |
| USO | -0.19 | -0.09 | -0.12 | -0.18 | -0.09 | 0.76 |
| UNG | -0.50 | -0.37 | -0.39 | -0.50 | -0.14 | 0.98 |

Six of eighteen differences are positive; none is below 5% (0.9 expected
under the null). Pooled equal-risk portfolios:

| Universe | Cost | HMM | EWMA | GARCH | Constant | HMM minus EWMA | p | 95% interval |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 13 futures | base | 0.18 | 0.20 | 0.17 | 0.20 | -0.02 | 0.61 | -0.13 to +0.12 |
| 13 futures | double | 0.17 | 0.18 | 0.16 | 0.20 | -0.01 | 0.58 | -0.12 to +0.12 |
| 5 ETFs | base | 0.18 | 0.36 | 0.32 | -0.19 | -0.18 | 0.99 | -0.37 to -0.02 |
| All 18 | base | 0.19 | 0.26 | 0.22 | 0.07 | -0.07 | 0.85 | -0.18 to +0.06 |

The WTI null is general. On the roll-free ETF series the HMM sizer is
significantly worse than EWMA, and the WTI edge (+0.12, still the two
April 2020 days) is the largest positive entry in the table and still not
significant. Realised daily volatility is within 20% of target for every
rule on every asset except the HMM on WTI ($1.40, the TG3 failure).

## E4. Regime drift

`08_regime_drift.py`, `outputs/regime_drift/`. Mean daily return by
one-step predicted state, 2015 to 2024, Newey-West test of the
difference:

| Asset | Mean, predicted calm | Mean, predicted volatile | Share calm | t | p |
|---|---:|---:|---:|---:|---:|
| WTI crude | -0.003 | +0.023 | 60% | -0.37 | 0.71 |
| Brent | -0.011 | +0.028 | 45% | -0.39 | 0.70 |
| Henry Hub | +0.019 | +0.008 | 38% | 0.09 | 0.93 |
| Gold | +0.033 | +0.026 | 76% | 0.14 | 0.89 |
| Silver | +0.004 | +0.085 | 74% | -0.86 | 0.39 |
| Copper | 0.000 | +0.027 | 47% | -0.47 | 0.64 |
| Corn | +0.024 | -0.033 | 67% | 0.71 | 0.48 |
| Soybeans | +0.023 | -0.083 | 77% | 1.45 | 0.15 |
| Wheat | +0.038 | -0.035 | 44% | 0.92 | 0.36 |
| S&P 500 | +0.028 | +0.062 | 58% | -0.78 | 0.44 |
| 10-year note | -0.006 | -0.007 | 60% | 0.09 | 0.93 |
| Euro | -0.006 | -0.006 | 68% | 0.02 | 0.98 |
| Yen | -0.009 | -0.012 | 55% | 0.12 | 0.90 |
| SPY | +0.034 | +0.070 | 60% | -0.79 | 0.43 |
| TLT | +0.020 | -0.044 | 63% | 1.65 | 0.10 |
| GLD | +0.014 | +0.057 | 64% | -1.16 | 0.25 |
| USO | +0.036 | -0.079 | 42% | 1.14 | 0.25 |
| UNG | -0.076 | -0.123 | 44% | 0.39 | 0.70 |

No asset shows a significant drift difference, and the equity sign is
the reverse of the in-sample atlas: on days the model predicted volatile,
the S&P 500 returned more, not less. The smoothed state means in E1 use
the whole sample, including the return on the day being labelled; the
one-step probability does not. The apparent "calm state carries the
premium" pattern is look-ahead.

Strategy Sharpe, pooled, base costs:

| Universe | Constant | Regime binary | Regime scaled | Vol-managed (Moreira-Muir) | Binary minus constant, p | Vol-managed minus constant, p | Binary minus vol-managed, p |
|---|---:|---:|---:|---:|---|---|---|
| 13 futures | 0.16 | 0.21 | 0.27 | 0.21 | +0.05, 0.40 | +0.05, 0.38 | +0.01, 0.50 |
| 5 ETFs | 0.19 | 0.42 | 0.40 | 0.55 | +0.23, 0.18 | +0.36, 0.04 | -0.12, 0.80 |
| All 18 | 0.17 | 0.29 | 0.32 | 0.31 | +0.12, 0.29 | +0.14, 0.19 | -0.02, 0.58 |

The one nominally significant line is the published result, not the
model: the Moreira-Muir volatility-managed rule on the five ETFs beats
constant long by 0.36 Sharpe units (p = 0.04, one test of six universes
and costs, so not significant after Holm). The regime rules sit between
constant long and the volatility-managed rule and never beat the latter.
Per asset, one regime-versus-constant difference in eighteen is below 5%,
against 0.9 expected.

## E5. Time-series momentum with a regime overlay

`09_tsmom_regime.py`, `outputs/tsmom_regime/`. Pooled, base costs:

| Universe | TSMOM, EWMA sized | HMM sized | GARCH sized | EWMA sized, cut in volatile state | HMM minus EWMA, p | Cut minus EWMA, p |
|---|---:|---:|---:|---:|---|---|
| 13 futures | 0.21 | 0.14 | 0.22 | 0.21 | -0.06, 0.82 | +0.01, 0.52 |
| 5 ETFs | 0.18 | 0.19 | 0.21 | 0.07 | +0.01, 0.48 | -0.12, 0.74 |
| All 18 | 0.21 | 0.17 | 0.23 | 0.17 | -0.04, 0.78 | -0.04, 0.59 |

Sizing momentum by the HMM is worse than sizing it by EWMA on the futures
portfolio. Cutting the position in the predicted-volatile state removes
about a quarter of the portfolio's volatility and leaves its Sharpe
unchanged: the cut takes out good days and bad days in equal measure.
Per asset, 8 of 18 cut differences are positive, one below 5% (copper,
+0.34), against 0.9 expected. The momentum crash story of Daniel and
Moskowitz is about equity cross-sectional momentum after bear markets;
it does not transfer to a one-day-ahead volatility state on single
futures here.

## E6. Cross-asset regime information

`10_cross_asset_regimes.py`, `outputs/cross_asset_regimes/`.

Regimes co-move in clusters. Correlation of one-step high-state
probabilities: gold and silver 0.79, corn and soybeans 0.67, WTI and
Brent 0.48, WTI and S&P 500 0.47, 10-year note and TLT 0.76, euro and
Brent 0.57. The grains are near-independent of everything else. The S&P
500 regime leads the WTI regime by about four days (correlation 0.47 at
lag zero, 0.48 at lag four, 0.37 at lag minus ten), a small asymmetry.

Does another market's regime improve an asset's own variance forecast?
Rolling-refit log-variance regressions, QLIKE out of sample 2016 to 2024,
Diebold-Mariano against the own-regime regression:

| Added regressor | Assets improved | Improved at 5% | Worsened at 5% | Improved after Holm |
|---|---:|---:|---:|---:|
| S&P 500 regime probability | 12 of 17 | 3 | 2 | 1 (Brent) |
| Lagged log VIX | 16 of 18 | 10 | 0 | 2 (gold, GLD) |
| Both | 15 of 17 | 8 | 1 | 2 |

The equity regime helps crude and natural gas a little and hurts the euro
and corn. The VIX helps almost everywhere and never hurts. A public
volatility index carries the cross-asset information better than the
model's own equity regime does.

## What was learned

1. The regime model is a description, not a forecaster. It says which
   of two volatility levels a market is in, with lags of weeks, and it
   says so reliably in most assets. It does not say what tomorrow's
   variance will be any better than a 21-day EWMA, and worse than a
   GARCH(1,1) that costs four parameters. Mincer-Zarnowitz R-squared
   near zero is the single number that explains every trading null in
   this programme.
2. Smoothed state statistics are not tradable statistics. The in-sample
   split of equity returns by state (calm +30% a year, volatile -8%)
   vanishes in real time. Any regime study that reports state-conditional
   means from the smoothed path is reporting look-ahead.
3. The published results replicate where they should. Volatility
   management (Moreira and Muir) improves the ETF portfolio; time-series
   momentum earns a modest positive Sharpe on the futures portfolio;
   Student-t emissions beat Gaussian everywhere. The pipeline is not
   broken; the regime signal is just not adding to these.
4. Eighteen series and nine strategy comparisons make 162 per-asset
   tests at base costs. Five p-values fall below 5%, against 8.1
   expected under the null, and none survives Holm. Reading the
   per-asset tables without the count line would have "found" copper
   momentum and euro volatility targeting.
5. Where a regime signal might still matter is not in sizing but in
   places this programme did not reach: options, where the state
   duration maps to a horizon; and the multivariate model of research
   Stage 5, where a shared regime factor could be estimated rather than
   proxied by the VIX.

## Limitations

- Yahoo continuous futures are front-month splices with roll gaps (corn
  -27% and natural gas +38% in a day are rolls, not markets). The ETF
  control handles this for five series; the other twelve carry the
  contamination in both the signal and the P&L. Every conclusion above
  holds on the ETFs as well, so it does not appear to matter here.
- Costs are one and two hundredths of a percent of notional per unit of
  turnover, which is a futures-like assumption applied to ETFs too.
- Range-based variance is missing on 701 silver days and 202 gold days
  (zero recorded range). The HAR regressors fall back to the squared
  return on those days.
- The pooled equal-risk portfolios weight assets by their own daily
  volatility target, not by a covariance estimate. They are a test
  device, not a portfolio anyone would hold.
- The bootstrap treats each rule as fixed. Thresholds (one half, 252
  days, 21-day span, 3-unit cap) were fixed in the programme document
  and not searched, but they were chosen by convention, not registered
  against alternatives.
- The E6 regression baseline (a log-variance regression on the own HMM
  variance) is weaker than the raw HMM forecast on most assets. The
  tests there are relative, own-regression against own-plus-covariate,
  and remain valid as such.

## Reproduction

```bash
python3 03_download_assets.py     # refuses to overwrite frozen files
python3 04_fit_signals.py         # about 10 minutes on 24 cores
python3 05_regime_atlas.py        # about 8 minutes
python3 06_vol_forecast_contest.py
python3 07_cross_asset_vol_target.py
python3 08_regime_drift.py
python3 09_tsmom_regime.py
python3 10_cross_asset_regimes.py
```
