# Second programme results

Run date: 9 September 2026. Plan: `RESEARCH_PROGRAMME_2.md`, written and
committed before any experiment ran. Script: `11_second_programme.py`.
Outputs: `outputs/second_programme/`. Scored period 2015 to 2024,
walk-forward, one-step probabilities only. Every number is exploratory.
No holdout was opened. The T2-on-returns signal was read from
`outputs/signals`, never refitted.

## Verdict in one paragraph

Six more questions, six more nulls for the regime model, and three
things worth knowing. Letting the volatility level move within a state
(E7) or combining the HMM with standard forecasters (E8) does not beat
GARCH-t; a GARCH-X with the regime probability in the variance equation
fits the regime term with a positive sign in 177 of 180 asset-years and
still forecasts worse than plain GARCH out of sample. A regime model on
realised variance (E9) is a different model with the same defect: it
tracks the level and loses to HAR. The composite cross-asset regime
(E12) hurts each asset's own forecast and adds nothing beyond the VIX.
The three findings that matter: the falling exit hazard of regime spells
(E10) is almost entirely an artefact of labelling by a filtered
probability, so the Markov assumption is not contradicted; the HMM sizer
turns over about half as much as EWMA (E11) and overtakes it at a cost
of about 4 basis points on the futures universe, which is above the one
basis point base cost but not absurd; and an equal-weight average of
EWMA, GARCH-t and HAR beats GARCH-t alone (E8, p = 0.022), with or
without the HMM in the average. Nothing becomes a gate candidate.

## E7. Within-state volatility dynamics

`outputs/second_programme/e7_within_state/`. Mean QLIKE, squared-return
proxy, pooled Diebold-Mariano on the cross-asset mean differential
(positive means the first model loses):

| Comparison | Mean QLIKE difference | Statistic | p | Assets where first wins |
|---|---:|---:|---:|---:|
| State-weighted EWMA vs GARCH-t | +0.019 | 3.58 | < 0.001 | 2 of 18 |
| State-weighted EWMA vs EWMA | -0.007 | -1.98 | 0.047 | 14 of 18 |
| State-weighted EWMA vs HMM | -0.083 | -1.92 | 0.055 | 15 of 18 |
| GARCH-X vs GARCH-t | +0.010 | 2.52 | 0.012 | 1 of 18 |
| GARCH-X vs EWMA | -0.016 | -1.86 | 0.063 | 15 of 18 |
| GARCH-X vs HMM | -0.092 | -2.37 | 0.018 | 18 of 18 |

Both hybrids beat the plain HMM, as they should: they are the HMM with a
tracking variance inside each state. Neither beats GARCH-t. The
state-weighted EWMA is a small, nominally significant improvement on the
plain EWMA (2 of 18 after Holm), which says the regime weights are worth
a little as a way of pooling squared returns, and that the little is
what a GARCH recursion already extracts.

GARCH-X is the sharper lesson. In sample the regime term is positive in
177 of 180 asset-year fits, with a median size of 0.56 times the
unconditional variance: the fit says the high state raises variance. Out
of sample the same term makes every asset but one worse. The regime
probability is a smoothed function of the same squared returns the
recursion already uses, lagged by the state's persistence; it adds a
parameter and no information.

The January models were rebuilt from their saved parameters with pi set
to the stationary distribution; the rebuilt one-step probabilities match
the saved signal to 6e-14 in every scored year
(`reconstruction_gaps.csv`), so the reconstruction is exact.

## E8. Forecast combination and encompassing

`outputs/second_programme/e8_combination/`. Equal-weight averages of the
one-day variance forecasts:

| Comparison | Mean QLIKE difference | Statistic | p | Assets where first wins |
|---|---:|---:|---:|---:|
| Mean of 4 (with HMM) vs mean of 3 | -0.001 | -0.49 | 0.62 | 9 of 18 |
| Geometric mean of 4 vs of 3 | -0.003 | -0.71 | 0.48 | 11 of 18 |
| Mean of 3 (EWMA, GARCH-t, HAR) vs GARCH-t | -0.008 | -2.29 | 0.022 | 12 of 18 |
| Geometric mean of 4 vs GARCH-t | -0.005 | -0.99 | 0.32 | 12 of 18 |
| Encompassing regression (GARCH-t, HMM) vs GARCH-t | +0.066 | 3.65 | < 0.001 | 0 of 18 |

Adding the HMM to a combination changes nothing (p = 0.62 and 0.48).
The combination result that is real does not involve the HMM: the plain
average of EWMA, GARCH-t and HAR beats GARCH-t on the pooled test and
on 12 of 18 assets (5 after Holm). That is the standard forecast
combination result (Bates and Granger 1969) showing up in a volatility
setting.

The encompassing regression, a Mincer-Zarnowitz regression of the
squared return on the GARCH-t and HMM forecasts jointly, refitted each
January, gives the HMM a median out-of-sample weight of 0.24 against
0.64 for GARCH-t, positive in 79% of asset-years. The fitted weights
hurt: the regression forecast loses to plain GARCH-t on every asset.
Estimated combination weights on noisy squared returns are worse than
equal weights, which is also the standard result.

## E9. Regime model on realised variance

`outputs/second_programme/e9_rv_regime/`. Two-state Gaussian HMM on the
log Garman-Klass variance, yearly refit, twelve starts, predictive
mixture forecast. Pooled Diebold-Mariano:

| Proxy | Comparison | Mean QLIKE difference | Statistic | p | Assets where RV-HMM wins |
|---|---|---:|---:|---:|---:|
| Squared return | RV-HMM vs HAR-GK | +0.167 | 2.67 | 0.008 | 3 of 18 |
| Squared return | RV-HMM vs GARCH-t | +0.314 | 4.73 | < 0.001 | 0 of 18 |
| Squared return | RV-HMM vs return HMM | +0.212 | 6.69 | < 0.001 | 2 of 18 |
| Garman-Klass | RV-HMM vs HAR-GK | +0.146 | 4.47 | < 0.001 | 0 of 18 |
| Garman-Klass | RV-HMM vs GARCH-t | +0.003 | 0.09 | 0.93 | 10 of 18 |
| Garman-Klass | RV-HMM vs return HMM | -0.064 | -2.94 | 0.003 | 15 of 18 |

On the range proxy it is trained towards, the RV regime model is level
with GARCH-t and beats the return HMM on 15 of 18 assets (11 after
Holm). Against HAR on the same input it loses everywhere. On the
squared-return proxy it loses to everything, for the bias reason E2
found for HAR: the range variance runs below the squared close-to-close
return, so a model of the range under-forecasts the return proxy.

The two models do not find the same regimes. Real-time label agreement
between the RV regime and the return regime runs from 51% (gold) to 91%
(natural gas); the RV states are shorter (WTI calm 86 days against 122;
gold 14 against 173). Multi-start stability is worse on the log-variance
input: Brent, wheat, USO and euro have years in which fewer than eight of
twelve starts reach the best likelihood. A regime on the level of
volatility is a less stable object than a regime on returns.

## E10. Duration dependence

`outputs/second_programme/e10_duration/`. Real-time label: one-step
high-state probability above one half. Pooled exit hazard by time in
state:

| State | 1 to 10 days | 11 to 30 | 31 to 90 | Over 90 |
|---|---:|---:|---:|---:|
| Calm | 0.132 | 0.026 | 0.019 | 0.008 |
| Volatile | 0.114 | 0.041 | 0.025 | 0.009 |

The hazard falls by a factor of fifteen. Logistic regression of exit on
log time-in-state with asset fixed effects: slope -0.79 (calm) and -0.63
(volatile), both with z beyond 25; every one of 36 asset-state slopes is
negative, 35 at 5%. Taken at face value this rejects the geometric
duration of a Markov chain.

It should not be taken at face value. The label is a threshold on a
filtered probability, not the state. Near the threshold the probability
crosses and recrosses, so short spells pile up at durations of one to
ten days even when the chain behind them is Markov. Control: simulate
from each asset's 2024 model, filter and label the simulated series the
same way, fit the same regression (ten draws per asset):

| Series | Calm slope | Volatile slope |
|---|---:|---:|
| Real data, mean across assets | -0.79 | -0.64 |
| Simulated Markov chain, filtered label | -0.73 (sd across assets 0.06) | -0.60 (sd 0.12) |
| Simulated Markov chain, true state | +0.18 | +0.24 |

The simulated Markov chain with the same filter reproduces the falling
hazard almost exactly. The real slopes are about one standard deviation
(calm) and a third of one (volatile) more negative than the null. The
true simulated states show the flat hazard the model assumes (the small
positive slopes are a finite-spell estimation artefact). Conclusion: the
duration dependence in the labelled regimes is a property of the
labelling, not evidence against the Markov assumption. A semi-Markov
model or a third state is not indicated by this test.

## E11. Break-even transaction cost

`outputs/second_programme/e11_break_even/`. E3 vol-target positions,
HMM against EWMA, costs varied from zero to 100 basis points of notional
per unit of turnover. The base cost in E3 was one basis point.

| Universe | Sharpe difference at 1 bp | At 10 bp | At 100 bp | Break-even cost | HMM turnover / EWMA turnover |
|---|---:|---:|---:|---:|---:|
| 13 futures | -0.015 | +0.039 | +0.574 | 4 bp | 0.57 |
| All 18 | -0.067 | -0.016 | +0.493 | 13 bp | 0.60 |

Per asset the HMM sizer's turnover is half the EWMA sizer's (median
ratio 0.49); only the two equity series turn over more under the HMM
(ratio 1.1 to 1.2), because the equity regime switches often. Six assets
already favour the HMM at zero cost, ten reach break-even below ten
basis points, and four (natural gas, S&P 500, SPY, USO) never do within
100 basis points.

The equal-risk futures portfolio breaks even at about four basis points
and favours the HMM by 0.04 Sharpe at ten. This is the one place the
model's slowness is a merit. Four basis points per unit of turnover is
above the E3 base cost, and the gain at realistic costs is small and not
tested for significance here, but a slow sizer is the honest use of a
persistent regime signal if it is used at all.

## E12. Composite regime factor against the VIX

`outputs/second_programme/e12_composite/`. T2 fitted yearly on the
equal-weight mean of the thirteen futures' returns each standardised by
its own trailing 252-day standard deviation. The composite regime is
very persistent (stay probabilities 0.95 to 0.99 in both states, calm
durations over 100 days), the calm-state degrees of freedom run to the
200 bound from 2021, and the one-step high-state probability has a
correlation of 0.23 with the log VIX. E6 regression, QLIKE 2016 to
2024, pooled:

| Added to own regime | Mean QLIKE difference | Statistic | p | Assets improved |
|---|---:|---:|---:|---:|
| Composite regime | +0.050 | 2.27 | 0.023 | 8 of 18 |
| Lagged log VIX | -0.162 | -2.47 | 0.014 | 16 of 18 |
| Both | -0.082 | -1.18 | 0.24 | 12 of 18 |
| Both against VIX alone | +0.080 | 4.43 | < 0.001 | 4 of 18 |

The composite regime makes the pooled forecast worse (the euro and
copper regressions blow up when it is added), and adding it to the VIX
makes the VIX regression worse on 14 of 18 assets (9 after Holm). A
cross-asset regime factor built this way is dominated by a public index.
Stage 5 of the research plan, the multivariate model, is a descriptive
exercise on this evidence, not a forecasting one.

## What was learned

- The HMM's defect is structural, not fixable by hybridisation. Every
  way of letting the level move inside a state (E7) converges on what
  GARCH already does, and the regime term in a GARCH-X is a lagged
  transform of the same squared returns.
- The forecast combination literature holds in volatility: equal
  weights beat GARCH-t alone, and estimated weights lose (E8). The HMM
  is neither needed nor harmful in the average.
- A regime model on realised variance is a different, less stable
  object (E9): shorter states, weaker multi-start agreement, and it
  still loses to a linear model of the same input.
- Labelling by a filtered probability creates duration dependence out of
  a Markov chain (E10). Any duration test on regime labels needs a
  simulated control. The Markov assumption survives.
- The one merit of the regime sizer is thrift (E11): half the turnover
  of EWMA, break-even at about four basis points on futures. That is a
  fact about costs, not about forecasting.
- A cross-asset regime factor loses to the VIX (E12), as the equity
  regime did in E6.

## Limitations

- Every test is exploratory and unregistered; the pooled p-values are
  read across twelve programme experiments in total.
- E7's GARCH-X uses the regime probability from the model fitted each
  January over the training years, which is what the model would have
  shown at the time, but the January model itself was selected by
  likelihood on those years.
- E9's Gaussian emissions on log variance are a modelling choice made
  without a Student-t comparison.
- E10's control uses ten draws per asset from the 2024 model only.
- E11's break-even Sharpe differences are point estimates without a
  bootstrap; the futures portfolio's advantage at ten basis points is
  0.04 Sharpe and would not survive one.
- E12's composite uses only the thirteen futures and a trailing
  standard deviation for standardisation; other constructions were not
  tried, by design.

## Reproduction

```bash
cd trading
python3 11_second_programme.py            # about eight minutes on 24 cores
python3 11_second_programme.py 10 11      # the two that run in seconds
```
