# Walk-forward result

Run date: 2 September 2026. Script: `02_walk_forward.py`. Gates:
`TRADING_GATES.md`, registered the same day before this run. Scored period:
2015 to 2024, 2,513 trading days. Outputs in `outputs/walk_forward`.

## Verdict

Every registered gate fails. The HMM rule's Sharpe ratio is higher than
the realised-volatility benchmark's under all six treatment and cost
combinations, but never significantly, and the whole margin comes from two
days in April 2020. There is no evidence that the T2 regime signal improves
a volatility-targeting rule over a 21-day exponentially weighted realised
volatility that needs no model.

| Gate | Rule | Result |
|---|---|---|
| TG1 | HMM beats B2 on Sharpe, p < 0.05, all three roll treatments, $0.01 costs | Fail. p between 0.20 and 0.23 |
| TG2 | The same at $0.02 costs | Fail. p between 0.19 and 0.22 |
| TG3 | Realised daily volatility within 20% of the $1.00 target | Fail. $1.40 |
| TG4 | Holdout sign | Sealed, not opened |

## Numbers

Raw treatment, $0.01 costs. The other treatments and the $0.02 run move
every figure by a few hundredths and change no conclusion; see
`outputs/walk_forward/metrics.csv`.

| Strategy | Sharpe | Annual return | Daily vol. | Max drawdown | Annual turnover | Mean position |
|---|---:|---:|---:|---:|---:|---:|
| HMM vol target | 0.07 | $1.62 | $1.40 | -$80.9 | 4.7 | 0.73 |
| B2 realised vol target | -0.05 | -$0.84 | $1.17 | -$68.9 | 9.8 | 0.82 |
| B1 constant long | 0.05 | $1.78 | $2.16 | -$114.0 | 0.1 | 1.00 |

Sharpe difference, HMM minus B2, stationary block bootstrap (mean block
21 days, 2,000 draws):

| Treatment | Cost | Observed | p-value | 95% interval |
|---|---|---:|---:|---|
| Raw | $0.01 | +0.12 | 0.23 | -0.14 to +0.41 |
| Raw | $0.02 | +0.12 | 0.22 | -0.13 to +0.42 |
| Zeroed | $0.01 | +0.12 | 0.21 | -0.12 to +0.40 |
| Zeroed | $0.02 | +0.12 | 0.20 | -0.12 to +0.40 |
| Dropped | $0.01 | +0.12 | 0.20 | -0.12 to +0.43 |
| Dropped | $0.02 | +0.12 | 0.19 | -0.12 to +0.44 |

Sharpe by year (raw, $0.01):

| | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HMM | -0.96 | 0.86 | 0.53 | -0.63 | 0.84 | -0.26 | 1.31 | 0.15 | -0.38 | 0.16 |
| B2 | -1.04 | 0.66 | 0.47 | -0.63 | 0.89 | -1.49 | 1.33 | 0.53 | -0.20 | 0.02 |

The two rules are near-identical in nine of ten years. The gap is 2020.

## What drove the result

Post-hoc reading, not a gate. On 20 April 2020 the price change was
-$55.90 and on 21 April +$47.64. The HMM rule held 0.64 units into both
days: its expected volatility had risen but not far, because two persistent
states cannot re-scale overnight. It lost $36 on the first day and made $30
on the second. The realised-volatility rule cut from 0.45 to 0.06 units
after the first day and so missed the rebound. That one pair of days is the
whole Sharpe margin and the whole TG3 failure:

| Excluding 20 and 21 April 2020 | HMM | B2 |
|---|---:|---:|
| Daily volatility of net P&L | $1.04 | $1.06 |
| Sharpe | 0.13 | 0.08 |
| Sharpe difference, bootstrap p-value | +0.05, p = 0.33 | |

Without those two days both rules track the $1.00 target and the HMM's
edge is 0.05 Sharpe units with a p-value of 0.33. That is noise.

Two further observations:

- Both rules under-forecast volatility on average. The mean expected daily
  volatility was $1.45 (HMM) and $1.57 (B2) against a realised standard
  deviation of $2.16 for the period, which is itself dominated by April
  2020. On ordinary days the forecasts are close to right, which is why
  the ex-April-2020 volatility lands on target.
- The HMM rule trades half as much as B2 (annual turnover 4.7 against 9.8)
  because filtered state probabilities move slowly. Lower costs are the one
  practical advantage it shows. At $0.02 a barrel that is worth about
  $0.05 a year per unit, which does not change any conclusion.

## Limitations

- The underlying had no drift over 2015 to 2024 (constant-long Sharpe
  0.05), so a long-only volatility target of any kind had little to work
  with. The test says the HMM does not size risk better than EWMA. It does
  not test the signal as a filter on a directional rule, which was not
  the registered question.
- A single yearly refit was registered. Per-year parameter estimates were
  not saved by this run; a later run should write them so that signal
  drift can be inspected.
- The P&L series carries a small roll-spread component on 4.8% of days
  (see `DATA_DECISION.md`). The three treatments agree, so it did not
  matter here.
- The bootstrap treats the strategy as fixed. It does not account for the
  choice of target, cap, or span, which were registered without search.

## Reproduction

```bash
python3 02_walk_forward.py
```

About four minutes. Output figures reproduce exactly from the fixed seeds.
