# Trading gates

Registered: 2 September 2026, before any backtest on real data. Every
threshold below was put to Tom as a choice with alternatives and confirmed.

These gates judge whether the selected two-state Student-t HMM (T2) adds
value to a volatility-targeting rule. They follow the same discipline as the
research `GATES.md`: thresholds are fixed before the result they judge is
seen, and a change needs a dated entry in the change log with the reason.

## Protocol

- Signal. Filtered probabilities only. Parameters are refitted every
  January on all data before that January, with the research multi-start
  estimator (one deterministic start, eleven random starts with fixed seeds,
  best likelihood kept). The first fit uses 2010 to 2014; the first scored
  year is 2015; the last is 2024.
- Expected volatility for day t is the standard deviation of the one-step
  predictive mixture, with state weights P(z_t | changes up to t-1) and the
  fitted state means and standard deviations.
- Position over day t = target volatility / expected volatility, capped at
  the leverage limit, decided at the close of day t-1.
- Target volatility: $1.00 per day per unit. Leverage limit: 3 units.
- P&L series and roll treatments as in `DATA_DECISION.md`: raw, roll day
  zeroed, roll day dropped. The roll day is the trading day after each CL
  expiry.
- Costs: $0.01 per barrel per unit of turnover (about one tick of
  bid-ask plus commission on a 1,000-barrel contract), and a second run at
  $0.02.

## Benchmarks

- B1: a constant one-unit long position.
- B2: the same volatility-target rule with expected volatility from a
  21-day exponentially weighted realised volatility of past changes, lagged
  one day, and the same target, cap and costs. This is the bar. It needs no
  model.

## Metrics

Sharpe ratio after costs (annualised, 252 days), annual return, annual
volatility, maximum drawdown, annual turnover, total cost, mean position.
The Sharpe difference between the HMM rule and B2 is tested with a
stationary block bootstrap (mean block 21 days, 2,000 draws, fixed seed)
that resamples both P&L series on the same days.

## Pass rules

- TG1. On 2015 to 2024 at $0.01 costs, the HMM rule's Sharpe ratio exceeds
  B2's, and the bootstrap one-sided p-value is below 0.05, under all three
  roll treatments.
- TG2. TG1 also holds at $0.02 costs.
- TG3. The HMM rule's realised daily volatility of net P&L on 2015 to 2024
  is within 20% of the $1.00 target under the raw treatment.
- TG4. Sealed holdout, one joint opening with the research Stage 6: on
  2025 to 2026 the sign of the HMM-minus-B2 Sharpe difference matches the
  2015 to 2024 sign, with parameters fitted through December 2024 and no
  refit. No threshold or choice changes after the opening.

A failed gate is recorded in the results document with the numbers. The
model is not retuned, the input is not changed, and the benchmark is not
weakened in response.

## Change log

- 2 September 2026: draft written before any backtest on real data.
- 2 September 2026: registered. Tom confirmed TG1 at p < 0.05 under all
  three roll treatments, costs of $0.01 and $0.02, a 20% volatility band
  with a 3-unit cap, and that the Master repo's "research only, no business
  ideas" rule covers this project as well.
