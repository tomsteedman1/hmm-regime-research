# WTI HMM Input Decision

Decision date: 11 August 2026

## Selected observation

The next WTI HMM will use the daily futures price change:

\[
x_t = P_t - P_{t-1}
\]

The unit is US dollars per barrel per trading day.

## Reason

The project describes WTI market conditions. WTI futures prices can cross
zero. Daily price changes remain defined across zero and preserve the direction
of each move.

The selected series includes 3,770 daily observations. On 20 April 2020, its
value is -$55.90 per barrel. The following observation is +$47.64 per barrel.
These signs match the direction of each price move.

## Comparison result

| Transformation | Observations | Missing | Stress occupancy |
|---|---:|---:|---:|
| Simple return | 3,770 | 1 | 2.15% |
| Valid log return | 3,768 | 3 | 13.83% |
| Price change | 3,770 | 1 | 4.77% |
| Arcsinh return, scale $10 | 3,770 | 1 | 1.49% |
| Arcsinh return, scale $20 | 3,770 | 1 | 1.86% |
| Arcsinh return, scale $40 | 3,770 | 1 | 1.78% |

Stress occupancy comes from the same diagnostic two-state Gaussian HMM. These
figures show sensitivity to the observation definition. They do not validate
any model.

## Rejected primary choices

Simple returns give the 21 April recovery a negative sign because the previous
price is negative. Log returns are undefined on 20 and 21 April. Arcsinh
returns remain valid, but their values and state paths depend on a scale choice.

## Known limits

Price-change volatility is measured in dollars. A one-dollar move can have a
different relative meaning when oil costs $20 and when it costs $100.

The current continuous futures series can contain contract-roll effects. The
calendar flag in the audit is a screen. It does not identify actual roll dates.

The Gaussian HMM assigns many isolated extreme days to its stress state. A
later stage will test Student-t emissions and state stability.

## Addendum, 8 October 2026: a scale-free candidate observation

Written before any HMM is fitted to it. The price change stays the
selected observation. A second observation is registered as a candidate,
to test the recorded limit that dollar changes are not scale-free:

\[
y_t = 100 \left[\operatorname{asinh}(P_t / 10) - \operatorname{asinh}(P_{t-1} / 10)\right]
\]

The scale is fixed at $10 before any fit and is not tuned. Reasons:

- It is defined on every day, including 20 and 21 April 2020, so the
  sample stays the same 3,770 observations.
- Above about $30 it is close to a log return in percent, so a $1 move at
  $20 and at $100 no longer count the same.
- $10 is the smallest scale in the 11 August comparison table, the one
  closest to a log return. The only result seen for it so far is the
  diagnostic Gaussian stress occupancy of 1.49%, which is not a gate result.

The factor of 100 only sets the units (percent-like), so the fitting
starts work on numbers of a usual size. It does not change any likelihood
comparison after the correction below.

A model on y and a model on x cannot be compared on raw likelihood. The
`GATES.md` change log of the same date registers the change-of-variables
correction that puts both in dollar-change units. Series name in
`series.py`: `wti_asinh10`.

## Reproduction

Run these commands from the project directory:

```bash
python 01_data_audit.py
python 01_return_comparison.py
python -m unittest -v test_return_comparison.py
```

Tables and charts are saved in `outputs/return_comparison`.
