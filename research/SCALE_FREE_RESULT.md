# Scale-free candidate result

Run date: 8 October 2026. Registered before any fit: `DATA_DECISION.md`
addendum and `GATES.md` change log, both of 8 October 2026. Scripts:
`06_model_comparison.py --series wti_asinh10`, then
`08_scale_free_comparison.py`. Outputs: `outputs/scale_free`.

## Verdict

T2 stays the selected model. The two-state Student-t model on asinh
returns (A2) fails gate G4. It is also worse than T2 on both registered
criteria once both are scored in dollar-change units. Its corrected BIC is
119.7 higher, and its corrected held-out log-likelihood is 51.3 lower.

The asinh transform does what it was chosen for away from zero. It fails
at the one place it was supposed to handle: the days when the price
crossed zero.

## Numbers

Both models are scored on the same 3,770 dollar changes. A2's likelihoods
carry the change-of-variables correction registered in `GATES.md`.

| Candidate | Observation | Parameters | Dollar BIC | Dollar held-out log-lik. 2019-2024 | G3 | G4 | Roll-day record | Smallest state share | Viterbi agreement with T2 |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| T2 | Price change | 9 | 13,281.1 | -2,911.6 | Pass | Pass | Pass | 27.9% | 100% |
| A2 | Asinh return, $10 | 9 | 13,400.8 | -2,963.0 | Pass | **Fail** | Pass | 35.9% | 79.1% |
| A2G | Asinh return, $10 | 7 | 13,983.2 | -4,379.8 | Pass | Fail | Pass | 1.5% | 73.1% |

A2's G4 subsamples:

| Subsample | Viterbi agreement | Top-state occupancy shift | High-state st. dev. |
|---|---:|---:|---:|
| 2010 to 2016 | 95.2% | +4.8 points | 2.94 |
| 2017 to 2024 | **77.4%** | **-22.6 points** | **11.99** |
| Without 20 and 21 April 2020 | 99.1% | +0.9 points | 3.44 |

On the full sample, A2's states have standard deviations of 1.49 and 3.76
(percent-like units), self-transitions of 0.989 and 0.980, and degrees of
freedom of 11.5 and 3.5.

## Why it fails

At $10 the transform is close to linear around zero, with slope 1/10. A
dollar move near zero therefore becomes a very large asinh move. On 20
April 2020 the observation is -339.9, and on 21 April it is +291.8. The
sample standard deviation is 7.7, and the largest move outside April and
May 2020 is 27.2. In dollars those two days were large. In asinh units
they are about 40 standard deviations each.

On the late half alone, the model spends its high state on the spring 2020
crash, with a standard deviation of 12. It is a different model from the
full-sample fit, so G4 fails. Without the two cross-zero days, A2 agrees
with its full-sample path on 99.1% of dates. The instability comes from
those two days.

## What it does show

Away from spring 2020 the transform removes most of the level effect.
These are raw sample standard deviations, not model estimates:

| Sample | Asinh return | Dollar change |
|---|---:|---:|
| 2010 to 2016 | 2.09 | $1.44 |
| 2017 to 2024 without March to June 2020 | 2.15 | $1.62 |

The relative size of daily moves was about the same in both halves. Part
of the dollar series' change between halves comes from the price level.
The limitation recorded for T2 is real. This candidate does not fix it,
because of the cross-zero days.

## Limitations of this test

- One scale ($10) was registered, and it was not tuned. A larger scale
  shrinks the April 2020 days but moves the transform back towards dollar
  changes. Choosing a scale after seeing this result would be tuning.
- The volatility clustering in T2's residuals is also present in A2's
  (Ljung-Box on squares, p < 0.001). The new observation does not help
  with it.

## Open, not registered

A log return in percent on the sample without 20 and 21 April 2020 is
scale-free and has no near-zero problem. It changes the sample the gates
were set on, so it would need its own dated entry before any fit. It is
listed in `TDL.md` as an option, not a plan.

## Reproduction

```bash
python 06_model_comparison.py --series wti_asinh10    # about 3 minutes
python 08_scale_free_comparison.py
```
