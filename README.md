# Hidden Markov regimes in WTI crude and eighteen other markets

[![tests](https://github.com/tomsteedman1/hmm-regime-research/actions/workflows/tests.yml/badge.svg)](https://github.com/tomsteedman1/hmm-regime-research/actions/workflows/tests.yml)
![Python 3.12 to 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
[![MIT licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

![WTI crude 2010 to 2024 with the high-volatility regime of the selected two-state Student-t HMM shaded](research/outputs/headline/wti_regimes.png)

## At a glance

| Question | Answer | Evidence |
|---|---|---|
| Does a hidden Markov model find stable volatility regimes in WTI crude? | Yes. Of six candidates, only the two-state Student-t model passed every gate. The gates were fixed before any fit. | [Model comparison](research/MODEL_COMPARISON_RESULT.md) |
| Does its regime signal size a WTI position better than 21-day realised volatility? | No. The Sharpe ratio was 0.12 higher (p = 0.23), and the whole margin came from 20 and 21 April 2020. | [Walk-forward test](trading/WALK_FORWARD_RESULT.md) |
| Does it forecast variance across eighteen futures and ETFs? | Less well than GARCH(1,1)-t on pooled QLIKE loss (p = 0.016). | [Cross-asset report](https://tomsteedman1.github.io/hmm-regime-research/trading/CROSS_ASSET_REPORT.html) |
| Does it improve any trading rule across those markets? | No pooled test favours it. Five of 162 per-asset tests fall below 5%, where chance alone gives 8.1. | [Cross-asset results](trading/CROSS_ASSET_RESULTS.md) |

The HMM is written from scratch in numpy and scipy: scaled forward-backward,
Baum-Welch with a Student-t EM step, and Viterbi decoding. Its likelihood and
state paths match `hmmlearn` in a cross-check test. Every pass and fail
threshold was written down before the result it judges, and the 2025 to 2026
data stays sealed. The trading results are mostly negative, and they are
reported in full.

## Start here

1. Run `python demo.py`. It needs no market data and takes a few seconds. It
   simulates two volatility regimes, fits the model, and shows how closely
   the fit recovers the true parameters and hidden states.
2. Read the [model comparison](research/MODEL_COMPARISON_RESULT.md). It
   shows why the two-state Student-t model was selected and where it
   still falls short.
3. Read the [walk-forward trading test](trading/WALK_FORWARD_RESULT.md). It
   is the one registered test of whether the regimes are worth money.
4. Open the [cross-asset report](https://tomsteedman1.github.io/hmm-regime-research/trading/CROSS_ASSET_REPORT.html). It is one page that covers
   eighteen markets and six questions.

## The two projects

1. **Research** (`research/`). Can a hidden Markov model find stable
   volatility regimes in daily WTI crude oil futures prices, 2010 to 2024?
2. **Trading** (`trading/`). Can the selected model's real-time regime
   signal improve a trading rule, after costs, against simple rules that
   need no model? The same question is then asked across eighteen futures
   and ETFs.

## How the work was done

Pass and fail thresholds were written down before the results they judge
were seen. A threshold can change only through a dated entry in a change
log, made before the affected result. The rules are in
[`research/GATES.md`](research/GATES.md) and
[`trading/TRADING_GATES.md`](trading/TRADING_GATES.md).

The price data was frozen and hashed before any model was fitted. The
2025 to 2026 data is a sealed holdout. No script reads it, and a test
fails if a trading script names a holdout file. It will be opened once,
for the research and trading tests together.

Every result document records the checks that failed as well as the
checks that passed.

## Results

### Research: model selection on WTI

The observation is the daily dollar price change. Log returns are
undefined on 20 April 2020, when the front-month price fell below zero
([`research/DATA_DECISION.md`](research/DATA_DECISION.md)).

Six candidates were tested: two, three and four states, each with
Gaussian or Student-t emissions. Only the two-state Student-t model (T2)
passed every gate.

| State | Implied daily st. dev. | Self-transition | Expected duration | Share of days |
|---|---:|---:|---:|---:|
| Low volatility | $1.09 | 99.16% | 119 days | 72.1% |
| High volatility | $2.39 | 98.15% | 54 days | 27.9% |

The two-state Gaussian model gave its high state to a few extreme days
around April 2020. The Student-t model takes those days into its fat
tails and finds a broad, persistent volatile regime. Its state means are
close to zero, so it finds volatility states, not trend states.

![State paths of the six candidate models on WTI, 2010 to 2024](research/outputs/model_comparison/01_state_paths.png)

Three later checks, each registered before it was run, left T2 in place.
Without the 180 futures roll days T2 keeps 96.2% Viterbi agreement with
its full fit. A scale-free candidate, the same model on asinh returns,
fails the sample-stability gate and has a BIC 119.7 higher once scored
in dollar units
([`research/SCALE_FREE_RESULT.md`](research/SCALE_FREE_RESULT.md)). The
from-scratch likelihood and decoding match `hmmlearn` in a cross-check
test.

Known limits: volatility clustering stays in the residuals, a dollar move
means different things at $20 and at $100 oil, and the low-state degrees
of freedom have a wide interval (8 to 39). Details in
[`research/MODEL_COMPARISON_RESULT.md`](research/MODEL_COMPARISON_RESULT.md).

### Trading: one registered test on WTI

A volatility-targeting rule sized by the HMM was tested against a rule
sized by 21-day realised volatility, 2015 to 2024. Every gate failed. The
HMM rule's Sharpe ratio was 0.12 higher (bootstrap p = 0.23), and the
whole margin came from 20 and 21 April 2020. Details in
[`trading/WALK_FORWARD_RESULT.md`](trading/WALK_FORWARD_RESULT.md).

### Trading: two exploratory programmes, eighteen series

These are labelled exploratory. Each plan was committed before any data
was downloaded or any experiment was run. The model was never refitted
or reselected to suit a trading result.

- Variance forecasting: GARCH(1,1)-t beats the HMM on pooled QLIKE loss
  (p = 0.016). The HMM forecast is unbiased, but its Mincer-Zarnowitz
  R-squared is close to zero.
- Volatility targeting, regime timing and momentum filters: no pooled
  test favours the model. Five of 162 per-asset tests fall below 5%,
  against 8.1 expected by chance.
- The VIX carries cross-asset regime information better than any regime
  built from the HMMs.
- An equal-weight average of EWMA, GARCH-t and HAR forecasts beats
  GARCH-t alone (p = 0.022), with or without the HMM.
- The HMM sizer trades about half as much as EWMA. It breaks even at about
  4 basis points of cost on the futures portfolio.
- Regime spells seem to get less likely to end as they get older. A
  simulated Markov chain labelled the same way shows the same pattern, so
  the Markov assumption stands.

Details in [`trading/CROSS_ASSET_RESULTS.md`](trading/CROSS_ASSET_RESULTS.md)
and [`trading/SECOND_PROGRAMME_RESULTS.md`](trading/SECOND_PROGRAMME_RESULTS.md).

## Code

- `research/regime_hmm.py`: K-state HMM with Gaussian or Student-t
  emissions. Scaled forward-backward, Baum-Welch, Viterbi, filtered
  and smoothed probabilities.
- `research/gaussian_hmm.py`: the first two-state Gaussian model, kept as
  the reference.
- `research/diagnostics.py`: probability integral transform, bootstrap,
  state durations.
- `trading/vol_models.py`: EWMA, GARCH(1,1)-t, HAR and the walk-forward
  HMM signal.
- `trading/backtest.py`, `trading/strategies.py`: positions, costs,
  metrics, block bootstrap.
- `trading/stats_tools.py`: QLIKE, Diebold-Mariano with Newey-West
  errors, Holm correction.
- `demo.py`: fits the model to simulated data and checks recovery.

The numbered scripts in each folder run the stages in order. Each folder
has its own README with the stage list.

## Running it

The results were produced with Python 3.14. The tests run on Python 3.12,
3.13 and 3.14 on every push. Install the packages and run the demo:

```bash
pip install -r requirements.txt
python demo.py
```

To rebuild the results, download the data:

```bash
python download_data.py
```

The price files are not in this repository. `download_data.py` gets the
2010 to 2024 samples from Yahoo Finance and checks their values against
`data_hashes.csv`. Yahoo revises history without notice. On 8 October
2026, 17 of 20 files matched. Gold futures, SPY and TLT have been revised
since the data was frozen, so results rebuilt from them can differ a
little from the result documents. The script reports each mismatch.

Then run the numbered scripts and tests:

```bash
cd research && python 01_data_audit.py      # and so on, see research/README.md
python -m pytest
cd ../trading && python 01_roll_audit.py   # and so on, see trading/README.md
python -m pytest
```

Tests that need the price files skip when the files are missing. The
`hmmlearn` cross-check test skips if `hmmlearn` is not installed. Install it
with `pip install hmmlearn` where a wheel exists for your Python version.

## A note on the documents

The result and gate documents are kept as written on their dates. Some
of them name files that are not in this public copy, such as task lists
and personal study notes. The two projects were first kept in separate
private repositories and were combined here with a new history.

## Licence

MIT. See [`LICENSE`](LICENSE).
