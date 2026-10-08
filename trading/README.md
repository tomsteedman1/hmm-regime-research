# HMM Regime Trading

Exploratory companion to the research project in `../research`.

## Questions

1. Registered (2 September 2026): can the filtered regime signal from the
   selected two-state Student-t HMM on WTI improve a volatility-targeting
   rule, after costs, against a simple realised-volatility rule that needs
   no model? Answer: no (`WALK_FORWARD_RESULT.md`).
2. Exploratory cross-asset programme (same day): does the same model
   describe, forecast, size or filter anything across eighteen series in
   energy, metals, grains, equities, rates and FX? Plan in
   `RESEARCH_PROGRAMME.md`, answers in `CROSS_ASSET_RESULTS.md`. Short
   version: it describes regimes well in most assets, forecasts variance
   worse than GARCH, and adds nothing to volatility targeting, regime
   timing or momentum that EWMA or the VIX does not already give.
3. Second exploratory programme (9 September 2026): six follow-up
   questions on within-state dynamics, forecast combination, a regime model
   on realised variance, regime durations, break-even costs and a composite
   cross-asset regime. Plan in `RESEARCH_PROGRAMME_2.md`, answers in
   `SECOND_PROGRAMME_RESULTS.md`. All six are null for the regime model.

## Scope guard

- The model is fixed: the two-state Student-t HMM (T2) selected under the
  research project's pre-registered gates. This project never reselects the
  model, changes its input, or tunes it to a trading result.
- Research code is imported from `../research` through `research_link.py`.
  Nothing is copied. `test_research_link.py` fails if the frozen WTI data
  differs from the data the results were produced on.
- Real time only. Every signal uses filtered probabilities and parameters
  fitted on data available at the time.
- The 2025 to 2026 holdout stays sealed until one joint opening with the
  research Stage 6 evaluation. Trading gates are registered before that.

## Results so far

The registered WTI test was run once on 2 September 2026 and failed every
gate. The HMM rule edges the realised-volatility rule by 0.12 Sharpe units
with a bootstrap p-value near 0.2, and the entire margin comes from 20 and
21 April 2020. Details in `WALK_FORWARD_RESULT.md`.

The cross-asset programme ran the same day on 2015 to 2024. No pooled test
favours the model; five of 162 per-asset strategy tests fall below 5%
against 8.1 expected by chance. The diagnosis is a Mincer-Zarnowitz
R-squared near zero: the regime forecast is unbiased but almost
uninformative day to day. Details in `CROSS_ASSET_RESULTS.md`.

The second programme ran on 9 September 2026 on the same sample. Three
results stand out. The falling exit hazard of regime spells also appears
in a simulated Markov chain labelled the same way, so the Markov
assumption stands. The HMM sizer trades about half as much as EWMA and
breaks even at about 4 basis points of cost on the futures portfolio. An
equal-weight average of EWMA, GARCH-t and HAR beats GARCH-t alone
(p = 0.022), with or without the HMM. Details in
`SECOND_PROGRAMME_RESULTS.md`.

## Files

| File | Purpose |
|---|---|
| `research_link.py` | Path to the research code, WTI data hash |
| `test_research_link.py` | Guards the frozen WTI data |
| `01_roll_audit.py` | Measures contract-roll contamination in the spliced series |
| `DATA_DECISION.md` | P&L series and roll treatment, decided on the audit |
| `DATA_MANIFEST.md` | Data used and the sealed holdout |
| `TRADING_GATES.md` | Benchmark, costs, metrics and pass thresholds, registered before the first run |
| `backtest.py` | P&L engine: positions, costs, metrics, block bootstrap |
| `02_walk_forward.py` | Walk-forward regime signal and volatility-target backtest |
| `test_backtest.py` | No-look-ahead, synthetic-regime and cost tests |
| `WALK_FORWARD_RESULT.md` | Result of the registered test: every gate failed |
| `RESEARCH_PROGRAMME.md` | Cross-asset plan, six experiments, written before any download |
| `03_download_assets.py` | Freezes eighteen series plus VIX and WTI OHLC, seals their holdouts |
| `assets.py` | Universe, frozen-file loaders, returns and range variance |
| `vol_models.py` | EWMA, GARCH(1,1)-t, HAR, walk-forward T2 signal with per-year parameters |
| `strategies.py` | Vol target, regime long, Moreira-Muir, time-series momentum, equal-risk portfolio |
| `stats_tools.py` | QLIKE, Diebold-Mariano with Newey-West, Holm |
| `04_fit_signals.py` | Caches walk-forward forecasts for every series (`outputs/signals`) |
| `05_regime_atlas.py` to `10_cross_asset_regimes.py` | Experiments E1 to E6 |
| `test_cross_asset.py` | No-look-ahead, parameter recovery, holdout guard |
| `CROSS_ASSET_RESULTS.md` | Results of E1 to E6 with every count and pooled test |
| `RESEARCH_PROGRAMME_2.md` | Second programme plan, six experiments, written before any run |
| `11_second_programme.py` | Experiments E7 to E12 |
| `SECOND_PROGRAMME_RESULTS.md` | Results of E7 to E12 |
| `render_cross_asset_report.py`, `CROSS_ASSET_REPORT.html` | Report page built from the outputs |

## Running

Download the data from the repository root first (see the root
`README.md`). `03_download_assets.py` is the original freezing script. It
is kept for the record, and `download_data.py` replaces it.

```bash
cd trading
python3 01_roll_audit.py
python3 02_walk_forward.py
python3 04_fit_signals.py          # cross-asset caches, about 10 minutes
python3 05_regime_atlas.py
python3 06_vol_forecast_contest.py
python3 07_cross_asset_vol_target.py
python3 08_regime_drift.py
python3 09_tsmom_regime.py
python3 10_cross_asset_regimes.py
python3 11_second_programme.py     # about eight minutes on 24 cores
python3 -m pytest
```
