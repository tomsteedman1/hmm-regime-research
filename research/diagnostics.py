"""Adequacy diagnostics, duration checks and bootstrap uncertainty for HMMs.

All predictive diagnostics use filtered state probabilities only. The
probability integral transform (PIT) of observation ``x_t`` is

    PIT_t = sum_k P(z_t = k | x_1, ..., x_{t-1}) F_k(x_t)

where ``F_k`` is the emission CDF of state ``k``. If the model is adequate,
the PIT values are independent and uniform on (0, 1).
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest, norm


def pit_values(model, observations):
    """Return one-step-ahead PIT values using filtered probabilities."""
    observations = np.asarray(observations, dtype=float)
    predictive = model.predictive_proba(observations)
    cdf = model.emission_cdf(observations)
    return (predictive * cdf).sum(axis=1)


def normal_scores(pit):
    """Map PIT values to standard normal residuals."""
    clipped = np.clip(pit, 1e-10, 1 - 1e-10)
    return norm.ppf(clipped)


def autocorrelation(values, lags):
    """Return sample autocorrelations at lags 1 to ``lags``."""
    centred = values - values.mean()
    denominator = (centred**2).sum()
    return np.array(
        [
            (centred[lag:] * centred[:-lag]).sum() / denominator
            for lag in range(1, lags + 1)
        ]
    )


def ljung_box(values, lags=20):
    """Return the Ljung-Box statistic and p-value."""
    values = np.asarray(values, dtype=float)
    length = len(values)
    rho = autocorrelation(values, lags)
    statistic = (
        length
        * (length + 2)
        * (rho**2 / (length - np.arange(1, lags + 1))).sum()
    )
    return statistic, chi2.sf(statistic, lags)


def pit_diagnostics(model, observations, lags=20):
    """Return the gate G6 statistics as a dictionary."""
    pit = pit_values(model, observations)
    residual = normal_scores(pit)
    ks_statistic, ks_p = kstest(pit, "uniform")
    lb_statistic, lb_p = ljung_box(residual, lags)
    lb2_statistic, lb2_p = ljung_box(residual**2, lags)
    return {
        "pit": pit,
        "residual": residual,
        "ks_statistic": ks_statistic,
        "ks_p_value": ks_p,
        "ljung_box_statistic": lb_statistic,
        "ljung_box_p_value": lb_p,
        "ljung_box_squared_statistic": lb2_statistic,
        "ljung_box_squared_p_value": lb2_p,
        "residual_mean": residual.mean(),
        "residual_stdev": residual.std(),
        "residual_skewness": pd.Series(residual).skew(),
        "residual_excess_kurtosis": pd.Series(residual).kurt(),
    }


def episode_durations(states):
    """Return the length of every uninterrupted state episode."""
    states = np.asarray(states)
    change = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(states)]])
    return pd.DataFrame(
        {
            "state": states[starts],
            "start": starts,
            "duration": ends - starts,
        }
    )


def duration_table(states, transition):
    """Compare realised Viterbi episode durations with geometric expectations."""
    episodes = episode_durations(states)
    records = []
    for state in range(transition.shape[0]):
        durations = episodes.loc[episodes["state"] == state, "duration"]
        stay = transition[state, state]
        expected = 1.0 / (1.0 - stay)
        records.append(
            {
                "state": state,
                "episodes": len(durations),
                "realised_mean_days": durations.mean() if len(durations) else np.nan,
                "realised_median_days": durations.median() if len(durations) else np.nan,
                "realised_max_days": durations.max() if len(durations) else np.nan,
                "expected_geometric_mean_days": expected,
                "realised_over_expected": (
                    durations.mean() / expected if len(durations) else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def parametric_bootstrap(model, length, n_draws=200, seed=20260901, fit_kwargs=None):
    """Refit the model on data simulated from itself.

    Returns one row per draw with the refitted parameters, states ordered by
    emission scale. The standard deviation of each column is the bootstrap
    standard error.
    """
    fit_kwargs = dict(fit_kwargs or {})
    records = []
    for draw in range(n_draws):
        simulated, _ = model.simulate(length, random_state=seed + draw)
        refit = type(model)(
            n_states=model.n_states,
            emission=model.emission,
            markov=model.markov,
            **fit_kwargs,
        ).fit(simulated)
        record = {"draw": draw, "iterations": refit.n_iter_}
        for state in range(model.n_states):
            record[f"mu_{state}"] = refit.mu[state]
            record[f"sigma_{state}"] = refit.sigma[state]
            record[f"stdev_{state}"] = refit.stdev[state]
            record[f"self_transition_{state}"] = refit.A[state, state]
            if refit.nu is not None:
                record[f"nu_{state}"] = refit.nu[state]
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_summary(model, draws):
    """Summarise bootstrap draws against the point estimates."""
    records = []
    for state in range(model.n_states):
        estimates = {
            "mu": model.mu[state],
            "sigma": model.sigma[state],
            "stdev": model.stdev[state],
            "self_transition": model.A[state, state],
        }
        if model.nu is not None:
            estimates["nu"] = model.nu[state]
        for name, estimate in estimates.items():
            column = draws[f"{name}_{state}"]
            records.append(
                {
                    "state": state,
                    "parameter": name,
                    "estimate": estimate,
                    "bootstrap_mean": column.mean(),
                    "standard_error": column.std(ddof=1),
                    "ci_2_5": column.quantile(0.025),
                    "ci_97_5": column.quantile(0.975),
                }
            )
    return pd.DataFrame(records)
