"""One-day-ahead variance forecasters and the walk-forward HMM signal.

Every function follows one convention: element ``t`` of a forecast uses data
up to and including day ``t - 1`` only. Fitted models are refitted each
January on all data before that January; the forecast for a day inside the
year comes from the parameters fitted at its start and the observations to
the day before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

import research_link  # noqa: F401
from multi_start import fit_multiple_starts_general  # noqa: E402
from regime_hmm import RegimeHMM  # noqa: E402

N_STARTS = 12
BASE_SEED = 20260811
FIRST_YEAR = 2015
LAST_YEAR = 2024


# ----------------------------------------------------------------------
# EWMA
# ----------------------------------------------------------------------

def ewma_variance(returns, span=21):
    """Exponentially weighted mean of squared returns, lagged one day."""
    series = pd.Series(np.asarray(returns, dtype=float))
    variance = (series ** 2).ewm(span=span, adjust=False).mean()
    return variance.shift(1).to_numpy()


# ----------------------------------------------------------------------
# GARCH(1,1) with Student-t innovations
# ----------------------------------------------------------------------

def _garch_recursion(r2, omega, alpha, beta, h0):
    h = np.empty(len(r2))
    h[0] = h0
    for t in range(1, len(r2)):
        h[t] = omega + alpha * r2[t - 1] + beta * h[t - 1]
    return h


def _t_loglik(x, h, nu):
    z2 = x ** 2 / h
    return (
        gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2))
    ).sum()


class GarchT:
    """GARCH(1,1) with a constant mean and Student-t innovations.

    Parameters are estimated by maximum likelihood on a transformed scale so
    that ``omega > 0``, ``0 < alpha``, ``0 < beta`` and ``alpha + beta < 1``,
    and ``nu > 2``. Conditional variance ``h_t`` uses returns to ``t - 1``.
    """

    def __init__(self):
        self.params_ = None

    @staticmethod
    def _unpack(theta):
        log_omega, a, b, log_nu = theta
        # alpha and beta share a logistic budget that keeps alpha + beta < 1.
        ea, eb = np.exp(a), np.exp(b)
        alpha = ea / (1 + ea + eb)
        beta = eb / (1 + ea + eb)
        return np.exp(log_omega), alpha, beta, 2.0 + np.exp(log_nu)

    def _negloglik(self, theta, x):
        omega, alpha, beta, nu = self._unpack(theta)
        h = _garch_recursion(x ** 2, omega, alpha, beta, x.var())
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e10
        return -_t_loglik(x, h, nu)

    def fit(self, returns):
        r = np.asarray(returns, dtype=float)
        self.mu_ = r.mean()
        x = r - self.mu_
        var = x.var()
        start = np.array([np.log(var * 0.05), np.log(0.05 / 0.9), np.log(0.9 / 0.05), np.log(6.0)])
        best = None
        for s in (start, start + np.array([0.5, 0.3, -0.3, 0.5]), start + np.array([-0.5, -0.3, 0.3, -0.5])):
            res = minimize(self._negloglik, s, args=(x,), method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        self.omega_, self.alpha_, self.beta_, self.nu_ = self._unpack(best.x)
        self.loglik_ = -best.fun
        self.params_ = dict(mu=self.mu_, omega=self.omega_, alpha=self.alpha_, beta=self.beta_, nu=self.nu_)
        return self

    def forecast(self, returns):
        """Conditional variance for each day given returns to the previous day."""
        r = np.asarray(returns, dtype=float)
        x = r - self.mu_
        persistence = self.alpha_ + self.beta_
        h0 = self.omega_ / (1 - persistence) if persistence < 1 else x.var()
        return _garch_recursion(x ** 2, self.omega_, self.alpha_, self.beta_, h0)


# ----------------------------------------------------------------------
# HAR on a daily variance proxy
# ----------------------------------------------------------------------

def har_design(proxy):
    """Lagged daily, weekly and monthly averages of a daily variance proxy.

    Row ``t`` holds averages of ``proxy[t-1]``, ``proxy[t-5..t-1]`` and
    ``proxy[t-22..t-1]``. The first 22 rows are NaN.
    """
    series = pd.Series(np.asarray(proxy, dtype=float))
    daily = series.shift(1)
    weekly = series.rolling(5).mean().shift(1)
    monthly = series.rolling(22).mean().shift(1)
    return np.column_stack([np.ones(len(series)), daily, weekly, monthly])


class HAR:
    """Corsi's heterogeneous autoregression by least squares, in levels."""

    def fit(self, proxy):
        X = har_design(proxy)
        y = np.asarray(proxy, dtype=float)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.beta_, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        # A negative fitted variance is replaced by a tenth of the smallest
        # training value; the floor is fixed at fit time so no later value leaks.
        self.floor_ = float(np.nanmin(y[keep])) * 0.1
        return self

    def forecast(self, proxy):
        X = har_design(proxy)
        f = X @ self.beta_
        return np.where(np.isfinite(f), np.maximum(f, self.floor_), np.nan)


# ----------------------------------------------------------------------
# Walk-forward driver
# ----------------------------------------------------------------------

def yearly_walk_forward(index, first_year, last_year, fit_and_forecast):
    """Refit each January on prior data and forecast the year.

    ``fit_and_forecast(train_mask, through_mask)`` receives boolean masks over
    ``index`` and returns forecasts for every row of ``through_mask``; the
    rows inside the year are kept. Returns an array aligned with ``index``.
    """
    out = np.full(len(index), np.nan)
    years = index.year
    for year in range(first_year, last_year + 1):
        train = years < year
        through = years <= year
        forecast = fit_and_forecast(train, through)
        in_year = years[through] == year
        out[years == year] = np.asarray(forecast)[in_year]
    return out


def make_t2(init_strategy, random_state):
    return RegimeHMM(n_states=2, emission="student_t", init_strategy=init_strategy, random_state=random_state)


def fit_t2(history, n_starts=N_STARTS, base_seed=BASE_SEED):
    model, records, _ = fit_multiple_starts_general(history, make_t2, n_starts, base_seed)
    return model, records


def hmm_signals(model, values):
    """Predictive mixture mean, variance and high-state probability per day."""
    weights = model.predictive_proba(values)
    means = model.mu
    variances = model.stdev ** 2
    mean = weights @ means
    second = weights @ (variances + means ** 2)
    variance = np.maximum(second - mean ** 2, 1e-12)
    high = int(np.argmax(model.stdev))
    return mean, variance, weights[:, high]


def walk_forward_hmm(returns, first_year=FIRST_YEAR, last_year=LAST_YEAR, n_starts=N_STARTS):
    """Yearly refit T2 signal. Returns a frame and the per-year parameters."""
    values = returns.to_numpy(dtype=float)
    index = returns.index
    out = pd.DataFrame(np.nan, index=index, columns=["hmm_mean", "hmm_var", "p_high"])
    params = []
    for year in range(first_year, last_year + 1):
        train = index.year < year
        through = index.year <= year
        model, records = fit_t2(values[train], n_starts=n_starts)
        mean, var, p_high = hmm_signals(model, values[through])
        in_year = index.year[through] == year
        rows = index.year == year
        out.loc[rows, "hmm_mean"] = mean[in_year]
        out.loc[rows, "hmm_var"] = var[in_year]
        out.loc[rows, "p_high"] = p_high[in_year]
        gaps = np.array([r["log_likelihood_gap"] for r in records])
        agree = np.array([r["agreement_with_best"] for r in records])
        high = int(np.argmax(model.stdev))
        low = 1 - high
        params.append(
            {
                "year": year,
                "train_obs": int(train.sum()),
                "mu_low": model.mu[low], "mu_high": model.mu[high],
                "sigma_low": model.sigma[low], "sigma_high": model.sigma[high],
                "nu_low": model.nu[low], "nu_high": model.nu[high],
                "stdev_low": model.stdev[low], "stdev_high": model.stdev[high],
                "stay_low": model.A[low, low], "stay_high": model.A[high, high],
                "loglik": model.score(values[train]),
                "starts_within_1": int((gaps <= 1.0).sum()),
                "min_agreement_within_1": float(agree[gaps <= 1.0].min()),
            }
        )
    return out, pd.DataFrame(params)


def walk_forward_garch(returns, first_year=FIRST_YEAR, last_year=LAST_YEAR):
    values = returns.to_numpy(dtype=float)
    params = []

    def step(train, through):
        model = GarchT().fit(values[train])
        params.append({"year": int(returns.index.year[through][-1]), **model.params_})
        return model.forecast(values[through])

    return yearly_walk_forward(returns.index, first_year, last_year, step), pd.DataFrame(params)


def walk_forward_har(proxy, first_year=FIRST_YEAR, last_year=LAST_YEAR):
    values = proxy.to_numpy(dtype=float)

    def step(train, through):
        return HAR().fit(values[train]).forecast(values[through])

    return yearly_walk_forward(proxy.index, first_year, last_year, step)
