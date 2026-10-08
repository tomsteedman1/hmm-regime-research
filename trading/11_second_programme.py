"""E7 to E12. The second exploratory programme, `RESEARCH_PROGRAMME_2.md`.

Every forecast for day t uses data to day t-1. The T2-on-returns signal in
``outputs/signals`` is read, never refitted. New fits (E9 realised-variance
regime model, E12 composite regime model) are walk-forward with the
research multi-start estimator. No holdout file is read.

    python3 11_second_programme.py            # everything
    python3 11_second_programme.py 7 10 11    # chosen experiments
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import backtest  # noqa: E402
import vol_models  # noqa: E402
from assets import FUTURES, load_range_variance, load_returns, load_vix  # noqa: E402
from experiments_common import COLOURS, PROJECT_DIR, TRADED, label, load_signals, scored_mask, write_json  # noqa: E402
from multi_start import fit_multiple_starts_general  # noqa: E402
from regime_hmm import RegimeHMM  # noqa: E402
from stats_tools import diebold_mariano, holm, qlike  # noqa: E402
from strategies import equal_risk_portfolio, vol_target  # noqa: E402
from vol_models import FIRST_YEAR, LAST_YEAR, GarchT, _garch_recursion, _t_loglik, yearly_walk_forward  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "outputs" / "second_programme"
SPAN = 21


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def dm_table(rows):
    dm = pd.DataFrame(rows)
    dm["p_holm"] = np.nan
    for _, i in dm.groupby("comparison").groups.items():
        dm.loc[i, "p_holm"] = holm(dm.loc[i, "p_value"].to_numpy())
    return dm


def pooled_test(diffs_by_comparison):
    """One Diebold-Mariano on the cross-asset mean daily loss differential."""
    rows = []
    for comparison, diffs in diffs_by_comparison.items():
        frame = pd.DataFrame(diffs)
        avg = frame.mean(axis=1, skipna=True).dropna()
        rows.append({"comparison": comparison, **diebold_mariano(avg.to_numpy(), np.zeros(len(avg))),
                     "first_wins_assets": int((frame.mean() < 0).sum()), "assets": int(frame.shape[1])})
    return pd.DataFrame(rows)


def counts_table(dm):
    out = []
    for comparison, g in dm.groupby("comparison"):
        out.append({"comparison": comparison, "assets": len(g),
                    "first_better": int((g["mean_diff"] < 0).sum()),
                    "first_better_p5": int(((g["mean_diff"] < 0) & (g["p_value"] < 0.05)).sum()),
                    "second_better_p5": int(((g["mean_diff"] > 0) & (g["p_value"] < 0.05)).sum()),
                    "holm_p5": int((g["p_holm"] < 0.05).sum())})
    return pd.DataFrame(out)


def stationary_distribution(A):
    values, vectors = np.linalg.eig(A.T)
    v = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return v / v.sum()


def reconstructed_models(key):
    """The January T2 model for each scored year, rebuilt from its saved parameters.

    The saved summary holds mu, sigma, nu and the stay probabilities; pi is
    set to the stationary distribution. Returns the models and the largest
    gap between the rebuilt one-step probability and the saved signal
    inside each scored year, which measures what the missing pi costs.
    """
    params = pd.read_csv(PROJECT_DIR / "outputs" / "signals" / f"{key}_hmm_params.csv").set_index("year")
    s = load_signals(key)
    values = s["return"].to_numpy(dtype=float)
    years = s.index.year
    models, gaps = {}, {}
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        p = params.loc[year]
        A = np.array([[p["stay_low"], 1 - p["stay_low"]], [1 - p["stay_high"], p["stay_high"]]])
        model = RegimeHMM(n_states=2, emission="student_t").set_parameters(
            stationary_distribution(A), A, np.array([p["mu_low"], p["mu_high"]]),
            np.array([p["sigma_low"], p["sigma_high"]]), np.array([p["nu_low"], p["nu_high"]]))
        models[year] = model
        through = years <= year
        p_high = model.predictive_proba(values[through])[:, 1]
        in_year = years[through] == year
        gaps[year] = float(np.nanmax(np.abs(p_high[in_year] - s["p_high"].to_numpy()[years == year])))
    return models, gaps


def valid_rows(s, columns):
    sc = scored_mask(s.index)
    proxy = s["r2"].to_numpy(dtype=float)
    valid = sc & np.isfinite(proxy) & (proxy > 0)
    for c in columns:
        valid &= np.isfinite(np.asarray(c, dtype=float))
    return valid


# ----------------------------------------------------------------------
# E7. Within-state volatility dynamics
# ----------------------------------------------------------------------

def state_weighted_ewma(returns, p_high, ewma_var, span=SPAN):
    """Two EWMA variances, each fed by the day's r^2 with weight = that day's
    one-step state probability; forecast = next day's probability mixture.

    The regime probability exists only from the first scored day, so both
    state variances start at that day's plain EWMA level (which uses data to
    the day before) and separate from there.
    """
    r2 = np.asarray(returns, dtype=float) ** 2
    p = np.asarray(p_high, dtype=float)
    base = np.asarray(ewma_var, dtype=float)
    lam = 1 - 2 / (span + 1)
    n = len(r2)
    v = np.full((n, 2), np.nan)
    state_var = None
    for t in range(n):
        if not np.isfinite(p[t]) or not np.isfinite(r2[t]):
            continue
        if state_var is None:
            if not np.isfinite(base[t]):
                continue
            state_var = np.array([base[t], base[t]])
        v[t] = state_var
        w = np.array([1 - p[t], p[t]])
        state_var = state_var + (1 - lam) * w * (r2[t] - state_var)
    forecast = np.full(n, np.nan)
    ok = np.isfinite(v[:, 0]) & np.isfinite(p)
    forecast[ok] = (1 - p[ok]) * v[ok, 0] + p[ok] * v[ok, 1]
    return forecast


class GarchX(GarchT):
    """GARCH(1,1)-t plus delta times an exogenous [0, 1] regressor in the variance."""

    @staticmethod
    def _unpack(theta):
        omega, alpha, beta, nu = GarchT._unpack(theta[:4])
        return omega, alpha, beta, nu, theta[4]

    def _recursion(self, x, z, omega, alpha, beta, delta, h0):
        h = _garch_recursion(x ** 2, omega, alpha, beta, h0) + delta * z
        # The exogenous term enters additively each day, not through the lag,
        # so the plain recursion plus the term is the model's variance.
        return h

    def _negloglik(self, theta, x, z):
        omega, alpha, beta, nu, delta = self._unpack(theta)
        h = self._recursion(x, z, omega, alpha, beta, delta, x.var())
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e10
        return -_t_loglik(x, h, nu)

    def fit(self, returns, z):
        r = np.asarray(returns, dtype=float)
        z = np.asarray(z, dtype=float)
        self.mu_ = r.mean()
        x = r - self.mu_
        var = x.var()
        base = np.array([np.log(var * 0.05), np.log(0.05 / 0.9), np.log(0.9 / 0.05), np.log(6.0), 0.0])
        best = None
        for s in (base, base + np.array([0.5, 0.3, -0.3, 0.5, var * 0.2]), base + np.array([-0.5, -0.3, 0.3, -0.5, -var * 0.1])):
            res = minimize(self._negloglik, s, args=(x, z), method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        self.omega_, self.alpha_, self.beta_, self.nu_, self.delta_ = self._unpack(best.x)
        self.loglik_ = -best.fun
        self.params_ = dict(mu=self.mu_, omega=self.omega_, alpha=self.alpha_, beta=self.beta_, nu=self.nu_, delta=self.delta_)
        return self

    def forecast(self, returns, z):
        x = np.asarray(returns, dtype=float) - self.mu_
        persistence = self.alpha_ + self.beta_
        h0 = self.omega_ / (1 - persistence) if persistence < 1 else x.var()
        return self._recursion(x, np.asarray(z, dtype=float), self.omega_, self.alpha_, self.beta_, self.delta_, h0)


def e7_one(key):
    s = load_signals(key)
    values = s["return"].to_numpy(dtype=float)
    years = s.index.year
    models, gaps = reconstructed_models(key)
    swe = state_weighted_ewma(values, s["p_high"].to_numpy(), s["ewma_var"].to_numpy())
    garchx = np.full(len(s), np.nan)
    params = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        through = years <= year
        train = years < year
        z_through = models[year].predictive_proba(values[through])[:, 1]
        z_train = z_through[train[through]]
        model = GarchX().fit(values[train], z_train)
        f = model.forecast(values[through], z_through)
        in_year = years[through] == year
        garchx[years == year] = f[in_year]
        params.append({"key": key, "year": year, **model.params_,
                       "delta_over_unconditional": model.delta_ / max(model.omega_ / max(1 - model.alpha_ - model.beta_, 1e-6), 1e-12)})
    return key, swe, garchx, pd.DataFrame(params), gaps


def e7():
    out = OUTPUT_DIR / "e7_within_state"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(min(len(TRADED), os.cpu_count() or 1)) as pool:
        results = pool.map(e7_one, TRADED)
    loss_rows, dm_rows, param_frames, gap_rows = [], [], [], []
    pooled = {}
    for key, swe, garchx, params, gaps in results:
        s = load_signals(key)
        param_frames.append(params)
        gap_rows.append({"key": key, **{f"gap_{y}": g for y, g in gaps.items()}, "max_gap": max(gaps.values())})
        fc = {"hmm": s["hmm_var"].to_numpy(), "garch": s["garch_var"].to_numpy(), "ewma": s["ewma_var"].to_numpy(),
              "state_ewma": swe, "garch_x": garchx}
        valid = valid_rows(s, fc.values())
        proxy = s["r2"].to_numpy()[valid]
        losses = {m: qlike(f[valid], proxy) for m, f in fc.items()}
        for m, l in losses.items():
            loss_rows.append({"key": key, "model": m, "mean_qlike": float(l.mean()), "days": int(valid.sum())})
        for first in ("state_ewma", "garch_x"):
            for second in ("garch", "hmm", "ewma"):
                comp = f"{first}_vs_{second}"
                dm_rows.append({"key": key, "comparison": comp, **diebold_mariano(losses[first], losses[second])})
                pooled.setdefault(comp, {})[key] = pd.Series(losses[first] - losses[second], index=s.index[valid])
    losses = pd.DataFrame(loss_rows)
    dm = dm_table(dm_rows)
    pooled = pooled_test(pooled)
    counts = counts_table(dm)
    params = pd.concat(param_frames)
    losses.to_csv(out / "mean_qlike.csv", index=False)
    dm.to_csv(out / "diebold_mariano.csv", index=False)
    pooled.to_csv(out / "pooled_tests.csv", index=False)
    counts.to_csv(out / "counts.csv", index=False)
    params.to_csv(out / "garch_x_params.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(out / "reconstruction_gaps.csv", index=False)
    write_json(out / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records")})
    print("\nE7 mean QLIKE (r2 proxy)")
    print(losses.pivot(index="key", columns="model", values="mean_qlike").round(4).to_string())
    print(pooled.round(4).to_string(index=False))
    print(counts.to_string(index=False))
    print("delta sign by asset-year: positive", int((params["delta"] > 0).sum()), "of", len(params))


# ----------------------------------------------------------------------
# E8. Forecast combination and encompassing
# ----------------------------------------------------------------------

def e8():
    out = OUTPUT_DIR / "e8_combination"
    out.mkdir(parents=True, exist_ok=True)
    base = ["ewma", "garch", "har"]
    loss_rows, dm_rows, enc_rows = [], [], []
    pooled = {}
    for key in TRADED:
        s = load_signals(key)
        cols = {m: s[f"{m}_var"].to_numpy(dtype=float) for m in base + ["hmm"]}
        valid = valid_rows(s, cols.values())
        proxy = s["r2"].to_numpy()[valid]
        F = {m: c[valid] for m, c in cols.items()}
        combos = {
            "mean3": np.mean([F[m] for m in base], axis=0),
            "mean4": np.mean([F[m] for m in base + ["hmm"]], axis=0),
            "geo3": np.exp(np.mean([np.log(F[m]) for m in base], axis=0)),
            "geo4": np.exp(np.mean([np.log(F[m]) for m in base + ["hmm"]], axis=0)),
            "garch": F["garch"], "hmm": F["hmm"],
        }
        losses = {m: qlike(f, proxy) for m, f in combos.items()}
        for m, l in losses.items():
            loss_rows.append({"key": key, "model": m, "mean_qlike": float(l.mean()), "days": int(valid.sum())})
        for comp, (a, b) in {"mean4_vs_mean3": ("mean4", "mean3"), "geo4_vs_geo3": ("geo4", "geo3"),
                             "mean3_vs_garch": ("mean3", "garch"), "geo4_vs_garch": ("geo4", "garch")}.items():
            dm_rows.append({"key": key, "comparison": comp, **diebold_mariano(losses[a], losses[b])})
            pooled.setdefault(comp, {})[key] = pd.Series(losses[a] - losses[b], index=s.index[valid])
        # Encompassing: r2 on garch and hmm forecasts jointly, yearly refit, out-of-sample weights.
        idx = s.index
        y = s["r2"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(s)), cols["garch"], cols["hmm"]])
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & (idx.year >= FIRST_YEAR)
        y_fit = np.where(ok, y, np.nan)
        betas = {}

        def step(train, through):
            keep = train & np.isfinite(y_fit)
            beta, *_ = np.linalg.lstsq(X[keep], y_fit[keep], rcond=None)
            betas[int(idx.year[through][-1])] = beta
            return X[through] @ beta
        fc = yearly_walk_forward(idx, FIRST_YEAR + 1, LAST_YEAR, step)
        for year, beta in betas.items():
            enc_rows.append({"key": key, "year": year, "intercept": beta[0], "w_garch": beta[1], "w_hmm": beta[2]})
        v2 = valid & np.isfinite(fc) & (fc > 0)
        lo = qlike(fc[v2], s["r2"].to_numpy()[v2])
        lg = qlike(cols["garch"][v2], s["r2"].to_numpy()[v2])
        dm_rows.append({"key": key, "comparison": "encompassing_vs_garch", **diebold_mariano(lo, lg)})
        pooled.setdefault("encompassing_vs_garch", {})[key] = pd.Series(lo - lg, index=idx[v2])
    losses = pd.DataFrame(loss_rows)
    dm = dm_table(dm_rows)
    pooled = pooled_test(pooled)
    counts = counts_table(dm)
    enc = pd.DataFrame(enc_rows)
    losses.to_csv(out / "mean_qlike.csv", index=False)
    dm.to_csv(out / "diebold_mariano.csv", index=False)
    pooled.to_csv(out / "pooled_tests.csv", index=False)
    counts.to_csv(out / "counts.csv", index=False)
    enc.to_csv(out / "encompassing_weights.csv", index=False)
    write_json(out / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records"),
                                      "median_weights": enc[["w_garch", "w_hmm"]].median().to_dict()})
    print("\nE8 mean QLIKE")
    print(losses.pivot(index="key", columns="model", values="mean_qlike").round(4).to_string())
    print(pooled.round(4).to_string(index=False))
    print(counts.to_string(index=False))
    print("encompassing weights, median across asset-years:", enc[["w_garch", "w_hmm"]].median().round(3).to_dict())
    print("share of asset-years with w_hmm > 0:", round(float((enc["w_hmm"] > 0).mean()), 3))


# ----------------------------------------------------------------------
# E9. Regime model on log realised variance
# ----------------------------------------------------------------------

def make_g2(init_strategy, random_state):
    return RegimeHMM(n_states=2, emission="gaussian", init_strategy=init_strategy, random_state=random_state)


def e9_one(key):
    started = time.time()
    s = load_signals(key)
    y = np.log(s["gk"].fillna(s["r2"]).where(lambda v: v > 0)).to_numpy(dtype=float)
    ok = np.isfinite(y)
    idx = s.index
    years = idx.year
    forecast = np.full(len(s), np.nan)
    p_high = np.full(len(s), np.nan)
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        train = (years < year) & ok
        through = (years <= year) & ok
        model, records, _ = fit_multiple_starts_general(y[train], make_g2, vol_models.N_STARTS, vol_models.BASE_SEED)
        w = model.predictive_proba(y[through])
        level = np.exp(model.mu + 0.5 * model.sigma ** 2)
        f = w @ level
        pos = np.where(through)[0]
        in_year = years[pos] == year
        forecast[pos[in_year]] = f[in_year]
        p_high[pos[in_year]] = w[in_year, int(np.argmax(model.mu))]
        gaps = np.array([r["log_likelihood_gap"] for r in records])
        rows.append({"key": key, "year": year, "mu_low": model.mu[0], "mu_high": model.mu[1],
                     "sigma_low": model.sigma[0], "sigma_high": model.sigma[1],
                     "stay_low": model.A[0, 0], "stay_high": model.A[1, 1],
                     "duration_low": 1 / (1 - model.A[0, 0]), "duration_high": 1 / (1 - model.A[1, 1]),
                     "starts_within_1": int((gaps <= 1.0).sum())})
    return key, forecast, p_high, pd.DataFrame(rows), time.time() - started


def e9():
    out = OUTPUT_DIR / "e9_rv_regime"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(min(len(TRADED), os.cpu_count() or 1)) as pool:
        results = pool.map(e9_one, TRADED)
    loss_rows, dm_rows, frames, dur_rows = [], [], [], []
    pooled = {}
    for key, forecast, p_high, params, seconds in results:
        print(f"  E9 {key:10s} {seconds:5.0f}s", flush=True)
        s = load_signals(key)
        frames.append(params)
        pd.DataFrame({"rv_hmm_var": forecast, "rv_p_high": p_high}, index=s.index).to_csv(out / f"{key}_rv_signals.csv")
        fc = {"rv_hmm": forecast, "har": s["har_var"].to_numpy(), "garch": s["garch_var"].to_numpy(),
              "hmm": s["hmm_var"].to_numpy(), "ewma": s["ewma_var"].to_numpy()}
        valid = valid_rows(s, fc.values())
        for proxy_name in ("r2", "gk"):
            proxy = s[proxy_name].to_numpy(dtype=float)
            v = valid & np.isfinite(proxy) & (proxy > 0)
            losses = {m: qlike(f[v], proxy[v]) for m, f in fc.items()}
            for m, l in losses.items():
                loss_rows.append({"key": key, "proxy": proxy_name, "model": m, "mean_qlike": float(l.mean()), "days": int(v.sum())})
            for rival in ("har", "garch", "hmm", "ewma"):
                comp = f"rv_hmm_vs_{rival}"
                dm_rows.append({"key": key, "proxy": proxy_name, "comparison": comp, **diebold_mariano(losses["rv_hmm"], losses[rival])})
                pooled.setdefault((proxy_name, comp), {})[key] = pd.Series(losses["rv_hmm"] - losses[rival], index=s.index[v])
        ret_params = pd.read_csv(PROJECT_DIR / "outputs" / "signals" / f"{key}_hmm_params.csv").set_index("year").loc[LAST_YEAR]
        last = params.set_index("year").loc[LAST_YEAR]
        sc = scored_mask(s.index)
        agree = float(np.nanmean((p_high[sc] > 0.5) == (s["p_high"].to_numpy()[sc] > 0.5)))
        dur_rows.append({"key": key, "rv_duration_low": last["duration_low"], "rv_duration_high": last["duration_high"],
                         "ret_duration_low": 1 / (1 - ret_params["stay_low"]), "ret_duration_high": 1 / (1 - ret_params["stay_high"]),
                         "label_agreement": agree, "rv_starts_within_1_min": int(params["starts_within_1"].min())})
    losses = pd.DataFrame(loss_rows)
    dm = pd.DataFrame(dm_rows)
    dm["p_holm"] = np.nan
    for _, i in dm.groupby(["proxy", "comparison"]).groups.items():
        dm.loc[i, "p_holm"] = holm(dm.loc[i, "p_value"].to_numpy())
    pooled_rows = []
    for (proxy_name, comp), diffs in pooled.items():
        frame = pd.DataFrame(diffs)
        avg = frame.mean(axis=1, skipna=True).dropna()
        pooled_rows.append({"proxy": proxy_name, "comparison": comp, **diebold_mariano(avg.to_numpy(), np.zeros(len(avg))),
                            "first_wins_assets": int((frame.mean() < 0).sum()), "assets": int(frame.shape[1])})
    pooled = pd.DataFrame(pooled_rows)
    counts = []
    for (proxy_name, comp), g in dm.groupby(["proxy", "comparison"]):
        counts.append({"proxy": proxy_name, "comparison": comp, "assets": len(g),
                       "first_better": int((g["mean_diff"] < 0).sum()),
                       "first_better_p5": int(((g["mean_diff"] < 0) & (g["p_value"] < 0.05)).sum()),
                       "second_better_p5": int(((g["mean_diff"] > 0) & (g["p_value"] < 0.05)).sum()),
                       "holm_p5": int((g["p_holm"] < 0.05).sum())})
    counts = pd.DataFrame(counts)
    durations = pd.DataFrame(dur_rows)
    losses.to_csv(out / "mean_qlike.csv", index=False)
    dm.to_csv(out / "diebold_mariano.csv", index=False)
    pooled.to_csv(out / "pooled_tests.csv", index=False)
    counts.to_csv(out / "counts.csv", index=False)
    pd.concat(frames).to_csv(out / "rv_hmm_params.csv", index=False)
    durations.to_csv(out / "durations.csv", index=False)
    write_json(out / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records")})
    print("\nE9 mean QLIKE, r2 proxy")
    print(losses[losses["proxy"] == "r2"].pivot(index="key", columns="model", values="mean_qlike").round(4).to_string())
    print(pooled.round(4).to_string(index=False))
    print(counts.to_string(index=False))
    print(durations.round(2).to_string(index=False))


# ----------------------------------------------------------------------
# E10. Duration dependence
# ----------------------------------------------------------------------

BUCKETS = [(1, 10), (11, 30), (31, 90), (91, 10 ** 6)]


def spells(labels):
    """(state, length) for each completed spell in a boolean label series; the
    last, censored spell is returned separately."""
    out = []
    start = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            out.append((bool(labels[start]), t - start))
            start = t
    return out[:-1], out[-1]


def exposure_rows(labels, key):
    """One row per day in state: (key, state, time_in_state, exit)."""
    rows = []
    completed, censored = spells(labels)
    for state, length in completed + [censored]:
        censored_flag = (state, length) == censored
        for d in range(1, length + 1):
            exit_ = (d == length) and not censored_flag
            rows.append((key, int(state), d, int(exit_)))
    return rows


def logistic_fit(X, y):
    """Plain logistic regression by Newton steps; returns coefficients and standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    H = (X * (p * (1 - p))[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def e10():
    from scipy.stats import norm
    out = OUTPUT_DIR / "e10_duration"
    out.mkdir(parents=True, exist_ok=True)
    all_rows, hazard_rows, asset_rows = [], [], []
    for key in TRADED:
        s = load_signals(key)
        sc = scored_mask(s.index)
        p = s["p_high"].to_numpy()[sc]
        labels = p > 0.5
        rows = exposure_rows(labels, key)
        all_rows += rows
        frame = pd.DataFrame(rows, columns=["key", "state", "d", "exit"])
        for state in (0, 1):
            g = frame[frame["state"] == state]
            mean_spell = g["exit"].sum() and len(g) / g["exit"].sum()
            const = 1 / mean_spell if mean_spell else np.nan
            for lo, hi in BUCKETS:
                b = g[(g["d"] >= lo) & (g["d"] <= hi)]
                hazard_rows.append({"key": key, "state": state, "bucket": f"{lo}-{hi if hi < 10**5 else 'inf'}",
                                    "days": len(b), "exits": int(b["exit"].sum()),
                                    "hazard": float(b["exit"].mean()) if len(b) else np.nan, "constant_hazard": const})
            X = np.column_stack([np.ones(len(g)), np.log(g["d"].to_numpy())])
            beta, se = logistic_fit(X, g["exit"].to_numpy(dtype=float))
            asset_rows.append({"key": key, "state": state, "spells": int(g["exit"].sum()), "mean_spell": mean_spell,
                               "slope_log_duration": beta[1], "se": se[1], "z": beta[1] / se[1],
                               "p_value": 2 * (1 - norm.cdf(abs(beta[1] / se[1])))})
    frame = pd.DataFrame(all_rows, columns=["key", "state", "d", "exit"])
    pooled_rows = []
    for state in (0, 1):
        g = frame[frame["state"] == state]
        dummies = pd.get_dummies(g["key"], drop_first=True).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), np.log(g["d"].to_numpy()), dummies])
        beta, se = logistic_fit(X, g["exit"].to_numpy(dtype=float))
        pooled_rows.append({"state": state, "days": len(g), "exits": int(g["exit"].sum()),
                            "slope_log_duration": beta[1], "se": se[1], "z": beta[1] / se[1],
                            "p_value": 2 * (1 - norm.cdf(abs(beta[1] / se[1])))})
    # Control: the label is a threshold on a filtered probability, not the
    # state, so even a true Markov chain produces short spells near the
    # boundary. Simulate from each asset's 2024 model, filter and label the
    # same way, and measure the same slope under the null.
    control_rows = []
    for key in TRADED:
        models, _ = reconstructed_models(key)
        model = models[LAST_YEAR]
        n_days = int(scored_mask(load_signals(key).index).sum())
        for draw in range(10):
            x, true_states = model.simulate(n_days, random_state=20260909 + draw)
            p = model.predictive_proba(x)[:, 1]
            for label_name, labels in (("filtered_label", p > 0.5), ("true_state", true_states == 1)):
                frame = pd.DataFrame(exposure_rows(labels, key), columns=["key", "state", "d", "exit"])
                for state in (0, 1):
                    g = frame[frame["state"] == state]
                    if g["exit"].sum() < 5:
                        continue
                    X = np.column_stack([np.ones(len(g)), np.log(g["d"].to_numpy())])
                    beta, se = logistic_fit(X, g["exit"].to_numpy(dtype=float))
                    control_rows.append({"key": key, "draw": draw, "label": label_name, "state": state,
                                         "slope_log_duration": beta[1]})
    control = pd.DataFrame(control_rows)
    control_summary = control.groupby(["label", "state"])["slope_log_duration"].describe()[["count", "mean", "std", "min", "max"]].reset_index()
    control.to_csv(out / "simulated_control.csv", index=False)
    control_summary.to_csv(out / "simulated_control_summary.csv", index=False)
    hazard = pd.DataFrame(hazard_rows)
    per_asset = pd.DataFrame(asset_rows)
    pooled = pd.DataFrame(pooled_rows)
    real_mean = per_asset.groupby("state")["slope_log_duration"].mean()
    sim_mean = control[control["label"] == "filtered_label"].groupby("state")["slope_log_duration"].mean()
    sim_sd = control[control["label"] == "filtered_label"].groupby(["state", "key"])["slope_log_duration"].mean().groupby("state").std()
    pooled["real_mean_asset_slope"] = real_mean.to_numpy()
    pooled["simulated_mean_slope_filtered_label"] = sim_mean.to_numpy()
    pooled["simulated_sd_across_assets"] = sim_sd.to_numpy()
    hazard.to_csv(out / "hazard_by_bucket.csv", index=False)
    per_asset.to_csv(out / "per_asset_logistic.csv", index=False)
    pooled.to_csv(out / "pooled_logistic.csv", index=False)
    pooled_hazard = hazard.groupby(["state", "bucket"]).apply(
        lambda g: pd.Series({"days": g["days"].sum(), "exits": g["exits"].sum(), "hazard": g["exits"].sum() / max(g["days"].sum(), 1)}),
        include_groups=False).reset_index()
    pooled_hazard.to_csv(out / "pooled_hazard.csv", index=False)
    write_json(out / "summary.json", {"pooled": pooled.to_dict("records"), "pooled_hazard": pooled_hazard.to_dict("records"),
                                      "simulated_control": control_summary.to_dict("records")})
    print("\nE10 simulated control (slope under a true Markov chain, same filter and label):")
    print(control_summary.round(4).to_string(index=False))
    print("\nE10 pooled hazard by time in state (0 calm, 1 volatile)")
    print(pooled_hazard.round(4).to_string(index=False))
    print(pooled.round(4).to_string(index=False))
    print("per-asset slopes negative:", int((per_asset["slope_log_duration"] < 0).sum()), "of", len(per_asset),
          "; negative at 5%:", int(((per_asset["slope_log_duration"] < 0) & (per_asset["p_value"] < 0.05)).sum()))


# ----------------------------------------------------------------------
# E11. Break-even transaction cost
# ----------------------------------------------------------------------

def sharpe_at_cost(gross, turnover, cost):
    return backtest.sharpe(gross - cost * turnover)


def break_even(gross_a, turn_a, gross_b, turn_b, lo=0.0, hi=1.0):
    f = lambda c: sharpe_at_cost(gross_a, turn_a, c) - sharpe_at_cost(gross_b, turn_b, c)
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return 0.0, flo, fhi  # HMM already ahead at zero cost
    if fhi < 0:
        return np.nan, flo, fhi  # never catches up inside 100 bp
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), flo, fhi


def e11():
    out = OUTPUT_DIR / "e11_break_even"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    store = {}
    for key in TRADED:
        s = load_signals(key)
        sc = scored_mask(s.index)
        r = s["return"].to_numpy(dtype=float)
        frames = {}
        for name in ("hmm", "ewma"):
            pos = np.where(sc, vol_target(s[f"{name}_var"].to_numpy()), np.nan)
            f = backtest.daily_pnl(r, pos, 0.0)
            f.index = s.index
            frames[name] = f[sc]
            store[(key, name)] = frames[name]
        be, at0, at100 = break_even(frames["hmm"]["gross"].to_numpy(), frames["hmm"]["turnover"].to_numpy(),
                                    frames["ewma"]["gross"].to_numpy(), frames["ewma"]["turnover"].to_numpy())
        rows.append({"key": key, "turnover_hmm": frames["hmm"]["turnover"].sum(), "turnover_ewma": frames["ewma"]["turnover"].sum(),
                     "turnover_ratio": frames["hmm"]["turnover"].sum() / frames["ewma"]["turnover"].sum(),
                     "sharpe_diff_at_0bp": at0, "sharpe_diff_at_100bp": at100, "break_even_bp": be * 100 if np.isfinite(be) else np.nan})
    table = pd.DataFrame(rows)
    pooled_rows = []
    for universe, keys in (("futures", FUTURES), ("all", TRADED)):
        grid = np.linspace(0, 1, 101)
        diffs = []
        for c in grid:
            nets = {name: equal_risk_portfolio({k: store[(k, name)]["gross"] - c * store[(k, name)]["turnover"] for k in keys}) for name in ("hmm", "ewma")}
            diffs.append(backtest.sharpe(nets["hmm"].to_numpy()) - backtest.sharpe(nets["ewma"].to_numpy()))
        diffs = np.array(diffs)
        cross = np.where((diffs[:-1] < 0) & (diffs[1:] >= 0))[0]
        be = float(grid[cross[0] + 1] * 100) if len(cross) else (0.0 if diffs[0] >= 0 else np.nan)
        pooled_rows.append({"universe": universe, "sharpe_diff_at_1bp": diffs[1], "sharpe_diff_at_10bp": diffs[10],
                            "sharpe_diff_at_100bp": diffs[-1], "break_even_bp": be,
                            "turnover_ratio": float(np.mean([table.set_index("key").loc[k, "turnover_ratio"] for k in keys]))})
        pd.DataFrame({"cost_bp": grid * 100, "sharpe_diff": diffs}).to_csv(out / f"pooled_curve_{universe}.csv", index=False)
    pooled = pd.DataFrame(pooled_rows)
    table.to_csv(out / "break_even_by_asset.csv", index=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    write_json(out / "summary.json", {"pooled": pooled.to_dict("records"), "median_turnover_ratio": float(table["turnover_ratio"].median()),
                                      "assets_break_even_below_10bp": int((table["break_even_bp"] < 10).sum())})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for universe in ("futures", "all"):
        curve = pd.read_csv(out / f"pooled_curve_{universe}.csv")
        ax.plot(curve["cost_bp"], curve["sharpe_diff"], label=f"pooled {universe}", color=COLOURS["hmm" if universe == "futures" else "other"])
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1, color=COLOURS["constant"], ls="--", lw=0.8, label="base cost 1 bp")
    ax.set_xlabel("Cost, basis points of notional per unit turnover")
    ax.set_ylabel("Sharpe: HMM target minus EWMA target")
    ax.set_title("E11. The HMM sizer's thrift against cost, 2015 to 2024")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_break_even.png", dpi=150)
    print("\nE11 break-even cost")
    print(table.round(3).to_string(index=False))
    print(pooled.round(3).to_string(index=False))


# ----------------------------------------------------------------------
# E12. Composite regime factor against the VIX
# ----------------------------------------------------------------------

def composite_return():
    parts = {}
    for key in FUTURES:
        r = load_returns(key)
        sd = r.rolling(252).std().shift(1)
        parts[key] = r / sd
    frame = pd.DataFrame(parts)
    comp = frame.mean(axis=1, skipna=True).where(frame.notna().sum(axis=1) >= 8)
    return comp.dropna().rename("composite")


def e12_fit(_=None):
    comp = composite_return()
    signal, params = vol_models.walk_forward_hmm(comp)
    signal["composite"] = comp
    return signal, params


def e12(signal=None, params=None):
    out = OUTPUT_DIR / "e12_composite"
    out.mkdir(parents=True, exist_ok=True)
    if signal is None:
        signal, params = e12_fit()
    signal.to_csv(out / "composite_signals.csv")
    params.to_csv(out / "composite_hmm_params.csv", index=False)
    vix = load_vix()
    from importlib import import_module
    log_var_regression = import_module("10_cross_asset_regimes").log_var_regression
    loss_rows, dm_rows = [], []
    pooled = {}
    corr_rows = []
    for key in TRADED:
        s = load_signals(key)
        idx = s.index
        proxy = s["gk"].fillna(s["r2"]).to_numpy()
        proxy = np.where(proxy > 0, proxy, np.nan)
        y = np.log(proxy)
        own = np.log(s["hmm_var"].to_numpy())
        cp = signal["p_high"].reindex(idx).to_numpy()
        lv = np.log(vix.reindex(idx).ffill().shift(1).to_numpy())
        ones = np.ones(len(idx))
        specs = {"own": np.column_stack([ones, own]),
                 "own_plus_composite": np.column_stack([ones, own, cp]),
                 "own_plus_vix": np.column_stack([ones, own, lv]),
                 "own_plus_both": np.column_stack([ones, own, cp, lv])}
        y_fit = np.where(idx.year >= FIRST_YEAR, y, np.nan)
        fc = log_var_regression(y_fit, specs, idx)
        sc = scored_mask(idx) & (idx.year >= FIRST_YEAR + 1)
        target = s["r2"].to_numpy()
        valid = sc & np.isfinite(target) & (target > 0)
        for spec in specs:
            valid &= np.isfinite(fc[spec])
        losses = {spec: qlike(fc[spec][valid], target[valid]) for spec in specs}
        for spec, l in losses.items():
            loss_rows.append({"key": key, "spec": spec, "mean_qlike": float(l.mean()), "days": int(valid.sum())})
        for spec in specs:
            if spec == "own":
                continue
            comp = f"{spec}_vs_own"
            dm_rows.append({"key": key, "comparison": comp, **diebold_mariano(losses[spec], losses["own"])})
            pooled.setdefault(comp, {})[key] = pd.Series(losses[spec] - losses["own"], index=idx[valid])
        comp2 = "own_plus_both_vs_own_plus_vix"
        dm_rows.append({"key": key, "comparison": comp2, **diebold_mariano(losses["own_plus_both"], losses["own_plus_vix"])})
        pooled.setdefault(comp2, {})[key] = pd.Series(losses["own_plus_both"] - losses["own_plus_vix"], index=idx[valid])
        sc2 = scored_mask(idx)
        corr_rows.append({"key": key, "corr_composite_regime": float(pd.Series(cp[sc2]).corr(s["p_high"][sc2].reset_index(drop=True))),
                          "corr_vix_regime": float(pd.Series(lv[sc2]).corr(s["p_high"][sc2].reset_index(drop=True)))})
    losses = pd.DataFrame(loss_rows)
    dm = dm_table(dm_rows)
    pooled = pooled_test(pooled)
    counts = counts_table(dm)
    corr = pd.DataFrame(corr_rows)
    sc = scored_mask(signal.index)
    corr_vix = float(np.corrcoef(signal["p_high"][sc], np.log(vix.reindex(signal.index).ffill())[sc])[0, 1])
    losses.to_csv(out / "mean_qlike.csv", index=False)
    dm.to_csv(out / "diebold_mariano.csv", index=False)
    pooled.to_csv(out / "pooled_tests.csv", index=False)
    counts.to_csv(out / "counts.csv", index=False)
    corr.to_csv(out / "regime_correlations.csv", index=False)
    write_json(out / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records"),
                                      "composite_p_high_vs_log_vix_corr": corr_vix})
    print("\nE12 composite regime")
    print(params.round(3).to_string(index=False))
    print("corr(composite p_high, log VIX) =", round(corr_vix, 3))
    print(losses.pivot(index="key", columns="spec", values="mean_qlike").round(4).to_string())
    print(pooled.round(4).to_string(index=False))
    print(counts.to_string(index=False))


# ----------------------------------------------------------------------

def main(argv):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = [int(a) for a in argv] or [7, 8, 9, 10, 11, 12]
    started = time.time()
    fits = {7: e7, 8: e8, 9: e9, 10: e10, 11: e11, 12: e12}
    for n in chosen:
        t0 = time.time()
        fits[n]()
        print(f"\n[E{n} done in {time.time() - t0:.0f}s]", flush=True)
    print(f"\nall done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
