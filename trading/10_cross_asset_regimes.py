"""E6. Cross-asset regime information.

Descriptive: co-occurrence and correlation of one-step high-state
probabilities across assets, and their lead-lag against the S&P 500.
Test: does the equity regime, or yesterday's VIX, improve each asset's own
one-day variance forecast? Rolling-refit log-variance regressions, QLIKE
out of sample, Diebold-Mariano against the own-regime model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assets import load_vix
from experiments_common import COLOURS, PROJECT_DIR, TRADED, label, load_signals, scored_mask, write_json
from stats_tools import diebold_mariano, holm, qlike
from vol_models import yearly_walk_forward

OUTPUT_DIR = PROJECT_DIR / "outputs" / "cross_asset_regimes"


def log_var_regression(y, X_by_spec, index):
    """Yearly-refit OLS of log variance; returns exp forecasts per spec."""
    out = {}
    for spec, X in X_by_spec.items():
        def step(train, through, X=X):
            keep = train & np.isfinite(y) & np.isfinite(X).all(axis=1)
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            resid = y[keep] - X[keep] @ beta
            smear = np.mean(np.exp(resid))  # Duan smearing for the log retransformation
            f = X[through] @ beta
            return np.exp(f) * smear
        out[spec] = yearly_walk_forward(index, 2015, 2024, step)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p_high = {}
    frames = {}
    for key in TRADED:
        s = load_signals(key)
        frames[key] = s
        p_high[key] = s["p_high"]
    P = pd.DataFrame(p_high)
    P = P[scored_mask(P.index)]
    corr = P.corr()
    corr.to_csv(OUTPUT_DIR / "p_high_correlation.csv")
    high = (P > 0.5)
    co = pd.DataFrame(index=P.columns, columns=P.columns, dtype=float)
    for a in P.columns:
        for b in P.columns:
            both = high[a] & high[b]
            either = high[a] | high[b]
            co.loc[a, b] = both.sum() / max(either.sum(), 1)
    co.to_csv(OUTPUT_DIR / "high_state_jaccard.csv")
    lead_lag = {}
    for key in TRADED:
        if key == "sp500":
            continue
        lead_lag[key] = {lag: P["sp500"].shift(lag).corr(P[key]) for lag in range(-10, 11)}
    lead_lag = pd.DataFrame(lead_lag).T
    lead_lag.to_csv(OUTPUT_DIR / "lead_lag_vs_sp500.csv")

    vix = load_vix()
    dm_rows, loss_rows = [], []
    for key in TRADED:
        s = frames[key]
        idx = s.index
        proxy = s["gk"].fillna(s["r2"]).to_numpy()
        proxy = np.where(proxy > 0, proxy, np.nan)
        y = np.log(proxy)
        own = np.log(s["hmm_var"].to_numpy())
        eq = P["sp500"].reindex(idx).to_numpy() if key != "sp500" else np.full(len(idx), np.nan)
        lv = np.log(vix.reindex(idx).ffill().shift(1).to_numpy())
        ones = np.ones(len(idx))
        specs = {"own": np.column_stack([ones, own]),
                 "own_plus_vix": np.column_stack([ones, own, lv])}
        if key != "sp500":
            specs["own_plus_equity_regime"] = np.column_stack([ones, own, eq])
            specs["own_plus_both"] = np.column_stack([ones, own, eq, lv])
        # Regression inputs are only defined where the HMM signal exists, so
        # the fit uses 2015 onwards data as it accrues (first fit on 2015 to
        # forecast 2016; 2015 is skipped in the scoring).
        y_fit = np.where(idx.year >= 2015, y, np.nan)
        fc = log_var_regression(y_fit, specs, idx)
        sc = scored_mask(idx) & (idx.year >= 2016)
        target = s["r2"].to_numpy()
        valid = sc & np.isfinite(target) & (target > 0)
        for spec in specs:
            valid &= np.isfinite(fc[spec])
        losses = {spec: qlike(fc[spec][valid], target[valid]) for spec in specs}
        losses["raw_hmm"] = qlike(s["hmm_var"].to_numpy()[valid], target[valid])
        for spec, l in losses.items():
            loss_rows.append({"key": key, "spec": spec, "mean_qlike": float(l.mean()), "days": int(valid.sum())})
        for spec in specs:
            if spec == "own":
                continue
            dm = diebold_mariano(losses[spec], losses["own"])
            dm_rows.append({"key": key, "comparison": f"{spec}_vs_own", **dm})
    dm = pd.DataFrame(dm_rows)
    dm["p_holm"] = np.nan
    for _, i in dm.groupby("comparison").groups.items():
        dm.loc[i, "p_holm"] = holm(dm.loc[i, "p_value"].to_numpy())
    losses = pd.DataFrame(loss_rows)
    dm.to_csv(OUTPUT_DIR / "diebold_mariano.csv", index=False)
    losses.to_csv(OUTPUT_DIR / "mean_qlike.csv", index=False)
    counts = dm.groupby("comparison").apply(lambda g: pd.Series({
        "assets": len(g), "improves": int((g["mean_diff"] < 0).sum()),
        "improves_p5": int(((g["mean_diff"] < 0) & (g["p_value"] < 0.05)).sum()),
        "worsens_p5": int(((g["mean_diff"] > 0) & (g["p_value"] < 0.05)).sum()),
        "holm_p5": int((g["p_holm"] < 0.05).sum())}), include_groups=False).reset_index()
    counts.to_csv(OUTPUT_DIR / "counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im = axes[0].imshow(corr.to_numpy(), vmin=-0.2, vmax=1, cmap="Blues")
    axes[0].set_xticks(range(len(corr)))
    axes[0].set_xticklabels([label(k) for k in corr.columns], rotation=90)
    axes[0].set_yticks(range(len(corr)))
    axes[0].set_yticklabels([label(k) for k in corr.index])
    axes[0].set_title("Correlation of one-step high-state probabilities, 2015 to 2024")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    for key in ("wti", "brent", "gold", "tnote", "euro", "copper"):
        axes[1].plot(lead_lag.columns, lead_lag.loc[key], label=label(key))
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Lag of S&P 500 regime (positive: S&P leads)")
    axes[1].set_ylabel("Correlation with asset's high-state probability")
    axes[1].set_title("Lead-lag against the equity regime")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_cross_asset.png", dpi=150)

    write_json(OUTPUT_DIR / "summary.json", {"counts": counts.to_dict("records")})
    print(corr.round(2).to_string())
    print(losses.pivot(index="key", columns="spec", values="mean_qlike").round(4).to_string())
    print(dm.round(3).to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
