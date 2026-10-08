"""E4. Does the calm state carry the risk premium?

Rules against a constant one-unit long: long when the one-step calm
probability exceeds one half; long scaled by that probability; and the
Moreira-Muir volatility-managed rule that needs no regime model. Also a
direct test of the return conditional on the predicted state.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_common import (COLOURS, COSTS, ETFS, FUTURES, PROJECT_DIR, TRADED, backtest, count_line, label,
                                load_signals, pnl_table, scored_mask, sharpe_test, summarise_tests, write_json)
from stats_tools import newey_west_variance
from strategies import equal_risk_portfolio, regime_long_binary, regime_long_scaled, vol_managed

OUTPUT_DIR = PROJECT_DIR / "outputs" / "regime_drift"
RULES = ("constant", "regime_binary", "regime_scaled", "vol_managed")


def conditional_mean_test(r, p_high):
    """Mean return in predicted-calm minus predicted-volatile days, Newey-West t."""
    calm = p_high <= 0.5
    if calm.sum() < 30 or (~calm).sum() < 30:
        return {"mean_calm": np.nan, "mean_volatile": np.nan, "t_stat": np.nan, "p_value": np.nan}
    d = np.where(calm, r, 0.0)
    # Difference of means via a regression of r on an indicator.
    X = np.column_stack([np.ones(len(r)), calm.astype(float)])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ beta
    # Newey-West on the score of the indicator coefficient.
    xtx_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    lags = int(np.floor(4 * (len(r) / 100) ** (2 / 9)))
    S = scores.T @ scores
    for k in range(1, lags + 1):
        g = scores[k:].T @ scores[:-k]
        S += (1 - k / (lags + 1)) * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    t = beta[1] / np.sqrt(cov[1, 1])
    from scipy.stats import norm
    return {"mean_calm": float(r[calm].mean()), "mean_volatile": float(r[~calm].mean()),
            "sd_calm": float(r[calm].std(ddof=1)), "sd_volatile": float(r[~calm].std(ddof=1)),
            "share_calm": float(calm.mean()), "t_stat": float(t), "p_value": float(2 * (1 - norm.cdf(abs(t))))}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows, test_rows, cond_rows = [], [], []
    store, scale = {}, {}
    for key in TRADED:
        s = load_signals(key)
        sc = scored_mask(s.index)
        r = s["return"].to_numpy()
        positions = {
            "constant": np.ones(len(s)),
            "regime_binary": regime_long_binary(s["p_high"].to_numpy()),
            "regime_scaled": regime_long_scaled(s["p_high"].to_numpy()),
            "vol_managed": vol_managed(r, s["ewma_var"].to_numpy()),
        }
        cond_rows.append({"key": key, **conditional_mean_test(r[sc], s["p_high"].to_numpy()[sc])})
        # Scale for the pooled portfolio: the scored-period realised volatility
        # of the constant long, applied to every rule on the asset alike.
        scale[key] = float(np.nanstd(r[sc], ddof=1))
        for cost_name, cost in COSTS.items():
            frames = pnl_table(r, positions, cost, sc, index=s.index)
            for name, frame in frames.items():
                store[(key, name, cost_name)] = frame
                metric_rows.append({"key": key, "strategy": name, "cost": cost_name, **backtest.metrics(frame)})
            for rule in RULES[1:]:
                boot = sharpe_test(frames[rule]["net"], frames["constant"]["net"])
                test_rows.append({"key": key, "cost": cost_name, "comparison": f"{rule}_vs_constant", **boot})
            boot = sharpe_test(frames["regime_binary"]["net"], frames["vol_managed"]["net"])
            test_rows.append({"key": key, "cost": cost_name, "comparison": "regime_binary_vs_vol_managed", **boot})
    metrics = pd.DataFrame(metric_rows)
    tests = summarise_tests(test_rows)
    cond = pd.DataFrame(cond_rows)
    from stats_tools import holm
    cond["p_holm"] = holm(cond["p_value"].to_numpy())
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "sharpe_tests.csv", index=False)
    cond.to_csv(OUTPUT_DIR / "conditional_means.csv", index=False)

    pooled_rows, port = [], {}
    for universe_name, keys in (("futures", FUTURES), ("etfs", ETFS), ("all", TRADED)):
        for cost_name in COSTS:
            nets = {rule: equal_risk_portfolio({k: store[(k, rule, cost_name)]["net"] / scale[k] for k in keys}) for rule in RULES}
            port[(universe_name, cost_name)] = nets
            row = {"universe": universe_name, "cost": cost_name}
            for rule, net in nets.items():
                row[f"sharpe_{rule}"] = backtest.sharpe(net)
            for rule in RULES[1:]:
                boot = sharpe_test(nets[rule], nets["constant"])
                row[f"diff_{rule}_vs_constant"] = boot["observed"]
                row[f"p_{rule}_vs_constant"] = boot["p_value"]
            boot = sharpe_test(nets["regime_binary"], nets["vol_managed"])
            row["diff_regime_binary_vs_vol_managed"] = boot["observed"]
            row["p_regime_binary_vs_vol_managed"] = boot["p_value"]
            pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUTPUT_DIR / "pooled.csv", index=False)
    comps = sorted(tests["comparison"].unique())
    counts = pd.DataFrame([count_line(tests[tests["cost"] == c], comp) | {"cost": c} for c in COSTS for comp in comps])
    counts.to_csv(OUTPUT_DIR / "counts.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    nets = port[("futures", "base")]
    for rule, colour in zip(RULES, (COLOURS["constant"], COLOURS["hmm"], "#6FA8DC", COLOURS["ewma"])):
        axes[0].plot(nets[rule].index, nets[rule].cumsum(), label=f"{rule} (Sharpe {backtest.sharpe(nets[rule]):.2f})", color=colour)
    axes[0].set_title("Equal-risk portfolio of 13 futures, long-only rules, base costs: cumulative net P&L (unit-vol scaled)")
    axes[0].legend()
    x = np.arange(len(cond))
    axes[1].bar(x - 0.2, cond["mean_calm"] * 252, 0.4, label="predicted calm", color=COLOURS["hmm"])
    axes[1].bar(x + 0.2, cond["mean_volatile"] * 252, 0.4, label="predicted volatile", color=COLOURS["ewma"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label(k) for k in cond["key"]], rotation=45, ha="right")
    axes[1].set_title("Annualised mean return by one-step predicted state, 2015 to 2024 (units: percent, WTI dollars)")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_regime_drift.png", dpi=150)

    write_json(OUTPUT_DIR / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records")})
    print(metrics[metrics["cost"] == "base"].pivot(index="key", columns="strategy", values="sharpe").round(3).to_string())
    print(cond.round(3).to_string(index=False))
    print(pooled.round(3).to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
