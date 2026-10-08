"""E3. The registered WTI volatility-target test on every asset.

HMM expected volatility against EWMA (the bar) and a constant long, with a
GARCH sizer as an extra reference. Costs of one and two hundredths of a
unit per unit of turnover. Sharpe differences by stationary block bootstrap,
Holm across assets, and one pooled equal-risk portfolio test.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_common import (COLOURS, COSTS, ETFS, FUTURES, PROJECT_DIR, TRADED, backtest, count_line, label,
                                load_signals, pnl_table, scored_mask, sharpe_test, summarise_tests, write_json)
from strategies import MAX_LEVERAGE, TARGET_VOL, equal_risk_portfolio, vol_target

OUTPUT_DIR = PROJECT_DIR / "outputs" / "cross_asset_vol_target"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows, test_rows = [], []
    store = {}
    for key in TRADED:
        s = load_signals(key)
        sc = scored_mask(s.index)
        positions = {
            "hmm": vol_target(s["hmm_var"].to_numpy()),
            "ewma": vol_target(s["ewma_var"].to_numpy()),
            "garch": vol_target(s["garch_var"].to_numpy()),
            "constant": np.ones(len(s)),
        }
        for cost_name, cost in COSTS.items():
            frames = pnl_table(s["return"].to_numpy(), positions, cost, sc, index=s.index)
            for name, frame in frames.items():
                store[(key, name, cost_name)] = frame
                metric_rows.append({"key": key, "strategy": name, "cost": cost_name, **backtest.metrics(frame)})
            for rival in ("ewma", "garch"):
                boot = sharpe_test(frames["hmm"]["net"], frames[rival]["net"])
                test_rows.append({"key": key, "cost": cost_name, "comparison": f"hmm_vs_{rival}", **boot})
    metrics = pd.DataFrame(metric_rows)
    tests = summarise_tests(test_rows)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "sharpe_tests.csv", index=False)

    pooled_rows, port_series = [], {}
    for universe_name, keys in (("futures", FUTURES), ("etfs", ETFS), ("all", TRADED)):
        for cost_name in COSTS:
            nets = {name: equal_risk_portfolio({k: store[(k, name, cost_name)]["net"] for k in keys})
                    for name in ("hmm", "ewma", "garch", "constant")}
            port_series[(universe_name, cost_name)] = nets
            row = {"universe": universe_name, "cost": cost_name}
            for name, net in nets.items():
                row[f"sharpe_{name}"] = backtest.sharpe(net)
                row[f"daily_vol_{name}"] = float(np.nanstd(net, ddof=1))
            for rival in ("ewma", "garch"):
                boot = sharpe_test(nets["hmm"], nets[rival])
                row[f"diff_vs_{rival}"] = boot["observed"]
                row[f"p_vs_{rival}"] = boot["p_value"]
                row[f"ci_vs_{rival}"] = f"{boot['ci_low']:+.2f} to {boot['ci_high']:+.2f}"
            pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUTPUT_DIR / "pooled.csv", index=False)
    counts = pd.DataFrame([count_line(tests[tests["cost"] == c], comp) | {"cost": c}
                           for c in COSTS for comp in ("hmm_vs_ewma", "hmm_vs_garch")])
    counts.to_csv(OUTPUT_DIR / "counts.csv", index=False)

    # Realised volatility against target, the TG3 analogue.
    band = metrics[(metrics["cost"] == "base")].pivot(index="key", columns="strategy", values="daily_vol")
    band.to_csv(OUTPUT_DIR / "realised_vol_vs_target.csv")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    nets = port_series[("futures", "base")]
    for name in ("hmm", "ewma", "garch", "constant"):
        axes[0].plot(nets[name].index, nets[name].cumsum(), label=f"{name} (Sharpe {backtest.sharpe(nets[name]):.2f})", color=COLOURS[name])
    axes[0].set_title("Equal-risk portfolio of 13 futures, volatility targeting, base costs: cumulative net P&L")
    axes[0].legend()
    sub = tests[(tests["cost"] == "base") & (tests["comparison"] == "hmm_vs_ewma")].set_index("key")
    x = np.arange(len(sub))
    axes[1].bar(x, sub["observed"], color=[COLOURS["hmm"] if p < 0.05 else "#9DB4C6" for p in sub["p_value"]])
    axes[1].errorbar(x, sub["observed"], yerr=[sub["observed"] - sub["ci_low"], sub["ci_high"] - sub["observed"]], fmt="none", color="black", lw=0.8)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label(k) for k in sub.index], rotation=45, ha="right")
    axes[1].set_title("Sharpe difference HMM minus EWMA per asset, 95% bootstrap interval (dark = p < 0.05)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_vol_target.png", dpi=150)

    write_json(OUTPUT_DIR / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records")})
    print(metrics[metrics["cost"] == "base"].pivot(index="key", columns="strategy", values="sharpe").round(3).to_string())
    print(tests[tests["cost"] == "base"].round(3).to_string(index=False))
    print(pooled.round(3).to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
