"""E5. Time-series momentum with regime sizing and a regime cut.

Sign of the past 252-day return, sized to one unit of daily volatility.
Variants: EWMA-sized (the standard), HMM-sized, GARCH-sized (extra), and
EWMA-sized with the position cut to zero when the one-step high-state
probability exceeds one half.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_common import (COLOURS, COSTS, ETFS, FUTURES, PROJECT_DIR, TRADED, backtest, count_line, label,
                                load_signals, pnl_table, scored_mask, sharpe_test, summarise_tests, write_json)
from strategies import CALM_THRESHOLD, equal_risk_portfolio, tsmom

OUTPUT_DIR = PROJECT_DIR / "outputs" / "tsmom_regime"
VARIANTS = ("tsmom_ewma", "tsmom_hmm", "tsmom_garch", "tsmom_ewma_cut")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows, test_rows = [], []
    store = {}
    for key in TRADED:
        s = load_signals(key)
        sc = scored_mask(s.index)
        r = s["return"].to_numpy()
        p_high = s["p_high"].to_numpy()
        positions = {
            "tsmom_ewma": tsmom(r, s["ewma_var"].to_numpy()),
            "tsmom_hmm": tsmom(r, s["hmm_var"].to_numpy()),
            "tsmom_garch": tsmom(r, s["garch_var"].to_numpy()),
            "tsmom_ewma_cut": tsmom(r, s["ewma_var"].to_numpy(), p_high=p_high, cut=CALM_THRESHOLD),
        }
        for cost_name, cost in COSTS.items():
            frames = pnl_table(r, positions, cost, sc, index=s.index)
            for name, frame in frames.items():
                store[(key, name, cost_name)] = frame
                metric_rows.append({"key": key, "strategy": name, "cost": cost_name, **backtest.metrics(frame)})
            for variant in VARIANTS[1:]:
                boot = sharpe_test(frames[variant]["net"], frames["tsmom_ewma"]["net"])
                test_rows.append({"key": key, "cost": cost_name, "comparison": f"{variant}_vs_tsmom_ewma", **boot})
    metrics = pd.DataFrame(metric_rows)
    tests = summarise_tests(test_rows)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "sharpe_tests.csv", index=False)

    pooled_rows, port = [], {}
    for universe_name, keys in (("futures", FUTURES), ("etfs", ETFS), ("all", TRADED)):
        for cost_name in COSTS:
            nets = {v: equal_risk_portfolio({k: store[(k, v, cost_name)]["net"] for k in keys}) for v in VARIANTS}
            port[(universe_name, cost_name)] = nets
            row = {"universe": universe_name, "cost": cost_name}
            for v, net in nets.items():
                row[f"sharpe_{v}"] = backtest.sharpe(net)
                row[f"daily_vol_{v}"] = float(np.nanstd(net, ddof=1))
            for v in VARIANTS[1:]:
                boot = sharpe_test(nets[v], nets["tsmom_ewma"])
                row[f"diff_{v}"] = boot["observed"]
                row[f"p_{v}"] = boot["p_value"]
            pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUTPUT_DIR / "pooled.csv", index=False)
    comps = sorted(tests["comparison"].unique())
    counts = pd.DataFrame([count_line(tests[tests["cost"] == c], comp) | {"cost": c} for c in COSTS for comp in comps])
    counts.to_csv(OUTPUT_DIR / "counts.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    nets = port[("futures", "base")]
    for v, colour in zip(VARIANTS, (COLOURS["ewma"], COLOURS["hmm"], COLOURS["garch"], COLOURS["other"])):
        axes[0].plot(nets[v].index, nets[v].cumsum(), label=f"{v} (Sharpe {backtest.sharpe(nets[v]):.2f})", color=colour)
    axes[0].set_title("Equal-risk portfolio of 13 futures, time-series momentum variants, base costs")
    axes[0].legend()
    sub = metrics[metrics["cost"] == "base"].pivot(index="key", columns="strategy", values="sharpe")
    x = np.arange(len(sub))
    w = 0.2
    for i, (v, colour) in enumerate(zip(VARIANTS, (COLOURS["ewma"], COLOURS["hmm"], COLOURS["garch"], COLOURS["other"]))):
        axes[1].bar(x + (i - 1.5) * w, sub[v], w, label=v, color=colour)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label(k) for k in sub.index], rotation=45, ha="right")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title("Sharpe per asset, base costs")
    axes[1].legend(ncol=4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_tsmom.png", dpi=150)

    write_json(OUTPUT_DIR / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records")})
    print(sub.round(3).to_string())
    print(pooled.round(3).to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
