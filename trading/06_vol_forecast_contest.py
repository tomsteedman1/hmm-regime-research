"""E2. One-day-ahead variance forecasting contest, 2015 to 2024.

HMM against EWMA, GARCH(1,1)-t and HAR-GK on QLIKE and squared error,
against the squared return and the Garman-Klass proxies. Diebold-Mariano
per asset (Newey-West), Holm across assets, and one pooled test on the
cross-asset average loss differential.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_common import COLOURS, PROJECT_DIR, TRADED, label, load_signals, scored_mask, write_json
from stats_tools import diebold_mariano, holm, qlike, squared_error

OUTPUT_DIR = PROJECT_DIR / "outputs" / "vol_forecast_contest"
MODELS = ["hmm", "ewma", "garch", "har"]
PROXIES = {"r2": "squared return", "gk": "Garman-Klass"}
LOSSES = {"qlike": qlike, "mse": squared_error}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_rows, dm_rows = [], []
    pooled_diff = {}  # (loss, proxy, rival) -> {key: Series of daily differential}
    mz_rows = []
    for key in TRADED:
        s = load_signals(key)
        sc = scored_mask(s.index)
        s = s[sc]
        for proxy_name in PROXIES:
            proxy = s[proxy_name].to_numpy()
            valid = np.isfinite(proxy) & (proxy > 0)
            for model in MODELS:
                valid &= np.isfinite(s[f"{model}_var"].to_numpy())
            for loss_name, loss_fn in LOSSES.items():
                losses = {m: loss_fn(s[f"{m}_var"].to_numpy()[valid], proxy[valid]) for m in MODELS}
                for m in MODELS:
                    loss_rows.append({"key": key, "proxy": proxy_name, "loss": loss_name, "model": m,
                                      "mean_loss": float(np.mean(losses[m])), "days": int(valid.sum())})
                for rival in MODELS[1:]:
                    dm = diebold_mariano(losses["hmm"], losses[rival])
                    dm_rows.append({"key": key, "proxy": proxy_name, "loss": loss_name,
                                    "comparison": f"hmm_vs_{rival}", **dm})
                    pooled_diff.setdefault((loss_name, proxy_name, rival), {})[key] = pd.Series(
                        losses["hmm"] - losses[rival], index=s.index[valid])
            # Mincer-Zarnowitz slope on the squared-return proxy, descriptive.
            if proxy_name == "r2":
                for m in MODELS:
                    x = s[f"{m}_var"].to_numpy()[valid]
                    y = proxy[valid]
                    X = np.column_stack([np.ones(len(x)), x])
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ beta
                    r2 = 1 - resid.var() / y.var()
                    mz_rows.append({"key": key, "model": m, "mz_intercept": beta[0], "mz_slope": beta[1], "mz_r2": r2,
                                    "mean_forecast": x.mean(), "mean_proxy": y.mean()})

    losses = pd.DataFrame(loss_rows)
    dm = pd.DataFrame(dm_rows)
    dm["p_holm"] = np.nan
    for _, idx in dm.groupby(["proxy", "loss", "comparison"]).groups.items():
        dm.loc[idx, "p_holm"] = holm(dm.loc[idx, "p_value"].to_numpy())
    losses.to_csv(OUTPUT_DIR / "mean_losses.csv", index=False)
    dm.to_csv(OUTPUT_DIR / "diebold_mariano.csv", index=False)
    pd.DataFrame(mz_rows).to_csv(OUTPUT_DIR / "mincer_zarnowitz.csv", index=False)

    # Ranking per asset on QLIKE / r2.
    rank = losses[(losses["loss"] == "qlike") & (losses["proxy"] == "r2")].pivot(index="key", columns="model", values="mean_loss")
    rank["hmm_rank"] = rank[MODELS].rank(axis=1)["hmm"]
    rank.to_csv(OUTPUT_DIR / "qlike_r2_by_asset.csv")

    pooled_rows = []
    for (loss_name, proxy_name, rival), diffs in pooled_diff.items():
        frame = pd.DataFrame(diffs)
        avg = frame.mean(axis=1, skipna=True).dropna()
        dmp = diebold_mariano(avg.to_numpy(), np.zeros(len(avg)))
        pooled_rows.append({"loss": loss_name, "proxy": proxy_name, "comparison": f"hmm_vs_{rival}", **dmp,
                            "hmm_wins_assets": int((frame.mean() < 0).sum()), "assets": int(frame.shape[1])})
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUTPUT_DIR / "pooled_tests.csv", index=False)

    counts = []
    for (proxy_name, loss_name, comp), sub in dm.groupby(["proxy", "loss", "comparison"]):
        counts.append({"proxy": proxy_name, "loss": loss_name, "comparison": comp, "assets": len(sub),
                       "hmm_better": int((sub["mean_diff"] < 0).sum()),
                       "hmm_better_p5": int(((sub["mean_diff"] < 0) & (sub["p_value"] < 0.05)).sum()),
                       "rival_better_p5": int(((sub["mean_diff"] > 0) & (sub["p_value"] < 0.05)).sum()),
                       "holm_p5": int((sub["p_holm"] < 0.05).sum())})
    counts = pd.DataFrame(counts)
    counts.to_csv(OUTPUT_DIR / "counts.csv", index=False)

    # Chart: QLIKE (r2) per asset, models as bars.
    fig, ax = plt.subplots(figsize=(13, 5))
    keys = list(rank.index)
    x = np.arange(len(keys))
    w = 0.2
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1.5) * w, rank[m], w, label=m.upper(), color=COLOURS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([label(k) for k in keys], rotation=45, ha="right")
    ax.set_ylabel("Mean QLIKE, squared-return proxy")
    ax.set_title("One-day-ahead variance forecast loss, 2015 to 2024 (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_qlike_by_asset.png", dpi=150)

    write_json(OUTPUT_DIR / "summary.json", {"pooled": pooled.to_dict("records"), "counts": counts.to_dict("records")})
    print(rank.round(4).to_string())
    print(pooled.round(4).to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
