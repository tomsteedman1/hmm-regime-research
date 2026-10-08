"""E1. Full-sample regime structure for every traded series.

Per asset: T2 and G2 on 2010 to 2024 with twelve starts, the BIC
preference, initialisation stability, and sample stability against the
early (2010 to 2016) and late (2017 to 2024) halves with eight starts.
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import research_link  # noqa: F401
from multi_start import fit_multiple_starts_general  # noqa: E402
from regime_hmm import RegimeHMM  # noqa: E402

from assets import TRADED, UNIVERSE, load_returns  # noqa: E402
from vol_models import BASE_SEED, make_t2  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "regime_atlas"


def make_g2(init_strategy, random_state):
    return RegimeHMM(n_states=2, emission="gaussian", init_strategy=init_strategy, random_state=random_state)


def describe(model, values, records):
    high = int(np.argmax(model.stdev))
    low = 1 - high
    path = model.decode(values)
    gaps = np.array([r["log_likelihood_gap"] for r in records])
    agree = np.array([r["agreement_with_best"] for r in records])
    within = gaps <= 1.0
    out = {
        "loglik": model.score(values),
        "bic": model.bic(values),
        "mu_low": model.mu[low], "mu_high": model.mu[high],
        "stdev_low": model.stdev[low], "stdev_high": model.stdev[high],
        "stay_low": model.A[low, low], "stay_high": model.A[high, high],
        "duration_low": 1 / (1 - model.A[low, low]), "duration_high": 1 / (1 - model.A[high, high]),
        "occupancy_high": float((path == high).mean()),
        "starts_within_1": int(within.sum()),
        "min_agreement_within_1": float(agree[within].min()),
    }
    if model.nu is not None:
        out["nu_low"] = model.nu[low]
        out["nu_high"] = model.nu[high]
    return out, path, high


def one(key):
    started = time.time()
    r = load_returns(key)
    values = r.to_numpy()
    t2, rec_t2, _ = fit_multiple_starts_general(values, make_t2, 12, BASE_SEED)
    g2, rec_g2, _ = fit_multiple_starts_general(values, make_g2, 12, BASE_SEED)
    d_t2, path_t2, high_t2 = describe(t2, values, rec_t2)
    d_g2, path_g2, high_g2 = describe(g2, values, rec_g2)
    row = {"key": key, "name": UNIVERSE[key]["name"], "group": UNIVERSE[key]["group"], "obs": len(values)}
    row.update({f"t2_{k}": v for k, v in d_t2.items()})
    row.update({f"g2_{k}": v for k, v in d_g2.items()})
    row["path_agreement_t2_g2"] = float(((path_t2 == high_t2) == (path_g2 == high_g2)).mean())
    # Sample stability of T2 against the two halves.
    for label, mask in (("early", r.index <= "2016-12-31"), ("late", r.index >= "2017-01-01")):
        sub, _, _ = fit_multiple_starts_general(values[mask], make_t2, 8, BASE_SEED)
        sub_high = int(np.argmax(sub.stdev))
        sub_path = sub.decode(values[mask])
        row[f"t2_{label}_agreement"] = float(((sub_path == sub_high) == (path_t2[mask] == high_t2)).mean())
        row[f"t2_{label}_stdev_high"] = sub.stdev[sub_high]
        row[f"t2_{label}_stdev_low"] = sub.stdev[1 - sub_high]
    # State-conditional realised statistics from the full-sample smoothed path
    # (descriptive only; not a real-time quantity).
    in_high = path_t2 == high_t2
    row["realised_mean_low"] = float(values[~in_high].mean())
    row["realised_mean_high"] = float(values[in_high].mean())
    row["realised_sd_low"] = float(values[~in_high].std(ddof=1))
    row["realised_sd_high"] = float(values[in_high].std(ddof=1))
    pd.DataFrame({"date": r.index, "return": values, "t2_high": in_high, "g2_high": path_g2 == high_g2}).to_csv(
        OUTPUT_DIR / f"{key}_paths.csv", index=False
    )
    row["seconds"] = time.time() - started
    print(f"{key:10s} {row['seconds']:5.0f}s  T2 high occ {row['t2_occupancy_high']:.2f}", flush=True)
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Pool(min(len(TRADED), os.cpu_count() or 1)) as pool:
        rows = pool.map(one, TRADED)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "regime_atlas.csv", index=False)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
