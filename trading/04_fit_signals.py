"""Walk-forward forecasts for every traded series, cached to ``outputs/signals``.

For each asset: T2 HMM (yearly refit, twelve starts), GARCH(1,1)-t (yearly
refit), HAR on the Garman-Klass variance (yearly refit) and EWMA(21). Every
element uses data to the previous day. Per-year fitted parameters are
saved so that signal drift can be inspected, as the WTI result asked for.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from assets import TRADED, load_range_variance, load_returns
import vol_models

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "signals"


def one(key: str) -> str:
    started = time.time()
    r = load_returns(key)
    gk = load_range_variance(key).reindex(r.index)
    # Where the range is undefined (zero range or bad high/low) the HAR
    # regressors fall back to the squared return, so the design has no holes.
    gk_filled = gk.fillna(r ** 2)

    hmm, hmm_params = vol_models.walk_forward_hmm(r)
    garch_var, garch_params = vol_models.walk_forward_garch(r)
    har_var = vol_models.walk_forward_har(gk_filled)
    ewma_var = vol_models.ewma_variance(r.to_numpy())

    frame = pd.DataFrame(
        {
            "return": r.to_numpy(),
            "r2": r.to_numpy() ** 2,
            "gk": gk.to_numpy(),
            "hmm_mean": hmm["hmm_mean"].to_numpy(),
            "hmm_var": hmm["hmm_var"].to_numpy(),
            "p_high": hmm["p_high"].to_numpy(),
            "garch_var": garch_var,
            "har_var": har_var,
            "ewma_var": ewma_var,
        },
        index=r.index,
    )
    frame.to_csv(OUTPUT_DIR / f"{key}_signals.csv")
    hmm_params.to_csv(OUTPUT_DIR / f"{key}_hmm_params.csv", index=False)
    garch_params.to_csv(OUTPUT_DIR / f"{key}_garch_params.csv", index=False)
    return f"{key:10s} {time.time() - started:6.0f}s"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sys.argv[1:] or TRADED
    with Pool(min(len(keys), os.cpu_count() or 1)) as pool:
        for line in pool.imap_unordered(one, keys):
            print(line, flush=True)


if __name__ == "__main__":
    main()
