"""Audit WTI data before it enters the HMM.

This script reads the frozen CL=F series in data/raw. It downloads from
Yahoo Finance only when no frozen file exists, and then freezes the download.
It checks data quality, compares price transformations, inspects April 2020,
and saves tables and charts in outputs/data_audit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "CL=F"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
ROLLING_WINDOW = 21
EXTREME_COUNT = 15
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "data_audit"
FROZEN_DATA = (
    Path(__file__).resolve().parent
    / "data"
    / "raw"
    / "wti_CL=F_2010-01-04_to_2024-12-30_downloaded_2026-08-11.csv"
)


def download_price() -> pd.Series:
    """Download the adjusted close series used by the first HMM."""
    raw = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("The WTI download returned no observations.")

    price = raw["Close"].squeeze().rename("price")
    price.index = pd.to_datetime(price.index).tz_localize(None)
    return price.sort_index()


def load_price(path: Path = FROZEN_DATA) -> pd.Series:
    """Read the frozen estimation sample, or download and freeze it once."""
    if path.exists():
        data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        return data["price"].astype(float).sort_index()
    price = download_price()
    path.parent.mkdir(parents=True, exist_ok=True)
    price.rename("price").to_csv(path, index_label="date")
    print(f"Frozen download saved to {path}. Record its hash in DATA_MANIFEST.md.")
    return price


def build_audit_table(price: pd.Series) -> pd.DataFrame:
    """Create checks and candidate transformations."""
    audit = price.to_frame()
    audit["price_change"] = audit["price"].diff()
    audit["simple_return"] = audit["price"].pct_change(fill_method=None)

    previous_price = audit["price"].shift(1)
    valid_log = (audit["price"] > 0) & (previous_price > 0)
    audit["log_return"] = np.nan
    audit.loc[valid_log, "log_return"] = np.log(
        audit.loc[valid_log, "price"] / previous_price.loc[valid_log]
    )

    valid_scaled = previous_price.abs() > 0
    audit["absolute_base_return"] = np.nan
    audit.loc[valid_scaled, "absolute_base_return"] = (
        audit.loc[valid_scaled, "price_change"]
        / previous_price.loc[valid_scaled].abs()
    )

    audit["log_return_valid"] = valid_log
    audit["missing_price"] = audit["price"].isna()
    audit["zero_price"] = audit["price"].eq(0)
    audit["negative_price"] = audit["price"].lt(0)
    audit["month_day"] = audit.index.day
    audit["possible_roll_window"] = audit["month_day"].between(14, 23)
    audit["rolling_volatility_21d"] = (
        audit["simple_return"].rolling(ROLLING_WINDOW).std()
        * np.sqrt(252)
    )
    return audit


def summary_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data-quality summary."""
    values = {
        "start_date": audit.index.min().date().isoformat(),
        "end_date": audit.index.max().date().isoformat(),
        "observations": len(audit),
        "duplicate_dates": int(audit.index.duplicated().sum()),
        "missing_prices": int(audit["missing_price"].sum()),
        "zero_prices": int(audit["zero_price"].sum()),
        "negative_prices": int(audit["negative_price"].sum()),
        "undefined_log_returns": int(audit["log_return"].isna().sum()),
        "minimum_price": float(audit["price"].min()),
        "maximum_price": float(audit["price"].max()),
    }
    return pd.DataFrame(
        {"check": list(values.keys()), "value": list(values.values())}
    )


def extreme_table(audit: pd.DataFrame) -> pd.DataFrame:
    """List the largest absolute simple returns."""
    columns = [
        "price",
        "price_change",
        "simple_return",
        "log_return",
        "absolute_base_return",
        "possible_roll_window",
    ]
    rows = audit.loc[audit["simple_return"].abs().nlargest(EXTREME_COUNT).index]
    return rows[columns].sort_values("simple_return")


def save_price_chart(audit: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(audit.index, audit["price"], color="#1f4e79", linewidth=0.9)
    negative = audit[audit["negative_price"]]
    ax.scatter(
        negative.index,
        negative["price"],
        color="#c00000",
        marker="o",
        s=35,
        label="Negative price",
        zorder=3,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(title="WTI continuous futures price", ylabel="USD per barrel")
    if not negative.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_raw_price.png", dpi=160)
    plt.close(fig)


def save_april_chart(april: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(april.index, april["price"], marker="o", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(title="WTI price around April 2020", ylabel="USD per barrel")

    axes[1].plot(
        april.index,
        april["simple_return"],
        marker="o",
        label="Simple return",
    )
    axes[1].plot(
        april.index,
        april["log_return"],
        marker="o",
        label="Log return",
    )
    axes[1].plot(
        april.index,
        april["absolute_base_return"],
        marker="o",
        label="Price change / absolute previous price",
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(title="Candidate transformations", ylabel="Daily value")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_april_2020.png", dpi=160)
    plt.close(fig)


def save_distribution_chart(audit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    audit["simple_return"].dropna().hist(ax=axes[0], bins=100, color="#4472c4")
    axes[0].set(title="Simple-return distribution", xlabel="Simple return")

    audit["log_return"].dropna().hist(ax=axes[1], bins=100, color="#70ad47")
    axes[1].set(title="Valid log-return distribution", xlabel="Log return")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_return_distributions.png", dpi=160)
    plt.close(fig)


def save_volatility_chart(audit: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        audit.index,
        audit["rolling_volatility_21d"],
        color="#7030a0",
        linewidth=0.9,
    )
    ax.set(
        title="WTI 21-day rolling annualised volatility",
        ylabel="Annualised volatility",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_rolling_volatility.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if FROZEN_DATA.exists():
        print(f"Reading frozen {TICKER} data from {FROZEN_DATA.name}.")
    else:
        print(f"Downloading {TICKER} from {START_DATE} to {END_DATE}.")

    price = load_price()
    audit = build_audit_table(price)
    summary = summary_table(audit)
    april = audit.loc["2020-04-13":"2020-04-27"]
    extremes = extreme_table(audit)

    audit.to_csv(OUTPUT_DIR / "wti_audit_data.csv", index_label="date")
    summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    april.to_csv(OUTPUT_DIR / "april_2020_detail.csv", index_label="date")
    extremes.to_csv(OUTPUT_DIR / "extreme_observations.csv", index_label="date")

    save_price_chart(audit)
    save_april_chart(april)
    save_distribution_chart(audit)
    save_volatility_chart(audit)

    print("\nData-quality summary")
    print(summary.to_string(index=False))
    print("\nApril 2020 detail")
    print(
        april[
            ["price", "simple_return", "log_return", "absolute_base_return"]
        ].to_string(float_format=lambda value: f"{value:.6f}")
    )
    print("\nLargest absolute simple returns")
    print(extremes.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nFiles saved in {OUTPUT_DIR}")
    print(
        "The possible-roll-window field is a screening flag. "
        "It does not identify a contract roll."
    )


if __name__ == "__main__":
    main()
