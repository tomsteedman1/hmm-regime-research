"""Audit contract-roll jumps in the frozen WTI series.

Yahoo's ``CL=F`` is a spliced front-month series with no roll adjustment. On
the day the front contract changes, the recorded price change mixes a real
market move with the calendar spread between the old and new contracts. This
script finds those days and measures how large the mixing is, so that
``DATA_DECISION.md`` can say whether the series is fit for P&L.

NYMEX CL expiry rule: trading in the contract for delivery month M ends three
business days before the 25th calendar day of month M-1. If the 25th is not a
business day, count back from the last business day before it. The frozen
file's own dates are used as the business-day calendar. Yahoo switches its
front-month series on the expiry day or the day after, so both days are
flagged and reported separately.

Outputs go to ``outputs/roll_audit``.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import research_link  # noqa: F401  (puts the research repository on the path)
from wti_data import CROSS_ZERO_DATES, load_changes, load_prices  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "roll_audit"
T2_SCALE = {"low": 1.0166, "high": 1.7391}  # from the research T2 summary


def expiry_dates(trading_days):
    """Return the CL expiry date for every delivery month in the sample."""
    days = pd.DatetimeIndex(trading_days)
    positions = pd.Series(np.arange(len(days)), index=days)
    expiries = []
    months = pd.period_range(days[0], days[-1], freq="M")
    for month in months:
        twenty_fifth = pd.Timestamp(year=month.year, month=month.month, day=25)
        if twenty_fifth > days[-1] or twenty_fifth < days[0]:
            continue
        anchor = days[days <= twenty_fifth][-1]
        anchor_position = positions[anchor]
        expiry_position = anchor_position - 3
        if expiry_position < 0:
            continue
        expiries.append(days[expiry_position])
    return pd.DatetimeIndex(expiries)


def flag_roll_days(changes):
    """Label each change as an expiry day, the day after expiry, or neither."""
    days = changes.index
    positions = pd.Series(np.arange(len(days)), index=days)
    expiries = expiry_dates(days)
    label = pd.Series("other", index=days, name="roll_flag")
    for expiry in expiries:
        if expiry in positions.index:
            label.loc[expiry] = "expiry_day"
            after = positions[expiry] + 1
            if after < len(days):
                label.iloc[after] = "day_after_expiry"
    return label, expiries


def describe(group):
    absolute = group.abs()
    return pd.Series(
        {
            "days": len(group),
            "mean": group.mean(),
            "median": group.median(),
            "std": group.std(),
            "mean_abs": absolute.mean(),
            "share_abs_over_1": (absolute > 1).mean(),
            "share_abs_over_2": (absolute > 2).mean(),
            "sum": group.sum(),
        }
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price = load_prices()
    changes = load_changes()
    label, expiries = flag_roll_days(changes)

    table = pd.DataFrame({"price_change_usd": changes, "roll_flag": label})
    table["price"] = price.reindex(table.index)
    table["abs_change"] = table["price_change_usd"].abs()

    summary = table.groupby("roll_flag")["price_change_usd"].apply(describe).unstack()
    summary = summary.loc[["expiry_day", "day_after_expiry", "other"]]
    summary["mean_abs_over_low_scale"] = summary["mean_abs"] / T2_SCALE["low"]
    summary["mean_abs_over_high_scale"] = summary["mean_abs"] / T2_SCALE["high"]
    summary.to_csv(OUTPUT_DIR / "roll_day_summary.csv")

    # The same table without 20 and 21 April 2020. The May 2020 contract expired
    # on 21 April, so the cross-zero days sit inside the flagged window and
    # dominate any mean or standard deviation that includes them.
    clean = table.drop(CROSS_ZERO_DATES, errors="ignore")
    summary_clean = clean.groupby("roll_flag")["price_change_usd"].apply(describe).unstack()
    summary_clean = summary_clean.loc[["expiry_day", "day_after_expiry", "other"]]
    summary_clean["median_abs"] = clean.groupby("roll_flag")["abs_change"].median()
    summary_clean["mean_abs_over_low_scale"] = summary_clean["mean_abs"] / T2_SCALE["low"]
    summary_clean.to_csv(OUTPUT_DIR / "roll_day_summary_excluding_cross_zero_days.csv")

    total_move = changes.sum()
    roll_move = table.loc[table["roll_flag"] != "other", "price_change_usd"].sum()
    cumulative = pd.DataFrame(
        {
            "all_changes": changes.cumsum(),
            "roll_days_only": table["price_change_usd"]
            .where(table["roll_flag"] != "other", 0.0)
            .cumsum(),
            "excluding_roll_days": table["price_change_usd"]
            .where(table["roll_flag"] == "other", 0.0)
            .cumsum(),
        }
    )
    cumulative.to_csv(OUTPUT_DIR / "cumulative_paths.csv")

    roll_days = table[table["roll_flag"] != "other"].copy()
    roll_days.to_csv(OUTPUT_DIR / "roll_day_changes.csv")

    # Which of the two flagged days carries the bigger move each month
    monthly = []
    for expiry in expiries:
        if expiry not in table.index:
            continue
        i = table.index.get_loc(expiry)
        if i + 1 >= len(table):
            continue
        monthly.append(
            {
                "expiry_day": expiry.date(),
                "expiry_change": table["price_change_usd"].iloc[i],
                "day_after_change": table["price_change_usd"].iloc[i + 1],
            }
        )
    monthly = pd.DataFrame(monthly)
    monthly["larger"] = np.where(
        monthly["expiry_change"].abs() >= monthly["day_after_change"].abs(),
        "expiry_day",
        "day_after_expiry",
    )
    monthly.to_csv(OUTPUT_DIR / "monthly_roll_pairs.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    bins = np.linspace(-8, 8, 65)
    for flag, colour in [("other", "#7C858E"), ("expiry_day", "#2F5D7C"), ("day_after_expiry", "#B4232C")]:
        values = table.loc[table["roll_flag"] == flag, "price_change_usd"].clip(-8, 8)
        axes[0].hist(values, bins=bins, density=True, alpha=0.5, label=flag, color=colour)
    axes[0].set_title("Daily price change by roll flag (density, clipped at ±$8)")
    axes[0].set_xlabel("USD per barrel")
    axes[0].legend()
    axes[1].plot(cumulative.index, cumulative["all_changes"], label="all changes", color="#182028")
    axes[1].plot(cumulative.index, cumulative["excluding_roll_days"], label="excluding flagged roll days", color="#2F5D7C")
    axes[1].plot(cumulative.index, cumulative["roll_days_only"], label="flagged roll days only", color="#B4232C")
    axes[1].axhline(0, color="#D3D8D2", linewidth=0.8)
    axes[1].set_title("Cumulative sum of daily changes")
    axes[1].set_ylabel("USD per barrel")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_roll_audit.png", dpi=160)

    print(f"Trading days: {len(changes)}; expiries found: {len(expiries)}")
    print("\nSummary by roll flag")
    print(summary.round(4).to_string())
    print("\nSummary by roll flag, excluding 20 and 21 April 2020")
    print(summary_clean.round(4).to_string())
    clean_roll = clean.loc[clean["roll_flag"] != "other", "price_change_usd"].sum()
    print(f"\nSum on flagged roll days excluding April 2020: {clean_roll:9.2f}")
    print(f"\nSum of all changes:            {total_move:9.2f}")
    print(f"Sum on flagged roll days:      {roll_move:9.2f}")
    print(f"Sum excluding flagged days:    {total_move - roll_move:9.2f}")
    print("\nWhich flagged day carries the larger move")
    print(monthly["larger"].value_counts().to_string())
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
