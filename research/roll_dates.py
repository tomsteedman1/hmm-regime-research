"""Contract-roll dates for spliced front-month futures series.

Ported unchanged from ``01_roll_audit.py`` in the HMM Regime Trading project
(2 September 2026), which found that the day after each expiry is the roll
day of Yahoo's ``CL=F`` series.

NYMEX CL expiry rule: trading in the contract for delivery month M ends three
business days before the 25th calendar day of month M-1. If the 25th is not a
business day, count back from the last business day before it. The series'
own dates are used as the business-day calendar.
"""

import numpy as np
import pandas as pd


def cl_expiry_dates(trading_days):
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


def cl_roll_days(trading_days):
    """Return the day after each CL expiry: the roll day of the spliced series."""
    days = pd.DatetimeIndex(trading_days)
    after = [days.get_loc(expiry) + 1 for expiry in cl_expiry_dates(days)]
    return days[[position for position in after if position < len(days)]]
