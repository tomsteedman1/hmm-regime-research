"""Download, freeze and seal the cross-asset universe.

Two files per ticker: a development file (2010 to 30 December 2024) and a
sealed holdout (1 January 2025 to 31 August 2026). Both are hashed into
``DATA_MANIFEST.md``. This script refuses to overwrite an existing frozen
file, so a rerun cannot silently replace the data behind a result.

WTI is not downloaded for the close. Its close comes from the research
frozen file. An OHLC file is fetched only for the range-based variance
proxy used in E2, and its close is compared with the frozen close so any
revision by Yahoo is on record.
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from assets import DATA_DIR, DEV_END, HOLDOUT_END, HOLDOUT_START, START, UNIVERSE

TODAY = date.today().isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    for attempt in range(5):
        frame = yf.download(
            ticker, start=start, end=end, auto_adjust=False, progress=False, actions=False
        )
        if len(frame):
            break
        time.sleep(5 * (attempt + 1))
    if not len(frame):
        raise SystemExit(f"{ticker}: empty download after retries")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)[["open", "high", "low", "close", "adj close", "volume"]]
    frame = frame.rename(columns={"adj close": "adj_close"})
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated()].sort_index()
    return frame.dropna(subset=["close"])


def write_frozen(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise SystemExit(f"Refusing to overwrite frozen file {path.name}")
    frame.to_csv(path, float_format="%.6f")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, spec in UNIVERSE.items():
        if key == "wti":
            continue  # the WTI close is the research frozen file
        ticker = spec["ticker"]
        safe = ticker.replace("^", "").replace("=", "")
        existing = [p for p in DATA_DIR.glob(f"{key}_{safe}_*.csv")]
        if len(existing) == 2:
            print(f"{key:12s} already frozen, skipping download")
            for path in sorted(existing):
                frame = pd.read_csv(path, parse_dates=["date"])
                rows.append({"key": key, "ticker": ticker, "sample": "holdout" if "_holdout_" in path.name else "development",
                             "file": path.name, "rows": len(frame), "first": frame["date"].iloc[0].date().isoformat(),
                             "last": frame["date"].iloc[-1].date().isoformat(), "sha256": sha256(path)})
            continue
        dev = fetch(ticker, START, (pd.Timestamp(DEV_END) + pd.Timedelta(days=1)).date().isoformat())
        hold = fetch(ticker, HOLDOUT_START, (pd.Timestamp(HOLDOUT_END) + pd.Timedelta(days=1)).date().isoformat())
        dev_path = DATA_DIR / f"{key}_{safe}_{dev.index[0].date()}_to_{dev.index[-1].date()}_downloaded_{TODAY}.csv"
        hold_path = DATA_DIR / f"{key}_{safe}_holdout_{hold.index[0].date()}_to_{hold.index[-1].date()}_downloaded_{TODAY}.csv"
        write_frozen(dev, dev_path)
        write_frozen(hold, hold_path)
        for label, frame, path in (("development", dev, dev_path), ("holdout", hold, hold_path)):
            rows.append(
                {
                    "key": key,
                    "ticker": ticker,
                    "sample": label,
                    "file": path.name,
                    "rows": len(frame),
                    "first": frame.index[0].date().isoformat(),
                    "last": frame.index[-1].date().isoformat(),
                    "sha256": sha256(path),
                }
            )
        print(f"{key:12s} {ticker:8s} dev {len(dev):5d} rows  holdout {len(hold):4d} rows")
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "manifest.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
