"""Download the development samples from Yahoo Finance.

The price files are not in this repository. This script fetches them and
writes them under the exact file names the scripts expect:

- ``research/data/raw``: the WTI close, 4 January 2010 to 30 December 2024.
- ``trading/data/raw``: the cross-asset universe over the same range, named
  as in ``trading/data/raw/manifest.csv``.

Yahoo Finance revises history without notice. Each file's values, rounded
to six decimals, are hashed and compared with ``data_hashes.csv``, made the
same way from the files the recorded results were produced on.
A mismatch is reported, not hidden: results rebuilt from a revised file can
differ from the ones in the result documents.

The 2025 to 2026 holdout is sealed and is not downloaded.

Usage: ``python3 download_data.py``. Existing files are never overwritten.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
RESEARCH_RAW = ROOT / "research" / "data" / "raw"
TRADING_RAW = ROOT / "trading" / "data" / "raw"
START = "2010-01-01"
END_EXCLUSIVE = "2024-12-31"

WTI_FILE = "wti_CL=F_2010-01-04_to_2024-12-30_downloaded_2026-08-11.csv"
EXPECTED = pd.read_csv(ROOT / "data_hashes.csv").set_index("file")["value_sha256"]


def value_sha256(path: Path) -> str:
    """Hash of the values rounded to six decimals, blind to float noise."""
    text = pd.read_csv(path, index_col=0).to_csv(float_format="%.6f", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def fetch(ticker: str, auto_adjust: bool) -> pd.DataFrame:
    frame = pd.DataFrame()
    for attempt in range(5):
        frame = yf.download(
            ticker, start=START, end=END_EXCLUSIVE, auto_adjust=auto_adjust,
            progress=False, actions=False,
        )
        if len(frame):
            break
        time.sleep(5 * (attempt + 1))
    if not len(frame):
        raise SystemExit(f"{ticker}: empty download after retries")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    return frame[~frame.index.duplicated()].sort_index()


def report(path: Path) -> bool:
    match = value_sha256(path) == EXPECTED[path.name]
    print(f"{'match   ' if match else 'MISMATCH'} {path.relative_to(ROOT)}")
    return match


def wti() -> bool:
    path = RESEARCH_RAW / WTI_FILE
    if not path.exists():
        # Same settings as research/01_data_audit.py.
        price = fetch("CL=F", auto_adjust=True)["Close"].rename("price")
        path.parent.mkdir(parents=True, exist_ok=True)
        price.to_csv(path, index_label="date")
    return report(path)


def universe() -> list[bool]:
    manifest = pd.read_csv(TRADING_RAW / "manifest.csv")
    results = []
    for row in manifest[manifest["sample"] == "development"].itertuples():
        path = TRADING_RAW / row.file
        if not path.exists():
            # Same settings as trading/03_download_assets.py.
            frame = fetch(row.ticker, auto_adjust=False)
            frame = frame.rename(columns=str.lower)[
                ["open", "high", "low", "close", "adj close", "volume"]
            ].rename(columns={"adj close": "adj_close"})
            frame.dropna(subset=["close"]).to_csv(path, float_format="%.6f")
        results.append(report(path))
    return results


def main() -> int:
    results = [wti(), *universe()]
    bad = results.count(False)
    print(f"\n{len(results) - bad} of {len(results)} files match the recorded hashes.")
    if bad:
        print("Yahoo has revised some history. Rebuilt results can differ from the result documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
