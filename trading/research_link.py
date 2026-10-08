"""Link to the research half of this repository.

Import this module before any research module. It puts ``research/`` on
``sys.path`` and records the hash of the frozen WTI values every result in
this project was produced against. ``test_research_link.py`` fails if the
local WTI file differs from them.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = PROJECT_DIR.parent / "research"
# Values rounded to six decimals; see ``value_sha256`` and ../data_hashes.csv.
RESEARCH_DATA_SHA256 = (
    "3e225ba917818a98f05f53bd7b3be4b86445f28f489122929a9422de5fbf7e3e"
)

if not RESEARCH_DIR.exists():
    raise FileNotFoundError(f"Research folder not found: {RESEARCH_DIR}")
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))


def value_sha256(path: Path) -> str:
    """Hash of a price file's values rounded to six decimals."""
    text = pd.read_csv(path, index_col=0).to_csv(float_format="%.6f", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()
