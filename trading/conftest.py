"""Skip the tests that read downloaded price files when they are missing.

Run ``python download_data.py`` from the repository root to fetch them. The
continuous-integration run on GitHub has no price data, so these tests skip
there and every other test runs.
"""

from pathlib import Path

import pytest

import research_link  # noqa: F401  puts research/ on sys.path
from wti_data import FROZEN_DATA

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

NEEDS_FROZEN_WTI = (
    "test_research_link.py::ResearchLinkTests::test_frozen_data_hash_matches_manifest",
    "test_research_link.py::ResearchLinkTests::test_selected_model_filters_the_frozen_sample",
)
NEEDS_ASSET_FILES = ("test_cross_asset.py::test_returns_and_units",)


def pytest_collection_modifyitems(config, items):
    have_wti = FROZEN_DATA.exists()
    have_assets = any(path.name != "manifest.csv" for path in RAW_DIR.glob("*.csv"))
    for item in items:
        if not have_wti and any(marker in item.nodeid for marker in NEEDS_FROZEN_WTI):
            item.add_marker(pytest.mark.skip(reason="frozen WTI file missing: run download_data.py"))
        if not have_assets and any(marker in item.nodeid for marker in NEEDS_ASSET_FILES):
            item.add_marker(pytest.mark.skip(reason="asset price files missing: run download_data.py"))
