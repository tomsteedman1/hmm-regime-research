"""Skip the tests that read price data when it has not been downloaded.

Run ``python download_data.py`` from the repository root to fetch the frozen
WTI file, then ``python 01_data_audit.py`` here to build the audit table. The
continuous-integration run on GitHub has no price data, so these tests skip
there and every other test runs.
"""

from pathlib import Path

import pytest

from series import SERIES

AUDIT_DATA = Path(__file__).resolve().parent / "outputs" / "data_audit" / "wti_audit_data.csv"

NEEDS_FROZEN_WTI = (
    "test_series.py::WtiSeriesTests::",
    "test_series.py::RollDateTests::",
    "test_series.py::AsinhSeriesTests::test_observation_matches_the_registered_formula",
    "test_series.py::GenericSeriesTests::test_wrong_observation_count_is_refused",
)
NEEDS_AUDIT_TABLE = ("test_return_comparison.py::",)


def pytest_collection_modifyitems(config, items):
    have_wti = SERIES["wti"].frozen_file.exists()
    have_audit = AUDIT_DATA.exists()
    for item in items:
        if not have_wti and any(marker in item.nodeid for marker in NEEDS_FROZEN_WTI):
            item.add_marker(pytest.mark.skip(reason="frozen WTI file missing: run download_data.py"))
        if not have_audit and any(marker in item.nodeid for marker in NEEDS_AUDIT_TABLE):
            item.add_marker(pytest.mark.skip(reason="audit table missing: run 01_data_audit.py"))
