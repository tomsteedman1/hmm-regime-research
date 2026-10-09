"""Skip the tests that read the frozen WTI file when it has not been downloaded.

Run ``python download_data.py`` from the repository root to fetch it. The
continuous-integration run on GitHub has no price data, so these tests skip
there and every other test runs.
"""

import pytest

from series import SERIES

NEEDS_FROZEN_WTI = (
    "test_series.py::WtiSeriesTests::",
    "test_series.py::RollDateTests::",
    "test_series.py::AsinhSeriesTests::test_observation_matches_the_registered_formula",
    "test_series.py::GenericSeriesTests::test_wrong_observation_count_is_refused",
    "test_return_comparison.py::",
)


def pytest_collection_modifyitems(config, items):
    if SERIES["wti"].frozen_file.exists():
        return
    skip = pytest.mark.skip(reason="frozen WTI file missing: run download_data.py")
    for item in items:
        if any(marker in item.nodeid for marker in NEEDS_FROZEN_WTI):
            item.add_marker(skip)
