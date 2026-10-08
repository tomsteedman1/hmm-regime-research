"""Automated checks for the WTI input decision."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from gaussian_hmm import GaussianHMM2State


PROJECT_DIR = Path(__file__).resolve().parent
MODULE_PATH = PROJECT_DIR / "01_return_comparison.py"
SPEC = importlib.util.spec_from_file_location("return_comparison", MODULE_PATH)
RETURN_COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETURN_COMPARISON)


class ReturnComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.price = RETURN_COMPARISON.load_price()
        cls.transformations = RETURN_COMPARISON.build_transformations(cls.price)

    def test_primary_input_acceptance_values(self):
        RETURN_COMPARISON.validate_primary_input(self.transformations)

    def test_log_return_break_is_explicit(self):
        log_return = self.transformations["log_return_valid"]
        self.assertTrue(np.isnan(log_return.loc["2020-04-20"]))
        self.assertTrue(np.isnan(log_return.loc["2020-04-21"]))

    def test_arcsinh_returns_are_complete_after_first_date(self):
        for scale in RETURN_COMPARISON.ASINH_SCALES:
            series = self.transformations[f"asinh_return_{int(scale)}"]
            self.assertEqual(series.isna().sum(), 1)
            self.assertTrue(np.isfinite(series.dropna().to_numpy()).all())

    def test_extracted_hmm_preserves_baseline(self):
        values = self.transformations["log_return_valid"].dropna().to_numpy()
        model = GaussianHMM2State(n_iter=200, tol=1e-5).fit(values)
        np.testing.assert_allclose(
            model.mu,
            [0.00043215, -0.00173935],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            model.sigma,
            [0.01713422, 0.05372713],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.diag(model.A),
            [0.98619569, 0.91687629],
            atol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
