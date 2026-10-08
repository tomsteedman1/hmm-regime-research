"""Tests for the registered input series."""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

import wti_data
from scipy import integrate, stats

from roll_dates import cl_expiry_dates, cl_roll_days
from series import SERIES, asinh_log_jacobian, asinh_return, get_series


class WtiSeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = get_series("wti")
        cls.changes = cls.series.load_observations()

    def test_observation_count_and_april_2020(self):
        self.assertEqual(len(self.changes), 3770)
        self.assertEqual(self.changes.name, "price_change_usd")
        self.assertAlmostEqual(self.changes.loc["2020-04-20"], -55.900002, places=5)
        self.assertAlmostEqual(self.changes.loc["2020-04-21"], 47.640001, places=5)

    def test_stability_samples(self):
        samples = self.series.stability_samples(self.changes)
        self.assertEqual(
            list(samples),
            ["full_2010_2024", "early_2010_2016", "late_2017_2024", "full_without_cross_zero_days"],
        )
        self.assertEqual(samples["early_2010_2016"].index.max(), pd.Timestamp("2016-12-30"))
        self.assertEqual(samples["late_2017_2024"].index.min().year, 2017)
        without = samples["full_without_cross_zero_days"]
        self.assertEqual(len(without), 3768)
        self.assertFalse(without.index.isin(self.series.excluded_dates).any())

    def test_train_test_split(self):
        train, test = self.series.train_test_split(self.changes)
        self.assertEqual(train.index.max().year, 2018)
        self.assertEqual(test.index.min(), pd.Timestamp("2019-01-02"))
        self.assertEqual(len(train) + len(test), len(self.changes))

    def test_wti_data_names_still_work(self):
        pd.testing.assert_series_equal(wti_data.load_changes(), self.changes)
        self.assertEqual(wti_data.EXPECTED_OBSERVATIONS, 3770)
        self.assertEqual(len(wti_data.CROSS_ZERO_DATES), 2)
        self.assertEqual(
            list(wti_data.stability_samples(self.changes)),
            list(self.series.stability_samples(self.changes)),
        )


class RollDateTests(unittest.TestCase):
    """The ported CL rule must give the roll audit's 180 expiries (2 Sep 2026)."""

    @classmethod
    def setUpClass(cls):
        cls.days = get_series("wti").load_observations().index

    def test_expiries_match_the_roll_audit(self):
        expiries = cl_expiry_dates(self.days)
        self.assertEqual(len(expiries), 180)
        self.assertEqual(
            list(expiries[:2].strftime("%Y-%m-%d")) + list(expiries[-2:].strftime("%Y-%m-%d")),
            ["2010-01-20", "2010-02-22", "2024-11-20", "2024-12-19"],
        )

    def test_roll_day_is_the_next_trading_day(self):
        expiries = cl_expiry_dates(self.days)
        rolls = cl_roll_days(self.days)
        self.assertEqual(len(rolls), 180)
        for expiry, roll in zip(expiries, rolls):
            self.assertEqual(self.days.get_loc(roll), self.days.get_loc(expiry) + 1)

    def test_record_sample_drops_the_roll_days(self):
        series = get_series("wti")
        values = series.load_observations()
        record = series.record_samples(values)
        self.assertEqual(list(record), ["full_without_roll_days"])
        self.assertEqual(len(record["full_without_roll_days"]), 3770 - 180)


class AsinhSeriesTests(unittest.TestCase):
    def test_observation_matches_the_registered_formula(self):
        series = get_series("wti_asinh10")
        price = series.load_prices()
        values = series.load_observations()
        self.assertEqual(len(values), 3770)
        expected = 100 * (np.arcsinh(price / 10) - np.arcsinh(price.shift(1) / 10))
        np.testing.assert_allclose(values, expected.dropna(), rtol=1e-12)
        # Defined through the negative price.
        self.assertTrue(np.isfinite(values.loc["2020-04-20"]))

    def test_jacobian_matches_a_numerical_derivative(self):
        for previous in (-30.0, 5.0, 60.0):
            for move in (-2.0, 0.5, 3.0):
                price = previous + move
                step = 1e-6
                y = lambda x: asinh_return(pd.Series([previous, previous + x])).iloc[1]
                numerical = (y(move + step) - y(move - step)) / (2 * step)
                self.assertAlmostEqual(
                    np.log(numerical), asinh_log_jacobian(pd.Series([price])).iloc[0], places=7
                )

    def test_corrected_density_integrates_to_one_in_dollars(self):
        """A density on y, carried to x by the Jacobian, is a density on x."""
        density_y = stats.t(df=4.0, loc=0.05, scale=2.0)
        for previous in (-37.6, 18.0, 85.0):
            def density_x(x):
                price = pd.Series([previous, previous + x])
                y = asinh_return(price).iloc[1]
                return density_y.pdf(y) * np.exp(asinh_log_jacobian(price).iloc[1])

            total, _ = integrate.quad(density_x, -np.inf, np.inf, limit=200)
            self.assertAlmostEqual(total, 1.0, places=6)


class GenericSeriesTests(unittest.TestCase):
    def test_no_excluded_dates_gives_three_samples(self):
        series = replace(SERIES["wti"], excluded_name="", excluded_key="")
        values = pd.Series(
            np.arange(10.0), index=pd.bdate_range("2016-12-26", periods=10)
        )
        samples = series.stability_samples(values)
        self.assertEqual(list(samples), ["full_2010_2024", "early_2010_2016", "late_2017_2024"])
        self.assertEqual(len(samples["early_2010_2016"]) + len(samples["late_2017_2024"]), 10)

    def test_wrong_observation_count_is_refused(self):
        series = replace(SERIES["wti"], expected_observations=3769)
        with self.assertRaises(ValueError):
            series.load_observations()

    def test_unknown_series_is_refused(self):
        with self.assertRaises(SystemExit):
            get_series("no_such_series")


if __name__ == "__main__":
    unittest.main()
