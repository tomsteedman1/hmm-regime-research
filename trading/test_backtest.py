"""Tests for the backtest engine and the walk-forward signal."""

import unittest

import numpy as np
import pandas as pd

import research_link  # noqa: F401
from regime_hmm import RegimeHMM

import backtest
from walk_forward import walk_forward_expected_vol


def t2_model():
    model = RegimeHMM(n_states=2, emission="student_t")
    model.set_parameters(
        pi=np.array([0.72, 0.28]),
        A=np.array([[0.9916, 0.0084], [0.0185, 0.9815]]),
        mu=np.array([0.051, -0.025]),
        sigma=np.array([1.017, 1.739]),
        nu=np.array([14.5, 4.25]),
    )
    return model


class SignalTests(unittest.TestCase):
    def test_hmm_signal_has_no_look_ahead(self):
        model = t2_model()
        obs, _ = model.simulate(600, random_state=1)
        short = backtest.hmm_expected_vol(model, obs[:400])
        long = backtest.hmm_expected_vol(model, obs)
        np.testing.assert_allclose(short, long[:400], rtol=1e-12)

    def test_realised_signal_has_no_look_ahead_and_lags_one_day(self):
        rng = np.random.default_rng(2)
        obs = rng.normal(size=300)
        short = backtest.realised_expected_vol(obs[:200])
        long = backtest.realised_expected_vol(obs)
        self.assertTrue(np.isnan(long[0]))
        np.testing.assert_allclose(short[1:], long[1:200], rtol=1e-12)
        # Changing the last observation must not change the last signal.
        bumped = obs.copy()
        bumped[-1] += 100.0
        np.testing.assert_allclose(
            backtest.realised_expected_vol(bumped)[-1], long[-1], rtol=1e-12
        )

    def test_synthetic_regimes_hit_target_and_cut_size_in_high_state(self):
        model = t2_model()
        obs, states = model.simulate(20000, random_state=3)
        expected = backtest.hmm_expected_vol(model, obs)
        position = backtest.vol_target_positions(expected, target_vol=1.0, max_leverage=3.0)
        pnl = backtest.daily_pnl(obs, position, cost_per_unit=0.0)
        realised = pnl["net"].std(ddof=1)
        # Student-t tails with nu=4.25 make the realised standard deviation
        # noisy; a 20% band is the check, not a claim of precision.
        self.assertLess(abs(realised - 1.0) / 1.0, 0.20)
        self.assertLess(position[states == 1].mean(), position[states == 0].mean())

    def test_walk_forward_uses_only_past_parameters(self):
        model = t2_model()
        obs, _ = model.simulate(1500, random_state=4)
        index = pd.bdate_range("2010-01-04", periods=1500)
        changes = pd.Series(obs, index=index)
        calls = []

        def fitter(history):
            calls.append(len(history))
            return t2_model()

        expected = walk_forward_expected_vol(
            changes, fitter, first_year=2012, refit_month=1
        )
        # The fitter sees only history that ends before each refit year.
        self.assertEqual(len(calls), len({year for year in index.year if year >= 2012}))
        for n, year in zip(calls, sorted({y for y in index.year if y >= 2012})):
            self.assertEqual(index[n - 1].year, year - 1)
        # Before the first year there is no signal.
        self.assertTrue(np.isnan(expected.loc[: "2011-12-31"]).all())
        self.assertTrue(np.isfinite(expected.loc["2012-01-01":]).all())
        # With fixed parameters the walk-forward signal equals the static one.
        static = backtest.hmm_expected_vol(model, obs)
        mask = index.year >= 2012
        np.testing.assert_allclose(expected.to_numpy()[mask], static[mask], rtol=1e-12)


class EngineTests(unittest.TestCase):
    def test_cost_arithmetic_on_a_three_day_example(self):
        changes = np.array([1.0, -2.0, 0.5])
        position = np.array([1.0, 2.0, 0.0])
        pnl = backtest.daily_pnl(changes, position, cost_per_unit=0.01)
        np.testing.assert_allclose(pnl["turnover"], [1.0, 1.0, 2.0])
        np.testing.assert_allclose(pnl["gross"], [1.0, -4.0, 0.0])
        np.testing.assert_allclose(pnl["net"], [0.99, -4.01, -0.02])

    def test_roll_treatments(self):
        changes = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, True, False])
        np.testing.assert_allclose(backtest.apply_roll_treatment(changes, mask, "raw"), changes)
        np.testing.assert_allclose(backtest.apply_roll_treatment(changes, mask, "zeroed"), [1.0, 0.0, 3.0])
        dropped = backtest.apply_roll_treatment(changes, mask, "dropped")
        self.assertTrue(np.isnan(dropped[1]))
        pnl = backtest.daily_pnl(dropped, np.ones(3), cost_per_unit=0.0)
        self.assertEqual(backtest.metrics(pnl)["days"], 2)

    def test_drawdown_and_sharpe(self):
        net = np.array([1.0, -3.0, 1.0, 1.0])
        self.assertAlmostEqual(backtest.max_drawdown(net), -3.0)
        self.assertAlmostEqual(
            backtest.sharpe(net), net.mean() / net.std(ddof=1) * np.sqrt(252)
        )

    def test_bootstrap_is_reproducible_and_neutral_on_identical_series(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=500)
        first = backtest.sharpe_difference_bootstrap(a, a.copy(), n_draws=200)
        second = backtest.sharpe_difference_bootstrap(a, a.copy(), n_draws=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed"], 0.0)
        self.assertEqual(first["p_value"], 1.0)  # every draw difference is exactly zero


if __name__ == "__main__":
    unittest.main()
