"""Validation checks for the reusable two-state Gaussian HMM."""

import unittest

import numpy as np

from gaussian_hmm import GaussianHMM2State
from multi_start import fit_multiple_starts


def simulate_hmm(length, pi, transition, means, sigmas, seed=42):
    """Simulate observations and hidden states from known parameters."""
    rng = np.random.default_rng(seed)
    states = np.zeros(length, dtype=int)
    observations = np.zeros(length)
    states[0] = rng.choice(2, p=pi)
    observations[0] = rng.normal(means[states[0]], sigmas[states[0]])
    for time in range(1, length):
        states[time] = rng.choice(2, p=transition[states[time - 1]])
        observations[time] = rng.normal(
            means[states[time]],
            sigmas[states[time]],
        )
    return observations, states


class GaussianHMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.true_pi = np.array([0.8, 0.2])
        cls.true_transition = np.array([[0.97, 0.03], [0.08, 0.92]])
        cls.true_means = np.array([0.0, -0.25])
        cls.true_sigmas = np.array([0.5, 2.0])
        cls.observations, cls.states = simulate_hmm(
            2500,
            cls.true_pi,
            cls.true_transition,
            cls.true_means,
            cls.true_sigmas,
        )
        cls.model = GaussianHMM2State(n_iter=300, tol=1e-7).fit(
            cls.observations
        )

    def test_likelihood_does_not_decrease(self):
        changes = np.diff(self.model.log_likelihood_history_)
        self.assertGreaterEqual(changes.min(), -1e-7)

    def test_posterior_rows_sum_to_one(self):
        posterior = self.model.predict_proba(self.observations)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(((posterior >= 0) & (posterior <= 1)).all())

    def test_viterbi_path_has_valid_states(self):
        decoded = self.model.decode(self.observations)
        self.assertEqual(len(decoded), len(self.observations))
        self.assertEqual(set(np.unique(decoded)), {0, 1})

    def test_recovers_simulated_volatility_and_persistence(self):
        np.testing.assert_allclose(
            self.model.sigma,
            self.true_sigmas,
            rtol=0.15,
        )
        np.testing.assert_allclose(
            np.diag(self.model.A),
            np.diag(self.true_transition),
            atol=0.04,
        )

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            GaussianHMM2State().fit(np.array([0.0, np.nan]))
        with self.assertRaises(ValueError):
            GaussianHMM2State().fit(np.array([[0.0, 1.0]]))

    def test_random_start_is_reproducible(self):
        first = GaussianHMM2State(
            n_iter=100,
            tol=1e-6,
            init_strategy="random",
            random_state=91,
        ).fit(self.observations[:500])
        second = GaussianHMM2State(
            n_iter=100,
            tol=1e-6,
            init_strategy="random",
            random_state=91,
        ).fit(self.observations[:500])
        np.testing.assert_allclose(first.mu, second.mu)
        np.testing.assert_allclose(first.sigma, second.sigma)
        np.testing.assert_allclose(first.A, second.A)

    def test_multiple_starts_select_highest_likelihood(self):
        best, records, _ = fit_multiple_starts(
            self.observations[:800],
            n_starts=4,
            base_seed=117,
            n_iter=150,
            tol=1e-6,
        )
        scores = [record["log_likelihood"] for record in records]
        selected = [record for record in records if record["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(best.score(self.observations[:800]), max(scores))
        self.assertGreaterEqual(max(scores), scores[0])


if __name__ == "__main__":
    unittest.main()
