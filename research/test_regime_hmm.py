"""Gate G1 and G2 checks for the K-state Gaussian and Student-t HMM."""

import unittest

import numpy as np

from gaussian_hmm import GaussianHMM2State
from multi_start import fit_multiple_starts_general
from regime_hmm import RegimeHMM, brute_force_inference

try:
    import hmmlearn.hmm as hmmlearn_hmm
except ImportError:  # pragma: no cover - depends on the environment
    hmmlearn_hmm = None


def make_truth(n_states, emission, seed):
    rng = np.random.default_rng(seed)
    model = RegimeHMM(n_states=n_states, emission=emission)
    stay = np.linspace(0.97, 0.90, n_states)
    A = np.empty((n_states, n_states))
    for state in range(n_states):
        A[state] = (1 - stay[state]) / (n_states - 1)
        A[state, state] = stay[state]
    pi = rng.dirichlet(np.ones(n_states))
    mu = np.linspace(0.0, -0.3, n_states)
    sigma = np.array([0.5, 2.0, 4.5][:n_states])
    nu = np.full(n_states, 5.0) if emission == "student_t" else None
    return model.set_parameters(pi, A, mu, sigma, nu)


class BruteForceTests(unittest.TestCase):
    """G1.1: exhaustive enumeration on a six-observation sequence."""

    def check_against_brute_force(self, model, observations):
        likelihood, posterior, path = brute_force_inference(model, observations)
        np.testing.assert_allclose(
            np.exp(model.score(observations)), likelihood, rtol=1e-8
        )
        np.testing.assert_allclose(
            model.predict_proba(observations), posterior, atol=1e-8
        )
        np.testing.assert_array_equal(model.decode(observations), path)
        # Filtered probabilities must match a brute-force forward pass too.
        filtered = model.filter_proba(observations)
        for time in range(len(observations)):
            _, partial_posterior, _ = brute_force_inference(
                model, observations[: time + 1]
            )
            np.testing.assert_allclose(
                filtered[time], partial_posterior[time], atol=1e-8
            )

    def test_two_state_gaussian(self):
        model = make_truth(2, "gaussian", 1)
        observations = np.array([0.1, -3.0, 2.5, 0.2, -0.1, 4.0])
        self.check_against_brute_force(model, observations)

    def test_three_state_student_t(self):
        model = make_truth(3, "student_t", 2)
        observations = np.array([0.1, -3.0, 12.5, 0.2, -0.1, 4.0])
        self.check_against_brute_force(model, observations)

    def test_independent_mixture(self):
        model = RegimeHMM(n_states=2, markov=False).set_parameters(
            [0.7, 0.3], [[0.7, 0.3], [0.7, 0.3]], [0.0, -0.2], [0.5, 2.0]
        )
        observations = np.array([0.1, -3.0, 2.5, 0.2, -0.1, 4.0])
        self.check_against_brute_force(model, observations)


class EquivalenceTests(unittest.TestCase):
    """The general model must reproduce the validated two-state class."""

    @classmethod
    def setUpClass(cls):
        truth = make_truth(2, "gaussian", 3)
        cls.observations, _ = truth.simulate(1500, random_state=7)

    def test_deterministic_start_matches(self):
        old = GaussianHMM2State(n_iter=200, tol=1e-7).fit(self.observations)
        new = RegimeHMM(n_states=2, n_iter=200, tol=1e-7).fit(self.observations)
        np.testing.assert_allclose(old.mu, new.mu, atol=1e-10)
        np.testing.assert_allclose(old.sigma, new.sigma, atol=1e-10)
        np.testing.assert_allclose(old.A, new.A, atol=1e-10)
        self.assertEqual(old.n_iter_, new.n_iter_)
        np.testing.assert_array_equal(
            old.decode(self.observations), new.decode(self.observations)
        )

    def test_random_start_matches(self):
        old = GaussianHMM2State(
            n_iter=200, tol=1e-7, init_strategy="random", random_state=11
        ).fit(self.observations)
        new = RegimeHMM(
            n_states=2, n_iter=200, tol=1e-7, init_strategy="random", random_state=11
        ).fit(self.observations)
        np.testing.assert_allclose(old.mu, new.mu, atol=1e-10)
        np.testing.assert_allclose(old.sigma, new.sigma, atol=1e-10)
        np.testing.assert_allclose(old.A, new.A, atol=1e-10)


class NumericalTests(unittest.TestCase):
    """G1.2 and G1.3 on every emission family."""

    def check_model(self, n_states, emission):
        truth = make_truth(n_states, emission, 5)
        observations, _ = truth.simulate(2000, random_state=9)
        model = RegimeHMM(
            n_states=n_states, emission=emission, n_iter=300, tol=1e-7
        ).fit(observations)
        changes = np.diff(model.log_likelihood_history_)
        self.assertGreaterEqual(changes.min(), -1e-7)
        posterior = model.predict_proba(observations)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)
        filtered = model.filter_proba(observations)
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(np.diff(model.sigma) >= 0))
        self.assertEqual(model.n_parameters, (n_states - 1) + n_states * (n_states - 1)
                         + (3 if emission == "student_t" else 2) * n_states)

    def test_gaussian_two_states(self):
        self.check_model(2, "gaussian")

    def test_gaussian_three_states(self):
        self.check_model(3, "gaussian")

    def test_student_t_two_states(self):
        self.check_model(2, "student_t")

    def test_student_t_three_states(self):
        self.check_model(3, "student_t")

    def test_conditional_score_splits_the_sequence(self):
        truth = make_truth(2, "gaussian", 6)
        observations, _ = truth.simulate(600, random_state=3)
        history, future = observations[:400], observations[400:]
        total = truth.score(observations)
        self.assertAlmostEqual(
            truth.score(history) + truth.conditional_score(history, future),
            total,
            places=9,
        )

    def test_mixture_has_no_persistence(self):
        truth = make_truth(2, "gaussian", 8)
        observations, _ = truth.simulate(2000, random_state=4)
        mixture = RegimeHMM(n_states=2, markov=False, n_iter=300).fit(observations)
        np.testing.assert_allclose(mixture.A[0], mixture.A[1], atol=1e-12)
        np.testing.assert_allclose(mixture.pi, mixture.A[0], atol=1e-12)
        markov = RegimeHMM(n_states=2, n_iter=300).fit(observations)
        self.assertGreater(markov.score(observations), mixture.score(observations))


class RecoveryTests(unittest.TestCase):
    """G2: parameter recovery on 5,000 simulated observations, three seeds.

    The estimator is the multi-start procedure used on real data: one
    deterministic start plus random starts, keeping the best likelihood. A
    single deterministic start reaches a local optimum for three states.
    """

    SEEDS = (101, 202, 303)
    N_STARTS = 4

    def check_recovery(self, n_states, emission):
        truth = make_truth(n_states, emission, 12)
        for seed in self.SEEDS:
            observations, _ = truth.simulate(5000, random_state=seed)
            model, _, _ = fit_multiple_starts_general(
                observations,
                lambda strategy, state: RegimeHMM(
                    n_states=n_states,
                    emission=emission,
                    n_iter=500,
                    tol=1e-7,
                    init_strategy=strategy,
                    random_state=state,
                ),
                n_starts=self.N_STARTS,
                base_seed=seed,
            )
            scale_tolerance = 0.20 if emission == "student_t" else 0.15
            np.testing.assert_allclose(
                model.sigma, truth.sigma, rtol=scale_tolerance,
                err_msg=f"scale, seed {seed}",
            )
            np.testing.assert_allclose(
                np.diag(model.A), np.diag(truth.A), atol=0.04,
                err_msg=f"persistence, seed {seed}",
            )
            self.assertTrue(
                np.all(np.abs(model.mu - truth.mu) <= 0.25 * truth.sigma),
                f"mean, seed {seed}: {model.mu} vs {truth.mu}",
            )
            if emission == "student_t":
                ratio = model.nu / truth.nu
                self.assertTrue(
                    np.all((ratio >= 1 / 2.0) & (ratio <= 2.0)),
                    f"nu, seed {seed}: {model.nu}",
                )

    def test_gaussian_two_states(self):
        self.check_recovery(2, "gaussian")

    def test_gaussian_three_states(self):
        self.check_recovery(3, "gaussian")

    def test_student_t_two_states(self):
        self.check_recovery(2, "student_t")

    def test_student_t_three_states(self):
        self.check_recovery(3, "student_t")


@unittest.skipUnless(hmmlearn_hmm is not None, "hmmlearn is not installed")
class LibraryCrossCheckTests(unittest.TestCase):
    """G1.4: identical parameters must give identical inference in hmmlearn."""

    def test_gaussian_inference_matches_hmmlearn(self):
        truth = make_truth(3, "gaussian", 21)
        observations, _ = truth.simulate(1000, random_state=5)
        library = hmmlearn_hmm.GaussianHMM(
            n_components=3, covariance_type="diag", init_params="", params=""
        )
        library.startprob_ = truth.pi
        library.transmat_ = truth.A
        library.means_ = truth.mu[:, None]
        library.covars_ = (truth.sigma**2)[:, None]
        column = observations[:, None]
        np.testing.assert_allclose(
            truth.score(observations), library.score(column), rtol=1e-6
        )
        np.testing.assert_allclose(
            truth.predict_proba(observations),
            library.predict_proba(column),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            truth.decode(observations), library.predict(column)
        )


if __name__ == "__main__":
    unittest.main()
