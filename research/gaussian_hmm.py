"""Reusable two-state Gaussian Hidden Markov Model."""

import numpy as np
from scipy.stats import norm


class GaussianHMM2State:
    """Two-state Gaussian HMM trained with the Baum-Welch algorithm."""

    def __init__(
        self,
        n_iter=200,
        tol=1e-5,
        init_strategy="deterministic",
        random_state=None,
    ):
        self.n_iter = n_iter
        self.tol = tol
        self.init_strategy = init_strategy
        self.random_state = random_state

    def _init(self, observations):
        if self.init_strategy not in {"deterministic", "random"}:
            raise ValueError("init_strategy must be deterministic or random")
        if self.init_strategy == "random":
            self._init_random(observations)
            return

        self.pi = np.array([0.5, 0.5])
        self.A = np.array([[0.95, 0.05], [0.05, 0.95]])

        ordered = np.argsort(np.abs(observations))
        half = len(observations) // 2
        calm = observations[ordered[:half]]
        volatile = observations[ordered[half:]]
        self.mu = np.array([calm.mean(), volatile.mean()])
        self.sigma = np.array(
            [calm.std() + 1e-6, volatile.std() + 1e-6]
        )

    def _init_random(self, observations):
        """Create reproducible valid parameters for a random start."""
        rng = np.random.default_rng(self.random_state)
        self.pi = rng.dirichlet(np.ones(2))

        persistence = rng.uniform(0.80, 0.99, size=2)
        self.A = np.array(
            [
                [persistence[0], 1 - persistence[0]],
                [1 - persistence[1], persistence[1]],
            ]
        )

        quantiles = rng.uniform(0.10, 0.90, size=2)
        self.mu = np.quantile(observations, quantiles)
        overall_sigma = max(observations.std(), 1e-4)
        multipliers = np.exp(rng.uniform(np.log(0.35), np.log(2.5), size=2))
        self.sigma = overall_sigma * multipliers

    def _emission_probabilities(self, observations):
        probabilities = np.zeros((len(observations), 2))
        for state in range(2):
            probabilities[:, state] = norm.pdf(
                observations,
                self.mu[state],
                self.sigma[state],
            )
        return np.clip(probabilities, 1e-300, None)

    def _forward(self, probabilities):
        length = len(probabilities)
        alpha = np.zeros((length, 2))
        scales = np.zeros(length)

        alpha[0] = self.pi * probabilities[0]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]

        for time in range(1, length):
            alpha[time] = (alpha[time - 1] @ self.A) * probabilities[time]
            scales[time] = alpha[time].sum() + 1e-300
            alpha[time] /= scales[time]
        return alpha, scales

    def _backward(self, probabilities, scales):
        length = len(probabilities)
        beta = np.zeros((length, 2))
        beta[-1] = 1.0

        for time in range(length - 2, -1, -1):
            beta[time] = self.A @ (
                probabilities[time + 1] * beta[time + 1]
            )
            beta[time] /= scales[time + 1]
        return beta

    def fit(self, observations):
        """Estimate model parameters from a one-dimensional NumPy array."""
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 1 or len(observations) < 2:
            raise ValueError("observations must be a one-dimensional array")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")

        self._init(observations)
        previous_log_likelihood = -np.inf
        self.log_likelihood_history_ = []

        for iteration in range(self.n_iter):
            probabilities = self._emission_probabilities(observations)
            alpha, scales = self._forward(probabilities)
            beta = self._backward(probabilities, scales)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Expected transition counts, summed over time. With scaled
            # alpha and beta, each date's joint state probabilities are
            # alpha[t] A (b[t+1] * beta[t+1]) / scales[t+1], and that
            # denominator equals the sum over all state pairs, so this
            # single matrix product replaces the per-date loop.
            weighted_next = (
                probabilities[1:] * beta[1:]
            ) / (scales[1:, None] + 1e-300)
            expected_transitions = self.A * (alpha[:-1].T @ weighted_next)

            self.pi = gamma[0]
            self.A = expected_transitions / (
                expected_transitions.sum(axis=1, keepdims=True) + 1e-300
            )
            for state in range(2):
                weights = gamma[:, state]
                weight_sum = weights.sum() + 1e-300
                self.mu[state] = (weights * observations).sum() / weight_sum
                self.sigma[state] = np.sqrt(
                    (
                        weights * (observations - self.mu[state]) ** 2
                    ).sum()
                    / weight_sum
                    + 1e-8
                )

            log_likelihood = np.log(scales + 1e-300).sum()
            self.log_likelihood_history_.append(log_likelihood)
            if abs(log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = log_likelihood

        self.n_iter_ = iteration + 1
        self.log_likelihood_ = log_likelihood

        if self.sigma[0] > self.sigma[1]:
            self._swap_states()
        return self

    def score(self, observations):
        """Return the log-likelihood of observations under the fitted model."""
        observations = self._check(observations)
        probabilities = self._emission_probabilities(observations)
        _, scales = self._forward(probabilities)
        return np.log(scales + 1e-300).sum()

    def conditional_score(self, history, observations):
        """Return log P(observations | history) under the fitted model."""
        history = self._check(history)
        observations = self._check(observations)
        joint = self.score(np.concatenate([history, observations]))
        return joint - self.score(history)

    def filter_proba(self, observations):
        """Return filtered state probabilities P(z_t | x_1, ..., x_t)."""
        observations = self._check(observations)
        probabilities = self._emission_probabilities(observations)
        alpha, _ = self._forward(probabilities)
        return alpha

    def predictive_proba(self, observations):
        """Return one-step-ahead state probabilities P(z_t | x_1, ..., x_{t-1})."""
        filtered = self.filter_proba(observations)
        predictive = np.empty_like(filtered)
        predictive[0] = self.pi
        predictive[1:] = filtered[:-1] @ self.A
        return predictive

    @staticmethod
    def _check(observations):
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 1 or len(observations) < 1:
            raise ValueError("observations must be a one-dimensional array")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")
        return observations

    def predict_proba(self, observations):
        """Return smoothed state probabilities from forward-backward inference."""
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 1 or len(observations) < 1:
            raise ValueError("observations must be a one-dimensional array")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")
        probabilities = self._emission_probabilities(observations)
        alpha, scales = self._forward(probabilities)
        beta = self._backward(probabilities, scales)
        posterior = alpha * beta
        posterior /= posterior.sum(axis=1, keepdims=True) + 1e-300
        return posterior

    def _swap_states(self):
        self.pi = self.pi[::-1]
        self.mu = self.mu[::-1]
        self.sigma = self.sigma[::-1]
        self.A = self.A[::-1, ::-1]

    def decode(self, observations):
        """Return the most likely state path from the Viterbi algorithm."""
        observations = np.asarray(observations, dtype=float)
        probabilities = self._emission_probabilities(observations)
        length = len(observations)
        delta = np.zeros((length, 2))
        previous_state = np.zeros((length, 2), dtype=int)

        delta[0] = np.log(self.pi + 1e-300) + np.log(probabilities[0])
        for time in range(1, length):
            for state in range(2):
                transition = delta[time - 1] + np.log(
                    self.A[:, state] + 1e-300
                )
                previous_state[time, state] = np.argmax(transition)
                delta[time, state] = (
                    transition[previous_state[time, state]]
                    + np.log(probabilities[time, state])
                )

        states = np.zeros(length, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for time in range(length - 2, -1, -1):
            states[time] = previous_state[time + 1, states[time + 1]]
        return states
