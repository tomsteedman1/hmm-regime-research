"""K-state Hidden Markov Model with Gaussian or Student-t emissions.

The two-state Gaussian case reproduces ``gaussian_hmm.GaussianHMM2State``
exactly: the same deterministic and random initialisation, the same scaled
forward-backward recursions and the same Baum-Welch updates. This module adds
three or more states, Student-t emissions, an independent-mixture option with
no Markov dependence, information criteria and simulation.

Student-t estimation uses the scale-mixture representation. Given the current
parameters, each observation receives a latent precision weight
``u = (nu + 1) / (nu + z^2)``. The location and scale updates are weighted
closed forms and the degrees of freedom are then maximised numerically. Each
step increases the expected complete-data log-likelihood, so the observed
log-likelihood does not decrease.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.stats import t as student_t


EMISSIONS = ("gaussian", "student_t")
NU_MIN = 2.1
NU_MAX = 200.0
NU_INITIAL = 8.0


class RegimeHMM:
    """Hidden Markov Model with K states and Gaussian or Student-t emissions."""

    def __init__(
        self,
        n_states=2,
        emission="gaussian",
        n_iter=300,
        tol=1e-7,
        init_strategy="deterministic",
        random_state=None,
        markov=True,
    ):
        if n_states < 2:
            raise ValueError("n_states must be at least 2")
        if emission not in EMISSIONS:
            raise ValueError(f"emission must be one of {EMISSIONS}")
        if init_strategy not in {"deterministic", "random"}:
            raise ValueError("init_strategy must be deterministic or random")
        self.n_states = int(n_states)
        self.emission = emission
        self.n_iter = n_iter
        self.tol = tol
        self.init_strategy = init_strategy
        self.random_state = random_state
        self.markov = markov
        self.nu = None

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def stdev(self):
        """Return the standard deviation implied by each state's emission."""
        if self.emission == "gaussian":
            return self.sigma.copy()
        return self.sigma * np.sqrt(self.nu / (self.nu - 2.0))

    @property
    def n_parameters(self):
        """Count free parameters as registered in GATES.md."""
        k = self.n_states
        count = (k - 1) + 2 * k
        if self.markov:
            count += k * (k - 1)
        if self.emission == "student_t":
            count += k
        return count

    def aic(self, observations):
        return 2 * self.n_parameters - 2 * self.score(observations)

    def bic(self, observations):
        observations = self._check(observations)
        return (
            self.n_parameters * np.log(len(observations))
            - 2 * self.score(observations)
        )

    def set_parameters(self, pi, A, mu, sigma, nu=None):
        """Set parameters directly, for tests and for library cross-checks."""
        self.pi = np.asarray(pi, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        if self.emission == "student_t":
            if nu is None:
                raise ValueError("Student-t emissions need nu")
            self.nu = np.asarray(nu, dtype=float)
        return self

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self, observations):
        k = self.n_states
        if self.init_strategy == "random":
            self._init_random(observations)
            return

        self.pi = np.full(k, 1.0 / k)
        off = 0.05 / (k - 1)
        self.A = np.full((k, k), off)
        np.fill_diagonal(self.A, 0.95)

        ordered = np.argsort(np.abs(observations))
        length = len(observations)
        bounds = [i * length // k for i in range(k + 1)]
        groups = [
            observations[ordered[bounds[i]:bounds[i + 1]]] for i in range(k)
        ]
        self.mu = np.array([group.mean() for group in groups])
        self.sigma = np.array([group.std() + 1e-6 for group in groups])
        if self.emission == "student_t":
            self.nu = np.full(k, NU_INITIAL)

    def _init_random(self, observations):
        """Create reproducible valid parameters for a random start.

        For two Gaussian states the draws match ``GaussianHMM2State`` exactly,
        so the same seed gives the same start in both implementations.
        """
        k = self.n_states
        rng = np.random.default_rng(self.random_state)
        self.pi = rng.dirichlet(np.ones(k))

        persistence = rng.uniform(0.80, 0.99, size=k)
        self.A = np.empty((k, k))
        for state in range(k):
            self.A[state] = (1 - persistence[state]) / (k - 1)
            self.A[state, state] = persistence[state]

        quantiles = rng.uniform(0.10, 0.90, size=k)
        self.mu = np.quantile(observations, quantiles)
        overall_sigma = max(observations.std(), 1e-4)
        multipliers = np.exp(rng.uniform(np.log(0.35), np.log(2.5), size=k))
        self.sigma = overall_sigma * multipliers
        if self.emission == "student_t":
            self.nu = np.exp(rng.uniform(np.log(3.0), np.log(30.0), size=k))

    # ------------------------------------------------------------------
    # Emission densities
    # ------------------------------------------------------------------

    def _emission_probabilities(self, observations):
        k = self.n_states
        probabilities = np.zeros((len(observations), k))
        for state in range(k):
            if self.emission == "gaussian":
                probabilities[:, state] = norm.pdf(
                    observations, self.mu[state], self.sigma[state]
                )
            else:
                probabilities[:, state] = student_t.pdf(
                    observations,
                    self.nu[state],
                    loc=self.mu[state],
                    scale=self.sigma[state],
                )
        return np.clip(probabilities, 1e-300, None)

    def emission_cdf(self, observations):
        """Return the emission CDF of each observation under each state."""
        observations = self._check(observations)
        k = self.n_states
        cdf = np.zeros((len(observations), k))
        for state in range(k):
            if self.emission == "gaussian":
                cdf[:, state] = norm.cdf(
                    observations, self.mu[state], self.sigma[state]
                )
            else:
                cdf[:, state] = student_t.cdf(
                    observations,
                    self.nu[state],
                    loc=self.mu[state],
                    scale=self.sigma[state],
                )
        return cdf

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward(self, probabilities):
        length = len(probabilities)
        alpha = np.zeros((length, self.n_states))
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
        beta = np.zeros((length, self.n_states))
        beta[-1] = 1.0

        for time in range(length - 2, -1, -1):
            beta[time] = self.A @ (probabilities[time + 1] * beta[time + 1])
            beta[time] /= scales[time + 1]
        return beta

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

    def predict_proba(self, observations):
        """Return smoothed state probabilities P(z_t | x_1, ..., x_T)."""
        observations = self._check(observations)
        probabilities = self._emission_probabilities(observations)
        alpha, scales = self._forward(probabilities)
        beta = self._backward(probabilities, scales)
        posterior = alpha * beta
        posterior /= posterior.sum(axis=1, keepdims=True) + 1e-300
        return posterior

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

    def decode(self, observations):
        """Return the most likely state path from the Viterbi algorithm."""
        observations = self._check(observations)
        probabilities = self._emission_probabilities(observations)
        length = len(observations)
        k = self.n_states
        log_transition = np.log(self.A + 1e-300)
        delta = np.zeros((length, k))
        previous_state = np.zeros((length, k), dtype=int)

        delta[0] = np.log(self.pi + 1e-300) + np.log(probabilities[0])
        for time in range(1, length):
            candidates = delta[time - 1][:, None] + log_transition
            previous_state[time] = np.argmax(candidates, axis=0)
            delta[time] = (
                candidates[previous_state[time], np.arange(k)]
                + np.log(probabilities[time])
            )

        states = np.zeros(length, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for time in range(length - 2, -1, -1):
            states[time] = previous_state[time + 1, states[time + 1]]
        return states

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, observations):
        """Estimate parameters by Baum-Welch from a one-dimensional array."""
        observations = self._check(observations)
        if len(observations) < 2:
            raise ValueError("observations must contain at least two values")

        self._init(observations)
        previous_log_likelihood = -np.inf
        self.log_likelihood_history_ = []

        for iteration in range(self.n_iter):
            probabilities = self._emission_probabilities(observations)
            alpha, scales = self._forward(probabilities)
            beta = self._backward(probabilities, scales)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            if self.markov:
                weighted_next = (probabilities[1:] * beta[1:]) / (
                    scales[1:, None] + 1e-300
                )
                expected_transitions = self.A * (alpha[:-1].T @ weighted_next)
                self.pi = gamma[0]
                self.A = expected_transitions / (
                    expected_transitions.sum(axis=1, keepdims=True) + 1e-300
                )
            else:
                weights = gamma.mean(axis=0)
                self.pi = weights
                self.A = np.tile(weights, (self.n_states, 1))

            self._update_emissions(observations, gamma)

            log_likelihood = np.log(scales + 1e-300).sum()
            self.log_likelihood_history_.append(log_likelihood)
            if abs(log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = log_likelihood

        self.n_iter_ = iteration + 1
        self.log_likelihood_ = log_likelihood
        self._order_states()
        return self

    def _update_emissions(self, observations, gamma):
        for state in range(self.n_states):
            weights = gamma[:, state]
            weight_sum = weights.sum() + 1e-300
            if self.emission == "gaussian":
                self.mu[state] = (weights * observations).sum() / weight_sum
                self.sigma[state] = np.sqrt(
                    (weights * (observations - self.mu[state]) ** 2).sum()
                    / weight_sum
                    + 1e-8
                )
                continue

            nu = self.nu[state]
            z2 = ((observations - self.mu[state]) / self.sigma[state]) ** 2
            u = (nu + 1.0) / (nu + z2)
            self.mu[state] = (weights * u * observations).sum() / (
                (weights * u).sum() + 1e-300
            )
            self.sigma[state] = np.sqrt(
                (weights * u * (observations - self.mu[state]) ** 2).sum()
                / weight_sum
                + 1e-8
            )
            self.nu[state] = self._maximise_nu(
                observations, weights, self.mu[state], self.sigma[state], nu
            )

    @staticmethod
    def _maximise_nu(observations, weights, mu, sigma, current):
        def negative_weighted_log_likelihood(nu):
            return -(
                weights
                * student_t.logpdf(observations, nu, loc=mu, scale=sigma)
            ).sum()

        result = minimize_scalar(
            negative_weighted_log_likelihood,
            bounds=(NU_MIN, NU_MAX),
            method="bounded",
            options={"xatol": 1e-4},
        )
        current_value = negative_weighted_log_likelihood(
            np.clip(current, NU_MIN, NU_MAX)
        )
        if result.fun <= current_value:
            return float(result.x)
        return float(np.clip(current, NU_MIN, NU_MAX))

    def _order_states(self):
        """Order states by emission scale so labels are comparable."""
        order = np.argsort(self.sigma, kind="stable")
        self.pi = self.pi[order]
        self.mu = self.mu[order]
        self.sigma = self.sigma[order]
        self.A = self.A[np.ix_(order, order)]
        if self.nu is not None:
            self.nu = self.nu[order]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, length, random_state=None):
        """Simulate observations and hidden states from the model."""
        rng = np.random.default_rng(random_state)
        k = self.n_states
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length)
        states[0] = rng.choice(k, p=self.pi)
        for time in range(1, length):
            states[time] = rng.choice(k, p=self.A[states[time - 1]])
        for state in range(k):
            mask = states == state
            count = int(mask.sum())
            if self.emission == "gaussian":
                draws = rng.normal(self.mu[state], self.sigma[state], count)
            else:
                draws = self.mu[state] + self.sigma[state] * rng.standard_t(
                    self.nu[state], count
                )
            observations[mask] = draws
        return observations, states

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _check(observations):
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 1 or len(observations) < 1:
            raise ValueError("observations must be a one-dimensional array")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")
        return observations


def brute_force_inference(model, observations):
    """Enumerate every state path for a short sequence.

    Returns the exact likelihood, smoothed posteriors and Viterbi path. Used
    only by tests, on sequences short enough to enumerate.
    """
    import itertools

    observations = np.asarray(observations, dtype=float)
    length = len(observations)
    k = model.n_states
    if length > 8:
        raise ValueError("brute force is limited to eight observations")
    probabilities = model._emission_probabilities(observations)

    total = 0.0
    posterior = np.zeros((length, k))
    best_path = None
    best_probability = -1.0
    for path in itertools.product(range(k), repeat=length):
        probability = model.pi[path[0]] * probabilities[0, path[0]]
        for time in range(1, length):
            probability *= (
                model.A[path[time - 1], path[time]]
                * probabilities[time, path[time]]
            )
        total += probability
        for time, state in enumerate(path):
            posterior[time, state] += probability
        if probability > best_probability:
            best_probability = probability
            best_path = np.array(path)
    return total, posterior / total, best_path
