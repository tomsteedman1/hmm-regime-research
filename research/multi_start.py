"""Multiple-start fitting for the two-state Gaussian HMM."""

import numpy as np

from gaussian_hmm import GaussianHMM2State


def fit_multiple_starts(
    observations,
    n_starts=12,
    base_seed=20260811,
    n_iter=300,
    tol=1e-7,
):
    """Fit one deterministic start and several reproducible random starts."""
    observations = np.asarray(observations, dtype=float)
    if n_starts < 2:
        raise ValueError("n_starts must be at least 2")

    models = []
    records = []
    paths = []

    for start in range(n_starts):
        deterministic = start == 0
        seed = None if deterministic else base_seed + start
        model = GaussianHMM2State(
            n_iter=n_iter,
            tol=tol,
            init_strategy="deterministic" if deterministic else "random",
            random_state=seed,
        ).fit(observations)
        path = model.decode(observations)
        fitted_score = model.score(observations)

        models.append(model)
        paths.append(path)
        records.append(
            {
                "start": start,
                "strategy": "deterministic" if deterministic else "random",
                "seed": seed,
                "log_likelihood": fitted_score,
                "iterations": model.n_iter_,
                "low_mean": model.mu[0],
                "high_mean": model.mu[1],
                "low_sigma": model.sigma[0],
                "high_sigma": model.sigma[1],
                "low_self_transition": model.A[0, 0],
                "high_self_transition": model.A[1, 1],
                "high_state_occupancy": (path == 1).mean(),
            }
        )

    scores = np.array([record["log_likelihood"] for record in records])
    best_index = int(np.argmax(scores))
    best_path = paths[best_index]
    for record, path in zip(records, paths):
        record["agreement_with_best"] = (path == best_path).mean()
        record["log_likelihood_gap"] = scores[best_index] - record["log_likelihood"]
        record["selected"] = record["start"] == best_index

    return models[best_index], records, paths


def fit_multiple_starts_general(
    observations,
    make_model,
    n_starts=12,
    base_seed=20260811,
):
    """Fit any model from one deterministic start and several random starts.

    ``make_model(init_strategy, random_state)`` must return an unfitted model
    with ``fit``, ``decode``, ``score`` and ``n_iter_``. States are compared
    after each model orders them, so the caller must use a model that orders
    its own states.
    """
    observations = np.asarray(observations, dtype=float)
    if n_starts < 2:
        raise ValueError("n_starts must be at least 2")

    models = []
    records = []
    paths = []
    for start in range(n_starts):
        deterministic = start == 0
        seed = None if deterministic else base_seed + start
        model = make_model(
            "deterministic" if deterministic else "random", seed
        ).fit(observations)
        path = model.decode(observations)
        models.append(model)
        paths.append(path)
        record = {
            "start": start,
            "strategy": "deterministic" if deterministic else "random",
            "seed": seed,
            "log_likelihood": model.score(observations),
            "iterations": model.n_iter_,
        }
        for state in range(model.n_states):
            record[f"mu_{state}"] = model.mu[state]
            record[f"sigma_{state}"] = model.sigma[state]
            record[f"self_transition_{state}"] = model.A[state, state]
            record[f"occupancy_{state}"] = (path == state).mean()
            if getattr(model, "nu", None) is not None:
                record[f"nu_{state}"] = model.nu[state]
        records.append(record)

    scores = np.array([record["log_likelihood"] for record in records])
    best_index = int(np.argmax(scores))
    best_path = paths[best_index]
    for record, path in zip(records, paths):
        record["agreement_with_best"] = (path == best_path).mean()
        record["log_likelihood_gap"] = scores[best_index] - record["log_likelihood"]
        record["selected"] = record["start"] == best_index
    return models[best_index], records, paths
