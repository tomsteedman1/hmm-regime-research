"""Guard the link to the research half of the repository."""

import unittest

import numpy as np

import research_link
from regime_hmm import RegimeHMM
from wti_data import FROZEN_DATA, load_changes


class ResearchLinkTests(unittest.TestCase):
    def test_frozen_data_hash_matches_manifest(self):
        digest = research_link.value_sha256(FROZEN_DATA)
        self.assertEqual(digest, research_link.RESEARCH_DATA_SHA256)

    def test_selected_model_filters_the_frozen_sample(self):
        changes = load_changes()
        model = RegimeHMM(n_states=2, emission="student_t")
        model.set_parameters(
            pi=np.array([0.5, 0.5]),
            A=np.array([[0.99, 0.01], [0.02, 0.98]]),
            mu=np.array([0.05, -0.03]),
            sigma=np.array([1.0, 1.7]),
            nu=np.array([14.0, 4.0]),
        )
        filtered = model.filter_proba(changes.to_numpy())
        self.assertEqual(filtered.shape, (len(changes), 2))
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
