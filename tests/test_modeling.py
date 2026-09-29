import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling import (  # noqa: E402
    INIT_RATING,
    benjamini_hochberg,
    design_matrix,
    feature_contrasts,
    fit_bt,
)


class ModelingTests(unittest.TestCase):
    def setUp(self):
        self.battles = pd.DataFrame(
            {
                "model_a_name": ["a", "a", "b", "b", "a", "b"],
                "model_b_name": ["b", "b", "a", "a", "b", "a"],
                "winner": ["model_a", "model_b", "model_a", "model_b", "model_a", "model_b"],
                "style_a": [1.0, 2.0, 1.0, 3.0, 1000.0, 2.0],
                "style_b": [0.0, 0.0, 2.0, 1.0, 0.0, 1.0],
            }
        )

    def test_design_encodes_opponents_and_winner(self):
        model_design, style_design, outcomes = design_matrix(
            self.battles, ["a", "b"], ["style"]
        )
        np.testing.assert_array_equal(model_design[0], [1.0, -1.0])
        np.testing.assert_array_equal(model_design[2], [-1.0, 1.0])
        np.testing.assert_array_equal(outcomes[:2], [1.0, 0.0])
        self.assertAlmostEqual(float(style_design.mean()), 0.0)
        self.assertAlmostEqual(float(style_design.std()), 1.0)

    def test_winsorization_is_explicit(self):
        raw = feature_contrasts(self.battles, ["style"], winsorize=False)
        trimmed = feature_contrasts(self.battles, ["style"], winsorize=True)
        self.assertLess(trimmed.max(), raw.max())

    def test_bt_ratings_follow_mean_rating_convention(self):
        ratings, coefficients, outcomes, estimator, design = fit_bt(
            self.battles, ["a", "b"], ["style"]
        )
        self.assertAlmostEqual(np.mean(list(ratings.values())), INIT_RATING)
        self.assertEqual(set(coefficients), {"style"})
        self.assertEqual(len(outcomes), len(self.battles))
        self.assertEqual(design.shape, (len(self.battles), 3))
        self.assertEqual(estimator.coef_.shape, (1, 3))

    def test_benjamini_hochberg_matches_known_example(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
        np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


if __name__ == "__main__":
    unittest.main()
