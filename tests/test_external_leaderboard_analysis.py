import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from external_leaderboard_analysis import (  # noqa: E402
    compare_scores,
    normalized_model_id,
    paired_bootstrap_delta,
    prepare_epoch_benchmark,
    validate_aliases,
    validate_epoch_payload,
)


class ExternalLeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.ratings = {
            "raw": {f"m{i}": float(i) for i in range(10)},
            "formatting_controlled": {f"m{i}": float(i) for i in range(10)},
            "joint_controlled": {f"m{i}": float(9 - i) for i in range(10)},
        }
        self.scores = {f"m{i}": float(i) for i in range(10)}
        self.common = sorted(self.scores)

    def test_normalization_does_not_remove_version_digits(self):
        self.assertEqual(
            normalized_model_id("Claude-3.5-Sonnet-20241022"),
            "claude35sonnet20241022",
        )

    def test_snapshot_hash_fails_closed(self):
        payload = b"audited bytes"
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(validate_epoch_payload(payload, digest), digest)
        with self.assertRaises(RuntimeError):
            validate_epoch_payload(payload, "0" * 64)

    def test_alias_targets_are_unique(self):
        validate_aliases()
        validate_aliases({"comparia-a": "epoch-a", "comparia-b": "epoch-b"})
        with self.assertRaises(ValueError):
            validate_aliases({"comparia-a": "epoch-a", "comparia-b": "epoch-a"})

    def test_compare_scores_uses_identical_support_and_known_direction(self):
        result = compare_scores(
            self.ratings,
            self.scores,
            self.common,
            np.random.default_rng(4),
            n_bootstrap=100,
            min_overlap=10,
        )
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["spearman"]["raw"], 1.0)
        self.assertAlmostEqual(result["spearman"]["formatting_controlled"], 1.0)
        self.assertAlmostEqual(result["spearman"]["joint_controlled"], -1.0)
        self.assertAlmostEqual(
            result["delta_vs_raw"]["formatting_controlled"]["point"], 0.0
        )
        self.assertAlmostEqual(
            result["delta_vs_raw"]["joint_controlled"]["point"], -2.0
        )

    def test_compare_scores_enforces_minimum_overlap(self):
        common = self.common[:9]
        result = compare_scores(
            self.ratings,
            self.scores,
            common,
            np.random.default_rng(2),
            n_bootstrap=10,
            min_overlap=10,
        )
        self.assertFalse(result["eligible"])
        self.assertNotIn("spearman", result)

    def test_compare_scores_rejects_mismatched_rating_support(self):
        ratings = {name: values.copy() for name, values in self.ratings.items()}
        del ratings["joint_controlled"]["m0"]
        with self.assertRaises(ValueError):
            compare_scores(
                ratings,
                self.scores,
                self.common,
                np.random.default_rng(2),
                n_bootstrap=10,
            )

    def test_paired_bootstrap_is_deterministic(self):
        base = np.arange(10, dtype=float)
        alternative = np.array([0, 2, 1, 3, 4, 5, 6, 8, 7, 9], dtype=float)
        external = np.arange(10, dtype=float)
        first = paired_bootstrap_delta(
            base, alternative, external, np.random.default_rng(8), 200
        )
        second = paired_bootstrap_delta(
            base, alternative, external, np.random.default_rng(8), 200
        )
        self.assertEqual(first, second)

    def test_matching_records_aliases_absence_and_ambiguity(self):
        frame = pd.DataFrame({
            "Model version": ["epoch-a", "epoch-b", "epoch-b"],
            "Score": [1.0, 2.0, 3.0],
            "Organization": ["A", "B", "B"],
        })
        prepared = prepare_epoch_benchmark(
            frame, ["comparia-a", "epoch-b", "missing"], "Score"
        )
        # The global alias table intentionally does not contain this test alias,
        # so exact matching finds only the ambiguous and absent cases.
        statuses = {
            row["comparia_model_id"]: row["status"]
            for row in prepared["records"]
        }
        self.assertEqual(statuses["epoch-b"], "excluded_ambiguous_source_scores")
        self.assertEqual(statuses["missing"], "not_present")
        self.assertEqual(prepared["scores"], {})

    def test_audited_alias_is_recorded_and_matched(self):
        frame = pd.DataFrame({
            "Model version": ["claude-3-7-sonnet-20250219"],
            "Score": [0.8],
            "Organization": ["Anthropic"],
        })
        prepared = prepare_epoch_benchmark(
            frame, ["claude-3-7-sonnet"], "Score"
        )
        record = prepared["records"][0]
        self.assertEqual(record["status"], "matched")
        self.assertEqual(record["match_rule"], "audited_same_build_alias")
        self.assertEqual(prepared["scores"], {"claude-3-7-sonnet": 0.8})

    def test_one_external_build_cannot_match_two_comparia_models(self):
        frame = pd.DataFrame({
            "Model version": ["claude-3-7-sonnet-20250219"],
            "Score": [0.8],
            "Organization": ["Anthropic"],
        })
        with self.assertRaises(ValueError):
            prepare_epoch_benchmark(
                frame,
                ["claude-3-7-sonnet", "claude-3-7-sonnet-20250219"],
                "Score",
            )

    def test_matching_requires_numeric_scores_and_required_schema(self):
        with self.assertRaises(ValueError):
            prepare_epoch_benchmark(
                pd.DataFrame({"Model version": ["m1"]}), ["m1"], "Score"
            )
        with self.assertRaises(ValueError):
            prepare_epoch_benchmark(
                pd.DataFrame({
                    "Model version": ["m1"],
                    "Score": ["not numeric"],
                }),
                ["m1"],
                "Score",
            )


if __name__ == "__main__":
    unittest.main()
