import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_FILES = (ROOT / "paper_draft.md", ROOT / "README.md")
STATUS_FILE = ROOT / "feedback.md"
EXTERNAL_RESULTS = ROOT / "results" / "external_leaderboard_results.json"
PRODUCTION_EXAMPLES = ROOT / "results" / "production_ranking_examples.json"


def read_all(paths):
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


class PublicationConsistencyTests(unittest.TestCase):
    def test_publication_prose_has_no_em_dashes(self):
        for path in PUBLICATION_FILES:
            with self.subTest(path=path.name):
                self.assertNotIn("—", path.read_text(encoding="utf-8"))

    def test_comparia_spelling_is_consistent(self):
        text = read_all((*PUBLICATION_FILES, STATUS_FILE))
        malformed = re.findall(r"(?<![\w-])(?:compar:IA|ComparIA)(?![\w-])", text)
        self.assertEqual(
            malformed,
            [],
            "Use 'Compar:IA' in prose and citations; keep 'comparia' only in "
            "literal dataset names and URLs.",
        )

    def test_sample_size_convention_is_stable(self):
        text = read_all((*PUBLICATION_FILES, STATUS_FILE))
        self.assertNotRegex(text, r"(?:about\s+138,000|~\s*137K)")
        self.assertIn("137,214", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("137,214", STATUS_FILE.read_text(encoding="utf-8"))
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        self.assertIn("127,010", paper)
        self.assertNotIn("127,893", paper)

    def test_external_ranking_labels_are_present(self):
        text = read_all(PUBLICATION_FILES)
        for label in ("raw", "formatting-controlled", "full joint-controlled"):
            with self.subTest(label=label):
                self.assertIn(label, text)

    def test_bt_abbreviation_is_defined_or_absent(self):
        for path in PUBLICATION_FILES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                if re.search(r"\bBT\b", text):
                    self.assertIn("Bradley-Terry (BT)", text)

    def test_external_results_and_publication_claims_are_aligned(self):
        results = json.loads(EXTERNAL_RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(results["epoch_snapshot"]["status"], "available")
        eligible = [
            record
            for record in results["capability_benchmarks"].values()
            if record["eligible"]
        ]
        self.assertEqual(len(eligible), 7)
        formatting_deltas = [
            record["delta_vs_raw"]["formatting_controlled"]["point"]
            for record in eligible
        ]
        self.assertTrue(all(delta < 0 for delta in formatting_deltas))
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        self.assertIn(results["epoch_snapshot"]["sha256"], paper)
        self.assertIn("suggestive face validity but inconclusive external validation", paper)
        for omitted_model in (
            "GPT-5.3",
            "Mistral Medium 2508",
            "Gemini 3.1 Flash Lite",
        ):
            self.assertIn(omitted_model, paper)

    def test_live_examples_are_labelled_as_a_distinct_snapshot(self):
        examples = json.loads(PRODUCTION_EXAMPLES.read_text(encoding="utf-8"))
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        self.assertEqual(examples["observed_at"], "2026-07-27")
        self.assertTrue(examples["live_page"]["counter_includes_ties"])
        self.assertIn("counter includes ties", paper)
        self.assertIn("not ranks reconstructed from the research release", paper)
        by_model = {row["model"]: row for row in examples["models"]}
        self.assertEqual(by_model["GPT-5.3"]["raw_rank"], 47)
        self.assertEqual(by_model["GPT-5.3"]["style_controlled_rank"], 1)
        self.assertFalse(
            by_model["Mistral Medium 2508"][
                "epoch_capabilities_index_matched"
            ]
        )

    def test_levels_of_reading_hypothesis_is_retained_and_qualified(self):
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        self.assertIn("level of reading", paper)
        self.assertIn("deliberately speculative and imperfect", paper)
        self.assertIn("do not test it directly", paper)


if __name__ == "__main__":
    unittest.main()
