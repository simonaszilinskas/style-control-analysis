import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript" / "paper.tex"
PUBLICATION_FILES = (PAPER, ROOT / "README.md")
STATUS_FILE = ROOT / "feedback.md"
EXTERNAL_RESULTS = ROOT / "results" / "external_leaderboard_results.json"
PRODUCTION_EXAMPLES = ROOT / "results" / "production_ranking_examples.json"
REASONING_AUDIT = ROOT / "results" / "reasoning_content_audit_results.json"
TURN_DEPTH_RESULTS = ROOT / "results" / "turn_depth_results.json"
TASK_RESULTS = ROOT / "results" / "task_results.json"
INTERACTION_RESULTS = ROOT / "results" / "formatting_interaction_results.json"


def read_all(paths):
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def flat(path):
    """Paper text with line wrapping collapsed, so claims can be matched as prose."""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


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
        self.assertIn("137,293", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("137,293", STATUS_FILE.read_text(encoding="utf-8"))
        paper = flat(PAPER)
        self.assertIn("127,092", paper)
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
        self.assertIn(
            results["epoch_snapshot"]["status"],
            {"available", "reused_cached_scores"},
        )
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
        paper = flat(PAPER)
        self.assertIn(results["epoch_snapshot"]["sha256"], paper)
        self.assertIn(
            "do not show that adjusted rankings better measure capability",
            paper,
        )
        for omitted_model in (
            "GPT-5.3",
            "Mistral Medium 2508",
            "Gemini 3.1 Flash Lite",
        ):
            self.assertIn(omitted_model, paper)

    def test_live_examples_are_labelled_as_a_distinct_snapshot(self):
        examples = json.loads(PRODUCTION_EXAMPLES.read_text(encoding="utf-8"))
        paper = flat(PAPER)
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

    def test_levels_of_reading_is_present_and_qualified(self):
        paper = flat(PAPER)
        self.assertIn("levels of reading", paper)
        self.assertIn("This is a hypothesis", paper)
        self.assertIn("turn count does not measure attention", paper)

    def test_depth_table_uses_the_pooled_interaction_model(self):
        results = json.loads(TURN_DEPTH_RESULTS.read_text(encoding="utf-8"))
        bold = results["primary"]["features"]["bold"]
        single = (pow(2.718281828459045, bold["main"]) - 1) * 100
        multi = (
            pow(2.718281828459045, bold["main"] + bold["interaction"]) - 1
        ) * 100
        paper = flat(PAPER)
        self.assertIn(f"+{single:.1f}", paper)
        self.assertIn(f"+{multi:.1f}", paper)
        self.assertNotIn("+31.0% win odds in single-turn", paper)

    def test_reasoning_audit_supports_the_manuscript_claim(self):
        audit = json.loads(REASONING_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["n_newly_retained_by_span_parser"], 86)
        self.assertEqual(
            audit["n_legacy_battles_removed_by_fail_closed_parser"], 7
        )
        self.assertEqual(audit["net_retained_battle_change"], 79)
        self.assertEqual(audit["n_missing_final_with_reasoning_content"], 0)

    def test_exploratory_task_bootstrap_is_reproducible(self):
        results = json.loads(TASK_RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(results["analysis"]["bootstrap_seed"], 42)
        self.assertEqual(results["analysis"]["bootstrap_resamples"], 400)

    def test_formatting_interactions_claim_matches_results(self):
        results = json.loads(INTERACTION_RESULTS.read_text(encoding="utf-8"))
        interactions = results["interactions"]
        self.assertEqual(len(interactions), 10)
        self.assertTrue(
            all(not record["sig_bh"] for record in interactions.values()),
            "The paper states that no pairwise formatting interaction "
            "survives Benjamini-Hochberg correction.",
        )
        paper = flat(PAPER)
        self.assertIn("None of the ten survives BH correction", paper)

    def test_figure_captions_do_not_duplicate_automatic_numbering(self):
        latex = PAPER.read_text(encoding="utf-8")
        self.assertNotRegex(latex, r"\\caption\{Figure\s+\d")


if __name__ == "__main__":
    unittest.main()
