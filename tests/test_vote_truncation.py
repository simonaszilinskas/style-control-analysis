import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_fr_arena import (  # noqa: E402
    _assistant_text,
    _attach_prefix_lengths,
    _conversation_prefix,
    _process_rows,
)


def conversation(*answers):
    messages = []
    for index, answer in enumerate(answers):
        messages.extend([
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": answer},
        ])
    return messages


class VoteTruncationTests(unittest.TestCase):
    def test_prefix_stops_after_voted_assistant_response(self):
        messages = conversation("first answer", "future answer")

        prefix = _conversation_prefix(messages, 0)

        self.assertEqual([m["role"] for m in prefix], ["user", "assistant"])
        self.assertEqual(_assistant_text(prefix), "first answer")

    def test_prefix_fails_closed_when_vote_turn_is_not_present(self):
        self.assertIsNone(_conversation_prefix(conversation("only answer"), 1))

    def test_process_rows_uses_visible_prefix_and_vote_time_depth(self):
        enough_words = " ".join(f"word{i}" for i in range(60))
        future_bold = "**future formatting must not be counted**"
        row = {
            "comparison_id": "c1",
            "choice": "a_better",
            "turn": 0,
            "model_a": "a",
            "model_b": "b",
            "full_conversation_a": conversation(enough_words, future_bold),
            "full_conversation_b": conversation(enough_words, future_bold),
            "metadata": {
                "languages": ["fr"],
                "tokens_a": 60,
                "tokens_b": 55,
                "categories": ["test"],
                "mode": "random",
            },
        }

        battles, tokens = _process_rows([row])

        self.assertEqual(len(battles), 1)
        self.assertEqual(battles[0]["vote_turn"], 0)
        self.assertEqual(battles[0]["conv_turns"], 1)
        self.assertEqual(battles[0]["bold_a"], 0.0)
        self.assertEqual(tokens[0]["tokens_a"], 60)

    def test_prefix_length_sums_only_turns_through_vote(self):
        battles = pd.DataFrame([
            {"conversation_pair_id": "c1", "vote_turn": 1},
            {"conversation_pair_id": "c2", "vote_turn": 0},
        ])
        tokens = pd.DataFrame([
            {"conversation_pair_id": "c1", "turn": 0, "tokens_a": 10, "tokens_b": 20},
            {"conversation_pair_id": "c1", "turn": 1, "tokens_a": 11, "tokens_b": 21},
            {"conversation_pair_id": "c1", "turn": 2, "tokens_a": 100, "tokens_b": 200},
            {"conversation_pair_id": "c2", "turn": 0, "tokens_a": 7, "tokens_b": 8},
        ])

        out = _attach_prefix_lengths(battles, tokens).set_index("conversation_pair_id")

        self.assertEqual(out.loc["c1", "length_a"], 21)
        self.assertEqual(out.loc["c1", "length_b"], 41)
        self.assertEqual(out.loc["c1", "post_vote_turns"], 1)
        self.assertEqual(out.loc["c2", "length_a"], 7)

    def test_missing_prefix_token_count_does_not_become_a_partial_sum(self):
        battles = pd.DataFrame([{"conversation_pair_id": "c1", "vote_turn": 1}])
        tokens = pd.DataFrame([
            {"conversation_pair_id": "c1", "turn": 0, "tokens_a": np.nan, "tokens_b": 3},
            {"conversation_pair_id": "c1", "turn": 1, "tokens_a": 5, "tokens_b": 4},
        ])

        out = _attach_prefix_lengths(battles, tokens)

        self.assertTrue(np.isnan(out.loc[0, "length_a"]))
        self.assertEqual(out.loc[0, "length_b"], 7)


if __name__ == "__main__":
    unittest.main()
