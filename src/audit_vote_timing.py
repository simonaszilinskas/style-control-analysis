#!/usr/bin/env python3
"""Audit whether full-conversation features include turns after the retained vote.

The consolidated source repeats the completed `full_conversation_*` on each
turn-level row.  The current battle builder retains the last decisive reaction
per comparison but computes features on that completed conversation.  This
audit measures how often later, unrated turns therefore enter the features.

    python src/audit_vote_timing.py
        -> results/vote_timing_audit_results.json
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from paths import BATTLES, RESULTS


RAW_CANDIDATES = [
    Path("comparia-fr-arena.parquet"),
    Path.home() / "Dev/comparia-theme-datasets/data/comparia-fr-arena/comparia-fr-arena.parquet",
]


def main():
    raw_path = next((path for path in RAW_CANDIDATES if path.exists()), None)
    if raw_path is None:
        raise FileNotFoundError(
            "Raw comparia-fr-arena.parquet not found; place it in the repo root."
        )

    battles = pd.read_parquet(
        BATTLES, columns=["conversation_pair_id", "conv_turns"]
    ).set_index("conversation_pair_id")
    ids = set(battles.index)
    turns = pq.read_table(
        raw_path, columns=["comparison_id", "turn", "choice"]
    ).to_pandas()
    turns = turns[turns["comparison_id"].isin(ids)]
    final_turn = turns.groupby("comparison_id")["turn"].max()
    decisive = turns[turns["choice"].isin(["a_better", "b_better"])]
    retained_vote_turn = decisive.groupby("comparison_id")["turn"].max()

    audit = battles.join(retained_vote_turn.rename("retained_vote_turn"))
    audit = audit.join(final_turn.rename("final_turn"))
    audit["post_vote_turns"] = audit["final_turn"] - audit["retained_vote_turn"]
    audit["visible_turns_at_vote"] = audit["retained_vote_turn"] + 1

    has_lookahead = audit["post_vote_turns"] > 0
    final_multi = audit["conv_turns"] >= 2
    visible_single = audit["visible_turns_at_vote"] == 1
    output = {
        "n_analyzed_battles": int(len(audit)),
        "n_with_post_vote_turns_in_full_conversation": int(has_lookahead.sum()),
        "share_with_post_vote_turns": float(has_lookahead.mean()),
        "n_classified_multi_turn_from_final_conversation": int(final_multi.sum()),
        "n_single_turn_at_vote_but_classified_multi_turn": int(
            (final_multi & visible_single).sum()
        ),
        "share_of_reported_multi_turn_stratum_single_at_vote": float(
            (final_multi & visible_single).sum() / final_multi.sum()
        ),
        "post_vote_turn_gap_counts": {
            str(int(k)): int(v)
            for k, v in audit["post_vote_turns"].value_counts().sort_index().items()
        },
    }
    out_path = RESULTS / "vote_timing_audit_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
