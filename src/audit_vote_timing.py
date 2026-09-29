#!/usr/bin/env python3
"""Audit temporal ordering in the vote-truncated battle table.

The consolidated source repeats completed conversations on each turn-level row.
The battle builder now truncates them at the retained vote.  This audit verifies
that the stored visible-turn count matches that vote and separately reports how
often the source conversation continued afterward.  Later source turns are
useful provenance, but none should enter the measured feature prefix.

    python src/audit_vote_timing.py
        -> results/vote_timing_audit_results.json
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from build_fr_arena import LOCAL_PATHS, _open
from paths import BATTLES, RESULTS


RAW_CANDIDATES = [Path(path) for path in LOCAL_PATHS]


def main():
    battles = pd.read_parquet(
        BATTLES, columns=["conversation_pair_id", "vote_turn", "conv_turns",
                          "final_turn", "post_vote_turns"]
    ).set_index("conversation_pair_id")
    ids = set(battles.index)
    raw_path = next((path for path in RAW_CANDIDATES if path.exists()), None)
    if raw_path is not None:
        turns = pq.read_table(
            raw_path, columns=["comparison_id", "turn", "choice"]
        ).to_pandas()
    else:
        pf = _open()
        turns = pd.concat([
            pf.read_row_group(rg, columns=["comparison_id", "turn", "choice"]).to_pandas()
            for rg in range(pf.num_row_groups)
        ], ignore_index=True)
    turns = turns[turns["comparison_id"].isin(ids)]
    final_turn = turns.groupby("comparison_id")["turn"].max()
    decisive = turns[turns["choice"].isin(["a_better", "b_better"])]
    retained_vote_turn = decisive.groupby("comparison_id")["turn"].max()

    audit = battles.join(retained_vote_turn.rename("raw_retained_vote_turn"))
    audit = audit.join(final_turn.rename("raw_final_turn"))
    audit["visible_turns_at_vote"] = audit["vote_turn"] + 1

    has_lookahead = audit["post_vote_turns"] > 0
    visible_multi = audit["conv_turns"] >= 2
    visible_single = audit["visible_turns_at_vote"] == 1
    vote_turn_matches = audit["vote_turn"] == audit["raw_retained_vote_turn"]
    final_turn_matches = audit["final_turn"] == audit["raw_final_turn"]
    visible_depth_matches = audit["conv_turns"] == audit["visible_turns_at_vote"]
    gap_matches = audit["post_vote_turns"] == audit["final_turn"] - audit["vote_turn"]
    output = {
        "n_analyzed_battles": int(len(audit)),
        "n_source_conversations_continuing_after_vote": int(has_lookahead.sum()),
        "share_source_conversations_continuing_after_vote": float(has_lookahead.mean()),
        "n_with_post_vote_turns_in_measured_features": 0,
        "share_with_post_vote_turns_in_measured_features": 0.0,
        "n_visible_multi_turn_at_vote": int(visible_multi.sum()),
        "n_visible_single_turn_at_vote": int(visible_single.sum()),
        "all_vote_turns_match_raw": bool(vote_turn_matches.all()),
        "all_final_turns_match_raw": bool(final_turn_matches.all()),
        "all_visible_depths_match_vote_turn": bool(visible_depth_matches.all()),
        "all_post_vote_gaps_match": bool(gap_matches.all()),
        "post_vote_turn_gap_counts": {
            str(int(k)): int(v)
            for k, v in audit["post_vote_turns"].value_counts().sort_index().items()
        },
    }
    validations = [vote_turn_matches, final_turn_matches, visible_depth_matches, gap_matches]
    if not all(check.all() for check in validations):
        raise AssertionError("vote-truncation audit failed; see mismatched validation fields")
    out_path = RESULTS / "vote_timing_audit_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
