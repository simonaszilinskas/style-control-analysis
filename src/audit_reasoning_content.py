#!/usr/bin/env python3
"""Audit hidden-reasoning serialization in the pinned Compar:IA release.

The audit reproduces the decisive-French, vote-time prefix filter without
persisting conversation text.  It reports:

* battles recovered by stripping paired ``<think>`` spans instead of dropping
  the whole message; and
* vote-time assistant messages whose final ``content`` is empty while a
  separate ``reasoning_content`` field is non-empty.

Only aggregate counts and model identifiers are written.
"""

from __future__ import annotations

import json
from collections import Counter

from build_fr_arena import (
    COLS,
    HF_PATH,
    HF_REVISION,
    _assistant_text,
    _conversation_prefix,
    _open,
)
from paths import RESULTS


def _legacy_assistant_text(messages):
    """Reproduce the superseded all-or-nothing tag filter for the audit."""
    parts = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if "<think>" not in content and "</think>" not in content:
            parts.append(content)
    return "\n".join(parts)


def _last_assistant_message(messages):
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return None


def _missing_final_with_reasoning(messages):
    message = _last_assistant_message(messages)
    if message is None:
        return False
    return not (message.get("content") or "").strip() and bool(
        (message.get("reasoning_content") or "").strip()
    )


def main():
    parquet = _open()
    candidate_rows = []
    for row_group in range(parquet.num_row_groups):
        for row in parquet.read_row_group(row_group, columns=COLS).to_pylist():
            if row["choice"] not in ("a_better", "b_better"):
                continue
            metadata = row["metadata"] or {}
            if "fr" not in (metadata.get("languages") or []):
                continue
            prefix_a = _conversation_prefix(
                row["full_conversation_a"], row["turn"]
            )
            prefix_b = _conversation_prefix(
                row["full_conversation_b"], row["turn"]
            )
            if prefix_a is None or prefix_b is None:
                continue
            visible_a = _assistant_text(prefix_a)
            visible_b = _assistant_text(prefix_b)
            legacy_a = _legacy_assistant_text(prefix_a)
            legacy_b = _legacy_assistant_text(prefix_b)
            candidate_rows.append(
                {
                    "comparison_id": row["comparison_id"],
                    "turn": int(row["turn"]),
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "current_valid": bool(visible_a and visible_b),
                    "legacy_valid": bool(legacy_a and legacy_b),
                    "missing_final_with_reasoning": (
                        _missing_final_with_reasoning(prefix_a)
                        or _missing_final_with_reasoning(prefix_b)
                    ),
                }
            )

    # Match each parser's battle builder independently: filter invalid rows,
    # then retain the last decisive vote per comparison.
    ordered = sorted(
        candidate_rows, key=lambda item: (item["comparison_id"], item["turn"])
    )

    def retained_by_parser(validity_key):
        by_comparison = {}
        for record in ordered:
            if record[validity_key]:
                by_comparison[record["comparison_id"]] = record
        return by_comparison

    current = retained_by_parser("current_valid")
    legacy = retained_by_parser("legacy_valid")
    recovered_ids = sorted(set(current) - set(legacy))
    excluded_ids = sorted(set(legacy) - set(current))
    recovered = [current[comparison_id] for comparison_id in recovered_ids]
    recovered_models = Counter()
    for record in recovered:
        recovered_models.update((record["model_a"], record["model_b"]))

    output = {
        "source": {
            "dataset": "ministere-culture/comparia-fr-arena",
            "revision": HF_REVISION,
            "path": HF_PATH,
        },
        "definition": {
            "unit": "last decisive French vote per comparison",
            "scope": "assistant messages through the retained vote only",
            "missing_final_query": (
                "last assistant message has blank content and non-blank "
                "reasoning_content on either side"
            ),
        },
        "n_retained_battles": len(current),
        "n_legacy_retained_battles": len(legacy),
        "n_newly_retained_by_span_parser": len(recovered_ids),
        "n_legacy_battles_removed_by_fail_closed_parser": len(excluded_ids),
        "net_retained_battle_change": len(current) - len(legacy),
        "n_missing_final_with_reasoning_content": sum(
            record["missing_final_with_reasoning"] for record in current.values()
        ),
        "recovered_model_appearances": dict(
            sorted(recovered_models.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    output_path = RESULTS / "reasoning_content_audit_results.json"
    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
