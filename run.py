#!/usr/bin/env python3
"""Run the documented analysis pipeline with explicit reproducibility profiles.

Profiles
--------
core
    Recompute the main paper analyses and Figures 1--3 from the committed
    battle table.  This is the default.
extended
    Run ``core`` plus every analysis that can use the committed auxiliary
    tables, the external snapshot/cache, and Figure 4.
full
    Rebuild the battle and auxiliary tables from the pinned gated Hugging Face
    release, run the raw-data audits, then run ``extended``.

Examples
--------
    python run.py
    python run.py --profile extended
    python run.py --profile full --reset-checkpoints
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

RAW_STEPS = [
    "build_fr_arena.py",
    "extract_prompts.py",
    "task_classify.py",
    "mattr_alt.py",
    "audit_vote_timing.py",
    "audit_reasoning_content.py",
]

CORE_ANALYSES = [
    "analyze_core.py",
    "formatting_interactions.py",
    "linguistic_analysis.py",
    "leaderboard_shift.py",
    "turn_depth_analysis.py",
    "topic_analysis.py",
    "endogeneity_analysis.py",
    "qualitative_analysis.py",
]

EXTENDED_ANALYSES = [
    "task_analysis.py",
    "robustness_random.py",
    "time_block_bootstrap.py",
    "mattr_stress.py",
    "analyze_mattr_alt.py",
    "external_leaderboard_analysis.py",
]

FIGURE_STEPS = [
    "generate_linguistic_figure.py",
    "generate_polish_figures.py",
    "generate_external_figure.py",
]


def _run(step: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, str(SRC / step), *(extra_args or [])]
    print(f"\n=== {' '.join(command[1:])} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("core", "extended", "full"),
        default="core",
        help="pipeline scope (default: core)",
    )
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help=(
            "with --profile full, discard generated row-group checkpoints "
            "before rebuilding them"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.reset_checkpoints and args.profile != "full":
        raise SystemExit("--reset-checkpoints is only valid with --profile full")

    if args.profile == "full":
        for step in RAW_STEPS:
            extra = (
                ["--reset-checkpoints"]
                if args.reset_checkpoints
                and step in {"build_fr_arena.py", "extract_prompts.py", "mattr_alt.py"}
                else []
            )
            _run(step, extra)

    battles = ROOT / "data" / "fr_battles.parquet"
    if not battles.exists():
        raise SystemExit(
            "data/fr_battles.parquet is missing. Run "
            "`python run.py --profile full` or `python src/build_fr_arena.py`."
        )

    for step in CORE_ANALYSES:
        _run(step)
    if args.profile in {"extended", "full"}:
        for step in EXTENDED_ANALYSES:
            _run(step)
    for step in FIGURE_STEPS[:2]:
        _run(step)
    if args.profile in {"extended", "full"}:
        _run(FIGURE_STEPS[2])

    print(
        f"\nCompleted the {args.profile!r} profile. "
        "Results are in results/ and figures in figures/."
    )


if __name__ == "__main__":
    main()
