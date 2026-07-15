#!/usr/bin/env python3
"""
Run the full analysis pipeline in order.

Assumes the battle table already exists at data/fr_battles.parquet; build it once
with `python src/build_fr_arena.py` (that step needs the comparia-fr-arena
dataset and is kept separate because it is slow). Results land in results/,
figures in figures/.

    python run.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    "analyze_core.py",            # §4.1-4.4  formatting BT, ranks, position bias
    "linguistic_analysis.py",     # §4.5      joint formatting+length+linguistic
    "leaderboard_shift.py",       # §4.5      standard vs formatting vs joint ranking
    "turn_depth_analysis.py",     # §4.6      reading-depth interactions
    "topic_analysis.py",          # §4.7      topic controls
    "endogeneity_analysis.py",    # §5.3      confounder vs mediator
    "qualitative_analysis.py",    # §5.4      winner-flip prevalence/asymmetry
    "generate_linguistic_figure.py",  # Figure 1
    "generate_polish_figures.py",     # Figures 2-3
]


def main():
    if not (ROOT / "data" / "fr_battles.parquet").exists():
        print("data/fr_battles.parquet not found. Build it first:\n"
              "  python src/build_fr_arena.py")
        sys.exit(1)
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        result = subprocess.run([sys.executable, str(SRC / step)])
        if result.returncode != 0:
            print(f"FAILED: {step}")
            sys.exit(result.returncode)
    print("\nDone. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
