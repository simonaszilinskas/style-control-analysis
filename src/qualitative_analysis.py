#!/usr/bin/env python3
"""
Qualitative analysis of winner-flipping battles (§5.4).

A battle "flips" when the standard and style-controlled BT models disagree on
which of its two models is stronger (the rating order reverses under style
control). We report the prevalence of flips and the formatting asymmetry within
them (does the vote winner format more than the loser?).

Note: comparia-fr-arena has no per-message reactions, so the user-reported
"clear formatting" attribute and the reaction-sourced illustrative examples of
the older analysis are not available here.

    python qualitative_analysis.py     # -> qualitative_results.json
"""

import json
import numpy as np
import pandas as pd

from analyze_core import fit, MIN_BATTLES, FORMATTING
from paths import BATTLES, RESULTS

INTENSITY = ["bold", "lists", "headers"]


def main():
    res = {}
    b = pd.read_parquet(BATTLES)
    d = b[b["winner"].isin(["model_a", "model_b"])].dropna(
        subset=[f"{f}_{s}" for f in FORMATTING for s in ("a", "b")]).copy()
    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy()

    standard, _ = fit(d, models, [])
    controlled, _ = fit(d, models, FORMATTING)

    a, bb = d["model_a_name"].to_numpy(), d["model_b_name"].to_numpy()
    std_sign = np.sign([standard[x] - standard[y] for x, y in zip(a, bb)])
    ctrl_sign = np.sign([controlled[x] - controlled[y] for x, y in zip(a, bb)])
    flip = (std_sign != 0) & (ctrl_sign != 0) & (std_sign != ctrl_sign)
    n_flip = int(flip.sum())
    res["n_battles"] = int(len(d))
    res["n_flips"] = n_flip
    res["flip_pct"] = float(100 * n_flip / len(d))
    print(f"battles: {len(d):,}  flips: {n_flip:,} ({100*n_flip/len(d):.2f}%)")

    # Formatting asymmetry within flips: does the vote winner format more?
    fd = d[flip].copy()
    fmt_a = fd[[f"{f}_a" for f in INTENSITY]].sum(axis=1).to_numpy()
    fmt_b = fd[[f"{f}_b" for f in INTENSITY]].sum(axis=1).to_numpy()
    win_a = (fd["winner"] == "model_a").to_numpy()
    winner_fmt = np.where(win_a, fmt_a, fmt_b)
    loser_fmt = np.where(win_a, fmt_b, fmt_a)
    res["winner_formats_more_pct"] = float(100 * np.mean(winner_fmt > loser_fmt))
    res["loser_formats_more_pct"] = float(100 * np.mean(loser_fmt > winner_fmt))
    print(f"in flips: winner formats more {res['winner_formats_more_pct']:.1f}%, "
          f"loser {res['loser_formats_more_pct']:.1f}%")

    # Which models appear most in flips.
    fm = pd.concat([fd["model_a_name"], fd["model_b_name"]]).value_counts().head(10)
    res["top_flip_models"] = {m: int(c) for m, c in fm.items()}

    with open(RESULTS / "qualitative_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved qualitative_results.json")


if __name__ == "__main__":
    main()
