#!/usr/bin/env python3
"""
Leaderboard shift under formatting-only vs joint presentation control (§4.7).

The §4.7 prose reports how much the ranking moves when we control for the full
presentation bundle rather than markdown alone: the standard-vs-controlled rating
correlations, how many models move >=10 ranks, and who the biggest movers are.
Those numbers were previously computed ad hoc; this script reproduces them from
the same common support and the same fit_bt used everywhere else, and saves them.

    python leaderboard_shift.py   # -> leaderboard_shift_results.json
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from modeling import fit_bt, FORMATTING, LINGUISTIC, MIN_BATTLES
from paths import BATTLES, RESULTS


def ranks(ratings):
    order = sorted(ratings, key=lambda m: -ratings[m])
    return {m: i + 1 for i, m in enumerate(order)}


def main():
    battles = pd.read_parquet(BATTLES)

    core = FORMATTING + ["length"] + LINGUISTIC
    sub = battles.dropna(subset=[f"{f}_{s}" for f in core for s in ("a", "b")]).copy()
    counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    sub = sub[sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)]
    print(f"Common support: {len(sub):,} battles, {len(models)} models")

    standard, _, _, _, _ = fit_bt(sub, models, [])
    fmt, _, _, _, _ = fit_bt(sub, models, FORMATTING)
    joint, _, _, _, _ = fit_bt(sub, models, core)

    common = [m for m in models if m in standard and m in fmt and m in joint]
    s = np.array([standard[m] for m in common])
    f = np.array([fmt[m] for m in common])
    j = np.array([joint[m] for m in common])

    r_fmt = float(np.corrcoef(s, f)[0, 1])
    r_joint = float(np.corrcoef(s, j)[0, 1])
    rho_joint = float(spearmanr(s, j).statistic)

    rk_std, rk_joint = ranks(standard), ranks(joint)
    deltas = {m: rk_std[m] - rk_joint[m] for m in common}   # + = rises under joint control
    movers10 = sorted((m for m in common if abs(deltas[m]) >= 10),
                      key=lambda m: deltas[m])
    top_fall = sorted(common, key=lambda m: deltas[m])[:6]
    top_rise = sorted(common, key=lambda m: -deltas[m])[:6]

    out = {
        "n_battles": int(len(sub)), "n_models": len(common),
        "corr_standard_vs_formatting": round(r_fmt, 4),
        "corr_standard_vs_joint_pearson": round(r_joint, 4),
        "corr_standard_vs_joint_spearman": round(rho_joint, 4),
        "n_move_ge_10_ranks": len(movers10),
        "biggest_fallers": {m: deltas[m] for m in top_fall},
        "biggest_risers": {m: deltas[m] for m in top_rise},
    }
    print(f"\ncorr(standard, formatting-controlled) = {r_fmt:.4f}")
    print(f"corr(standard, joint-controlled)       = {r_joint:.4f}  (Spearman {rho_joint:.4f})")
    print(f"models moving >= 10 ranks under joint control: {len(movers10)}")
    print("\nbiggest fallers (rank delta, - = falls):")
    for m in top_fall:
        print(f"  {m:32s} {deltas[m]:+d}")
    print("biggest risers:")
    for m in top_rise:
        print(f"  {m:32s} {deltas[m]:+d}")

    with open(RESULTS / "leaderboard_shift_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved leaderboard_shift_results.json")


if __name__ == "__main__":
    main()
