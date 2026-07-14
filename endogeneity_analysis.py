#!/usr/bin/env python3
"""
Endogeneity: is formatting a confounder or a mediator? (§5.3)

Three formatting-intensity tests on the comparia-fr-arena battle table, none of
which need model metadata beyond the battles themselves:

  1. quality-formatting correlation: does model formatting intensity track the
     standard BT rating, and does that association shrink under style control?
  2. tier-stratified style effects: is the formatting premium larger among
     weaker models (confounder) than stronger ones (mediator)?
  3. rating change vs formatting intensity across models.

    python endogeneity_analysis.py     # -> endogeneity_results.json
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analyze_core import fit, MIN_BATTLES, FORMATTING

np.random.seed(42)
INTENSITY = ["bold", "lists", "headers"]


def model_formatting_intensity(d, models):
    """Mean (bold+lists+headers) per response, per model, over its battles."""
    tot = {m: [] for m in models}
    for side, mcol in (("a", "model_a_name"), ("b", "model_b_name")):
        v = d[[f"{f}_{side}" for f in INTENSITY]].sum(axis=1).to_numpy()
        for m, x in zip(d[mcol].to_numpy(), v):
            if m in tot:
                tot[m].append(x)
    return {m: float(np.mean(tot[m])) if tot[m] else np.nan for m in models}


def main():
    res = {}
    b = pd.read_parquet("fr_battles.parquet")
    d = b[b["winner"].isin(["model_a", "model_b"])].dropna(
        subset=[f"{f}_{s}" for f in FORMATTING for s in ("a", "b")]).copy()
    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy()

    standard, _ = fit(d, models, [])
    controlled, _ = fit(d, models, FORMATTING)
    intensity = model_formatting_intensity(d, models)

    ms = [m for m in models if not np.isnan(intensity[m])]
    x = np.array([intensity[m] for m in ms])
    rs = np.array([standard[m] for m in ms])
    rc = np.array([controlled[m] for m in ms])

    # Test 1: quality-formatting correlation, standard and controlled.
    pr = pearsonr(x, rs)
    sp = spearmanr(x, rs)
    res["quality_formatting"] = {
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sp.statistic),
        "pearson_r_controlled": float(pearsonr(x, rc).statistic)}
    print(f"formatting vs standard rating: r={pr.statistic:.3f} (p={pr.pvalue:.1e}), rho={sp.statistic:.3f}")
    print(f"formatting vs controlled rating: r={pearsonr(x, rc).statistic:.3f}")

    # Test 3: rating change vs formatting intensity.
    dr = rc - rs
    prc = pearsonr(x, dr)
    res["rating_change_vs_formatting_r"] = float(prc.statistic)
    res["rating_change_vs_formatting_p"] = float(prc.pvalue)
    print(f"rating change vs formatting: r={prc.statistic:.3f} (p={prc.pvalue:.1e})")

    # Test 2: tier-stratified style effects (tiers by standard rating).
    q1, q2 = np.quantile(rs, 1/3), np.quantile(rs, 2/3)
    tiers = {m: ("top" if standard[m] >= q2 else "bottom" if standard[m] <= q1 else "middle")
             for m in ms}
    res["tiers"] = {}
    for t in ("bottom", "middle", "top"):
        tmodels = [m for m in ms if tiers[m] == t]
        sub = d[d["model_a_name"].isin(tmodels) & d["model_b_name"].isin(tmodels)]
        if len(sub) < 200:
            continue
        _, coefs = fit(sub, sorted(tmodels), FORMATTING)
        res["tiers"][t] = {"n_battles": int(len(sub)),
                           **{f: float((np.exp(coefs[f]) - 1) * 100) for f in INTENSITY}}
        print(f"  {t:7s} n={len(sub):6d}  " +
              "  ".join(f"{f}={(np.exp(coefs[f])-1)*100:+.1f}%" for f in INTENSITY))

    with open("endogeneity_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved endogeneity_results.json")


if __name__ == "__main__":
    main()
