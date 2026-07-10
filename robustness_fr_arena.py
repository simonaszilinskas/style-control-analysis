#!/usr/bin/env python3
"""
Robustness of the linguistic extension (§4.7) on an independent dataset:
comparia-fr-arena.

The main analysis and §4.7 use the older votes+reactions data. comparia-fr-arena
is the newer consolidated arena export (a different collection window, a
partly different model roster, arena votes only). We rebuild the same feature set
on it (markdown headers/lists/bold/code_blocks/emoji with the identical regexes,
length from output tokens, and the same readability/diversity/structure features)
and fit the same joint Bradley-Terry model, using the winsorized standardized
contrasts, per-model strengths, 1000x bootstrap and Benjamini-Hochberg of
linguistic_analysis.py. If the headline conclusions hold on this second dataset,
they are not an artefact of one export.

Perplexity is not included here: CamemBERT pseudo-perplexity needs a GPU and is
only available on the older export, where §4.7 already shows it adds nothing.

Input fr_arena_battles.parquet is built by build_fr_arena.py (streams the gated
HF dataset). Run:  python robustness_fr_arena.py  ->  fr_arena_results.json
"""

import json
import numpy as np
import pandas as pd

import linguistic_analysis as LA

JOINT = LA.FORMATTING + ["length"] + LA.LINGUISTIC


def main():
    b = pd.read_parquet("fr_arena_battles.parquet")
    # Adapt column names/winner encoding to the shape linguistic_analysis expects.
    b["model_a_name"] = b["model_a"]
    b["model_b_name"] = b["model_b"]
    b["winner"] = np.where(b["winner"] == b["model_a"], "model_a", "model_b")

    sub = b.dropna(subset=[f"{f}_{s}" for f in JOINT for s in ("a", "b")]).copy()
    counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= LA.MIN_BATTLES].index)
    sub = sub[sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)]
    print(f"comparia-fr-arena common support: {len(sub):,} battles, {len(models)} models")

    results = {"dataset": "comparia-fr-arena",
               "common_support": {"battles": int(len(sub)), "models": len(models)}}

    # Nested fits + accuracy, mirroring §4.7.
    specs = {"formatting": LA.FORMATTING,
             "formatting+length": LA.FORMATTING + ["length"],
             "joint": JOINT}
    fits, fit_stats = {}, {}
    for name, feats in specs.items():
        _, coefs, y, lr, X = LA.fit_bt(sub, models, feats)
        fits[name] = coefs
        fit_stats[name] = {"acc": float(lr.score(X, y)), "n": int(len(y)), "k": X.shape[1]}
    results["bt_coefficients"] = fits
    results["model_fit"] = fit_stats

    # Bootstrap the joint coefficients (1000x) + BH.
    print(f"Bootstrapping joint coefficients ({LA.N_BOOTSTRAP}x)...")
    boot = {f: [] for f in JOINT}
    for i in range(LA.N_BOOTSTRAP):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{LA.N_BOOTSTRAP}")
        s = sub.sample(n=len(sub), replace=True)
        try:
            _, c, _, _, _ = LA.fit_bt(s, models, JOINT)
            for f in JOINT:
                boot[f].append(c[f])
        except Exception:
            continue
    ci, pvals = {}, {}
    for f in JOINT:
        arr = np.array(boot[f])
        point = fits["joint"][f]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * max(np.mean(arr <= 0) if point >= 0 else np.mean(arr >= 0), 1 / (len(arr) + 1))
        ci[f] = {"point": float(point), "ci": [float(lo), float(hi)],
                 "odds_pct": float((np.exp(point) - 1) * 100), "p": float(min(p, 1.0))}
        pvals[f] = min(p, 1.0)
    for f, a in zip(JOINT, LA.bh([pvals[f] for f in JOINT])):
        ci[f]["p_bh"] = float(a)
        ci[f]["sig_bh"] = bool(a < 0.05)
    results["joint_coefficients"] = ci

    print("\ncomparia-fr-arena joint coefficients (95% CI, BH):")
    for f in JOINT:
        c = ci[f]
        print(f"  {f:16s} {c['point']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"({c['odds_pct']:+.1f}%)  {'***' if c['sig_bh'] else 'n.s.'}")
    print("\nFit accuracy:  " + "  ".join(f"{n}={fit_stats[n]['acc']:.4f}" for n in specs))

    with open("fr_arena_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nSaved fr_arena_results.json")


if __name__ == "__main__":
    main()
