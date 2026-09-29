#!/usr/bin/env python3
"""
Formatting pairwise interactions (reviewer request: does bold x headers etc. matter?).

Same battle table and filtering as analyze_core.py (decisive votes, dropna on
the 5 formatting features both sides, models with >= 100 battles,
winsorize=False -- the published core formatting spec). Adds all 10 pairwise
products of the standardized formatting contrasts to the Bradley-Terry design,
standardizing the products themselves so every coefficient is per-SD. Bootstrap
400x for CIs/p on the interaction terms, BH across the 10-term family, and a
likelihood-ratio test against the no-interaction model.

    python formatting_interactions.py   # -> formatting_interaction_results.json
"""

import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from modeling import FORMATTING, benjamini_hochberg as bh, design_matrix, fit_bt, log_likelihood
from paths import BATTLES, RESULTS

np.random.seed(42)
MIN_BATTLES = 100
N_BOOT = 400

PAIRS = list(itertools.combinations(FORMATTING, 2))


def fit_interaction(d, models):
    """Model contrasts + standardized formatting contrasts + standardized
    pairwise products of those contrasts."""
    model_design, style_design, outcomes = design_matrix(d, models, FORMATTING, winsorize=False)
    products = np.column_stack([
        style_design[:, FORMATTING.index(a)] * style_design[:, FORMATTING.index(b)]
        for a, b in PAIRS
    ])
    inter_design = StandardScaler().fit_transform(products)
    design = np.hstack([model_design, style_design, inter_design])

    estimator = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    estimator.fit(design, outcomes)

    n, k = len(models), len(FORMATTING)
    main = dict(zip(FORMATTING, estimator.coef_[0][n:n + k]))
    inter = {f"{a}_x_{b}": c for (a, b), c in zip(PAIRS, estimator.coef_[0][n + k:])}
    return main, inter, outcomes, estimator, design


def main():
    res = {}
    b = pd.read_parquet(BATTLES)
    d = b[b["winner"].isin(["model_a", "model_b"])].dropna(
        subset=[f"{f}_{s}" for f in FORMATTING for s in ("a", "b")]).copy()
    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy()
    print(f"battles: {len(d):,}  models: {len(models)}")
    res["n_battles"], res["n_models"] = int(len(d)), len(models)

    # No-interaction model (published spec) for the LR test.
    _, coefs0, outcomes0, est0, design0 = fit_bt(d, models, FORMATTING, winsorize=False)

    # Interaction model.
    main_coefs, inter, outcomes1, est1, design1 = fit_interaction(d, models)
    res["main_effects"] = {f: {"coef": float(main_coefs[f]),
                               "odds_pct": float((np.exp(main_coefs[f]) - 1) * 100)}
                           for f in FORMATTING}
    print("main effects (interaction model):")
    for f in FORMATTING:
        print(f"  {f:12s} {main_coefs[f]:+.4f} ({(np.exp(main_coefs[f]) - 1) * 100:+.1f}%)")

    # Bootstrap the 10 interaction terms.
    print(f"bootstrapping ({N_BOOT}x)...")
    boot = {k: [] for k in inter}
    for i in range(N_BOOT):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_BOOT}")
        s = d.sample(n=len(d), replace=True)
        try:
            _, bi, _, _, _ = fit_interaction(s, models)
            for k in boot:
                boot[k].append(bi[k])
        except Exception:
            continue

    res["interactions"] = {}
    pvals = {}
    for k in inter:
        arr = np.array(boot[k])
        pt = inter[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * max(np.mean(arr <= 0) if pt >= 0 else np.mean(arr >= 0), 1 / (len(arr) + 1))
        pvals[k] = min(p, 1.0)
        res["interactions"][k] = {
            "coef": float(pt),
            "odds_pct": float((np.exp(pt) - 1) * 100),
            "ci": [float(lo), float(hi)],
            "odds_ci": [float((np.exp(lo) - 1) * 100), float((np.exp(hi) - 1) * 100)],
            "p": float(pvals[k]),
        }
    for k, a in zip(inter, bh([pvals[k] for k in inter])):
        res["interactions"][k]["p_bh"] = float(a)
        res["interactions"][k]["sig_bh"] = bool(a < 0.05)

    print(f"\n  {'term':22s}{'odds/SD':>10s}{'95% CI (odds%)':>22s}{'p_bh':>10s}  sig")
    for k in inter:
        c = res["interactions"][k]
        star = "yes" if c["sig_bh"] else "no"
        print(f"  {k:22s}{c['odds_pct']:+9.2f}%"
              f"  [{c['odds_ci'][0]:+.2f},{c['odds_ci'][1]:+.2f}]{c['p_bh']:10.4f}  {star}")
    n_sig = sum(res["interactions"][k]["sig_bh"] for k in inter)
    res["n_sig_bh"] = n_sig
    print(f"significant interaction terms (BH): {n_sig}/{len(inter)}")

    # Model comparison: with vs without interactions.
    ll0 = log_likelihood(outcomes0, design0, est0)
    ll1 = log_likelihood(outcomes1, design1, est1)
    acc0 = float(est0.score(design0, outcomes0))
    acc1 = float(est1.score(design1, outcomes1))
    lr_stat = 2 * (ll1 - ll0)
    lr_p = float(chi2.sf(lr_stat, df=len(PAIRS)))
    res["model_comparison"] = {
        "log_likelihood_no_interaction": float(ll0),
        "log_likelihood_interaction": float(ll1),
        "accuracy_no_interaction": acc0,
        "accuracy_interaction": acc1,
        "lr_statistic": float(lr_stat),
        "lr_df": len(PAIRS),
        "lr_p_value": lr_p,
    }
    print(f"\naccuracy: no-interaction={acc0:.4f}  interaction={acc1:.4f}")
    print(f"log-likelihood: no-interaction={ll0:.2f}  interaction={ll1:.2f}")
    print(f"LR statistic={lr_stat:.2f}  df={len(PAIRS)}  p={lr_p:.4f}")

    with open(RESULTS / "formatting_interaction_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved formatting_interaction_results.json")


if __name__ == "__main__":
    main()
