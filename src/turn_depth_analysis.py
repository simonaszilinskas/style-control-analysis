#!/usr/bin/env python3
"""
Compar:IA Style Control, Vote-Time Conversation-Depth Extension

This exploratory analysis tests whether presentation associations differ
between conversation prefixes with one visible user turn and those with two or
more at the retained vote. The battle builder truncates both response text and
turn count at that vote. The interaction remains observational: pre-vote depth
is endogenous and is not itself a direct measure of attentiveness.

The test is a single pooled Bradley-Terry model with per-model strengths, the
standardized A-B style contrasts, and each contrast interacted with a multi-turn
indicator. The interaction coefficient is the object of interest: negative means
the feature-vote association is smaller in the visible multi-turn stratum. Standardization is
done once on the full vote sample so the main effect and the interaction share a
scale. We bootstrap 1000x (resampling battles) and apply Benjamini-Hochberg.

    python turn_depth_analysis.py   # -> turn_depth_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from linguistic_analysis import bh, FORMATTING, LINGUISTIC
from paths import BATTLES, RESULTS

np.random.seed(42)

MIN_BATTLES = 100
N_BOOTSTRAP = 1000


def _contrasts(d, feats):
    """Standardized, winsorized A-B contrasts (same construction as
    linguistic_analysis._design), fit on whatever rows are passed."""
    cols = []
    for s in feats:
        diff = d[f"{s}_a"].to_numpy(float) - d[f"{s}_b"].to_numpy(float)
        lo, hi = np.nanpercentile(diff, [1, 99])
        cols.append(np.clip(diff, lo, hi))
    return np.column_stack(cols)


def _fit_interaction(d, models, feats, scaler):
    """Pooled BT: per-model dummies + standardized contrasts + contrast x multiturn.
    scaler is pre-fit on the full sample so single/multi share a feature scale."""
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    Xm = np.zeros((len(d), n))
    rows = np.arange(len(d))
    Xm[rows, d["model_a_name"].map(idx).to_numpy()] = 1
    Xm[rows, d["model_b_name"].map(idx).to_numpy()] = -1
    y = (d["winner"].to_numpy() == "model_a").astype(float)

    Xs = scaler.transform(_contrasts(d, feats))
    mt = d["multiturn"].to_numpy(float)[:, None]
    Xi = Xs * mt  # interaction: contrast active only when multi-turn
    X = np.hstack([Xm, Xs, Xi])

    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)
    k = len(feats)
    main = dict(zip(feats, lr.coef_[0][n:n + k]))
    inter = dict(zip(feats, lr.coef_[0][n + k:n + 2 * k]))
    return main, inter


def _fit_subset(d, models, feats, scaler):
    """Formatting BT on one subset, contrasts on the shared scaler (for the
    descriptive single-vs-multi coefficient comparison)."""
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    Xm = np.zeros((len(d), n))
    rows = np.arange(len(d))
    Xm[rows, d["model_a_name"].map(idx).to_numpy()] = 1
    Xm[rows, d["model_b_name"].map(idx).to_numpy()] = -1
    y = (d["winner"].to_numpy() == "model_a").astype(float)
    Xs = scaler.transform(_contrasts(d, feats))
    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(np.hstack([Xm, Xs]), y)
    return dict(zip(feats, lr.coef_[0][n:]))


def _prep(feats, require_ling=False):
    # comparia-fr-arena battle table already carries conv_turns and every feature.
    battles = pd.read_parquet(BATTLES)
    d = battles[battles["winner"].isin(["model_a", "model_b"])].copy()
    d = d.dropna(subset=["conv_turns"])
    d = d[d["conv_turns"] >= 1]
    d = d.dropna(subset=[f"{s}_a" for s in feats] + [f"{s}_b" for s in feats])
    d["multiturn"] = (d["conv_turns"] >= 2).astype(int)

    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy()
    return d, models


def run(feats, label, require_ling):
    d, models = _prep(feats, require_ling)
    n_single = int((d["multiturn"] == 0).sum())
    n_multi = int((d["multiturn"] == 1).sum())
    print(f"\n=== {label} ===")
    print(f"battles: {len(d):,}  single-turn: {n_single:,}  multi-turn: {n_multi:,}  "
          f"models: {len(models)}")

    scaler = StandardScaler().fit(_contrasts(d, feats))
    main, inter = _fit_interaction(d, models, feats, scaler)

    single = d[d["multiturn"] == 0]
    multi = d[d["multiturn"] == 1]
    c_single = _fit_subset(single, models, feats, scaler)
    c_multi = _fit_subset(multi, models, feats, scaler)

    print(f"\n  {'feature':14s}{'single':>10s}{'multi':>10s}{'interaction':>14s}")
    for f in feats:
        print(f"  {f:14s}{c_single[f]:+10.4f}{c_multi[f]:+10.4f}{inter[f]:+14.4f}")

    # Bootstrap the interaction coefficients (resample battles).
    print(f"  bootstrapping interactions ({N_BOOTSTRAP}x)...")
    boot = {f: [] for f in feats}
    for b in range(N_BOOTSTRAP):
        if (b + 1) % 250 == 0:
            print(f"    {b+1}/{N_BOOTSTRAP}")
        s = d.sample(n=len(d), replace=True)
        try:
            _, bi = _fit_interaction(s, models, feats, scaler)
            for f in feats:
                boot[f].append(bi[f])
        except Exception:
            continue

    out_feats = {}
    pvals = {}
    for f in feats:
        arr = np.array(boot[f])
        pt = inter[f]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * max(np.mean(arr <= 0) if pt >= 0 else np.mean(arr >= 0), 1 / (len(arr) + 1))
        pvals[f] = min(p, 1.0)
        out_feats[f] = {"single": float(c_single[f]), "multi": float(c_multi[f]),
                        "main": float(main[f]), "interaction": float(pt),
                        "interaction_ci": [float(lo), float(hi)], "p": float(min(p, 1.0))}
    adj = bh([pvals[f] for f in feats])
    for f, a in zip(feats, adj):
        out_feats[f]["p_bh"] = float(a)
        out_feats[f]["sig_bh"] = bool(a < 0.05)

    print(f"\n  Interaction (multi-turn slope shift), BH-corrected:")
    for f in feats:
        c = out_feats[f]
        star = "***" if c["sig_bh"] else "n.s."
        print(f"    {f:14s} {c['interaction']:+.4f} "
              f"[{c['interaction_ci'][0]:+.4f},{c['interaction_ci'][1]:+.4f}] "
              f"p_BH={c['p_bh']:.4f} {star}")

    return {"label": label, "n_battles": len(d), "n_single": n_single,
            "n_multi": n_multi, "n_models": len(models), "features": out_feats}


def main():
    results = {}
    # Primary: formatting only, all vote battles (max power, no linguistic merge).
    # This is exactly the "glance at the shape" channel: if presentation wins by a
    # quick look, its pull should fall once the reader engages over several turns.
    results["primary"] = run(FORMATTING, "Formatting only (all vote battles)",
                             require_ling=False)
    # Secondary: joint set on the linguistically-covered subset.
    results["joint"] = run(FORMATTING + ["length"] + LINGUISTIC,
                           "Joint (formatting + length + linguistic)", require_ling=True)
    with open(RESULTS / "turn_depth_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nSaved turn_depth_results.json")


if __name__ == "__main__":
    main()
