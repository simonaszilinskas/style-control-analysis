#!/usr/bin/env python3
"""
Compar:IA Style Control, Linguistic Extension

The main pipeline (clean_and_analyze.py) controls preferences for markdown
*formatting* (headers, lists, bold, code blocks, emoji) but not for answer
*length* or for *linguistic* properties of the text. This script adds both and
asks the question the formatting-only model cannot: once we also hold length,
readability, lexical diversity and sentence structure fixed, which presentation
features still move a vote, and do the formatting effects survive?

The linguistic features (readability REL/CLI/FKG, diversity TTR/MATTR, structure
ASL/long-sentence-ratio, and CamemBERT pseudo-perplexity) were computed by
Maayeesha Farzana for her internship analysis and are merged in here, keyed by
conversation_pair_id, onto the exact same battle table the formatting analysis
uses (75.7% of battles carry them; the A/B framing matches 1:1). Length is the
model's output-token count from the dataset metadata, not a whitespace word
count, which also settles the French-tokenization concern raised in the review.

Method mirrors clean_and_analyze.py exactly (per-model Bradley-Terry dummies +
standardized A-B style contrasts, sklearn logistic, 1000x bootstrap, Benjamini-
Hochberg), so every coefficient here is directly comparable to the formatting
coefficients there.

    python linguistic_analysis.py     # -> linguistic_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

SCALE, BASE, INIT_RATING = 400, 10, 1000
MIN_BATTLES = 100
N_BOOTSTRAP = 1000

FORMATTING = ["headers", "lists", "bold", "code_blocks", "emoji"]
READABILITY = ["rel", "cli", "fkg"]
DIVERSITY = ["ttr", "mattr"]
STRUCTURE = ["asl", "long_sent_ratio"]
LINGUISTIC = READABILITY + DIVERSITY + STRUCTURE


# --------------------------------------------------------------------------- #
# Bradley-Terry (identical construction to clean_and_analyze.py)
# --------------------------------------------------------------------------- #
def _design(battles, models, style_features):
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    d = battles[battles["winner"].isin(["model_a", "model_b"])]
    if style_features:
        d = d.dropna(subset=[f"{s}_a" for s in style_features]
                     + [f"{s}_b" for s in style_features])
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)]

    Xm = np.zeros((len(d), n))
    rows = np.arange(len(d))
    Xm[rows, d["model_a_name"].map(idx).to_numpy()] = 1
    Xm[rows, d["model_b_name"].map(idx).to_numpy()] = -1
    y = (d["winner"].to_numpy() == "model_a").astype(float)

    if style_features:
        cols = []
        for s in style_features:
            diff = d[f"{s}_a"].to_numpy(float) - d[f"{s}_b"].to_numpy(float)
            # Winsorize each A-B contrast at 1/99% before standardizing. Several
            # features are heavy-tailed (perplexity, readability grades, raw bold
            # counts up to ~1000); without this a handful of extreme battles drive
            # the coefficient. Bounded features (TTR/MATTR in [0,1]) are unaffected.
            lo, hi = np.nanpercentile(diff, [1, 99])
            cols.append(np.clip(diff, lo, hi))
        Xs = StandardScaler().fit_transform(np.column_stack(cols))
    else:
        Xs = np.empty((len(d), 0))
    return Xm, Xs, y


def fit_bt(battles, models, style_features):
    Xm, Xs, y = _design(battles, models, style_features)
    X = np.hstack([Xm, Xs])
    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)
    n = len(models)
    ratings = dict(zip(models, INIT_RATING + SCALE * lr.coef_[0][:n] / np.log(BASE)))
    coefs = dict(zip(style_features, lr.coef_[0][n:]))
    return ratings, coefs, y, lr, X


def fit_pooled(battles, models, style_features):
    """Reduced-form logit: style contrasts + intercept, NO per-model dummies.
    The gap between this and fit_bt is the part of a style effect that was really
    model strength (the confounder-vs-mediator question, seen feature by feature)."""
    _, Xs, y = _design(battles, models, style_features)
    X = np.column_stack([np.ones(len(Xs)), Xs])
    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)
    return dict(zip(style_features, lr.coef_[0][1:]))


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (same procedure as the main script)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    for i, idx in enumerate(order):
        adj[idx] = p[idx] * m / (i + 1)
    for i in range(m - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return np.minimum(adj, 1.0)


def log_likelihood(y, X, lr):
    p = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------------------------------------------------------- #
def main():
    results = {}
    battles = pd.read_parquet("battles_bt_styled.parquet")
    ling = pd.read_parquet("linguistic_features.parquet")
    n_before = len(battles)
    battles = battles.merge(ling, on="conversation_pair_id", how="left")
    have_ling = battles["rel_a"].notna()
    have_ppl = battles["ppl_a"].notna()
    print(f"Battles: {n_before:,}; with linguistic features: {have_ling.sum():,} "
          f"({have_ling.mean()*100:.1f}%); with perplexity: {have_ppl.sum():,}")
    results["coverage"] = {"battles": int(n_before),
                           "with_linguistic": int(have_ling.sum()),
                           "with_perplexity": int(have_ppl.sum())}

    # Common support: battles that carry every formatting + length + linguistic
    # feature, so all nested models are compared on identical data.
    core = FORMATTING + ["length"] + LINGUISTIC
    sub = battles.dropna(subset=[f"{f}_{s}" for f in core for s in ("a", "b")]).copy()
    counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    sub = sub[sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)]
    print(f"Common support: {len(sub):,} battles, {len(models)} models "
          f"(>= {MIN_BATTLES} battles)")
    results["common_support"] = {"battles": int(len(sub)), "models": len(models)}

    # Nested feature sets.
    specs = {
        "formatting": FORMATTING,
        "formatting+length": FORMATTING + ["length"],
        "joint": FORMATTING + ["length"] + LINGUISTIC,
    }
    fits, lls = {}, {}
    for name, feats in specs.items():
        ratings, coefs, y, lr, X = fit_bt(sub, models, feats)
        fits[name] = coefs
        lls[name] = {"loglik": log_likelihood(y, X, lr), "n": int(len(y)),
                     "k": X.shape[1], "acc": float(lr.score(X, y))}
    results["bt_coefficients"] = fits
    results["model_fit"] = lls

    print("\nStyle coefficients across nested models (log-odds per SD):")
    allf = FORMATTING + ["length"] + LINGUISTIC
    hdr = f"  {'feature':16s}" + "".join(f"{n:>20s}" for n in specs)
    print(hdr)
    for f in allf:
        cells = "".join(f"{('' if f not in fits[n] else f'{fits[n][f]:+.4f}'):>20s}"
                        for n in specs)
        print(f"  {f:16s}{cells}")

    print("\nModel fit (higher loglik / acc = better):")
    for name, s in lls.items():
        print(f"  {name:20s} loglik={s['loglik']:.1f}  acc={s['acc']:.4f}  k={s['k']}")

    # Reduced-form vs BT for the joint set (confounder-vs-mediator, per feature).
    pooled = fit_pooled(sub, models, allf)
    results["pooled_vs_bt"] = {f: {"pooled": pooled[f], "bt": fits["joint"][f]} for f in allf}
    print("\nReduced-form (no model strengths) vs BT, joint set:")
    for f in allf:
        print(f"  {f:16s} pooled {pooled[f]:+.4f}   BT {fits['joint'][f]:+.4f}")

    # Bootstrap the joint coefficients (1000x) + BH, matching the main pipeline.
    print(f"\nBootstrapping joint coefficients ({N_BOOTSTRAP}x)...")
    boot = {f: [] for f in allf}
    for b in range(N_BOOTSTRAP):
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{N_BOOTSTRAP}")
        s = sub.sample(n=len(sub), replace=True)
        try:
            _, c, _, _, _ = fit_bt(s, models, allf)
            for f in allf:
                boot[f].append(c[f])
        except Exception:
            continue
    ci, pvals = {}, {}
    for f in allf:
        arr = np.array(boot[f])
        point = fits["joint"][f]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * max(np.mean(arr <= 0) if point >= 0 else np.mean(arr >= 0), 1 / (len(arr) + 1))
        ci[f] = {"point": float(point), "ci": [float(lo), float(hi)],
                 "odds_pct": float((np.exp(point) - 1) * 100), "p": float(min(p, 1.0))}
        pvals[f] = min(p, 1.0)
    adj = bh([pvals[f] for f in allf])
    for f, a in zip(allf, adj):
        ci[f]["p_bh"] = float(a)
        ci[f]["sig_bh"] = bool(a < 0.05)
    results["joint_coefficients"] = ci

    print("\nJoint coefficients with 95% CI (BH-corrected):")
    for f in allf:
        c = ci[f]
        print(f"  {f:16s} {c['point']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"({c['odds_pct']:+.1f}%)  p_BH={c['p_bh']:.4f}  {'***' if c['sig_bh'] else 'n.s.'}")

    # Perplexity: does it add anything on top of the joint set? (subset with ppl)
    ppl_sub = sub.dropna(subset=["ppl_a", "ppl_b"])
    if len(ppl_sub) > 1000:
        pc = pd.concat([ppl_sub["model_a_name"], ppl_sub["model_b_name"]]).value_counts()
        pm = sorted(pc[pc >= MIN_BATTLES].index)
        _, cppl, _, _, _ = fit_bt(ppl_sub, pm, allf + ["ppl"])
        results["perplexity"] = {"n_battles": int(len(ppl_sub)), "coef_ppl": float(cppl["ppl"]),
                                 "odds_pct": float((np.exp(cppl["ppl"]) - 1) * 100)}
        print(f"\nPerplexity (n={len(ppl_sub):,}): coef={cppl['ppl']:+.4f} "
              f"({(np.exp(cppl['ppl'])-1)*100:+.1f}% odds/SD) on top of the joint set")

    with open("linguistic_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nSaved linguistic_results.json")


if __name__ == "__main__":
    main()
