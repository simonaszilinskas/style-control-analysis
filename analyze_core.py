#!/usr/bin/env python3
"""
Core formatting analysis on the comparia-fr-arena battle table (§4.1-4.6).

Standard vs formatting-style-controlled Bradley-Terry, rank changes with 1000x
bootstrap + Benjamini-Hochberg, position bias, arena modes, and the single-
feature ablation. Reads fr_battles.parquet (one row per decisive French battle,
votes only). This replaces the votes+reactions clean_and_analyze.py pipeline;
there are no reactions in comparia-fr-arena.

    python analyze_core.py     # -> core_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
SCALE, BASE, INIT = 400, 10, 1000
MIN_BATTLES = 100
N_BOOT = 1000
FORMATTING = ["headers", "lists", "bold", "code_blocks", "emoji"]


def design(d, models, feats):
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    Xm = np.zeros((len(d), n))
    rows = np.arange(len(d))
    Xm[rows, d["model_a_name"].map(idx).to_numpy()] = 1
    Xm[rows, d["model_b_name"].map(idx).to_numpy()] = -1
    y = (d["winner"].to_numpy() == "model_a").astype(float)
    if feats:
        cols = [d[f"{f}_a"].to_numpy(float) - d[f"{f}_b"].to_numpy(float) for f in feats]
        Xs = StandardScaler().fit_transform(np.column_stack(cols))
    else:
        Xs = np.empty((len(d), 0))
    return Xm, Xs, y


def fit(d, models, feats):
    Xm, Xs, y = design(d, models, feats)
    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(np.hstack([Xm, Xs]), y)
    n = len(models)
    ratings = dict(zip(models, INIT + SCALE * lr.coef_[0][:n] / np.log(BASE)))
    coefs = dict(zip(feats, lr.coef_[0][n:]))
    return ratings, coefs


def bh(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    for i, idx in enumerate(order):
        adj[idx] = p[idx] * m / (i + 1)
    for i in range(m - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return np.minimum(adj, 1.0)


def ranks(r):
    return {m: i + 1 for i, m in enumerate(sorted(r, key=lambda x: -r[x]))}


def main():
    from scipy.stats import binomtest
    res = {}
    b = pd.read_parquet("fr_battles.parquet")
    d = b[b["winner"].isin(["model_a", "model_b"])].dropna(
        subset=[f"{f}_{s}" for f in FORMATTING for s in ("a", "b")]).copy()
    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy()
    print(f"battles: {len(d):,}  models: {len(models)}")
    res["n_battles"], res["n_models"] = int(len(d)), len(models)

    standard, _ = fit(d, models, [])
    controlled, coefs = fit(d, models, FORMATTING)
    res["style_coefficients"] = {f: {"coef": float(coefs[f]),
                                     "odds_pct": float((np.exp(coefs[f]) - 1) * 100)}
                                 for f in FORMATTING}
    r_corr = float(np.corrcoef([standard[m] for m in models],
                               [controlled[m] for m in models])[0, 1])
    res["bt_correlation"] = r_corr
    rk_s, rk_c = ranks(standard), ranks(controlled)
    print(f"corr(standard, formatting-controlled) = {r_corr:.4f}")
    for f in FORMATTING:
        print(f"  {f:12s} {coefs[f]:+.4f} ({(np.exp(coefs[f])-1)*100:+.1f}%)")

    # Ablation: one feature at a time.
    res["ablation"] = {}
    for f in FORMATTING:
        _, c1 = fit(d, models, [f])
        res["ablation"][f] = {"coef": float(c1[f]), "odds_pct": float((np.exp(c1[f]) - 1) * 100)}

    # Position bias.
    aw = int((d["winner"] == "model_a").sum())
    tot = len(d)
    bt = binomtest(aw, tot, 0.5)
    res["position_bias"] = {"a_win_rate": aw / tot, "p_value": float(bt.pvalue),
                            "a_wins": aw, "b_wins": tot - aw}
    print(f"position bias: A wins {aw/tot*100:.2f}%  p={bt.pvalue:.4f}")

    # Arena modes.
    res["modes"] = {}
    if "mode" in d.columns:
        for m, sub in d.groupby("mode"):
            if len(sub) >= 50:
                res["modes"][str(m)] = {"n": int(len(sub)),
                                        "a_win_rate": float((sub["winner"] == "model_a").mean())}

    # Bootstrap rank changes + BH.
    print(f"bootstrapping ({N_BOOT}x)...")
    boot_s = {m: [] for m in models}
    boot_c = {m: [] for m in models}
    boot_coef = {f: [] for f in FORMATTING}
    for i in range(N_BOOT):
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N_BOOT}")
        s = d.sample(n=len(d), replace=True)
        try:
            rs, _ = fit(s, models, [])
            rc, cc = fit(s, models, FORMATTING)
            for m in models:
                boot_s[m].append(rs[m])
                boot_c[m].append(rc[m])
            for f in FORMATTING:
                boot_coef[f].append(cc[f])
        except Exception:
            continue

    # style coefficient CIs + BH
    cpv = {}
    for f in FORMATTING:
        arr = np.array(boot_coef[f])
        pt = coefs[f]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * max(np.mean(arr <= 0) if pt >= 0 else np.mean(arr >= 0), 1 / (len(arr) + 1))
        cpv[f] = min(p, 1.0)
        res["style_coefficients"][f].update(
            {"ci": [float(lo), float(hi)], "odds_ci": [float((np.exp(lo)-1)*100), float((np.exp(hi)-1)*100)], "p": float(min(p, 1.0))})
    for f, a in zip(FORMATTING, bh([cpv[f] for f in FORMATTING])):
        res["style_coefficients"][f]["p_bh"] = float(a)
        res["style_coefficients"][f]["sig_bh"] = bool(a < 0.05)

    # rank change significance + BH
    sig = []
    for m in models:
        diffs = np.array(boot_c[m]) - np.array(boot_s[m])
        pt = controlled[m] - standard[m]
        p = 2 * max(np.mean(diffs <= 0) if pt >= 0 else np.mean(diffs >= 0), 1 / (len(diffs) + 1))
        sig.append({"model": m, "rank_std": rk_s[m], "rank_ctrl": rk_c[m],
                    "rank_delta": rk_s[m] - rk_c[m], "rating_delta": float(pt),
                    "p": float(min(p, 1.0))})
    for e, a in zip(sig, bh([e["p"] for e in sig])):
        e["p_bh"] = float(a)
        e["sig_bh"] = bool(a < 0.05)
    sig.sort(key=lambda e: -abs(e["rank_delta"]))
    res["rank_changes"] = sig
    res["n_sig_bh"] = sum(e["sig_bh"] for e in sig)
    res["rankings"] = {m: {"rank_std": rk_s[m], "rank_ctrl": rk_c[m],
                           "rating_std": float(standard[m]), "rating_ctrl": float(controlled[m])}
                       for m in models}
    print(f"models with significant rating change (BH): {res['n_sig_bh']}/{len(models)}")

    with open("core_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved core_results.json")


if __name__ == "__main__":
    main()
