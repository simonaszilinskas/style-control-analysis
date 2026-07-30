#!/usr/bin/env python3
"""
Compar:IA Style Control, Topic Controls

Presentation features may proxy for topic: technical questions invite code
blocks and lists, some subjects invite longer answers. Because topic is a
property of the shared prompt, it differences out of the pairwise A-B model
(Delta topic = 0), so "controlling for topic" here means letting each topic have
its own style slope, i.e. topic x style interactions, not a topic main effect.

Topic comes from the conversations export `categories` field (an LLM-assigned
taxonomy, ~18 classes, present for 94% of battles). A conversation can carry two
categories; we use the first as its primary topic.

Two questions:

  Q1. How do formatting associations vary by topic? Fit the headline formatting
      Bradley-Terry model *within* each major topic and compare bold/lists/
      headers. Similar estimates are a descriptive sensitivity check, not proof
      that topic-related confounding is absent.

  Q2. Is the vote-time conversation-depth interaction robust to topic? Re-fit the
      formatting x multi-turn interaction model with topic x formatting
      interactions added, so every topic gets its own formatting slope. Compare
      the resulting multi-turn interactions with the primary specification;
      this does not eliminate unmeasured confounding.

    python topic_analysis.py   # -> topic_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from linguistic_analysis import bh, FORMATTING
from turn_depth_analysis import _contrasts
from paths import BATTLES, RESULTS

np.random.seed(42)

MIN_BATTLES = 100          # min battles per model, pooled
MIN_TOPIC_MODEL = 50       # min battles per model within a topic stratum
MIN_TOPIC_STRATUM = 2500   # min battles for a topic to get its own stratified fit
MIN_TOPIC_DUMMY = 1500     # min battles for a topic to get its own interaction dummy
N_BOOT_Q1 = 400
N_BOOT_Q2 = 1000


def _load():
    # comparia-fr-arena battle table carries primary_topic (categories[0]) and conv_turns.
    b = pd.read_parquet(BATTLES).rename(columns={"primary_topic": "topic"})
    d = b[b["winner"].isin(["model_a", "model_b"])].copy()
    d = d.dropna(subset=["topic"] + [f"{s}_a" for s in FORMATTING] + [f"{s}_b" for s in FORMATTING])
    return d


def _fit_formatting(d, models, scaler):
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    Xm = np.zeros((len(d), n))
    rows = np.arange(len(d))
    Xm[rows, d["model_a_name"].map(idx).to_numpy()] = 1
    Xm[rows, d["model_b_name"].map(idx).to_numpy()] = -1
    y = (d["winner"].to_numpy() == "model_a").astype(float)
    Xs = scaler.transform(_contrasts(d, FORMATTING))
    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(np.hstack([Xm, Xs]), y)
    return dict(zip(FORMATTING, lr.coef_[0][n:]))


# --------------------------------------------------------------------------- #
# Q1: formatting association within each topic
# --------------------------------------------------------------------------- #
def q1_topic_stratified():
    d = _load()
    scaler = StandardScaler().fit(_contrasts(d, FORMATTING))  # shared scale across topics
    topics = [t for t, c in d["topic"].value_counts().items() if c >= MIN_TOPIC_STRATUM]
    print(f"\n=== Q1: formatting association within each topic ({len(topics)} topics) ===")

    out = {}
    # pooled reference (all topics, same model set logic) for comparison
    for label, sub in [("ALL", d)] + [(t, d[d["topic"] == t]) for t in topics]:
        counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
        thr = MIN_BATTLES if label == "ALL" else MIN_TOPIC_MODEL
        models = sorted(counts[counts >= thr].index)
        s = sub[sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)].copy()
        point = _fit_formatting(s, models, scaler)
        boot = {f: [] for f in FORMATTING}
        for b in range(N_BOOT_Q1):
            r = s.sample(n=len(s), replace=True)
            try:
                c = _fit_formatting(r, models, scaler)
                for f in FORMATTING:
                    boot[f].append(c[f])
            except Exception:
                continue
        rec = {"n_battles": int(len(s)), "n_models": len(models), "coef": {}}
        for f in FORMATTING:
            arr = np.array(boot[f])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rec["coef"][f] = {"beta": float(point[f]),
                              "odds_pct": float((np.exp(point[f]) - 1) * 100),
                              "ci": [float(lo), float(hi)]}
        out[label] = rec
        cells = "  ".join(f"{f}={(np.exp(point[f])-1)*100:+5.1f}%" for f in FORMATTING)
        print(f"  {label[:34]:34s} n={len(s):6d} m={len(models):3d}  {cells}")
    return out


# --------------------------------------------------------------------------- #
# Q2: vote-time conversation-depth interaction, with topic x formatting interactions
# --------------------------------------------------------------------------- #
def _primary_topic_dummies(d):
    counts = d["topic"].value_counts()
    keep = [t for t, c in counts.items() if c >= MIN_TOPIC_DUMMY]
    topic = d["topic"].where(d["topic"].isin(keep), "Other")
    levels = sorted(topic.unique())
    ref = levels[0]  # reference topic absorbed by the per-model + style baseline
    return topic, [t for t in levels if t != ref], ref


def _fit_q2(d, models, topic_levels, topic_series, scaler):
    """Model dummies + formatting contrasts + contrast x multiturn + contrast x topic."""
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    Xm = np.zeros((len(d), n))
    rows = np.arange(len(d))
    Xm[rows, d["model_a_name"].map(idx).to_numpy()] = 1
    Xm[rows, d["model_b_name"].map(idx).to_numpy()] = -1
    y = (d["winner"].to_numpy() == "model_a").astype(float)

    Xs = scaler.transform(_contrasts(d, FORMATTING))          # k main effects
    mt = d["multiturn"].to_numpy(float)[:, None]
    Xmt = Xs * mt                                             # k multiturn interactions
    blocks = [Xm, Xs, Xmt]
    for t in topic_levels:                                    # k interactions per extra topic
        ind = (topic_series.to_numpy() == t).astype(float)[:, None]
        blocks.append(Xs * ind)
    X = np.hstack(blocks)
    lr = LogisticRegression(fit_intercept=False, penalty=None, max_iter=5000)
    lr.fit(X, y)
    k = len(FORMATTING)
    return dict(zip(FORMATTING, lr.coef_[0][n + k:n + 2 * k]))   # multiturn interactions


def q2_reading_depth_topic_controlled():
    d = _load()
    d = d.dropna(subset=["conv_turns"])
    d["multiturn"] = (d["conv_turns"] >= 2).astype(int)
    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    d = d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy()
    topic_series, topic_levels, ref = _primary_topic_dummies(d)
    print("\n=== Q2: conversation-length interaction with topic x formatting controls ===")
    print(f"battles: {len(d):,}  single: {(d.multiturn==0).sum():,}  "
          f"multi: {(d.multiturn==1).sum():,}  models: {len(models)}  "
          f"topic dummies: {len(topic_levels)} (ref={ref})")

    scaler = StandardScaler().fit(_contrasts(d, FORMATTING))
    point = _fit_q2(d, models, topic_levels, topic_series, scaler)
    boot = {f: [] for f in FORMATTING}
    for b in range(N_BOOT_Q2):
        if (b + 1) % 250 == 0:
            print(f"    {b+1}/{N_BOOT_Q2}")
        ridx = np.random.randint(0, len(d), len(d))
        r = d.iloc[ridx]
        ts = topic_series.iloc[ridx]
        try:
            c = _fit_q2(r, models, topic_levels, ts, scaler)
            for f in FORMATTING:
                boot[f].append(c[f])
        except Exception:
            continue

    out = {}
    pvals = {}
    for f in FORMATTING:
        arr = np.array(boot[f])
        pt = point[f]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * max(np.mean(arr <= 0) if pt >= 0 else np.mean(arr >= 0), 1 / (len(arr) + 1))
        pvals[f] = min(p, 1.0)
        out[f] = {"interaction": float(pt), "ci": [float(lo), float(hi)], "p": float(min(p, 1.0))}
    adj = bh([pvals[f] for f in FORMATTING])
    for f, a in zip(FORMATTING, adj):
        out[f]["p_bh"] = float(a)
        out[f]["sig_bh"] = bool(a < 0.05)
    print("\n  multi-turn interaction (topic-controlled), BH:")
    for f in FORMATTING:
        c = out[f]
        print(f"    {f:12s} {c['interaction']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"p_BH={c['p_bh']:.4f} {'***' if c['sig_bh'] else 'n.s.'}")
    return {"n_battles": int(len(d)), "n_models": len(models),
            "n_topic_dummies": len(topic_levels), "interactions": out}


def main():
    results = {"q1_topic_stratified": q1_topic_stratified(),
               "q2_reading_depth_topic_controlled": q2_reading_depth_topic_controlled()}
    with open(RESULTS / "topic_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nSaved topic_results.json")


if __name__ == "__main__":
    main()
