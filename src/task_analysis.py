#!/usr/bin/env python3
"""
Task-type controls (§4.8): is the formatting premium a proxy for task form?

Task type (write code / translate / summarize / draft / ...) drives presentation
more directly than subject matter: a list request produces lists, a coding
request produces code blocks. Like topic (§4.7), task is a property of the shared
prompt, so it differences out of the pairwise model and can only enter through
task x style interactions. We refit the formatting Bradley-Terry model within
each task and check whether the bold premium holds; we also read code_blocks
per task as a sanity check (it should matter where code is actually requested).

    python src/task_analysis.py    # -> results/task_results.json
"""

import json
import numpy as np
import pandas as pd

from analyze_core import fit, MIN_BATTLES, FORMATTING
from paths import BATTLES, DATA, RESULTS

MIN_TASK_MODEL = 50       # min battles per model within a task stratum
MIN_TASK_STRATUM = 2500   # min battles for a task to get its own stratified fit
N_BOOT = 400
SEED = 42


def _load():
    b = pd.read_parquet(BATTLES)
    tasks = pd.read_parquet(DATA / "battle_tasks.parquet")
    b = b.merge(tasks, on="conversation_pair_id", how="left")
    task_coverage = float(b["task"].notna().mean())
    d = b[b["winner"].isin(["model_a", "model_b"])].copy()
    d = d.dropna(subset=["task"] + [f"{s}_a" for s in FORMATTING] + [f"{s}_b" for s in FORMATTING])
    return d, task_coverage


def main():
    rng = np.random.default_rng(SEED)
    d, task_coverage = _load()
    print(f"battles with a task label: {task_coverage*100:.3f}%")
    strata = [t for t, c in d["task"].value_counts().items()
              if c >= MIN_TASK_STRATUM and t != "other"]
    print(f"tasks analysed: {strata}")

    res = {}
    for label, sub in [("ALL", d)] + [(t, d[d["task"] == t]) for t in strata]:
        counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
        thr = MIN_BATTLES if label == "ALL" else MIN_TASK_MODEL
        models = sorted(counts[counts >= thr].index)
        s = sub[sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)].copy()
        point = fit(s, models, FORMATTING)[1]
        boot = {f: [] for f in FORMATTING}
        for _ in range(N_BOOT):
            indices = rng.integers(0, len(s), size=len(s))
            r = s.iloc[indices]
            try:
                c = fit(r, models, FORMATTING)[1]
                for f in FORMATTING:
                    boot[f].append(c[f])
            except Exception:
                continue
        rec = {"n_battles": int(len(s)), "n_models": len(models), "coef": {}}
        for f in FORMATTING:
            arr = np.array(boot[f])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rec["coef"][f] = {"odds_pct": float((np.exp(point[f]) - 1) * 100),
                              "odds_ci": [float((np.exp(lo) - 1) * 100), float((np.exp(hi) - 1) * 100)]}
        res[label] = rec
        cells = "  ".join(f"{f}={rec['coef'][f]['odds_pct']:+5.1f}%" for f in FORMATTING)
        print(f"  {label[:14]:14s} n={len(s):6d} m={len(models):3d}  {cells}")

    output = {
        "analysis": {
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": SEED,
            "intervals": "unadjusted percentile intervals; exploratory",
            "task_label_coverage": task_coverage,
        },
        "strata": res,
    }
    with open(RESULTS / "task_results.json", "w") as fh:
        json.dump(output, fh, indent=2)
    print("Saved results/task_results.json")


if __name__ == "__main__":
    main()
