#!/usr/bin/env python3
"""
Robustness: are the associations stable on random-pair battles only? (§4.9)

Custom (user-selected) pairings are non-random: a user picks models for reasons
that may correlate with expected quality and style. The cleanest primary would
be random-pair battles. We refit the headline coefficients (formatting-only, and
the joint bold/length/MATTR) on random battles only and compare to the full
sample. If they agree, the mode mix is not driving the result.

    python src/robustness_random.py    # -> results/robustness_random_results.json
"""

import json
import numpy as np
import pandas as pd

from analyze_core import fit, MIN_BATTLES, FORMATTING
from linguistic_analysis import fit_bt, LINGUISTIC
from paths import BATTLES, RESULTS

CORE = FORMATTING + ["length"] + LINGUISTIC


def _prep(d):
    counts = pd.concat([d["model_a_name"], d["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    return d[d["model_a_name"].isin(models) & d["model_b_name"].isin(models)].copy(), models


def main():
    b = pd.read_parquet(BATTLES)
    dec = b[b["winner"].isin(["model_a", "model_b"])]
    res = {}
    for label, sub in [("all", dec),
                       ("random", dec[dec["mode"] == "random"])]:
        d = sub.dropna(subset=[f"{f}_{s}" for f in FORMATTING for s in ("a", "b")])
        d, models = _prep(d)
        _, fcoef = fit(d, models, FORMATTING)      # formatting-only
        dj = sub.dropna(subset=[f"{f}_{s}" for f in CORE for s in ("a", "b")])
        dj, mj = _prep(dj)
        _, jcoef, _, _, _ = fit_bt(dj, mj, CORE)   # joint (winsorized)
        def odds_pct(coef):
            return float((np.exp(coef) - 1) * 100)

        res[label] = {
            "n_formatting": int(len(d)), "n_joint": int(len(dj)), "n_models": len(models),
            "formatting": {f: odds_pct(fcoef[f]) for f in FORMATTING},
            "joint": {
                f: odds_pct(jcoef[f])
                for f in ["bold", "length", "mattr", "ttr", "headers", "lists"]
            },
        }
        print(f"\n{label} (formatting n={len(d):,}, joint n={len(dj):,}, models {len(models)})")
        print(
            "  formatting-only: "
            + "  ".join(f"{f}={odds_pct(fcoef[f]):+.1f}%" for f in FORMATTING)
        )
        print(
            "  joint: "
            + "  ".join(
                f"{f}={odds_pct(jcoef[f]):+.1f}%"
                for f in ["bold", "length", "mattr", "ttr"]
            )
        )

    with open(RESULTS / "robustness_random_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved results/robustness_random_results.json")


if __name__ == "__main__":
    main()
