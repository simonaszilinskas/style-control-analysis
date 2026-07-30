#!/usr/bin/env python3
"""
MATTR stress tests (§4.10): is MATTR's length-independence and its association real? (part 1)

From the battle table (no re-streaming): response-length distribution, the share
of responses too short for the 50-token MATTR window, the MATTR-length
correlation overall and within length bins, and the joint MATTR coefficient
refit within short and long response strata. (The alternative-metric checks,
MTLD and content-word MATTR, need the raw text and live in mattr_alt.py.)

    python src/mattr_stress.py    # -> results/mattr_stress_results.json
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from linguistic_analysis import fit_bt, LINGUISTIC, FORMATTING, MIN_BATTLES
from paths import BATTLES, RESULTS

CORE = FORMATTING + ["length"] + LINGUISTIC


def main():
    b = pd.read_parquet(BATTLES)
    # per-response length and MATTR, stacking both sides.
    length = pd.concat([b["length_a"], b["length_b"]]).dropna()
    mattr = pd.concat([b["mattr_a"], b["mattr_b"]])
    both = pd.DataFrame({"length": pd.concat([b["length_a"], b["length_b"]]),
                         "mattr": mattr}).dropna()

    res = {}
    res["length_tokens"] = {"median": float(length.median()), "q25": float(length.quantile(.25)),
                            "q75": float(length.quantile(.75)), "p5": float(length.quantile(.05)),
                            "p95": float(length.quantile(.95))}
    # MATTR is NaN when a response has < 50 word tokens (its window); report that share.
    short = pd.concat([b["mattr_a"].isna(), b["mattr_b"].isna()]).mean()
    res["share_response_no_mattr"] = float(short)
    print(f"response length tokens: median {length.median():.0f}  IQR [{length.quantile(.25):.0f},{length.quantile(.75):.0f}]")
    print(f"share of responses with no MATTR (< 50 word tokens): {short*100:.1f}%")

    # MATTR-length correlation overall and by length quartile.
    rho = spearmanr(both["length"], both["mattr"]).statistic
    res["mattr_length_spearman_overall"] = float(rho)
    print(f"MATTR-length Spearman overall: {rho:+.3f}")
    both["bin"] = pd.qcut(both["length"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    res["mattr_length_spearman_by_bin"] = {}
    for name, g in both.groupby("bin", observed=True):
        r = spearmanr(g["length"], g["mattr"]).statistic
        res["mattr_length_spearman_by_bin"][str(name)] = float(r)
        print(f"  {name} (len {g['length'].min():.0f}-{g['length'].max():.0f}): rho {r:+.3f}")

    # Joint MATTR coefficient within short vs long response strata (by max response length).
    dec = b[b["winner"].isin(["model_a", "model_b"])].dropna(
        subset=[f"{f}_{s}" for f in CORE for s in ("a", "b")]).copy()
    dec["maxlen"] = dec[["length_a", "length_b"]].max(axis=1)
    med = dec["maxlen"].median()
    res["mattr_by_stratum"] = {}
    def odds_pct(coef):
        return float((np.exp(coef) - 1) * 100)

    for label, sub in [("short", dec[dec["maxlen"] <= med]), ("long", dec[dec["maxlen"] > med])]:
        counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
        models = sorted(counts[counts >= MIN_BATTLES].index)
        s = sub[sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)]
        _, coef, _, _, _ = fit_bt(s, models, CORE)
        res["mattr_by_stratum"][label] = {
            "n": int(len(s)),
            "mattr": odds_pct(coef["mattr"]),
            "bold": odds_pct(coef["bold"]),
            "length": odds_pct(coef["length"]),
        }
        print(
            f"joint {label} answers (n={len(s):,}): "
            f"MATTR {odds_pct(coef['mattr']):+.1f}%  "
            f"length {odds_pct(coef['length']):+.1f}%"
        )

    with open(RESULTS / "mattr_stress_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved results/mattr_stress_results.json")


if __name__ == "__main__":
    main()
