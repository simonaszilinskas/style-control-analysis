#!/usr/bin/env python3
"""
Time-block bootstrap (§4.9): are the headline CIs robust to temporal dependence?

comparia-fr-arena carries no user identifier, so we cannot cluster by user. As a
partial substitute for the independence assumption, we block-bootstrap by
calendar week: resample the ~89 weekly blocks with replacement (keeping each
week's battles together) and refit, which preserves within-week dependence
(shifting model roster, user population, prompt mix). We compare the resulting
95% intervals for bold and MATTR to the ordinary i.i.d. battle bootstrap.

    python src/time_block_bootstrap.py    # -> results/time_block_results.json
"""

import json
import numpy as np
import pandas as pd

from analyze_core import fit, MIN_BATTLES, FORMATTING
from linguistic_analysis import fit_bt, LINGUISTIC
from paths import BATTLES, DATA, RESULTS

CORE = FORMATTING + ["length"] + LINGUISTIC
N_BOOT = 500
np.random.seed(42)


def main():
    b = pd.read_parquet(BATTLES)
    ts = pd.read_parquet(DATA / "timestamps.parquet")
    b = b.merge(ts, on="conversation_pair_id", how="left")
    b = b[b["winner"].isin(["model_a", "model_b"])].copy()
    b["week"] = pd.to_datetime(b["timestamp"], utc=True).dt.strftime("%G-W%V")
    b = b.dropna(subset=["week"])

    # formatting sample and joint sample, with a fixed model set.
    fmt = b.dropna(subset=[f"{f}_{s}" for f in FORMATTING for s in ("a", "b")])
    jnt = b.dropna(subset=[f"{f}_{s}" for f in CORE for s in ("a", "b")])
    counts = pd.concat([fmt["model_a_name"], fmt["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    fmt = fmt[fmt["model_a_name"].isin(models) & fmt["model_b_name"].isin(models)]
    jnt = jnt[jnt["model_a_name"].isin(models) & jnt["model_b_name"].isin(models)]
    weeks = sorted(fmt["week"].unique())
    print(f"weeks: {len(weeks)}  formatting battles: {len(fmt):,}  joint battles: {len(jnt):,}")

    o = lambda c: (np.exp(c) - 1) * 100
    pt = {"bold_fmt": o(fit(fmt, models, FORMATTING)[1]["bold"]),
          "bold_joint": o(fit_bt(jnt, models, CORE)[1]["bold"]),
          "mattr_joint": o(fit_bt(jnt, models, CORE)[1]["mattr"])}

    fmt_by = {w: idx.to_numpy() for w, idx in fmt.groupby("week").groups.items()}
    jnt_by = {w: idx.to_numpy() for w, idx in jnt.groupby("week").groups.items()}
    boot = {k: [] for k in pt}
    for i in range(N_BOOT):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_BOOT}")
        draw = np.random.choice(weeks, size=len(weeks), replace=True)
        fi = np.concatenate([fmt_by[w] for w in draw])
        ji = np.concatenate([jnt_by[w] for w in draw if w in jnt_by])
        try:
            boot["bold_fmt"].append(o(fit(fmt.loc[fi], models, FORMATTING)[1]["bold"]))
            c = fit_bt(jnt.loc[ji], models, CORE)[1]
            boot["bold_joint"].append(o(c["bold"]))
            boot["mattr_joint"].append(o(c["mattr"]))
        except Exception:
            continue

    res = {"n_weeks": len(weeks), "n_boot": N_BOOT, "coef": {}}
    for k in pt:
        arr = np.array(boot[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        res["coef"][k] = {"point": float(pt[k]), "block_ci": [float(lo), float(hi)],
                          "block_ci_width": float(hi - lo)}
        print(f"  {k:12s} {pt[k]:+.1f}%  block-bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]  (width {hi-lo:.1f})")

    with open(RESULTS / "time_block_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved results/time_block_results.json")


if __name__ == "__main__":
    main()
