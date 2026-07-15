"""
Figures for the reading-depth (§4.6) and topic-control (§4.7) sections.
Reads turn_depth_results.json and topic_results.json. Matches the house style of
generate_figures.py / generate_linguistic_figure.py.

    python generate_polish_figures.py   # -> figures/fig10_reading_depth.{png,pdf}
                                        #    figures/fig11_topic_controls.{png,pdf}
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS, FIGURES

OUT = FIGURES
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FMT = "#7570b3"       # formatting
LEN = "#000000"       # length
DIV = "#d95f02"       # diversity
NSIG = "#bdbdbd"
GREY = "#9a9ab0"


def odds(b):
    return (np.exp(b) - 1) * 100


# --------------------------------------------------------------------------- #
# Figure 10: reading depth
# --------------------------------------------------------------------------- #
def fig_reading_depth():
    td = json.load(open(RESULTS / "turn_depth_results.json"))
    prim = td["primary"]["features"]        # formatting, single/multi/interaction
    joint = td["joint"]["features"]         # has length + mattr too

    # Left panel: single-turn vs multi-turn odds%, dumbbell.
    # 5 formatting features (from the formatting-only model) + length + MATTR (joint).
    rows = [("bold", prim["bold"], FMT), ("lists", prim["lists"], FMT),
            ("headers", prim["headers"], FMT), ("code_blocks", prim["code_blocks"], FMT),
            ("emoji", prim["emoji"], FMT),
            ("length", joint["length"], LEN), ("mattr", joint["mattr"], DIV)]
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    for yi, (_, v, col) in zip(y, rows):
        s, m = odds(v["single"]), odds(v["multi"])
        ax1.plot([s, m], [yi, yi], color=GREY, lw=1.5, zorder=1)
        ax1.scatter([s], [yi], color=col, s=48, zorder=3, marker="o")
        ax1.scatter([m], [yi], color=col, s=48, zorder=3, marker="D",
                    edgecolor="white", linewidth=0.6)
    ax1.axvline(0, color="black", lw=0.8, ls="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("win-odds change per SD (%)")
    ax1.set_title("Single-turn (circle) vs multi-turn (diamond)\n"
                  "formatting and length shrink, only diversity holds")
    handles = [plt.Line2D([], [], marker="o", ls="", color="black", label="single-turn (quick read)"),
               plt.Line2D([], [], marker="D", ls="", color="black", label="multi-turn (attentive read)")]
    ax1.legend(handles=handles, loc="lower right", frameon=True)

    # Right panel: interaction coefficients with 95% CI (the multi-turn slope shift).
    irows = [("bold", prim["bold"], FMT), ("lists", prim["lists"], FMT),
             ("headers", prim["headers"], FMT), ("code_blocks", prim["code_blocks"], FMT),
             ("emoji", prim["emoji"], FMT),
             ("length", joint["length"], LEN), ("mattr", joint["mattr"], DIV)]
    y2 = np.arange(len(irows))[::-1]
    for yi, (_, v, col) in zip(y2, irows):
        c = col if v["sig_bh"] else NSIG
        ci = v["interaction_ci"]
        ax2.plot(ci, [yi, yi], color=c, lw=2, zorder=1)
        ax2.scatter([v["interaction"]], [yi], color=c, s=44, zorder=2)
    ax2.axvline(0, color="black", lw=0.8, ls="--")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([r[0] for r in irows])
    ax2.set_xlabel("multi-turn interaction (log-odds per SD)")
    ax2.set_title("Slope shift with reading depth\n(95% CI; grey = n.s. after BH)")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig10_reading_depth.{ext}")
    print("wrote", OUT / "fig10_reading_depth.png")


# --------------------------------------------------------------------------- #
# Figure 11: topic controls
# --------------------------------------------------------------------------- #
def fig_topic_controls():
    tp = json.load(open(RESULTS / "topic_results.json"))
    q1 = tp["q1_topic_stratified"]
    q2 = tp["q2_reading_depth_topic_controlled"]["interactions"]
    td = json.load(open(RESULTS / "turn_depth_results.json"))["primary"]["features"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: bold effect by topic, 95% CI, showing it is positive everywhere.
    topics = [(t, r) for t, r in q1.items() if t != "ALL"]
    topics.sort(key=lambda kv: kv[1]["coef"]["bold"]["odds_pct"])
    labels, pts, los, his = [], [], [], []
    for t, r in topics:
        c = r["coef"]["bold"]
        labels.append(t.replace(" & Formal Science & Technology", " & Tech")
                      .replace(" & Economics & Finance", "")[:26])
        pts.append(c["odds_pct"])
        los.append(odds(c["ci"][0]))
        his.append(odds(c["ci"][1]))
    y = np.arange(len(labels))
    for yi, p, lo, hi in zip(y, pts, los, his):
        ax1.plot([lo, hi], [yi, yi], color=FMT, lw=2, zorder=1)
        ax1.scatter([p], [yi], color=FMT, s=42, zorder=2)
    pooled = q1["ALL"]["coef"]["bold"]["odds_pct"]
    ax1.axvline(pooled, color="black", lw=1.0, ls="--", label=f"pooled (+{pooled:.0f}%)")
    ax1.axvline(0, color="grey", lw=0.7, ls=":")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("bold win-odds change per SD (%)")
    ax1.set_title("Bold formatting is positive in every topic\n(within-topic fit, 95% CI)")
    ax1.legend(loc="lower right", frameon=True)

    # Right: reading-depth interaction, uncontrolled (§4.6) vs topic-controlled (§4.7).
    feats = ["bold", "lists", "headers", "code_blocks", "emoji"]
    y2 = np.arange(len(feats))[::-1]
    for yi, f in zip(y2, feats):
        u, c = td[f]["interaction"], q2[f]["interaction"]
        ax2.plot([u, c], [yi, yi], color=GREY, lw=1.5, zorder=1)
        ax2.scatter([u], [yi], color=GREY, s=46, zorder=2, marker="o", label="_")
        ax2.scatter([c], [yi], color=FMT, s=46, zorder=3, marker="D",
                    edgecolor="white", linewidth=0.6)
    ax2.axvline(0, color="black", lw=0.8, ls="--")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(feats)
    ax2.set_xlabel("multi-turn interaction (log-odds per SD)")
    ax2.set_title("Reading-depth effect survives topic control\n(circle = §4.6, diamond = + topic x style)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=GREY, label="no topic control (§4.6)"),
               plt.Line2D([], [], marker="D", ls="", color=FMT, label="topic-controlled (§4.7)")]
    ax2.legend(handles=handles, loc="lower left", frameon=True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig11_topic_controls.{ext}")
    print("wrote", OUT / "fig11_topic_controls.png")


if __name__ == "__main__":
    fig_reading_depth()
    fig_topic_controls()
