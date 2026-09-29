"""
Figure for the linguistic extension (§4.2): joint style coefficients with
bootstrap CIs, coloured by feature family, and the formatting -> +length ->
joint shrinkage. Reads linguistic_results.json. Matches the house style of
generate_figures.py.

    python generate_linguistic_figure.py   # -> figures/fig9_linguistic.{png,pdf}
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS, FIGURES

OUT = FIGURES
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FAMILY = {"headers": "formatting", "lists": "formatting", "bold": "formatting",
          "code_blocks": "formatting", "emoji": "formatting", "length": "length",
          "rel": "readability", "cli": "readability", "fkg": "readability",
          "ttr": "diversity", "mattr": "diversity",
          "asl": "structure", "long_sent_ratio": "structure"}
COLOR = {"formatting": "#7570b3", "length": "#000000", "readability": "#1b9e77",
         "diversity": "#d95f02", "structure": "#e7298a"}
NSIG = "#bdbdbd"
LABEL = {"headers": "Headers", "lists": "Lists", "bold": "Bold",
         "code_blocks": "Code blocks", "emoji": "Emoji", "length": "Length",
         "rel": "Readability (REL)", "cli": "Coleman-Liau", "fkg": "Flesch-Kincaid",
         "ttr": "TTR", "mattr": "MATTR", "asl": "Mean sentence length",
         "long_sent_ratio": "Long-sentence share"}
# Right panel: one distinct colour and marker per formatting feature.
LINE = {"bold": ("#7570b3", "o"), "lists": ("#66a61e", "s"), "headers": ("#e6ab02", "^"),
        "code_blocks": ("#1f78b4", "D"), "emoji": ("#a6761d", "v"), "length": ("#000000", "o")}

res = json.load(open(RESULTS / "linguistic_results.json"))
jc = res["joint_coefficients"]
order = ["headers", "lists", "bold", "code_blocks", "emoji", "length",
         "rel", "cli", "fkg", "ttr", "mattr", "asl", "long_sent_ratio"]
order = [f for f in order if f in jc]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})

# Left: joint coefficient forest with 95% CI.
y = np.arange(len(order))[::-1]
for yi, f in zip(y, order):
    c = jc[f]
    col = COLOR[FAMILY[f]] if c["sig_bh"] else NSIG
    ax1.plot(c["ci"], [yi, yi], color=col, lw=2, zorder=1)
    ax1.scatter([c["point"]], [yi], color=col, s=40, zorder=2)
ax1.axvline(0, color="black", lw=0.8, ls="--")
ax1.set_yticks(y)
ax1.set_yticklabels([LABEL[f] for f in order])
ax1.set_xlabel("joint style coefficient (log-odds per SD)")
ax1.set_title("Joint model: what moves a vote\n(95% CI, BH-corrected; grey = n.s.)")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in COLOR.items()]
ax1.legend(handles=handles, loc="upper left", frameon=True)

# Right: formatting -> +length -> joint for the markdown features + length.
nested = res["bt_coefficients"]
specs = ["formatting", "formatting+length", "joint"]
feats_r = ["bold", "lists", "headers", "code_blocks", "emoji", "length"]
x = np.arange(len(specs))
for f in feats_r:
    ys = [nested[s].get(f, np.nan) for s in specs]
    col, mk = LINE[f]
    ax2.plot(x, ys, marker=mk, ls="-", lw=1.6, ms=6, color=col, label=LABEL[f])
ax2.axhline(0, color="black", lw=0.6, ls=":")
ax2.set_xticks(x)
ax2.set_xticklabels(["formatting", "+ length", "joint"])
ax2.set_ylabel("coefficient (log-odds per SD)")
ax2.set_title("Formatting associations shrink as length\nand linguistics are added")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig9_linguistic.{ext}")
print("wrote", OUT / "fig9_linguistic.png")
