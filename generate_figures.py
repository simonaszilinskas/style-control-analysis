"""
Generate publication-quality figures for the ComparIA style control paper.

Produces 8 figures:
  1. Forest plot of style coefficients with bootstrap CIs
  2. Scatter plot of standard vs style-controlled BT ratings
  3. Rank change waterfall for top significant movers
  4. Tier-stratified style effects grouped bar chart
  5. Rating change vs formatting intensity scatter
  6. Ablation: single-feature vs joint coefficients
  7. Reaction vs vote BT ratings scatter
  8. Winner-flipping formatting asymmetry

Usage:
    python generate_figures.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette
C_SIG = "#2166ac"       # significant (blue)
C_NSIG = "#969696"      # not significant (gray)
C_RISE = "#4daf4a"      # rank rise (green)
C_DROP = "#e41a1c"      # rank drop (red)
C_REASON = "#ff7f00"    # reasoning models (orange)
C_BOLD = "#1b9e77"
C_LISTS = "#d95f02"
C_HEADERS = "#7570b3"

# ── Load data ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

with open(DATA_DIR / "clean_analysis_results.json") as f:
    results = json.load(f)

with open(DATA_DIR / "endogeneity_results.json") as f:
    endo = json.load(f)

with open(DATA_DIR / "qualitative_results.json") as f:
    qual = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Forest plot of style coefficients
# ═══════════════════════════════════════════════════════════════════════════
def fig1_forest_plot():
    cis = results["style_coefficient_cis"]
    features = ["bold", "lists", "headers", "code_blocks", "emoji"]
    labels = ["Bold", "Lists", "Headers", "Code blocks", "Emoji"]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    y_pos = np.arange(len(features))[::-1]

    for i, (feat, label) in enumerate(zip(features, labels)):
        d = cis[feat]
        point = d["odds_point"]
        lo = d["odds_ci_low"]
        hi = d["odds_ci_high"]
        sig = d["significant_bh"]
        color = C_SIG if sig else C_NSIG

        y = y_pos[i]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(point, y, "o", color=color, markersize=7, zorder=5)

        # Annotation
        if sig:
            ax.text(hi + 0.8, y, f"+{point:.1f}%", va="center", fontsize=8.5,
                    color=color, fontweight="bold")
        else:
            ax.text(hi + 0.8, y, f"{point:+.1f}%", va="center", fontsize=8.5,
                    color=color)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in win odds per SD (%)")
    ax.set_title("Style coefficients with 95% bootstrap CIs (BH-corrected)")
    ax.set_xlim(-5, 23)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    sig_patch = mpatches.Patch(color=C_SIG, label="Significant (FDR < 0.05)")
    nsig_patch = mpatches.Patch(color=C_NSIG, label="Not significant")
    ax.legend(handles=[sig_patch, nsig_patch], loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_forest_plot.pdf")
    fig.savefig(OUT_DIR / "fig1_forest_plot.png")
    plt.close(fig)
    print("  [OK] fig1_forest_plot")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Standard vs style-controlled BT ratings scatter
# ═══════════════════════════════════════════════════════════════════════════
def fig2_scatter_ratings():
    std = results["rankings"]["standard"]
    ctrl = results["rankings"]["controlled"]

    # Notable models to label (biggest movers)
    label_models = {
        "mistral-large-2512", "o3-mini", "kimi-k2-thinking",
        "claude-4-5-sonnet", "gemini-3-flash-preview",
        "EuroLLM-22B-Instruct-2512",
    }

    fig, ax = plt.subplots(figsize=(6, 5.5))

    models = sorted(std.keys())
    for m in models:
        x = std[m]["rating"]
        y = ctrl[m]["rating"]
        is_labeled = m in label_models

        alpha = 0.85 if is_labeled else 0.45
        size = 30 if is_labeled else 18

        ax.scatter(x, y, c=C_SIG, s=size, alpha=alpha, edgecolors="white",
                   linewidths=0.3, zorder=3 if is_labeled else 2)

    # Diagonal reference line
    lims = [680, 1200]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.6, zorder=1)

    # Labels for notable models with manual offsets to reduce overlap
    offsets = {
        "mistral-large-2512": (6, -14),
        "o3-mini": (-55, -12),
        "kimi-k2-thinking": (-80, 8),
        "claude-4-5-sonnet": (6, -14),
        "gemini-3-flash-preview": (-15, 8),
        "EuroLLM-22B-Instruct-2512": (6, -13),
    }

    for m in label_models:
        x = std[m]["rating"]
        y = ctrl[m]["rating"]
        dx, dy = offsets.get(m, (5, 5))
        short = m.replace("-instruct", "").replace("-Instruct", "")
        ax.annotate(short, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

    ax.set_xlabel("Standard BT Rating")
    ax.set_ylabel("Style-Controlled BT Rating")
    ax.set_title(f"Standard vs. Style-Controlled Ratings (r = {results['bt_correlation']:.3f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: above line = rises after control, below = drops
    ax.text(720, 760, "Rise after\nstyle control", fontsize=8, color=C_RISE,
            alpha=0.7, fontstyle="italic")
    ax.text(760, 710, "Drop after\nstyle control", fontsize=8, color=C_DROP,
            alpha=0.7, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_scatter_ratings.pdf")
    fig.savefig(OUT_DIR / "fig2_scatter_ratings.png")
    plt.close(fig)
    print("  [OK] fig2_scatter_ratings")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Rank change waterfall, top 20 movers
# ═══════════════════════════════════════════════════════════════════════════
def fig3_rank_changes():
    rank_sig = results["rank_significance"]

    # Filter to significant only, sort by absolute rank change, take top 20
    sig_models = [m for m in rank_sig if m.get("significant_bh", False)]
    sorted_models = sorted(sig_models, key=lambda x: abs(x["rank_change"]), reverse=True)[:20]
    sorted_models.sort(key=lambda x: x["rank_change"])

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = []
    changes = []
    colors = []

    for m in sorted_models:
        labels.append(m["model"])
        changes.append(m["rank_change"])
        colors.append(C_RISE if m["rank_change"] > 0 else C_DROP)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, changes, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rank change after style control")
    ax.set_title("Top 20 rank changes (positive = rises after control)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Legend
    rise_patch = mpatches.Patch(color=C_RISE, label="Rises (less formatted)")
    drop_patch = mpatches.Patch(color=C_DROP, label="Drops (more formatted)")
    ax.legend(handles=[rise_patch, drop_patch],
              loc="lower right", frameon=True, fontsize=8,
              fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_rank_changes.pdf")
    fig.savefig(OUT_DIR / "fig3_rank_changes.png")
    plt.close(fig)
    print("  [OK] fig3_rank_changes")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Tier-stratified style effects
# ═══════════════════════════════════════════════════════════════════════════
def fig4_tier_effects():
    # Use the interaction model implied effects from the paper draft (Table 5)
    # These are the composite effects that include main + interaction terms
    tiers_data = endo["tier_stratified_coefficients"]

    features = ["Bold", "Lists", "Headers"]
    feat_keys = ["bold", "lists", "headers"]
    tiers = ["bottom-bottom", "middle-middle", "top-top"]
    tier_nice = ["Bottom-Bottom", "Middle-Middle", "Top-Top"]
    tier_labels = ["{}\n(n={:,})".format(tn, tiers_data[t]["n_battles"]) for tn, t in zip(tier_nice, tiers)]

    # From the paper's Table 5 (interaction model implied odds changes)
    # These values come from main + tier interaction effects
    interaction = endo["interaction_model"]
    main = interaction["main_style_effects"]
    top_int = interaction["top_interactions"]
    mid_int = interaction["mid_interactions"]

    implied = {}
    for fk in feat_keys:
        base = main[fk]
        implied[fk] = {
            "bottom-bottom": (np.exp(base) - 1) * 100,  # baseline (reference)
            "middle-middle": (np.exp(base + mid_int[fk]) - 1) * 100,
            "top-top": (np.exp(base + top_int[fk]) - 1) * 100,
        }

    fig, ax = plt.subplots(figsize=(5.5, 4))

    x = np.arange(len(tiers))
    width = 0.22
    feat_colors = [C_BOLD, C_LISTS, C_HEADERS]

    for i, (fk, fl, fc) in enumerate(zip(feat_keys, features, feat_colors)):
        vals = [implied[fk][t] for t in tiers]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=fl, color=fc, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("Win odds change per SD (%)")
    ax.set_title("Style effects by model-pair tier (interaction model)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(implied["bold"]["bottom-bottom"], implied["lists"]["bottom-bottom"]) * 1.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_tier_effects.pdf")
    fig.savefig(OUT_DIR / "fig4_tier_effects.png")
    plt.close(fig)
    print("  [OK] fig4_tier_effects")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Rating change vs formatting intensity
# ═══════════════════════════════════════════════════════════════════════════
def fig5_rating_vs_formatting():
    battles = pd.read_parquet(DATA_DIR / "battles_bt_styled.parquet")
    std = results["rankings"]["standard"]
    ctrl = results["rankings"]["controlled"]

    # Compute per-model mean formatting intensity (bold + lists + headers)
    fmt_a = battles.groupby("model_a_name")[["bold_a", "lists_a", "headers_a"]].mean()
    fmt_a.columns = ["bold", "lists", "headers"]
    fmt_b = battles.groupby("model_b_name")[["bold_b", "lists_b", "headers_b"]].mean()
    fmt_b.columns = ["bold", "lists", "headers"]
    fmt = pd.concat([fmt_a, fmt_b]).groupby(level=0).mean()
    fmt["total"] = fmt["bold"] + fmt["lists"] + fmt["headers"]

    models = sorted(set(std.keys()) & set(fmt.index))
    x_fmt = np.array([fmt.loc[m, "total"] for m in models])
    y_change = np.array([ctrl[m]["rating"] - std[m]["rating"] for m in models])

    r = np.corrcoef(x_fmt, y_change)[0, 1]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x_fmt, y_change, c=C_SIG, s=20, alpha=0.6, edgecolors="white", linewidths=0.3)

    # Fit line
    z = np.polyfit(x_fmt, y_change, 1)
    x_line = np.linspace(x_fmt.min(), x_fmt.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=C_DROP, linewidth=1.2, alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # Label only the most extreme outliers
    for m, xv, yv in zip(models, x_fmt, y_change):
        if abs(yv) > 55 or xv > 90:
            ax.annotate(m, (xv, yv), fontsize=6.5, color="#555555",
                        textcoords="offset points", xytext=(4, -10))

    ax.set_xlabel("Mean formatting intensity (bold + lists + headers)")
    ax.set_ylabel("Rating change after style control")
    ax.set_title(f"Rating change vs. formatting intensity (r = {r:.3f})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_rating_vs_formatting.pdf")
    fig.savefig(OUT_DIR / "fig5_rating_vs_formatting.png")
    plt.close(fig)
    print("  [OK] fig5_rating_vs_formatting")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Ablation, single-feature vs joint coefficients
# ═══════════════════════════════════════════════════════════════════════════
def fig6_ablation():
    ablation = results["ablation"]
    joint = results["style_coefficients"]

    features = ["bold", "lists", "headers", "code_blocks", "emoji"]
    labels = ["Bold", "Lists", "Headers", "Code\nblocks", "Emoji"]

    single_vals = [ablation[f]["odds_change_pct"] for f in features]
    joint_vals = [joint[f]["odds_change_pct"] for f in features]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    y_pos = np.arange(len(features))[::-1]

    for i, (sv, jv, label) in enumerate(zip(single_vals, joint_vals, labels)):
        y = y_pos[i]
        # Line connecting the two dots
        ax.plot([jv, sv], [y, y], color="#cccccc", linewidth=1.5, zorder=1)
        # Joint (filled)
        ax.plot(jv, y, "o", color=C_SIG, markersize=8, zorder=3, label="Joint model" if i == 0 else "")
        # Single (hollow)
        ax.plot(sv, y, "D", color=C_DROP, markersize=7, zorder=3, markerfacecolor="white",
                markeredgewidth=1.5, label="Single feature" if i == 0 else "")

        # Annotations
        ax.text(jv - 0.8, y + 0.25, f"{jv:+.1f}%", fontsize=7.5, ha="right", color=C_SIG)
        ax.text(sv + 0.8, y + 0.25, f"+{sv:.1f}%", fontsize=7.5, ha="left", color=C_DROP)

    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in win odds per SD (%)")
    ax.set_title("Single-feature vs. joint model coefficients")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_ablation.pdf")
    fig.savefig(OUT_DIR / "fig6_ablation.png")
    plt.close(fig)
    print("  [OK] fig6_ablation")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Reaction vs vote BT ratings
# ═══════════════════════════════════════════════════════════════════════════
def _fit_bt(battles_df, min_battles=50):
    """Fit a standard Bradley-Terry model, return {model: rating}."""
    decisive = battles_df[battles_df["winner"].isin(["model_a", "model_b"])].copy()
    models = sorted(set(decisive["model_a_name"]) | set(decisive["model_b_name"]))

    # Filter to models with enough battles
    counts = pd.concat([decisive["model_a_name"], decisive["model_b_name"]]).value_counts()
    models = [m for m in models if counts.get(m, 0) >= min_battles]
    model_idx = {m: i for i, m in enumerate(models)}

    rows_X = []
    rows_y = []
    for _, row in decisive.iterrows():
        a, b = row["model_a_name"], row["model_b_name"]
        if a not in model_idx or b not in model_idx:
            continue
        x = np.zeros(len(models))
        x[model_idx[a]] = 1
        x[model_idx[b]] = -1
        rows_X.append(x)
        rows_y.append(1 if row["winner"] == "model_a" else 0)

    X = np.array(rows_X)
    y = np.array(rows_y)

    lr = LogisticRegression(fit_intercept=False, max_iter=2000, C=1e9)
    lr.fit(X, y)

    ratings = {}
    for m, i in model_idx.items():
        ratings[m] = 1000 + 400 * lr.coef_[0][i] / np.log(10)
    return ratings


def fig7_reaction_vs_vote():
    battles = pd.read_parquet(DATA_DIR / "battles_bt_styled.parquet")

    vote_battles = battles[battles["source"] == "vote"]
    reaction_battles = battles[battles["source"] == "reaction"]

    vote_bt = _fit_bt(vote_battles, min_battles=30)
    reaction_bt = _fit_bt(reaction_battles, min_battles=10)

    common = sorted(set(vote_bt.keys()) & set(reaction_bt.keys()))
    x_vote = np.array([vote_bt[m] for m in common])
    y_react = np.array([reaction_bt[m] for m in common])

    r = np.corrcoef(x_vote, y_react)[0, 1]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(x_vote, y_react, c=C_SIG, s=22, alpha=0.6, edgecolors="white", linewidths=0.3)

    lo = min(x_vote.min(), y_react.min()) - 20
    hi = max(x_vote.max(), y_react.max()) + 20
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Vote-derived BT Rating")
    ax.set_ylabel("Reaction-derived BT Rating")
    ax.set_title(f"Reaction vs. vote BT ratings ({len(common)} models, r = {r:.3f})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_reaction_vs_vote.pdf")
    fig.savefig(OUT_DIR / "fig7_reaction_vs_vote.png")
    plt.close(fig)
    print("  [OK] fig7_reaction_vs_vote")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Winner-flipping formatting asymmetry
# ═══════════════════════════════════════════════════════════════════════════
def fig8_flip_asymmetry():
    fmt = qual["formatting_in_flips"]
    winner_more = fmt["vote_winner_formats_more_pct"]
    loser_more = fmt["vote_loser_formats_more_pct"]
    tied = 100.0 - winner_more - loser_more
    n_total = qual["summary"]["winner_flipping_battles"]

    categories = ["Vote winner\nformats more", "Equal\nformatting", "Vote loser\nformats more"]
    values = [winner_more, tied, loser_more]
    colors = [C_DROP, C_NSIG, C_RISE]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5, width=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("% of winner-flipping battles")
    ax.set_title(f"Formatting asymmetry in winner-flipping battles (n={n_total:,})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_flip_asymmetry.pdf")
    fig.savefig(OUT_DIR / "fig8_flip_asymmetry.png")
    plt.close(fig)
    print("  [OK] fig8_flip_asymmetry")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig1_forest_plot()
    fig2_scatter_ratings()
    fig3_rank_changes()
    fig4_tier_effects()
    fig5_rating_vs_formatting()
    fig6_ablation()
    fig7_reaction_vs_vote()
    fig8_flip_asymmetry()
    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix in (".pdf", ".png"):
            print(f"  {f.name}")
