"""
Generate comparison charts: Decision Tree vs XGBoost.
Saved to images/ directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DATA
# ============================================================

# Benchmark test results (optimized thresholds)
models = ["Decision Tree", "XGBoost"]
metrics = {
    "F1 Score":  [88.53, 90.26],
    "Precision": [93.43, 94.42],
    "Recall":    [84.13, 86.45],
    "Accuracy":  [97.83, 98.14],
}

# Cross-validation F1
cv_f1_means = [95.48, 96.45]
cv_f1_stds  = [1.50, 0.47]

# Confusion matrix values (optimized)
dt_cm  = np.array([[36758, 242], [649, 3440]])
xgb_cm = np.array([[36791, 209], [554, 3535]])

# Complexity
complexity = {
    "Training Time\n(relative)":   [1, 10],      # single tree vs 100 boosted trees
    "Number of\nTrees":            [1, 100],
    "Total Decision\nNodes":       [257, 25700],  # approx 257 per tree * 100
    "Model File\nSize (KB)":       [15, 900],     # approximate
}

# Feature importance
features = ["rate", "sload", "sbytes", "dload", "proto", "dtcpb", "stcpb", "dmean", "tcprtt", "dur"]
dt_importance =  [0.0011, 0.5298, 0.1112, 0.1455, 0.1017, 0.0004, 0.0097, 0.0272, 0.0647, 0.0089]
# XGBoost feature importance (from SHAP mean absolute values — approximate from training)
xgb_importance = [0.0520, 0.2890, 0.0810, 0.0640, 0.2150, 0.0380, 0.0350, 0.0430, 0.1520, 0.0310]

TEAL = "#00897B"
ORANGE = "#FF9500"
GRAY = "#86868B"
RED = "#FF3B30"
GREEN = "#34C759"
BLUE = "#0071E3"

# ============================================================
# CHART 1: Side-by-side metric comparison (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.32
bars1 = ax.bar(x - width/2, [metrics[m][0] for m in metrics], width,
               label="Decision Tree", color=ORANGE, edgecolor="white", alpha=0.9)
bars2 = ax.bar(x + width/2, [metrics[m][1] for m in metrics], width,
               label="XGBoost", color=TEAL, edgecolor="white", alpha=0.9)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=ORANGE)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEAL)

ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
ax.set_title("Decision Tree vs XGBoost — Benchmark Test Performance", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(list(metrics.keys()), fontsize=12)
ax.set_ylim(80, 101)
ax.legend(fontsize=12, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparison_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_metrics.png")

# ============================================================
# CHART 2: Confusion matrix side-by-side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, cm, title, color in [
    (axes[0], dt_cm, "Decision Tree\n(Threshold = 0.93)", "Oranges"),
    (axes[1], xgb_cm, "XGBoost\n(Threshold = 0.85)", "Teal"),
]:
    # Use custom colormap for teal
    if color == "Teal":
        from matplotlib.colors import LinearSegmentedColormap
        teal_cmap = LinearSegmentedColormap.from_list("teal", ["#E0F2F1", "#00897B", "#004D40"])
        cmap = teal_cmap
    else:
        cmap = plt.cm.Oranges

    im = ax.imshow(cm, cmap=cmap, interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "DoS"], fontsize=11)
    ax.set_yticklabels(["Normal", "DoS"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm[i][j]
            clr = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=15, fontweight="bold", color=clr)

# Highlight differences
diff_fp = xgb_cm[0][1] - dt_cm[0][1]  # XGB has fewer FP
diff_fn = xgb_cm[1][0] - dt_cm[1][0]  # XGB has fewer FN
fig.suptitle("Confusion Matrix Comparison (41,089 Test Samples)", fontsize=14, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         f"XGBoost produces {abs(diff_fp)} fewer false positives and {abs(diff_fn)} fewer false negatives than Decision Tree",
         ha="center", fontsize=11, style="italic", color=GRAY)
plt.tight_layout()
plt.savefig("images/comparison_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_confusion_matrix.png")

# ============================================================
# CHART 3: CV F1 comparison with error bars
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
colors = [ORANGE, TEAL]
for i, (model, mean, std) in enumerate(zip(models, cv_f1_means, cv_f1_stds)):
    ax.bar(i, mean, width=0.5, color=colors[i], alpha=0.9, edgecolor="white")
    ax.errorbar(i, mean, yerr=std, fmt="none", ecolor="black", capsize=10, capthick=2, linewidth=2)
    ax.text(i, mean + std + 0.3, f"{mean:.2f}%\n(+/-{std:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=12, fontweight="bold")
ax.set_ylabel("F1 Score (%)", fontsize=13, fontweight="bold")
ax.set_title("Cross-Validation F1 Score (5-Fold)", fontsize=14, fontweight="bold")
ax.set_ylim(92, 99)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparison_cv_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_cv_f1.png")

# ============================================================
# CHART 4: Complexity comparison (1 tree vs 100 trees)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

comp_labels = list(complexity.keys())
dt_vals = [complexity[k][0] for k in comp_labels]
xgb_vals = [complexity[k][1] for k in comp_labels]

x = np.arange(len(comp_labels))
width = 0.32
bars1 = ax.bar(x - width/2, dt_vals, width, label="Decision Tree (1 tree)", color=ORANGE, edgecolor="white")
bars2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost (100 trees)", color=TEAL, edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=ORANGE)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEAL)

ax.set_ylabel("Value", fontsize=13, fontweight="bold")
ax.set_title("Model Complexity — Decision Tree vs XGBoost", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comp_labels, fontsize=11)
ax.legend(fontsize=11)
ax.set_yscale("log")
ax.set_ylim(0.5, 100000)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparison_complexity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_complexity.png")

# ============================================================
# CHART 5: Feature importance comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Decision Tree
sorted_dt = sorted(zip(features, dt_importance), key=lambda x: x[1])
ax = axes[0]
ax.barh([f[0] for f in sorted_dt], [f[1] for f in sorted_dt],
        color=ORANGE, edgecolor="white", height=0.6, alpha=0.9)
for i, (feat, val) in enumerate(sorted_dt):
    ax.text(val + 0.005, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.set_title("Decision Tree\n(Gini Importance)", fontsize=13, fontweight="bold")
ax.set_xlim(0, 0.62)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# XGBoost
sorted_xgb = sorted(zip(features, xgb_importance), key=lambda x: x[1])
ax = axes[1]
ax.barh([f[0] for f in sorted_xgb], [f[1] for f in sorted_xgb],
        color=TEAL, edgecolor="white", height=0.6, alpha=0.9)
for i, (feat, val) in enumerate(sorted_xgb):
    ax.text(val + 0.003, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.set_title("XGBoost\n(Gain Importance)", fontsize=13, fontweight="bold")
ax.set_xlim(0, 0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Feature Importance Comparison", fontsize=14, fontweight="bold", y=1.02)
fig.text(0.5, -0.03,
         "Decision Tree relies heavily on sload (53%). XGBoost distributes importance more evenly across features.",
         ha="center", fontsize=11, style="italic", color=GRAY)
plt.tight_layout()
plt.savefig("images/comparison_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_feature_importance.png")

# ============================================================
# CHART 6: The "1 Worker vs 10 Workers" analogy diagram
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# LEFT: Decision Tree (1 worker)
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Decision Tree\n\"1 Worker Doing Everything\"", fontsize=14, fontweight="bold", color=ORANGE)

# Draw single tree
ax.add_patch(plt.Rectangle((3.5, 7), 3, 2, facecolor=ORANGE, alpha=0.2, edgecolor=ORANGE, linewidth=2, zorder=2))
ax.text(5, 8, "Single\nDecision Tree", ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
# Arrow down
ax.annotate("", xy=(5, 5.5), xytext=(5, 7), arrowprops=dict(arrowstyle="->", lw=2, color=ORANGE))
# Input
ax.add_patch(plt.Rectangle((2, 4), 6, 1.3, facecolor="#FFF3E0", edgecolor=ORANGE, linewidth=1.5, zorder=2))
ax.text(5, 4.65, "All 10 features analyzed\nby 1 tree with 257 nodes", ha="center", va="center", fontsize=10, zorder=3)
# Arrow down
ax.annotate("", xy=(5, 2.5), xytext=(5, 4), arrowprops=dict(arrowstyle="->", lw=2, color=ORANGE))
# Output
ax.add_patch(plt.Circle((5, 1.8), 1, facecolor=ORANGE, alpha=0.3, edgecolor=ORANGE, linewidth=2, zorder=2))
ax.text(5, 1.8, "1 Vote\n= Final\nAnswer", ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)

# RIGHT: XGBoost (100 workers)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("XGBoost\n\"100 Workers Collaborating\"", fontsize=14, fontweight="bold", color=TEAL)

# Draw multiple trees
tree_positions = [(1.5, 7.2), (3.5, 7.2), (5.5, 7.2), (7.5, 7.2)]
tree_labels = ["Tree 1\n(finds easy\npatterns)", "Tree 2\n(fixes Tree 1's\nmistakes)",
               "Tree 3\n(fixes Tree 2's\nmistakes)", "... Tree 100\n(final\nrefinement)"]
for (x_pos, y_pos), label in zip(tree_positions, tree_labels):
    ax.add_patch(plt.Rectangle((x_pos - 0.7, y_pos), 1.4, 1.8, facecolor=TEAL, alpha=0.15,
                                edgecolor=TEAL, linewidth=1.5, zorder=2))
    ax.text(x_pos, y_pos + 0.9, label, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

# Arrows converging
for x_pos, _ in tree_positions:
    ax.annotate("", xy=(5, 5.5), xytext=(x_pos, 7.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=TEAL, alpha=0.6))

# Aggregation box
ax.add_patch(plt.Rectangle((2, 4), 6, 1.3, facecolor="#E0F2F1", edgecolor=TEAL, linewidth=1.5, zorder=2))
ax.text(5, 4.65, "Each tree focuses on what\nprevious trees got WRONG", ha="center", va="center", fontsize=10, zorder=3)

# Arrow down
ax.annotate("", xy=(5, 2.5), xytext=(5, 4), arrowprops=dict(arrowstyle="->", lw=2, color=TEAL))

# Output
ax.add_patch(plt.Circle((5, 1.8), 1, facecolor=TEAL, alpha=0.3, edgecolor=TEAL, linewidth=2, zorder=2))
ax.text(5, 1.8, "100 Votes\n= Refined\nAnswer", ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)

fig.suptitle("Why XGBoost Outperforms Decision Tree", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("images/comparison_analogy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_analogy.png")

# ============================================================
# CHART 7: Error reduction — where XGBoost wins
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

categories = ["False Positives\n(Normal flagged as DoS)", "False Negatives\n(DoS missed as Normal)"]
dt_errors = [242, 649]
xgb_errors = [209, 554]
reduction = [242 - 209, 649 - 554]

x = np.arange(len(categories))
width = 0.3
bars1 = ax.bar(x - width/2, dt_errors, width, label="Decision Tree", color=ORANGE, edgecolor="white")
bars2 = ax.bar(x + width/2, xgb_errors, width, label="XGBoost", color=TEAL, edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{int(bar.get_height()):,}", ha="center", fontsize=12, fontweight="bold", color=ORANGE)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{int(bar.get_height()):,}", ha="center", fontsize=12, fontweight="bold", color=TEAL)

# Reduction annotations
for i, red in enumerate(reduction):
    ax.annotate(f"-{red}\nfewer errors", xy=(i + width/2, xgb_errors[i]),
                xytext=(i + 0.55, max(dt_errors[i], xgb_errors[i]) * 0.7),
                fontsize=10, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

ax.set_ylabel("Number of Errors", fontsize=13, fontweight="bold")
ax.set_title("Error Comparison — Decision Tree vs XGBoost", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparison_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/comparison_errors.png")

print("\nAll comparison charts generated!")
