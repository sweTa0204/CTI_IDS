import matplotlib.pyplot as plt
import numpy as np

# Data from the analysis
models = [
    "Random Forest",
    "XGBoost",
    "MLP",
    "SVM",
    "Logistic Regression",
]
f1_scores = [0.999, 0.998, 0.989, 0.972, 0.965]

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Create the bar plot
bars = ax.bar(models, f1_scores, color="skyblue")

# Add data labels
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval,
        f"{yval:.3f}",
        va="bottom",
        ha="center",
    )

# Customize the plot
ax.set_title("Model F1-Score Comparison", fontsize=16)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_ylim(0.9, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.savefig("d:/Edu/Final Project/Phase_1 Final/03_model_training/models/comparison/model_f1_score_comparison.png")

print("Comparison chart generated successfully.")
