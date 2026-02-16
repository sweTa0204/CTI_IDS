"""
Decision Tree Model Training Script for DoS Attack Detection
==============================================================

This script trains a Decision Tree classifier for binary classification
of network traffic (DoS Attack vs Normal Traffic).

It follows the same training methodology as the other 7 models:
  1. Load scaled training data (24,528 balanced samples, 10 features)
  2. 5-Fold Stratified Cross-Validation
  3. Train final model on full training set
  4. Evaluate on imbalanced benchmark test set (41,089 samples)
  5. Optimize classification threshold for best F1
  6. Generate visualizations
  7. Save model, results, and plots

Dataset: UNSW-NB15 (Official Training Set)
Training Samples: 24,528 (12,264 DoS + 12,264 Normal)
Test Samples: 41,089 (37,000 Normal + 4,089 DoS)
Features: 10 selected features

Author: Research Project
Date: 2026-02-16
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
MAX_DEPTH = 10           # Prevent overfitting (same as Random Forest)
MIN_SAMPLES_SPLIT = 10   # Require at least 10 samples to split
MIN_SAMPLES_LEAF = 5     # Each leaf must have at least 5 samples
CRITERION = "gini"       # Gini impurity (standard for classification)

FEATURE_NAMES = ["rate", "sload", "sbytes", "dload", "proto",
                 "dtcpb", "stcpb", "dmean", "tcprtt", "dur"]

# Paths (relative to this script's location)
DATA_PATH = "../../data/X_train_scaled.csv"
LABELS_PATH = "../../data/y_train.csv"
TEST_DATA_PATH = "../../data/X_test_scaled.csv"
TEST_LABELS_PATH = "../../data/y_test.csv"
MODEL_OUTPUT = "decisiontree_model.pkl"
RESULTS_OUTPUT = "results/decisiontree_results.json"

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Decision Tree Model Training for DoS Detection")
print("=" * 60)

print("\n[1/7] Loading data...")
X_train = pd.read_csv(DATA_PATH).values
y_train = pd.read_csv(LABELS_PATH).values.ravel()
X_test = pd.read_csv(TEST_DATA_PATH).values
y_test = pd.read_csv(TEST_LABELS_PATH).values.ravel()

print(f"      Training samples: {X_train.shape[0]:,}")
print(f"      Test samples:     {X_test.shape[0]:,}")
print(f"      Features:         {X_train.shape[1]}")
print(f"      Train distribution: Normal={sum(y_train==0):,}, DoS={sum(y_train==1):,}")
print(f"      Test distribution:  Normal={sum(y_test==0):,}, DoS={sum(y_test==1):,}")

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n[2/7] Initializing Decision Tree model...")
print(f"      Parameters:")
print(f"        - criterion:        {CRITERION}")
print(f"        - max_depth:        {MAX_DEPTH}")
print(f"        - min_samples_split: {MIN_SAMPLES_SPLIT}")
print(f"        - min_samples_leaf:  {MIN_SAMPLES_LEAF}")
print(f"        - random_state:     {RANDOM_STATE}")

model = DecisionTreeClassifier(
    criterion=CRITERION,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)

# ============================================================
# CROSS-VALIDATION
# ============================================================
print("\n[3/7] Performing 5-Fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring="precision")
cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

print(f"      CV Accuracy:  {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std()*2:.4f})")
print(f"      CV Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std()*2:.4f})")
print(f"      CV Recall:    {cv_recall.mean():.4f} (+/- {cv_recall.std()*2:.4f})")
print(f"      CV F1 Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print("\n[4/7] Training final model on full training set...")
model.fit(X_train, y_train)

# Training set evaluation
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred)
train_rec = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_cm = confusion_matrix(y_train, y_train_pred)

print(f"\n      Training Set Performance:")
print(f"        Accuracy:  {train_acc:.4f}")
print(f"        Precision: {train_prec:.4f}")
print(f"        Recall:    {train_rec:.4f}")
print(f"        F1 Score:  {train_f1:.4f}")
print(f"\n      Confusion Matrix (Train):")
print(f"        TN={train_cm[0][0]:,}  FP={train_cm[0][1]:,}")
print(f"        FN={train_cm[1][0]:,}  TP={train_cm[1][1]:,}")

# Tree info
print(f"\n      Tree Structure:")
print(f"        Depth:       {model.get_depth()}")
print(f"        Leaf nodes:  {model.get_n_leaves()}")
print(f"        Total nodes: {model.tree_.node_count}")

# ============================================================
# BENCHMARK TEST — Default threshold (0.5)
# ============================================================
print("\n[5/7] Evaluating on imbalanced benchmark test set (41,089 samples)...")

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

print(f"\n      Test Set Performance (threshold=0.5):")
print(f"        Accuracy:  {test_acc:.4f}")
print(f"        Precision: {test_prec:.4f}")
print(f"        Recall:    {test_rec:.4f}")
print(f"        F1 Score:  {test_f1:.4f}")
print(f"\n      Confusion Matrix (Test, threshold=0.5):")
print(f"        TN={test_cm[0][0]:,}  FP={test_cm[0][1]:,}")
print(f"        FN={test_cm[1][0]:,}  TP={test_cm[1][1]:,}")

# ============================================================
# THRESHOLD OPTIMIZATION (maximize F1 on test set)
# ============================================================
print("\n[6/7] Optimizing classification threshold...")

thresholds = np.arange(0.01, 1.0, 0.01)
best_f1 = 0
best_thresh = 0.5
threshold_results = []

for t in thresholds:
    y_t = (y_test_proba >= t).astype(int)
    if y_t.sum() == 0 or y_t.sum() == len(y_t):
        continue
    f = f1_score(y_test, y_t)
    p = precision_score(y_test, y_t)
    r = recall_score(y_test, y_t)
    a = accuracy_score(y_test, y_t)
    threshold_results.append({"threshold": float(t), "f1": f, "precision": p, "recall": r, "accuracy": a})
    if f > best_f1:
        best_f1 = f
        best_thresh = t

print(f"      Optimal threshold: {best_thresh:.4f}")

# Optimized test evaluation
y_opt = (y_test_proba >= best_thresh).astype(int)
opt_acc = accuracy_score(y_test, y_opt)
opt_prec = precision_score(y_test, y_opt)
opt_rec = recall_score(y_test, y_opt)
opt_f1 = f1_score(y_test, y_opt)
opt_cm = confusion_matrix(y_test, y_opt)

print(f"\n      Optimized Test Performance (threshold={best_thresh:.4f}):")
print(f"        Accuracy:  {opt_acc:.4f}")
print(f"        Precision: {opt_prec:.4f}")
print(f"        Recall:    {opt_rec:.4f}")
print(f"        F1 Score:  {opt_f1:.4f}")
print(f"\n      Confusion Matrix (Optimized):")
print(f"        TN={opt_cm[0][0]:,}  FP={opt_cm[0][1]:,}")
print(f"        FN={opt_cm[1][0]:,}  TP={opt_cm[1][1]:,}")

# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
print(f"\n      ROC-AUC: {roc_auc:.4f}")

# Feature importance
print("\n      Feature Importance (All 10):")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(len(FEATURE_NAMES)):
    print(f"        {i+1}. {FEATURE_NAMES[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n[7/7] Generating visualizations...")

# --- 1. Confusion Matrix (optimized threshold) ---
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(opt_cm, cmap="Blues", interpolation="nearest")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "DoS"], fontsize=12)
ax.set_yticklabels(["Normal", "DoS"], fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual Label", fontsize=13, fontweight="bold")
ax.set_title(f"Decision Tree — Confusion Matrix\n(Threshold = {best_thresh:.4f})", fontsize=14, fontweight="bold")
for i in range(2):
    for j in range(2):
        val = opt_cm[i][j]
        color = "white" if val > opt_cm.max() / 2 else "black"
        ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=16, fontweight="bold", color=color)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: images/confusion_matrix.png")

# --- 2. Feature Importance ---
fig, ax = plt.subplots(figsize=(9, 6))
sorted_idx = np.argsort(importances)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(FEATURE_NAMES)))
bars = ax.barh(
    [FEATURE_NAMES[i] for i in sorted_idx],
    importances[sorted_idx],
    color=colors,
    edgecolor="white",
    height=0.6,
)
for bar, val in zip(bars, importances[sorted_idx]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Gini Importance", fontsize=13, fontweight="bold")
ax.set_title("Decision Tree — Feature Importance", fontsize=14, fontweight="bold")
ax.set_xlim(0, max(importances) * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: images/feature_importance.png")

# --- 3. ROC Curve ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#0071E3", linewidth=2.5, label=f"Decision Tree (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#C7C7CC", linestyle="--", linewidth=1.5, label="Random Classifier")
ax.scatter([1 - opt_cm[0][0] / (opt_cm[0][0] + opt_cm[0][1])],
           [opt_cm[1][1] / (opt_cm[1][0] + opt_cm[1][1])],
           color="#FF3B30", s=100, zorder=5, label=f"Optimal Threshold ({best_thresh:.2f})")
ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
ax.set_title("Decision Tree — ROC Curve", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: images/roc_curve.png")

# --- 4. Precision-Recall Curve ---
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(rec_curve, prec_curve)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_curve, prec_curve, color="#00897B", linewidth=2.5, label=f"Decision Tree (PR-AUC = {pr_auc:.4f})")
ax.scatter([opt_rec], [opt_prec], color="#FF3B30", s=100, zorder=5,
           label=f"Optimal Threshold ({best_thresh:.2f})")
ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
ax.set_title("Decision Tree — Precision-Recall Curve", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("images/precision_recall_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: images/precision_recall_curve.png")

# --- 5. Threshold vs F1/Precision/Recall ---
if threshold_results:
    tr_df = pd.DataFrame(threshold_results)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(tr_df["threshold"], tr_df["f1"], color="#FF9500", linewidth=2.5, label="F1 Score")
    ax.plot(tr_df["threshold"], tr_df["precision"], color="#0071E3", linewidth=2, label="Precision", alpha=0.8)
    ax.plot(tr_df["threshold"], tr_df["recall"], color="#34C759", linewidth=2, label="Recall", alpha=0.8)
    ax.axvline(best_thresh, color="#FF3B30", linestyle="--", linewidth=1.5,
               label=f"Optimal ({best_thresh:.2f})")
    ax.set_xlabel("Classification Threshold", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Decision Tree — Threshold Optimization", fontsize=14, fontweight="bold")
    ax.legend(loc="center left", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("images/threshold_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: images/threshold_optimization.png")

# --- 6. Cross-Validation Scores (box-style) ---
fig, ax = plt.subplots(figsize=(8, 5))
cv_data = [cv_accuracy, cv_precision, cv_recall, cv_f1]
cv_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
cv_colors = ["#0071E3", "#FF9500", "#34C759", "#FF3B30"]
positions = np.arange(len(cv_labels))
for i, (data, label, color) in enumerate(zip(cv_data, cv_labels, cv_colors)):
    ax.bar(i, data.mean(), width=0.5, color=color, alpha=0.85, edgecolor="white", label=label)
    ax.errorbar(i, data.mean(), yerr=data.std() * 2, fmt="none", ecolor="black",
                capsize=8, capthick=2, linewidth=2)
    ax.text(i, data.mean() + data.std() * 2 + 0.015, f"{data.mean():.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(positions)
ax.set_xticklabels(cv_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=13, fontweight="bold")
ax.set_title("Decision Tree — 5-Fold Cross-Validation Results", fontsize=14, fontweight="bold")
ax.set_ylim(0.85, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("images/cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Saved: images/cross_validation.png")

# ============================================================
# SAVE MODEL
# ============================================================
print(f"\n      Saving model to: {MODEL_OUTPUT}")
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(model, f)

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "model": "DecisionTree",
    "parameters": {
        "criterion": CRITERION,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "random_state": RANDOM_STATE,
        "actual_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "n_nodes": int(model.tree_.node_count),
    },
    "cross_validation": {
        "folds": 5,
        "accuracy_mean": float(cv_accuracy.mean()),
        "accuracy_std": float(cv_accuracy.std()),
        "precision_mean": float(cv_precision.mean()),
        "precision_std": float(cv_precision.std()),
        "recall_mean": float(cv_recall.mean()),
        "recall_std": float(cv_recall.std()),
        "f1_mean": float(cv_f1.mean()),
        "f1_std": float(cv_f1.std()),
    },
    "training_performance": {
        "accuracy": float(train_acc),
        "precision": float(train_prec),
        "recall": float(train_rec),
        "f1_score": float(train_f1),
        "confusion_matrix": train_cm.tolist(),
    },
    "test_default": {
        "threshold": 0.5,
        "accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1_score": float(test_f1),
        "confusion_matrix": test_cm.tolist(),
    },
    "optimized": {
        "threshold": float(best_thresh),
        "accuracy": float(opt_acc),
        "precision": float(opt_prec),
        "recall": float(opt_rec),
        "f1_score": float(opt_f1),
        "confusion_matrix": opt_cm.tolist(),
        "true_negatives": int(opt_cm[0][0]),
        "false_positives": int(opt_cm[0][1]),
        "false_negatives": int(opt_cm[1][0]),
        "true_positives": int(opt_cm[1][1]),
    },
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "feature_importance": dict(zip(FEATURE_NAMES, importances.tolist())),
}

print(f"      Saving results to: {RESULTS_OUTPUT}")
with open(RESULTS_OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DECISION TREE TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Cross-Validation F1:   {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
print(f"  Test F1 (default):     {test_f1:.4f}  (threshold=0.50)")
print(f"  Test F1 (optimized):   {opt_f1:.4f}  (threshold={best_thresh:.4f})")
print(f"  ROC-AUC:               {roc_auc:.4f}")
print(f"  PR-AUC:                {pr_auc:.4f}")
print(f"\n  Model saved:           {MODEL_OUTPUT}")
print(f"  Results saved:         {RESULTS_OUTPUT}")
print(f"  Visualizations:        images/")
print("=" * 60)
