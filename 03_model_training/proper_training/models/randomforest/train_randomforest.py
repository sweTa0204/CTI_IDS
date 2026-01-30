"""
Random Forest Model Training Script for DoS Attack Detection
=============================================================

This script trains a Random Forest classifier for binary classification
of network traffic (DoS Attack vs Normal Traffic).

Dataset: UNSW-NB15 (Official Training Set)
Training Samples: 24,528 (12,264 DoS + 12,264 Normal)
Features: 10 selected features

Author: Research Project
Date: 2026-01-28
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
N_JOBS = -1  # Use all CPU cores

# Paths (relative to project root)
DATA_PATH = "../../data/X_train_scaled.csv"
LABELS_PATH = "../../data/y_train.csv"
MODEL_OUTPUT = "randomforest_model.pkl"
RESULTS_OUTPUT = "../../results/randomforest_results.json"

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Random Forest Model Training for DoS Detection")
print("=" * 60)

print("\n[1/4] Loading training data...")
X_train = pd.read_csv(DATA_PATH).values
y_train = pd.read_csv(LABELS_PATH).values.ravel()

print(f"      Training samples: {X_train.shape[0]:,}")
print(f"      Features: {X_train.shape[1]}")
print(f"      Class distribution: Normal={sum(y_train==0):,}, DoS={sum(y_train==1):,}")

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n[2/4] Initializing Random Forest model...")
print(f"      Parameters:")
print(f"        - n_estimators: {N_ESTIMATORS}")
print(f"        - max_depth: {MAX_DEPTH}")
print(f"        - n_jobs: {N_JOBS} (parallel processing)")
print(f"        - random_state: {RANDOM_STATE}")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

# ============================================================
# CROSS-VALIDATION
# ============================================================
print("\n[3/4] Performing 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')
cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

print(f"      CV Accuracy:  {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std()*2:.4f})")
print(f"      CV Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std()*2:.4f})")
print(f"      CV Recall:    {cv_recall.mean():.4f} (+/- {cv_recall.std()*2:.4f})")
print(f"      CV F1 Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print("\n[4/4] Training final model on full dataset...")
model.fit(X_train, y_train)

# Training set evaluation
y_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred)
train_precision = precision_score(y_train, y_pred)
train_recall = recall_score(y_train, y_pred)
train_f1 = f1_score(y_train, y_pred)
train_cm = confusion_matrix(y_train, y_pred)

print(f"\n      Training Set Performance:")
print(f"        - Accuracy:  {train_accuracy:.4f}")
print(f"        - Precision: {train_precision:.4f}")
print(f"        - Recall:    {train_recall:.4f}")
print(f"        - F1 Score:  {train_f1:.4f}")

print(f"\n      Confusion Matrix:")
print(f"        TN={train_cm[0][0]:,}  FP={train_cm[0][1]:,}")
print(f"        FN={train_cm[1][0]:,}  TP={train_cm[1][1]:,}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n      Feature Importance (Top 5):")
feature_names = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(5):
    print(f"        {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================
print(f"\n      Saving model to: {MODEL_OUTPUT}")
with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(model, f)

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "model": "RandomForest",
    "parameters": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_STATE
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
        "f1_std": float(cv_f1.std())
    },
    "training_performance": {
        "accuracy": float(train_accuracy),
        "precision": float(train_precision),
        "recall": float(train_recall),
        "f1_score": float(train_f1),
        "confusion_matrix": train_cm.tolist()
    },
    "feature_importance": dict(zip(feature_names, importances.tolist()))
}

print(f"      Saving results to: {RESULTS_OUTPUT}")
with open(RESULTS_OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
