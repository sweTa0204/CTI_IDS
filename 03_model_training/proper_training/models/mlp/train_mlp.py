"""
Multi-Layer Perceptron (MLP) Model Training Script for DoS Attack Detection
============================================================================

This script trains a Neural Network (MLP) classifier for binary classification
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
from sklearn.neural_network import MLPClassifier
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
HIDDEN_LAYERS = (100, 50)  # Two hidden layers: 100 and 50 neurons
MAX_ITER = 500
EARLY_STOPPING = True
ACTIVATION = 'relu'
SOLVER = 'adam'

# Paths (relative to project root)
DATA_PATH = "../../data/X_train_scaled.csv"
LABELS_PATH = "../../data/y_train.csv"
MODEL_OUTPUT = "mlp_model.pkl"
RESULTS_OUTPUT = "../../results/mlp_results.json"

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("MLP (Neural Network) Model Training for DoS Detection")
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
print("\n[2/4] Initializing MLP model...")
print(f"      Network Architecture:")
print(f"        - Input layer: {X_train.shape[1]} neurons (features)")
print(f"        - Hidden layer 1: {HIDDEN_LAYERS[0]} neurons")
print(f"        - Hidden layer 2: {HIDDEN_LAYERS[1]} neurons")
print(f"        - Output layer: 2 neurons (classes)")
print(f"      Parameters:")
print(f"        - activation: {ACTIVATION}")
print(f"        - solver: {SOLVER}")
print(f"        - max_iter: {MAX_ITER}")
print(f"        - early_stopping: {EARLY_STOPPING}")
print(f"        - random_state: {RANDOM_STATE}")

model = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation=ACTIVATION,
    solver=SOLVER,
    max_iter=MAX_ITER,
    early_stopping=EARLY_STOPPING,
    random_state=RANDOM_STATE,
    verbose=False
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

# Network information
print(f"\n      Network Info:")
print(f"        - Iterations completed: {model.n_iter_}")
print(f"        - Final loss: {model.loss_:.6f}")
print(f"        - Number of layers: {model.n_layers_}")

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
    "model": "MLP",
    "parameters": {
        "hidden_layer_sizes": HIDDEN_LAYERS,
        "activation": ACTIVATION,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
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
    "training_info": {
        "iterations": int(model.n_iter_),
        "final_loss": float(model.loss_),
        "n_layers": int(model.n_layers_)
    }
}

print(f"      Saving results to: {RESULTS_OUTPUT}")
with open(RESULTS_OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
