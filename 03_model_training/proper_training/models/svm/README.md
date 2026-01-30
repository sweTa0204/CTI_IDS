# Support Vector Machine (SVM) Model for DoS Attack Detection

## Overview

This directory contains the SVM classifier with RBF kernel trained for binary classification of network traffic into DoS Attack or Normal Traffic.

---

## Model Information

| Property | Value |
|----------|-------|
| **Algorithm** | Support Vector Machine (SVM) |
| **Kernel** | RBF (Radial Basis Function) |
| **Task** | Binary Classification |
| **Classes** | 0 = Normal Traffic, 1 = DoS Attack |
| **Training Samples** | 24,528 (balanced: 12,264 DoS + 12,264 Normal) |
| **Features** | 10 |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kernel` | rbf | Radial Basis Function kernel |
| `C` | 1.0 | Regularization parameter (penalty for misclassification) |
| `gamma` | scale | Kernel coefficient (1 / (n_features * X.var())) |
| `probability` | True | Enable probability estimates |
| `random_state` | 42 | Seed for reproducibility |

---

## Features Used

The model uses 10 network traffic features:

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `rate` | Connection rate (packets/second) | Numeric |
| 2 | `sload` | Source bits per second | Numeric |
| 3 | `sbytes` | Source to destination bytes | Numeric |
| 4 | `dload` | Destination bits per second | Numeric |
| 5 | `proto` | Protocol type (encoded) | Categorical |
| 6 | `dtcpb` | Destination TCP base sequence number | Numeric |
| 7 | `stcpb` | Source TCP base sequence number | Numeric |
| 8 | `dmean` | Mean of packet size from destination | Numeric |
| 9 | `tcprtt` | TCP round trip time | Numeric |
| 10 | `dur` | Connection duration | Numeric |

**Preprocessing Applied:**
- Categorical encoding: `proto` encoded using LabelEncoder
- Feature scaling: StandardScaler (mean=0, std=1) - **Critical for SVM**

---

## Training Performance

### Cross-Validation Results (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| **Accuracy** | 92.66% | ±0.74% |
| **Precision** | 94.21% | ±0.89% |
| **Recall** | 90.86% | ±1.12% |
| **F1 Score** | 92.26% | ±0.74% |

### Training Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 92.72% |
| **Precision** | 93.94% |
| **Recall** | 91.32% |
| **F1 Score** | 92.62% |

### Confusion Matrix (Training Set)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 11,533 (TN) | 731 (FP) |
| **Actual DoS** | 1,053 (FN) | 11,211 (TP) |

---

## External Benchmark Performance

Tested on **completely unseen data** from the Official UNSW-NB15 Testing Set.

**Benchmark Dataset:** 41,089 samples (4,089 DoS + 37,000 Normal)

| Metric | Value |
|--------|-------|
| **Accuracy** | 85.72% |
| **Precision** | 40.11% |
| **Recall** | **88.24%** |
| **F1 Score** | 55.15% |

### Confusion Matrix (Benchmark)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 31,612 (TN) | 5,388 (FP) |
| **Actual DoS** | 481 (FN) | 3,608 (TP) |

**Key Insight:** SVM achieves **88.24% recall** but lower precision due to higher false positive rate on imbalanced test data.

---

## Why SVM?

SVM was included in this research because:

1. **Maximum Margin Classifier**: Finds the optimal hyperplane that maximizes the margin between classes
2. **Kernel Trick**: RBF kernel maps data to higher dimensions for non-linear separation
3. **Theoretical Foundation**: Strong mathematical guarantees from statistical learning theory
4. **Effective in High Dimensions**: Works well even when features > samples
5. **Robust to Outliers**: Only support vectors influence the decision boundary

### How SVM with RBF Kernel Works

```
Original Feature Space              Higher-Dimensional Space
     (Non-separable)                    (Separable)

    ●  ○  ○  ●                           ●    ●
  ○  ●    ●  ○        RBF Kernel        /  ●  \
    ●  ○  ●  ○       ─────────────►    ○ ─────── ○
  ○    ●  ○  ●                          \  ○  /
    ○  ●  ●  ○                           ○    ○

● = DoS Attack                     Hyperplane separates classes
○ = Normal Traffic                 in transformed space
```

### RBF Kernel Formula

```
K(x, x') = exp(-γ ||x - x'||²)

where:
- γ (gamma) = kernel coefficient
- ||x - x'|| = Euclidean distance between points
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `svm_model.pkl` | Trained SVM model (pickle format) |
| `train_svm.py` | Training script (reproducible) |
| `README.md` | This documentation file |

---

## How to Load and Use the Model

```python
import pickle
import pandas as pd

# Load the trained model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessing artifacts (from ../../data/)
with open('../../data/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../../data/proto_encoder.pkl', 'rb') as f:
    proto_encoder = pickle.load(f)

# Prepare your data (must have same 10 features)
features = ['rate', 'sload', 'sbytes', 'dload', 'proto',
            'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']

# Example prediction
X_new = your_data[features].copy()
X_new['proto'] = proto_encoder.transform(X_new['proto'].astype(str))
X_new_scaled = scaler.transform(X_new)  # IMPORTANT: SVM requires scaled features

predictions = model.predict(X_new_scaled)
# 0 = Normal, 1 = DoS Attack

# Get prediction probabilities (slower for SVM)
probabilities = model.predict_proba(X_new_scaled)
```

---

## Important Notes for SVM

### Feature Scaling is Critical

SVM with RBF kernel is **sensitive to feature scales**. Always use the same scaler:

```python
# Wrong - will give incorrect predictions
raw_predictions = model.predict(X_new)  # DON'T DO THIS

# Correct - scale features first
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)   # DO THIS
```

### Computational Complexity

- **Training Time**: O(n² to n³) - can be slow on large datasets
- **Prediction Time**: O(n_sv × n_features) where n_sv = number of support vectors
- **Memory**: Stores all support vectors

---

## Comparison with Tree-Based Models

| Aspect | SVM (RBF) | XGBoost | Random Forest |
|--------|-----------|---------|---------------|
| **CV F1** | 92.26% | 96.45% | 96.22% |
| **Benchmark Recall** | 88.24% | 95.28% | 94.35% |
| **Training Speed** | Slow | Medium | Fast |
| **Feature Scaling** | Required | Not required | Not required |
| **Interpretability** | Low | Medium | Medium |

---

## When to Use SVM

**Good for:**
- Smaller datasets (< 100k samples)
- High-dimensional data
- When a clear margin of separation exists
- Binary classification tasks

**Not ideal for:**
- Very large datasets (use Linear SVM or SGDClassifier instead)
- When interpretability is crucial
- Multi-class problems (though possible with one-vs-one)

---

## Reproducibility

To reproduce this model:

1. Ensure the training data is in place:
   - `../../data/X_train_scaled.csv`
   - `../../data/y_train.csv`

2. Run the training script:
   ```bash
   python train_svm.py
   ```

3. The script uses `random_state=42` for reproducibility.

---

## References

- **SVM Paper**: Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
- **RBF Kernel**: Scholkopf, B., & Smola, A. J. (2002). Learning with Kernels.
- **UNSW-NB15 Dataset**: Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.

---

*Last Updated: 2026-01-28*
