# Random Forest Model for DoS Attack Detection

## Overview

This directory contains the Random Forest classifier trained for binary classification of network traffic into DoS Attack or Normal Traffic.

---

## Model Information

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest (Ensemble of Decision Trees) |
| **Task** | Binary Classification |
| **Classes** | 0 = Normal Traffic, 1 = DoS Attack |
| **Training Samples** | 24,528 (balanced: 12,264 DoS + 12,264 Normal) |
| **Features** | 10 |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 100 | Number of trees in the forest |
| `max_depth` | 10 | Maximum depth of each tree |
| `n_jobs` | -1 | Use all CPU cores for parallel training |
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
- Feature scaling: StandardScaler (mean=0, std=1)

---

## Training Performance

### Cross-Validation Results (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| **Accuracy** | 96.20% | ±0.63% |
| **Precision** | 97.19% | ±0.72% |
| **Recall** | 95.16% | ±0.85% |
| **F1 Score** | 96.22% | ±0.63% |

### Training Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.11% |
| **Precision** | 97.83% |
| **Recall** | 96.35% |
| **F1 Score** | 97.08% |

### Confusion Matrix (Training Set)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 11,998 (TN) | 266 (FP) |
| **Actual DoS** | 442 (FN) | 11,822 (TP) |

---

## External Benchmark Performance

Tested on **completely unseen data** from the Official UNSW-NB15 Testing Set.

**Benchmark Dataset:** 41,089 samples (4,089 DoS + 37,000 Normal)

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.44% |
| **Precision** | 61.01% |
| **Recall** | **94.35%** |
| **F1 Score** | 74.10% |

### Confusion Matrix (Benchmark)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 34,534 (TN) | 2,466 (FP) |
| **Actual DoS** | 231 (FN) | 3,858 (TP) |

**Key Insight:** The model achieves **94.35% recall**, successfully detecting 94% of DoS attacks on unseen data.

---

## Why Random Forest?

Random Forest was selected for this research because:

1. **Ensemble Method**: Combines multiple decision trees to reduce overfitting
2. **Bagging**: Each tree is trained on a random subset of data (bootstrap sampling)
3. **Feature Randomness**: Each split considers a random subset of features
4. **Robustness**: Less sensitive to outliers and noise than single trees
5. **Interpretability**: Provides feature importance scores
6. **No Preprocessing Required**: Can handle unscaled features (though we scale for consistency)
7. **Parallelizable**: Training can be distributed across CPU cores

### How Random Forest Works

```
Input Data
    │
    ▼
┌─────────────────────────────────────┐
│     Bootstrap Sampling (Bagging)    │
│   Create n random subsets of data   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────┐ ┌─────────┐     ┌─────────┐
│ Tree 1  │ │ Tree 2  │ ... │ Tree n  │
│ (subset)│ │ (subset)│     │ (subset)│
└─────────┘ └─────────┘     └─────────┘
    │           │               │
    ▼           ▼               ▼
┌─────────────────────────────────────┐
│         Majority Voting             │
│   Combine predictions from all trees │
└─────────────────────────────────────┘
    │
    ▼
Final Prediction (0 or 1)
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `randomforest_model.pkl` | Trained Random Forest model (pickle format) |
| `train_randomforest.py` | Training script (reproducible) |
| `README.md` | This documentation file |

---

## How to Load and Use the Model

```python
import pickle
import pandas as pd

# Load the trained model
with open('randomforest_model.pkl', 'rb') as f:
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
X_new_scaled = scaler.transform(X_new)

predictions = model.predict(X_new_scaled)
# 0 = Normal, 1 = DoS Attack

# Get prediction probabilities
probabilities = model.predict_proba(X_new_scaled)
# probabilities[:, 0] = P(Normal)
# probabilities[:, 1] = P(DoS)
```

---

## Feature Importance

Random Forest provides feature importance based on mean decrease in impurity (Gini importance):

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `sload` | Source load is primary attack indicator |
| 2 | `rate` | Connection rate differentiates attacks |
| 3 | `sbytes` | Byte volume patterns |
| 4 | `dload` | Destination load patterns |
| 5 | `tcprtt` | TCP timing anomalies |

*Extract exact values using `model.feature_importances_`*

---

## Comparison with XGBoost

| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| **Method** | Bagging (parallel trees) | Boosting (sequential trees) |
| **CV F1** | 96.22% | 96.45% |
| **Benchmark Recall** | 94.35% | 95.28% |
| **Training Speed** | Faster (parallel) | Slower (sequential) |
| **Overfitting Risk** | Lower | Higher (needs tuning) |

---

## Reproducibility

To reproduce this model:

1. Ensure the training data is in place:
   - `../../data/X_train_scaled.csv`
   - `../../data/y_train.csv`

2. Run the training script:
   ```bash
   python train_randomforest.py
   ```

3. The script uses `random_state=42` for reproducibility.

---

## References

- **Random Forest Paper**: Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- **UNSW-NB15 Dataset**: Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.

---

*Last Updated: 2026-01-28*
