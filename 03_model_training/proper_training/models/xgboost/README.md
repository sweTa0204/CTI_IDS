# XGBoost Model for DoS Attack Detection

## Overview

This directory contains the XGBoost (eXtreme Gradient Boosting) classifier trained for binary classification of network traffic into DoS Attack or Normal Traffic.

---

## Model Information

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost (Gradient Boosted Decision Trees) |
| **Task** | Binary Classification |
| **Classes** | 0 = Normal Traffic, 1 = DoS Attack |
| **Training Samples** | 24,528 (balanced: 12,264 DoS + 12,264 Normal) |
| **Features** | 10 |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 100 | Number of boosting rounds (trees) |
| `max_depth` | 6 | Maximum depth of each tree |
| `learning_rate` | 0.1 | Step size shrinkage to prevent overfitting |
| `eval_metric` | logloss | Evaluation metric for validation |
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
| **Accuracy** | 96.42% | ±0.47% |
| **Precision** | 97.08% | ±0.52% |
| **Recall** | 95.71% | ±0.68% |
| **F1 Score** | 96.45% | ±0.47% |

### Training Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.47% |
| **Precision** | 98.16% |
| **Recall** | 96.75% |
| **F1 Score** | 97.45% |

### Confusion Matrix (Training Set)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 11,963 (TN) | 301 (FP) |
| **Actual DoS** | 319 (FN) | 11,945 (TP) |

---

## External Benchmark Performance

Tested on **completely unseen data** from the Official UNSW-NB15 Testing Set.

**Benchmark Dataset:** 41,089 samples (4,089 DoS + 37,000 Normal)

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.81% |
| **Precision** | 66.78% |
| **Recall** | **95.28%** |
| **F1 Score** | 78.52% |

### Confusion Matrix (Benchmark)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 35,062 (TN) | 1,938 (FP) |
| **Actual DoS** | 193 (FN) | 3,896 (TP) |

**Key Insight:** The model achieves **95.28% recall**, meaning it correctly identifies 95% of all DoS attacks. The lower precision is due to class imbalance in the test set (9:1 Normal:DoS ratio).

---

## Why XGBoost?

XGBoost was selected for this research because:

1. **Gradient Boosting**: Builds trees sequentially, with each tree correcting errors of previous trees
2. **Regularization**: Built-in L1 and L2 regularization prevents overfitting
3. **Handling Imbalanced Data**: Effective with class weights and evaluation metrics
4. **Feature Importance**: Provides interpretable feature importance scores for XAI integration
5. **Speed**: Optimized implementation with parallel processing
6. **State-of-the-art**: Consistently top performer in machine learning competitions

---

## Files in This Directory

| File | Description |
|------|-------------|
| `xgboost_model.pkl` | Trained XGBoost model (pickle format) |
| `train_xgboost.py` | Training script (reproducible) |
| `README.md` | This documentation file |

---

## How to Load and Use the Model

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
with open('xgboost_model.pkl', 'rb') as f:
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
```

---

## Feature Importance

XGBoost provides feature importance scores based on how frequently each feature is used in tree splits:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `rate` | High - Connection rate is key DoS indicator |
| 2 | `sload` | High - Source load indicates attack volume |
| 3 | `sbytes` | Medium - Byte patterns differ in attacks |
| 4 | `tcprtt` | Medium - RTT anomalies in DoS |
| 5 | `dur` | Medium - Duration patterns |

*Exact importance scores can be extracted using `model.feature_importances_`*

---

## Reproducibility

To reproduce this model:

1. Ensure the training data is in place:
   - `../../data/X_train_scaled.csv`
   - `../../data/y_train.csv`

2. Run the training script:
   ```bash
   python train_xgboost.py
   ```

3. The script uses `random_state=42` for reproducibility.

---

## References

- **XGBoost Paper**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.
- **UNSW-NB15 Dataset**: Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.

---

*Last Updated: 2026-01-28*
