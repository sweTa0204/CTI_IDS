# Logistic Regression Model for DoS Attack Detection

## Overview

This directory contains the Logistic Regression classifier trained for binary classification of network traffic into DoS Attack or Normal Traffic.

---

## Model Information

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression (Linear Model) |
| **Task** | Binary Classification |
| **Classes** | 0 = Normal Traffic, 1 = DoS Attack |
| **Training Samples** | 24,528 (balanced: 12,264 DoS + 12,264 Normal) |
| **Features** | 10 |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `penalty` | l2 | L2 regularization (Ridge) to prevent overfitting |
| `solver` | lbfgs | Limited-memory BFGS optimizer |
| `max_iter` | 1000 | Maximum iterations for convergence |
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
| **Accuracy** | 86.64% | ±1.15% |
| **Precision** | 90.11% | ±1.24% |
| **Recall** | 82.05% | ±1.82% |
| **F1 Score** | 86.27% | ±1.15% |

### Training Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 86.76% |
| **Precision** | 89.83% |
| **Recall** | 82.91% |
| **F1 Score** | 86.23% |

### Confusion Matrix (Training Set)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 11,111 (TN) | 1,153 (FP) |
| **Actual DoS** | 2,097 (FN) | 10,167 (TP) |

---

## External Benchmark Performance

Tested on **completely unseen data** from the Official UNSW-NB15 Testing Set.

**Benchmark Dataset:** 41,089 samples (4,089 DoS + 37,000 Normal)

| Metric | Value |
|--------|-------|
| **Accuracy** | 82.69% |
| **Precision** | 33.68% |
| **Recall** | **76.25%** |
| **F1 Score** | 46.72% |

### Confusion Matrix (Benchmark)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 30,860 (TN) | 6,140 (FP) |
| **Actual DoS** | 971 (FN) | 3,118 (TP) |

**Key Insight:** Logistic Regression shows the lowest performance among the 5 models, indicating that DoS attack detection requires capturing non-linear patterns that tree-based models handle better.

---

## Why Logistic Regression?

Logistic Regression was included in this research because:

1. **Baseline Model**: Provides a baseline for comparing complex models
2. **Interpretability**: Coefficients directly show feature importance
3. **Speed**: Very fast training and prediction
4. **Probabilistic Output**: Outputs calibrated probabilities
5. **Well-Understood**: Extensive theoretical foundation

### How Logistic Regression Works

```
Decision Function:

z = w₀ + w₁x₁ + w₂x₂ + ... + w₁₀x₁₀

P(DoS|x) = σ(z) = 1 / (1 + e^(-z))

where:
- wᵢ = learned coefficients
- xᵢ = input features
- σ = sigmoid function
```

### Sigmoid Function

```
P(y=1)
   1 ┼─────────────────────────●●●●●●●
     │                      ●●
     │                   ●●
 0.5 ┼─────────────────●
     │              ●●
     │           ●●
   0 ┼●●●●●●●●●●●──────────────────────
     └─────────────────────────────────► z
               Decision Boundary
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `logisticregression_model.pkl` | Trained Logistic Regression model (pickle format) |
| `train_logisticregression.py` | Training script (reproducible) |
| `README.md` | This documentation file |

---

## How to Load and Use the Model

```python
import pickle
import pandas as pd

# Load the trained model
with open('logisticregression_model.pkl', 'rb') as f:
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

# Access coefficients for interpretability
print("Feature Coefficients:", model.coef_[0])
print("Intercept:", model.intercept_[0])
```

---

## Feature Importance via Coefficients

Logistic Regression provides direct interpretability through coefficients:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| Positive coefficient | > 0 | Increases probability of DoS classification |
| Negative coefficient | < 0 | Decreases probability of DoS classification |
| Larger magnitude | |Coefficient|| | More influential feature |

**Note**: Coefficients are only meaningful when features are standardized (which we do).

---

## Comparison with Other Models

| Aspect | Logistic Regression | XGBoost | Random Forest | SVM | MLP |
|--------|---------------------|---------|---------------|-----|-----|
| **CV F1** | 86.27% | **96.45%** | 96.22% | 92.26% | 94.32% |
| **Benchmark Recall** | 76.25% | **95.28%** | 94.35% | 88.24% | 92.08% |
| **Interpretability** | **High** | Medium | Medium | Low | Low |
| **Training Speed** | **Fastest** | Medium | Fast | Slow | Medium |
| **Non-linear Patterns** | No | Yes | Yes | Yes | Yes |

---

## Limitations

1. **Linear Decision Boundary**: Can only capture linear relationships
2. **Feature Engineering Required**: Needs manual feature interactions for non-linear patterns
3. **Lower Performance**: Underperforms on complex pattern recognition tasks
4. **Sensitive to Outliers**: Outliers can significantly affect coefficients

---

## When to Use Logistic Regression

**Good for:**
- Baseline comparison
- Interpretability requirements
- Real-time predictions (very fast)
- When relationships are approximately linear

**Not ideal for:**
- Complex non-linear patterns (like DoS detection)
- High-dimensional data without feature selection
- When accuracy is paramount

---

## Reproducibility

To reproduce this model:

1. Ensure the training data is in place:
   - `../../data/X_train_scaled.csv`
   - `../../data/y_train.csv`

2. Run the training script:
   ```bash
   python train_logisticregression.py
   ```

3. The script uses `random_state=42` for reproducibility.

---

## References

- **Logistic Regression**: Cox, D. R. (1958). The regression analysis of binary sequences.
- **L-BFGS**: Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization.
- **UNSW-NB15 Dataset**: Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.

---

*Last Updated: 2026-01-28*
