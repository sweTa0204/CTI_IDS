# Multi-Layer Perceptron (MLP) Neural Network for DoS Attack Detection

## Overview

This directory contains the MLP Neural Network classifier trained for binary classification of network traffic into DoS Attack or Normal Traffic.

---

## Model Information

| Property | Value |
|----------|-------|
| **Algorithm** | Multi-Layer Perceptron (Feedforward Neural Network) |
| **Task** | Binary Classification |
| **Classes** | 0 = Normal Traffic, 1 = DoS Attack |
| **Training Samples** | 24,528 (balanced: 12,264 DoS + 12,264 Normal) |
| **Features** | 10 |

---

## Network Architecture

```
Input Layer          Hidden Layer 1      Hidden Layer 2      Output Layer
(10 neurons)         (100 neurons)       (50 neurons)        (2 neurons)

    ●  ────────────────►  ●  ───────────────►  ●  ───────────►  ● (Normal)
    ●                     ●                    ●
    ●                     ●                    ●                 ● (DoS)
    ●        ReLU         ●        ReLU        ●      Softmax
    ●  ────────────────►  ●  ───────────────►  ●  ───────────►
    ●                     ●                    ●
    ●                     ●                    ●
    ●                     .                    .
    ●                     .                    .
    ●                     .                    .

 Features            100 neurons          50 neurons         Probabilities
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_layer_sizes` | (100, 50) | Two hidden layers with 100 and 50 neurons |
| `activation` | relu | Rectified Linear Unit activation function |
| `solver` | adam | Adaptive Moment Estimation optimizer |
| `max_iter` | 500 | Maximum training iterations |
| `early_stopping` | True | Stop training when validation score plateaus |
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
- Feature scaling: StandardScaler (mean=0, std=1) - **Critical for Neural Networks**

---

## Training Performance

### Cross-Validation Results (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| **Accuracy** | 94.26% | ±0.60% |
| **Precision** | 95.38% | ±0.72% |
| **Recall** | 93.02% | ±0.88% |
| **F1 Score** | 94.32% | ±0.60% |

### Training Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.70% |
| **Precision** | 95.25% |
| **Recall** | 94.10% |
| **F1 Score** | 94.67% |

### Confusion Matrix (Training Set)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 11,707 (TN) | 557 (FP) |
| **Actual DoS** | 723 (FN) | 11,541 (TP) |

---

## External Benchmark Performance

Tested on **completely unseen data** from the Official UNSW-NB15 Testing Set.

**Benchmark Dataset:** 41,089 samples (4,089 DoS + 37,000 Normal)

| Metric | Value |
|--------|-------|
| **Accuracy** | 90.63% |
| **Precision** | 51.64% |
| **Recall** | **92.08%** |
| **F1 Score** | 66.17% |

### Confusion Matrix (Benchmark)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 33,474 (TN) | 3,526 (FP) |
| **Actual DoS** | 324 (FN) | 3,765 (TP) |

**Key Insight:** MLP achieves **92.08% recall**, detecting 92% of DoS attacks on unseen data.

---

## Why MLP (Neural Network)?

MLP was included in this research because:

1. **Universal Approximator**: Can learn any continuous function with enough neurons
2. **Non-Linear Patterns**: Captures complex, non-linear relationships in data
3. **Automatic Feature Learning**: Learns feature representations internally
4. **Scalability**: Can be extended to deep learning architectures
5. **Foundation for XAI**: Compatible with techniques like LIME and SHAP for explainability

### How MLP Works

```
Forward Pass:
─────────────

Input (x)  →  Hidden Layer 1  →  Hidden Layer 2  →  Output
              z1 = W1·x + b1      z2 = W2·a1 + b2    y = softmax(z3)
              a1 = ReLU(z1)       a2 = ReLU(z2)

Backpropagation:
────────────────

Loss = CrossEntropy(y, y_true)
       ↓
Gradients computed via chain rule
       ↓
Weights updated: W = W - α·∇W (Adam optimizer)
```

### ReLU Activation

```
ReLU(x) = max(0, x)

      │
    y │         ╱
      │       ╱
      │     ╱
    ──┼────●────────► x
      │    0
      │
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `mlp_model.pkl` | Trained MLP model (pickle format) |
| `train_mlp.py` | Training script (reproducible) |
| `README.md` | This documentation file |

---

## How to Load and Use the Model

```python
import pickle
import pandas as pd

# Load the trained model
with open('mlp_model.pkl', 'rb') as f:
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
X_new_scaled = scaler.transform(X_new)  # IMPORTANT: NN requires scaled features

predictions = model.predict(X_new_scaled)
# 0 = Normal, 1 = DoS Attack

# Get prediction probabilities
probabilities = model.predict_proba(X_new_scaled)
# probabilities[:, 0] = P(Normal)
# probabilities[:, 1] = P(DoS)
```

---

## Important Notes for MLP

### Feature Scaling is Critical

Neural networks are **very sensitive to feature scales**:

```python
# Wrong - will give poor predictions
raw_predictions = model.predict(X_new)  # DON'T DO THIS

# Correct - scale features first
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)   # DO THIS
```

### Why (100, 50) Architecture?

| Layer | Neurons | Rationale |
|-------|---------|-----------|
| Input | 10 | Number of features |
| Hidden 1 | 100 | ~10x input size, captures complex patterns |
| Hidden 2 | 50 | Compression layer, learns abstractions |
| Output | 2 | Binary classification |

---

## Comparison with Other Models

| Aspect | MLP | XGBoost | Random Forest | SVM |
|--------|-----|---------|---------------|-----|
| **CV F1** | 94.32% | 96.45% | 96.22% | 92.26% |
| **Benchmark Recall** | 92.08% | 95.28% | 94.35% | 88.24% |
| **Feature Scaling** | Required | Not required | Not required | Required |
| **Interpretability** | Low | Medium | Medium | Low |
| **Training Speed** | Medium | Medium | Fast | Slow |

---

## XAI Integration Potential

MLP is compatible with several explainability techniques:

| Technique | Applicability | Description |
|-----------|---------------|-------------|
| **LIME** | High | Local explanations via linear approximations |
| **SHAP** | High | Shapley values for feature contributions |
| **Gradient-based** | High | Saliency maps, integrated gradients |
| **Feature Ablation** | Medium | Remove features and observe impact |

---

## Reproducibility

To reproduce this model:

1. Ensure the training data is in place:
   - `../../data/X_train_scaled.csv`
   - `../../data/y_train.csv`

2. Run the training script:
   ```bash
   python train_mlp.py
   ```

3. The script uses `random_state=42` for reproducibility.

---

## References

- **Backpropagation**: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
- **Adam Optimizer**: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
- **ReLU**: Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines.
- **UNSW-NB15 Dataset**: Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.

---

*Last Updated: 2026-01-28*
