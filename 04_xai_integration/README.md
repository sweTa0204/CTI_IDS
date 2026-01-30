# 04_xai_integration - XAI Integration Documentation

## Overview

This module provides **Explainable AI (XAI)** capabilities for the DoS detection model. We use **SHAP TreeExplainer** to explain WHY the XGBoost model makes its predictions.

---

## What is XAI?

**XAI = Explainable Artificial Intelligence**

| Without XAI | With XAI |
|-------------|----------|
| "This is a DoS attack" | "This is a DoS attack BECAUSE rate is 12x higher than normal and sload is 10x higher" |

XAI answers the question: **WHY did the model make this prediction?**

---

## XAI Method: SHAP TreeExplainer

### What is SHAP?

**SHAP = SHapley Additive exPlanations**

SHAP calculates how much each feature contributed to the prediction:

```
Prediction: DoS Attack (94% confidence)

Feature Contributions:
  rate:   +0.35  (pushes toward DoS)
  sload:  +0.28  (pushes toward DoS)
  sbytes: +0.15  (pushes toward DoS)
  proto:  +0.05  (pushes toward DoS)
  ...
  ─────────────
  Total = 0.94 (94% DoS probability)
```

### Why TreeExplainer?

SHAP has multiple methods. We use **TreeExplainer** because:

| Reason | Explanation |
|--------|-------------|
| **Optimized for XGBoost** | TreeExplainer is specifically designed for tree-based models |
| **Fast** | Computes in seconds, not hours |
| **Exact** | Mathematically exact values, not approximations |
| **Guaranteed to work** | 100% compatible with XGBoost |

---

## Why NOT LIME?

### What is LIME?

**LIME = Local Interpretable Model-agnostic Explanations**

LIME is another XAI method that works with any model.

### Why We Chose SHAP Over LIME

| Criteria | SHAP TreeExplainer | LIME |
|----------|-------------------|------|
| **Speed** | Fast (seconds) | Slow (can take minutes) |
| **Accuracy** | Exact for trees | Approximation |
| **Designed for XGBoost** | Yes | No (model-agnostic) |
| **Consistency** | Always consistent | Can vary between runs |
| **Academic acceptance** | Gold standard for trees | Good, but less precise |
| **Implementation complexity** | Simple | More complex |

### Our Decision

**We use SHAP only** because:

1. **SHAP TreeExplainer is the gold standard** for tree-based models like XGBoost
2. **LIME adds unnecessary complexity** without significant benefit
3. **One XAI method is sufficient** for academic rigor
4. **Our research novelty is in the Mitigation Framework**, not in using multiple XAI methods

---

## Implementation Details

### Files in This Directory

| File | Purpose |
|------|---------|
| `shap_explainer.py` | Main SHAP explainer class |
| `test_shap.py` | Test script to verify SHAP works |
| `sample_shap_output.json` | Sample output from test run |
| `README.md` | This documentation file |

### How It Works

```
Step 1: Load trained XGBoost model
           |
           v
Step 2: Initialize SHAP TreeExplainer
           |
           v
Step 3: Pass network traffic record (10 features)
           |
           v
Step 4: SHAP calculates contribution of each feature
           |
           v
Step 5: Return explanation with:
        - Prediction (DoS/Normal)
        - Confidence (probability)
        - SHAP values for all 10 features
        - Top 3 contributing features
```

### Code Example

```python
from shap_explainer import SHAPExplainer

# Initialize
explainer = SHAPExplainer()
explainer.load_model()
explainer.initialize_explainer()

# Explain a single record
features = [1200, 850000, 5000000, 50000, 6, 12345, 67890, 500, 0.01, 2]
explanation = explainer.explain_single(features, record_id=1)

print(explanation)
```

---

## Output Format

### Single Record Explanation

```json
{
    "record_id": 20459,
    "prediction": "DoS",
    "prediction_code": 1,
    "confidence": 0.9996,
    "probability_dos": 0.9996,
    "probability_normal": 0.0004,
    "shap_values": {
        "rate": 0.1234,
        "sload": 2.4836,
        "sbytes": 0.7366,
        "dload": 0.1523,
        "proto": 4.0827,
        "dtcpb": 0.0234,
        "stcpb": 0.0156,
        "dmean": 0.0089,
        "tcprtt": 0.0045,
        "dur": 0.0012
    },
    "top_features": ["proto", "sload", "sbytes"],
    "base_value": -1.2345,
    "feature_values": {
        "rate": 1200.0,
        "sload": 850000.0,
        "sbytes": 5000000.0,
        "dload": 50000.0,
        "proto": 6.0,
        "dtcpb": 12345.0,
        "stcpb": 67890.0,
        "dmean": 500.0,
        "tcprtt": 0.01,
        "dur": 2.0
    }
}
```

### Understanding SHAP Values

| SHAP Value | Meaning |
|------------|---------|
| **Positive (+)** | Pushes prediction toward DoS |
| **Negative (-)** | Pushes prediction toward Normal |
| **Larger absolute value** | Stronger contribution |
| **Smaller absolute value** | Weaker contribution |

---

## Test Results

We tested SHAP on 5 sample records (3 DoS, 2 Normal):

### Results Summary

| Sample | Actual | Predicted | Confidence | Match |
|--------|--------|-----------|------------|-------|
| 1 | DoS | DoS | 99.96% | CORRECT |
| 2 | DoS | DoS | 54.53% | CORRECT |
| 3 | DoS | DoS | 99.91% | CORRECT |
| 4 | Normal | Normal | 83.32% | CORRECT |
| 5 | Normal | Normal | 99.97% | CORRECT |

**All 5 predictions were correct!**

### DoS Detection Example (Sample 1)

```
Record ID:        20459
Actual Label:     DoS
Model Prediction: DoS
Confidence:       99.96%

Top 3 Contributing Features:
  1. proto:  +4.0827 -> DoS (main cause)
  2. sload:  +2.4836 -> DoS
  3. sbytes: +0.7366 -> DoS
```

**Interpretation:** This traffic was flagged as DoS primarily because of unusual protocol behavior (proto) and high source load (sload).

### Normal Traffic Example (Sample 5)

```
Record ID:        8457
Actual Label:     Normal
Model Prediction: Normal
Confidence:       99.97%

Top 3 Contributing Features:
  1. dload:  -3.6993 -> Normal (main cause)
  2. tcprtt: -1.3943 -> Normal
  3. sload:  -1.3462 -> Normal
```

**Interpretation:** This traffic was classified as Normal because destination load, round-trip time, and source load were all within normal ranges.

### Most Frequent Top Features in DoS Detections

| Feature | Frequency in Top 3 |
|---------|-------------------|
| proto | 3 times |
| sload | 2 times |
| sbytes | 2 times |
| dload | 1 time |
| tcprtt | 1 time |

---

## Connection to Mitigation Framework

The SHAP output feeds directly into the Mitigation Framework (Objective 4):

```
SHAP Output                    Mitigation Framework
───────────                    ────────────────────
top_features: [proto, sload]   --> Attack Classifier
                                   "Volumetric Flood"

confidence: 0.9996             --> Severity Calculator
                                   "HIGH"

shap_values: {...}             --> Mitigation Generator
                                   "iptables -m limit..."
```

---

## How to Run

### Test SHAP Explainer

```bash
cd 04_xai_integration
python test_shap.py
```

### Use in Your Code

```python
from shap_explainer import SHAPExplainer

# Initialize once
explainer = SHAPExplainer()
explainer.load_model()
explainer.initialize_explainer()

# Explain multiple records
explanations = explainer.explain_batch(X_test_data)

# Or explain only DoS detections
dos_explanations = explainer.explain_dos_only(X_test_data, threshold=0.8517)
```

---

## Academic Justification

### Why SHAP is Appropriate

1. **Lundberg & Lee (2017)** introduced SHAP as a unified approach to interpreting predictions
2. **TreeExplainer** provides exact Shapley values for tree ensembles
3. **Widely cited** in ML interpretability literature (>15,000 citations)
4. **Recommended** for XGBoost by the XGBoost documentation itself

### Citation

```
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
model predictions. Advances in neural information processing systems, 30.
```

---

## Summary

| Item | Details |
|------|---------|
| **XAI Method** | SHAP TreeExplainer |
| **Why SHAP** | Optimized for XGBoost, fast, exact |
| **Why NOT LIME** | Slower, approximate, adds complexity |
| **Output** | Feature contributions for each prediction |
| **Connection** | Feeds into Mitigation Framework |
| **Status** | COMPLETED and TESTED |

---

*Created: 2026-01-29*
*Status: Step 1 COMPLETED*
