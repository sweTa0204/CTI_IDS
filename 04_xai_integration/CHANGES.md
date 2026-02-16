# 04_xai_integration — Changes

## Overview

This directory contains the SHAP (SHapley Additive exPlanations) integration module that provides explainability for the XGBoost DoS detection model.

---

## Modified File: `shap_explainer.py`

### Change: Directional SHAP Feature Selection

**What was changed:**
The `explain_single()` method in the `SHAPExplainer` class was updated to select top contributing features **directionally** rather than by absolute value.

**Before (old behavior):**
```python
# Sort by absolute contribution to get top features
sorted_features = sorted(feature_contributions.items(),
                         key=lambda x: abs(x[1]),
                         reverse=True)
top_features = [f[0] for f in sorted_features[:3]]
```
This selected the 3 features with the largest absolute SHAP values, regardless of direction. For a DoS prediction, a feature with a large **negative** SHAP (pushing AWAY from DoS, toward Normal) could be incorrectly listed as a "top contributor" to the DoS detection.

**After (new behavior):**
```python
# Select top features pushing TOWARD the prediction
if prediction == 1:  # DoS
    # Positive SHAP = pushes toward DoS class
    directional = sorted(
        [(k, v) for k, v in feature_contributions.items() if v > 0],
        key=lambda x: x[1], reverse=True,
    )
else:  # Normal
    # Negative SHAP = pushes toward Normal class
    directional = sorted(
        [(k, v) for k, v in feature_contributions.items() if v < 0],
        key=lambda x: abs(x[1]), reverse=True,
    )
# Fallback to absolute sort if no features match the expected direction
if not directional:
    directional = sorted(feature_contributions.items(),
                         key=lambda x: abs(x[1]), reverse=True)
top_features = [f[0] for f in directional[:3]]
```

**Why this matters:**
- For DoS predictions: Only features with **positive SHAP** (pushing toward DoS) are selected as top contributors. These are the actual attack indicators.
- For Normal predictions: Only features with **negative SHAP** (pushing toward Normal) are selected.
- A **fallback** to absolute sorting is included for edge cases where all features push in the opposite direction.
- This directly improves attack classification accuracy, because the `AttackClassifier` uses these top features to determine attack type (Volumetric, Protocol Exploit, Slowloris, Amplification).

**Impact:**
This change is propagated consistently across `04_xai_integration`, `05_mitigation_framework`, `06_complete_testing`, and `06_dashboard` — all components now use directional SHAP selection.
