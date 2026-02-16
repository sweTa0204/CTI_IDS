# 06_complete_testing — Changes

## Overview

This directory contains end-to-end testing scripts that validate the full detection pipeline (preprocessing → XGBoost → SHAP → classification → mitigation).

---

## Modified Files

### 1. `demo_single_sample.py`

**Change: Directional top-feature selection for DoS display**

The `demo_pipeline()` function was updated to select top contributing features using **only positive SHAP values** (features pushing toward DoS), rather than sorting by absolute value.

**Before:**
```python
# Top features
top_features = [f[0] for f in sorted_shap[:3]]
```

**After:**
```python
# Top features: only those with positive SHAP (pushing toward DoS)
dos_features = sorted(
    [(k, v) for k, v in shap_dict.items() if v > 0],
    key=lambda x: x[1], reverse=True,
)
top_features = [f[0] for f in dos_features[:3]]
```

**Why:** When demonstrating a DoS detection, the "top 3 contributing features" should only include features that actually contributed to the DoS prediction (positive SHAP). Features with negative SHAP values pushed the model toward Normal and should not be listed as DoS contributors.

The display of all SHAP values (sorted by absolute magnitude) is preserved separately for the full table printout, so no information is lost.

---

### 2. `run_complete_test.py`

**Change: Directional top-feature selection in `explain_single_with_threshold()`**

The function was updated to select top features based on prediction direction:

**Before:**
```python
# Get top features (sorted by absolute SHAP value)
sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
top_features = [f[0] for f in sorted_features]
```

**After:**
```python
# Select top features pushing TOWARD the prediction
if prediction == 1:  # DoS
    directional = sorted(
        [(k, v) for k, v in shap_dict.items() if v > 0],
        key=lambda x: x[1], reverse=True,
    )
else:  # Normal
    directional = sorted(
        [(k, v) for k, v in shap_dict.items() if v < 0],
        key=lambda x: abs(x[1]), reverse=True,
    )
top_features = [f[0] for f in directional]
```

**Why:** Same rationale as above — top features should be directionally consistent with the prediction. For DoS: only positive SHAP features. For Normal: only negative SHAP features. This ensures that the attack classifier receives correct inputs and the complete test pipeline matches the dashboard behavior.

---

## Consistency Note

Both changes align with the identical update made in:
- `04_xai_integration/shap_explainer.py` (the `SHAPExplainer` class)
- `06_dashboard/src/pipeline.py` (the dashboard's `_top_features_for_row()` function)

All 4 locations now use directional SHAP selection consistently.
