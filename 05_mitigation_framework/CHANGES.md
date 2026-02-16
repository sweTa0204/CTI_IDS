# 05_mitigation_framework — Changes

## Overview

This directory contains the attack classification, severity calculation, and mitigation command generation modules that form the automated response framework.

---

## Modified File: `attack_classifier.py`

### Change: Refined Slowloris Detection Thresholds

**What was changed:**
The `_calc_slowloris_score()` method in the `AttackClassifier` class was updated to use stricter thresholds for detecting Slowloris-type attacks.

**Before (old behavior):**
```python
# Slowloris typically has low rate contribution or negative
rate_shap = shap_values.get('rate', 0)
if rate_shap < 0.1:  # Low or negative rate contribution
    score += 0.2

# Check for high sbytes over time (persistent connection)
sbytes_shap = shap_values.get('sbytes', 0)
if sbytes_shap > 0.2 and rate_shap < 0.1:
    score += 0.2
```
The threshold `rate_shap < 0.1` was too permissive — it would match records where `rate` had a small positive SHAP (slightly pushing toward DoS), which is inconsistent with Slowloris behavior.

**After (new behavior):**
```python
# Slowloris typically has negative rate contribution (rate pushes toward Normal)
rate_shap = shap_values.get('rate', 0)
if rate_shap <= 0:  # Rate pushing toward Normal = low-rate attack
    score += 0.2

# Check for high sbytes over time (persistent connection)
sbytes_shap = shap_values.get('sbytes', 0)
if sbytes_shap > 0.2 and rate_shap <= 0:
    score += 0.2
```

**Why this matters:**
- Slowloris attacks are characterized by **low packet rates** (slow, persistent connections). In SHAP terms, the `rate` feature should have a **negative** contribution (pushing toward Normal), because the rate itself is low.
- The old threshold (`< 0.1`) could incorrectly flag records where `rate` had a small positive SHAP as potential Slowloris attacks.
- The new threshold (`<= 0`) strictly requires that `rate` pushes toward Normal, which is consistent with the low-rate nature of Slowloris attacks.
- This aligns with the directional SHAP selection changes made across all other modules.

**Impact:**
More accurate Slowloris classification, reducing false Slowloris labels on high-rate attacks that happen to have moderate rate SHAP values.
