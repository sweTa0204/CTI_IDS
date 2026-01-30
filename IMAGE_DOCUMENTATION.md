# Image Documentation

## From Detection to Defense: An XAI Powered DoS Prevention System

This document provides detailed documentation for all visualizations generated throughout the research project.

---

## Table of Contents

1. [Model Training Images (Objective 2)](#model-training-images-objective-2)
2. [XAI Integration Images (Objective 3)](#xai-integration-images-objective-3)
3. [Mitigation Framework Images (Objective 4)](#mitigation-framework-images-objective-4)
4. [Complete Test Results Images](#complete-test-results-images)

---

## Model Training Images (Objective 2)

**Location:** `03_model_training/proper_training/images/`

### 01. Testing Set Distribution
**File:** `01_testing_set_distribution.png`

**Description:** Shows the class distribution of the testing dataset used to evaluate model performance.

**Key Information:**
- Dataset: UNSW-NB15 Testing Split
- Total samples: 6,132
- DoS samples: 3,066 (50%)
- Normal samples: 3,066 (50%)
- Purpose: Balanced evaluation of model performance

---

### 02. Training Set Distribution
**File:** `02_training_set_distribution.png`

**Description:** Shows the class distribution of the training dataset used to train the XGBoost model.

**Key Information:**
- Dataset: UNSW-NB15 Training Split
- Total samples: 24,528
- DoS samples: 12,264 (50%)
- Normal samples: 12,264 (50%)
- Purpose: Balanced training for unbiased model

---

### 03. Model Performance Training
**File:** `03_model_performance_training.png`

**Description:** Bar chart comparing performance metrics (Accuracy, Precision, Recall, F1-Score) across different models tested during training.

**Key Information:**
- Models compared: XGBoost, Random Forest, SVM, Logistic Regression
- XGBoost achieved best overall performance
- Metrics shown: Accuracy, Precision, Recall, F1-Score
- All metrics in percentage (%)

---

### 04. XGBoost Confusion Matrix (Training)
**File:** `04_xgboost_confusion_matrix_training.png`

**Description:** Confusion matrix heatmap showing model predictions vs actual labels on the training data.

**Key Information:**
- True Positives (DoS correctly identified as DoS)
- True Negatives (Normal correctly identified as Normal)
- False Positives (Normal incorrectly classified as DoS)
- False Negatives (DoS missed and classified as Normal)

---

### 05. XGBoost Confusion Matrix (Testing)
**File:** `05_xgboost_confusion_matrix_testing.png`

**Description:** Confusion matrix heatmap showing model predictions vs actual labels on the held-out testing data.

**Key Information:**
- Validation of model generalization
- Testing data was never seen during training
- Shows model's real-world performance expectations

---

### 06. XGBoost Feature Importance
**File:** `06_xgboost_feature_importance.png`

**Description:** Bar chart showing the relative importance of each feature in the XGBoost model's decision-making process.

**Key Information:**
- 10 features used in the model
- Feature ranking by importance:
  1. rate (packet rate)
  2. sload (source load)
  3. sbytes (source bytes)
  4. dload (destination load)
  5. proto (protocol type)
  6. dtcpb (destination TCP base)
  7. stcpb (source TCP base)
  8. dmean (destination mean)
  9. tcprtt (TCP round-trip time)
  10. dur (duration)

---

## XAI Integration Images (Objective 3)

**Location:** `04_xai_integration/images/`

### 07. SHAP Summary Plot
**File:** `07_shap_summary_plot.png`

**Description:** Global SHAP summary plot showing how each feature contributes to DoS detection across all samples.

**Key Information:**
- X-axis: SHAP value (impact on model output)
- Y-axis: Features ranked by importance
- Color: Feature value (red = high, blue = low)
- Each dot represents one sample
- Shows both direction and magnitude of feature impact
- Generated from 500 random samples

**Interpretation:**
- Features at the top have the highest impact
- Points to the right increase DoS probability
- Points to the left decrease DoS probability (favor Normal)
- Color indicates the feature's actual value for that sample

---

### 08. SHAP Waterfall Plot (DoS Example)
**File:** `08_shap_waterfall_dos.png`

**Description:** Detailed SHAP explanation for a specific DoS detection case, showing how each feature contributed to the final prediction.

**Key Information:**
- Shows a single high-confidence DoS detection
- Red bars: Features increasing DoS likelihood
- Blue bars: Features decreasing DoS likelihood
- Base value: Model's average prediction
- Final value: This sample's prediction score
- Each bar shows the feature name, value, and SHAP contribution

**Use Case:**
- Demonstrates explainability for security analysts
- Shows WHY a specific traffic sample was flagged as DoS
- Enables validation of model reasoning

---

### 09. SHAP Waterfall Plot (Normal Example)
**File:** `09_shap_waterfall_normal.png`

**Description:** Detailed SHAP explanation for a Normal traffic classification, showing how each feature contributed to the non-DoS prediction.

**Key Information:**
- Shows a high-confidence Normal traffic sample
- Mostly blue bars (decreasing DoS likelihood)
- Demonstrates model's reasoning for benign traffic
- Contrasts with DoS example to show difference in feature patterns

**Use Case:**
- Validates model correctly identifies normal traffic
- Shows different feature patterns for normal vs attack traffic
- Helps understand what makes traffic appear benign

---

## Mitigation Framework Images (Objective 4)

**Location:** `05_mitigation_framework/images/`

### 10. Attack Type Distribution (100 Sample Test)
**File:** `10_attack_type_distribution.png`

**Description:** Bar chart showing the distribution of detected DoS attack types from the initial 100-sample benchmark test.

**Key Information:**
- Attack types classified from DoS detections:
  - Protocol Exploit: 51%
  - Volumetric Flood: 46%
  - Amplification: 2%
  - Slowloris: 1%
- Based on SHAP feature analysis
- Classification rules use feature thresholds

---

## Complete Test Results Images

**Location:** `05_mitigation_framework/complete_test/`

These images represent the complete pipeline test on ALL 41,089 official benchmark samples using the optimized threshold (0.8517).

### 11. Confusion Matrix Heatmap (Complete Test)
**File:** `confusion_matrix_heatmap.png`

**Description:** Confusion matrix visualization for the complete benchmark test showing all 41,089 sample predictions with optimized threshold.

**Key Metrics:**
| Metric | Value |
|--------|-------|
| True Positives (DoS→DoS) | 3,535 |
| True Negatives (Normal→Normal) | 36,791 |
| False Positives (Normal→DoS) | 209 |
| False Negatives (DoS→Normal) | 554 |

**Analysis:**
- Excellent accuracy (98.14%) - matches original benchmark exactly
- High precision (94.42%) - very few false alarms (only 209)
- Good recall (86.45%) - detects most attacks (3,535 out of 4,089)
- Results achieved using optimized threshold (0.8517) and saved preprocessing (scaler + encoder)

---

### 12. Attack Type Distribution (Complete Test)
**File:** `attack_type_distribution.png`

**Description:** Pie chart showing the distribution of classified attack types from all 3,744 DoS predictions.

**Key Information:**
| Attack Type | Count | Percentage |
|-------------|-------|------------|
| Volumetric Flood | 3,043 | 81.3% |
| Protocol Exploit | 660 | 17.6% |
| Amplification | 36 | 1.0% |
| Slowloris | 5 | 0.1% |

**Interpretation:**
- Volumetric Flood dominates (high traffic volume attacks)
- Protocol Exploit is second (TCP-based attacks)
- Amplification and Slowloris are rare in this dataset

---

### 13. Severity Distribution (Complete Test)
**File:** `severity_distribution.png`

**Description:** Bar chart showing the severity levels assigned to all DoS predictions.

**Key Information:**
| Severity Level | Count | Percentage | Escalation Required |
|----------------|-------|------------|---------------------|
| CRITICAL | 3,743 | 100.0% | Yes |
| HIGH | 1 | 0.0% | Yes |
| MEDIUM | 0 | 0.0% | No |
| LOW | 0 | 0.0% | No |

**Analysis:**
- 100% of detections require escalation (all CRITICAL or HIGH)
- High severity due to optimized threshold (0.8517) - only high-confidence predictions pass
- Severity calculation includes:
  - Model confidence (>95% = CRITICAL base with optimized threshold)
  - Attack type modifier (Amplification +15%, Volumetric +10%)
  - Feature modifiers (high rate, high sload)

---

### 14. Performance Metrics (Complete Test)
**File:** `performance_metrics.png`

**Description:** Bar chart comparing the four main performance metrics from the complete benchmark test with optimized threshold.

**Key Metrics:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 98.14% | Overall correctness - EXCELLENT |
| Precision | 94.42% | Correct DoS out of all DoS predictions - EXCELLENT |
| Recall | 86.45% | DoS detected out of all actual DoS - GOOD |
| F1-Score | 90.26% | Harmonic mean of Precision and Recall - EXCELLENT |

**Important Note - Threshold Optimization:**
The excellent results were achieved using an **optimized threshold of 0.8517**:
- **High Precision (94.42%)**: Very few false alarms (only 209 false positives out of 37,000 normal samples)
- **Good Recall (86.45%)**: Most DoS attacks detected (3,535 out of 4,089)
- **Excellent F1-Score (90.26%)**: Balanced performance

The threshold optimization process:
1. Model trained on balanced data (50/50 split)
2. Default threshold (0.5) gives high recall but low precision on imbalanced data
3. Optimized threshold (0.8517) balances precision and recall for best F1 score
4. Critical: Must use saved scaler and encoder from training for correct results

---

## Summary Statistics

### Total Images Generated: 14

| Module | Count | Description |
|--------|-------|-------------|
| Model Training | 6 | Training/testing distributions, confusion matrices, feature importance |
| XAI Integration | 3 | SHAP summary plot, waterfall examples (DoS and Normal) |
| Mitigation Framework | 1 | Initial attack type distribution |
| Complete Test | 4 | Confusion matrix, attack types, severity, performance metrics |

### Image Specifications

- **Format:** PNG
- **Resolution:** 300 DPI
- **Background:** White
- **Style:** Publication-quality with clear labels and legends

---

## File Locations Summary

```
CTI_IDS/
├── 03_model_training/
│   └── proper_training/
│       └── images/
│           ├── 01_testing_set_distribution.png
│           ├── 02_training_set_distribution.png
│           ├── 03_model_performance_training.png
│           ├── 04_xgboost_confusion_matrix_training.png
│           ├── 05_xgboost_confusion_matrix_testing.png
│           └── 06_xgboost_feature_importance.png
│
├── 04_xai_integration/
│   └── images/
│       ├── 07_shap_summary_plot.png
│       ├── 08_shap_waterfall_dos.png
│       └── 09_shap_waterfall_normal.png
│
└── 05_mitigation_framework/
    ├── images/
    │   └── 10_attack_type_distribution.png
    └── complete_test/
        ├── confusion_matrix_heatmap.png
        ├── attack_type_distribution.png
        ├── severity_distribution.png
        └── performance_metrics.png
```

---

## Usage for Research Paper

All images are designed to be included directly in the research paper:

1. **Methodology Section:**
   - Images 01-02: Dataset distribution
   - Image 06: Feature selection justification

2. **Results Section:**
   - Images 03-05: Model performance
   - Images 07-09: XAI explanations
   - Images 11-14: Complete benchmark results

3. **Discussion Section:**
   - Image 10: Attack classification effectiveness
   - Images 12-13: Practical implications for SOC

---

*Generated: 2026-01-30*
*Project: From Detection to Defense: An XAI Powered DoS Prevention System*
