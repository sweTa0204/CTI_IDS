# XAI-Powered DoS Detection and Mitigation System
## Complete Project Documentation

**Project Lead:** Akash Madanu
**Last Updated:** 2026-01-30
**Status:** COMPLETE

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Information](#2-dataset-information)
3. [Data Split Details](#3-data-split-details)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Training](#5-model-training)
6. [Benchmark Results](#6-benchmark-results)
7. [XAI Integration (SHAP)](#7-xai-integration-shap)
8. [Mitigation Framework](#8-mitigation-framework)
9. [Complete Pipeline](#9-complete-pipeline)
10. [Directory Structure](#10-directory-structure)
11. [How to Run](#11-how-to-run)
12. [Key Files Reference](#12-key-files-reference)

---

# 1. Project Overview

## What is This Project?

This project develops an **Explainable AI (XAI) powered Intrusion Detection System (IDS)** that:

1. **Detects DoS attacks** using machine learning (XGBoost)
2. **Explains WHY** an attack was detected (using SHAP)
3. **Classifies the attack type** (Volumetric, Protocol Exploit, Slowloris, Amplification)
4. **Calculates severity level** (CRITICAL, HIGH, MEDIUM, LOW)
5. **Generates mitigation commands** (iptables, tc)

## Research Objectives

| Objective | Description | Status |
|-----------|-------------|--------|
| **Objective 1** | Dataset Preparation & Feature Engineering | COMPLETED |
| **Objective 2** | ML Model Training (5 models) | COMPLETED |
| **Objective 3** | XAI Integration (SHAP) | COMPLETED |
| **Objective 4** | Mitigation Framework | COMPLETED |

## Key Achievements

- **98.14% Accuracy** on external benchmark data
- **90.26% F1 Score** with optimized threshold
- **94.42% Precision** with only 209 false alarms out of 37,000 normal traffic
- **86.45% Recall** - catches most DoS attacks
- **Complete Pipeline** from detection to mitigation

---

# 2. Dataset Information

## UNSW-NB15 Dataset

The project uses the **UNSW-NB15** dataset, a widely-used network intrusion detection benchmark.

### Official Source

- **Provider:** University of New South Wales (UNSW)
- **Link:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Total Records:** 257,673 (175,341 training + 82,332 testing)

### Our Dataset Files

Located in: `01_data_preparation/data/official_datasets/`

| File Name | Records | Purpose |
|-----------|---------|---------|
| `UNSW_NB15_TRAINING_175341.csv` | 175,341 | Model Training |
| `UNSW_NB15_TESTING_82332.csv` | 82,332 | Benchmark Testing |

### Attack Categories in UNSW-NB15

The dataset contains 10 attack categories:

| Category | Description |
|----------|-------------|
| **DoS** | Denial of Service attacks (our focus) |
| Normal | Legitimate traffic |
| Exploits | Exploiting vulnerabilities |
| Fuzzers | Fuzzing attacks |
| Generic | Generic attacks |
| Reconnaissance | Network scanning |
| Shellcode | Shellcode injection |
| Worms | Worm propagation |
| Backdoor | Backdoor access |
| Analysis | Protocol analysis attacks |

**Our model focuses on: DoS vs Normal (Binary Classification)**

---

# 3. Data Split Details

## Training Data (UNSW_NB15_TRAINING_175341.csv)

| Category | Records | Percentage |
|----------|---------|------------|
| **DoS** | **12,264** | 7.0% |
| Normal | 56,000 | 31.9% |
| Other Attacks | 107,077 | 61.1% |
| **TOTAL** | **175,341** | 100% |

### What We Used for Training

We used a **balanced subset** for training:

```
Training Set (Balanced):
┌─────────────────────────────────────┐
│  DoS Attacks       │  12,264 (50%)  │
│  Normal Traffic    │  12,264 (50%)  │
├─────────────────────────────────────┤
│  TOTAL             │  24,528        │
└─────────────────────────────────────┘
```

**Why balanced?** Equal representation of both classes helps the model learn to distinguish DoS from Normal without bias.

## Testing Data (UNSW_NB15_TESTING_82332.csv)

| Category | Records | Percentage |
|----------|---------|------------|
| **DoS** | **4,089** | 5.0% |
| Normal | 37,000 | 44.9% |
| Other Attacks | 41,243 | 50.1% |
| **TOTAL** | **82,332** | 100% |

### What We Used for Benchmark Testing

We used **DoS + Normal** records only:

```
Benchmark Test Set (Real-World Imbalanced):
┌─────────────────────────────────────┐
│  DoS Attacks       │   4,089 (10%)  │
│  Normal Traffic    │  37,000 (90%)  │
├─────────────────────────────────────┤
│  TOTAL             │  41,089        │
└─────────────────────────────────────┘
```

**Why 41,089 (not 82,332)?** Our model is a DoS detector - it only classifies DoS vs Normal. The other 41,243 records are different attack types (Exploits, Fuzzers, etc.) which our model doesn't handle.

## Data Flow Summary

```
TRAINING FILE (175,341 records)
        │
        ├── DoS: 12,264
        ├── Normal: 56,000 → Take 12,264 (balanced)
        └── Others: 107,077 (not used)
        │
        ▼
   TRAINING SET: 24,528 (balanced 50/50)
        │
        ▼
   Train XGBoost Model


TESTING FILE (82,332 records)
        │
        ├── DoS: 4,089
        ├── Normal: 37,000
        └── Others: 41,243 (not used)
        │
        ▼
   BENCHMARK SET: 41,089 (imbalanced 10/90)
        │
        ▼
   Evaluate Model Performance
```

---

# 4. Feature Engineering

## 10 Selected Features

From the original 42 features, we selected 10 most discriminative features:

| # | Feature | Full Name | Description |
|---|---------|-----------|-------------|
| 1 | `rate` | Packets per second | Network traffic rate |
| 2 | `sload` | Source bits per second | Source load |
| 3 | `sbytes` | Source to destination bytes | Total bytes sent |
| 4 | `dload` | Destination bits per second | Destination load |
| 5 | `proto` | Protocol | Network protocol (encoded) |
| 6 | `dtcpb` | Destination TCP base sequence | TCP sequence number |
| 7 | `stcpb` | Source TCP base sequence | TCP sequence number |
| 8 | `dmean` | Destination packet mean size | Average packet size |
| 9 | `tcprtt` | TCP round-trip time | Connection latency |
| 10 | `dur` | Duration | Connection duration |

## Why These Features?

These features were selected based on:
- **Correlation analysis** with DoS attacks
- **Variance analysis** for discrimination ability
- **Feature importance** from preliminary models
- **Domain knowledge** of DoS attack characteristics

## Feature Preprocessing

1. **Protocol Encoding:** `proto` (categorical) → LabelEncoder → numeric
2. **Scaling:** StandardScaler (mean=0, std=1)
3. **Missing Values:** Filled with median

### Important: Saved Preprocessors

Located in: `03_model_training/proper_training/data/`

| File | Purpose |
|------|---------|
| `feature_scaler.pkl` | StandardScaler fitted on training data |
| `proto_encoder.pkl` | LabelEncoder for protocol column |

**CRITICAL:** For testing, always use `transform()` NOT `fit_transform()` with saved preprocessors!

---

# 5. Model Training

## Models Trained

We trained 5 machine learning models:

| # | Model | Type |
|---|-------|------|
| 1 | XGBoost | Gradient Boosting |
| 2 | Random Forest | Ensemble |
| 3 | SVM | Support Vector Machine |
| 4 | MLP | Neural Network |
| 5 | Logistic Regression | Linear |

## Training Configuration

```python
# Common Configuration
RANDOM_STATE = 42  # For reproducibility
CROSS_VALIDATION = 5-Fold Stratified

# XGBoost Parameters
n_estimators = 100
max_depth = 6
learning_rate = 0.1
```

## Cross-Validation Results (Training)

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 Score |
|-------|-------------|--------------|-----------|-------------|
| **XGBoost** | 96.45% | 96.89% | 95.95% | 96.45% |
| Random Forest | 96.22% | 96.75% | 95.63% | 96.22% |
| MLP | 94.32% | 95.38% | 93.02% | 94.32% |
| SVM | 92.26% | 93.45% | 90.88% | 92.26% |
| Logistic Regression | 86.64% | 90.11% | 82.05% | 86.27% |

## Model Files

Located in: `03_model_training/proper_training/models/<model_name>/`

| Model | File |
|-------|------|
| XGBoost | `xgboost_model.pkl` + `xgboost_model.json` |
| Random Forest | `randomforest_model.pkl` |
| SVM | `svm_model.pkl` |
| MLP | `mlp_model.pkl` |
| Logistic Regression | `logisticregression_model.pkl` |

---

# 6. Benchmark Results

## Default Threshold (0.5)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | 94.81% | 66.78% | 95.28% | 78.52% |
| Random Forest | 93.44% | 61.01% | 94.35% | 74.10% |
| MLP | 90.63% | 51.64% | 92.08% | 66.17% |
| SVM | 85.72% | 40.11% | 88.24% | 55.15% |
| Logistic Regression | 82.69% | 33.68% | 76.25% | 46.72% |

**Note:** Low precision is due to class imbalance (90% Normal, 10% DoS).

## Optimized Threshold Results

### What is Threshold Optimization?

```
Default (threshold = 0.5):
  If P(DoS) >= 0.5 → Predict DoS

Optimized (threshold = 0.8517):
  If P(DoS) >= 0.8517 → Predict DoS
```

Higher threshold = fewer false alarms, but may miss some attacks.

### Optimized Results

| Model | Accuracy | Precision | Recall | F1 Score | Threshold |
|-------|----------|-----------|--------|----------|-----------|
| **XGBoost** | **98.14%** | **94.42%** | **86.45%** | **90.26%** | **0.8517** |
| Random Forest | 97.93% | 89.86% | 89.26% | 89.56% | 0.6865 |
| MLP | 97.14% | 88.43% | 82.02% | 85.11% | 0.8448 |
| SVM | 95.86% | 82.47% | 74.10% | 78.06% | 0.9300 |
| Logistic Regression | 88.42% | 44.48% | 66.06% | 53.16% | 0.7468 |

## XGBoost Confusion Matrix (Optimized)

```
                    ACTUAL
                Normal      DoS
              ┌──────────┬──────────┐
    Predicted │  36,791  │    554   │
    Normal    │   (TN)   │   (FN)   │  Missed 554 attacks (13.5%)
              ├──────────┼──────────┤
    Predicted │    209   │   3,535  │
    DoS       │   (FP)   │   (TP)   │  Only 209 false alarms (0.56%)
              └──────────┴──────────┘

Key Metrics:
  ✓ True Negative Rate: 99.44% (correctly identified normal)
  ✓ False Positive Rate: 0.56% (false alarms)
  ✓ True Positive Rate: 86.45% (attacks detected)
  ✓ False Negative Rate: 13.55% (attacks missed)
```

## Why XGBoost is Best

| Criteria | XGBoost Performance |
|----------|---------------------|
| Highest Accuracy | 98.14% |
| Highest Precision | 94.42% |
| Highest F1 Score | 90.26% |
| Lowest False Alarms | 209 (0.56%) |
| AUC (ROC) | 0.9915 (Excellent) |

**SELECTED MODEL: XGBoost with Threshold 0.8517**

---

# 7. XAI Integration (SHAP)

## What is SHAP?

**SHAP = SHapley Additive exPlanations**

SHAP explains WHY the model made a prediction by calculating how much each feature contributed.

## Example Explanation

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

## Why SHAP TreeExplainer?

| Reason | Explanation |
|--------|-------------|
| **Optimized for XGBoost** | Specifically designed for tree-based models |
| **Fast** | Computes in seconds |
| **Exact** | Mathematically exact values, not approximations |
| **Consistent** | Same input always produces same output |

## Why NOT LIME?

| SHAP TreeExplainer | LIME |
|-------------------|------|
| Fast (seconds) | Slow (minutes) |
| Exact for trees | Approximation |
| Consistent | Can vary between runs |
| Simple to implement | More complex |

**Decision: SHAP only (sufficient for academic rigor)**

## SHAP Output Format

```json
{
    "record_id": 20459,
    "prediction": "DoS",
    "confidence": 0.9996,
    "shap_values": {
        "rate": 0.1234,
        "sload": 2.4836,
        "sbytes": 0.7366,
        "proto": 4.0827,
        ...
    },
    "top_features": ["proto", "sload", "sbytes"]
}
```

## Files

Located in: `04_xai_integration/`

| File | Purpose |
|------|---------|
| `shap_explainer.py` | Main SHAP explainer class |
| `test_shap.py` | Test script |

---

# 8. Mitigation Framework

## What is the Mitigation Framework?

The Mitigation Framework converts DoS detections into actionable security responses:

```
Detection → Classification → Severity → Mitigation Commands
```

## Attack Classification

Based on SHAP feature contributions, attacks are classified into 4 types:

| Attack Type | Key Features | Description |
|-------------|--------------|-------------|
| **Volumetric Flood** | rate, sbytes, sload | High volume traffic |
| **Protocol Exploit** | proto, tcprtt, stcpb, dtcpb | Protocol manipulation |
| **Slowloris** | dur, dmean | Slow, prolonged connections |
| **Amplification** | dload, dbytes | Response larger than request |

## Severity Levels

| Level | Confidence Threshold | Action |
|-------|---------------------|--------|
| **CRITICAL** | >= 95% | Immediate block |
| **HIGH** | 90% - 95% | Priority response |
| **MEDIUM** | 75% - 90% | Monitor closely |
| **LOW** | 60% - 75% | Log and observe |

## Mitigation Commands Generated

### Rate Limiting (tc)
```bash
tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 400ms
```

### Firewall Rules (iptables)
```bash
iptables -A INPUT -p tcp --dport 80 -m limit --limit 100/sec -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j DROP
```

### SYN Flood Protection
```bash
iptables -A INPUT -p tcp --syn -m limit --limit 50/sec -j ACCEPT
sysctl -w net.ipv4.tcp_syncookies=1
```

## Files

Located in: `05_mitigation_framework/`

| File | Purpose |
|------|---------|
| `attack_classifier.py` | Classifies attack type from SHAP |
| `severity_calculator.py` | Calculates severity level |
| `mitigation_generator.py` | Generates mitigation commands |
| `alert_generator.py` | Creates security alerts |
| `main.py` | Main framework class |

---

# 9. Complete Pipeline

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPLETE PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: Network Traffic Record (10 features)
        │
        ▼
Step 2: Feature Scaling (using saved scaler)
        │
        ▼
Step 3: XGBoost Prediction (threshold = 0.8517)
        │
        ├── If P(DoS) < 0.8517 → Normal Traffic → End
        │
        └── If P(DoS) >= 0.8517 → DoS Attack Detected
                │
                ▼
Step 4: SHAP Explanation (feature contributions)
        │
        ▼
Step 5: Attack Classification
        │ Based on top contributing features:
        │ - rate, sbytes, sload → Volumetric Flood
        │ - proto, tcprtt → Protocol Exploit
        │ - dur, dmean → Slowloris
        │ - dload → Amplification
        │
        ▼
Step 6: Severity Calculation
        │ Based on confidence:
        │ - >= 95% → CRITICAL
        │ - 90-95% → HIGH
        │ - 75-90% → MEDIUM
        │ - 60-75% → LOW
        │
        ▼
Step 7: Mitigation Generation
        │ Generate:
        │ - iptables commands
        │ - tc rate limiting
        │ - System hardening
        │
        ▼
Step 8: Alert Output
        {
          "attack_type": "Volumetric Flood",
          "severity": "CRITICAL",
          "mitigation": [...],
          "explanation": "High rate + sload"
        }
```

## Complete Test Results

Running all 41,089 samples through the complete pipeline:

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.14% |
| **Precision** | 94.42% |
| **Recall** | 86.45% |
| **F1 Score** | 90.26% |
| **Threshold** | 0.8517 |

### Confusion Matrix

| | Predicted Normal | Predicted DoS |
|---|---|---|
| **Actual Normal** | 36,791 (TN) | 209 (FP) |
| **Actual DoS** | 554 (FN) | 3,535 (TP) |

### Attack Type Distribution (Detected)

| Attack Type | Count | Percentage |
|-------------|-------|------------|
| Volumetric Flood | 3,043 | 81.3% |
| Protocol Exploit | 660 | 17.6% |
| Amplification | 36 | 1.0% |
| Slowloris | 5 | 0.1% |

### Severity Distribution

| Severity | Count | Percentage |
|----------|-------|------------|
| CRITICAL | 3,743 | 99.97% |
| HIGH | 1 | 0.03% |

---

# 10. Directory Structure

```
CTI_IDS/
│
├── 01_data_preparation/
│   └── data/
│       └── official_datasets/
│           ├── UNSW_NB15_TRAINING_175341.csv  (175,341 records)
│           ├── UNSW_NB15_TESTING_82332.csv    (82,332 records)
│           └── README.md
│
├── 03_model_training/
│   └── proper_training/
│       ├── data/
│       │   ├── X_train_scaled.csv     (24,528 training samples)
│       │   ├── y_train.csv            (training labels)
│       │   ├── X_test_scaled.csv      (41,089 benchmark samples)
│       │   ├── y_test.csv             (benchmark labels)
│       │   ├── feature_scaler.pkl     (saved StandardScaler)
│       │   └── proto_encoder.pkl      (saved LabelEncoder)
│       │
│       ├── models/
│       │   ├── xgboost/
│       │   │   └── xgboost_model.pkl
│       │   ├── randomforest/
│       │   ├── svm/
│       │   ├── mlp/
│       │   └── logisticregression/
│       │
│       ├── results/
│       │   ├── training_results.json
│       │   ├── benchmark_results.json
│       │   └── benchmark_results_optimized.json
│       │
│       ├── images/
│       │   └── (visualization images)
│       │
│       └── RESULT_DISCUSSION.md
│
├── 04_xai_integration/
│   ├── shap_explainer.py
│   ├── test_shap.py
│   ├── images/
│   └── README.md
│
├── 05_mitigation_framework/
│   ├── attack_classifier.py
│   ├── severity_calculator.py
│   ├── mitigation_generator.py
│   ├── alert_generator.py
│   ├── main.py
│   ├── prepare_test_data.py
│   ├── mappings/
│   └── images/
│
├── 06_complete_testing/           (Complete Pipeline Testing)
│   ├── run_complete_test.py       (main test script)
│   ├── generate_visualizations.py
│   ├── confusion_matrix.json      (performance metrics)
│   ├── summary_report.json        (full test summary)
│   ├── attack_distribution.json   (attack type breakdown)
│   ├── complete_results.json      (all 41,089 results)
│   ├── confusion_matrix_heatmap.png
│   ├── attack_type_distribution.png
│   ├── severity_distribution.png
│   └── performance_metrics.png
│
├── _ARCHIVE/                  (old/deprecated files)
│
├── PROJECT_DOCUMENTATION.md   (THIS FILE)
├── README.md
├── IMPLEMENTATION_PLAN.md
├── OBJECTIVE_3_4_DOCUMENTATION.md
└── IMAGE_DOCUMENTATION.md
```

---

# 11. How to Run

## Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

## Run Complete Pipeline Test

```bash
cd 06_complete_testing
python run_complete_test.py
```

This will:
1. Load the XGBoost model
2. Process all 41,089 benchmark samples
3. Apply SHAP explanations to DoS detections
4. Classify attack types
5. Calculate severity levels
6. Generate mitigation recommendations
7. Output results to JSON files

## Run SHAP Test

```bash
cd 04_xai_integration
python test_shap.py
```

## Use in Your Code

```python
from mitigation_framework.main import MitigationFramework

# Initialize
framework = MitigationFramework()

# Process a single record
features = [1200, 850000, 5000000, 50000, 6, 12345, 67890, 500, 0.01, 2]
result = framework.process_single(features)

print(result)
# {
#   "prediction": "DoS",
#   "confidence": 0.95,
#   "attack_type": "Volumetric Flood",
#   "severity": "CRITICAL",
#   "mitigation": [...]
# }
```

---

# 12. Key Files Reference

## Critical Files

| File | Path | Purpose |
|------|------|---------|
| XGBoost Model | `03_model_training/proper_training/models/xgboost/xgboost_model.pkl` | Trained model |
| Feature Scaler | `03_model_training/proper_training/data/feature_scaler.pkl` | Preprocessing |
| Proto Encoder | `03_model_training/proper_training/data/proto_encoder.pkl` | Protocol encoding |
| SHAP Explainer | `04_xai_integration/shap_explainer.py` | XAI class |
| Complete Test | `06_complete_testing/run_complete_test.py` | Full pipeline |

## Result Files

| File | Path | Content |
|------|------|---------|
| Training Results | `03_model_training/proper_training/results/training_results.json` | CV metrics |
| Benchmark Results | `03_model_training/proper_training/results/benchmark_results_optimized.json` | Test metrics |
| Complete Test Summary | `06_complete_testing/summary_report.json` | Pipeline results |

## Configuration Values

| Parameter | Value | Location |
|-----------|-------|----------|
| Optimized Threshold | 0.8517 | Used in prediction |
| Random State | 42 | For reproducibility |
| Features | 10 | rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur |

---

# Summary

## What We Built

1. **DoS Detection Model (XGBoost)**
   - Trained on 24,528 balanced samples
   - Tested on 41,089 imbalanced samples
   - 98.14% accuracy, 90.26% F1 score

2. **Explainable AI (SHAP)**
   - Explains why each detection was made
   - Identifies top contributing features

3. **Attack Classification**
   - 4 attack types: Volumetric, Protocol, Slowloris, Amplification

4. **Severity Assessment**
   - 4 levels: CRITICAL, HIGH, MEDIUM, LOW

5. **Mitigation Generation**
   - iptables, tc commands
   - Actionable security responses

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Training samples | 24,528 (balanced 50/50) |
| Benchmark samples | 41,089 (imbalanced 10/90) |
| DoS in training | 12,264 |
| DoS in benchmark | 4,089 |
| Accuracy | 98.14% |
| F1 Score | 90.26% |
| Optimal Threshold | 0.8517 |
| Features | 10 |

---

*This document provides a complete reference for the XAI-Powered DoS Detection and Mitigation System project.*

*Created: 2026-01-30*
