# Visualization Gallery for DoS Detection Research

## Overview

This directory contains 6 key visualizations for the research paper "From Detection to Defense: An XAI Powered DoS Prevention System with Implementable Mitigation Protocols".

---

## Data Sources

| Dataset | Source | Samples | Purpose |
|---------|--------|---------|---------|
| **Training Data** | UNSW-NB15 Training CSV | 24,528 (balanced) | Model training |
| **Testing Data** | UNSW-NB15 Testing CSV | 41,089 (imbalanced) | Benchmark evaluation |

**Important:** Training and Testing are from SEPARATE CSV files. The model never sees testing data during training.

---

## Image Inventory (6 Images)

| # | Filename | Data Source | Description |
|---|----------|-------------|-------------|
| 1 | `01_testing_set_distribution.png` | **Testing** | Testing/Benchmark dataset composition |
| 2 | `02_training_set_distribution.png` | **Training** | Training dataset composition |
| 3 | `03_model_performance_training.png` | **Training** | All 5 models - Cross-validation results |
| 4 | `04_xgboost_confusion_matrix_training.png` | **Training** | XGBoost predictions on training data |
| 5 | `05_xgboost_confusion_matrix_testing.png` | **Testing** | XGBoost predictions on benchmark data |
| 6 | `06_xgboost_feature_importance.png` | **Training** | Feature importance learned by XGBoost |

---

## Detailed Descriptions

### 1. Testing Set Distribution (`01_testing_set_distribution.png`)

- **Data:** UNSW-NB15 Testing CSV (External Benchmark)
- **Samples:** 41,089 total
  - Normal Traffic: 37,000 (90%)
  - DoS Attacks: 4,089 (10%)
- **Purpose:** Shows real-world imbalanced distribution for benchmarking

---

### 2. Training Set Distribution (`02_training_set_distribution.png`)

- **Data:** UNSW-NB15 Training CSV
- **Samples:** 24,528 total
  - Normal Traffic: 12,264 (50%)
  - DoS Attacks: 12,264 (50%)
- **Purpose:** Shows balanced training data used for model training

---

### 3. Model Performance - Training (`03_model_performance_training.png`)

- **Data:** Training data with 5-Fold Cross-Validation
- **Models:** XGBoost, Random Forest, MLP, SVM, Logistic Regression
- **Metrics:** Accuracy, Precision, Recall, F1 Score

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 96.45% | 96.89% | 95.95% | 96.45% |
| Random Forest | 96.22% | 96.75% | 95.63% | 96.22% |
| MLP | 94.32% | 95.38% | 93.02% | 94.32% |
| SVM | 92.26% | 93.45% | 90.88% | 92.26% |
| Logistic Reg. | 86.64% | 90.11% | 82.05% | 86.27% |

**Best Performer:** XGBoost (96.45% F1 Score)

---

### 4. XGBoost Confusion Matrix - Training (`04_xgboost_confusion_matrix_training.png`)

- **Data:** Training data (24,528 samples)
- **Model:** XGBoost

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 11,886 (TN) | 378 (FP) |
| **Actual DoS** | 497 (FN) | 11,767 (TP) |

- Accuracy: 96.43%
- Precision: 96.89%
- Recall: 95.95%

---

### 5. XGBoost Confusion Matrix - Testing (`05_xgboost_confusion_matrix_testing.png`)

- **Data:** Testing/Benchmark data (41,089 samples - UNSEEN)
- **Model:** XGBoost with optimized threshold (0.8517)

|  | Predicted Normal | Predicted DoS |
|--|------------------|---------------|
| **Actual Normal** | 36,791 (TN) | 209 (FP) |
| **Actual DoS** | 528 (FN) | 3,561 (TP) |

- Accuracy: 97.76%
- Precision: 94.41%
- Recall: 87.09%
- F1 Score: 90.57%

---

### 6. XGBoost Feature Importance (`06_xgboost_feature_importance.png`)

- **Data:** Learned from training data
- **Top Features:**

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | rate | 0.28 | Connection rate (packets/sec) |
| 2 | sload | 0.22 | Source bits per second |
| 3 | sbytes | 0.15 | Source to destination bytes |
| 4 | dload | 0.12 | Destination bits per second |
| 5 | proto | 0.08 | Protocol type |

---

## Research Paper Usage

| Section | Recommended Images |
|---------|-------------------|
| **Methodology** | 01, 02 (Data distributions) |
| **Training Results** | 03, 04 (Training performance) |
| **Benchmark Results** | 05 (Testing confusion matrix) |
| **Discussion** | 06 (Feature importance) |

---

*Last Updated: 2026-01-29*
