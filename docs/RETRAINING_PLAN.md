# Model Retraining Plan - Proper Methodology

**Created**: 2026-01-28
**Status**: COMPLETED

---

## Overview

Retraining all models using the correct UNSW-NB15 train/test split:
- **Training**: Official Training Set (175,341 records)
- **Testing**: Official Testing Set (82,332 records)

---

## Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Training Sample Size | ALL DoS (12,264) + Equal Normal | More data = better model |
| Normal Sampling Strategy | Stratified by Protocol | Preserves traffic diversity |
| Features | Same 10 features | Consistency |
| Balancing | 50/50 (DoS:Normal) | Binary classification standard |

---

## Phase Tracking

### Phase 1: Setup & Archive
| Step | Description | Status |
|------|-------------|--------|
| 1.1 | Archive old directories | [x] DONE |
| 1.2 | Create new directory structure | [x] DONE |
| 1.3 | Verify official datasets in place | [x] DONE |

### Phase 2: Data Analysis & Visualization
| Step | Description | Status |
|------|-------------|--------|
| 2.1 | Analyze training set distribution | [x] DONE |
| 2.2 | Analyze testing set distribution | [x] DONE |
| 2.3 | Generate distribution images (separate PNGs) | [x] DONE |

### Phase 3: Data Preparation
| Step | Description | Status |
|------|-------------|--------|
| 3.1 | Extract ALL DoS from training set (12,264) | [x] DONE |
| 3.2 | Stratified sample Normal traffic (12,264) | [x] DONE |
| 3.3 | Validate sampling (compare distributions) | [x] DONE |
| 3.4 | Create balanced training dataset (24,528) | [x] DONE |
| 3.5 | Generate sampling validation images | [x] DONE |

### Phase 4: Feature Engineering
| Step | Description | Status |
|------|-------------|--------|
| 4.1 | Extract 10 features | [x] DONE |
| 4.2 | Encode categorical (proto) | [x] DONE |
| 4.3 | Scale features (StandardScaler) | [x] DONE |
| 4.4 | Save scaler for benchmark use | [x] DONE |
| 4.5 | Save processed training data | [x] DONE |

### Phase 5: Model Training
| Step | Description | Status |
|------|-------------|--------|
| 5.1 | Train XGBoost | [x] DONE |
| 5.2 | Train Random Forest | [x] DONE |
| 5.3 | Train SVM | [x] DONE |
| 5.4 | Train MLP | [x] DONE |
| 5.5 | Train Logistic Regression | [x] DONE |
| 5.6 | Generate model performance images | [x] DONE |
| 5.7 | Save all models (.pkl) | [x] DONE |

### Phase 6: External Benchmarking
| Step | Description | Status |
|------|-------------|--------|
| 6.1 | Load testing set (82,332 records) | [x] DONE |
| 6.2 | Extract DoS + Normal for binary classification | [x] DONE |
| 6.3 | Apply same preprocessing (10 features, scaling) | [x] DONE |
| 6.4 | Test all 5 models | [x] DONE |
| 6.5 | Generate benchmark images (separate PNGs) | [x] DONE |
| 6.6 | Save benchmark results (JSON) | [x] DONE |

### Phase 7: Documentation
| Step | Description | Status |
|------|-------------|--------|
| 7.1 | Create methodology documentation | [x] DONE |
| 7.2 | Generate final comparison images | [x] DONE |
| 7.3 | Update tracking plan | [x] DONE |

---

## Final Results

### Training Performance (5-Fold CV on 24,528 samples)

| Model | CV F1 Score | Train F1 |
|-------|-------------|----------|
| XGBoost | 0.9645 | 0.9745 |
| RandomForest | 0.9622 | 0.9708 |
| MLP | 0.9432 | 0.9467 |
| SVM | 0.9226 | 0.9262 |
| LogisticRegression | 0.8627 | 0.8623 |

### External Benchmark (41,089 unseen samples)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.9481 | 0.6678 | **0.9528** | 0.7852 |
| RandomForest | 0.9344 | 0.6101 | 0.9435 | 0.7410 |
| MLP | 0.9063 | 0.5164 | 0.9208 | 0.6617 |
| SVM | 0.8572 | 0.4011 | 0.8824 | 0.5515 |
| LogisticRegression | 0.8269 | 0.3368 | 0.7625 | 0.4672 |

**Key Finding**: XGBoost achieves **95.28% recall** on external benchmark - catches 95% of DoS attacks!

---

## Data Summary

### Training Data (Created)
- **Source**: UNSW_NB15_BENCHMARK_DATA_175341.csv (Official Training Set)
- **DoS Samples**: 12,264 (ALL available)
- **Normal Samples**: 12,264 (Stratified by protocol)
- **Total**: 24,528 balanced samples
- **Features**: 10 (rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur)

### Benchmark Data (External Testing)
- **Source**: UNSW_NB15_MODEL_SOURCE_82332.csv (Official Testing Set)
- **DoS Samples**: 4,089
- **Normal Samples**: 37,000
- **Total for Testing**: 41,089

---

## Sampling Methodology

### Normal Traffic Stratified Sampling

```
Total Normal in Training Set: 56,000
Target Normal Samples: 12,264

Actual Protocol Distribution Sampled:
  TCP: 8,568 (69.86%)
  UDP: 3,049 (24.86%)
  ARP:   626 (5.10%)
  OSPF:   14 (0.11%)
  IGMP:    4 (0.03%)
  ICMP:    3 (0.02%)

Total: 12,264 Normal samples
```

**Rationale**: Stratified sampling by protocol ensures:
1. Traffic diversity is preserved
2. Model learns from TCP, UDP, and ARP patterns
3. Methodology is defensible in research paper
4. Reproducible with random_state=42

---

## Directory Structure

```
CTI_IDS/
├── _ARCHIVE/                          ← Old work (reference only)
├── 01_data_preparation/
│   └── data/
│       └── official_datasets/         ← Source data
├── 03_model_training/
│   └── proper_training/               ← NEW training (completed)
│       ├── data/                      ← Processed training data
│       │   ├── balanced_dos_detection_dataset.csv
│       │   ├── X_train_scaled.csv
│       │   ├── y_train.csv
│       │   ├── feature_scaler.pkl
│       │   └── proto_encoder.pkl
│       ├── models/                    ← Trained models
│       │   ├── xgboost_model.pkl
│       │   ├── randomforest_model.pkl
│       │   ├── svm_model.pkl
│       │   ├── mlp_model.pkl
│       │   └── logisticregression_model.pkl
│       ├── images/                    ← All visualizations (12 PNGs)
│       └── results/                   ← JSON results
│           ├── training_results.json
│           └── benchmark_results.json
├── RETRAINING_PLAN.md                 ← This file
└── XAI_MITIGATION_FRAMEWORK_PROPOSAL.md
```

---

## Generated Images

| # | Filename | Description |
|---|----------|-------------|
| 01 | 01_training_attack_distribution.png | Training set attack category distribution |
| 02 | 02_testing_attack_distribution.png | Testing set attack category distribution |
| 03 | 03_training_normal_protocol_dist.png | Normal traffic protocol distribution |
| 04 | 04_dos_vs_normal_comparison.png | DoS vs Normal comparison |
| 05 | 05_stratified_sampling_validation.png | Stratified sampling validation |
| 06 | 06_balanced_dataset_distribution.png | Balanced training dataset |
| 07 | 07_training_performance_comparison.png | Model training performance |
| 08 | 08_training_metrics_all.png | All training metrics |
| 09 | 09_benchmark_performance_comparison.png | External benchmark performance |
| 10 | 10_training_vs_benchmark_f1.png | Training vs benchmark F1 |
| 11 | 11_confusion_matrices.png | Confusion matrices |
| 12 | 12_dos_detection_rate.png | DoS detection rate (recall) |

---

## Notes

- Using `random_state=42` throughout for reproducibility
- All preprocessing identical for training and benchmarking
- Scaler fitted on training data, applied to benchmark data
- Lower precision on benchmark due to class imbalance (4,089 DoS vs 37,000 Normal)
- High recall (95.28%) indicates excellent DoS attack detection capability

---

*Completed: 2026-01-28*
