# Official UNSW-NB15 Datasets

## Dataset Organization

This folder contains the official UNSW-NB15 datasets from the [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset) source.

---

## Files (Renamed for Clarity)

### UNSW_NB15_TRAINING_175341.csv
- **Records**: 175,341
- **Original Name**: UNSW_NB15_training-set.csv
- **Purpose**: Model Training
- **Usage**: This is where our model's training data came from
- **Note**: Contains balanced DoS + Normal samples used to train XGBoost

### UNSW_NB15_TESTING_82332.csv
- **Records**: 82,332
- **Original Name**: UNSW_NB15_testing-set.csv
- **Purpose**: Benchmark Testing / Evaluation
- **Usage**: Used to evaluate model performance on unseen data
- **DoS Samples**: 4,089
- **Normal Samples**: 37,000
- **Total for Evaluation**: 41,089 (DoS + Normal only)

---

## Data Pipeline

```
TRAINING DATA (175,341 records)
UNSW_NB15_TRAINING_175341.csv
              │
              ▼
    Filter DoS + Normal (balanced)
              │
              ▼
    proper_training/ (30,660 records)
    - X_train_scaled.csv (24,528)
    - X_test_scaled.csv (6,132)
              │
              ▼
    XGBoost Model Trained
    - feature_scaler.pkl (saved)
    - proto_encoder.pkl (saved)
    - xgboost_model.json (saved)


BENCHMARK DATA (82,332 records)
UNSW_NB15_TESTING_82332.csv
              │
              ▼
    Filter DoS + Normal (41,089)
              │
              ▼
    Apply SAVED scaler & encoder
              │
              ▼
    Complete Pipeline Test
              │
              ▼
    Results: 98.14% Accuracy
```

---

## Complete Test Results (Optimized Threshold: 0.8517)

| Metric | Value |
|--------|-------|
| Accuracy | 98.14% |
| Precision | 94.42% |
| Recall | 86.45% |
| F1-Score | 90.26% |

| Confusion Matrix | Predicted Normal | Predicted DoS |
|------------------|------------------|---------------|
| Actual Normal | 36,791 (TN) | 209 (FP) |
| Actual DoS | 554 (FN) | 3,535 (TP) |

---

## Important Notes

1. **Use Saved Preprocessing**: Always use the saved `feature_scaler.pkl` and `proto_encoder.pkl` from training
2. **Threshold Optimization**: Default threshold (0.5) gives poor precision on imbalanced data; use 0.8517
3. **10 Features Used**: rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur

---

## Historical Note

The original Kaggle source had swapped file names (training labeled as testing and vice versa). We obtained the correct files from the official UNSW OneDrive source and renamed them for clarity:
- `UNSW_NB15_BENCHMARK_DATA_175341.csv` → `UNSW_NB15_TRAINING_175341.csv`
- `UNSW_NB15_MODEL_SOURCE_82332.csv` → `UNSW_NB15_TESTING_82332.csv`

---

*Last Updated: 2026-01-30*
