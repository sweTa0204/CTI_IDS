# 1D-CNN Model Training Report for DoS Detection

## Executive Summary

This document provides analysis of the 1D-CNN (1D Convolutional Neural Network) model
for DoS attack detection, with comparison to XGBoost and LSTM.

**Key Finding:** 1D-CNN achieved 86.38% F1 Score,
positioning it between LSTM and XGBoost in performance.

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Conv Filters | 64, 128 |
| Kernel Size | 3 |
| Dropout Rate | 0.3 |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Batch Size | 64 |

## Why 1D-CNN?

1D Convolutional Neural Networks are designed to:
- Detect **local patterns** in sequential data
- Find **feature signatures** that indicate attacks
- Process data **faster** than recurrent networks (LSTM)
- Capture **spatial relationships** between adjacent features

## Results Summary

### Default Threshold (0.5)

| Metric | Value |
|--------|-------|
| Accuracy | 93.98% |
| Precision | 63.99% |
| Recall | 90.36% |
| F1 Score | 74.93% |
| AUC | 0.9780 |

### Optimized Threshold (0.8700)

| Metric | Value |
|--------|-------|
| Accuracy | 97.42% |
| Precision | 90.92% |
| Recall | 82.27% |
| F1 Score | 86.38% |
| AUC | 0.9780 |

## Three-Model Comparison

| Metric | XGBoost | LSTM | 1D-CNN | Best |
|--------|---------|------|--------|------|
| Accuracy | 97.76% | 96.89% | 97.42% | XGBoost |
| Precision | 94.41% | 88.12% | 90.92% | XGBoost |
| Recall | 87.09% | 79.48% | 82.27% | XGBoost |
| F1 Score | 90.57% | 83.58% | 86.38% | XGBoost |
| AUC | 0.9915 | 0.9683 | 0.9780 | XGBoost |

## Confusion Matrix (Optimized)

```
                ACTUAL
            Normal    DoS
          +--------+--------+
Predicted | 36,664 |    336 |  Normal
          +--------+--------+
Predicted |    725 |  3,364 |  DoS
          +--------+--------+

TN = 36,664 (Normal correctly identified)
FP = 725 (False alarms)
FN = 336 (Missed attacks)
TP = 3,364 (Attacks detected)
```

## Speed Comparison

| Model | Training Time | Prediction Speed |
|-------|--------------|------------------|
| XGBoost | ~2-3 seconds | ~0.1 ms/sample |
| LSTM | ~2-3 minutes | ~1-2 ms/sample |
| 1D-CNN | ~1-2 minutes | ~0.5-1 ms/sample |

**1D-CNN is faster than LSTM** because:
- No recurrent connections (no sequential dependency)
- Parallel convolution operations
- Simpler backpropagation

## Conclusion

### Model Rankings (for this dataset)

1. **XGBoost** - Best overall (90.57% F1)
2. **1D-CNN** - Second best (86.38% F1)
3. **LSTM** - Third (83.58% F1)

### Why XGBoost Still Wins

- UNSW-NB15 provides **tabular flow features**, not raw sequences
- XGBoost is **optimized for tabular data**
- Neural networks (CNN, LSTM) need **sequential/image data** to show advantage

### Value of This Analysis

By implementing XGBoost, LSTM, and 1D-CNN, we demonstrate:
1. Comprehensive exploration of different model architectures
2. Understanding of when each model type excels
3. Rigorous comparative analysis with proper threshold optimization

---

*Generated: 2026-02-03 18:07:41*
