# LOGISTIC REGRESSION TRAINING REPORT
Generated: 2025-09-17 12:38:42

## MODEL CONFIGURATION
- Algorithm: Logistic Regression
- Features: 10 network traffic features
- Feature Scaling: StandardScaler applied
- Training Time: 0.0096 seconds
- Hyperparameter Tuning Time: 1.74 seconds
- Cross-Validation: 3-fold
- Class Balance: Balanced (50-50)

## BEST HYPERPARAMETERS
- C (Regularization): 100.0
- Penalty: l2
- Solver: liblinear
- Max Iterations: 1000


## PERFORMANCE METRICS
- Accuracy: 0.7818 (78.18%)
- Precision: 0.7709 (77.09%)
- Recall: 0.8020 (80.20%)
- F1-Score: 0.7861 (78.61%)
- ROC-AUC: 0.8530 (85.30%)
- CV F1-Score: 0.7825 (78.25%)

## TOP 5 IMPORTANT FEATURES (by coefficient magnitude)
1. dload: 0.6910 (69.1%) ↘️
2. sbytes: 0.1146 (11.5%) ↗️
3. dmean: 0.0480 (4.8%) ↗️
4. rate: 0.0451 (4.5%) ↗️
5. tcprtt: 0.0337 (3.4%) ↘️

## CONFUSION MATRIX ANALYSIS
- True Negatives (Normal correctly classified): 623
- False Positives (Normal misclassified as DoS): 195
- False Negatives (DoS misclassified as Normal): 162
- True Positives (DoS correctly classified): 656

## LINEAR MODEL INSIGHTS
- Linear decision boundary learned from data
- Feature coefficients represent direct influence on DoS probability
- Positive coefficients increase DoS likelihood
- Negative coefficients decrease DoS likelihood
- Model highly interpretable compared to tree-based methods

## ANALYSIS SUMMARY
✅ Feature Scaling: COMPLETED (StandardScaler)
✅ Hyperparameter Tuning: COMPLETED
✅ Training: COMPLETED
✅ Evaluation: COMPLETED  
✅ Model Saved: COMPLETED
✅ Visualizations: COMPLETED
⏳ XAI Analysis: PENDING (Layer 2)

## NEXT STEPS
1. Train SVM model (final model)
2. Compare all 4 models comprehensively
3. Proceed to Layer 2: XAI/SHAP analysis

---
Logistic Regression Training Successfully Completed
