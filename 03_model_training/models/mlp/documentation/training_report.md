# MLP NEURAL NETWORK TRAINING REPORT
Generated: 2025-09-17 14:33:43

## NEURAL NETWORK CONFIGURATION
- Algorithm: Multi-Layer Perceptron (MLP)
- Architecture: (150, 75, 25)
- Total Parameters: 14901
- Activation Function: relu
- Solver: adam
- Training Time: 1.57 seconds
- Tuning Time: 26.63 seconds
- Convergence: 31 iterations

## PERFORMANCE METRICS
- Accuracy: 0.9248 (92.48%)
- Precision: 0.9627 (96.27%)
- Recall: 0.8839 (88.39%)
- F1-Score: 0.9216 (92.16%)
- ROC-AUC: 0.9735 (97.35%)
- CV F1-Score: 0.9192 (91.92%)

## NEURAL NETWORK INSIGHTS
- Hidden Layer Configuration: (150, 75, 25)
- Network Depth: 3 hidden layers
- Parameter Count: 14901 (learnable weights + biases)
- Early Stopping: Activated

## 5-MODEL COMPARISON COMPLETE
1. XGBoost: 95.54% (Tree-based leader)
2. Random Forest: 95.29% (Tree-based strong)
3. MLP: 92.48% (Neural network)
4. SVM: 90.04% (Kernel method)
5. Logistic Regression: 78.18% (Linear baseline)

## NEURAL NETWORK ANALYSIS
- Performance Position: 3rd place in 5-model comparison
- Neural vs Tree-based: Lower performance compared to XGBoost/Random Forest
- Neural vs Traditional: Superior to SVM and Logistic Regression
- Complexity Trade-off: Higher complexity than traditional ML with justified performance gain

## RESEARCH CONTRIBUTIONS
- Completes comprehensive ML paradigm comparison
- Demonstrates neural network performance on tabular cybersecurity data
- Establishes tree-based model superiority for DoS detection
- Provides neural network baseline for future research

## NEXT STEPS
- Layer 2: XAI/SHAP analysis for top performing models
- Production deployment with optimal model selection
- Research documentation for publication

---
MLP Neural Network Training Successfully Completed
5-Model Comparison Framework Complete
