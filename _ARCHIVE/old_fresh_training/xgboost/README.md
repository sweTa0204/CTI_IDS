# XGBoost Model - DoS Attack Detection

## Overview

XGBoost (Extreme Gradient Boosting) is a powerful ensemble learning algorithm that builds multiple decision trees sequentially, where each tree corrects the errors of the previous ones. In this project, XGBoost is used for binary classification to distinguish between **DoS (Denial of Service) attacks** and **Normal network traffic**.

This model achieved the **highest performance** among all five models tested, with an accuracy of **95.78%** and an F1-Score of **95.75%**. The model was trained on 6,542 samples and tested on 1,636 samples from the UNSW-NB15 dataset.

**Key Characteristics:**
- Algorithm Type: Gradient Boosting (Ensemble of Decision Trees)
- Classification Task: Binary (DoS Attack vs Normal Traffic)
- Best Performing Model in this research
- Training Time: 0.28 seconds

---

## Files

### xgboost_model.pkl
This is the trained XGBoost model saved in Python's pickle format. The file contains the complete model object including all 100 decision trees, their learned parameters, split points, and feature thresholds. This file can be loaded directly to make predictions on new network traffic data without retraining. The model size is approximately 386 KB.

### training_results.json
This JSON file contains all the training metadata and performance metrics. It includes:
- **model_name**: Identifier for the model ("XGBoost")
- **training_date**: When the model was trained
- **best_parameters**: The hyperparameters used for training
  - colsample_bytree: 0.8 (80% of features used per tree)
  - learning_rate: 0.2 (step size for gradient descent)
  - max_depth: 10 (maximum depth of each tree)
  - n_estimators: 100 (number of trees in the ensemble)
  - subsample: 1.0 (100% of samples used per tree)
  - random_state: 42 (for reproducibility)
- **performance_metrics**: All evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- **training_time_seconds**: Time taken to train the model

### feature_names.json
This JSON file lists the 10 features used by the model for making predictions. These features were selected through the feature engineering process and represent the most important network traffic characteristics for DoS detection:
1. **rate** - Packet transmission rate
2. **sload** - Source to destination bytes per second
3. **sbytes** - Source to destination bytes
4. **dload** - Destination to source bytes per second
5. **proto** - Protocol type (encoded)
6. **dtcpb** - Destination TCP base sequence number
7. **stcpb** - Source TCP base sequence number
8. **dmean** - Mean of packet size from destination
9. **tcprtt** - TCP round-trip time
10. **dur** - Connection duration

### train_xgboost.py
The Python script used to train this model. It loads the preprocessed dataset, splits it into training and testing sets, trains the XGBoost classifier with the specified hyperparameters, evaluates performance, and saves all outputs.

### generate_separate_images.py
A utility script that generates individual high-resolution (300 DPI) PNG images for each visualization. This script loads the trained model and creates publication-ready figures.

### images/ (folder)
Contains four separate high-resolution visualizations:

#### images/confusion_matrix.png
The confusion matrix shows the model's prediction breakdown:
- **True Negatives (Top-Left)**: Normal traffic correctly identified as Normal
- **False Positives (Top-Right)**: Normal traffic incorrectly flagged as DoS attack
- **False Negatives (Bottom-Left)**: DoS attacks missed (incorrectly classified as Normal)
- **True Positives (Bottom-Right)**: DoS attacks correctly detected

#### images/roc_curve.png
The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate at various classification thresholds. The area under this curve (AUC) is 0.9913, indicating excellent discrimination capability. A perfect classifier would have AUC = 1.0, while a random classifier would have AUC = 0.5. The blue shaded area represents the model's performance advantage over random guessing.

#### images/feature_importance.png
This horizontal bar chart shows which features contribute most to the model's predictions. Features with higher importance scores have more influence on the classification decision. This helps understand what network traffic characteristics are most indicative of DoS attacks. The most important features for this model are 'proto' (protocol type) and 'sload' (source load).

#### images/performance_metrics.png
A bar chart displaying all five key performance metrics:
- Accuracy: Overall correctness of predictions
- Precision: Of all predicted DoS attacks, how many were actually attacks
- Recall: Of all actual DoS attacks, how many were correctly detected
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under the ROC curve

---

## Results Interpretation

### Overall Performance
The XGBoost model achieved **95.78% accuracy**, meaning it correctly classified approximately 96 out of every 100 network traffic samples. This is the best performance among all five models tested in this research.

### Detection Capability
- **Precision (96.52%)**: When the model predicts a DoS attack, it is correct 96.52% of the time. This means only 3.48% of alerts are false alarms, which is important for real-world deployment where too many false positives can overwhelm security teams.

- **Recall (94.99%)**: The model successfully detects 94.99% of all actual DoS attacks. This means only about 5% of attacks go undetected, providing strong security coverage.

- **F1-Score (95.75%)**: This balanced metric combines precision and recall, showing the model maintains excellent performance on both fronts without sacrificing one for the other.

### Classification Quality
- **ROC-AUC (99.13%)**: This near-perfect score indicates the model has excellent ability to distinguish between DoS attacks and normal traffic across all possible classification thresholds. It means the model consistently ranks actual attacks higher than normal traffic in terms of attack probability.

### Confusion Matrix Breakdown
From the confusion matrix:
- **790 True Negatives**: Normal traffic correctly classified
- **28 False Positives**: Normal traffic incorrectly flagged as attacks (3.4% false alarm rate)
- **41 False Negatives**: DoS attacks missed (5.0% miss rate)
- **777 True Positives**: DoS attacks correctly detected

### Feature Insights
The feature importance analysis reveals that:
- **Protocol type (proto)** is the most important feature, suggesting different protocols have distinct patterns during DoS attacks
- **Source load (sload)** is the second most important, indicating that bandwidth consumption is a key indicator
- **Destination mean packet size (dmean)** and **TCP round-trip time (tcprtt)** also contribute significantly

### Cross-Validation Stability
The 5-fold cross-validation F1-Score of **95.26% (±0.30%)** demonstrates that the model's performance is stable and not dependent on a particular train-test split. The low standard deviation indicates consistent performance across different data subsets.

### Practical Implications
This model is suitable for real-world DoS detection because:
1. High precision minimizes false alarms that could fatigue security analysts
2. High recall ensures most attacks are caught
3. Fast training (0.28 seconds) allows for quick model updates
4. The model can process predictions in milliseconds, suitable for real-time detection
