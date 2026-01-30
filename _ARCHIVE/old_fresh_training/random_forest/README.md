# Random Forest Model - DoS Attack Detection

## Overview

Random Forest is an ensemble learning algorithm that constructs multiple decision trees during training and outputs the mode of the classes for classification. Each tree is trained on a random subset of the data with a random subset of features, making the model robust against overfitting and capable of capturing complex patterns.

In this project, Random Forest is used for binary classification to distinguish between **DoS (Denial of Service) attacks** and **Normal network traffic**. This model achieved the **second-highest performance** among all five models tested, with an accuracy of **95.29%** and an F1-Score of **95.21%**. The model was trained on 6,542 samples and tested on 1,636 samples from the UNSW-NB15 dataset.

**Key Characteristics:**
- Algorithm Type: Ensemble of Decision Trees (Bagging)
- Classification Task: Binary (DoS Attack vs Normal Traffic)
- Number of Trees: 200
- Training Time: 0.37 seconds

---

## Files

### random_forest_model.pkl
This is the trained Random Forest model saved in Python's pickle format. The file contains the complete ensemble of 200 decision trees, each with their learned split rules and thresholds. The model can be loaded directly to make predictions on new network traffic data without retraining.

### training_results.json
This JSON file contains all the training metadata and performance metrics. It includes:
- **model_name**: Identifier for the model ("Random Forest")
- **training_date**: When the model was trained
- **best_parameters**: The hyperparameters used for training
  - n_estimators: 200 (number of trees in the forest)
  - max_depth: 20 (maximum depth of each tree)
  - min_samples_split: 2 (minimum samples required to split a node)
  - min_samples_leaf: 1 (minimum samples required at a leaf node)
  - max_features: sqrt (square root of total features considered per split)
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

### train_random_forest.py
The Python script used to train this model. It loads the preprocessed dataset, splits it into training and testing sets, trains the Random Forest classifier with the specified hyperparameters, evaluates performance, and saves all outputs.

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
The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate at various classification thresholds. The area under this curve (AUC) is 0.9901, indicating excellent discrimination capability. A perfect classifier would have AUC = 1.0, while a random classifier would have AUC = 0.5. The green shaded area represents the model's performance advantage over random guessing.

#### images/feature_importance.png
This horizontal bar chart shows which features contribute most to the model's predictions based on the Gini importance (mean decrease in impurity). Features with higher importance scores have more influence on the classification decision. This helps understand what network traffic characteristics are most indicative of DoS attacks.

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
The Random Forest model achieved **95.29% accuracy**, meaning it correctly classified approximately 95 out of every 100 network traffic samples. This is the second-best performance among all five models tested, slightly behind XGBoost.

### Detection Capability
- **Precision (96.84%)**: When the model predicts a DoS attack, it is correct 96.84% of the time. This is the highest precision among all models, meaning it produces the fewest false alarms.

- **Recall (93.64%)**: The model successfully detects 93.64% of all actual DoS attacks. This means about 6.4% of attacks go undetected.

- **F1-Score (95.21%)**: This balanced metric combines precision and recall, showing the model maintains strong performance on both fronts.

### Classification Quality
- **ROC-AUC (99.01%)**: This near-perfect score indicates the model has excellent ability to distinguish between DoS attacks and normal traffic across all possible classification thresholds.

### Confusion Matrix Breakdown
From the confusion matrix:
- **792 True Negatives**: Normal traffic correctly classified
- **26 False Positives**: Normal traffic incorrectly flagged as attacks (3.2% false alarm rate)
- **52 False Negatives**: DoS attacks missed (6.4% miss rate)
- **766 True Positives**: DoS attacks correctly detected

### Feature Insights
The feature importance analysis from Random Forest reveals which features are most useful for detecting DoS attacks. The model uses the mean decrease in Gini impurity to calculate importance, which indicates how much each feature contributes to pure (homogeneous) node splits across all trees.

### Cross-Validation Stability
The 5-fold cross-validation F1-Score of **95.25%** demonstrates that the model's performance is stable and not dependent on a particular train-test split. This indicates consistent performance across different data subsets.

### Comparison with XGBoost
While Random Forest achieved slightly lower performance than XGBoost (95.29% vs 95.78% accuracy), it offers some advantages:
1. Simpler to tune with fewer hyperparameters
2. Naturally resistant to overfitting due to bagging
3. Provides interpretable feature importance scores
4. Can be parallelized easily during training

### Practical Implications
This model is suitable for real-world DoS detection because:
1. Highest precision minimizes false alarms that could fatigue security analysts
2. Good recall ensures most attacks are caught
3. Fast training (0.37 seconds) allows for quick model updates
4. The ensemble nature makes it robust against noise in network data
