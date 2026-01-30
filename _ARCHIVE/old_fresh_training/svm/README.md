# SVM Model - DoS Attack Detection

## Overview

Support Vector Machine (SVM) is a powerful supervised learning algorithm that finds the optimal hyperplane to separate different classes in a high-dimensional feature space. Using the RBF (Radial Basis Function) kernel, SVM can capture non-linear relationships in the data by implicitly mapping features into a higher-dimensional space.

In this project, SVM is used for binary classification to distinguish between **DoS (Denial of Service) attacks** and **Normal network traffic**. This model achieved an accuracy of **89.61%** and an F1-Score of **89.12%**. The model was trained on 6,542 samples and tested on 1,636 samples from the UNSW-NB15 dataset.

**Key Characteristics:**
- Algorithm Type: Support Vector Machine with RBF Kernel
- Classification Task: Binary (DoS Attack vs Normal Traffic)
- Kernel: Radial Basis Function (RBF)
- Training Time: 2.16 seconds

---

## Files

### svm_model.pkl
This is the trained SVM model saved in Python's pickle format. The file contains the support vectors, dual coefficients, and all learned parameters needed for making predictions. The model can be loaded directly to make predictions on new network traffic data without retraining.

### training_results.json
This JSON file contains all the training metadata and performance metrics. It includes:
- **model_name**: Identifier for the model ("SVM")
- **training_date**: When the model was trained
- **best_parameters**: The hyperparameters used for training
  - C: 10.0 (regularization parameter)
  - gamma: scale (kernel coefficient)
  - kernel: rbf (Radial Basis Function)
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

### train_svm.py
The Python script used to train this model. It loads the preprocessed dataset, splits it into training and testing sets, trains the SVM classifier with the specified hyperparameters, evaluates performance, and saves all outputs.

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
The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate at various classification thresholds. The area under this curve (AUC) is 0.9530, indicating good discrimination capability. A perfect classifier would have AUC = 1.0, while a random classifier would have AUC = 0.5. The purple shaded area represents the model's performance advantage over random guessing.

#### images/model_configuration.png
Since SVM with RBF kernel does not provide direct feature importance scores, this visualization displays the model configuration and key parameters used for training. It includes kernel type, regularization parameter, and dataset information.

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
The SVM model achieved **89.61% accuracy**, meaning it correctly classified approximately 90 out of every 100 network traffic samples. While this is lower than the tree-based models (XGBoost and Random Forest), it still provides reasonable detection capability.

### Detection Capability
- **Precision (93.55%)**: When the model predicts a DoS attack, it is correct 93.55% of the time. This means only 6.45% of alerts are false alarms.

- **Recall (85.09%)**: The model successfully detects 85.09% of all actual DoS attacks. This means about 15% of attacks go undetected, which is higher than the tree-based models.

- **F1-Score (89.12%)**: This balanced metric combines precision and recall, showing the model has good but not exceptional balance between the two.

### Classification Quality
- **ROC-AUC (95.30%)**: This score indicates good ability to distinguish between DoS attacks and normal traffic, though lower than XGBoost (99.13%) and Random Forest (99.01%).

### Confusion Matrix Breakdown
From the confusion matrix:
- **764 True Negatives**: Normal traffic correctly classified
- **54 False Positives**: Normal traffic incorrectly flagged as attacks (6.6% false alarm rate)
- **122 False Negatives**: DoS attacks missed (14.9% miss rate)
- **696 True Positives**: DoS attacks correctly detected

### Why SVM Has Lower Performance
Several factors contribute to SVM's lower performance compared to tree-based models:
1. **Non-linear decision boundaries**: While RBF kernel helps, tree-based models naturally handle complex interactions better
2. **Feature scaling sensitivity**: Although data was scaled, SVM is more sensitive to feature distributions
3. **Computational complexity**: SVM's quadratic programming optimization may not find the global optimum for this dataset

### Cross-Validation Stability
The 5-fold cross-validation F1-Score of **89.10%** is consistent with the test set performance, indicating the model generalizes well despite its lower overall accuracy.

### Practical Implications
This model may be suitable for specific scenarios:
1. When interpretability of the decision boundary is important
2. As a baseline comparison for more complex models
3. In resource-constrained environments where simpler models are preferred
4. When combined with other models in an ensemble approach

### Recommendations
For production DoS detection, consider:
1. Using XGBoost or Random Forest as primary classifiers
2. Using SVM as a secondary verification model
3. Exploring different kernel functions (polynomial, sigmoid)
4. Fine-tuning the C and gamma parameters further
