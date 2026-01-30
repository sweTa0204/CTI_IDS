# Logistic Regression Model - DoS Attack Detection

## Overview

Logistic Regression is a linear classification algorithm that models the probability of a binary outcome using the logistic (sigmoid) function. Despite its simplicity, it serves as an important baseline model and provides fully interpretable coefficients that show how each feature contributes to the prediction.

In this project, Logistic Regression is used for binary classification to distinguish between **DoS (Denial of Service) attacks** and **Normal network traffic**. This model achieved an accuracy of **78.18%** and an F1-Score of **78.61%**. While this is the lowest performing model among the five tested, it provides valuable insights through its interpretable coefficients. The model was trained on 6,542 samples and tested on 1,636 samples from the UNSW-NB15 dataset.

**Key Characteristics:**
- Algorithm Type: Linear Classification (Logistic/Sigmoid Function)
- Classification Task: Binary (DoS Attack vs Normal Traffic)
- Fully Interpretable: Coefficient-based feature importance
- Training Time: 0.02 seconds (fastest among all models)

---

## Files

### logistic_regression_model.pkl
This is the trained Logistic Regression model saved in Python's pickle format. The file contains the learned coefficients for each feature and the intercept term. The model can be loaded directly to make predictions on new network traffic data without retraining.

### training_results.json
This JSON file contains all the training metadata and performance metrics. It includes:
- **model_name**: Identifier for the model ("Logistic Regression")
- **training_date**: When the model was trained
- **best_parameters**: The hyperparameters used for training
  - C: 100.0 (inverse of regularization strength)
  - max_iter: 1000 (maximum iterations for solver)
  - penalty: l2 (L2 regularization)
  - solver: liblinear (optimization algorithm)
  - random_state: 42 (for reproducibility)
- **performance_metrics**: All evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- **training_time_seconds**: Time taken to train the model
- **feature_coefficients**: The learned weight for each feature

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

### train_logistic_regression.py
The Python script used to train this model. It loads the preprocessed dataset, splits it into training and testing sets, trains the Logistic Regression classifier with the specified hyperparameters, evaluates performance, and saves all outputs.

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
The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate at various classification thresholds. The area under this curve (AUC) is 0.8530, indicating moderate discrimination capability. A perfect classifier would have AUC = 1.0, while a random classifier would have AUC = 0.5. The red shaded area represents the model's performance advantage over random guessing.

#### images/feature_coefficients.png
This horizontal bar chart shows the learned coefficients for each feature:
- **Positive coefficients** (green): Features that increase the probability of DoS attack when their values increase
- **Negative coefficients** (red): Features that decrease the probability of DoS attack when their values increase

The magnitude of each coefficient indicates the strength of that feature's influence on the prediction.

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
The Logistic Regression model achieved **78.18% accuracy**, meaning it correctly classified approximately 78 out of every 100 network traffic samples. This is the lowest performance among all five models, which is expected given the linear nature of the algorithm and the complex, non-linear patterns in network traffic data.

### Detection Capability
- **Precision (77.09%)**: When the model predicts a DoS attack, it is correct 77.09% of the time. This means about 23% of alerts are false alarms.

- **Recall (80.20%)**: The model successfully detects 80.20% of all actual DoS attacks. This means about 20% of attacks go undetected.

- **F1-Score (78.61%)**: This balanced metric combines precision and recall, showing moderate overall detection performance.

### Classification Quality
- **ROC-AUC (85.30%)**: This moderate score indicates the model can distinguish between DoS attacks and normal traffic better than random, but not as effectively as the non-linear models.

### Confusion Matrix Breakdown
From the confusion matrix:
- **656 True Negatives**: Normal traffic correctly classified
- **162 False Positives**: Normal traffic incorrectly flagged as attacks (19.8% false alarm rate)
- **162 False Negatives**: DoS attacks missed (19.8% miss rate)
- **656 True Positives**: DoS attacks correctly detected

### Feature Coefficient Analysis
The learned coefficients reveal how each feature influences the prediction:

**Features that INCREASE DoS probability (positive coefficients):**
- **sbytes (+1.91)**: Higher source bytes strongly indicate DoS attack
- **dmean (+0.76)**: Larger destination packet sizes suggest attack
- **rate (+0.73)**: Higher packet rates indicate attack
- **dur (+0.12)**: Longer connection duration slightly indicates attack

**Features that DECREASE DoS probability (negative coefficients):**
- **dload (-10.97)**: Higher destination load strongly indicates normal traffic (DoS attacks typically have low response from destination)
- **tcprtt (-0.55)**: Higher TCP round-trip time suggests normal traffic
- **proto (-0.31)**: Protocol type influences prediction
- **dtcpb (-0.30)**: Destination TCP base sequence affects classification
- **stcpb (-0.21)**: Source TCP base sequence affects classification
- **sload (-0.17)**: Source load has slight negative influence

The strong negative coefficient for **dload** (-10.97) is particularly interesting, as it suggests that DoS attacks have significantly lower destination-to-source traffic compared to normal connections.

### Why Logistic Regression Has Lower Performance
Several factors explain the lower performance:
1. **Linear decision boundary**: Cannot capture complex non-linear patterns in network traffic
2. **Feature interactions ignored**: Does not model interactions between features
3. **Simple model capacity**: Limited ability to represent complex attack signatures

### Cross-Validation Stability
The 5-fold cross-validation F1-Score of **78.17%** is almost identical to the test set performance, indicating stable generalization without overfitting.

### Value as a Baseline
Despite lower performance, Logistic Regression provides valuable contributions:
1. **Interpretability**: Clear understanding of feature contributions
2. **Speed**: Fastest training time (0.02 seconds)
3. **Baseline**: Establishes minimum expected performance
4. **Insights**: Coefficient analysis reveals important feature relationships

### Practical Implications
This model may be useful in specific scenarios:
1. When full interpretability is required for regulatory compliance
2. As a quick baseline for comparison with other models
3. In resource-constrained environments where simplicity is prioritized
4. For educational purposes to understand feature relationships
5. As part of an ensemble where diversity is valuable
