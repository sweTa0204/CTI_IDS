# MLP (Neural Network) Model - DoS Attack Detection

## Overview

Multi-Layer Perceptron (MLP) is a type of feedforward artificial neural network that consists of multiple layers of neurons with non-linear activation functions. The network learns complex patterns by adjusting connection weights through backpropagation during training.

In this project, MLP is used for binary classification to distinguish between **DoS (Denial of Service) attacks** and **Normal network traffic**. This model achieved an accuracy of **92.11%** and an F1-Score of **92.15%**. The model was trained on 6,542 samples and tested on 1,636 samples from the UNSW-NB15 dataset.

**Key Characteristics:**
- Algorithm Type: Feedforward Neural Network
- Classification Task: Binary (DoS Attack vs Normal Traffic)
- Architecture: 10 -> 150 -> 75 -> 25 -> 1
- Training Time: 1.47 seconds

---

## Files

### mlp_model.pkl
This is the trained MLP model saved in Python's pickle format. The file contains all the learned weights and biases for the neural network layers, along with the network architecture configuration. The model can be loaded directly to make predictions on new network traffic data without retraining.

### training_results.json
This JSON file contains all the training metadata and performance metrics. It includes:
- **model_name**: Identifier for the model ("MLP (Multi-Layer Perceptron)")
- **training_date**: When the model was trained
- **best_parameters**: The hyperparameters used for training
  - hidden_layer_sizes: [150, 75, 25] (three hidden layers)
  - activation: relu (Rectified Linear Unit)
  - alpha: 0.01 (L2 regularization term)
  - learning_rate_init: 0.01 (initial learning rate)
  - max_iter: 500 (maximum iterations)
  - random_state: 42 (for reproducibility)
  - early_stopping: true (stops when validation score stops improving)
- **performance_metrics**: All evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- **training_time_seconds**: Time taken to train the model
- **network_architecture**: Details about the network structure and convergence

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

### train_mlp.py
The Python script used to train this model. It loads the preprocessed dataset, splits it into training and testing sets, trains the MLP classifier with the specified hyperparameters, evaluates performance, and saves all outputs.

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
The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate against the False Positive Rate at various classification thresholds. The area under this curve (AUC) is 0.9746, indicating very good discrimination capability. A perfect classifier would have AUC = 1.0, while a random classifier would have AUC = 0.5. The orange shaded area represents the model's performance advantage over random guessing.

#### images/network_architecture.png
This visualization displays the neural network architecture showing the flow of information through the network:
- **Input Layer**: 10 neurons (one per feature)
- **Hidden Layer 1**: 150 neurons with ReLU activation
- **Hidden Layer 2**: 75 neurons with ReLU activation
- **Hidden Layer 3**: 25 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)

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
The MLP model achieved **92.11% accuracy**, meaning it correctly classified approximately 92 out of every 100 network traffic samples. This places it as the third-best performing model, behind XGBoost and Random Forest.

### Detection Capability
- **Precision (91.76%)**: When the model predicts a DoS attack, it is correct 91.76% of the time. This means about 8.2% of alerts are false alarms.

- **Recall (92.54%)**: The model successfully detects 92.54% of all actual DoS attacks. This means about 7.5% of attacks go undetected.

- **F1-Score (92.15%)**: This balanced metric combines precision and recall, showing the model has good balance between false positives and false negatives.

### Classification Quality
- **ROC-AUC (97.46%)**: This high score indicates excellent ability to distinguish between DoS attacks and normal traffic across all classification thresholds.

### Confusion Matrix Breakdown
From the confusion matrix:
- **749 True Negatives**: Normal traffic correctly classified
- **69 False Positives**: Normal traffic incorrectly flagged as attacks (8.4% false alarm rate)
- **61 False Negatives**: DoS attacks missed (7.5% miss rate)
- **757 True Positives**: DoS attacks correctly detected

### Network Architecture Insights
The MLP uses a deep architecture with three hidden layers:
- **Layer 1 (150 neurons)**: Captures basic patterns in the input features
- **Layer 2 (75 neurons)**: Combines basic patterns into more complex representations
- **Layer 3 (25 neurons)**: Further abstracts features before final classification

The ReLU activation function enables the network to learn non-linear decision boundaries while avoiding the vanishing gradient problem.

### Training Convergence
The model converged in **24 iterations** with early stopping enabled. This indicates:
- The model learned quickly from the data
- Early stopping prevented overfitting
- The validation loss stopped improving after 24 epochs

### Cross-Validation Stability
The 5-fold cross-validation F1-Score of **91.65%** is close to the test set performance, indicating stable generalization across different data splits.

### Comparison with Other Models
MLP provides a different learning paradigm compared to tree-based models:
- **Advantages**: Can learn complex non-linear patterns, good generalization with regularization
- **Disadvantages**: Less interpretable than tree-based models, sensitive to hyperparameter choices

### Practical Implications
This model is suitable for DoS detection when:
1. A balance between precision and recall is desired
2. The slightly higher false positive rate is acceptable
3. Neural network infrastructure is available for deployment
4. Model interpretability is not the primary concern
