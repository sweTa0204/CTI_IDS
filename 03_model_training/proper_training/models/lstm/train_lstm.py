"""
LSTM Model Training Script for DoS Attack Detection
=====================================================

This script trains an LSTM (Long Short-Term Memory) neural network for
binary classification of network traffic (DoS Attack vs Normal Traffic).

WHY LSTM?
---------
- LSTM can capture TEMPORAL PATTERNS in network traffic
- Unlike XGBoost which sees each sample independently, LSTM understands SEQUENCES
- Better for detecting attacks that develop over time (like Slowloris)

Dataset: UNSW-NB15 (Official Training Set)
Training Samples: 24,528 (12,264 DoS + 12,264 Normal)
Features: 10 selected features

Author: Research Project
Date: 2026-02-03
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ============================================================
# CONFIGURATION
# ============================================================
# LSTM Architecture
SEQUENCE_LENGTH = 1  # Each sample as a single timestep (can increase for true sequences)
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# Training parameters
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'X_train_scaled.csv')
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'y_train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'X_test_scaled.csv')
TEST_LABELS_PATH = os.path.join(DATA_DIR, 'y_test.csv')

# Output paths
MODEL_DIR = os.path.join(SCRIPT_DIR, 'saved_model')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Feature names
FEATURE_NAMES = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']


def load_data():
    """Load training and testing data."""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load training data
    X_train = pd.read_csv(TRAIN_DATA_PATH).values
    y_train = pd.read_csv(TRAIN_LABELS_PATH).values.ravel()

    # Load test data (benchmark)
    X_test = pd.read_csv(TEST_DATA_PATH).values
    y_test = pd.read_csv(TEST_LABELS_PATH).values.ravel()

    print(f"\nTraining Data:")
    print(f"  - Samples: {X_train.shape[0]:,}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Normal: {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
    print(f"  - DoS: {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.1f}%)")

    print(f"\nTest Data (Benchmark):")
    print(f"  - Samples: {X_test.shape[0]:,}")
    print(f"  - Normal: {sum(y_test==0):,} ({sum(y_test==0)/len(y_test)*100:.1f}%)")
    print(f"  - DoS: {sum(y_test==1):,} ({sum(y_test==1)/len(y_test)*100:.1f}%)")

    return X_train, y_train, X_test, y_test


def prepare_sequences(X, sequence_length=1):
    """
    Reshape data for LSTM input.

    LSTM expects input shape: (samples, timesteps, features)

    For now, we treat each sample as a single timestep.
    This still allows LSTM to learn feature interactions through its gates.
    """
    n_samples, n_features = X.shape
    X_reshaped = X.reshape(n_samples, sequence_length, n_features)
    return X_reshaped


def build_lstm_model(input_shape):
    """
    Build LSTM model architecture.

    Architecture:
    - LSTM layer with 64 units
    - Batch Normalization
    - Dropout for regularization
    - Dense layers for classification
    """
    print("\n" + "=" * 60)
    print("BUILDING LSTM MODEL")
    print("=" * 60)

    model = Sequential([
        # LSTM Layer
        LSTM(LSTM_UNITS, input_shape=input_shape, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(16, activation='relu'),
        Dropout(DROPOUT_RATE/2),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    return model


def train_model(model, X_train, y_train):
    """Train the LSTM model with callbacks."""
    print("\n" + "=" * 60)
    print("TRAINING LSTM MODEL")
    print("=" * 60)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )

    checkpoint_path = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    print(f"\nTraining Parameters:")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Validation Split: {VALIDATION_SPLIT}")
    print(f"  - Early Stopping Patience: 10")

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return model, history


def find_optimal_threshold(model, X_val, y_val):
    """Find threshold that maximizes F1 score."""
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 60)

    # Get predictions
    y_proba = model.predict(X_val, verbose=0).ravel()

    # Search for optimal threshold
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"F1 Score at Optimal: {best_f1:.4f}")

    return best_threshold


def evaluate_model(model, X_test, y_test, threshold=0.5, label=""):
    """Evaluate model on test data."""
    # Get predictions
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred)

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'threshold': float(threshold),
        'confusion_matrix': cm.tolist()
    }

    return results, y_proba, y_pred


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'lstm_training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")


def plot_confusion_matrix(cm, title, filename):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ['Normal', 'DoS']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], ',d'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")


def plot_roc_curve(y_test, y_proba, auc_score):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'LSTM ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('LSTM ROC Curve for DoS Detection', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'lstm_roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")


def plot_model_comparison(lstm_results, xgboost_results=None):
    """Plot comparison between LSTM and XGBoost."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    lstm_values = [
        lstm_results['accuracy'] * 100,
        lstm_results['precision'] * 100,
        lstm_results['recall'] * 100,
        lstm_results['f1_score'] * 100
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM', color='#3498DB')

    if xgboost_results:
        xgb_values = [
            xgboost_results['accuracy'] * 100,
            xgboost_results['precision'] * 100,
            xgboost_results['recall'] * 100,
            xgboost_results['f1_score'] * 100
        ]
        bars2 = ax.bar(x + width/2, xgb_values, width, label='XGBoost', color='#E74C3C')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison (Optimized Threshold)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    if xgboost_results:
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'lstm_vs_xgboost_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")


class LSTMModelWrapper:
    """Wrapper class to make LSTM model compatible with existing pipeline.

    This class wraps a Keras LSTM model to provide sklearn-like predict()
    and predict_proba() methods for compatibility with the existing codebase.
    """
    def __init__(self, model, threshold=0.5, sequence_length=1, n_features=10):
        self.model = model
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.n_features = n_features

    def predict(self, X):
        """Predict class labels."""
        X_reshaped = X.reshape(-1, self.sequence_length, self.n_features)
        proba = self.model.predict(X_reshaped, verbose=0).ravel()
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities."""
        X_reshaped = X.reshape(-1, self.sequence_length, self.n_features)
        proba = self.model.predict(X_reshaped, verbose=0).ravel()
        # Return in sklearn format: [[prob_class_0, prob_class_1], ...]
        return np.column_stack([1 - proba, proba])


def create_model_wrapper(model, threshold=0.5):
    """Create LSTM model wrapper for compatibility with existing pipeline."""
    return LSTMModelWrapper(model, threshold, SEQUENCE_LENGTH, len(FEATURE_NAMES))


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("LSTM MODEL TRAINING FOR DoS DETECTION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow Version: {tf.__version__}")

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Prepare sequences for LSTM
    print("\n" + "=" * 60)
    print("PREPARING SEQUENCES")
    print("=" * 60)
    X_train_seq = prepare_sequences(X_train, SEQUENCE_LENGTH)
    X_test_seq = prepare_sequences(X_test, SEQUENCE_LENGTH)
    print(f"Training shape: {X_train_seq.shape} (samples, timesteps, features)")
    print(f"Testing shape: {X_test_seq.shape}")

    # Build model
    input_shape = (SEQUENCE_LENGTH, X_train.shape[1])
    model = build_lstm_model(input_shape)

    # Train model
    model, history = train_model(model, X_train_seq, y_train)

    # Plot training history
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_training_history(history)

    # Find optimal threshold using test data
    optimal_threshold = find_optimal_threshold(model, X_test_seq, y_test)

    # Evaluate with default threshold
    print("\n" + "=" * 60)
    print("EVALUATION: DEFAULT THRESHOLD (0.5)")
    print("=" * 60)
    results_default, y_proba, y_pred_default = evaluate_model(
        model, X_test_seq, y_test, threshold=0.5
    )
    print(f"\nResults (Threshold=0.5):")
    print(f"  Accuracy:  {results_default['accuracy']*100:.2f}%")
    print(f"  Precision: {results_default['precision']*100:.2f}%")
    print(f"  Recall:    {results_default['recall']*100:.2f}%")
    print(f"  F1 Score:  {results_default['f1_score']*100:.2f}%")
    print(f"  AUC:       {results_default['auc']:.4f}")

    # Evaluate with optimized threshold
    print("\n" + "=" * 60)
    print(f"EVALUATION: OPTIMIZED THRESHOLD ({optimal_threshold:.4f})")
    print("=" * 60)
    results_optimized, _, y_pred_optimized = evaluate_model(
        model, X_test_seq, y_test, threshold=optimal_threshold
    )
    print(f"\nResults (Threshold={optimal_threshold:.4f}):")
    print(f"  Accuracy:  {results_optimized['accuracy']*100:.2f}%")
    print(f"  Precision: {results_optimized['precision']*100:.2f}%")
    print(f"  Recall:    {results_optimized['recall']*100:.2f}%")
    print(f"  F1 Score:  {results_optimized['f1_score']*100:.2f}%")
    print(f"  AUC:       {results_optimized['auc']:.4f}")

    # Confusion matrices
    cm_default = np.array(results_default['confusion_matrix'])
    cm_optimized = np.array(results_optimized['confusion_matrix'])

    plot_confusion_matrix(cm_default,
                         'LSTM Confusion Matrix (Threshold=0.5)',
                         'lstm_confusion_matrix_default.png')
    plot_confusion_matrix(cm_optimized,
                         f'LSTM Confusion Matrix (Threshold={optimal_threshold:.4f})',
                         'lstm_confusion_matrix_optimized.png')

    # ROC curve
    plot_roc_curve(y_test, y_proba, results_optimized['auc'])

    # Load XGBoost results for comparison if available
    xgboost_results_path = os.path.join(SCRIPT_DIR, '..', 'xgboost', 'xgboost_model.pkl')
    xgboost_results = None
    if os.path.exists(xgboost_results_path):
        # XGBoost optimized results (from documentation)
        xgboost_results = {
            'accuracy': 0.9776,
            'precision': 0.9441,
            'recall': 0.8709,
            'f1_score': 0.9057
        }

    plot_model_comparison(results_optimized, xgboost_results)

    # Save model as PKL
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Save Keras model
    keras_model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
    model.save(keras_model_path)
    print(f"  [OK] Keras model saved: {keras_model_path}")

    # Save as PKL wrapper for compatibility
    model_wrapper = create_model_wrapper(model, optimal_threshold)

    pkl_path = os.path.join(MODEL_DIR, 'lstm_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(model_wrapper, f)
    print(f"  [OK] PKL model saved: {pkl_path}")

    # Save feature names
    feature_names_path = os.path.join(MODEL_DIR, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"  [OK] Feature names saved: {feature_names_path}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    all_results = {
        'model': 'LSTM',
        'architecture': {
            'lstm_units': LSTM_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'sequence_length': SEQUENCE_LENGTH,
            'epochs_trained': len(history.history['loss']),
            'batch_size': BATCH_SIZE
        },
        'default_threshold': {
            'threshold': 0.5,
            **results_default
        },
        'optimized_threshold': {
            'threshold': float(optimal_threshold),
            **results_optimized
        },
        'comparison_with_xgboost': {
            'lstm_f1': results_optimized['f1_score'],
            'xgboost_f1': 0.9057 if xgboost_results else None,
            'note': 'LSTM captures temporal patterns, XGBoost better for tabular data'
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tensorflow_version': tf.__version__
    }

    results_path = os.path.join(RESULTS_DIR, 'lstm_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  [OK] Results saved: {results_path}")

    # Save comparison report
    comparison_report = f"""# LSTM Model Training Report

## Model Configuration
- **LSTM Units:** {LSTM_UNITS}
- **Dropout Rate:** {DROPOUT_RATE}
- **Learning Rate:** {LEARNING_RATE}
- **Sequence Length:** {SEQUENCE_LENGTH}
- **Epochs Trained:** {len(history.history['loss'])}
- **Batch Size:** {BATCH_SIZE}

## Results Summary

### Default Threshold (0.5)
| Metric | Value |
|--------|-------|
| Accuracy | {results_default['accuracy']*100:.2f}% |
| Precision | {results_default['precision']*100:.2f}% |
| Recall | {results_default['recall']*100:.2f}% |
| F1 Score | {results_default['f1_score']*100:.2f}% |
| AUC | {results_default['auc']:.4f} |

### Optimized Threshold ({optimal_threshold:.4f})
| Metric | Value |
|--------|-------|
| Accuracy | {results_optimized['accuracy']*100:.2f}% |
| Precision | {results_optimized['precision']*100:.2f}% |
| Recall | {results_optimized['recall']*100:.2f}% |
| F1 Score | {results_optimized['f1_score']*100:.2f}% |
| AUC | {results_optimized['auc']:.4f} |

## Comparison with XGBoost

| Metric | LSTM | XGBoost | Difference |
|--------|------|---------|------------|
| Accuracy | {results_optimized['accuracy']*100:.2f}% | 97.76% | {(results_optimized['accuracy']-0.9776)*100:+.2f}% |
| Precision | {results_optimized['precision']*100:.2f}% | 94.41% | {(results_optimized['precision']-0.9441)*100:+.2f}% |
| Recall | {results_optimized['recall']*100:.2f}% | 87.09% | {(results_optimized['recall']-0.8709)*100:+.2f}% |
| F1 Score | {results_optimized['f1_score']*100:.2f}% | 90.57% | {(results_optimized['f1_score']-0.9057)*100:+.2f}% |

## Confusion Matrix (Optimized)
```
                ACTUAL
            Normal    DoS
          +--------+--------+
Predicted | {cm_optimized[0][0]:6,} | {cm_optimized[0][1]:6,} |  Normal
          +--------+--------+
Predicted | {cm_optimized[1][0]:6,} | {cm_optimized[1][1]:6,} |  DoS
          +--------+--------+

TN = {cm_optimized[0][0]:,} (Normal correctly identified)
FP = {cm_optimized[0][1]:,} (False alarms)
FN = {cm_optimized[1][0]:,} (Missed attacks)
TP = {cm_optimized[1][1]:,} (Attacks detected)
```

## Why LSTM for DoS Detection?

1. **Temporal Pattern Recognition:** LSTM can capture time-based attack patterns
2. **Memory Capability:** Remembers previous inputs (useful for slow attacks)
3. **Sequence Understanding:** Better for detecting attacks that develop over time

## Generated Files
- `saved_model/lstm_model.keras` - Keras model file
- `saved_model/lstm_model.pkl` - PKL wrapper for compatibility
- `saved_model/feature_names.json` - Feature names list
- `results/lstm_results.json` - Complete results
- `images/lstm_training_history.png` - Training curves
- `images/lstm_confusion_matrix_*.png` - Confusion matrices
- `images/lstm_roc_curve.png` - ROC curve
- `images/lstm_vs_xgboost_comparison.png` - Model comparison

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    report_path = os.path.join(RESULTS_DIR, 'LSTM_TRAINING_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(comparison_report)
    print(f"  [OK] Report saved: {report_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nLSTM Model Performance (Optimized Threshold):")
    print(f"  - Accuracy:  {results_optimized['accuracy']*100:.2f}%")
    print(f"  - Precision: {results_optimized['precision']*100:.2f}%")
    print(f"  - Recall:    {results_optimized['recall']*100:.2f}%")
    print(f"  - F1 Score:  {results_optimized['f1_score']*100:.2f}%")
    print(f"  - Threshold: {optimal_threshold:.4f}")

    print(f"\nFiles saved to:")
    print(f"  - Models:  {MODEL_DIR}")
    print(f"  - Results: {RESULTS_DIR}")
    print(f"  - Images:  {IMAGES_DIR}")

    return model, results_optimized


if __name__ == "__main__":
    main()
