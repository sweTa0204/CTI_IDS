#!/usr/bin/env python3
"""
Generate Separate High-Resolution Images for XGBoost
====================================================
Creates individual PNG files for each visualization (300 DPI).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# High resolution settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

def load_model_and_data():
    """Load trained model and test data"""
    # Load model
    model = joblib.load('xgboost_model.pkl')

    # Load feature names
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)

    # Load data
    data_path = "../../../01_data_preparation/data/final_scaled_dataset.csv"
    df = pd.read_csv(data_path)

    X = df.drop('label', axis=1)
    y = df['label']

    # Same split as training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Load metrics
    with open('training_results.json', 'r') as f:
        results = json.load(f)
    metrics = results['performance_metrics']

    return model, X_test, y_test, y_pred, y_pred_proba, feature_names, metrics

def generate_confusion_matrix(y_test, y_pred):
    """Generate high-resolution confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'DoS Attack'],
                yticklabels=['Normal', 'DoS Attack'],
                annot_kws={'size': 16},
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('Actual Label', fontsize=14)
    ax.set_title('XGBoost - Confusion Matrix', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: images/confusion_matrix.png")

def generate_roc_curve(y_test, y_pred_proba, metrics):
    """Generate high-resolution ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2.5,
            label=f'XGBoost (AUC = {metrics["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')

    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('XGBoost - ROC Curve', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/roc_curve.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: images/roc_curve.png")

def generate_feature_importance(model, feature_names):
    """Generate high-resolution feature importance chart"""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
                   color='steelblue', edgecolor='navy', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=11)

    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title('XGBoost - Feature Importance', fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.savefig('images/feature_importance.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: images/feature_importance.png")

def generate_performance_metrics(metrics):
    """Generate high-resolution performance metrics bar chart"""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_values = [
        metrics['accuracy'] * 100,
        metrics['precision'] * 100,
        metrics['recall'] * 100,
        metrics['f1_score'] * 100,
        metrics['roc_auc'] * 100
    ]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_title('XGBoost - Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim([0, 110])
    ax.tick_params(axis='x', labelsize=12)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

    plt.tight_layout()
    plt.savefig('images/performance_metrics.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: images/performance_metrics.png")

def main():
    print("="*60)
    print("GENERATING SEPARATE HIGH-RESOLUTION IMAGES")
    print("XGBoost Model - 300 DPI")
    print("="*60)

    model, X_test, y_test, y_pred, y_pred_proba, feature_names, metrics = load_model_and_data()

    print("\nGenerating images...")
    generate_confusion_matrix(y_test, y_pred)
    generate_roc_curve(y_test, y_pred_proba, metrics)
    generate_feature_importance(model, feature_names)
    generate_performance_metrics(metrics)

    print("\n" + "="*60)
    print("ALL IMAGES GENERATED")
    print("="*60)
    print("\nFiles in images/:")
    print("  - confusion_matrix.png (300 DPI)")
    print("  - roc_curve.png (300 DPI)")
    print("  - feature_importance.png (300 DPI)")
    print("  - performance_metrics.png (300 DPI)")

if __name__ == "__main__":
    main()
