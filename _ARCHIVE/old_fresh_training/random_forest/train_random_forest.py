#!/usr/bin/env python3
"""
Random Forest Model Training - Fresh Execution
===============================================
Clean training script with proper paths and no hardcoded values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for clean images
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_data():
    """Load the final scaled dataset"""
    data_path = "../../../01_data_preparation/data/final_scaled_dataset.csv"
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target distribution:\n{df['label'].value_counts()}")
    return df

def prepare_data(df):
    """Prepare features and target, split data"""
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, list(X.columns)

def train_model(X_train, y_train):
    """Train Random Forest with proven hyperparameters"""
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    print(f"Training completed in {training_time:.2f} seconds")

    return model, training_time

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model and compute all metrics"""
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    metrics['cv_f1_score'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print("RANDOM FOREST RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']*100:.2f}%")
    print(f"CV F1:     {metrics['cv_f1_score']*100:.2f}% (+/- {metrics['cv_f1_std']*100:.2f}%)")
    print("="*50)

    return metrics, cm, y_pred, y_pred_proba

def generate_performance_figure(metrics, cm, y_test, y_pred_proba, feature_names, model):
    """Generate a clean, non-overlapping performance figure"""

    fig = plt.figure(figsize=(14, 10))

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'DoS'],
                yticklabels=['Normal', 'DoS'],
                annot_kws={'size': 14},
                ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=10)

    # 2. ROC Curve
    ax2 = fig.add_subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, 'g-', linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    # 3. Feature Importance
    ax3 = fig.add_subplot(2, 2, 3)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    ax3.barh(importance_df['Feature'], importance_df['Importance'], color='forestgreen')
    ax3.set_xlabel('Importance', fontsize=11)
    ax3.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=10)
    ax3.tick_params(axis='y', labelsize=9)

    # 4. Performance Metrics
    ax4 = fig.add_subplot(2, 2, 4)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                     metrics['f1_score'], metrics['roc_auc']]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax4.bar(metric_names, [v*100 for v in metric_values], color=colors)

    for bar, val in zip(bars, metric_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)

    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylim([0, 110])
    ax4.tick_params(axis='x', labelsize=10)

    plt.tight_layout(pad=2.0)
    plt.savefig('random_forest_performance.png', bbox_inches='tight', facecolor='white')
    plt.close()

    print("\nSaved: random_forest_performance.png")

def save_results(model, metrics, training_time, feature_names):
    """Save model and results"""

    joblib.dump(model, 'random_forest_model.pkl')
    print("Saved: random_forest_model.pkl")

    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("Saved: feature_names.json")

    results = {
        "model_name": "Random Forest",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_parameters": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42
        },
        "performance_metrics": {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "roc_auc": metrics['roc_auc'],
            "cv_f1_score": metrics['cv_f1_score']
        },
        "training_time_seconds": training_time
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved: training_results.json")

    return results

def main():
    """Main execution"""
    print("="*60)
    print("RANDOM FOREST MODEL TRAINING - FRESH EXECUTION")
    print("="*60)

    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    model, training_time = train_model(X_train, y_train)
    metrics, cm, y_pred, y_pred_proba = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )
    generate_performance_figure(metrics, cm, y_test, y_pred_proba, feature_names, model)
    results = save_results(model, metrics, training_time, feature_names)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nFiles created:")
    print("  - random_forest_model.pkl (trained model)")
    print("  - random_forest_performance.png (performance figure)")
    print("  - training_results.json (metrics)")
    print("  - feature_names.json (feature list)")

    return results

if __name__ == "__main__":
    main()
