#!/usr/bin/env python3
"""
External Benchmarking - Test Models on Completely Unseen Data
=============================================================
Tests all 5 trained models on the 175,341 record dataset that
the models have NEVER seen during training.

This proves model generalization to new data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
import joblib
import json
import os
from datetime import datetime

# Create output directories
os.makedirs('benchmark_results', exist_ok=True)
os.makedirs('benchmark_images', exist_ok=True)

# High resolution settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

print("="*70)
print("EXTERNAL BENCHMARKING - TESTING ON COMPLETELY UNSEEN DATA")
print("="*70)

# ============================================================================
# STEP 1: Load Original Training Data for Scaler Fitting
# ============================================================================
print("\n[STEP 1] Loading original training data for scaler fitting...")

# The 10 features our models use
FEATURE_NAMES = ['rate', 'sload', 'sbytes', 'dload', 'proto', 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']

# Load original training data (before scaling) to fit scaler
original_data_path = "../../01_data_preparation/data/statistical_features.csv"
original_df = pd.read_csv(original_data_path)
print(f"  Original training data: {original_df.shape[0]:,} records")

# Fit scaler on original training data
scaler = StandardScaler()
scaler.fit(original_df[FEATURE_NAMES])
print(f"  Scaler fitted on {len(FEATURE_NAMES)} features")

# ============================================================================
# STEP 2: Load Benchmark Data (175,341 records - NEVER SEEN)
# ============================================================================
print("\n[STEP 2] Loading benchmark data (COMPLETELY UNSEEN)...")

benchmark_path = "../../01_data_preparation/data/official_datasets/UNSW_NB15_BENCHMARK_DATA_175341.csv"
benchmark_df = pd.read_csv(benchmark_path)
print(f"  Benchmark data loaded: {benchmark_df.shape[0]:,} records")
print(f"  Columns: {benchmark_df.shape[1]}")

# Check attack distribution
print("\n  Attack Category Distribution:")
attack_dist = benchmark_df['attack_cat'].value_counts()
for cat, count in attack_dist.items():
    print(f"    {cat}: {count:,} ({count/len(benchmark_df)*100:.1f}%)")

# ============================================================================
# STEP 3: Extract DoS and Normal for Binary Classification
# ============================================================================
print("\n[STEP 3] Extracting DoS and Normal samples...")

dos_samples = benchmark_df[benchmark_df['attack_cat'] == 'DoS'].copy()
normal_samples = benchmark_df[benchmark_df['attack_cat'] == 'Normal'].copy()

print(f"  DoS samples: {len(dos_samples):,}")
print(f"  Normal samples: {len(normal_samples):,}")

# Combine for binary classification
benchmark_binary = pd.concat([dos_samples, normal_samples], ignore_index=True)
print(f"  Total for binary classification: {len(benchmark_binary):,}")

# Create binary labels (0=Normal, 1=DoS)
benchmark_binary['binary_label'] = (benchmark_binary['attack_cat'] == 'DoS').astype(int)

# ============================================================================
# STEP 4: Extract and Scale Features
# ============================================================================
print("\n[STEP 4] Extracting and scaling features...")

# Check if all features exist
missing_features = [f for f in FEATURE_NAMES if f not in benchmark_binary.columns]
if missing_features:
    print(f"  ERROR: Missing features: {missing_features}")
    exit(1)

# Extract features
X_benchmark = benchmark_binary[FEATURE_NAMES].copy()
y_benchmark = benchmark_binary['binary_label'].copy()

print(f"  Features extracted: {X_benchmark.shape}")
print(f"  Labels: {y_benchmark.value_counts().to_dict()}")

# Encode 'proto' feature (categorical to numeric)
# Use LabelEncoder to convert protocol strings to numbers
from sklearn.preprocessing import LabelEncoder
if X_benchmark['proto'].dtype == 'object':
    print(f"  Encoding 'proto' feature (categorical to numeric)...")
    proto_encoder = LabelEncoder()
    X_benchmark['proto'] = proto_encoder.fit_transform(X_benchmark['proto'].astype(str))
    print(f"  Proto values encoded: {len(proto_encoder.classes_)} unique protocols")

# Handle any missing values
if X_benchmark.isnull().sum().sum() > 0:
    print(f"  Warning: Found {X_benchmark.isnull().sum().sum()} missing values, filling with 0")
    X_benchmark = X_benchmark.fillna(0)

# Convert to numeric and handle any remaining issues
X_benchmark = X_benchmark.apply(pd.to_numeric, errors='coerce').fillna(0)

# Scale using the fitted scaler
X_benchmark_scaled = scaler.transform(X_benchmark)
print(f"  Features scaled using original training scaler")

# ============================================================================
# STEP 5: Load All Trained Models
# ============================================================================
print("\n[STEP 5] Loading trained models...")

models = {
    'XGBoost': joblib.load('xgboost/xgboost_model.pkl'),
    'Random Forest': joblib.load('random_forest/random_forest_model.pkl'),
    'SVM': joblib.load('svm/svm_model.pkl'),
    'MLP': joblib.load('mlp/mlp_model.pkl'),
    'Logistic Regression': joblib.load('logistic_regression/logistic_regression_model.pkl')
}

print(f"  Loaded {len(models)} models")

# ============================================================================
# STEP 6: Benchmark All Models
# ============================================================================
print("\n[STEP 6] Running benchmark tests...")
print("="*70)

results = {}
all_predictions = {}
all_probabilities = {}

for model_name, model in models.items():
    print(f"\n  Testing {model_name}...")

    # Predict
    y_pred = model.predict(X_benchmark_scaled)
    y_pred_proba = model.predict_proba(X_benchmark_scaled)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_benchmark, y_pred),
        'precision': precision_score(y_benchmark, y_pred),
        'recall': recall_score(y_benchmark, y_pred),
        'f1_score': f1_score(y_benchmark, y_pred),
        'roc_auc': roc_auc_score(y_benchmark, y_pred_proba)
    }

    results[model_name] = metrics
    all_predictions[model_name] = y_pred
    all_probabilities[model_name] = y_pred_proba

    print(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {metrics['precision']*100:.2f}%")
    print(f"    Recall:    {metrics['recall']*100:.2f}%")
    print(f"    F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"    ROC-AUC:   {metrics['roc_auc']*100:.2f}%")

# ============================================================================
# STEP 7: Generate Separate High-Resolution Images
# ============================================================================
print("\n[STEP 7] Generating benchmark images...")

model_colors = {
    'XGBoost': '#3498db',
    'Random Forest': '#2ecc71',
    'MLP': '#e67e22',
    'SVM': '#9b59b6',
    'Logistic Regression': '#e74c3c'
}

# -----------------------------------------------------------------------------
# IMAGE 1: Benchmark Performance - All Metrics
# -----------------------------------------------------------------------------
print("  Generating: 01_benchmark_all_metrics.png")

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.15

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metric_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

for i, (metric_key, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
    values = [results[m][metric_key] * 100 for m in models.keys()]
    offset = (i - 2) * width
    bars = ax.bar(x + offset, values, width, label=metric_label, color=metric_colors[i], edgecolor='black', linewidth=0.5)

ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
ax.set_title('External Benchmark Results - All Metrics\n(Tested on 175,341 UNSEEN Records)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(list(models.keys()), fontsize=11)
ax.set_ylim([50, 110])
ax.legend(loc='upper right', fontsize=10, ncol=5)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=90, color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('benchmark_images/01_benchmark_all_metrics.png', bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# IMAGE 2: F1-Score Comparison
# -----------------------------------------------------------------------------
print("  Generating: 02_benchmark_f1_score.png")

fig, ax = plt.subplots(figsize=(12, 7))

f1_values = [results[m]['f1_score'] * 100 for m in models.keys()]
bars = ax.bar(list(models.keys()), f1_values, color=list(model_colors.values()), edgecolor='black', linewidth=1.5)

for bar, val in zip(bars, f1_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('F1-Score (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
ax.set_title('External Benchmark - F1-Score Comparison\n(175,341 Unseen Records)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([50, 105])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_images/02_benchmark_f1_score.png', bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# IMAGE 3: ROC Curves - All Models
# -----------------------------------------------------------------------------
print("  Generating: 03_benchmark_roc_curves.png")

fig, ax = plt.subplots(figsize=(10, 8))

for model_name in models.keys():
    fpr, tpr, _ = roc_curve(y_benchmark, all_probabilities[model_name])
    auc_val = results[model_name]['roc_auc']
    ax.plot(fpr, tpr, color=model_colors[model_name], linewidth=2.5,
            label=f'{model_name} (AUC = {auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('External Benchmark - ROC Curves\n(175,341 Unseen Records)', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_images/03_benchmark_roc_curves.png', bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# IMAGE 4: Confusion Matrices - All Models
# -----------------------------------------------------------------------------
print("  Generating: 04_benchmark_confusion_matrices.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

cmap_colors = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']

for idx, (model_name, y_pred) in enumerate(all_predictions.items()):
    ax = axes[idx]
    cm = confusion_matrix(y_benchmark, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_colors[idx],
                xticklabels=['Normal', 'DoS'],
                yticklabels=['Normal', 'DoS'],
                annot_kws={'size': 12},
                ax=ax)

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')

# Hide the 6th subplot
axes[5].axis('off')

plt.suptitle('External Benchmark - Confusion Matrices\n(175,341 Unseen Records)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('benchmark_images/04_benchmark_confusion_matrices.png', bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# IMAGE 5: Training vs Benchmark Comparison
# -----------------------------------------------------------------------------
print("  Generating: 05_training_vs_benchmark.png")

# Training results (from original training)
training_f1 = {
    'XGBoost': 95.75,
    'Random Forest': 95.21,
    'MLP': 92.15,
    'SVM': 89.12,
    'Logistic Regression': 78.61
}

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.35

training_vals = [training_f1[m] for m in models.keys()]
benchmark_vals = [results[m]['f1_score'] * 100 for m in models.keys()]

bars1 = ax.bar(x - width/2, training_vals, width, label='Training (8,178 samples)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, benchmark_vals, width, label='Benchmark (175,341 unseen)', color='#e74c3c', edgecolor='black')

# Add value labels
for bar, val in zip(bars1, training_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar, val in zip(bars2, benchmark_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('F1-Score (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
ax.set_title('Training vs External Benchmark Performance', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(list(models.keys()), fontsize=11)
ax.set_ylim([50, 110])
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_images/05_training_vs_benchmark.png', bbox_inches='tight', facecolor='white')
plt.close()

# -----------------------------------------------------------------------------
# IMAGE 6: Benchmark Summary Table
# -----------------------------------------------------------------------------
print("  Generating: 06_benchmark_summary_table.png")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

table_data = []
for model_name in models.keys():
    m = results[model_name]
    table_data.append([
        model_name,
        f"{m['accuracy']*100:.2f}%",
        f"{m['precision']*100:.2f}%",
        f"{m['recall']*100:.2f}%",
        f"{m['f1_score']*100:.2f}%",
        f"{m['roc_auc']*100:.2f}%"
    ])

columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=['#3498db']*6
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_text_props(fontweight='bold', color='white')
    table[(0, i)].set_facecolor('#2c3e50')

# Color rows
row_colors = ['#d5f4e6', '#d5f4e6', '#ffeaa7', '#ffeaa7', '#fab1a0']
for row in range(1, len(table_data) + 1):
    for col in range(len(columns)):
        table[(row, col)].set_facecolor(row_colors[row-1])

ax.set_title('External Benchmark Results Summary\n(Tested on 175,341 Completely Unseen Records)',
             fontsize=18, fontweight='bold', pad=20, y=0.95)

plt.tight_layout()
plt.savefig('benchmark_images/06_benchmark_summary_table.png', bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# STEP 8: Save Results to JSON
# ============================================================================
print("\n[STEP 8] Saving benchmark results...")

benchmark_report = {
    "benchmark_info": {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_dataset": "UNSW_NB15_BENCHMARK_DATA_175341.csv",
        "total_records": len(benchmark_df),
        "dos_samples": len(dos_samples),
        "normal_samples": len(normal_samples),
        "binary_samples_tested": len(benchmark_binary),
        "features_used": FEATURE_NAMES
    },
    "model_results": {}
}

for model_name, metrics in results.items():
    benchmark_report["model_results"][model_name] = {
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1_score": metrics['f1_score'],
        "roc_auc": metrics['roc_auc']
    }

with open('benchmark_results/external_benchmark_results.json', 'w') as f:
    json.dump(benchmark_report, f, indent=4)

print(f"  Results saved to: benchmark_results/external_benchmark_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EXTERNAL BENCHMARKING COMPLETE")
print("="*70)

print(f"\nBenchmark Dataset:")
print(f"  - Total records: {len(benchmark_df):,}")
print(f"  - DoS samples: {len(dos_samples):,}")
print(f"  - Normal samples: {len(normal_samples):,}")
print(f"  - Binary classification samples: {len(benchmark_binary):,}")

print(f"\nModel Performance on UNSEEN Data:")
print("-"*50)
for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
    print(f"  {model_name:20s}: F1={metrics['f1_score']*100:.2f}%  Acc={metrics['accuracy']*100:.2f}%")

print(f"\nImages Generated (300 DPI):")
print("  benchmark_images/")
print("    01_benchmark_all_metrics.png")
print("    02_benchmark_f1_score.png")
print("    03_benchmark_roc_curves.png")
print("    04_benchmark_confusion_matrices.png")
print("    05_training_vs_benchmark.png")
print("    06_benchmark_summary_table.png")

print(f"\nResults File:")
print("  benchmark_results/external_benchmark_results.json")

print("\n" + "="*70)
print("GENERALIZATION PROOF: Models tested on completely unseen data!")
print("="*70)
