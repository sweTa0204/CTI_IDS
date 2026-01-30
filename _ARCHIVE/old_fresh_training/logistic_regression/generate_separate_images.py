#!/usr/bin/env python3
"""
Generate Separate High-Resolution Images for Logistic Regression Model
=======================================================================
Creates individual 300 DPI images for publication use.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import joblib
import json
import os

# Create images directory
os.makedirs('images', exist_ok=True)

# High resolution settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Load model and data
print("Loading model and data...")
model = joblib.load('logistic_regression_model.pkl')

with open('training_results.json', 'r') as f:
    results = json.load(f)

with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Load test data to regenerate predictions
data_path = "../../../01_data_preparation/data/final_scaled_dataset.csv"
df = pd.read_csv(data_path)
X = df.drop('label', axis=1)
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
metrics = results['performance_metrics']

# 1. Confusion Matrix
print("Generating confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal', 'DoS'],
            yticklabels=['Normal', 'DoS'],
            annot_kws={'size': 16},
            ax=ax)
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('Actual Label', fontsize=14)
ax.set_title('Logistic Regression - Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: images/confusion_matrix.png")

# 2. ROC Curve
print("Generating ROC curve...")
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, 'red', linewidth=2.5, label=f'Logistic Regression (AUC = {metrics["roc_auc"]:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.3, color='red')
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('Logistic Regression - ROC Curve', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=12)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/roc_curve.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: images/roc_curve.png")

# 3. Feature Coefficients
print("Generating feature coefficients...")
fig, ax = plt.subplots(figsize=(10, 6))
coefficients = model.coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values('Coefficient', ascending=True)

colors = ['#e74c3c' if c < 0 else '#27ae60' for c in coef_df['Coefficient']]
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
ax.set_xlabel('Coefficient Value', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
ax.set_title('Logistic Regression - Feature Coefficients', fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='y', labelsize=11)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add value labels
for bar, val in zip(bars, coef_df['Coefficient']):
    x_pos = val + 0.3 if val >= 0 else val - 0.3
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', ha=ha, fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#27ae60', label='Positive (increases DoS probability)'),
                   Patch(facecolor='#e74c3c', label='Negative (decreases DoS probability)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('images/feature_coefficients.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: images/feature_coefficients.png")

# 4. Performance Metrics
print("Generating performance metrics...")
fig, ax = plt.subplots(figsize=(10, 6))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                 metrics['f1_score'], metrics['roc_auc']]

colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax.bar(metric_names, [v*100 for v in metric_values], color=colors, edgecolor='black', linewidth=1.2)

for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_title('Logistic Regression - Performance Metrics', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim([0, 110])
ax.tick_params(axis='x', labelsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/performance_metrics.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: images/performance_metrics.png")

print("\nAll images generated successfully!")
print("Location: images/")
