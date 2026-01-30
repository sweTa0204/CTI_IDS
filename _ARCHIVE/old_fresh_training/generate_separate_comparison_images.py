#!/usr/bin/env python3
"""
Generate Separate High-Resolution Model Comparison Images
=========================================================
Creates individual 300 DPI images for each comparison visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create images directory
os.makedirs('comparison_images', exist_ok=True)

# High resolution settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Model data (from training results)
models = ['XGBoost', 'Random Forest', 'MLP', 'SVM', 'Logistic Reg.']
accuracy = [95.78, 95.29, 92.11, 89.61, 78.18]
precision = [96.52, 96.84, 91.76, 93.55, 77.09]
recall = [94.99, 93.64, 92.54, 85.09, 80.20]
f1_score = [95.75, 95.21, 92.15, 89.12, 78.61]
roc_auc = [99.13, 99.01, 97.46, 95.30, 85.30]
training_time = [0.28, 0.37, 1.47, 2.16, 0.02]

# Colors for each model
model_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']

# ============================================================================
# IMAGE 1: Model Performance Comparison - All Metrics (Fixed overlapping)
# ============================================================================
print("Generating Image 1: Model Performance Comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.15  # Narrower bars to prevent overlap

metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score,
    'ROC-AUC': roc_auc
}

metric_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

for i, (metric_name, values) in enumerate(metrics.items()):
    offset = (i - 2) * width  # Center the bars
    bars = ax.bar(x + offset, values, width, label=metric_name, color=metric_colors[i], edgecolor='black', linewidth=0.5)

    # Add value labels on top of bars (smaller font, rotated)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim([70, 110])
ax.legend(loc='upper right', fontsize=10, ncol=5)
ax.grid(axis='y', alpha=0.3)

# Add horizontal line at 90% for reference
ax.axhline(y=90, color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('comparison_images/01_model_performance_all_metrics.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: comparison_images/01_model_performance_all_metrics.png")

# ============================================================================
# IMAGE 2: F1-Score Comparison (Primary Metric)
# ============================================================================
print("Generating Image 2: F1-Score Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(models, f1_score, color=model_colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, f1_score):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('F1-Score (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
ax.set_title('Model Comparison - F1-Score (Primary Metric)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([70, 105])
ax.tick_params(axis='x', labelsize=12)
ax.grid(axis='y', alpha=0.3)

# Highlight best model
best_idx = f1_score.index(max(f1_score))
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('comparison_images/02_f1_score_comparison.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: comparison_images/02_f1_score_comparison.png")

# ============================================================================
# IMAGE 3: Training Time Comparison
# ============================================================================
print("Generating Image 3: Training Time Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(models, training_time, color=model_colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, training_time):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
ax.set_title('Model Comparison - Training Time', fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='x', labelsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_images/03_training_time_comparison.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: comparison_images/03_training_time_comparison.png")

# ============================================================================
# IMAGE 4: Model Summary Table
# ============================================================================
print("Generating Image 4: Model Summary Table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Table data
table_data = [
    ['XGBoost', '95.78%', '96.52%', '94.99%', '95.75%', '99.13%', '0.28s', 'Best Overall'],
    ['Random Forest', '95.29%', '96.84%', '93.64%', '95.21%', '99.01%', '0.37s', 'Best Precision'],
    ['MLP', '92.11%', '91.76%', '92.54%', '92.15%', '97.46%', '1.47s', 'Balanced'],
    ['SVM', '89.61%', '93.55%', '85.09%', '89.12%', '95.30%', '2.16s', 'High Precision'],
    ['Logistic Reg.', '78.18%', '77.09%', '80.20%', '78.61%', '85.30%', '0.02s', 'Fastest']
]

columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time', 'Strength']

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=['#3498db']*8
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)

# Style header row
for i in range(len(columns)):
    table[(0, i)].set_text_props(fontweight='bold', color='white')
    table[(0, i)].set_facecolor('#2c3e50')

# Color rows based on model performance
row_colors = ['#d5f4e6', '#d5f4e6', '#ffeaa7', '#ffeaa7', '#fab1a0']
for row in range(1, len(table_data) + 1):
    for col in range(len(columns)):
        table[(row, col)].set_facecolor(row_colors[row-1])

ax.set_title('Model Performance Summary', fontsize=18, fontweight='bold', pad=20, y=0.95)

plt.tight_layout()
plt.savefig('comparison_images/04_model_summary_table.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: comparison_images/04_model_summary_table.png")

# ============================================================================
# IMAGE 5: Accuracy vs Training Time (Trade-off Analysis)
# ============================================================================
print("Generating Image 5: Accuracy vs Training Time...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with model names
for i, model in enumerate(models):
    ax.scatter(training_time[i], accuracy[i], s=300, c=model_colors[i],
               edgecolors='black', linewidth=2, label=model, zorder=5)
    ax.annotate(model, (training_time[i], accuracy[i]),
                textcoords="offset points", xytext=(10, 10), fontsize=11, fontweight='bold')

ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Trade-off: Accuracy vs Training Time', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

# Add quadrant labels
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('comparison_images/05_accuracy_vs_training_time.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: comparison_images/05_accuracy_vs_training_time.png")

print("\n" + "="*60)
print("All comparison images generated successfully!")
print("="*60)
print("\nImages saved in: comparison_images/")
print("  1. 01_model_performance_all_metrics.png")
print("  2. 02_f1_score_comparison.png")
print("  3. 03_training_time_comparison.png")
print("  4. 04_model_summary_table.png")
print("  5. 05_accuracy_vs_training_time.png")
