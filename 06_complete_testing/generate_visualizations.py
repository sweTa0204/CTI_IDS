"""
Generate Visualizations for Complete Test Results
==================================================

This script creates publication-quality visualizations from the complete test results:
1. Confusion Matrix Heatmap
2. Attack Type Distribution (Pie Chart)
3. Severity Distribution (Bar Chart)
4. Model Performance Metrics (Bar Chart)

Author: Research Project
Date: 2026-01-30
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Paths
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_results():
    """Load all result files."""
    with open(os.path.join(OUTPUT_DIR, 'summary_report.json'), 'r') as f:
        summary = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'confusion_matrix.json'), 'r') as f:
        cm = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'attack_distribution.json'), 'r') as f:
        attack_dist = json.load(f)
    return summary, cm, attack_dist


def plot_confusion_matrix(cm):
    """Generate confusion matrix heatmap."""
    print("\n[1/4] Generating Confusion Matrix Heatmap...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create matrix
    matrix = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])

    # Plot heatmap
    im = ax.imshow(matrix, cmap='Blues')

    # Labels
    labels = ['Normal', 'DoS']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() / 2 else 'black'
            ax.text(j, i, f'{val:,}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)

    # Add labels for each cell
    cell_labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i + 0.3, f'({cell_labels[i][j]})', ha='center', va='center',
                    fontsize=10, color=color, alpha=0.7)

    plt.title('Confusion Matrix - Complete Benchmark Test\n(68,264 samples)',
              fontsize=14, fontweight='bold')
    plt.colorbar(im, label='Count')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def plot_attack_distribution(attack_dist):
    """Generate attack type distribution pie chart."""
    print("\n[2/4] Generating Attack Type Distribution...")

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = list(attack_dist['attack_types'].keys())
    sizes = list(attack_dist['attack_types'].values())
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

    # Explode the largest slice
    explode = [0.05 if s == max(sizes) else 0 for s in sizes]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=colors, explode=explode,
        shadow=True, textprops={'fontsize': 11}
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax.set_title(f'Attack Type Distribution\n(Total DoS Predictions: {attack_dist["total_dos_predictions"]:,})',
                 fontsize=14, fontweight='bold')

    # Add legend with counts
    legend_labels = [f'{l}: {s:,}' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Attack Types", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'attack_type_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def plot_severity_distribution(summary):
    """Generate severity distribution bar chart."""
    print("\n[3/4] Generating Severity Distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    severity_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    severity_colors = {
        'LOW': '#27ae60',
        'MEDIUM': '#f39c12',
        'HIGH': '#e67e22',
        'CRITICAL': '#e74c3c'
    }

    counts = [summary['severity_distribution'].get(s, 0) for s in severity_order]
    colors = [severity_colors[s] for s in severity_order]

    bars = ax.bar(severity_order, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Severity Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
    ax.set_title('Severity Distribution of DoS Detections\n(Complete Benchmark Test)',
                 fontsize=14, fontweight='bold')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'severity_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def plot_performance_metrics(cm):
    """Generate model performance metrics bar chart."""
    print("\n[4/4] Generating Performance Metrics...")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [cm['accuracy'], cm['precision'], cm['recall'], cm['f1_score']]
    colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']

    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics\n(XGBoost on UNSW-NB15 Benchmark)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)

    # Add reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'performance_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS FOR COMPLETE TEST RESULTS")
    print("=" * 70)

    # Load results
    summary, cm, attack_dist = load_results()

    # Generate all plots
    images = []

    img1 = plot_confusion_matrix(cm)
    images.append(img1)

    img2 = plot_attack_distribution(attack_dist)
    images.append(img2)

    img3 = plot_severity_distribution(summary)
    images.append(img3)

    img4 = plot_performance_metrics(cm)
    images.append(img4)

    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(images)} visualizations:")
    for img in images:
        print(f"  - {os.path.basename(img)}")

    return images


if __name__ == "__main__":
    main()
