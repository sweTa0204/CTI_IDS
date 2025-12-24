#!/usr/bin/env python3
"""
Generate PPT-Ready Visualizations for External Benchmarking Results
IMPROVED VERSION - Better spacing, no overlapping text
Creates publication-quality figures for faculty presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style for professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

# Create output directory
output_dir = 'Phase_3_Validation_and_XAI/01_Test_Benchmarking/results/ppt_figures'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# FIGURE 1: Training vs Testing Accuracy Comparison (IMPROVED)
# =============================================================================
def create_accuracy_comparison():
    """Bar chart comparing training and testing accuracy - NO OVERLAP"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    training = [95.54, 96.27, 94.74, 95.50, 99.13]
    testing = [96.59, 87.03, 95.20, 90.93, 99.53]
    
    x = np.arange(len(metrics))
    width = 0.30  # Slightly narrower bars for more spacing
    
    bars1 = ax.bar(x - width/2, training, width, label='Training (8,178 samples)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, testing, width, label='Testing (68,264 samples)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars - POSITIONED ABOVE with more space
    for bar, val in zip(bars1, training):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, testing):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=12)
    ax.set_title('Training vs External Testing Performance\nDoS Detection Model Validation', 
                 fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold', fontsize=11)
    
    # Legend at BOTTOM to avoid overlap with bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=2, fontsize=12, framealpha=0.9)
    
    ax.set_ylim(0, 115)  # Extra space at top for labels
    
    # Adjust bottom margin for legend
    plt.subplots_adjust(bottom=0.18)
    
    plt.savefig(f'{output_dir}/01_training_vs_testing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 01_training_vs_testing_comparison.png")

# =============================================================================
# FIGURE 2: Confusion Matrix Heatmap (IMPROVED)
# =============================================================================
def create_confusion_matrix():
    """Professional confusion matrix visualization - CLEAN LAYOUT"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Confusion matrix values
    cm = np.array([[54260, 1740],
                   [589, 11675]])
    
    # Create heatmap with better formatting
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', 
                xticklabels=['Predicted\nNormal', 'Predicted\nDoS'],
                yticklabels=['Actual\nNormal', 'Actual\nDoS'],
                annot_kws={'size': 18, 'weight': 'bold'},
                linewidths=3, linecolor='white',
                cbar_kws={'label': 'Number of Samples', 'shrink': 0.8})
    
    ax.set_xlabel('', fontweight='bold', fontsize=13)
    ax.set_ylabel('', fontweight='bold', fontsize=13)
    ax.set_title('Confusion Matrix - External Test Dataset\n(68,264 Total Samples)', 
                 fontweight='bold', fontsize=15, pad=20)
    
    # Rotate labels for better readability
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 02_confusion_matrix.png")

# =============================================================================
# FIGURE 3: Detection Rate Pie Charts (IMPROVED)
# =============================================================================
def create_detection_pie():
    """Pie charts showing detection rates - CLEAN LAYOUT"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== LEFT PIE: DoS Detection =====
    dos_detected = 11675
    dos_missed = 589
    total_dos = dos_detected + dos_missed
    
    colors1 = ['#27ae60', '#e74c3c']
    wedges1, texts1, autotexts1 = ax1.pie(
        [dos_detected, dos_missed], 
        labels=None,  # Remove labels from pie
        colors=colors1, 
        autopct='%1.1f%%', 
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        explode=(0.02, 0.02),
        shadow=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Set percentage text color to white for visibility
    for autotext in autotexts1:
        autotext.set_color('white')
    
    ax1.set_title('DoS Attack Detection Rate', fontweight='bold', fontsize=14, pad=15)
    
    # Add legend instead of labels on pie
    ax1.legend(wedges1, [f'Detected: {dos_detected:,}', f'Missed: {dos_missed:,}'], 
               loc='lower center', fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.1))
    
    # Add total below
    ax1.text(0.5, -0.25, f'Total DoS Attacks: {total_dos:,}', 
             transform=ax1.transAxes, ha='center', fontsize=11, style='italic')
    
    # ===== RIGHT PIE: Normal Traffic =====
    normal_correct = 54260
    normal_false = 1740
    total_normal = normal_correct + normal_false
    
    colors2 = ['#3498db', '#e74c3c']
    wedges2, texts2, autotexts2 = ax2.pie(
        [normal_correct, normal_false], 
        labels=None,  # Remove labels from pie
        colors=colors2, 
        autopct='%1.1f%%', 
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        explode=(0.02, 0.02),
        shadow=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    for autotext in autotexts2:
        autotext.set_color('white')
    
    ax2.set_title('Normal Traffic Classification', fontweight='bold', fontsize=14, pad=15)
    
    ax2.legend(wedges2, [f'Correct: {normal_correct:,}', f'False Alarm: {normal_false:,}'], 
               loc='lower center', fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.1))
    
    ax2.text(0.5, -0.25, f'Total Normal Traffic: {total_normal:,}', 
             transform=ax2.transAxes, ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legends
    plt.savefig(f'{output_dir}/03_detection_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 03_detection_rates.png")

# =============================================================================
# FIGURE 7: Dataset Comparison (IMPROVED)
# =============================================================================
def create_dataset_comparison():
    """Shows training vs testing dataset sizes - CLEAN LAYOUT"""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Data
    labels = ['Training Dataset', 'External Test Dataset']
    dos_samples = [4089, 12264]
    normal_samples = [4089, 56000]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dos_samples, width, label='DoS Attacks', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, normal_samples, width, label='Normal Traffic', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    
    # Add values ABOVE bars with good spacing
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1500, 
                f'{int(height):,}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#c0392b')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1500, 
                f'{int(height):,}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#2980b9')
    
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax.set_title('Dataset Composition: Training vs External Testing', 
                 fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Set y-axis limit with room for labels
    ax.set_ylim(0, 65000)
    
    # Add total annotations at bottom using text box
    props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8)
    ax.text(0, -8000, f'Total: 8,178\n(Balanced 50:50)', ha='center', fontsize=10, 
            fontweight='bold', bbox=props)
    ax.text(1, -8000, f'Total: 68,264\n(Realistic 82:18)', ha='center', fontsize=10, 
            fontweight='bold', bbox=props)
    
    # Extend bottom margin for annotations
    ax.set_ylim(-12000, 65000)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 07_dataset_comparison.png")

# =============================================================================
# Generate Selected Figures (1, 2, 3, 7)
# =============================================================================
def main():
    print("=" * 60)
    print("Generating IMPROVED PPT Visualizations (No Overlapping)")
    print("=" * 60)
    
    create_accuracy_comparison()
    create_confusion_matrix()
    create_detection_pie()
    create_dataset_comparison()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)
    print("\nGenerated Files (IMPROVED):")
    print("   1. 01_training_vs_testing_comparison.png")
    print("   2. 02_confusion_matrix.png")
    print("   3. 03_detection_rates.png")
    print("   7. 07_dataset_comparison.png")
    print("\nReady for your PPT!")

if __name__ == "__main__":
    main()
