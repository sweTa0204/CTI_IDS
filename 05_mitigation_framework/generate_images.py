"""
Generate Images for Research Paper
===================================

This script generates publication-quality images for the paper:
1. SHAP Summary Plot (global feature importance)
2. SHAP Waterfall Plot - DoS Example
3. SHAP Waterfall Plot - Normal Example
4. Attack Type Distribution from Benchmark Results

Author: Research Project
Date: 2026-01-30
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '04_xai_integration'))

import shap
from shap_explainer import SHAPExplainer

# Paths
MODEL_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                          'models', 'xgboost', 'xgboost_model.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                               'data', 'X_test_scaled.csv')
TEST_LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                                 'data', 'y_test.csv')
# Output directories - images go to their respective module folders
XAI_IMAGES_DIR = os.path.join(BASE_DIR, '04_xai_integration', 'images')
MITIGATION_IMAGES_DIR = os.path.join(BASE_DIR, '05_mitigation_framework', 'images')
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results.json')

# Feature names
FEATURE_NAMES = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']


def generate_shap_summary_plot():
    """Generate SHAP summary plot showing global feature importance."""

    print("\n[1/4] Generating SHAP Summary Plot...")

    # Initialize explainer
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()

    # Load test data
    X_test = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)

    # Sample for SHAP (use 500 samples for speed)
    np.random.seed(42)
    n_samples = min(500, len(X_test))
    sample_idx = np.random.choice(len(X_test), size=n_samples, replace=False)
    X_sample = X_test.iloc[sample_idx]

    # Calculate SHAP values
    print("  Calculating SHAP values for 500 samples...")
    shap_values = explainer.explainer.shap_values(X_sample)

    # For binary classification, use class 1 (DoS) SHAP values
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Create figure
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=FEATURE_NAMES, show=False)
    plt.title('SHAP Feature Importance for DoS Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = os.path.join(XAI_IMAGES_DIR, '07_shap_summary_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def generate_shap_waterfall_dos():
    """Generate SHAP waterfall plot for a DoS detection example."""

    print("\n[2/4] Generating SHAP Waterfall Plot (DoS)...")

    # Initialize explainer
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()

    # Load test data
    X_test = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)

    # Find a high-confidence DoS prediction
    dos_indices = y_test[y_test.iloc[:, 0] == 1].index.tolist()
    np.random.seed(42)

    # Get predictions and find high confidence DoS
    best_idx = None
    best_conf = 0
    for idx in dos_indices[:100]:  # Check first 100
        features = X_test.iloc[idx].values.reshape(1, -1)
        pred_proba = explainer.model.predict_proba(features)[0][1]
        if pred_proba > best_conf:
            best_conf = pred_proba
            best_idx = idx

    print(f"  Selected DoS sample with {best_conf*100:.1f}% confidence")

    # Get SHAP values for this sample
    features = X_test.iloc[best_idx].values.reshape(1, -1)
    shap_values = explainer.explainer.shap_values(features)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Get base value
    if isinstance(explainer.explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.explainer.expected_value[1]
    else:
        base_value = explainer.explainer.expected_value

    # Create waterfall plot manually
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    sorted_features = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_shap = sv[sorted_idx]
    sorted_values = features[0][sorted_idx]

    # Create horizontal bar chart
    colors = ['#ff0d57' if s > 0 else '#1e88e5' for s in sorted_shap]
    y_pos = np.arange(len(sorted_features))

    bars = ax.barh(y_pos, sorted_shap, color=colors, edgecolor='black', linewidth=0.5)

    # Add feature names and values
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{f} = {sorted_values[i]:.3f}" for i, f in enumerate(sorted_features)])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_shap)):
        x_pos = bar.get_width()
        ha = 'left' if val >= 0 else 'right'
        offset = 0.02 if val >= 0 else -0.02
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=9)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP Value (impact on DoS prediction)', fontsize=11)
    ax.set_title(f'SHAP Explanation for DoS Detection\n(Prediction: DoS, Confidence: {best_conf*100:.1f}%)',
                 fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff0d57', label='Increases DoS likelihood'),
                       Patch(facecolor='#1e88e5', label='Decreases DoS likelihood')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    # Save
    output_path = os.path.join(XAI_IMAGES_DIR, '08_shap_waterfall_dos.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def generate_shap_waterfall_normal():
    """Generate SHAP waterfall plot for a Normal traffic example."""

    print("\n[3/4] Generating SHAP Waterfall Plot (Normal)...")

    # Initialize explainer
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()

    # Load test data
    X_test = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)

    # Find a high-confidence Normal prediction
    normal_indices = y_test[y_test.iloc[:, 0] == 0].index.tolist()
    np.random.seed(42)

    # Get predictions and find high confidence Normal
    best_idx = None
    best_conf = 0
    for idx in normal_indices[:100]:  # Check first 100
        features = X_test.iloc[idx].values.reshape(1, -1)
        pred_proba = explainer.model.predict_proba(features)[0][0]  # Class 0 probability
        if pred_proba > best_conf:
            best_conf = pred_proba
            best_idx = idx

    print(f"  Selected Normal sample with {best_conf*100:.1f}% confidence")

    # Get SHAP values for this sample
    features = X_test.iloc[best_idx].values.reshape(1, -1)
    shap_values = explainer.explainer.shap_values(features)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Still use class 1 SHAP values (but they'll be negative)
    else:
        sv = shap_values[0]

    # Create waterfall plot manually
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    sorted_features = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_shap = sv[sorted_idx]
    sorted_values = features[0][sorted_idx]

    # Create horizontal bar chart
    colors = ['#ff0d57' if s > 0 else '#1e88e5' for s in sorted_shap]
    y_pos = np.arange(len(sorted_features))

    bars = ax.barh(y_pos, sorted_shap, color=colors, edgecolor='black', linewidth=0.5)

    # Add feature names and values
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{f} = {sorted_values[i]:.3f}" for i, f in enumerate(sorted_features)])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_shap)):
        x_pos = bar.get_width()
        ha = 'left' if val >= 0 else 'right'
        offset = 0.02 if val >= 0 else -0.02
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=9)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP Value (impact on DoS prediction)', fontsize=11)
    ax.set_title(f'SHAP Explanation for Normal Traffic\n(Prediction: Normal, Confidence: {best_conf*100:.1f}%)',
                 fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff0d57', label='Increases DoS likelihood'),
                       Patch(facecolor='#1e88e5', label='Decreases DoS likelihood')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    # Save
    output_path = os.path.join(XAI_IMAGES_DIR, '09_shap_waterfall_normal.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def generate_attack_type_distribution():
    """Generate attack type distribution from benchmark results."""

    print("\n[4/4] Generating Attack Type Distribution...")

    # Load test results
    if not os.path.exists(RESULTS_PATH):
        print(f"  [ERROR] Results file not found: {RESULTS_PATH}")
        return None

    with open(RESULTS_PATH, 'r') as f:
        results = json.load(f)

    # Handle different formats (list or dict)
    if isinstance(results, list):
        alerts = results
    else:
        alerts = results.get('mitigations', results)

    # Count attack types
    attack_counts = {}
    for alert in alerts:
        if isinstance(alert, dict):
            attack_type = alert.get('classification', {}).get('attack_type')
            if attack_type and attack_type != 'None':
                attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

    if not attack_counts:
        print("  [WARNING] No attack types found in results")
        return None

    print(f"  Attack types found: {attack_counts}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    labels = list(attack_counts.keys())
    sizes = list(attack_counts.values())
    total = sum(sizes)
    percentages = [s/total*100 for s in sizes]

    # Colors
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6'][:len(labels)]

    # Create bar chart
    bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, count, pct in zip(bars, sizes, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Number of Detections', fontsize=12)
    ax.set_title('DoS Attack Type Distribution (Benchmark Test Data)',
                 fontsize=14, fontweight='bold')

    # Add total annotation
    ax.text(0.98, 0.98, f'Total DoS Detections: {total}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    # Save
    output_path = os.path.join(MITIGATION_IMAGES_DIR, '10_attack_type_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def main():
    """Generate all images."""

    print("=" * 70)
    print("GENERATING IMAGES FOR RESEARCH PAPER")
    print("=" * 70)

    # Ensure output directories exist
    os.makedirs(XAI_IMAGES_DIR, exist_ok=True)
    os.makedirs(MITIGATION_IMAGES_DIR, exist_ok=True)

    # Generate all images
    images = []

    try:
        img1 = generate_shap_summary_plot()
        if img1:
            images.append(img1)
    except Exception as e:
        print(f"  [ERROR] SHAP summary plot failed: {e}")

    try:
        img2 = generate_shap_waterfall_dos()
        if img2:
            images.append(img2)
    except Exception as e:
        print(f"  [ERROR] DoS waterfall plot failed: {e}")

    try:
        img3 = generate_shap_waterfall_normal()
        if img3:
            images.append(img3)
    except Exception as e:
        print(f"  [ERROR] Normal waterfall plot failed: {e}")

    try:
        img4 = generate_attack_type_distribution()
        if img4:
            images.append(img4)
    except Exception as e:
        print(f"  [ERROR] Attack distribution plot failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("IMAGE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(images)} images:")
    for img in images:
        print(f"  - {os.path.basename(img)}")

    print(f"\nOutput directories:")
    print(f"  - XAI images: {XAI_IMAGES_DIR}")
    print(f"  - Mitigation images: {MITIGATION_IMAGES_DIR}")

    return images


if __name__ == "__main__":
    main()
