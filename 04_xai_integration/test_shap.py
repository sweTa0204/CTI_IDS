"""
Test Script for SHAP Explainer
==============================

This script tests the SHAP explainer on sample data to verify it works correctly.

Run this script to:
1. Load the XGBoost model
2. Initialize SHAP TreeExplainer
3. Test on a few sample records
4. Display the explanations

Author: Research Project
Date: 2026-01-29
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shap_explainer import SHAPExplainer, format_explanation_for_display

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                          'models', 'xgboost', 'xgboost_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                         'data', 'X_train_scaled.csv')
LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                           'data', 'y_train.csv')


def test_shap_explainer():
    """Main test function for SHAP explainer."""

    print("=" * 70)
    print("SHAP EXPLAINER TEST")
    print("=" * 70)

    # Step 1: Initialize explainer
    print("\n[Step 1] Initializing SHAP Explainer...")
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()
    print("[OK] SHAP Explainer initialized successfully!")

    # Step 2: Load sample data
    print("\n[Step 2] Loading sample data...")
    X_train = pd.read_csv(DATA_PATH)
    y_train = pd.read_csv(LABELS_PATH)

    print(f"[OK] Loaded {len(X_train)} training samples")
    print(f"  - Normal: {sum(y_train.values.ravel() == 0)}")
    print(f"  - DoS: {sum(y_train.values.ravel() == 1)}")

    # Step 3: Select sample records (mix of DoS and Normal)
    print("\n[Step 3] Selecting sample records for testing...")

    # Get indices of DoS and Normal samples
    dos_indices = y_train[y_train.iloc[:, 0] == 1].index.tolist()
    normal_indices = y_train[y_train.iloc[:, 0] == 0].index.tolist()

    # Select 3 DoS and 2 Normal samples
    np.random.seed(42)
    sample_dos = np.random.choice(dos_indices, size=3, replace=False)
    sample_normal = np.random.choice(normal_indices, size=2, replace=False)
    sample_indices = list(sample_dos) + list(sample_normal)

    print(f"[OK] Selected {len(sample_indices)} samples:")
    print(f"  - DoS samples (indices): {list(sample_dos)}")
    print(f"  - Normal samples (indices): {list(sample_normal)}")

    # Step 4: Generate explanations
    print("\n[Step 4] Generating SHAP explanations...")

    explanations = []
    for idx in sample_indices:
        features = X_train.iloc[idx].values
        actual_label = "DoS" if y_train.iloc[idx, 0] == 1 else "Normal"
        explanation = explainer.explain_single(features, record_id=int(idx))
        explanation['actual_label'] = actual_label
        explanations.append(explanation)

    print(f"[OK] Generated {len(explanations)} explanations")

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("SHAP EXPLANATION RESULTS")
    print("=" * 70)

    for i, exp in enumerate(explanations):
        print(f"\n--- Sample {i+1} ---")
        print(f"Record ID:        {exp['record_id']}")
        print(f"Actual Label:     {exp['actual_label']}")
        print(f"Model Prediction: {exp['prediction']}")
        print(f"Confidence:       {exp['confidence']*100:.2f}%")
        print(f"P(DoS):           {exp['probability_dos']:.4f}")
        print(f"P(Normal):        {exp['probability_normal']:.4f}")
        print(f"\nTop 3 Contributing Features:")

        # Sort SHAP values by absolute value
        sorted_shap = sorted(exp['shap_values'].items(),
                             key=lambda x: abs(x[1]),
                             reverse=True)[:3]

        for j, (feature, shap_val) in enumerate(sorted_shap):
            feature_val = exp['feature_values'][feature]
            sign = "+" if shap_val >= 0 else ""
            direction = "-> DoS" if shap_val > 0 else "-> Normal"
            print(f"  {j+1}. {feature:10s}: {sign}{shap_val:.4f} {direction}")
            print(f"     (actual value: {feature_val:.4f})")

        # Check if prediction matches actual
        match = "[OK] CORRECT" if exp['prediction'] == exp['actual_label'] else "[X] INCORRECT"
        print(f"\nPrediction Match: {match}")

    # Step 6: Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = explainer.get_summary_stats(explanations)
    print(f"\nTotal samples tested:     {summary['total_records']}")
    print(f"DoS predictions:          {summary['dos_detections']}")
    print(f"Normal predictions:       {summary['normal_detections']}")

    if summary.get('top_features_frequency'):
        print(f"\nMost frequent top features (in DoS detections):")
        for feature, count in summary['top_features_frequency'].items():
            print(f"  - {feature}: appeared in top 3 for {count} samples")

    # Step 7: Save sample output
    print("\n[Step 6] Saving sample output...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'sample_shap_output.json')

    # Convert for JSON serialization
    output_data = {
        "test_info": {
            "samples_tested": len(explanations),
            "model_path": MODEL_PATH,
            "data_source": "Training data (X_train_scaled.csv)"
        },
        "explanations": explanations,
        "summary": summary
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Output saved to: {output_path}")

    # Final status
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the explanations above")
    print("2. Verify SHAP values make sense")
    print("3. If approved, proceed to Step 2 (Attack Classification)")

    return explanations, summary


if __name__ == "__main__":
    explanations, summary = test_shap_explainer()
