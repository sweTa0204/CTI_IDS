"""
Test Script for Attack Classifier
==================================

This script tests the attack classifier using SHAP explanations from the SHAP explainer.

Run this script to:
1. Load SHAP explanations (from sample_shap_output.json or generate new ones)
2. Classify each DoS detection into attack types
3. Display classification results with reasoning

Author: Research Project
Date: 2026-01-29
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack_classifier import AttackClassifier, get_attack_statistics

# Also import SHAP explainer to generate fresh explanations if needed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '04_xai_integration'))
from shap_explainer import SHAPExplainer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_OUTPUT_PATH = os.path.join(BASE_DIR, '04_xai_integration', 'sample_shap_output.json')
MODEL_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                          'models', 'xgboost', 'xgboost_model.pkl')


def test_with_sample_data():
    """Test attack classifier with pre-generated SHAP explanations."""

    print("=" * 70)
    print("ATTACK CLASSIFIER TEST")
    print("=" * 70)

    # Step 1: Load SHAP explanations
    print("\n[Step 1] Loading SHAP explanations...")

    if os.path.exists(SHAP_OUTPUT_PATH):
        with open(SHAP_OUTPUT_PATH, 'r') as f:
            shap_data = json.load(f)
        explanations = shap_data.get('explanations', [])
        print(f"[OK] Loaded {len(explanations)} explanations from sample_shap_output.json")
    else:
        print("[!] sample_shap_output.json not found. Generating new explanations...")
        explanations = generate_fresh_explanations()

    # Step 2: Initialize Attack Classifier
    print("\n[Step 2] Initializing Attack Classifier...")
    classifier = AttackClassifier()
    print("[OK] Attack Classifier initialized!")
    print(f"     Supported attack types: {list(classifier.attack_types.keys())}")

    # Step 3: Classify each DoS detection
    print("\n[Step 3] Classifying attack types...")

    classifications = []
    for exp in explanations:
        classification = classifier.classify(exp)
        classification['record_id'] = exp.get('record_id')
        classification['original_prediction'] = exp.get('prediction')
        classification['original_confidence'] = exp.get('confidence')
        classifications.append(classification)

    print(f"[OK] Classified {len(classifications)} records")

    # Step 4: Display results
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    for i, result in enumerate(classifications):
        print(f"\n--- Record {result['record_id']} ---")
        print(f"Original Prediction:  {result['original_prediction']}")
        print(f"Original Confidence:  {result['original_confidence']*100:.2f}%")
        print(f"\nAttack Classification:")
        print(f"  Type:        {result['attack_type']}")
        print(f"  Description: {result['attack_description']}")
        print(f"  Confidence:  {result['confidence']*100:.2f}%")
        print(f"  Category:    {result['mitigation_category']}")
        print(f"\nPrimary Indicators: {result['primary_indicators']}")
        print(f"Reasoning: {result['reasoning']}")

        if result.get('all_scores'):
            print(f"\nAll Attack Type Scores:")
            for attack_type, score in sorted(result['all_scores'].items(),
                                              key=lambda x: x[1],
                                              reverse=True):
                bar = "#" * int(score * 20)
                print(f"  {attack_type:18s}: {score:.4f} {bar}")

    # Step 5: Summary statistics
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)

    # Filter only DoS classifications (not None)
    dos_classifications = [c for c in classifications if c['attack_type'] != 'None']

    if dos_classifications:
        stats = get_attack_statistics(dos_classifications)

        print(f"\nTotal DoS detections classified: {stats['total_classifications']}")
        print(f"\nAttack Type Distribution:")
        for attack_type, count in stats['attack_type_counts'].items():
            percentage = stats['attack_type_percentages'][attack_type]
            bar = "#" * int(percentage / 5)
            print(f"  {attack_type:18s}: {count} ({percentage:.1f}%) {bar}")

        print(f"\nMost Common Attack Type: {stats['most_common_type']}")
        print(f"  (Appeared {stats['most_common_count']} times)")
    else:
        print("\nNo DoS detections to classify.")

    # Step 6: Save output
    print("\n[Step 6] Saving classification results...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'sample_classification_output.json')

    output_data = {
        "test_info": {
            "samples_classified": len(classifications),
            "dos_detections": len(dos_classifications),
            "attack_types": list(classifier.attack_types.keys())
        },
        "classifications": classifications,
        "statistics": get_attack_statistics(dos_classifications) if dos_classifications else None
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Output saved to: {output_path}")

    # Final status
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the attack classifications above")
    print("2. Verify reasoning makes sense")
    print("3. If approved, proceed to Step 3 (Severity Calculator)")

    return classifications


def generate_fresh_explanations():
    """Generate fresh SHAP explanations if sample file not found."""
    import pandas as pd
    import numpy as np

    DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                             'data', 'X_train_scaled.csv')
    LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                               'data', 'y_train.csv')

    # Initialize explainer
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()

    # Load data
    X_train = pd.read_csv(DATA_PATH)
    y_train = pd.read_csv(LABELS_PATH)

    # Get sample indices
    dos_indices = y_train[y_train.iloc[:, 0] == 1].index.tolist()
    normal_indices = y_train[y_train.iloc[:, 0] == 0].index.tolist()

    np.random.seed(42)
    sample_dos = np.random.choice(dos_indices, size=3, replace=False)
    sample_normal = np.random.choice(normal_indices, size=2, replace=False)
    sample_indices = list(sample_dos) + list(sample_normal)

    # Generate explanations
    explanations = []
    for idx in sample_indices:
        features = X_train.iloc[idx].values
        explanation = explainer.explain_single(features, record_id=int(idx))
        explanations.append(explanation)

    print(f"[OK] Generated {len(explanations)} fresh explanations")
    return explanations


def test_with_synthetic_data():
    """Test classifier with synthetic SHAP explanations to verify all attack types."""

    print("\n" + "=" * 70)
    print("SYNTHETIC DATA TEST (All Attack Types)")
    print("=" * 70)

    classifier = AttackClassifier()

    # Create synthetic explanations for each attack type
    synthetic_tests = [
        {
            "name": "Volumetric Flood Pattern",
            "shap_explanation": {
                "prediction": "DoS",
                "confidence": 0.95,
                "shap_values": {
                    "rate": 0.8, "sload": 0.6, "sbytes": 0.4,
                    "dload": 0.1, "proto": 0.05, "dtcpb": 0.02,
                    "stcpb": 0.01, "dmean": 0.03, "tcprtt": 0.01, "dur": 0.02
                },
                "top_features": ["rate", "sload", "sbytes"],
                "feature_values": {
                    "rate": 1500.0, "sload": 900000.0, "sbytes": 50000.0,
                    "dload": 100.0, "proto": 6, "dtcpb": 0, "stcpb": 0,
                    "dmean": 50, "tcprtt": 0.01, "dur": 0.5
                }
            },
            "expected_type": "Volumetric Flood"
        },
        {
            "name": "Protocol Exploit Pattern",
            "shap_explanation": {
                "prediction": "DoS",
                "confidence": 0.88,
                "shap_values": {
                    "rate": 0.1, "sload": 0.1, "sbytes": 0.05,
                    "dload": 0.05, "proto": 2.5, "dtcpb": 0.4,
                    "stcpb": 0.3, "dmean": 0.02, "tcprtt": 0.01, "dur": 0.02
                },
                "top_features": ["proto", "dtcpb", "stcpb"],
                "feature_values": {
                    "rate": 100.0, "sload": 5000.0, "sbytes": 500.0,
                    "dload": 100.0, "proto": 1, "dtcpb": 12345, "stcpb": 54321,
                    "dmean": 50, "tcprtt": 0.01, "dur": 0.1
                }
            },
            "expected_type": "Protocol Exploit"
        },
        {
            "name": "Slowloris Pattern",
            "shap_explanation": {
                "prediction": "DoS",
                "confidence": 0.82,
                "shap_values": {
                    "rate": -0.1, "sload": 0.05, "sbytes": 0.3,
                    "dload": 0.02, "proto": 0.05, "dtcpb": 0.01,
                    "stcpb": 0.01, "dmean": 0.02, "tcprtt": 0.05, "dur": 0.6
                },
                "top_features": ["dur", "sbytes", "rate"],
                "feature_values": {
                    "rate": 2.0, "sload": 100.0, "sbytes": 5000.0,
                    "dload": 50.0, "proto": 6, "dtcpb": 0, "stcpb": 0,
                    "dmean": 50, "tcprtt": 0.5, "dur": 300.0
                }
            },
            "expected_type": "Slowloris"
        },
        {
            "name": "Amplification Pattern",
            "shap_explanation": {
                "prediction": "DoS",
                "confidence": 0.91,
                "shap_values": {
                    "rate": 0.2, "sload": 0.1, "sbytes": 0.1,
                    "dload": 0.7, "proto": 0.15, "dtcpb": 0.01,
                    "stcpb": 0.01, "dmean": 0.02, "tcprtt": 0.01, "dur": 0.02
                },
                "top_features": ["dload", "rate", "proto"],
                "feature_values": {
                    "rate": 500.0, "sload": 1000.0, "sbytes": 100.0,
                    "dload": 50000.0, "proto": 17, "dtcpb": 0, "stcpb": 0,
                    "dmean": 1000, "tcprtt": 0.01, "dur": 0.1
                }
            },
            "expected_type": "Amplification"
        },
        {
            "name": "Normal Traffic (No Attack)",
            "shap_explanation": {
                "prediction": "Normal",
                "confidence": 0.95,
                "shap_values": {
                    "rate": -0.2, "sload": -0.15, "sbytes": -0.1,
                    "dload": -0.05, "proto": -0.05, "dtcpb": -0.02,
                    "stcpb": -0.02, "dmean": -0.01, "tcprtt": -0.01, "dur": -0.02
                },
                "top_features": ["rate", "sload", "sbytes"],
                "feature_values": {
                    "rate": 10.0, "sload": 500.0, "sbytes": 200.0,
                    "dload": 300.0, "proto": 6, "dtcpb": 0, "stcpb": 0,
                    "dmean": 50, "tcprtt": 0.01, "dur": 0.05
                }
            },
            "expected_type": "None"
        }
    ]

    print("\nTesting all attack type patterns...\n")

    all_passed = True
    for test in synthetic_tests:
        result = classifier.classify(test["shap_explanation"])
        passed = result["attack_type"] == test["expected_type"]
        status = "[OK]" if passed else "[X]"

        if not passed:
            all_passed = False

        print(f"{status} {test['name']}")
        print(f"    Expected: {test['expected_type']}")
        print(f"    Got:      {result['attack_type']} (confidence: {result['confidence']:.2f})")
        if result.get('all_scores'):
            top_score = max(result['all_scores'].items(), key=lambda x: x[1])
            print(f"    Top score: {top_score[0]} = {top_score[1]:.4f}")
        print()

    print("-" * 70)
    if all_passed:
        print("[OK] All synthetic tests PASSED!")
    else:
        print("[!] Some tests failed - classifier may need tuning")

    return all_passed


if __name__ == "__main__":
    # Run main test with real/sample data
    classifications = test_with_sample_data()

    # Run synthetic data test
    test_with_synthetic_data()
