"""
Demo: Single Sample Through Complete Pipeline
==============================================

This script demonstrates the complete XAI-powered DoS detection
and mitigation pipeline using a single sample.

Perfect for presentations - shows each step clearly with explanations.

Usage:
    python demo_single_sample.py

Author: Research Project
Date: 2026-01-30
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '04_xai_integration'))
sys.path.insert(0, os.path.join(BASE_DIR, '05_mitigation_framework'))

# Import components
from shap_explainer import SHAPExplainer
from attack_classifier import AttackClassifier
from severity_calculator import SeverityCalculator
from mitigation_generator import MitigationGenerator

# Paths
MODEL_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                          'models', 'xgboost', 'xgboost_model.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                               'data', 'X_test_scaled.csv')
TEST_LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                                 'data', 'y_test.csv')

# Constants
OPTIMIZED_THRESHOLD = 0.8517
FEATURE_NAMES = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print a section divider."""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def demo_pipeline(sample_index=None, find_dos=True):
    """
    Demonstrate the complete pipeline with a single sample.

    Args:
        sample_index: Specific sample index to use (None = auto-select)
        find_dos: If True and no index given, find a DoS sample
    """

    print_header("XAI-POWERED DoS DETECTION & MITIGATION DEMO")
    print("\n  This demo shows ONE sample flowing through the complete pipeline:")
    print("  INPUT -> DETECT -> EXPLAIN -> CLASSIFY -> ASSESS -> MITIGATE")

    # =========================================================================
    # STEP 0: LOAD COMPONENTS
    # =========================================================================
    print_section("STEP 0: Loading Components")

    print("\n  Loading XGBoost model...")
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()
    print("  [OK] XGBoost model loaded")

    print("\n  Loading Attack Classifier...")
    classifier = AttackClassifier()
    print("  [OK] Attack Classifier loaded")

    print("\n  Loading Severity Calculator...")
    severity_calc = SeverityCalculator()
    print("  [OK] Severity Calculator loaded")

    print("\n  Loading Mitigation Generator...")
    mitigation_gen = MitigationGenerator()
    print("  [OK] Mitigation Generator loaded")

    # =========================================================================
    # STEP 1: SELECT SAMPLE
    # =========================================================================
    print_section("STEP 1: DATA INPUT")

    # Load test data
    X_test = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)

    # Select sample
    if sample_index is None:
        if find_dos:
            # Find a DoS sample with high probability
            dos_indices = y_test[y_test.iloc[:, 0] == 1].index.tolist()
            sample_index = dos_indices[42]  # Pick a specific one for consistency
        else:
            sample_index = 100  # Normal sample

    # Get sample data
    features = X_test.iloc[sample_index].values
    actual_label = "DoS" if y_test.iloc[sample_index, 0] == 1 else "Normal"

    print(f"\n  Sample Index: {sample_index}")
    print(f"  Actual Label: {actual_label}")
    print(f"\n  Input Features (10 features):")
    print("  " + "-" * 50)
    for i, (name, value) in enumerate(zip(FEATURE_NAMES, features)):
        print(f"    {i+1}. {name:8s} = {value:>12.4f}")
    print("  " + "-" * 50)

    # =========================================================================
    # STEP 2: XGBoost DETECTION
    # =========================================================================
    print_section("STEP 2: DoS DETECTION (XGBoost)")

    start_time = time.time()

    # Get prediction
    features_array = np.array(features).reshape(1, -1)
    prediction_proba = explainer.model.predict_proba(features_array)[0]
    prob_normal = prediction_proba[0]
    prob_dos = prediction_proba[1]

    # Apply threshold
    prediction = "DoS" if prob_dos >= OPTIMIZED_THRESHOLD else "Normal"

    print(f"\n  Model Output:")
    print(f"    P(Normal) = {prob_normal:.4f} ({prob_normal*100:.2f}%)")
    print(f"    P(DoS)    = {prob_dos:.4f} ({prob_dos*100:.2f}%)")
    print(f"\n  Threshold: {OPTIMIZED_THRESHOLD} ({OPTIMIZED_THRESHOLD*100:.2f}%)")
    print(f"\n  Decision: {prob_dos:.4f} {'>=':^3} {OPTIMIZED_THRESHOLD}")

    if prediction == "DoS":
        print(f"\n  >>> RESULT: DoS ATTACK DETECTED (Confidence: {prob_dos*100:.2f}%)")
    else:
        print(f"\n  >>> RESULT: Normal Traffic (Confidence: {prob_normal*100:.2f}%)")
        print("\n  [Pipeline ends here for normal traffic]")
        return

    # =========================================================================
    # STEP 3: SHAP EXPLAINABILITY
    # =========================================================================
    print_section("STEP 3: EXPLAINABILITY (SHAP TreeExplainer)")

    # Calculate SHAP values
    shap_values = explainer.explainer.shap_values(features_array)

    # Handle binary classification output
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Class 1 (DoS) SHAP values
    else:
        sv = shap_values[0]

    # Create SHAP dictionary
    shap_dict = {name: float(sv[i]) for i, name in enumerate(FEATURE_NAMES)}

    # Sort by absolute value
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n  SHAP Values (Feature Contributions to DoS Prediction):")
    print("  " + "-" * 50)
    print(f"  {'Feature':<10} {'SHAP Value':>12} {'Direction':>15}")
    print("  " + "-" * 50)
    for name, value in sorted_shap:
        direction = "-> DoS" if value > 0 else "-> Normal"
        bar = "*" * min(int(abs(value) * 5), 20)
        print(f"    {name:<10} {value:>+12.4f}  {direction:<10} {bar}")
    print("  " + "-" * 50)

    # Top features
    top_features = [f[0] for f in sorted_shap[:3]]
    print(f"\n  TOP 3 CONTRIBUTING FEATURES: {', '.join(top_features)}")
    print(f"\n  EXPLANATION: This traffic was flagged as DoS primarily because of")
    print(f"               high {top_features[0]} and {top_features[1]} values.")

    # =========================================================================
    # STEP 4: ATTACK CLASSIFICATION
    # =========================================================================
    print_section("STEP 4: ATTACK CLASSIFICATION")

    # Build SHAP explanation dict for classifier
    shap_explanation = {
        'prediction': prediction,
        'confidence': prob_dos,
        'shap_values': shap_dict,
        'top_features': top_features,
        'feature_values': {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)}
    }

    # Classify attack based on SHAP explanation
    classification = classifier.classify(shap_explanation)
    attack_type = classification.get('attack_type', 'Unknown')

    print(f"\n  Based on top contributing features: {top_features}")
    print(f"\n  Classification Rules:")
    print("    - rate, sbytes, sload  -> VOLUMETRIC FLOOD")
    print("    - proto, tcprtt, stcpb -> PROTOCOL EXPLOIT")
    print("    - dur, dmean           -> SLOWLORIS")
    print("    - dload, dbytes        -> AMPLIFICATION")

    print(f"\n  >>> ATTACK TYPE: {attack_type}")
    print(f"\n  Description: {classification.get('description', 'N/A')}")

    # =========================================================================
    # STEP 5: SEVERITY ASSESSMENT
    # =========================================================================
    print_section("STEP 5: SEVERITY ASSESSMENT")

    confidence = prob_dos
    severity_result = severity_calc.calculate(classification, shap_explanation)
    severity_level = severity_result.get('level', 'UNKNOWN')

    print(f"\n  Model Confidence: {confidence*100:.2f}%")
    print(f"\n  Severity Thresholds:")
    print("    - >= 95%  -> CRITICAL (Immediate action required)")
    print("    - 90-95%  -> HIGH (Priority response)")
    print("    - 75-90%  -> MEDIUM (Monitor closely)")
    print("    - 60-75%  -> LOW (Log and observe)")

    print(f"\n  >>> SEVERITY LEVEL: {severity_level}")
    print(f"\n  Escalation Required: {severity_result.get('escalation_required', False)}")

    # =========================================================================
    # STEP 6: MITIGATION GENERATION
    # =========================================================================
    print_section("STEP 6: MITIGATION GENERATION")

    source_ip = "192.168.1.100"  # Example attacker IP
    dest_ip = "10.0.0.1"  # Example target IP
    interface = "eth0"

    mitigation = mitigation_gen.generate(
        classification=classification,
        severity=severity_result,
        source_ip=source_ip,
        interface=interface
    )

    print(f"\n  Attack Type: {attack_type}")
    print(f"  Source IP: {source_ip}")
    print(f"  Target IP: {dest_ip}")
    print(f"  Interface: {interface}")

    print(f"\n  Generated Mitigation Commands:")
    print("  " + "-" * 50)

    commands = mitigation.get('commands', [])
    for i, cmd in enumerate(commands[:5], 1):  # Show first 5 commands
        print(f"  {i}. {cmd}")

    if len(commands) > 5:
        print(f"  ... and {len(commands) - 5} more commands")

    print("  " + "-" * 50)

    # =========================================================================
    # STEP 7: FINAL OUTPUT
    # =========================================================================
    print_section("STEP 7: COMPLETE SECURITY ALERT")

    elapsed_time = time.time() - start_time

    desc = classification.get('attack_description', classification.get('description', 'Attack detected'))
    print("")
    print("  +====================================================================+")
    print("  |                       SECURITY ALERT                              |")
    print("  +====================================================================+")
    print(f"  |  Timestamp:     2026-01-30 12:00:00                              |")
    print(f"  |  Source IP:     {source_ip:<50}|")
    print(f"  |  Destination:   {dest_ip}:80                                       |")
    print("  +--------------------------------------------------------------------+")
    print(f"  |  DETECTION:     DoS ATTACK ({prob_dos*100:.1f}% confidence)                     |")
    print(f"  |  ATTACK TYPE:   {attack_type:<50}|")
    print(f"  |  SEVERITY:      {severity_level:<50}|")
    print("  +--------------------------------------------------------------------+")
    print(f"  |  EXPLANATION:                                                    |")
    print(f"  |  Top features: {', '.join(top_features):<51}|")
    print(f"  |  {desc[:64]:<64}|")
    print("  +--------------------------------------------------------------------+")
    print(f"  |  RECOMMENDED ACTIONS:                                            |")
    print(f"  |  1. Block source IP: iptables -A INPUT -s {source_ip} -j DROP   |")
    print(f"  |  2. Apply rate limiting on interface                             |")
    print(f"  |  3. Enable SYN cookies protection                                |")
    print(f"  |  4. Escalate to security team                                    |")
    print("  +====================================================================+")

    print(f"\n  Processing Time: {elapsed_time*1000:.2f} ms")

    print_header("DEMO COMPLETE")
    print("\n  The sample has flowed through all 7 phases of the pipeline:")
    print("    1. INPUT       - Network traffic features")
    print("    2. DETECT      - XGBoost classification")
    print("    3. EXPLAIN     - SHAP feature contributions")
    print("    4. CLASSIFY    - Attack type identification")
    print("    5. ASSESS      - Severity calculation")
    print("    6. MITIGATE    - Command generation")
    print("    7. OUTPUT      - Complete security alert")
    print("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Demo single sample through pipeline')
    parser.add_argument('--index', type=int, default=None,
                        help='Specific sample index to use')
    parser.add_argument('--normal', action='store_true',
                        help='Demo with a normal sample instead of DoS')

    args = parser.parse_args()

    demo_pipeline(
        sample_index=args.index,
        find_dos=not args.normal
    )
