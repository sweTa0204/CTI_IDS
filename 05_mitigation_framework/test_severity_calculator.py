"""
Test Script for Severity Calculator
====================================

This script tests the severity calculator using classification results
from the attack classifier.

Run this script to:
1. Load classification results (from sample_classification_output.json)
2. Load SHAP explanations (from sample_shap_output.json)
3. Calculate severity for each detection
4. Display severity assessments

Author: Research Project
Date: 2026-01-29
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from severity_calculator import SeverityCalculator, get_severity_statistics

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFICATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'sample_classification_output.json')
SHAP_OUTPUT_PATH = os.path.join(BASE_DIR, '04_xai_integration', 'sample_shap_output.json')


def test_with_sample_data():
    """Test severity calculator with pre-generated data."""

    print("=" * 70)
    print("SEVERITY CALCULATOR TEST")
    print("=" * 70)

    # Step 1: Load classification results
    print("\n[Step 1] Loading classification results...")

    if os.path.exists(CLASSIFICATION_PATH):
        with open(CLASSIFICATION_PATH, 'r') as f:
            classification_data = json.load(f)
        classifications = classification_data.get('classifications', [])
        print(f"[OK] Loaded {len(classifications)} classifications")
    else:
        print("[!] Classification file not found. Run test_attack_classifier.py first.")
        return None

    # Step 2: Load SHAP explanations
    print("\n[Step 2] Loading SHAP explanations...")

    shap_explanations = []
    if os.path.exists(SHAP_OUTPUT_PATH):
        with open(SHAP_OUTPUT_PATH, 'r') as f:
            shap_data = json.load(f)
        shap_explanations = shap_data.get('explanations', [])
        print(f"[OK] Loaded {len(shap_explanations)} SHAP explanations")
    else:
        print("[!] SHAP output file not found. Proceeding without SHAP data.")

    # Step 3: Initialize Severity Calculator
    print("\n[Step 3] Initializing Severity Calculator...")
    calculator = SeverityCalculator()
    print("[OK] Severity Calculator initialized!")
    print(f"     Severity levels: {list(calculator.severity_levels.keys())}")

    # Step 4: Calculate severity for each classification
    print("\n[Step 4] Calculating severity levels...")

    severity_results = []
    for i, classification in enumerate(classifications):
        # Match SHAP explanation by record_id
        shap_exp = None
        record_id = classification.get('record_id')
        for exp in shap_explanations:
            if exp.get('record_id') == record_id:
                shap_exp = exp
                break

        severity = calculator.calculate(classification, shap_exp)
        severity['record_id'] = record_id
        severity['attack_type'] = classification.get('attack_type')
        severity_results.append(severity)

    print(f"[OK] Calculated severity for {len(severity_results)} records")

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("SEVERITY ASSESSMENT RESULTS")
    print("=" * 70)

    for result in severity_results:
        print(f"\n--- Record {result['record_id']} ---")
        print(f"Attack Type:      {result['attack_type']}")

        if result['severity'] is None:
            print(f"Severity:         N/A (Normal traffic)")
            print(f"Description:      {result['description']}")
            continue

        # Visual severity indicator
        severity = result['severity']
        if severity == "CRITICAL":
            indicator = "[!!!!] CRITICAL"
        elif severity == "HIGH":
            indicator = "[!!!]  HIGH"
        elif severity == "MEDIUM":
            indicator = "[!!]   MEDIUM"
        else:
            indicator = "[!]    LOW"

        print(f"Severity:         {indicator}")
        print(f"Severity Score:   {result['severity_score']*100:.1f}%")
        print(f"Description:      {result['description']}")
        print(f"Escalation:       {'YES - Required' if result['escalation_required'] else 'No'}")

        print(f"\nScore Breakdown:")
        breakdown = result.get('score_breakdown', {})
        print(f"  - Base confidence:     {breakdown.get('base_confidence', 0)*100:.1f}%")
        print(f"  - Attack type modifier: +{breakdown.get('attack_type_modifier', 0)*100:.1f}%")
        print(f"  - Feature modifier:     +{breakdown.get('feature_modifier', 0)*100:.1f}%")
        print(f"  - Total score:          {breakdown.get('total_score', 0)*100:.1f}%")

        print(f"\nRecommended Actions:")
        for action in result.get('recommended_actions', []):
            print(f"  -> {action}")

        print(f"\nReasoning: {result['reasoning']}")

    # Step 6: Summary statistics
    print("\n" + "=" * 70)
    print("SEVERITY SUMMARY")
    print("=" * 70)

    # Filter only DoS detections (non-None severity)
    dos_severities = [r for r in severity_results if r['severity'] is not None]

    if dos_severities:
        stats = get_severity_statistics(dos_severities)

        print(f"\nTotal DoS detections assessed: {stats['total_assessments']}")
        print(f"\nSeverity Level Distribution:")

        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = stats['level_counts'].get(level, 0)
            percentage = stats['level_percentages'].get(level, 0)
            bar_length = int(percentage / 5)
            bar = "#" * bar_length if bar_length > 0 else ""
            print(f"  {level:8s}: {count} ({percentage:.1f}%) {bar}")

        print(f"\nEscalation Required: {stats['escalation_required_count']} "
              f"({stats['escalation_percentage']:.1f}%)")
        print(f"Average Severity Score: {stats['average_severity_score']*100:.1f}%")
        print(f"Most Severe Level Present: {stats['most_severe_level']}")
    else:
        print("\nNo DoS detections to assess.")

    # Step 7: Save output
    print("\n[Step 7] Saving severity results...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'sample_severity_output.json')

    output_data = {
        "test_info": {
            "records_assessed": len(severity_results),
            "dos_detections": len(dos_severities),
            "severity_levels": list(calculator.severity_levels.keys())
        },
        "severity_results": severity_results,
        "statistics": get_severity_statistics(dos_severities) if dos_severities else None
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Output saved to: {output_path}")

    # Final status
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the severity assessments above")
    print("2. Verify reasoning makes sense")
    print("3. If approved, proceed to Step 4 (Mitigation Generator)")

    return severity_results


def test_with_synthetic_data():
    """Test severity calculator with synthetic data to verify all severity levels."""

    print("\n" + "=" * 70)
    print("SYNTHETIC DATA TEST (All Severity Levels)")
    print("=" * 70)

    calculator = SeverityCalculator()

    # Create synthetic scenarios for each severity level
    synthetic_tests = [
        {
            "name": "CRITICAL Scenario (95%+ confidence, Amplification attack)",
            "classification": {
                "attack_type": "Amplification",
                "confidence": 0.98,
                "mitigation_category": "amplification_filtering"
            },
            "shap_explanation": {
                "confidence": 0.98,
                "shap_values": {"dload": 3.5, "sload": 0.5, "rate": 0.8, "proto": 0.3},
                "feature_values": {"dload": 50000, "sload": 1000}
            },
            "expected_level": "CRITICAL"
        },
        {
            "name": "HIGH Scenario (90-95% confidence, Volumetric attack)",
            "classification": {
                "attack_type": "Volumetric Flood",
                "confidence": 0.92,
                "mitigation_category": "rate_limiting"
            },
            "shap_explanation": {
                "confidence": 0.92,
                "shap_values": {"rate": 1.2, "sload": 0.8, "sbytes": 0.5},
                "feature_values": {"rate": 1500, "sload": 900000}
            },
            "expected_level": "HIGH"
        },
        {
            "name": "MEDIUM Scenario (75-90% confidence, Protocol attack)",
            "classification": {
                "attack_type": "Protocol Exploit",
                "confidence": 0.82,
                "mitigation_category": "protocol_filtering"
            },
            "shap_explanation": {
                "confidence": 0.82,
                "shap_values": {"proto": 1.5, "stcpb": 0.4, "rate": 0.2},
                "feature_values": {"proto": 1}
            },
            "expected_level": "MEDIUM"
        },
        {
            "name": "LOW Scenario (60-75% confidence, Generic attack)",
            "classification": {
                "attack_type": "Generic DoS",
                "confidence": 0.68,
                "mitigation_category": "general_protection"
            },
            "shap_explanation": {
                "confidence": 0.68,
                "shap_values": {"rate": 0.3, "sload": 0.2, "sbytes": 0.1},
                "feature_values": {"rate": 200}
            },
            "expected_level": "LOW"
        },
        {
            "name": "Normal Traffic (No attack)",
            "classification": {
                "attack_type": "None",
                "confidence": 0.95,
                "mitigation_category": None
            },
            "shap_explanation": {
                "confidence": 0.95,
                "shap_values": {"rate": -0.2, "sload": -0.15},
                "feature_values": {"rate": 50}
            },
            "expected_level": None
        }
    ]

    print("\nTesting all severity level scenarios...\n")

    all_passed = True
    for test in synthetic_tests:
        result = calculator.calculate(test["classification"], test["shap_explanation"])

        passed = result["severity"] == test["expected_level"]
        status = "[OK]" if passed else "[X]"

        if not passed:
            all_passed = False

        print(f"{status} {test['name']}")
        print(f"    Expected: {test['expected_level']}")
        print(f"    Got:      {result['severity']} (score: {result['severity_score']*100:.1f}%)")
        if result.get('escalation_required'):
            print(f"    Escalation: Required")
        print()

    print("-" * 70)
    if all_passed:
        print("[OK] All synthetic tests PASSED!")
    else:
        print("[!] Some tests failed - calculator may need tuning")

    return all_passed


if __name__ == "__main__":
    # Run main test with real/sample data
    severity_results = test_with_sample_data()

    # Run synthetic data test
    test_with_synthetic_data()
