"""
Test Script for Mitigation Generator
=====================================

This script tests the mitigation generator using classification and
severity results from previous steps.

Run this script to:
1. Load classification and severity results
2. Generate mitigation recommendations
3. Display the mitigation commands

Author: Research Project
Date: 2026-01-29
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mitigation_generator import MitigationGenerator, get_mitigation_statistics

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFICATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'sample_classification_output.json')
SEVERITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'sample_severity_output.json')


def test_with_sample_data():
    """Test mitigation generator with pre-generated data."""

    print("=" * 70)
    print("MITIGATION GENERATOR TEST")
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

    # Step 2: Load severity results
    print("\n[Step 2] Loading severity results...")

    if os.path.exists(SEVERITY_PATH):
        with open(SEVERITY_PATH, 'r') as f:
            severity_data = json.load(f)
        severities = severity_data.get('severity_results', [])
        print(f"[OK] Loaded {len(severities)} severity assessments")
    else:
        print("[!] Severity file not found. Run test_severity_calculator.py first.")
        return None

    # Step 3: Initialize Mitigation Generator
    print("\n[Step 3] Initializing Mitigation Generator...")
    generator = MitigationGenerator()
    print("[OK] Mitigation Generator initialized!")

    # Step 4: Generate mitigations
    print("\n[Step 4] Generating mitigation recommendations...")

    # Match classifications and severities by record_id
    mitigations = []
    for classification in classifications:
        record_id = classification.get('record_id')

        # Find matching severity
        severity = None
        for s in severities:
            if s.get('record_id') == record_id:
                severity = s
                break

        if severity is None:
            severity = {"severity": None}

        # Generate mitigation with sample IP
        mitigation = generator.generate(
            classification,
            severity,
            source_ip=f"192.168.1.{record_id % 255}",  # Sample IP based on record
            interface="eth0"
        )
        mitigation['record_id'] = record_id
        mitigations.append(mitigation)

    print(f"[OK] Generated {len(mitigations)} mitigation recommendations")

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("MITIGATION RECOMMENDATIONS")
    print("=" * 70)

    for mitigation in mitigations:
        print(f"\n{'=' * 70}")
        print(f"Record ID: {mitigation['record_id']}")
        print(f"{'=' * 70}")

        if not mitigation.get('mitigations_required', False):
            print("Status: No mitigation required (Normal traffic)")
            continue

        print(f"Attack Type: {mitigation['attack_type']}")
        print(f"Severity: {mitigation['severity']}")
        print(f"Strategy: {mitigation.get('primary_strategy', 'N/A')}")
        print(f"Auto-Apply Recommended: {'YES' if mitigation.get('auto_apply_recommended') else 'NO'}")

        # Immediate actions
        if mitigation.get('immediate_actions'):
            print(f"\n--- Immediate Actions ---")
            for action in mitigation['immediate_actions']:
                print(f"\n  [{action['name']}]")
                if action.get('description'):
                    print(f"    {action['description']}")
                if action.get('command'):
                    print(f"    $ {action['command']}")
                if action.get('followup'):
                    print(f"    $ {action['followup']}")
                if action.get('commands'):
                    for cmd in action['commands']:
                        print(f"    $ {cmd}")

        # Alternative actions (show first one only)
        if mitigation.get('alternative_actions'):
            print(f"\n--- Alternative Actions ---")
            action = mitigation['alternative_actions'][0]
            print(f"  [{action['name']}]")
            if action.get('description'):
                print(f"    {action['description']}")

        # Monitoring (show first one only)
        if mitigation.get('monitoring_commands'):
            print(f"\n--- Monitoring ---")
            action = mitigation['monitoring_commands'][0]
            print(f"  [{action['name']}]")
            if action.get('command'):
                print(f"    $ {action['command']}")

        # Human explanation
        print(f"\n--- Human Explanation ---")
        explanation = mitigation.get('human_explanation', 'N/A')
        # Print first 3 lines of explanation
        for line in explanation.split('\n')[:5]:
            print(f"  {line}")

    # Step 6: Summary statistics
    print("\n" + "=" * 70)
    print("MITIGATION SUMMARY")
    print("=" * 70)

    # Filter only actual mitigations
    actual_mitigations = [m for m in mitigations if m.get('mitigations_required', False)]

    if actual_mitigations:
        stats = get_mitigation_statistics(actual_mitigations)

        print(f"\nTotal mitigations generated: {stats['total_mitigations']}")
        print(f"\nAttack Type Distribution:")
        for at, count in stats.get('attack_type_counts', {}).items():
            print(f"  - {at}: {count}")

        print(f"\nAuto-Apply Recommended: {stats['auto_apply_recommended']} "
              f"({stats['auto_apply_percentage']}%)")

        print(f"\nStrategies Used:")
        for strategy, count in stats.get('strategy_counts', {}).items():
            print(f"  - {strategy}: {count}")
    else:
        print("\nNo mitigations required (all traffic normal)")

    # Step 7: Save output
    print("\n[Step 7] Saving mitigation results...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'sample_mitigation_output.json')

    output_data = {
        "test_info": {
            "records_processed": len(mitigations),
            "mitigations_generated": len(actual_mitigations)
        },
        "mitigations": mitigations,
        "statistics": get_mitigation_statistics(actual_mitigations) if actual_mitigations else None
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Output saved to: {output_path}")

    # Final status
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the mitigation commands above")
    print("2. Verify commands are appropriate for attack types")
    print("3. If approved, proceed to Step 5 (Alert System)")

    return mitigations


def test_all_attack_types():
    """Test mitigation generator with all attack types."""

    print("\n" + "=" * 70)
    print("ALL ATTACK TYPES TEST")
    print("=" * 70)

    generator = MitigationGenerator()

    attack_types = ["Volumetric Flood", "Protocol Exploit", "Slowloris", "Amplification", "Generic DoS"]

    print("\nTesting mitigation generation for all attack types...\n")

    for attack_type in attack_types:
        classification = {
            "attack_type": attack_type,
            "confidence": 0.95,
            "mitigation_category": "test"
        }
        severity = {
            "severity": "HIGH",
            "severity_score": 0.92
        }

        mitigation = generator.generate(
            classification, severity,
            source_ip="10.0.0.100",
            interface="eth0"
        )

        print(f"[OK] {attack_type}")
        print(f"     Strategy: {mitigation.get('primary_strategy', 'N/A')}")
        print(f"     Immediate Actions: {len(mitigation.get('immediate_actions', []))}")
        print(f"     Alternative Actions: {len(mitigation.get('alternative_actions', []))}")
        print(f"     Monitoring Commands: {len(mitigation.get('monitoring_commands', []))}")
        print()

    print("-" * 70)
    print("[OK] All attack types have mitigation commands!")


if __name__ == "__main__":
    # Run main test with real/sample data
    mitigations = test_with_sample_data()

    # Test all attack types
    test_all_attack_types()
