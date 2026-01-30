"""
DoS Detection and Mitigation Framework - Main Entry Point
==========================================================

This is the main entry point for the complete DoS detection and mitigation system.

Pipeline:
1. XGBoost Model     -> Predicts DoS or Normal
2. SHAP Explainer    -> Explains WHY (feature contributions)
3. Attack Classifier -> Classifies attack TYPE
4. Severity Calculator -> Determines severity LEVEL
5. Mitigation Generator -> Generates ACTIONS
6. Alert Generator   -> Combines everything into ALERT

Usage:
    python main.py                    # Run demo with sample data
    python main.py --test             # Run on test data
    python main.py --record <id>      # Explain specific record

Author: Research Project
Date: 2026-01-29
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Add parent directory and XAI module to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '04_xai_integration'))

# Import SHAP explainer from XAI module
from shap_explainer import SHAPExplainer

# Import mitigation framework components
from alert_generator import AlertGenerator, get_alert_statistics


# Paths (BASE_DIR already defined above)
MODEL_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                          'models', 'xgboost', 'xgboost_model.pkl')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                                'data', 'X_train_scaled.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                               'data', 'X_test_scaled.csv')
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                                  'data', 'y_train.csv')
TEST_LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                                 'data', 'y_test.csv')


def run_demo():
    """Run demo with sample data from training set."""

    print("")
    print("=" * 75)
    print("    DoS DETECTION AND MITIGATION FRAMEWORK - DEMO")
    print("=" * 75)
    print("")

    # Step 1: Initialize components
    print("[1/4] Initializing SHAP Explainer...")
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()
    print("      [OK] SHAP Explainer ready")

    print("[2/4] Initializing Alert Generator...")
    alert_gen = AlertGenerator()
    print("      [OK] Alert Generator ready (includes classifier, severity, mitigation)")

    # Step 2: Load sample data
    print("[3/4] Loading sample data...")
    X_train = pd.read_csv(TRAIN_DATA_PATH)
    y_train = pd.read_csv(TRAIN_LABELS_PATH)

    # Select samples: 2 DoS, 1 Normal
    dos_indices = y_train[y_train.iloc[:, 0] == 1].index.tolist()
    normal_indices = y_train[y_train.iloc[:, 0] == 0].index.tolist()

    np.random.seed(42)
    sample_indices = list(np.random.choice(dos_indices, size=2, replace=False))
    sample_indices.append(np.random.choice(normal_indices, size=1)[0])

    print(f"      [OK] Selected {len(sample_indices)} samples")

    # Step 3: Generate alerts
    print("[4/4] Generating detection alerts...")
    print("")

    alerts = []
    for idx in sample_indices:
        features = X_train.iloc[idx].values
        actual_label = "DoS" if y_train.iloc[idx, 0] == 1 else "Normal"

        # Generate SHAP explanation
        shap_exp = explainer.explain_single(features, record_id=int(idx))
        shap_exp['actual_label'] = actual_label

        # Generate complete alert
        alert = alert_gen.generate_alert(
            shap_exp,
            source_ip=f"192.168.1.{idx % 255}",
            destination_ip="10.0.0.1",
            interface="eth0"
        )
        alert['actual_label'] = actual_label
        alerts.append(alert)

        # Display alert
        print(alert_gen.format_for_console(alert))
        print("")

        # Show actual vs predicted
        prediction = alert['detection']['prediction']
        match = "[OK] CORRECT" if prediction == actual_label else "[X] INCORRECT"
        print(f"Actual Label: {actual_label} | Predicted: {prediction} | {match}")
        print("")
        print("-" * 75)

    # Summary
    print("")
    print("=" * 75)
    print("                           DEMO SUMMARY")
    print("=" * 75)
    print("")

    stats = get_alert_statistics(alerts)
    print(f"Total alerts generated:    {stats['total_alerts']}")
    print(f"DoS detections:            {stats['dos_detections']}")
    print(f"Normal traffic:            {stats['normal_traffic']}")
    print(f"Escalation required:       {stats['escalation_required']}")
    print("")
    print("Attack type distribution:")
    for at, count in stats.get('attack_type_distribution', {}).items():
        print(f"  - {at}: {count}")
    print("")
    print("Severity distribution:")
    for level, count in stats['severity_distribution'].items():
        if count > 0:
            print(f"  - {level}: {count}")

    # Save output
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'demo_output.json')
    alert_gen.save_alerts(alerts, output_path)
    print("")
    print(f"[OK] Alerts saved to: {output_path}")
    print("")
    print("=" * 75)
    print("                        DEMO COMPLETED")
    print("=" * 75)

    return alerts


def run_on_test_data(n_samples=10, dos_only=True):
    """Run on external test data (benchmark)."""

    print("")
    print("=" * 75)
    print("    DoS DETECTION - BENCHMARK TEST DATA")
    print("=" * 75)
    print("")

    # Check if test data exists
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR] Test data not found: {TEST_DATA_PATH}")
        print("Please ensure the test data is available.")
        return None

    # Initialize components
    print("[1/4] Initializing SHAP Explainer...")
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()
    print("      [OK] Ready")

    print("[2/4] Initializing Alert Generator...")
    alert_gen = AlertGenerator()
    print("      [OK] Ready")

    # Load test data
    print("[3/4] Loading test data...")
    X_test = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)
    print(f"      [OK] Loaded {len(X_test)} test samples")

    # Select samples
    if dos_only:
        # Get only DoS samples
        dos_indices = y_test[y_test.iloc[:, 0] == 1].index.tolist()
        np.random.seed(42)
        sample_indices = list(np.random.choice(dos_indices, size=min(n_samples, len(dos_indices)), replace=False))
        print(f"      [OK] Selected {len(sample_indices)} DoS samples for analysis")
    else:
        # Random samples
        np.random.seed(42)
        sample_indices = list(np.random.choice(len(X_test), size=n_samples, replace=False))
        print(f"      [OK] Selected {len(sample_indices)} random samples")

    # Generate alerts
    print("[4/4] Generating alerts...")
    print("")

    alerts = []
    for idx in sample_indices:
        features = X_test.iloc[idx].values
        actual_label = "DoS" if y_test.iloc[idx, 0] == 1 else "Normal"

        shap_exp = explainer.explain_single(features, record_id=int(idx))
        shap_exp['actual_label'] = actual_label

        alert = alert_gen.generate_alert(
            shap_exp,
            source_ip=f"192.168.1.{idx % 255}",
            destination_ip="10.0.0.1"
        )
        alert['actual_label'] = actual_label
        alerts.append(alert)

    # Display summary
    print("=" * 75)
    print("                         TEST RESULTS")
    print("=" * 75)
    print("")

    stats = get_alert_statistics(alerts)
    print(f"Total samples:         {stats['total_alerts']}")
    print(f"DoS detections:        {stats['dos_detections']}")
    print(f"Normal predictions:    {stats['normal_traffic']}")
    print("")

    # Check accuracy
    correct = sum(1 for a in alerts if a['detection']['prediction'] == a['actual_label'])
    accuracy = correct / len(alerts) * 100
    print(f"Prediction accuracy:   {correct}/{len(alerts)} ({accuracy:.2f}%)")
    print("")

    print("Attack type distribution:")
    for at, count in stats.get('attack_type_distribution', {}).items():
        print(f"  - {at}: {count}")
    print("")

    print("Severity distribution:")
    for level, count in stats['severity_distribution'].items():
        if count > 0:
            print(f"  - {level}: {count}")
    print("")

    print(f"Escalation required:   {stats['escalation_required']} ({stats['escalation_percentage']:.1f}%)")

    # Save output
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'test_results.json')
    alert_gen.save_alerts(alerts, output_path)
    print("")
    print(f"[OK] Results saved to: {output_path}")
    print("")
    print("=" * 75)

    return alerts


def explain_single_record(record_id, data_source="train"):
    """Explain a specific record."""

    print(f"\n[INFO] Explaining record {record_id} from {data_source} data\n")

    # Load data
    if data_source == "train":
        X = pd.read_csv(TRAIN_DATA_PATH)
        y = pd.read_csv(TRAIN_LABELS_PATH)
    else:
        X = pd.read_csv(TEST_DATA_PATH)
        y = pd.read_csv(TEST_LABELS_PATH)

    if record_id >= len(X):
        print(f"[ERROR] Record {record_id} not found. Max index: {len(X)-1}")
        return None

    # Initialize
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()

    alert_gen = AlertGenerator()

    # Generate explanation
    features = X.iloc[record_id].values
    actual_label = "DoS" if y.iloc[record_id, 0] == 1 else "Normal"

    shap_exp = explainer.explain_single(features, record_id=record_id)
    shap_exp['actual_label'] = actual_label

    alert = alert_gen.generate_alert(shap_exp)
    alert['actual_label'] = actual_label

    # Display
    print(alert_gen.format_for_console(alert))
    print("")
    print(f"Actual Label: {actual_label}")

    return alert


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="DoS Detection and Mitigation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run demo with training data
  python main.py --test             # Run on test/benchmark data
  python main.py --test -n 20       # Run on 20 test samples
  python main.py --record 12345     # Explain specific record
        """
    )

    parser.add_argument('--test', action='store_true',
                        help='Run on external test data (benchmark)')
    parser.add_argument('-n', '--num-samples', type=int, default=10,
                        help='Number of samples to process (default: 10)')
    parser.add_argument('--record', type=int,
                        help='Explain a specific record by index')
    parser.add_argument('--all', action='store_true',
                        help='Include normal samples (default: DoS only)')

    args = parser.parse_args()

    if args.record is not None:
        # Explain specific record
        explain_single_record(args.record)
    elif args.test:
        # Run on test data
        run_on_test_data(n_samples=args.num_samples, dos_only=not args.all)
    else:
        # Run demo
        run_demo()


if __name__ == "__main__":
    main()
