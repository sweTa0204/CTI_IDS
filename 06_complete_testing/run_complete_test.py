"""
Complete Pipeline Test - All Samples (WITH OPTIMIZED THRESHOLD)
================================================================

This script runs ALL 68,264 benchmark samples through the complete pipeline:
1. XGBoost Model -> Prediction (DoS/Normal) with OPTIMIZED THRESHOLD
2. SHAP Explainer -> Feature contributions
3. Attack Classifier -> Attack type identification
4. Severity Calculator -> Severity assessment
5. Mitigation Generator -> Actionable recommendations

IMPORTANT: Uses optimized threshold of 0.8517 (not default 0.5)
This matches the benchmark results from proper_training.

Outputs:
- complete_results.json: All alerts
- summary_report.json: Statistics and metrics
- confusion_matrix.json: Accuracy metrics
- attack_distribution.json: Attack type breakdown

Author: Research Project
Date: 2026-01-30
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# Add paths for imports
# Now at CTI_IDS/06_complete_testing/, so go up 1 level to get to CTI_IDS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '04_xai_integration'))
sys.path.insert(0, os.path.join(BASE_DIR, '05_mitigation_framework'))

# Import components
from shap_explainer import SHAPExplainer
from alert_generator import AlertGenerator, get_alert_statistics

# Paths
MODEL_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                          'models', 'xgboost', 'xgboost_model.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                               'data', 'X_test_scaled.csv')
TEST_LABELS_PATH = os.path.join(BASE_DIR, '03_model_training', 'proper_training',
                                 'data', 'y_test.csv')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# OPTIMIZED THRESHOLD from benchmark results
# This was found by searching for the threshold that maximizes F1 score
OPTIMIZED_THRESHOLD = 0.8517


def explain_single_with_threshold(explainer, features, record_id, threshold=0.5):
    """
    Generate SHAP explanation with custom threshold.

    This is a modified version of explain_single that uses a custom threshold
    instead of the default 0.5.
    """
    features_array = np.array(features).reshape(1, -1)

    # Get model prediction probabilities
    prediction_proba = explainer.model.predict_proba(features_array)[0]

    # Use CUSTOM THRESHOLD instead of 0.5
    prediction = int(prediction_proba[1] >= threshold)
    confidence = float(prediction_proba[1] if prediction == 1 else prediction_proba[0])

    # Calculate SHAP values
    shap_values = explainer.explainer.shap_values(features_array)

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Class 1 (DoS) SHAP values
    else:
        sv = shap_values[0]

    # Get base value
    if isinstance(explainer.explainer.expected_value, (list, np.ndarray)):
        base_value = float(explainer.explainer.expected_value[1])
    else:
        base_value = float(explainer.explainer.expected_value)

    # Feature names
    feature_names = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                     'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']

    # Create SHAP value dictionary
    shap_dict = {fn: float(sv[i]) for i, fn in enumerate(feature_names)}

    # Get top features (sorted by absolute SHAP value)
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [f[0] for f in sorted_features]

    # Feature values dictionary
    feature_values = {fn: float(features_array[0][i]) for i, fn in enumerate(feature_names)}

    return {
        "record_id": record_id,
        "prediction": "DoS" if prediction == 1 else "Normal",
        "prediction_code": prediction,
        "confidence": round(confidence, 4),
        "probability_dos": round(float(prediction_proba[1]), 4),
        "probability_normal": round(float(prediction_proba[0]), 4),
        "threshold_used": threshold,
        "shap_values": shap_dict,
        "feature_values": feature_values,
        "base_value": base_value,
        "top_features": top_features
    }


def run_complete_test(threshold=OPTIMIZED_THRESHOLD):
    """Run the complete test on all samples with specified threshold."""

    start_time = time.time()

    print("")
    print("=" * 80)
    print("    COMPLETE PIPELINE TEST - ALL BENCHMARK SAMPLES")
    print("=" * 80)
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Using OPTIMIZED THRESHOLD: {threshold} ({threshold*100:.2f}%)")
    print("=" * 80)
    print("")

    # =========================================================================
    # Step 1: Initialize Components
    # =========================================================================
    print("[Step 1/6] Initializing Components...")
    print("-" * 80)

    print("  Loading XGBoost model and SHAP explainer...")
    explainer = SHAPExplainer(model_path=MODEL_PATH)
    explainer.load_model()
    explainer.initialize_explainer()
    print("  [OK] SHAP Explainer initialized")

    print("  Loading Alert Generator (classifier + severity + mitigation)...")
    alert_gen = AlertGenerator()
    print("  [OK] Alert Generator initialized")

    # =========================================================================
    # Step 2: Load All Test Data
    # =========================================================================
    print("")
    print("[Step 2/6] Loading Test Data...")
    print("-" * 80)

    X_test = pd.read_csv(TEST_DATA_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)

    total_samples = len(X_test)
    dos_count = sum(y_test.iloc[:, 0] == 1)
    normal_count = sum(y_test.iloc[:, 0] == 0)

    print(f"  Total samples: {total_samples:,}")
    print(f"  DoS samples:   {dos_count:,} ({dos_count/total_samples*100:.1f}%)")
    print(f"  Normal samples: {normal_count:,} ({normal_count/total_samples*100:.1f}%)")
    print("  [OK] Test data loaded")

    # =========================================================================
    # Step 3: Run Complete Pipeline on All Samples
    # =========================================================================
    print("")
    print("[Step 3/6] Running Complete Pipeline on ALL Samples...")
    print("-" * 80)
    print("  Pipeline: XGBoost -> SHAP -> Classification -> Severity -> Mitigation")
    print(f"  Threshold: {threshold} (optimized for F1 score)")
    print("")

    # Initialize results storage
    all_alerts = []
    predictions = []
    actual_labels = []

    # Progress tracking
    progress_interval = 5000

    print(f"  Processing {total_samples:,} samples...")
    print("")

    for idx in range(total_samples):
        # Get features and actual label
        features = X_test.iloc[idx].values
        actual_label = "DoS" if y_test.iloc[idx, 0] == 1 else "Normal"
        actual_labels.append(actual_label)

        # Step 3a: SHAP Explanation WITH OPTIMIZED THRESHOLD
        shap_exp = explain_single_with_threshold(
            explainer, features, record_id=int(idx), threshold=threshold
        )
        shap_exp['actual_label'] = actual_label

        # Store prediction
        predictions.append(shap_exp.get('prediction'))

        # Step 3b-d: Generate complete alert (classification + severity + mitigation)
        alert = alert_gen.generate_alert(
            shap_exp,
            source_ip=f"192.168.{(idx // 256) % 256}.{idx % 256}",
            destination_ip="10.0.0.1",
            interface="eth0"
        )
        alert['actual_label'] = actual_label
        alert['sample_index'] = idx
        alert['threshold_used'] = threshold

        all_alerts.append(alert)

        # Progress reporting
        if (idx + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (total_samples - idx - 1) / rate
            print(f"  Processed: {idx + 1:,}/{total_samples:,} ({(idx+1)/total_samples*100:.1f}%) "
                  f"| Rate: {rate:.1f}/s | ETA: {eta/60:.1f} min")

    processing_time = time.time() - start_time
    print("")
    print(f"  [OK] All {total_samples:,} samples processed in {processing_time/60:.1f} minutes")

    # =========================================================================
    # Step 4: Calculate Metrics
    # =========================================================================
    print("")
    print("[Step 4/6] Calculating Metrics...")
    print("-" * 80)

    # Confusion Matrix
    tp = sum(1 for p, a in zip(predictions, actual_labels) if p == "DoS" and a == "DoS")
    tn = sum(1 for p, a in zip(predictions, actual_labels) if p == "Normal" and a == "Normal")
    fp = sum(1 for p, a in zip(predictions, actual_labels) if p == "DoS" and a == "Normal")
    fn = sum(1 for p, a in zip(predictions, actual_labels) if p == "Normal" and a == "DoS")

    accuracy = (tp + tn) / total_samples * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Threshold Used: {threshold} ({threshold*100:.2f}%)")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"    True Positives (DoS->DoS):       {tp:,}")
    print(f"    True Negatives (Normal->Normal): {tn:,}")
    print(f"    False Positives (Normal->DoS):   {fp:,}")
    print(f"    False Negatives (DoS->Normal):   {fn:,}")
    print("")
    print(f"  Performance Metrics:")
    print(f"    Accuracy:  {accuracy:.2f}%")
    print(f"    Precision: {precision:.2f}%")
    print(f"    Recall:    {recall:.2f}%")
    print(f"    F1-Score:  {f1_score:.2f}%")

    # Attack Type Distribution (only for DoS predictions)
    dos_predictions_count = tp + fp
    attack_types = Counter()
    severity_levels = Counter()
    escalation_count = 0

    for alert in all_alerts:
        if alert['detection']['prediction'] == 'DoS':
            at = alert['classification'].get('attack_type', 'Unknown')
            attack_types[at] += 1

            level = alert['severity'].get('level', 'Unknown')
            severity_levels[level] += 1

            if alert['severity'].get('escalation_required', False):
                escalation_count += 1

    print("")
    print(f"  Attack Type Distribution (from {dos_predictions_count:,} DoS predictions):")
    for at, count in attack_types.most_common():
        pct = count / dos_predictions_count * 100 if dos_predictions_count > 0 else 0
        print(f"    {at}: {count:,} ({pct:.1f}%)")

    print("")
    print(f"  Severity Distribution:")
    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_levels.get(level, 0)
        pct = count / dos_predictions_count * 100 if dos_predictions_count > 0 else 0
        print(f"    {level}: {count:,} ({pct:.1f}%)")

    print("")
    esc_pct = escalation_count / dos_predictions_count * 100 if dos_predictions_count > 0 else 0
    print(f"  Escalation Required: {escalation_count:,} ({esc_pct:.1f}% of DoS predictions)")

    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print("")
    print("[Step 5/6] Saving Results...")
    print("-" * 80)

    # Save complete results (all alerts)
    results_path = os.path.join(OUTPUT_DIR, 'complete_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_alerts, f, indent=2)
    print(f"  [OK] Complete results: {results_path}")

    # Save confusion matrix
    confusion_matrix = {
        "threshold_used": threshold,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.json')
    with open(cm_path, 'w') as f:
        json.dump(confusion_matrix, f, indent=2)
    print(f"  [OK] Confusion matrix: {cm_path}")

    # Save attack distribution
    attack_dist = {
        "threshold_used": threshold,
        "total_dos_predictions": dos_predictions_count,
        "attack_types": dict(attack_types),
        "percentages": {at: round(count/dos_predictions_count*100, 2)
                       for at, count in attack_types.items()} if dos_predictions_count > 0 else {}
    }
    dist_path = os.path.join(OUTPUT_DIR, 'attack_distribution.json')
    with open(dist_path, 'w') as f:
        json.dump(attack_dist, f, indent=2)
    print(f"  [OK] Attack distribution: {dist_path}")

    # Save summary report
    total_time = time.time() - start_time
    summary = {
        "test_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_samples": total_samples,
            "dos_samples": dos_count,
            "normal_samples": normal_count,
            "threshold_used": threshold,
            "processing_time_minutes": round(total_time / 60, 2),
            "samples_per_second": round(total_samples / total_time, 2)
        },
        "model_performance": confusion_matrix,
        "attack_type_distribution": dict(attack_types),
        "severity_distribution": dict(severity_levels),
        "escalation_stats": {
            "escalation_required": escalation_count,
            "percentage_of_dos": round(esc_pct, 2)
        },
        "pipeline_components": [
            "XGBoost Classifier (threshold optimized)",
            "SHAP TreeExplainer",
            "Attack Classifier (4 types)",
            "Severity Calculator (4 levels)",
            "Mitigation Generator (iptables/tc)"
        ],
        "threshold_explanation": {
            "default_threshold": 0.5,
            "optimized_threshold": threshold,
            "reason": "Threshold optimized to maximize F1 score on imbalanced data"
        }
    }
    summary_path = os.path.join(OUTPUT_DIR, 'summary_report.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Summary report: {summary_path}")

    # =========================================================================
    # Step 6: Final Summary
    # =========================================================================
    print("")
    print("[Step 6/6] Test Complete!")
    print("-" * 80)

    print("")
    print("=" * 80)
    print("                         COMPLETE TEST SUMMARY")
    print("=" * 80)
    print("")
    print(f"  Threshold Used:           {threshold} ({threshold*100:.2f}%)")
    print(f"  Total Samples Processed:  {total_samples:,}")
    print(f"  Processing Time:          {total_time/60:.2f} minutes")
    print(f"  Processing Rate:          {total_samples/total_time:.1f} samples/second")
    print("")
    print("  Model Performance (with optimized threshold):")
    print(f"    Accuracy:   {accuracy:.2f}%")
    print(f"    Precision:  {precision:.2f}%")
    print(f"    Recall:     {recall:.2f}%")
    print(f"    F1-Score:   {f1_score:.2f}%")
    print("")
    print("  Output Files:")
    print(f"    - complete_results.json   ({total_samples:,} alerts)")
    print(f"    - confusion_matrix.json   (accuracy metrics)")
    print(f"    - attack_distribution.json (attack type breakdown)")
    print(f"    - summary_report.json     (complete summary)")
    print("")
    print("=" * 80)
    print(f"    Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return all_alerts, summary


if __name__ == "__main__":
    # Run with optimized threshold
    alerts, summary = run_complete_test(threshold=OPTIMIZED_THRESHOLD)
