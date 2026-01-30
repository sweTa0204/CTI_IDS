"""
SHAP Explainer for DoS Detection Model
======================================

This module provides XAI (Explainable AI) capabilities for the XGBoost DoS detection model
using SHAP (SHapley Additive exPlanations) TreeExplainer.

Purpose:
- Explain WHY the model predicts a traffic record as DoS or Normal
- Provide feature contribution scores for each prediction
- Support the mitigation framework (Objective 4)

Author: Research Project
Date: 2026-01-29
"""

import pickle
import numpy as np
import pandas as pd
import shap
import json
import os

# Feature names (must match training data)
FEATURE_NAMES = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for XGBoost DoS detection model.

    This class provides methods to:
    1. Load the trained XGBoost model
    2. Initialize SHAP TreeExplainer
    3. Generate explanations for single or multiple predictions
    4. Format explanations for downstream use (mitigation framework)
    """

    def __init__(self, model_path=None):
        """
        Initialize the SHAP Explainer.

        Parameters:
        -----------
        model_path : str, optional
            Path to the trained XGBoost model (.pkl file)
            If None, uses default path relative to this file
        """
        self.model = None
        self.explainer = None
        self.feature_names = FEATURE_NAMES

        # Set default model path
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, '03_model_training', 'proper_training',
                                      'models', 'xgboost', 'xgboost_model.pkl')

        self.model_path = model_path

    def load_model(self):
        """Load the XGBoost model from pickle file."""
        print(f"Loading model from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("Model loaded successfully!")
        return self

    def initialize_explainer(self):
        """Initialize SHAP TreeExplainer for the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("TreeExplainer initialized successfully!")
        return self

    def explain_single(self, features, record_id=None):
        """
        Generate SHAP explanation for a single record.

        Parameters:
        -----------
        features : array-like
            Feature values for a single record (10 features)
        record_id : int, optional
            Record identifier

        Returns:
        --------
        dict : Explanation dictionary containing:
            - record_id: Record identifier
            - prediction: "DoS" or "Normal"
            - confidence: Prediction probability
            - shap_values: Dictionary of feature contributions
            - top_features: List of top 3 contributing features
            - base_value: Model's base prediction
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        # Ensure features is 2D array
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        elif isinstance(features, (list, np.ndarray)):
            features_array = np.array(features).reshape(1, -1)
        else:
            features_array = np.array([features])

        # Get model prediction
        prediction_proba = self.model.predict_proba(features_array)[0]
        prediction = int(prediction_proba[1] >= 0.5)  # Default threshold
        confidence = float(prediction_proba[1] if prediction == 1 else prediction_proba[0])

        # Get SHAP values
        shap_values = self.explainer.shap_values(features_array)

        # For binary classification, shap_values might be a list [class_0, class_1]
        if isinstance(shap_values, list):
            # Use class 1 (DoS) SHAP values
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Create feature contribution dictionary
        feature_contributions = {}
        for i, fname in enumerate(self.feature_names):
            feature_contributions[fname] = float(sv[i])

        # Sort by absolute contribution to get top features
        sorted_features = sorted(feature_contributions.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True)
        top_features = [f[0] for f in sorted_features[:3]]

        # Get base value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = float(self.explainer.expected_value[1])  # Class 1 base
        else:
            base_value = float(self.explainer.expected_value)

        return {
            "record_id": record_id,
            "prediction": "DoS" if prediction == 1 else "Normal",
            "prediction_code": prediction,
            "confidence": round(confidence, 4),
            "probability_dos": round(float(prediction_proba[1]), 4),
            "probability_normal": round(float(prediction_proba[0]), 4),
            "shap_values": {k: round(v, 4) for k, v in feature_contributions.items()},
            "top_features": top_features,
            "base_value": round(base_value, 4),
            "feature_values": {self.feature_names[i]: float(features_array[0][i])
                              for i in range(len(self.feature_names))}
        }

    def explain_batch(self, features_df, record_ids=None):
        """
        Generate SHAP explanations for multiple records.

        Parameters:
        -----------
        features_df : pd.DataFrame or np.ndarray
            Feature values for multiple records
        record_ids : list, optional
            List of record identifiers

        Returns:
        --------
        list : List of explanation dictionaries
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        # Convert to numpy if DataFrame
        if isinstance(features_df, pd.DataFrame):
            features_array = features_df.values
        else:
            features_array = np.array(features_df)

        n_samples = features_array.shape[0]

        if record_ids is None:
            record_ids = list(range(n_samples))

        explanations = []
        for i in range(n_samples):
            explanation = self.explain_single(features_array[i], record_ids[i])
            explanations.append(explanation)

        return explanations

    def explain_dos_only(self, features_df, record_ids=None, threshold=0.5):
        """
        Generate SHAP explanations ONLY for records predicted as DoS.

        Parameters:
        -----------
        features_df : pd.DataFrame or np.ndarray
            Feature values for multiple records
        record_ids : list, optional
            List of record identifiers
        threshold : float
            Classification threshold (default 0.5)

        Returns:
        --------
        list : List of explanation dictionaries for DoS predictions only
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        # Convert to numpy if DataFrame
        if isinstance(features_df, pd.DataFrame):
            features_array = features_df.values
        else:
            features_array = np.array(features_df)

        n_samples = features_array.shape[0]

        if record_ids is None:
            record_ids = list(range(n_samples))

        # Get all predictions first
        predictions_proba = self.model.predict_proba(features_array)[:, 1]

        # Find DoS predictions
        dos_indices = np.where(predictions_proba >= threshold)[0]

        print(f"Total records: {n_samples}")
        print(f"DoS detections (threshold={threshold}): {len(dos_indices)}")

        explanations = []
        for idx in dos_indices:
            explanation = self.explain_single(features_array[idx], record_ids[idx])
            explanations.append(explanation)

        return explanations

    def get_summary_stats(self, explanations):
        """
        Get summary statistics from a list of explanations.

        Parameters:
        -----------
        explanations : list
            List of explanation dictionaries

        Returns:
        --------
        dict : Summary statistics
        """
        if not explanations:
            return {"error": "No explanations provided"}

        # Count predictions
        dos_count = sum(1 for e in explanations if e['prediction'] == 'DoS')
        normal_count = len(explanations) - dos_count

        # Average feature contributions for DoS predictions
        dos_explanations = [e for e in explanations if e['prediction'] == 'DoS']

        avg_contributions = {}
        if dos_explanations:
            for feature in self.feature_names:
                values = [e['shap_values'][feature] for e in dos_explanations]
                avg_contributions[feature] = round(np.mean(values), 4)

        # Top features across all DoS detections
        feature_counts = {f: 0 for f in self.feature_names}
        for e in dos_explanations:
            for f in e['top_features']:
                feature_counts[f] += 1

        top_features_overall = sorted(feature_counts.items(),
                                       key=lambda x: x[1],
                                       reverse=True)[:5]

        return {
            "total_records": len(explanations),
            "dos_detections": dos_count,
            "normal_detections": normal_count,
            "dos_percentage": round(dos_count / len(explanations) * 100, 2),
            "average_contributions_dos": avg_contributions,
            "top_features_frequency": dict(top_features_overall)
        }


def format_explanation_for_display(explanation):
    """
    Format a single explanation for human-readable display.

    Parameters:
    -----------
    explanation : dict
        Explanation dictionary from SHAPExplainer

    Returns:
    --------
    str : Formatted string for display
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Record ID: {explanation['record_id']}")
    lines.append(f"Prediction: {explanation['prediction']} ({explanation['confidence']*100:.1f}% confidence)")
    lines.append("-" * 60)
    lines.append("SHAP Feature Contributions:")

    # Sort by absolute value
    sorted_shap = sorted(explanation['shap_values'].items(),
                         key=lambda x: abs(x[1]),
                         reverse=True)

    for feature, value in sorted_shap:
        feature_val = explanation['feature_values'].get(feature, 'N/A')
        sign = "+" if value >= 0 else ""
        lines.append(f"  {feature:10s}: {sign}{value:.4f}  (value: {feature_val})")

    lines.append("-" * 60)
    lines.append(f"Top contributing features: {', '.join(explanation['top_features'])}")
    lines.append("=" * 60)

    return "\n".join(lines)


# Main execution for testing
if __name__ == "__main__":
    print("SHAP Explainer Module")
    print("This module is meant to be imported, not run directly.")
    print("Use test_shap.py to test the explainer.")
