"""
Attack Classifier for DoS Detection
====================================

This module classifies DoS attacks into 4 types based on SHAP feature contributions:
1. Volumetric Flood - High rate, sload, sbytes
2. Protocol Exploit - Abnormal protocol behavior
3. Slowloris - Long duration, low rate
4. Amplification - High dload compared to sload

The classification helps determine appropriate mitigation strategies.

Author: Research Project
Date: 2026-01-29
"""

import numpy as np
from typing import Dict, List, Any, Optional


# Attack type definitions with descriptions
ATTACK_TYPES = {
    "Volumetric Flood": {
        "description": "High-volume traffic flood attempting to overwhelm network resources",
        "indicators": ["rate", "sload", "sbytes"],
        "mitigation_category": "rate_limiting"
    },
    "Protocol Exploit": {
        "description": "Attack exploiting protocol weaknesses (e.g., SYN flood, ICMP flood)",
        "indicators": ["proto", "stcpb", "dtcpb"],
        "mitigation_category": "protocol_filtering"
    },
    "Slowloris": {
        "description": "Slow, persistent attack keeping connections open to exhaust resources",
        "indicators": ["dur", "rate", "sbytes"],
        "mitigation_category": "timeout_reduction"
    },
    "Amplification": {
        "description": "Attack using amplification techniques (response larger than request)",
        "indicators": ["dload", "sload", "proto"],
        "mitigation_category": "amplification_filtering"
    },
    "Generic DoS": {
        "description": "DoS attack that doesn't fit specific patterns",
        "indicators": [],
        "mitigation_category": "general_protection"
    }
}


class AttackClassifier:
    """
    Classifies DoS attacks based on SHAP explanations.

    Uses feature contribution patterns to identify attack types:
    - Volumetric: High rate + sload contributions
    - Protocol: High proto contribution
    - Slowloris: High dur + low/negative rate
    - Amplification: dload >> sload ratio
    """

    def __init__(self):
        """Initialize the attack classifier."""
        self.attack_types = ATTACK_TYPES

    def classify(self, shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an attack based on SHAP explanation.

        Parameters:
        -----------
        shap_explanation : dict
            Output from SHAPExplainer.explain_single() containing:
            - prediction: "DoS" or "Normal"
            - shap_values: dict of feature contributions
            - top_features: list of top 3 features
            - feature_values: actual feature values
            - confidence: prediction confidence

        Returns:
        --------
        dict : Classification result containing:
            - attack_type: Name of attack type
            - attack_description: Human-readable description
            - confidence: Classification confidence
            - primary_indicators: Features that indicate this attack type
            - reasoning: Why this classification was made
            - mitigation_category: Category for mitigation lookup
        """
        # If prediction is Normal, no attack classification needed
        if shap_explanation.get('prediction') == 'Normal':
            return {
                "attack_type": "None",
                "attack_description": "Traffic classified as normal - no attack detected",
                "confidence": shap_explanation.get('confidence', 0),
                "primary_indicators": [],
                "reasoning": "Model prediction is Normal",
                "mitigation_category": None
            }

        shap_values = shap_explanation.get('shap_values', {})
        feature_values = shap_explanation.get('feature_values', {})
        top_features = shap_explanation.get('top_features', [])

        # Calculate scores for each attack type
        scores = {
            "Volumetric Flood": self._score_volumetric(shap_values, feature_values, top_features),
            "Protocol Exploit": self._score_protocol(shap_values, feature_values, top_features),
            "Slowloris": self._score_slowloris(shap_values, feature_values, top_features),
            "Amplification": self._score_amplification(shap_values, feature_values, top_features)
        }

        # Find the attack type with highest score
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # If no clear pattern, classify as Generic DoS
        if best_score < 0.3:
            best_type = "Generic DoS"
            best_score = 0.5

        # Build reasoning based on top features
        reasoning = self._build_reasoning(best_type, shap_values, feature_values, top_features)

        return {
            "attack_type": best_type,
            "attack_description": self.attack_types[best_type]["description"],
            "confidence": round(best_score, 4),
            "primary_indicators": self._get_primary_indicators(best_type, top_features),
            "reasoning": reasoning,
            "mitigation_category": self.attack_types[best_type]["mitigation_category"],
            "all_scores": {k: round(v, 4) for k, v in scores.items()}
        }

    def _score_volumetric(self, shap_values: Dict, feature_values: Dict,
                          top_features: List[str]) -> float:
        """
        Score for Volumetric Flood attack.
        High score if rate, sload, sbytes have high positive SHAP values.
        """
        score = 0.0

        # Check if volumetric indicators are in top features
        volumetric_features = ['rate', 'sload', 'sbytes']
        top_count = sum(1 for f in volumetric_features if f in top_features)
        score += top_count * 0.25

        # Check SHAP values for volumetric features
        for feature in volumetric_features:
            shap_val = shap_values.get(feature, 0)
            if shap_val > 0.5:
                score += 0.15
            elif shap_val > 0.1:
                score += 0.05

        # Bonus if rate is the top feature
        if top_features and top_features[0] == 'rate':
            score += 0.1

        return min(score, 1.0)

    def _score_protocol(self, shap_values: Dict, feature_values: Dict,
                        top_features: List[str]) -> float:
        """
        Score for Protocol Exploit attack.
        High score if proto has high positive SHAP value.
        """
        score = 0.0

        # Check if proto is a top feature
        if 'proto' in top_features:
            position = top_features.index('proto')
            score += (3 - position) * 0.2  # Higher score if proto is #1

        # Check SHAP value for proto
        proto_shap = shap_values.get('proto', 0)
        if proto_shap > 2.0:
            score += 0.4
        elif proto_shap > 1.0:
            score += 0.25
        elif proto_shap > 0.5:
            score += 0.15

        # Check TCP sequence features
        for feature in ['stcpb', 'dtcpb']:
            if shap_values.get(feature, 0) > 0.3:
                score += 0.1

        return min(score, 1.0)

    def _score_slowloris(self, shap_values: Dict, feature_values: Dict,
                         top_features: List[str]) -> float:
        """
        Score for Slowloris attack.
        High score if dur is high AND rate is low/negative.
        """
        score = 0.0

        # Check if dur is a top feature with positive contribution
        if 'dur' in top_features:
            dur_shap = shap_values.get('dur', 0)
            if dur_shap > 0.3:
                score += 0.3

        # Slowloris typically has low rate contribution or negative
        rate_shap = shap_values.get('rate', 0)
        if rate_shap < 0.1:  # Low or negative rate contribution
            score += 0.2

        # Check for high sbytes over time (persistent connection)
        sbytes_shap = shap_values.get('sbytes', 0)
        if sbytes_shap > 0.2 and rate_shap < 0.1:
            score += 0.2

        # Bonus if dur is #1 top feature
        if top_features and top_features[0] == 'dur':
            score += 0.2

        return min(score, 1.0)

    def _score_amplification(self, shap_values: Dict, feature_values: Dict,
                             top_features: List[str]) -> float:
        """
        Score for Amplification attack.
        High score if dload >> sload (response larger than request).
        """
        score = 0.0

        # Check if dload is a top feature
        if 'dload' in top_features:
            position = top_features.index('dload')
            score += (3 - position) * 0.15

        # Check SHAP values - dload should be high, sload moderate or low
        dload_shap = shap_values.get('dload', 0)
        sload_shap = shap_values.get('sload', 0)

        if dload_shap > 0.5:
            score += 0.25

        # Amplification signature: dload contribution > sload contribution
        if dload_shap > sload_shap * 1.5:
            score += 0.25

        # Check actual feature values for amplification ratio
        dload_val = feature_values.get('dload', 0)
        sload_val = feature_values.get('sload', 1)  # Avoid division by zero
        if sload_val > 0 and dload_val / (sload_val + 0.001) > 2:
            score += 0.2

        return min(score, 1.0)

    def _get_primary_indicators(self, attack_type: str,
                                top_features: List[str]) -> List[str]:
        """Get primary indicators for the classified attack type."""
        type_indicators = self.attack_types.get(attack_type, {}).get('indicators', [])

        # Return intersection of type indicators and actual top features
        primary = [f for f in top_features if f in type_indicators]

        # If no intersection, return top 2 features
        if not primary:
            primary = top_features[:2] if top_features else []

        return primary

    def _build_reasoning(self, attack_type: str, shap_values: Dict,
                         feature_values: Dict, top_features: List[str]) -> str:
        """Build human-readable reasoning for the classification."""

        if attack_type == "Volumetric Flood":
            parts = []
            if 'rate' in top_features:
                parts.append(f"high packet rate (SHAP: {shap_values.get('rate', 0):.2f})")
            if 'sload' in top_features:
                parts.append(f"high source bandwidth (SHAP: {shap_values.get('sload', 0):.2f})")
            if 'sbytes' in top_features:
                parts.append(f"high bytes transferred (SHAP: {shap_values.get('sbytes', 0):.2f})")
            return f"Classified as Volumetric Flood due to: {', '.join(parts) if parts else 'high traffic volume indicators'}"

        elif attack_type == "Protocol Exploit":
            proto_shap = shap_values.get('proto', 0)
            return f"Classified as Protocol Exploit due to: unusual protocol behavior (SHAP: {proto_shap:.2f})"

        elif attack_type == "Slowloris":
            dur_shap = shap_values.get('dur', 0)
            rate_shap = shap_values.get('rate', 0)
            return f"Classified as Slowloris due to: long connection duration (SHAP: {dur_shap:.2f}) with low rate (SHAP: {rate_shap:.2f})"

        elif attack_type == "Amplification":
            dload_shap = shap_values.get('dload', 0)
            return f"Classified as Amplification due to: high destination load (SHAP: {dload_shap:.2f}) indicating response amplification"

        else:
            return f"Classified as Generic DoS - attack pattern does not match specific type"

    def classify_batch(self, shap_explanations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple attacks.

        Parameters:
        -----------
        shap_explanations : list
            List of SHAP explanation dictionaries

        Returns:
        --------
        list : List of classification results
        """
        return [self.classify(exp) for exp in shap_explanations]

    def get_attack_type_info(self, attack_type: str) -> Optional[Dict]:
        """Get information about a specific attack type."""
        return self.attack_types.get(attack_type)

    def get_all_attack_types(self) -> Dict:
        """Get all attack type definitions."""
        return self.attack_types


def get_attack_statistics(classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics from a list of attack classifications.

    Parameters:
    -----------
    classifications : list
        List of classification results from AttackClassifier

    Returns:
    --------
    dict : Statistics including counts and percentages
    """
    total = len(classifications)
    if total == 0:
        return {"error": "No classifications provided"}

    # Count each attack type
    type_counts = {}
    for c in classifications:
        attack_type = c.get('attack_type', 'Unknown')
        type_counts[attack_type] = type_counts.get(attack_type, 0) + 1

    # Calculate percentages
    type_percentages = {k: round(v / total * 100, 2) for k, v in type_counts.items()}

    # Most common attack type
    most_common = max(type_counts, key=type_counts.get) if type_counts else None

    return {
        "total_classifications": total,
        "attack_type_counts": type_counts,
        "attack_type_percentages": type_percentages,
        "most_common_type": most_common,
        "most_common_count": type_counts.get(most_common, 0)
    }


# Main execution for testing
if __name__ == "__main__":
    print("Attack Classifier Module")
    print("This module is meant to be imported, not run directly.")
    print("Use test_attack_classifier.py to test the classifier.")
