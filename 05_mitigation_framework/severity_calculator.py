"""
Severity Calculator for DoS Detection
======================================

This module calculates severity levels for DoS attacks based on:
1. Model confidence (prediction probability)
2. Attack type
3. Feature magnitudes

Severity Levels:
- LOW: Monitor only
- MEDIUM: Apply rate limiting, increase logging
- HIGH: Immediate throttling, alert security team
- CRITICAL: Auto-block recommended, escalate to SOC

Author: Research Project
Date: 2026-01-29
"""

from typing import Dict, Any, List, Optional


# Severity level definitions
SEVERITY_LEVELS = {
    "LOW": {
        "level": 1,
        "color": "green",
        "description": "Low-confidence detection, requires monitoring only",
        "actions": ["Monitor traffic patterns", "Log for analysis"],
        "escalation_required": False
    },
    "MEDIUM": {
        "level": 2,
        "color": "yellow",
        "description": "Moderate-confidence detection, apply protective measures",
        "actions": ["Apply rate limiting", "Increase logging verbosity", "Monitor closely"],
        "escalation_required": False
    },
    "HIGH": {
        "level": 3,
        "color": "orange",
        "description": "High-confidence detection, immediate action required",
        "actions": ["Apply immediate throttling", "Alert security team", "Prepare blocking rules"],
        "escalation_required": True
    },
    "CRITICAL": {
        "level": 4,
        "color": "red",
        "description": "Very high confidence with severe indicators, auto-block recommended",
        "actions": ["Apply auto-blocking", "Escalate to SOC immediately", "Activate incident response"],
        "escalation_required": True
    }
}

# Attack type severity modifiers
ATTACK_TYPE_MODIFIERS = {
    "Volumetric Flood": 0.1,      # High volume attacks are often more severe
    "Protocol Exploit": 0.05,     # Protocol exploits can be targeted
    "Slowloris": 0.0,             # Slowloris is persistent but slower impact
    "Amplification": 0.15,        # Amplification can have massive impact
    "Generic DoS": 0.0            # No modifier for generic
}


class SeverityCalculator:
    """
    Calculates severity levels for DoS detections.

    Severity is determined by:
    1. Base confidence from model prediction
    2. Attack type modifier
    3. Feature-based adjustments (extreme values)
    """

    def __init__(self):
        """Initialize the severity calculator."""
        self.severity_levels = SEVERITY_LEVELS
        self.attack_modifiers = ATTACK_TYPE_MODIFIERS

    def calculate(self, classification: Dict[str, Any],
                  shap_explanation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate severity for a classified attack.

        Parameters:
        -----------
        classification : dict
            Output from AttackClassifier.classify() containing:
            - attack_type: Type of attack
            - confidence: Classification confidence
            - mitigation_category: Category for mitigation

        shap_explanation : dict, optional
            Output from SHAPExplainer.explain_single() containing:
            - confidence: Model prediction confidence
            - feature_values: Actual feature values

        Returns:
        --------
        dict : Severity assessment containing:
            - severity: Severity level (LOW/MEDIUM/HIGH/CRITICAL)
            - severity_score: Numerical score (0-1)
            - description: Human-readable description
            - recommended_actions: List of recommended actions
            - escalation_required: Whether to escalate
            - reasoning: Why this severity was assigned
        """
        # If no attack (Normal traffic), return None severity
        if classification.get('attack_type') == 'None':
            return {
                "severity": None,
                "severity_score": 0.0,
                "description": "No attack detected - no severity assessment needed",
                "recommended_actions": [],
                "escalation_required": False,
                "reasoning": "Traffic classified as normal"
            }

        # Get base confidence from SHAP explanation or classification
        if shap_explanation:
            base_confidence = shap_explanation.get('confidence', 0.5)
        else:
            base_confidence = classification.get('confidence', 0.5)

        # Get attack type modifier
        attack_type = classification.get('attack_type', 'Generic DoS')
        type_modifier = self.attack_modifiers.get(attack_type, 0.0)

        # Calculate feature-based modifier
        feature_modifier = 0.0
        if shap_explanation:
            feature_modifier = self._calculate_feature_modifier(
                shap_explanation.get('feature_values', {}),
                shap_explanation.get('shap_values', {})
            )

        # Calculate total severity score
        severity_score = min(1.0, base_confidence + type_modifier + feature_modifier)

        # Determine severity level based on score
        severity_level = self._score_to_level(severity_score)

        # Build reasoning
        reasoning = self._build_reasoning(
            base_confidence, attack_type, type_modifier,
            feature_modifier, severity_score, severity_level
        )

        return {
            "severity": severity_level,
            "severity_score": round(severity_score, 4),
            "severity_level_info": self.severity_levels[severity_level],
            "description": self.severity_levels[severity_level]["description"],
            "recommended_actions": self.severity_levels[severity_level]["actions"],
            "escalation_required": self.severity_levels[severity_level]["escalation_required"],
            "reasoning": reasoning,
            "score_breakdown": {
                "base_confidence": round(base_confidence, 4),
                "attack_type_modifier": round(type_modifier, 4),
                "feature_modifier": round(feature_modifier, 4),
                "total_score": round(severity_score, 4)
            }
        }

    def _score_to_level(self, score: float) -> str:
        """
        Convert numerical score to severity level.

        Thresholds:
        - LOW: 0.60 - 0.75
        - MEDIUM: 0.75 - 0.90
        - HIGH: 0.90 - 0.95
        - CRITICAL: 0.95+
        """
        if score >= 0.95:
            return "CRITICAL"
        elif score >= 0.90:
            return "HIGH"
        elif score >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_feature_modifier(self, feature_values: Dict,
                                     shap_values: Dict) -> float:
        """
        Calculate additional severity modifier based on extreme feature values.

        Returns a value between 0 and 0.1 based on:
        - Very high SHAP values (extreme contributions)
        - Multiple high-contributing features
        """
        modifier = 0.0

        # Count features with very high positive SHAP contribution
        extreme_count = sum(1 for v in shap_values.values() if v > 1.0)
        if extreme_count >= 3:
            modifier += 0.05
        elif extreme_count >= 2:
            modifier += 0.03

        # Check for extremely high single SHAP value
        max_shap = max(shap_values.values()) if shap_values else 0
        if max_shap > 3.0:
            modifier += 0.05
        elif max_shap > 2.0:
            modifier += 0.02

        return min(modifier, 0.1)  # Cap at 0.1

    def _build_reasoning(self, base_confidence: float, attack_type: str,
                         type_modifier: float, feature_modifier: float,
                         total_score: float, severity_level: str) -> str:
        """Build human-readable reasoning for severity assessment."""
        parts = []

        # Base confidence reasoning
        if base_confidence >= 0.95:
            parts.append(f"very high model confidence ({base_confidence*100:.1f}%)")
        elif base_confidence >= 0.90:
            parts.append(f"high model confidence ({base_confidence*100:.1f}%)")
        elif base_confidence >= 0.75:
            parts.append(f"moderate model confidence ({base_confidence*100:.1f}%)")
        else:
            parts.append(f"lower model confidence ({base_confidence*100:.1f}%)")

        # Attack type reasoning
        if type_modifier > 0:
            parts.append(f"{attack_type} attack type (+{type_modifier*100:.0f}% severity)")

        # Feature reasoning
        if feature_modifier > 0:
            parts.append(f"extreme feature contributions (+{feature_modifier*100:.0f}%)")

        reasoning = f"Severity {severity_level} assigned due to: {', '.join(parts)}. "
        reasoning += f"Total severity score: {total_score*100:.1f}%"

        return reasoning

    def calculate_batch(self, classifications: List[Dict[str, Any]],
                        shap_explanations: Optional[List[Dict[str, Any]]] = None
                        ) -> List[Dict[str, Any]]:
        """
        Calculate severity for multiple classifications.

        Parameters:
        -----------
        classifications : list
            List of classification results
        shap_explanations : list, optional
            List of SHAP explanations (same order as classifications)

        Returns:
        --------
        list : List of severity assessments
        """
        if shap_explanations is None:
            shap_explanations = [None] * len(classifications)

        severities = []
        for i, classification in enumerate(classifications):
            shap_exp = shap_explanations[i] if i < len(shap_explanations) else None
            severity = self.calculate(classification, shap_exp)
            severity['record_id'] = classification.get('record_id')
            severities.append(severity)

        return severities

    def get_severity_info(self, level: str) -> Optional[Dict]:
        """Get information about a specific severity level."""
        return self.severity_levels.get(level)

    def get_all_severity_levels(self) -> Dict:
        """Get all severity level definitions."""
        return self.severity_levels


def get_severity_statistics(severity_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics from a list of severity assessments.

    Parameters:
    -----------
    severity_results : list
        List of severity assessment dictionaries

    Returns:
    --------
    dict : Statistics including counts and percentages
    """
    # Filter out None severities (normal traffic)
    valid_results = [r for r in severity_results if r.get('severity') is not None]
    total = len(valid_results)

    if total == 0:
        return {"error": "No severity assessments to analyze"}

    # Count each severity level
    level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in valid_results:
        level = r.get('severity', 'LOW')
        level_counts[level] = level_counts.get(level, 0) + 1

    # Calculate percentages
    level_percentages = {k: round(v / total * 100, 2) for k, v in level_counts.items()}

    # Count escalation required
    escalation_count = sum(1 for r in valid_results if r.get('escalation_required', False))

    # Average severity score
    avg_score = sum(r.get('severity_score', 0) for r in valid_results) / total

    # Most severe level present
    most_severe = None
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level_counts[level] > 0:
            most_severe = level
            break

    return {
        "total_assessments": total,
        "level_counts": level_counts,
        "level_percentages": level_percentages,
        "escalation_required_count": escalation_count,
        "escalation_percentage": round(escalation_count / total * 100, 2),
        "average_severity_score": round(avg_score, 4),
        "most_severe_level": most_severe
    }


# Main execution for testing
if __name__ == "__main__":
    print("Severity Calculator Module")
    print("This module is meant to be imported, not run directly.")
    print("Use test_severity_calculator.py to test the calculator.")
