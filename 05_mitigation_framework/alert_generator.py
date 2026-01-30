"""
Alert Generator for DoS Detection
==================================

This module combines all components to generate complete detection alerts:
1. SHAP explanation (from XAI module)
2. Attack classification
3. Severity assessment
4. Mitigation recommendations

Produces human-readable alerts with actionable information.

Author: Research Project
Date: 2026-01-29
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack_classifier import AttackClassifier
from severity_calculator import SeverityCalculator
from mitigation_generator import MitigationGenerator


class AlertGenerator:
    """
    Generates complete detection alerts by combining all framework components.
    """

    def __init__(self):
        """Initialize the alert generator with all required components."""
        self.classifier = AttackClassifier()
        self.severity_calc = SeverityCalculator()
        self.mitigation_gen = MitigationGenerator()

    def generate_alert(self, shap_explanation: Dict[str, Any],
                       source_ip: str = "UNKNOWN",
                       destination_ip: str = "UNKNOWN",
                       interface: str = "eth0") -> Dict[str, Any]:
        """
        Generate a complete alert from a SHAP explanation.

        Parameters:
        -----------
        shap_explanation : dict
            Output from SHAPExplainer.explain_single()
        source_ip : str
            Source IP address
        destination_ip : str
            Destination IP address
        interface : str
            Network interface

        Returns:
        --------
        dict : Complete alert containing all components
        """
        timestamp = datetime.now().isoformat()

        # Step 1: Classify the attack
        classification = self.classifier.classify(shap_explanation)

        # Step 2: Calculate severity
        severity = self.severity_calc.calculate(classification, shap_explanation)

        # Step 3: Generate mitigations
        mitigation = self.mitigation_gen.generate(
            classification, severity, source_ip, interface
        )

        # Build complete alert
        alert = {
            "alert_id": f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{shap_explanation.get('record_id', 0)}",
            "timestamp": timestamp,
            "record_id": shap_explanation.get('record_id'),

            # Detection info
            "detection": {
                "prediction": shap_explanation.get('prediction'),
                "confidence": shap_explanation.get('confidence'),
                "probability_dos": shap_explanation.get('probability_dos'),
                "probability_normal": shap_explanation.get('probability_normal')
            },

            # Network info
            "network_info": {
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "interface": interface
            },

            # XAI explanation
            "xai_explanation": {
                "method": "SHAP TreeExplainer",
                "top_features": shap_explanation.get('top_features', []),
                "shap_values": shap_explanation.get('shap_values', {}),
                "feature_values": shap_explanation.get('feature_values', {}),
                "base_value": shap_explanation.get('base_value')
            },

            # Attack classification
            "classification": {
                "attack_type": classification.get('attack_type'),
                "attack_description": classification.get('attack_description'),
                "classification_confidence": classification.get('confidence'),
                "primary_indicators": classification.get('primary_indicators'),
                "reasoning": classification.get('reasoning'),
                "mitigation_category": classification.get('mitigation_category')
            },

            # Severity assessment
            "severity": {
                "level": severity.get('severity'),
                "score": severity.get('severity_score'),
                "description": severity.get('description'),
                "escalation_required": severity.get('escalation_required'),
                "reasoning": severity.get('reasoning')
            },

            # Mitigation recommendations
            "mitigation": {
                "required": mitigation.get('mitigations_required', False),
                "auto_apply_recommended": mitigation.get('auto_apply_recommended', False),
                "primary_strategy": mitigation.get('primary_strategy'),
                "immediate_actions": mitigation.get('immediate_actions', []),
                "alternative_actions": mitigation.get('alternative_actions', []),
                "monitoring_commands": mitigation.get('monitoring_commands', [])
            },

            # Action checklist
            "action_checklist": self._generate_checklist(classification, severity, mitigation),

            # Human explanation
            "human_explanation": self._generate_human_explanation(
                shap_explanation, classification, severity, mitigation
            )
        }

        return alert

    def _generate_checklist(self, classification: Dict, severity: Dict,
                            mitigation: Dict) -> List[str]:
        """Generate action checklist based on severity."""
        checklist = []

        if classification.get('attack_type') == 'None':
            return ["No action required - traffic is normal"]

        severity_level = severity.get('severity', 'LOW')

        # Add actions based on severity
        if severity_level in ["HIGH", "CRITICAL"]:
            checklist.append("Apply immediate mitigation (see commands below)")
            checklist.append("Alert security team")

        if severity_level == "CRITICAL":
            checklist.append("Escalate to SOC immediately")
            checklist.append("Consider auto-blocking")

        checklist.append("Monitor traffic for 5 minutes")
        checklist.append("Capture traffic for analysis if attack persists")
        checklist.append("Document incident in ticketing system")

        return checklist

    def _generate_human_explanation(self, shap_exp: Dict, classification: Dict,
                                    severity: Dict, mitigation: Dict) -> str:
        """Generate complete human-readable explanation."""
        lines = []

        prediction = shap_exp.get('prediction', 'Unknown')
        confidence = shap_exp.get('confidence', 0) * 100

        if prediction == "Normal":
            return "This traffic has been classified as NORMAL. No attack detected."

        lines.append(f"This traffic has been classified as a DoS attack with {confidence:.1f}% confidence.")
        lines.append("")

        # Why it's an attack
        top_features = shap_exp.get('top_features', [])
        shap_values = shap_exp.get('shap_values', {})
        feature_values = shap_exp.get('feature_values', {})

        lines.append("WHY THIS IS FLAGGED AS AN ATTACK:")
        for i, feature in enumerate(top_features[:3], 1):
            shap_val = shap_values.get(feature, 0)
            feat_val = feature_values.get(feature, 0)
            direction = "increases" if shap_val > 0 else "decreases"
            lines.append(f"  {i}. {feature} = {feat_val:.2f} (SHAP: {shap_val:+.2f}) - {direction} attack likelihood")

        lines.append("")

        # Attack type
        attack_type = classification.get('attack_type', 'Unknown')
        lines.append(f"ATTACK TYPE: {attack_type}")
        lines.append(f"  {classification.get('attack_description', '')}")
        lines.append("")

        # Severity
        severity_level = severity.get('severity', 'Unknown')
        lines.append(f"SEVERITY: {severity_level}")
        lines.append(f"  {severity.get('description', '')}")

        if severity.get('escalation_required'):
            lines.append("  [!] ESCALATION REQUIRED")

        lines.append("")

        # What to do
        lines.append("RECOMMENDED ACTION:")
        lines.append(f"  {mitigation.get('primary_strategy', 'Apply general protection')}")

        return "\n".join(lines)

    def generate_batch(self, shap_explanations: List[Dict],
                       source_ips: Optional[List[str]] = None,
                       destination_ip: str = "UNKNOWN",
                       interface: str = "eth0") -> List[Dict]:
        """
        Generate alerts for multiple detections.

        Parameters:
        -----------
        shap_explanations : list
            List of SHAP explanation dictionaries
        source_ips : list, optional
            List of source IPs (uses placeholder if not provided)
        destination_ip : str
            Destination IP address
        interface : str
            Network interface

        Returns:
        --------
        list : List of complete alerts
        """
        if source_ips is None:
            source_ips = ["UNKNOWN"] * len(shap_explanations)

        alerts = []
        for i, shap_exp in enumerate(shap_explanations):
            source_ip = source_ips[i] if i < len(source_ips) else "UNKNOWN"
            alert = self.generate_alert(shap_exp, source_ip, destination_ip, interface)
            alerts.append(alert)

        return alerts

    def format_for_console(self, alert: Dict) -> str:
        """Format alert for console display."""
        lines = []

        # Header
        lines.append("")
        lines.append("=" * 75)
        lines.append("                         DoS DETECTION ALERT")
        lines.append("=" * 75)

        # Detection summary
        lines.append("")
        lines.append("DETECTION SUMMARY")
        lines.append("-" * 75)
        lines.append(f"Alert ID:         {alert['alert_id']}")
        lines.append(f"Timestamp:        {alert['timestamp']}")
        lines.append(f"Record ID:        {alert['record_id']}")
        lines.append(f"Source IP:        {alert['network_info']['source_ip']}")
        lines.append(f"Destination IP:   {alert['network_info']['destination_ip']}")
        lines.append(f"Verdict:          {alert['detection']['prediction']}")
        lines.append(f"Confidence:       {alert['detection']['confidence']*100:.2f}%")

        # Check if normal traffic
        if alert['detection']['prediction'] == 'Normal':
            lines.append("")
            lines.append("STATUS: Normal traffic - no attack detected")
            lines.append("=" * 75)
            return "\n".join(lines)

        # XAI Explanation
        lines.append("")
        lines.append("=" * 75)
        lines.append("                         XAI EXPLANATION")
        lines.append("=" * 75)
        lines.append("")
        lines.append("TOP CONTRIBUTING FEATURES:")
        lines.append("-" * 75)

        xai = alert['xai_explanation']
        for feature in xai['top_features'][:5]:
            shap_val = xai['shap_values'].get(feature, 0)
            feat_val = xai['feature_values'].get(feature, 0)
            direction = "-> DoS" if shap_val > 0 else "-> Normal"
            lines.append(f"  {feature:12s}  SHAP: {shap_val:+.4f}  Value: {feat_val:.4f}  {direction}")

        # Attack Classification
        lines.append("")
        lines.append("=" * 75)
        lines.append("                      ATTACK CLASSIFICATION")
        lines.append("=" * 75)
        lines.append("")

        classification = alert['classification']
        severity = alert['severity']

        lines.append(f"Attack Type:      {classification['attack_type']}")
        lines.append(f"Severity:         {severity['level']}")
        lines.append(f"Escalation:       {'REQUIRED' if severity['escalation_required'] else 'Not required'}")
        lines.append("")
        lines.append(f"Description:")
        lines.append(f"  {classification['attack_description']}")
        lines.append("")
        lines.append(f"Reasoning:")
        lines.append(f"  {classification['reasoning']}")

        # Mitigation Recommendations
        lines.append("")
        lines.append("=" * 75)
        lines.append("                    RECOMMENDED MITIGATIONS")
        lines.append("=" * 75)
        lines.append("")

        mitigation = alert['mitigation']
        lines.append(f"Strategy: {mitigation['primary_strategy']}")
        lines.append(f"Auto-Apply: {'Recommended' if mitigation['auto_apply_recommended'] else 'Manual review suggested'}")
        lines.append("")

        if mitigation['immediate_actions']:
            lines.append("IMMEDIATE ACTIONS:")
            lines.append("-" * 75)
            for action in mitigation['immediate_actions']:
                lines.append(f"  [{action['name']}]")
                if action.get('command'):
                    lines.append(f"    $ {action['command']}")
                if action.get('followup'):
                    lines.append(f"    $ {action['followup']}")
            lines.append("")

        if mitigation['monitoring_commands']:
            lines.append("MONITORING COMMANDS:")
            lines.append("-" * 75)
            for cmd in mitigation['monitoring_commands'][:2]:
                lines.append(f"  [{cmd['name']}]")
                if cmd.get('command'):
                    lines.append(f"    $ {cmd['command']}")
            lines.append("")

        # Action Checklist
        lines.append("=" * 75)
        lines.append("                        ACTION CHECKLIST")
        lines.append("=" * 75)
        lines.append("")
        for item in alert['action_checklist']:
            lines.append(f"  [ ] {item}")

        lines.append("")
        lines.append("=" * 75)

        return "\n".join(lines)

    def save_alert(self, alert: Dict, output_path: str):
        """Save alert to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(alert, f, indent=2)

    def save_alerts(self, alerts: List[Dict], output_path: str):
        """Save multiple alerts to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(alerts, f, indent=2)


def get_alert_statistics(alerts: List[Dict]) -> Dict[str, Any]:
    """
    Get statistics from a list of alerts.

    Parameters:
    -----------
    alerts : list
        List of alert dictionaries

    Returns:
    --------
    dict : Statistics
    """
    total = len(alerts)
    if total == 0:
        return {"error": "No alerts to analyze"}

    # Count detections
    dos_count = sum(1 for a in alerts if a['detection']['prediction'] == 'DoS')
    normal_count = total - dos_count

    # Count by attack type
    attack_types = {}
    for a in alerts:
        at = a['classification'].get('attack_type', 'Unknown')
        if at != 'None':
            attack_types[at] = attack_types.get(at, 0) + 1

    # Count by severity
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for a in alerts:
        level = a['severity'].get('level')
        if level and level in severity_counts:
            severity_counts[level] += 1

    # Count escalation required
    escalation_count = sum(1 for a in alerts if a['severity'].get('escalation_required', False))

    return {
        "total_alerts": total,
        "dos_detections": dos_count,
        "normal_traffic": normal_count,
        "attack_type_distribution": attack_types,
        "severity_distribution": severity_counts,
        "escalation_required": escalation_count,
        "escalation_percentage": round(escalation_count / dos_count * 100, 2) if dos_count > 0 else 0
    }


# Main execution
if __name__ == "__main__":
    print("Alert Generator Module")
    print("This module is meant to be imported, not run directly.")
    print("Use main.py to run the complete system.")
