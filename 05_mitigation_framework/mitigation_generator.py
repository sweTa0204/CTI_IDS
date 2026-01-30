"""
Mitigation Generator for DoS Detection
=======================================

This module generates specific mitigation commands based on:
1. Attack type classification
2. Severity level
3. Feature indicators

Provides actionable commands for:
- iptables (packet filtering)
- tc (traffic control)
- System configuration
- Monitoring commands

Author: Research Project
Date: 2026-01-29
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class MitigationGenerator:
    """
    Generates specific mitigation commands based on attack classification
    and severity assessment.
    """

    def __init__(self, mapping_path: Optional[str] = None):
        """
        Initialize the mitigation generator.

        Parameters:
        -----------
        mapping_path : str, optional
            Path to feature_to_action.json mapping file
        """
        if mapping_path is None:
            mapping_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'mappings', 'feature_to_action.json'
            )

        self.mapping_path = mapping_path
        self.mappings = None
        self.load_mappings()

    def load_mappings(self):
        """Load the feature-to-action mapping file."""
        try:
            with open(self.mapping_path, 'r') as f:
                self.mappings = json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] Mapping file not found: {self.mapping_path}")
            print("[WARNING] Using default mappings")
            self.mappings = self._get_default_mappings()
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in mapping file: {e}")
            self.mappings = self._get_default_mappings()

    def _get_default_mappings(self) -> Dict:
        """Return default mappings if file not found."""
        return {
            "attack_type_mitigations": {
                "Volumetric Flood": {
                    "immediate_actions": [
                        {"name": "Rate Limiting", "command": "iptables -A INPUT -s {source_ip} -m limit --limit 100/s -j ACCEPT"}
                    ]
                },
                "Protocol Exploit": {
                    "immediate_actions": [
                        {"name": "Enable SYN Cookies", "command": "echo 1 > /proc/sys/net/ipv4/tcp_syncookies"}
                    ]
                },
                "Slowloris": {
                    "immediate_actions": [
                        {"name": "Reduce Timeout", "command": "echo 10 > /proc/sys/net/ipv4/tcp_keepalive_time"}
                    ]
                },
                "Amplification": {
                    "immediate_actions": [
                        {"name": "Block DNS Amp", "command": "iptables -A INPUT -p udp --sport 53 -m length --length 512: -j DROP"}
                    ]
                },
                "Generic DoS": {
                    "immediate_actions": [
                        {"name": "Rate Limiting", "command": "iptables -A INPUT -s {source_ip} -m limit --limit 50/s -j ACCEPT"}
                    ]
                }
            }
        }

    def generate(self, classification: Dict[str, Any],
                 severity: Dict[str, Any],
                 source_ip: str = "ATTACKER_IP",
                 interface: str = "eth0") -> Dict[str, Any]:
        """
        Generate mitigation commands for a classified attack.

        Parameters:
        -----------
        classification : dict
            Output from AttackClassifier.classify()
        severity : dict
            Output from SeverityCalculator.calculate()
        source_ip : str
            Source IP address to apply mitigations to
        interface : str
            Network interface (default: eth0)

        Returns:
        --------
        dict : Mitigation recommendation containing:
            - attack_type: Type of attack
            - severity: Severity level
            - immediate_actions: Commands to apply immediately
            - alternative_actions: Alternative approaches
            - monitoring_commands: Commands for monitoring
            - human_explanation: Plain English explanation
        """
        attack_type = classification.get('attack_type', 'Generic DoS')
        severity_level = severity.get('severity')

        # If no attack, return no mitigations
        if attack_type == 'None' or severity_level is None:
            return {
                "attack_type": "None",
                "severity": None,
                "mitigations_required": False,
                "immediate_actions": [],
                "alternative_actions": [],
                "monitoring_commands": [],
                "human_explanation": "No attack detected - no mitigation required"
            }

        # Get attack-specific mitigations
        attack_mitigations = self.mappings.get('attack_type_mitigations', {}).get(
            attack_type, self.mappings.get('attack_type_mitigations', {}).get('Generic DoS', {})
        )

        # Generate timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Process immediate actions
        immediate_actions = self._process_actions(
            attack_mitigations.get('immediate_actions', []),
            source_ip, interface, timestamp
        )

        # Process alternative actions
        alternative_actions = self._process_actions(
            attack_mitigations.get('alternative_actions', []),
            source_ip, interface, timestamp
        )

        # Process monitoring commands
        monitoring_commands = self._process_actions(
            attack_mitigations.get('monitoring_commands', []),
            source_ip, interface, timestamp
        )

        # Build human explanation
        human_explanation = self._build_explanation(
            attack_type, severity_level, classification, immediate_actions
        )

        # Determine if auto-apply is recommended
        severity_actions = self.mappings.get('severity_based_actions', {}).get(severity_level, {})
        auto_apply = severity_actions.get('auto_apply', False)

        return {
            "attack_type": attack_type,
            "severity": severity_level,
            "mitigations_required": True,
            "auto_apply_recommended": auto_apply,
            "primary_strategy": attack_mitigations.get('primary_strategy', 'General protection'),
            "immediate_actions": immediate_actions,
            "alternative_actions": alternative_actions,
            "monitoring_commands": monitoring_commands,
            "human_explanation": human_explanation,
            "notification_level": severity_actions.get('notification', 'Log only'),
            "escalation_actions": severity_actions.get('actions', [])
        }

    def _process_actions(self, actions: List[Dict], source_ip: str,
                         interface: str, timestamp: str) -> List[Dict]:
        """Process action templates with variable substitution."""
        processed = []

        for action in actions:
            processed_action = {
                "name": action.get('name', 'Unnamed Action'),
                "description": action.get('description', '')
            }

            # Process single command
            if 'command' in action:
                processed_action['command'] = self._substitute_vars(
                    action['command'], source_ip, interface, timestamp
                )

            # Process followup command
            if 'followup' in action:
                processed_action['followup'] = self._substitute_vars(
                    action['followup'], source_ip, interface, timestamp
                )

            # Process multiple commands
            if 'commands' in action:
                processed_action['commands'] = [
                    self._substitute_vars(cmd, source_ip, interface, timestamp)
                    for cmd in action['commands']
                ]

            # Process config (for server configs)
            if 'config' in action:
                processed_action['config'] = action['config']

            processed.append(processed_action)

        return processed

    def _substitute_vars(self, command: str, source_ip: str,
                         interface: str, timestamp: str) -> str:
        """Substitute variables in command templates."""
        # Use simple string replacement instead of .format() to avoid
        # conflicts with shell commands that use curly braces (like awk)
        result = command
        result = result.replace('{source_ip}', source_ip)
        result = result.replace('{interface}', interface)
        result = result.replace('{timestamp}', timestamp)
        return result

    def _build_explanation(self, attack_type: str, severity: str,
                           classification: Dict, actions: List[Dict]) -> str:
        """Build human-readable explanation of mitigations."""

        explanation_parts = []

        # Attack description
        explanation_parts.append(
            f"A {attack_type} attack has been detected with {severity} severity."
        )

        # Reasoning
        reasoning = classification.get('reasoning', '')
        if reasoning:
            explanation_parts.append(f"\nWhy: {reasoning}")

        # Recommended actions
        if actions:
            explanation_parts.append("\nRecommended immediate actions:")
            for i, action in enumerate(actions[:3], 1):  # Top 3 actions
                explanation_parts.append(f"  {i}. {action['name']}: {action.get('description', '')}")

        # Severity-based guidance
        if severity == "CRITICAL":
            explanation_parts.append(
                "\n[CRITICAL] Immediate action required. Auto-blocking recommended. "
                "Escalate to SOC immediately."
            )
        elif severity == "HIGH":
            explanation_parts.append(
                "\n[HIGH] Urgent action required. Apply throttling and alert security team."
            )
        elif severity == "MEDIUM":
            explanation_parts.append(
                "\n[MEDIUM] Apply rate limiting and monitor closely."
            )
        else:
            explanation_parts.append(
                "\n[LOW] Monitor traffic patterns. No immediate action required."
            )

        return "\n".join(explanation_parts)

    def generate_batch(self, classifications: List[Dict],
                       severities: List[Dict],
                       source_ips: Optional[List[str]] = None,
                       interface: str = "eth0") -> List[Dict]:
        """
        Generate mitigations for multiple detections.

        Parameters:
        -----------
        classifications : list
            List of classification results
        severities : list
            List of severity assessments
        source_ips : list, optional
            List of source IPs (uses placeholder if not provided)
        interface : str
            Network interface

        Returns:
        --------
        list : List of mitigation recommendations
        """
        if source_ips is None:
            source_ips = ["ATTACKER_IP"] * len(classifications)

        mitigations = []
        for i, (classification, severity) in enumerate(zip(classifications, severities)):
            source_ip = source_ips[i] if i < len(source_ips) else "ATTACKER_IP"
            mitigation = self.generate(classification, severity, source_ip, interface)
            mitigation['record_id'] = classification.get('record_id')
            mitigations.append(mitigation)

        return mitigations

    def get_attack_type_info(self, attack_type: str) -> Optional[Dict]:
        """Get mitigation info for a specific attack type."""
        return self.mappings.get('attack_type_mitigations', {}).get(attack_type)

    def format_for_display(self, mitigation: Dict) -> str:
        """Format mitigation for console display."""
        lines = []
        lines.append("=" * 70)
        lines.append("MITIGATION RECOMMENDATIONS")
        lines.append("=" * 70)

        lines.append(f"\nAttack Type: {mitigation['attack_type']}")
        lines.append(f"Severity: {mitigation['severity']}")
        lines.append(f"Strategy: {mitigation.get('primary_strategy', 'N/A')}")
        lines.append(f"Auto-Apply: {'Yes' if mitigation.get('auto_apply_recommended') else 'No'}")

        # Immediate actions
        if mitigation.get('immediate_actions'):
            lines.append("\n" + "-" * 40)
            lines.append("IMMEDIATE ACTIONS")
            lines.append("-" * 40)
            for action in mitigation['immediate_actions']:
                lines.append(f"\n[{action['name']}]")
                if action.get('description'):
                    lines.append(f"  Description: {action['description']}")
                if action.get('command'):
                    lines.append(f"  Command: {action['command']}")
                if action.get('followup'):
                    lines.append(f"  Followup: {action['followup']}")
                if action.get('commands'):
                    for cmd in action['commands']:
                        lines.append(f"  -> {cmd}")

        # Alternative actions
        if mitigation.get('alternative_actions'):
            lines.append("\n" + "-" * 40)
            lines.append("ALTERNATIVE ACTIONS")
            lines.append("-" * 40)
            for action in mitigation['alternative_actions']:
                lines.append(f"\n[{action['name']}]")
                if action.get('description'):
                    lines.append(f"  Description: {action['description']}")
                if action.get('command'):
                    lines.append(f"  Command: {action['command']}")
                if action.get('commands'):
                    for cmd in action['commands']:
                        lines.append(f"  -> {cmd}")

        # Monitoring
        if mitigation.get('monitoring_commands'):
            lines.append("\n" + "-" * 40)
            lines.append("MONITORING COMMANDS")
            lines.append("-" * 40)
            for action in mitigation['monitoring_commands']:
                lines.append(f"\n[{action['name']}]")
                if action.get('command'):
                    lines.append(f"  -> {action['command']}")

        lines.append("\n" + "=" * 70)
        lines.append("HUMAN EXPLANATION")
        lines.append("=" * 70)
        lines.append(mitigation.get('human_explanation', 'No explanation available'))

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


def get_mitigation_statistics(mitigations: List[Dict]) -> Dict[str, Any]:
    """
    Get statistics from a list of mitigation recommendations.

    Parameters:
    -----------
    mitigations : list
        List of mitigation recommendation dictionaries

    Returns:
    --------
    dict : Statistics
    """
    # Filter only actual mitigations
    actual_mitigations = [m for m in mitigations if m.get('mitigations_required', False)]
    total = len(actual_mitigations)

    if total == 0:
        return {"error": "No mitigations to analyze"}

    # Count by attack type
    attack_counts = {}
    for m in actual_mitigations:
        at = m.get('attack_type', 'Unknown')
        attack_counts[at] = attack_counts.get(at, 0) + 1

    # Count auto-apply recommendations
    auto_apply_count = sum(1 for m in actual_mitigations if m.get('auto_apply_recommended', False))

    # Most common strategies
    strategies = [m.get('primary_strategy', 'Unknown') for m in actual_mitigations]
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    return {
        "total_mitigations": total,
        "attack_type_counts": attack_counts,
        "auto_apply_recommended": auto_apply_count,
        "auto_apply_percentage": round(auto_apply_count / total * 100, 2),
        "strategy_counts": strategy_counts
    }


# Main execution
if __name__ == "__main__":
    print("Mitigation Generator Module")
    print("This module is meant to be imported, not run directly.")
    print("Use test_mitigation_generator.py to test the generator.")
