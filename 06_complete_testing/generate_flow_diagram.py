"""
Generate Detailed Flow Diagram for Presentation
================================================

Creates a clean, professional flow diagram showing the complete
XAI-Powered DoS Detection and Mitigation pipeline.

Author: Research Project
Date: 2026-01-30
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')

def create_flow_diagram():
    """Create the main flow diagram."""

    fig, ax = plt.subplots(figsize=(20, 28))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)
    ax.axis('off')

    # Colors
    colors = {
        'title': '#1a1a2e',
        'phase_bg': '#16213e',
        'phase_text': '#ffffff',
        'box_input': '#e8f4f8',
        'box_process': '#fff3e0',
        'box_decision': '#fff9c4',
        'box_output': '#e8f5e9',
        'box_critical': '#ffebee',
        'arrow': '#0f4c75',
        'border': '#3282b8'
    }

    # Title
    ax.text(10, 27.5, 'XAI-POWERED DoS DETECTION AND MITIGATION SYSTEM',
            fontsize=20, fontweight='bold', ha='center', va='center',
            color=colors['title'])
    ax.text(10, 26.8, 'Complete Pipeline Flow',
            fontsize=14, ha='center', va='center', color='#666666')

    # =========================================================================
    # PHASE 1: INPUT
    # =========================================================================
    y_start = 25.5

    # Phase header
    phase1_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase1_box)
    ax.text(10, y_start, 'PHASE 1: DATA INPUT', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # Network traffic box
    y_pos = y_start - 1.5
    traffic_box = FancyBboxPatch((2, y_pos-0.6), 16, 1.2,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor=colors['box_input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(traffic_box)
    ax.text(10, y_pos+0.2, 'Network Traffic Record (10 Features)', fontsize=11, fontweight='bold',
            ha='center', va='center')
    ax.text(10, y_pos-0.25, 'rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur',
            fontsize=9, ha='center', va='center', color='#555555')

    # Arrow
    ax.annotate('', xy=(10, y_pos-1.2), xytext=(10, y_pos-0.7),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Preprocessing box
    y_pos = y_start - 3.2
    preprocess_box = FancyBboxPatch((3, y_pos-0.5), 14, 1.0,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor=colors['box_process'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(10, y_pos+0.15, 'Preprocessing', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(10, y_pos-0.2, 'Protocol Encoding (LabelEncoder) + Feature Scaling (StandardScaler)',
            fontsize=9, ha='center', va='center', color='#555555')

    # =========================================================================
    # PHASE 2: DETECTION
    # =========================================================================
    y_start = 21.0

    # Arrow from Phase 1
    ax.annotate('', xy=(10, y_start+0.5), xytext=(10, y_start+1.0),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Phase header
    phase2_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase2_box)
    ax.text(10, y_start, 'PHASE 2: DoS DETECTION (XGBoost Model)', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # XGBoost box
    y_pos = y_start - 1.5
    xgb_box = FancyBboxPatch((3, y_pos-0.5), 14, 1.0,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=colors['box_process'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(xgb_box)
    ax.text(10, y_pos+0.15, 'XGBoost Classifier', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(10, y_pos-0.2, 'Trained on 24,528 samples | Outputs P(DoS) probability',
            fontsize=9, ha='center', va='center', color='#555555')

    # Arrow
    ax.annotate('', xy=(10, y_pos-1.0), xytext=(10, y_pos-0.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Decision diamond
    y_pos = y_start - 3.3
    diamond_points = [(10, y_pos+0.6), (12, y_pos), (10, y_pos-0.6), (8, y_pos)]
    diamond = plt.Polygon(diamond_points, facecolor=colors['box_decision'],
                          edgecolor=colors['border'], linewidth=2)
    ax.add_patch(diamond)
    ax.text(10, y_pos+0.1, 'P(DoS) >=', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(10, y_pos-0.2, '0.8517?', fontsize=9, fontweight='bold', ha='center', va='center')

    # Yes arrow (down)
    ax.annotate('', xy=(10, y_pos-1.3), xytext=(10, y_pos-0.65),
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))
    ax.text(10.3, y_pos-1.0, 'YES', fontsize=9, fontweight='bold', color='#2e7d32')

    # No arrow (right)
    ax.annotate('', xy=(16, y_pos), xytext=(12.1, y_pos),
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=2))
    ax.text(14, y_pos+0.25, 'NO', fontsize=9, fontweight='bold', color='#c62828')

    # Normal traffic box (right)
    normal_box = FancyBboxPatch((16, y_pos-0.4), 3, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor='#e0e0e0', edgecolor='#9e9e9e', linewidth=2)
    ax.add_patch(normal_box)
    ax.text(17.5, y_pos, 'Normal\nNo Action', fontsize=8, ha='center', va='center')

    # =========================================================================
    # PHASE 3: EXPLAINABILITY
    # =========================================================================
    y_start = 16.0

    # Phase header
    phase3_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase3_box)
    ax.text(10, y_start, 'PHASE 3: EXPLAINABILITY (SHAP TreeExplainer)', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # SHAP box
    y_pos = y_start - 1.8
    shap_box = FancyBboxPatch((2, y_pos-0.8), 16, 1.6,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=colors['box_process'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(shap_box)
    ax.text(10, y_pos+0.4, 'SHAP Calculates Feature Contributions', fontsize=11, fontweight='bold',
            ha='center', va='center')
    ax.text(10, y_pos, 'Each feature gets a SHAP value showing how much it pushed toward DoS or Normal',
            fontsize=9, ha='center', va='center', color='#555555')
    ax.text(10, y_pos-0.4, 'Output: Top 3 contributing features (e.g., proto: +4.08, sload: +2.48, sbytes: +0.74)',
            fontsize=9, ha='center', va='center', color='#555555', style='italic')

    # Arrow
    ax.annotate('', xy=(10, y_pos-1.4), xytext=(10, y_pos-0.9),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # =========================================================================
    # PHASE 4: CLASSIFICATION
    # =========================================================================
    y_start = 12.5

    # Phase header
    phase4_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase4_box)
    ax.text(10, y_start, 'PHASE 4: ATTACK CLASSIFICATION', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # Classification boxes
    y_pos = y_start - 1.6

    # Box 1: Volumetric
    vol_box = FancyBboxPatch((1.5, y_pos-0.5), 4, 1.0,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#ffcdd2', edgecolor='#c62828', linewidth=2)
    ax.add_patch(vol_box)
    ax.text(3.5, y_pos+0.2, 'VOLUMETRIC', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(3.5, y_pos-0.15, 'rate, sbytes, sload', fontsize=8, ha='center', va='center', color='#555555')

    # Box 2: Protocol
    proto_box = FancyBboxPatch((6, y_pos-0.5), 4, 1.0,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor='#fff9c4', edgecolor='#f9a825', linewidth=2)
    ax.add_patch(proto_box)
    ax.text(8, y_pos+0.2, 'PROTOCOL', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(8, y_pos-0.15, 'proto, tcprtt, stcpb', fontsize=8, ha='center', va='center', color='#555555')

    # Box 3: Slowloris
    slow_box = FancyBboxPatch((10.5, y_pos-0.5), 4, 1.0,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(slow_box)
    ax.text(12.5, y_pos+0.2, 'SLOWLORIS', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(12.5, y_pos-0.15, 'dur, dmean', fontsize=8, ha='center', va='center', color='#555555')

    # Box 4: Amplification
    amp_box = FancyBboxPatch((15, y_pos-0.5), 4, 1.0,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#bbdefb', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(amp_box)
    ax.text(17, y_pos+0.2, 'AMPLIFICATION', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(17, y_pos-0.15, 'dload, dbytes', fontsize=8, ha='center', va='center', color='#555555')

    # Arrow
    ax.annotate('', xy=(10, y_pos-1.1), xytext=(10, y_pos-0.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # =========================================================================
    # PHASE 5: SEVERITY
    # =========================================================================
    y_start = 9.2

    # Phase header
    phase5_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase5_box)
    ax.text(10, y_start, 'PHASE 5: SEVERITY ASSESSMENT', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # Severity boxes
    y_pos = y_start - 1.6

    # Critical
    crit_box = FancyBboxPatch((1.5, y_pos-0.5), 4, 1.0,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#b71c1c', edgecolor='#7f0000', linewidth=2)
    ax.add_patch(crit_box)
    ax.text(3.5, y_pos+0.2, 'CRITICAL', fontsize=9, fontweight='bold', ha='center', va='center', color='white')
    ax.text(3.5, y_pos-0.15, '>= 95%', fontsize=8, ha='center', va='center', color='#ffcdd2')

    # High
    high_box = FancyBboxPatch((6, y_pos-0.5), 4, 1.0,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#e65100', edgecolor='#bf360c', linewidth=2)
    ax.add_patch(high_box)
    ax.text(8, y_pos+0.2, 'HIGH', fontsize=9, fontweight='bold', ha='center', va='center', color='white')
    ax.text(8, y_pos-0.15, '90-95%', fontsize=8, ha='center', va='center', color='#ffe0b2')

    # Medium
    med_box = FancyBboxPatch((10.5, y_pos-0.5), 4, 1.0,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#ffc107', edgecolor='#ff8f00', linewidth=2)
    ax.add_patch(med_box)
    ax.text(12.5, y_pos+0.2, 'MEDIUM', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(12.5, y_pos-0.15, '75-90%', fontsize=8, ha='center', va='center', color='#555555')

    # Low
    low_box = FancyBboxPatch((15, y_pos-0.5), 4, 1.0,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#4caf50', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(low_box)
    ax.text(17, y_pos+0.2, 'LOW', fontsize=9, fontweight='bold', ha='center', va='center', color='white')
    ax.text(17, y_pos-0.15, '60-75%', fontsize=8, ha='center', va='center', color='#c8e6c9')

    # Arrow
    ax.annotate('', xy=(10, y_pos-1.1), xytext=(10, y_pos-0.55),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # =========================================================================
    # PHASE 6: MITIGATION
    # =========================================================================
    y_start = 5.9

    # Phase header
    phase6_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase6_box)
    ax.text(10, y_start, 'PHASE 6: MITIGATION GENERATION', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # Mitigation commands box
    y_pos = y_start - 2.0
    mit_box = FancyBboxPatch((2, y_pos-1.0), 16, 2.0,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#f5f5f5', edgecolor=colors['border'], linewidth=2)
    ax.add_patch(mit_box)
    ax.text(10, y_pos+0.6, 'Generate Actionable Commands', fontsize=11, fontweight='bold',
            ha='center', va='center')
    ax.text(10, y_pos+0.15, 'Rate Limiting:  tc qdisc add dev eth0 root tbf rate 100mbit',
            fontsize=9, ha='center', va='center', family='monospace', color='#1565c0')
    ax.text(10, y_pos-0.2, 'Firewall:  iptables -A INPUT -s <source_ip> -j DROP',
            fontsize=9, ha='center', va='center', family='monospace', color='#1565c0')
    ax.text(10, y_pos-0.55, 'SYN Protection:  sysctl -w net.ipv4.tcp_syncookies=1',
            fontsize=9, ha='center', va='center', family='monospace', color='#1565c0')

    # Arrow
    ax.annotate('', xy=(10, y_pos-1.6), xytext=(10, y_pos-1.1),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # =========================================================================
    # PHASE 7: OUTPUT
    # =========================================================================
    y_start = 2.2

    # Phase header
    phase7_box = FancyBboxPatch((1, y_start-0.4), 18, 0.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['phase_bg'], edgecolor='none')
    ax.add_patch(phase7_box)
    ax.text(10, y_start, 'PHASE 7: SECURITY ALERT OUTPUT', fontsize=12, fontweight='bold',
            ha='center', va='center', color=colors['phase_text'])

    # Output box
    y_pos = y_start - 1.3
    output_box = FancyBboxPatch((3, y_pos-0.6), 14, 1.2,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=colors['box_output'], edgecolor='#2e7d32', linewidth=3)
    ax.add_patch(output_box)
    ax.text(10, y_pos+0.25, 'COMPLETE SECURITY ALERT', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#1b5e20')
    ax.text(10, y_pos-0.2, 'Detection + Explanation + Attack Type + Severity + Mitigation Commands',
            fontsize=9, ha='center', va='center', color='#555555')

    # Save
    plt.tight_layout()
    plt.savefig('pipeline_flow_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Saved: pipeline_flow_diagram.png")


def create_simple_flow():
    """Create a simpler horizontal flow for quick overview."""

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(9, 5.5, 'XAI-Powered DoS Detection Pipeline - Overview',
            fontsize=16, fontweight='bold', ha='center', va='center')

    # Boxes
    boxes = [
        (0.5, 2, 'INPUT', 'Network\nTraffic\n(10 features)', '#e3f2fd', '#1976d2'),
        (3.3, 2, 'DETECT', 'XGBoost\nClassifier\n(98% acc)', '#fff3e0', '#f57c00'),
        (6.1, 2, 'EXPLAIN', 'SHAP\nExplainer\n(WHY?)', '#f3e5f5', '#7b1fa2'),
        (8.9, 2, 'CLASSIFY', 'Attack\nType\n(4 types)', '#ffebee', '#c62828'),
        (11.7, 2, 'ASSESS', 'Severity\nLevel\n(4 levels)', '#fff8e1', '#ff8f00'),
        (14.5, 2, 'MITIGATE', 'iptables\ntc commands\n(action)', '#e8f5e9', '#2e7d32'),
    ]

    for x, y, title, content, bg_color, border_color in boxes:
        box = FancyBboxPatch((x, y-0.8), 2.5, 2.2,
                              boxstyle="round,pad=0.02,rounding_size=0.2",
                              facecolor=bg_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(box)
        ax.text(x+1.25, y+1.0, title, fontsize=10, fontweight='bold',
                ha='center', va='center', color=border_color)
        ax.text(x+1.25, y+0.1, content, fontsize=9, ha='center', va='center',
                color='#333333', linespacing=1.2)

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#37474f', lw=2)
    for i in range(5):
        x_start = 0.5 + 2.5 + i * 2.8
        ax.annotate('', xy=(x_start + 0.3, 2.6), xytext=(x_start, 2.6),
                    arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('pipeline_simple_overview.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Saved: pipeline_simple_overview.png")


if __name__ == "__main__":
    print("\nGenerating Flow Diagrams for Presentation...")
    print("=" * 50)

    create_flow_diagram()
    create_simple_flow()

    print("\nDone! Files created:")
    print("  1. pipeline_flow_diagram.png (Detailed 7-phase flow)")
    print("  2. pipeline_simple_overview.png (Simple horizontal overview)")
