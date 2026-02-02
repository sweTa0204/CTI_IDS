"""
Generate High-Level Presentation Diagrams
==========================================

Creates clear, professional diagrams for:
1. Mitigation Framework Flow
2. Complete End-to-End Pipeline

Author: Research Project
Date: 2026-02-02
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'presentation_diagrams')
os.makedirs(IMAGES_DIR, exist_ok=True)


def draw_box(ax, x, y, width, height, text, color, text_color='white', fontsize=11):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=2.5
    )
    ax.add_patch(box)

    # Add text
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=text_color, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.6,head_length=0.8',
        color=color,
        linewidth=3,
        zorder=1
    )
    ax.add_patch(arrow)

    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label,
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))


def generate_mitigation_framework_diagram():
    """Generate Mitigation Framework high-level diagram."""

    print("\n[1/2] Generating Mitigation Framework Diagram...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Mitigation Framework Flow',
            fontsize=20, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))

    # STEP 1: XAI OUTPUT (Top)
    draw_box(ax, 4.5, 8, 5, 0.8, 'XAI OUTPUT (SHAP)', '#2C3E50', fontsize=12)
    ax.text(7, 7.6, 'Top Features: [tcprtt, dload, dmean]',
            fontsize=9, ha='center', style='italic')
    ax.text(7, 7.3, 'Confidence: 95.18%',
            fontsize=9, ha='center', style='italic')

    # Arrow down
    draw_arrow(ax, 7, 8, 7, 6.9)

    # STEP 2: ATTACK CLASSIFICATION
    draw_box(ax, 4.5, 6, 5, 0.7, 'ATTACK CLASSIFIER', '#E74C3C', fontsize=12)

    # Classification details (4 boxes)
    class_y = 5
    draw_box(ax, 0.5, class_y, 2.8, 0.5, 'Volumetric\nFlood', '#3498DB', fontsize=9)
    draw_box(ax, 3.6, class_y, 2.8, 0.5, 'Protocol\nExploit', '#9B59B6', fontsize=9)
    draw_box(ax, 6.7, class_y, 2.8, 0.5, 'Slowloris', '#F39C12', fontsize=9)
    draw_box(ax, 9.8, class_y, 2.8, 0.5, 'Amplification', '#2ECC71', fontsize=9)

    # Arrow down from classifier to types
    for x in [1.9, 5.0, 8.1, 11.2]:
        draw_arrow(ax, 7, 6, x, 5.5)

    # Selected type highlight
    ax.text(7, 4.5, '✓ Selected: Protocol Exploit',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#9B59B6',
                     edgecolor='black', linewidth=2, alpha=0.8),
            color='white')

    # Arrow down
    draw_arrow(ax, 7, 4.3, 7, 3.8)

    # STEP 3: SEVERITY ASSESSMENT
    draw_box(ax, 4.5, 3, 5, 0.7, 'SEVERITY CALCULATOR', '#F39C12', fontsize=12)

    # Severity levels
    sev_y = 2
    draw_box(ax, 1, sev_y, 2.5, 0.5, 'CRITICAL\n≥95%', '#C0392B', fontsize=9)
    draw_box(ax, 3.8, sev_y, 2.5, 0.5, 'HIGH\n90-95%', '#E67E22', fontsize=9)
    draw_box(ax, 6.6, sev_y, 2.5, 0.5, 'MEDIUM\n75-90%', '#F39C12', fontsize=9)
    draw_box(ax, 9.4, sev_y, 2.5, 0.5, 'LOW\n60-75%', '#95A5A6', fontsize=9)

    # Arrow down from severity to levels
    for x in [2.25, 5.05, 7.85, 10.65]:
        draw_arrow(ax, 7, 3, x, 2.5)

    # Selected severity
    ax.text(7, 1.4, '✓ Assessed: CRITICAL',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C0392B',
                     edgecolor='black', linewidth=2, alpha=0.8),
            color='white')

    # Arrow down
    draw_arrow(ax, 7, 1.2, 7, 0.9)

    # STEP 4: MITIGATION GENERATION
    draw_box(ax, 3, 0.1, 8, 0.7, 'MITIGATION COMMANDS GENERATED', '#27AE60', fontsize=12)

    # Footer note
    ax.text(7, -0.5, '⚡ Processing Time: <3ms per sample',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'mitigation_framework_highlevel.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def generate_complete_pipeline_diagram():
    """Generate Complete End-to-End Pipeline diagram."""

    print("\n[2/2] Generating Complete Pipeline Diagram...")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(8, 9.5, 'Complete XAI-Powered DoS Detection & Mitigation Pipeline',
            fontsize=20, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                     edgecolor='black', linewidth=2))

    # PHASE 1: INPUT
    draw_box(ax, 0.5, 7.5, 2.5, 1, 'PHASE 1\nINPUT\n\nNetwork Traffic\n10 Features',
             '#34495E', fontsize=10)
    ax.text(1.75, 6.9, 'rate, sload, sbytes,\ndload, proto, tcprtt...',
            fontsize=7, ha='center', style='italic')

    # Arrow
    draw_arrow(ax, 3, 8, 3.8, 8, '')

    # PHASE 2: DETECTION
    draw_box(ax, 4, 7.5, 2.5, 1, 'PHASE 2\nDETECT\n\nXGBoost Model\n98.14% Accuracy',
             '#2980B9', fontsize=10)
    ax.text(5.25, 6.9, 'Threshold: 0.8517\nF1 Score: 90.26%',
            fontsize=7, ha='center', style='italic')

    # Arrow
    draw_arrow(ax, 6.5, 8, 7.3, 8, 'DoS?')

    # PHASE 3: EXPLAINABILITY
    draw_box(ax, 7.5, 7.5, 2.5, 1, 'PHASE 3\nEXPLAIN\n\nSHAP TreeExplainer\nFeature Contributions',
             '#8E44AD', fontsize=10)
    ax.text(8.75, 6.9, 'Top 3 Features\nSHAP Values',
            fontsize=7, ha='center', style='italic')

    # Arrow down to mitigation framework
    draw_arrow(ax, 8.75, 7.5, 8.75, 6.5, '')

    # Mitigation Framework Section (in box)
    # Background box for mitigation framework
    mitigation_box = FancyBboxPatch(
        (0.3, 0.5), 15.4, 5.5,
        boxstyle="round,pad=0.15",
        facecolor='#ECF0F1',
        edgecolor='black',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(mitigation_box)

    ax.text(8, 5.9, 'MITIGATION FRAMEWORK',
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))

    # PHASE 4: CLASSIFICATION
    draw_box(ax, 1, 4.5, 3, 0.9, 'PHASE 4\nCLASSIFY\n\nAttack Type',
             '#E74C3C', fontsize=10)
    ax.text(2.5, 3.9, 'Volumetric / Protocol\nSlowloris / Amplification',
            fontsize=7, ha='center', style='italic')

    # Arrow right
    draw_arrow(ax, 4, 5, 5.2, 5, '')

    # PHASE 5: SEVERITY
    draw_box(ax, 5.5, 4.5, 3, 0.9, 'PHASE 5\nASSESS\n\nSeverity Level',
             '#F39C12', fontsize=10)
    ax.text(7, 3.9, 'CRITICAL / HIGH\nMEDIUM / LOW',
            fontsize=7, ha='center', style='italic')

    # Arrow right
    draw_arrow(ax, 8.5, 5, 9.7, 5, '')

    # PHASE 6: MITIGATION
    draw_box(ax, 10, 4.5, 3, 0.9, 'PHASE 6\nMITIGATE\n\nCommand Generation',
             '#27AE60', fontsize=10)
    ax.text(11.5, 3.9, 'iptables, tc, sysctl\nFirewall Rules',
            fontsize=7, ha='center', style='italic')

    # Arrow down
    draw_arrow(ax, 11.5, 4.5, 11.5, 3.5, '')

    # PHASE 7: OUTPUT
    draw_box(ax, 9.5, 2.5, 4, 0.9, 'PHASE 7: OUTPUT\n\nComplete Security Alert + Mitigation Commands',
             '#16A085', fontsize=10)

    # Example output
    output_box = FancyBboxPatch(
        (1, 0.8), 14, 1.3,
        boxstyle="round,pad=0.1",
        facecolor='#2C3E50',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)

    ax.text(8, 1.75, 'Example Alert:', fontsize=9, ha='center',
            color='white', fontweight='bold')
    ax.text(8, 1.45, 'DoS Attack Detected (95.18% confidence) | Type: Protocol Exploit | Severity: CRITICAL',
            fontsize=8, ha='center', color='white')
    ax.text(8, 1.15, 'Top Features: tcprtt, dload, dmean | Mitigation: Block IP + Rate Limiting + SYN Cookies',
            fontsize=8, ha='center', color='white')
    ax.text(8, 0.9, '✓ Commands: iptables -A INPUT -s 192.168.1.100 -j DROP | tc qdisc add dev eth0...',
            fontsize=7, ha='center', color='lightgreen', style='italic')

    # Footer stats
    ax.text(8, 0.2, '📊 Benchmark: 41,089 samples | 98.14% Accuracy | 209 False Alarms | <3ms per sample',
            fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'complete_pipeline_highlevel.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  [OK] Saved: {output_path}")
    return output_path


def main():
    """Generate all presentation diagrams."""

    print("=" * 70)
    print("GENERATING PRESENTATION DIAGRAMS")
    print("=" * 70)

    diagrams = []

    # Generate Mitigation Framework diagram
    try:
        diagram1 = generate_mitigation_framework_diagram()
        diagrams.append(diagram1)
    except Exception as e:
        print(f"  [ERROR] Mitigation Framework diagram failed: {e}")

    # Generate Complete Pipeline diagram
    try:
        diagram2 = generate_complete_pipeline_diagram()
        diagrams.append(diagram2)
    except Exception as e:
        print(f"  [ERROR] Complete Pipeline diagram failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGRAM GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(diagrams)} diagrams:")
    for diagram in diagrams:
        print(f"  - {os.path.basename(diagram)}")

    print(f"\nOutput directory: {IMAGES_DIR}")
    print("\nThese diagrams are optimized for presentations!")

    return diagrams


if __name__ == "__main__":
    main()
