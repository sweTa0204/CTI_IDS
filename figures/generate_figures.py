"""
Generate Publication-Quality Figures for CTI-IDS Project Documentation
Figures 5.1 through 5.5

Author: Generated for Akash Madanu's project
"Detection to Defense: An XAI-Powered DoS Prevention System"
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Color palette
SHAP_POS = '#FF0051'
SHAP_NEG = '#008BFB'
BLUE = '#2563EB'
TEAL = '#0D9488'
AMBER = '#F59E0B'
RED = '#DC2626'
SLATE = '#475569'
LIGHT_BG = '#F8FAFC'
BORDER = '#E2E8F0'


# ============================================================
# FIGURE 5.1: CONFUSION MATRIX
# ============================================================
def generate_figure_5_1():
    """Confusion Matrix of XGBoost Model on External Test Dataset"""
    print("Generating Figure 5.1: Confusion Matrix...")

    fig, ax = plt.subplots(figsize=(8, 7.5))

    cm = np.array([[36791, 209],
                    [554,  3535]])
    total = cm.sum()
    labels = ['Normal', 'Attack']

    cmap = LinearSegmentedColormap.from_list('cm_blue', ['#EFF6FF', '#1E40AF'])
    cm_norm = cm.astype(float) / cm.max()

    # Draw cells
    for i in range(2):
        for j in range(2):
            color = cmap(cm_norm[i, j])
            text_color = 'white' if cm_norm[i, j] > 0.35 else '#1E293B'

            rect = FancyBboxPatch(
                (j - 0.46, i - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor='white', linewidth=3
            )
            ax.add_patch(rect)

            # Count
            ax.text(j, i - 0.1, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=24, fontweight='bold',
                    color=text_color)

            # Percentage
            pct = cm[i, j] / total * 100
            ax.text(j, i + 0.22, f'({pct:.2f}%)',
                    ha='center', va='center', fontsize=12,
                    color=text_color, alpha=0.85)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(1.55, -0.55)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_xlabel('\nPredicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label\n', fontsize=14, fontweight='bold')

    ax.set_title(
        'Confusion Matrix of XGBoost Model\n'
        'External Test Dataset (N = 41,089 | Threshold = 0.8517)',
        fontsize=14, fontweight='bold', pad=20, linespacing=1.5
    )

    # Metrics box
    acc = (cm[0, 0] + cm[1, 1]) / total * 100
    prec = cm[1, 1] / (cm[1, 1] + cm[0, 1]) * 100
    rec = cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100
    f1 = 2 * prec * rec / (prec + rec)
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) * 100

    metrics = (
        f'Accuracy: {acc:.2f}%    Precision: {prec:.2f}%    '
        f'Recall: {rec:.2f}%    F1: {f1:.2f}%    FPR: {fpr:.2f}%'
    )
    fig.text(
        0.5, 0.02, metrics,
        ha='center', va='bottom', fontsize=10.5, color=SLATE,
        bbox=dict(boxstyle='round,pad=0.6', facecolor=LIGHT_BG,
                  edgecolor=BORDER, linewidth=1)
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.subplots_adjust(bottom=0.13)
    path = os.path.join(OUTPUT_DIR, 'figure_5_1_confusion_matrix.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {path}')


# ============================================================
# FIGURE 5.2: COMPARATIVE MODEL PERFORMANCE
# ============================================================
def generate_figure_5_2():
    """Comparative Performance of Trained Models on External Test Dataset"""
    print("Generating Figure 5.2: Comparative Performance...")

    models   = ['XGBoost', 'Random\nForest', 'Decision\nTree', '1D-CNN',
                'MLP', 'LSTM', 'SVM', 'Logistic\nRegression']
    accuracy  = [98.14, 97.93, 97.83, 97.42, 97.14, 96.89, 95.86, 88.42]
    precision = [94.42, 89.86, 93.43, 90.92, 88.43, 88.12, 82.47, 44.48]
    recall    = [86.45, 89.26, 84.13, 82.27, 82.02, 79.48, 74.10, 66.05]
    f1        = [90.26, 89.56, 88.53, 86.38, 85.11, 83.58, 78.06, 53.16]

    x = np.arange(len(models))
    width = 0.19

    fig, ax = plt.subplots(figsize=(15, 7.5))

    colors = ['#2563EB', '#0D9488', '#F59E0B', '#DC2626']

    b1 = ax.bar(x - 1.5*width, accuracy,  width, label='Accuracy',
                color=colors[0], edgecolor='white', linewidth=0.5, zorder=3)
    b2 = ax.bar(x - 0.5*width, precision, width, label='Precision',
                color=colors[1], edgecolor='white', linewidth=0.5, zorder=3)
    b3 = ax.bar(x + 0.5*width, recall,    width, label='Recall',
                color=colors[2], edgecolor='white', linewidth=0.5, zorder=3)
    b4 = ax.bar(x + 1.5*width, f1,        width, label='F1 Score',
                color=colors[3], edgecolor='white', linewidth=0.5, zorder=3)

    # Value labels
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            if h > 55:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=6.5, color=SLATE, rotation=90)

    # Highlight best model
    ax.axvspan(-0.5, 0.5, alpha=0.07, color=BLUE, zorder=0)
    ax.text(0, 38, 'Best Model', ha='center', fontsize=9,
            color=BLUE, fontweight='bold', style='italic')

    ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Comparative Performance of Trained Models\n'
        'External Test Dataset (Optimized Thresholds)',
        fontsize=14, fontweight='bold', pad=18, linespacing=1.5
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(30, 110)
    ax.set_yticks(range(40, 101, 10))

    ax.yaxis.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
              edgecolor=BORDER, ncol=4, bbox_to_anchor=(0.99, 0.99))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_5_2_comparative_performance.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {path}')


# ============================================================
# FIGURE 5.3: SHAP SUMMARY PLOT (Global Feature Importance)
# ============================================================
def generate_figure_5_3():
    """SHAP Summary Plot Showing Global Feature Importance"""
    print("Generating Figure 5.3: SHAP Summary Plot...")

    # Mean |SHAP| values for DoS class (from sample_shap_output.json)
    features  = ['proto', 'sload', 'tcprtt', 'dload', 'sbytes',
                 'stcpb', 'dur', 'dtcpb', 'rate', 'dmean']
    mean_shap = [2.2363, 1.3953, 0.4639, 0.4604, 0.2469,
                 0.1501, 0.0357, 0.0265, 0.0081, 0.0022]

    full_names = {
        'proto':  'Protocol Type (proto)',
        'sload':  'Source Load (sload)',
        'tcprtt': 'TCP Round-Trip Time (tcprtt)',
        'dload':  'Destination Load (dload)',
        'sbytes': 'Source Bytes (sbytes)',
        'stcpb':  'Source TCP Base Seq (stcpb)',
        'dur':    'Duration (dur)',
        'dtcpb':  'Dest TCP Base Seq (dtcpb)',
        'rate':   'Connection Rate (rate)',
        'dmean':  'Dest Packet Mean (dmean)',
    }

    # Reverse for horizontal bars (most important at top)
    features_r = features[::-1]
    values_r   = mean_shap[::-1]
    names_r    = [full_names[f] for f in features_r]

    fig, ax = plt.subplots(figsize=(11, 7))

    max_val = max(values_r)
    cmap = LinearSegmentedColormap.from_list('shap_bar', ['#C7D2FE', '#EF4444'])
    colors = [cmap(v / max_val) for v in values_r]

    bars = ax.barh(range(len(features_r)), values_r, height=0.62,
                   color=colors, edgecolor='white', linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values_r):
        ax.text(val + max_val * 0.015, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left',
                fontsize=10.5, color=SLATE, fontweight='bold')

    ax.set_yticks(range(len(names_r)))
    ax.set_yticklabels(names_r, fontsize=11)
    ax.set_xlabel('Mean |SHAP Value|  (Average Impact on Model Output)',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        'SHAP Global Feature Importance\n'
        'XGBoost DoS Detection Model',
        fontsize=14, fontweight='bold', pad=18, linespacing=1.5
    )

    ax.text(0.97, 0.06,
            'Higher |SHAP| value = greater\ninfluence on prediction',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=SLATE, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=LIGHT_BG,
                      edgecolor=BORDER))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.xaxis.grid(True, alpha=0.25, linestyle='--', zorder=0)
    ax.set_xlim(0, max_val * 1.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_5_3_shap_summary.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {path}')


# ============================================================
# FIGURE 5.4: SHAP WATERFALL PLOT
# ============================================================
def generate_figure_5_4():
    """SHAP Waterfall Plot for a Sample DoS Detection"""
    print("Generating Figure 5.4: SHAP Waterfall Plot...")

    # Record 20459 — DoS, 99.96% confidence (from sample_shap_output.json)
    base_value = 0.0032

    features    = ['proto', 'sload', 'sbytes', 'dload', 'stcpb',
                   'dtcpb',  'tcprtt', 'dmean',  'dur',   'rate']
    shap_values = [4.0827,  2.4836,  0.7366,  0.6995,  0.2405,
                   0.0256, -0.1031, -0.0978, -0.1351, -0.0673]

    # Sort by absolute SHAP value ascending (smallest at bottom)
    idx = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]))
    s_feat = [features[i]    for i in idx]
    s_shap = [shap_values[i] for i in idx]

    # Cumulative sums bottom → top
    cumulative = [base_value]
    for sv in s_shap:
        cumulative.append(cumulative[-1] + sv)
    final_value = cumulative[-1]

    n = len(s_feat)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    for i in range(n):
        start = cumulative[i]
        w = s_shap[i]
        color = SHAP_POS if w >= 0 else SHAP_NEG

        ax.barh(i, w, left=start, height=0.55,
                color=color, edgecolor='white', linewidth=0.5, zorder=3)

        # Connector line
        if i < n - 1:
            end_x = cumulative[i + 1]
            ax.plot([end_x, end_x], [i + 0.275, i + 0.725],
                    color='#CBD5E1', linewidth=1.0, zorder=2)

        # Value label — always place to the RIGHT of the bar to avoid
        # overlapping with the y-axis feature names.
        if abs(w) > 0.30:
            ax.text(start + w / 2, i, f'{w:+.3f}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white', zorder=4)
        else:
            # For small bars, place label to the right of whichever end
            # is farther from the y-axis (i.e. always on the positive side)
            right_edge = max(start, start + w) + 0.1
            ax.text(right_edge, i, f'{w:+.4f}',
                    ha='left', va='center',
                    fontsize=8, color=SLATE, fontweight='bold', zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels(s_feat, fontsize=11, fontweight='bold')

    # Reference lines
    ax.axvline(x=base_value, color='#94A3B8', linestyle=':', linewidth=0.8,
               alpha=0.5, zorder=1)

    # Base value label
    ax.text(base_value, -1.0, f'E[f(x)] = {base_value:.4f}',
            ha='center', va='top', fontsize=10.5, color=SLATE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=LIGHT_BG,
                      edgecolor=BORDER))

    # Final value label
    ax.text(final_value, n + 0.35,
            f'f(x) = {final_value:.2f}   \u2192   P(DoS) = 99.96%',
            ha='center', va='bottom', fontsize=11.5, color=RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#FEF2F2',
                      edgecolor='#FECACA'))

    ax.set_xlabel('SHAP Value (Log-Odds Contribution)', fontsize=12,
                  fontweight='bold')
    ax.set_title(
        'SHAP Waterfall Plot for a Sample DoS Detection\n'
        'Individual Feature Contributions to Prediction (Record #20459)',
        fontsize=14, fontweight='bold', pad=18, linespacing=1.5
    )

    pos_p = mpatches.Patch(color=SHAP_POS, label='Pushes toward DoS (+)')
    neg_p = mpatches.Patch(color=SHAP_NEG, label='Pushes toward Normal (\u2212)')
    ax.legend(handles=[pos_p, neg_p], loc='lower right', fontsize=10.5,
              framealpha=0.95, edgecolor=BORDER)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.set_ylim(-1.4, n + 0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_5_4_shap_waterfall.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {path}')


# ============================================================
# FIGURE 5.5: END-TO-END FLOW DIAGRAM  (research-paper style)
# ============================================================
def generate_figure_5_5():
    """End-to-End Flow from Detection to Mitigation Output"""
    print("Generating Figure 5.5: End-to-End Flow Diagram...")

    # Large canvas — gives every element room to breathe
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Title ──
    ax.text(12, 15.2,
            'End-to-End Pipeline: From Network Detection to Mitigation',
            ha='center', va='center', fontsize=24, fontweight='bold',
            color='#0F172A')
    ax.text(12, 14.55,
            'Detection \u2192 Explanation \u2192 Classification \u2192 Response',
            ha='center', va='center', fontsize=16, color='#64748B')

    # ── Box dimensions ──
    bw, bh = 5.8, 3.4

    # ── Stage definitions  (two-row Z-pattern) ──
    stages = [
        # Row 1: left → right   (y = 9.2)
        dict(x=0.8,  y=9.2, num='1',
             label='Network Traffic\nInput',
             detail='PCAP  /  CSV  /  Live Capture',
             color='#3B82F6', bg='#DBEAFE'),
        dict(x=9.1,  y=9.2, num='2',
             label='Feature\nExtraction',
             detail='10 Selected Features',
             color='#8B5CF6', bg='#EDE9FE'),
        dict(x=17.4, y=9.2, num='3',
             label='XGBoost Binary\nClassification',
             detail='Threshold: 0.8517  |  F1: 90.26%',
             color='#06B6D4', bg='#CFFAFE'),
        # Row 2: right → left   (y = 3.0)
        dict(x=17.4, y=3.0, num='4',
             label='SHAP Explainability\nAnalysis',
             detail='Per-Feature Attribution',
             color='#F59E0B', bg='#FEF3C7'),
        dict(x=9.1,  y=3.0, num='5',
             label='Attack Classification\n& Severity Assessment',
             detail='Type + Severity Score',
             color='#EF4444', bg='#FEE2E2'),
        dict(x=0.8,  y=3.0, num='6',
             label='Mitigation Command\nGeneration',
             detail='Firewall Rules  /  Rate Limiting',
             color='#10B981', bg='#D1FAE5'),
    ]

    # ── Draw boxes ──
    for s in stages:
        x, y = s['x'], s['y']
        cx, cy = x + bw / 2, y + bh / 2     # centre

        # Drop shadow
        shadow = FancyBboxPatch(
            (x + 0.10, y - 0.10), bw, bh,
            boxstyle="round,pad=0.22",
            facecolor='#94A3B8', edgecolor='none', alpha=0.18, zorder=1)
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (x, y), bw, bh,
            boxstyle="round,pad=0.22",
            facecolor=s['bg'], edgecolor=s['color'], linewidth=3.5, zorder=2)
        ax.add_patch(box)

        # Coloured header banner
        hdr = FancyBboxPatch(
            (x + 0.12, y + bh - 0.72), bw - 0.24, 0.60,
            boxstyle="round,pad=0.10",
            facecolor=s['color'], edgecolor='none', zorder=3)
        ax.add_patch(hdr)

        # Stage number (white on coloured banner)
        ax.text(cx, y + bh - 0.42,
                f"Stage {s['num']}", ha='center', va='center',
                fontsize=16, fontweight='bold', color='white', zorder=4)

        # Main label
        ax.text(cx, cy - 0.10,
                s['label'], ha='center', va='center',
                fontsize=19, fontweight='bold', color='#1E293B',
                zorder=4, linespacing=1.4)

        # Subtitle / detail
        ax.text(cx, y + 0.45,
                s['detail'], ha='center', va='center',
                fontsize=13, color=SLATE, zorder=4, style='italic')

    # ── Helper: thick arrow shorthand ──
    def draw_arrow(x1, y1, x2, y2, color='#475569', lw=3.5, label=None,
                   label_side='above', rad=0.0):
        conn = f'arc3,rad={rad}' if rad else 'arc3,rad=0'
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='-|>',
                color=color, lw=lw, mutation_scale=22,
                connectionstyle=conn,
            ), zorder=5)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.35 if label_side == 'above' else -0.35
            if x1 == x2:                       # vertical arrow
                mx += 0.55
                my = (y1 + y2) / 2
                offset = 0
            ax.text(mx, my + offset, label,
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, linewidth=1.2, alpha=0.92),
                    zorder=6)

    # Row 1 centres
    r1y = 9.2 + bh / 2
    r2y = 3.0 + bh / 2

    # Row 1:  1 → 2  → 3
    draw_arrow(0.8 + bw, r1y,  9.1, r1y,
               label='Extract Features')
    draw_arrow(9.1 + bw, r1y,  17.4, r1y,
               label='Classify Traffic')

    # Vertical:  3 → 4  (attack branch)
    draw_arrow(17.4 + bw / 2, 9.2,  17.4 + bw / 2, 3.0 + bh,
               color='#EF4444', lw=4.0,
               label='Attack Detected')

    # Row 2:  4 → 5  → 6   (right to left)
    draw_arrow(17.4, r2y,  9.1 + bw, r2y,
               label='Classify Attack')
    draw_arrow(9.1, r2y,   0.8 + bw, r2y,
               label='Generate Response')

    # ── Normal-traffic branch (from Stage 3 upward-right) ──
    nx, ny = 22.2, 13.6
    draw_arrow(17.4 + bw, r1y,  nx - 0.8, ny - 0.45,
               color='#10B981', lw=3.0, rad=-0.25)

    # "Normal Traffic — No Action Required" badge
    badge = FancyBboxPatch(
        (nx - 1.55, ny - 0.6), 3.1, 1.2,
        boxstyle="round,pad=0.20",
        facecolor='#D1FAE5', edgecolor='#10B981', linewidth=2.5, zorder=5)
    ax.add_patch(badge)
    ax.text(nx, ny + 0.15, 'Normal Traffic',
            ha='center', va='center', fontsize=15, fontweight='bold',
            color='#10B981', zorder=6)
    ax.text(nx, ny - 0.30, 'No Action Required',
            ha='center', va='center', fontsize=12, color='#10B981',
            style='italic', zorder=6)

    # ── Dashed enclosure around the "response pipeline" (stages 4-6) ──
    enc = FancyBboxPatch(
        (0.3, 1.8), 23.3, 5.8,
        boxstyle="round,pad=0.30",
        facecolor='none', edgecolor='#CBD5E1', linewidth=1.5,
        linestyle='--', zorder=0)
    ax.add_patch(enc)
    ax.text(12, 1.5, 'Response Pipeline  (triggered only for detected attacks)',
            ha='center', va='top', fontsize=13, color='#94A3B8',
            style='italic', zorder=1)

    path = os.path.join(OUTPUT_DIR, 'figure_5_5_end_to_end_flow.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {path}')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print(f'Output directory: {OUTPUT_DIR}\n')
    generate_figure_5_1()
    generate_figure_5_2()
    generate_figure_5_3()
    generate_figure_5_4()
    generate_figure_5_5()
    print(f'\nAll 5 figures generated successfully in:\n  {OUTPUT_DIR}')
