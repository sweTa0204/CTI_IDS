"""Plotly chart helpers — clean white theme for publication."""

import plotly.graph_objects as go
import plotly.express as px

# Shared layout for all charts (white, clean, Inter font)
_BASE_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, -apple-system, sans-serif", color="#1D1D1F", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)

# Colors
MODEL_COLORS = {
    "XGBoost": "#0071E3",
    "RandomForest": "#34C759",
    "1D-CNN": "#FF9500",
    "MLP": "#AF52DE",
    "LSTM": "#FF2D55",
    "SVM": "#5AC8FA",
    "LogisticRegression": "#8E8E93",
}

SEVERITY_COLORS = {
    "CRITICAL": "#FF3B30",
    "HIGH": "#FF9500",
    "MEDIUM": "#FFCC00",
    "LOW": "#34C759",
}

ATTACK_COLORS = {
    "Volumetric Flood": "#FF3B30",
    "Protocol Exploit": "#FF9500",
    "Slowloris": "#FFCC00",
    "Amplification": "#AF52DE",
}


def model_comparison_bar(model_results):
    """Grouped bar chart comparing all models on key metrics."""
    models = list(model_results.keys())
    metrics = ["f1", "accuracy", "precision", "recall"]
    labels = ["F1 Score", "Accuracy", "Precision", "Recall"]

    fig = go.Figure()
    for m in models:
        vals = [model_results[m][met] for met in metrics]
        fig.add_trace(go.Bar(
            name=m, x=labels, y=vals,
            marker_color=MODEL_COLORS.get(m, "#8E8E93"),
            text=[f"{v:.1f}" for v in vals],
            textposition="auto",
            textfont=dict(size=8),
            insidetextanchor="middle",
            hovertemplate="%{x}: %{y:.1f}%<extra>" + m + "</extra>",
        ))

    fig.update_layout(
        title="Benchmark Performance — All Models (Optimized Thresholds)",
        barmode="group",
        yaxis=dict(range=[0, 110], title="Score (%)", gridcolor="#E5E5EA"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        height=500,
        **_BASE_LAYOUT,
    )
    return fig


def confusion_matrix_heatmap(tp, tn, fp, fn, title="Confusion Matrix"):
    """Publication-grade confusion matrix heatmap."""
    z = [[tn, fp], [fn, tp]]
    labels = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Normal", "Predicted DoS"],
        y=["Actual Normal", "Actual DoS"],
        text=labels, texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        colorscale=[[0, "#D6E4F0"], [0.5, "#4A90D9"], [1, "#0071E3"]],
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text=title, y=0.98),
        height=420,
        xaxis=dict(side="top"),
        margin=dict(l=40, r=20, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", color="#1D1D1F", size=12),
    )
    return fig


def attack_donut(results):
    """Donut chart of detection distribution (Normal + attack types)."""
    total = len(results)
    if not total:
        return go.Figure()

    # Count Normal traffic
    normal_count = sum(1 for r in results if r["prediction"] == "Normal")

    # Count each attack type among DoS records
    types = {}
    for r in results:
        if r["prediction"] == "DoS":
            t = r.get("attack_type") or "Unknown"
            types[t] = types.get(t, 0) + 1

    # Build slices: Normal first, then attack types
    labels = ["Normal"] + list(types.keys())
    values = [normal_count] + list(types.values())
    colors = ["#34C759"] + [ATTACK_COLORS.get(t, "#8E8E93") for t in types.keys()]

    # Legend labels with count + percentage
    legend_labels = [f"{l} — {v:,} ({v/total*100:.1f}%)" for l, v in zip(labels, values)]

    fig = go.Figure(go.Pie(
        labels=legend_labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo="percent",
        textposition="inside",
        textfont=dict(size=11, color="white"),
        insidetextorientation="horizontal",
    ))
    fig.update_layout(
        title=dict(text=f"Detection Distribution ({total:,} records)", y=0.98),
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.05, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
        margin=dict(l=20, r=20, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", color="#1D1D1F", size=12),
    )
    return fig


def severity_bar(results):
    """Bar chart of severity level distribution among DoS detections."""
    dos_results = [r for r in results if r["prediction"] == "DoS"]
    total_dos = len(dos_results)
    sevs = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in dos_results:
        s = r.get("severity")
        if s in sevs:
            sevs[s] += 1

    max_val = max(sevs.values()) if any(sevs.values()) else 1
    fig = go.Figure(go.Bar(
        x=list(sevs.keys()), y=list(sevs.values()),
        marker_color=[SEVERITY_COLORS[s] for s in sevs.keys()],
        text=list(sevs.values()), textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=f"Severity — {total_dos:,} DoS Detections",
        height=380,
        yaxis=dict(title="Count", gridcolor="#E5E5EA", range=[0, max_val * 1.15]),
        **_BASE_LAYOUT,
    )
    return fig


def shap_waterfall(shap_vals, feature_names=None):
    """Horizontal bar chart showing SHAP feature contributions."""
    sorted_sv = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [s[0] for s in sorted_sv]
    values = [s[1] for s in sorted_sv]
    colors = ["#FF3B30" if v > 0 else "#0071E3" for v in values]

    fig = go.Figure(go.Bar(
        y=features, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="auto",
        insidetextfont=dict(size=9, color="white"),
        outsidetextfont=dict(size=9, color="#1D1D1F"),
        insidetextanchor="middle",
    ))
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="SHAP Value",
        height=400,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=40, t=60, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", color="#1D1D1F", size=12),
        annotations=[dict(
            text="<span style='color:#FF3B30'>\u25a0</span> Toward DoS &nbsp;&nbsp; "
                 "<span style='color:#0071E3'>\u25a0</span> Toward Normal",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=11),
        )],
    )
    return fig


def results_confusion_matrix(results):
    """Build confusion matrix from detection results (when labels are available)."""
    tp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "DoS")
    tn = sum(1 for r in results if r["prediction"] == "Normal" and r["actual"] == "Normal")
    fp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "Normal")
    fn = sum(1 for r in results if r["prediction"] == "Normal" and r["actual"] == "DoS")
    return confusion_matrix_heatmap(tp, tn, fp, fn, title="Detection Results — Confusion Matrix")
