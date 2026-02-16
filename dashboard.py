"""
XAI-Powered DoS Detection & Mitigation Dashboard
==================================================
Apple/Google-inspired Streamlit dashboard that integrates
the complete detection pipeline: Upload → Preprocess → Detect → Explain → Mitigate

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "03_model_training" / "proper_training" / "data"
MODEL_DIR = BASE_DIR / "03_model_training" / "proper_training" / "models"
XAI_DIR = BASE_DIR / "04_xai_integration"
MIT_DIR = BASE_DIR / "05_mitigation_framework"

# Add project dirs to path for imports
sys.path.insert(0, str(XAI_DIR))
sys.path.insert(0, str(MIT_DIR))

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["rate", "sload", "sbytes", "dload", "proto",
                 "dtcpb", "stcpb", "dmean", "tcprtt", "dur"]
THRESHOLD = 0.8517

SEVERITY_COLORS = {
    "CRITICAL": "#FF3B30", "HIGH": "#FF9500",
    "MEDIUM": "#FFCC00", "LOW": "#34C759", None: "#8E8E93"
}
ATTACK_COLORS = {
    "Volumetric Flood": "#FF3B30", "Protocol Exploit": "#FF9500",
    "Slowloris": "#FFCC00", "Amplification": "#AF52DE",
    "Generic DoS": "#8E8E93"
}

# ---------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI DoS Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ---- Apple / Google inspired theme ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stApp { background-color: #FAFAFA; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1D1D1F 0%, #2C2C2E 100%);
    }
    section[data-testid="stSidebar"] * { color: #F5F5F7 !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #A1A1A6 !important; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0071E3 0%, #40C8E0 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 1.8rem; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.15s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.10); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1D1D1F; letter-spacing: -1px; }
    .metric-label { font-size: 0.8rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
    }

    /* Severity badges */
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
             font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px; }
    .badge-critical { background: #FF3B30; color: white; }
    .badge-high     { background: #FF9500; color: white; }
    .badge-medium   { background: #FFCC00; color: #1D1D1F; }
    .badge-low      { background: #34C759; color: white; }
    .badge-normal   { background: #E5E5EA; color: #3A3A3C; }
    .badge-dos      { background: #FF3B30; color: white; }

    /* Terminal command blocks */
    .cmd-block {
        background: #1D1D1F;
        color: #34C759;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.8rem;
        margin: 0.4rem 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed #D1D1D6;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: white;
    }

    /* Step indicator */
    .step-done { color: #34C759; font-weight: 600; }
    .step-text { color: #3A3A3C; }

    /* Pipeline flow */
    .pipeline-step {
        display: inline-block;
        background: white;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin: 0.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 0.85rem;
        font-weight: 500;
    }
    .pipeline-arrow { display: inline-block; color: #C7C7CC; margin: 0 0.3rem; font-size: 1.2rem; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load XGBoost model from pickle."""
    model_path = MODEL_DIR / "xgboost" / "xgboost_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    """Load fitted StandardScaler."""
    with open(DATA_DIR / "feature_scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoder():
    """Load fitted LabelEncoder for protocol."""
    with open(DATA_DIR / "proto_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_shap_explainer(_model):
    """Create SHAP TreeExplainer."""
    import shap
    return shap.TreeExplainer(_model)


@st.cache_data
def load_sample_data():
    """Load pre-processed test data and labels."""
    X = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    if list(X.columns) == list(range(len(X.columns))):
        X.columns = FEATURE_NAMES
    y = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
    return X, y


def load_model_results():
    """Load comparison results for all models."""
    results = {
        "XGBoost": {"accuracy": 97.76, "precision": 94.41, "recall": 87.09,
                     "f1": 90.57, "auc": 0.9915, "threshold": 0.8517},
    }
    lstm_path = MODEL_DIR / "lstm" / "results" / "lstm_results.json"
    if lstm_path.exists():
        with open(lstm_path) as f:
            d = json.load(f)
            opt = d.get("optimized", d.get("optimized_threshold", {}))
            results["LSTM"] = {
                "accuracy": round(opt.get("accuracy", 0) * 100, 2),
                "precision": round(opt.get("precision", 0) * 100, 2),
                "recall": round(opt.get("recall", 0) * 100, 2),
                "f1": round(opt.get("f1_score", 0) * 100, 2),
                "auc": round(opt.get("auc", 0), 4),
                "threshold": opt.get("threshold", 0.5),
            }
    cnn_path = MODEL_DIR / "cnn1d" / "results" / "cnn1d_results.json"
    if cnn_path.exists():
        with open(cnn_path) as f:
            d = json.load(f)
            opt = d.get("optimized", {})
            results["1D-CNN"] = {
                "accuracy": round(opt.get("accuracy", 0) * 100, 2),
                "precision": round(opt.get("precision", 0) * 100, 2),
                "recall": round(opt.get("recall", 0) * 100, 2),
                "f1": round(opt.get("f1_score", 0) * 100, 2),
                "auc": round(opt.get("auc", 0), 4),
                "threshold": opt.get("threshold", 0.5),
            }
    return results


# ---------------------------------------------------------------------------
# PREPROCESSING HELPERS
# ---------------------------------------------------------------------------
def preprocess_raw_csv(df):
    """Preprocess a raw UNSW-NB15 CSV into model-ready format."""
    steps = []

    # Step 1 — Filter DoS + Normal
    original_count = len(df)
    label_col = None
    for c in ["attack_cat", "label"]:
        if c in df.columns:
            label_col = c
            break
    if label_col == "attack_cat":
        df = df[df["attack_cat"].isin(["DoS", "Normal", "", " "])].copy()
        if "label" in df.columns:
            y_labels = df["label"].values
        else:
            y_labels = (df["attack_cat"] == "DoS").astype(int).values
    elif label_col == "label":
        y_labels = df["label"].values
    else:
        y_labels = None
    steps.append(f"Filtered DoS + Normal: {original_count:,} -> {len(df):,} records")

    # Step 2 — Select features
    available = [f for f in FEATURE_NAMES if f in df.columns]
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    X = df[available].copy()
    for m in missing:
        X[m] = 0
    X = X[FEATURE_NAMES]
    steps.append(f"Selected {len(available)}/10 features" +
                 (f" (missing: {missing})" if missing else ""))

    # Step 3 — Encode proto
    encoder = load_encoder()
    if X["proto"].dtype == object:
        known = set(encoder.classes_)
        X["proto"] = X["proto"].apply(lambda v: v if v in known else "tcp")
        X["proto"] = encoder.transform(X["proto"])
        steps.append("Protocol encoded (string -> numeric)")
    else:
        steps.append("Protocol already numeric")

    # Step 4 — Missing values
    n_missing = X.isnull().sum().sum()
    X = X.fillna(X.median())
    steps.append(f"Missing values filled ({n_missing} found)")

    # Step 5 — Scale
    scaler = load_scaler()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_NAMES, index=X.index)
    steps.append("Features scaled (StandardScaler)")

    return X_scaled, y_labels, steps


def detect_csv_type(df):
    """Determine CSV type: raw UNSW-NB15, CIC (CICFlowMeter), or preprocessed."""
    cols_stripped = set(c.strip() for c in df.columns)
    # CIC format (CIC-IDS2017, CIC-DDoS2019): CICFlowMeter output with ~80 cols
    if "Flow Duration" in cols_stripped and "Total Fwd Packets" in cols_stripped:
        return "cic"
    cols = set(df.columns)
    feature_set = set(FEATURE_NAMES)
    if len(df.columns) > 15:
        return "raw"
    if feature_set.issubset(cols) or len(df.columns) == 10:
        return "preprocessed"
    return "unknown"


def preprocess_cic_csv(df):
    """Preprocess a CIC-format CSV (CIC-IDS2017, CIC-DDoS2019) through adapter.

    Maps CICFlowMeter features to the 10 model features, sets unavailable
    features (stcpb, dtcpb, tcprtt) to neutral values, and scales.
    """
    steps = []
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Step 1 — Extract labels
    y_labels = None
    if "Label" in df.columns:
        labels_raw = df["Label"].str.strip()
        y_labels = (labels_raw != "BENIGN").astype(int).values
        n_attack = int(y_labels.sum())
        n_benign = int((y_labels == 0).sum())
        attack_types = labels_raw[labels_raw != "BENIGN"].unique().tolist()
        type_str = ", ".join(attack_types[:5])
        if len(attack_types) > 5:
            type_str += f", ... (+{len(attack_types) - 5} more)"
        steps.append(f"CIC dataset detected — {len(df):,} records "
                     f"({n_attack:,} attack, {n_benign:,} benign)")
        steps.append(f"Attack types: {type_str}")
    else:
        steps.append(f"CIC dataset detected — {len(df):,} records (no labels)")

    # Step 2 — Map CIC features to model features
    dur_us = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0).values
    dur = np.maximum(dur_us / 1e6, 0.0)

    rate = pd.to_numeric(df["Flow Packets/s"], errors="coerce").fillna(0).values
    sbytes = pd.to_numeric(
        df["Total Length of Fwd Packets"], errors="coerce"
    ).fillna(0).values.astype(float)
    bwd_bytes = pd.to_numeric(
        df["Total Length of Bwd Packets"], errors="coerce"
    ).fillna(0).values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sload = np.where(dur > 0, sbytes * 8.0 / dur, 0.0)
        dload = np.where(dur > 0, bwd_bytes * 8.0 / dur, 0.0)

    dmean = pd.to_numeric(
        df["Bwd Packet Length Mean"], errors="coerce"
    ).fillna(0).values.astype(float)

    steps.append("Feature mapping: 7/10 features extracted from CIC columns")

    # Step 3 — Protocol encoding
    encoder = load_encoder()
    PROTO_NUM_TO_NAME = {
        6: "tcp", 17: "udp", 1: "icmp", 2: "igmp", 47: "gre",
        41: "ipv6", 89: "ospf", 132: "sctp", 50: "esp", 51: "ah",
    }
    if "Protocol" in df.columns:
        proto_nums = pd.to_numeric(
            df["Protocol"], errors="coerce"
        ).fillna(6).astype(int).values
        known = set(encoder.classes_)
        proto_names = [PROTO_NUM_TO_NAME.get(p, str(p)) for p in proto_nums]
        proto_names = [n if n in known else "tcp" for n in proto_names]
        proto = encoder.transform(proto_names).astype(float)
        steps.append("Protocol column encoded via LabelEncoder")
    else:
        tcp_val = float(encoder.transform(["tcp"])[0])
        proto = np.full(len(df), tcp_val)
        steps.append("No Protocol column — defaulting to TCP")

    # Step 4 — Unavailable features set to neutral (training mean → scales to 0)
    scaler = load_scaler()
    dtcpb  = np.full(len(df), scaler.mean_[5])
    stcpb  = np.full(len(df), scaler.mean_[6])
    tcprtt = np.full(len(df), scaler.mean_[8])
    steps.append("Unavailable features (stcpb, dtcpb, tcprtt) set to neutral")

    # Step 5 — Assemble + scale
    X_raw = np.column_stack([
        rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur,
    ])
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = pd.DataFrame(
        scaler.transform(X_raw), columns=FEATURE_NAMES, index=df.index,
    )
    steps.append("Features scaled (StandardScaler)")

    return X_scaled, y_labels, steps


# ---------------------------------------------------------------------------
# ANALYSIS ENGINE
# ---------------------------------------------------------------------------
def classify_attack(top_features, shap_vals):
    """Classify attack type based on SHAP top features."""
    volumetric = {"rate", "sload", "sbytes"}
    protocol = {"proto", "tcprtt", "stcpb", "dtcpb"}
    slowloris = {"dur", "dmean"}
    amplification = {"dload"}

    top_set = set(top_features[:3])
    scores = {
        "Volumetric Flood": len(top_set & volumetric) / max(len(volumetric), 1),
        "Protocol Exploit": len(top_set & protocol) / max(len(protocol), 1),
        "Slowloris": len(top_set & slowloris) / max(len(slowloris), 1),
        "Amplification": len(top_set & amplification) / max(len(amplification), 1),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "Volumetric Flood"
    return best, scores[best]


def calc_severity(confidence):
    """Calculate severity level from confidence."""
    if confidence >= 0.95:
        return "CRITICAL"
    elif confidence >= 0.90:
        return "HIGH"
    elif confidence >= 0.75:
        return "MEDIUM"
    elif confidence >= 0.60:
        return "LOW"
    return None


def get_mitigation_commands(attack_type, severity):
    """Return mitigation commands for an attack type."""
    commands = {
        "Volumetric Flood": [
            "iptables -A INPUT -p tcp --syn -m limit --limit 10/s --limit-burst 20 -j ACCEPT",
            "iptables -A INPUT -p tcp --syn -j DROP",
            "tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 400ms",
        ],
        "Protocol Exploit": [
            "echo 1 > /proc/sys/net/ipv4/tcp_syncookies",
            "iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT",
            "iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP",
            "iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP",
        ],
        "Slowloris": [
            "sysctl -w net.ipv4.tcp_fin_timeout=30",
            "sysctl -w net.ipv4.tcp_keepalive_time=300",
            "iptables -A INPUT -p tcp -m connlimit --connlimit-above 10 -j DROP",
        ],
        "Amplification": [
            "iptables -A INPUT -p udp --sport 53 -m length --length 512: -j DROP",
            "iptables -A INPUT -p udp -m limit --limit 100/s -j ACCEPT",
            "iptables -A INPUT -p udp -j DROP",
        ],
    }
    return commands.get(attack_type, commands["Volumetric Flood"])


def run_batch_detection(X_df, y_labels):
    """Run detection on all records."""
    model = load_model()
    explainer = load_shap_explainer(model)

    X_arr = X_df.values if hasattr(X_df, 'values') else np.array(X_df)
    probas = model.predict_proba(X_arr)
    shap_values = explainer.shap_values(X_arr)

    results = []
    for i in range(len(X_arr)):
        p_dos = float(probas[i][1])
        is_dos = p_dos >= THRESHOLD
        prediction = "DoS" if is_dos else "Normal"
        confidence = p_dos if is_dos else (1 - p_dos)

        sv = {FEATURE_NAMES[j]: float(shap_values[i][j]) for j in range(len(FEATURE_NAMES))}

        # Top features: only those pushing TOWARD the prediction
        if is_dos:
            dos_features = sorted(
                [(k, v) for k, v in sv.items() if v > 0],
                key=lambda x: x[1], reverse=True,
            )
            top3 = [f[0] for f in dos_features[:3]]
        else:
            normal_features = sorted(
                [(k, v) for k, v in sv.items() if v < 0],
                key=lambda x: abs(x[1]), reverse=True,
            )
            top3 = [f[0] for f in normal_features[:3]]

        attack_type = None
        severity = None
        if is_dos:
            attack_type, _ = classify_attack(top3, sv)
            severity = calc_severity(p_dos)

        actual = None
        if y_labels is not None and i < len(y_labels):
            actual = "DoS" if y_labels[i] == 1 else "Normal"

        results.append({
            "index": i,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "p_dos": round(p_dos * 100, 2),
            "attack_type": attack_type or "—",
            "severity": severity or "—",
            "top_features": top3,
            "shap_values": sv,
            "actual": actual,
        })
    return results


# ---------------------------------------------------------------------------
# PLOTLY CHART HELPERS
# ---------------------------------------------------------------------------
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#1D1D1F"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def make_confusion_matrix(results):
    tp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "DoS")
    tn = sum(1 for r in results if r["prediction"] == "Normal" and r["actual"] == "Normal")
    fp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "Normal")
    fn = sum(1 for r in results if r["prediction"] == "Normal" and r["actual"] == "DoS")

    z = [[tn, fp], [fn, tp]]
    labels = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]

    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred Normal", "Pred DoS"], y=["Actual Normal", "Actual DoS"],
        text=labels, texttemplate="%{text}", textfont=dict(size=14, color="white"),
        colorscale=[[0, "#E5E5EA"], [0.5, "#64D2FF"], [1, "#0071E3"]],
        showscale=False,
    ))
    fig.update_layout(title="Confusion Matrix", height=350, **CHART_LAYOUT)
    fig.update_xaxes(side="top")
    return fig, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def make_attack_donut(results):
    dos_results = [r for r in results if r["prediction"] == "DoS"]
    if not dos_results:
        return go.Figure()
    types = {}
    for r in dos_results:
        t = r["attack_type"]
        types[t] = types.get(t, 0) + 1

    fig = go.Figure(go.Pie(
        labels=list(types.keys()), values=list(types.values()),
        hole=0.55,
        marker=dict(colors=[ATTACK_COLORS.get(t, "#8E8E93") for t in types.keys()]),
        textinfo="label+percent", textfont=dict(size=12),
    ))
    fig.update_layout(title="Attack Type Distribution", height=350,
                      showlegend=False, **CHART_LAYOUT)
    return fig


def make_severity_bar(results):
    dos_results = [r for r in results if r["prediction"] == "DoS"]
    sevs = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in dos_results:
        s = r["severity"]
        if s in sevs:
            sevs[s] += 1

    fig = go.Figure(go.Bar(
        x=list(sevs.keys()), y=list(sevs.values()),
        marker_color=[SEVERITY_COLORS[s] for s in sevs.keys()],
        text=list(sevs.values()), textposition="outside",
        textfont=dict(size=13, color="#1D1D1F"),
    ))
    fig.update_layout(title="Severity Distribution", height=350,
                      yaxis_title="Count", **CHART_LAYOUT)
    return fig


def make_shap_bar(shap_vals):
    sorted_sv = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [s[0] for s in sorted_sv]
    values = [s[1] for s in sorted_sv]
    colors = ["#FF3B30" if v > 0 else "#0071E3" for v in values]

    fig = go.Figure(go.Bar(
        y=features, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="SHAP Value (impact on DoS prediction)",
        height=380, **CHART_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def make_model_comparison(model_results):
    models = list(model_results.keys())
    metrics = ["f1", "accuracy", "precision", "recall"]
    metric_labels = ["F1 Score", "Accuracy", "Precision", "Recall"]
    bar_colors = ["#0071E3", "#34C759", "#FF9500"]

    fig = go.Figure()
    for i, m in enumerate(models):
        vals = [model_results[m][met] for met in metrics]
        fig.add_trace(go.Bar(
            name=m, x=metric_labels, y=vals,
            marker_color=bar_colors[i % len(bar_colors)],
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont=dict(size=11),
        ))
    fig.update_layout(
        title="Model Comparison (Optimized Thresholds)",
        barmode="group", height=400,
        yaxis=dict(range=[0, 105], title="Score (%)"),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        **CHART_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# UI: SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["Dashboard", "Analyze", "About"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("##### Model Info")
    st.markdown(f"**Model:** XGBoost")
    st.markdown(f"**Threshold:** {THRESHOLD}")
    st.markdown(f"**F1 Score:** 90.57%")
    st.markdown(f"**AUC:** 0.9915")
    st.markdown("---")
    st.markdown(
        "<small style='color:#86868B'>XAI-Powered DoS Detection<br>"
        "UNSW-NB15 Dataset<br>SHAP TreeExplainer</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>XAI-Powered DoS Detection & Mitigation</h1>
    <p>Explainable AI for Network Security — Real-time detection, explanation, and automated response</p>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# PAGE: DASHBOARD
# ===================================================================
if page == "Dashboard":

    model_results = load_model_results()
    xgb = model_results["XGBoost"]

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{xgb['accuracy']}%", "Accuracy"),
        (c2, f"{xgb['precision']}%", "Precision"),
        (c3, f"{xgb['recall']}%", "Recall"),
        (c4, f"{xgb['f1']}%", "F1 Score"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Load sample data for charts (cache in session_state)
    try:
        if "dashboard_results" not in st.session_state:
            X_sample, y_sample = load_sample_data()
            with st.spinner("Running detection on test dataset..."):
                st.session_state["dashboard_results"] = run_batch_detection(X_sample, y_sample)

        results = st.session_state["dashboard_results"]

        col_a, col_b = st.columns(2)
        with col_a:
            cm_fig, _ = make_confusion_matrix(results)
            st.plotly_chart(cm_fig, width="stretch")
        with col_b:
            donut_fig = make_attack_donut(results)
            st.plotly_chart(donut_fig, width="stretch")
    except Exception as e:
        st.warning(f"Could not load sample data for charts: {e}")

    # Model comparison
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    comp_fig = make_model_comparison(model_results)
    st.plotly_chart(comp_fig, width="stretch")

    comp_df = pd.DataFrame(model_results).T
    comp_df.index.name = "Model"
    st.dataframe(comp_df.style.format({
        "accuracy": "{:.2f}%", "precision": "{:.2f}%",
        "recall": "{:.2f}%", "f1": "{:.2f}%", "auc": "{:.4f}",
        "threshold": "{:.4f}",
    }), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detection Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:1rem 0">
        <span class="pipeline-step">Upload CSV</span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step">Preprocess</span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step">XGBoost Detection</span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step">SHAP Explanation</span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step">Attack Classification</span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step">Severity Assessment</span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step">Mitigation Commands</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===================================================================
# PAGE: ANALYZE
# ===================================================================
elif page == "Analyze":

    # Upload section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Network Traffic Data</div>',
                unsafe_allow_html=True)

    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag & drop your CSV here",
            type=["csv"],
            help="Supports raw UNSW-NB15, CIC-IDS2017/DDoS2019, or preprocessed (10 features)",
        )

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("Load Sample Data", type="primary",
                                help="Load the test dataset (41,089 samples)")

    st.markdown('</div>', unsafe_allow_html=True)

    # Determine data source
    X_data = None
    y_labels = None
    preprocessing_steps = []

    if use_sample:
        X_data, y_labels = load_sample_data()
        preprocessing_steps = ["Sample test data loaded (already preprocessed)",
                               f"{len(X_data):,} records with 10 features"]
        st.session_state["data_loaded"] = True
        st.session_state["X_data"] = X_data
        st.session_state["y_labels"] = y_labels
        st.session_state["steps"] = preprocessing_steps

    elif uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        csv_type = detect_csv_type(df_raw)

        if csv_type == "raw":
            X_data, y_labels, preprocessing_steps = preprocess_raw_csv(df_raw)
        elif csv_type == "cic":
            X_data, y_labels, preprocessing_steps = preprocess_cic_csv(df_raw)
        elif csv_type == "preprocessed":
            if len(df_raw.columns) == 10 and list(df_raw.columns) != FEATURE_NAMES:
                df_raw.columns = FEATURE_NAMES
            X_data = df_raw[FEATURE_NAMES] if all(f in df_raw.columns for f in FEATURE_NAMES) else df_raw
            y_labels = None
            preprocessing_steps = ["Preprocessed data detected (10 features)",
                                   f"{len(X_data):,} records loaded",
                                   "No additional preprocessing needed"]
        else:
            st.error(f"Unrecognized CSV format ({len(df_raw.columns)} columns). "
                     "Expected raw UNSW-NB15, CIC-IDS2017/DDoS2019, "
                     "or preprocessed (10 cols).")

        if X_data is not None:
            st.session_state["data_loaded"] = True
            st.session_state["X_data"] = X_data
            st.session_state["y_labels"] = y_labels
            st.session_state["steps"] = preprocessing_steps

    # Show data from session state
    if st.session_state.get("data_loaded"):
        X_data = st.session_state["X_data"]
        y_labels = st.session_state["y_labels"]
        preprocessing_steps = st.session_state.get("steps", [])

        # Preprocessing steps
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preprocessing Status</div>',
                    unsafe_allow_html=True)
        for step in preprocessing_steps:
            st.markdown(f'<span class="step-done">&#10003;</span> '
                        f'<span class="step-text">{step}</span>', unsafe_allow_html=True)

        st.markdown("<br>**Data Preview:**", unsafe_allow_html=True)
        st.dataframe(X_data.head(), width="stretch", height=200)
        st.markdown('</div>', unsafe_allow_html=True)

        # Run detection
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        max_records = st.slider("Records to analyze", 100, min(len(X_data), 41089),
                                min(len(X_data), 5000), step=100)

        run_btn = st.button("Run Detection", type="primary")

        if run_btn:
            X_subset = X_data.iloc[:max_records]
            y_subset = y_labels[:max_records] if y_labels is not None else None

            with st.spinner(f"Analyzing {max_records:,} records..."):
                results = run_batch_detection(X_subset, y_subset)

            st.session_state["results"] = results
            st.session_state["analyzed"] = True

        st.markdown('</div>', unsafe_allow_html=True)

    # Show results
    if st.session_state.get("analyzed"):
        results = st.session_state["results"]

        total = len(results)
        dos_count = sum(1 for r in results if r["prediction"] == "DoS")
        normal_count = total - dos_count
        has_actual = results[0]["actual"] is not None if results else False

        if has_actual:
            fp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "Normal")
        else:
            fp = 0

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{total:,}", "Total Analyzed"),
            (c2, f"{dos_count:,}", "DoS Detected"),
            (c3, f"{normal_count:,}", "Normal Traffic"),
            (c4, f"{fp:,}", "False Alarms" if has_actual else "N/A"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col_a, col_b = st.columns(2)
        with col_a:
            donut = make_attack_donut(results)
            st.plotly_chart(donut, width="stretch")
        with col_b:
            sev_bar = make_severity_bar(results)
            st.plotly_chart(sev_bar, width="stretch")

        # Results table
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detection Results</div>',
                    unsafe_allow_html=True)
        st.markdown("*Click a row number below to see detailed SHAP explanation*")

        table_data = []
        for r in results:
            row = {
                "#": r["index"],
                "Prediction": r["prediction"],
                "Confidence": f"{r['confidence']:.1f}%",
                "Attack Type": r["attack_type"],
                "Severity": r["severity"],
            }
            if has_actual:
                row["Actual"] = r["actual"]
                row["Correct"] = "Yes" if r["prediction"] == r["actual"] else "No"
            table_data.append(row)

        results_df = pd.DataFrame(table_data)
        st.dataframe(results_df, width="stretch", height=350)
        st.markdown('</div>', unsafe_allow_html=True)

        # Detail view — select a record
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detailed Record Analysis</div>',
                    unsafe_allow_html=True)

        dos_indices = [r["index"] for r in results if r["prediction"] == "DoS"]
        default_idx = dos_indices[0] if dos_indices else 0

        selected = st.selectbox(
            "Select record to analyze",
            options=[r["index"] for r in results],
            index=([r["index"] for r in results].index(default_idx)
                   if default_idx in [r["index"] for r in results] else 0),
            format_func=lambda x: (
                f"Record #{x} — "
                f"{next(r['prediction'] for r in results if r['index']==x)} "
                f"({next(r['confidence'] for r in results if r['index']==x):.1f}%)"
            ),
        )

        rec = next(r for r in results if r["index"] == selected)

        # Detection + Classification cards
        det_col, cls_col = st.columns(2)

        with det_col:
            badge_cls = "badge-dos" if rec["prediction"] == "DoS" else "badge-normal"
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1.2rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                <div style="font-weight:600;margin-bottom:0.5rem">Detection</div>
                <span class="badge {badge_cls}">{rec['prediction']}</span>
                <div style="margin-top:0.8rem;color:#86868B;font-size:0.85rem">
                    Confidence: <strong style="color:#1D1D1F">{rec['confidence']:.1f}%</strong>
                </div>
                <div style="color:#86868B;font-size:0.85rem">
                    P(DoS): <strong style="color:#1D1D1F">{rec['p_dos']:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cls_col:
            sev = rec["severity"]
            sev_badge = f"badge-{sev.lower()}" if sev != "—" else "badge-normal"
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1.2rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                <div style="font-weight:600;margin-bottom:0.5rem">Classification</div>
                <div style="font-size:0.9rem">Attack: <strong>{rec['attack_type']}</strong></div>
                <div style="margin-top:0.5rem">
                    Severity: <span class="badge {sev_badge}">{sev}</span>
                </div>
                <div style="margin-top:0.5rem;color:#86868B;font-size:0.85rem">
                    Top features: {', '.join(rec['top_features'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # SHAP bar chart
        shap_fig = make_shap_bar(rec["shap_values"])
        st.plotly_chart(shap_fig, width="stretch")

        # Feature table
        feat_data = []
        for f in FEATURE_NAMES:
            sv = rec["shap_values"].get(f, 0)
            feat_data.append({
                "Feature": f,
                "SHAP Value": f"{sv:+.4f}",
                "Direction": "Toward DoS" if sv > 0 else "Toward Normal",
                "|Impact|": abs(sv),
            })
        feat_df = pd.DataFrame(feat_data).sort_values("|Impact|", ascending=False)
        st.dataframe(feat_df.drop(columns=["|Impact|"]), width="stretch",
                     hide_index=True, height=280)

        # Mitigation commands
        if rec["prediction"] == "DoS":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Recommended Mitigation — {rec['attack_type']}**")
            commands = get_mitigation_commands(rec["attack_type"], rec["severity"])
            for cmd in commands:
                st.markdown(f'<div class="cmd-block">$ {cmd}</div>',
                            unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ===================================================================
# PAGE: ABOUT
# ===================================================================
elif page == "About":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About This Project</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **XAI-Powered DoS Detection and Mitigation System** is a research project that combines
    machine learning with Explainable AI (XAI) to detect Denial-of-Service attacks in network
    traffic and generate automated, transparent mitigation responses.

    **Key Contributions:**
    - Trained 7 ML models; XGBoost selected with 90.57% F1 Score
    - SHAP TreeExplainer provides mathematically exact explanations
    - Novel mitigation framework that uses XAI output to classify attacks and generate commands
    - Complete transparent pipeline from detection to response

    **Dataset:** UNSW-NB15 (University of New South Wales)
    - Training: 24,528 samples (balanced 50/50 DoS/Normal)
    - Testing: 41,089 samples (imbalanced 90/10 Normal/DoS — real-world scenario)

    **Technology Stack:** Python, XGBoost, SHAP, Scikit-learn, TensorFlow, Streamlit, Plotly
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Architecture</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0">
        <span class="pipeline-step" style="background:#E3F2FD">1. Data Input<br><small>10 features</small></span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step" style="background:#E8F5E9">2. XGBoost<br><small>Detection</small></span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step" style="background:#FFF3E0">3. SHAP<br><small>Explanation</small></span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step" style="background:#FCE4EC">4. Classify<br><small>Attack Type</small></span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step" style="background:#F3E5F5">5. Severity<br><small>Assessment</small></span>
        <span class="pipeline-arrow">&rarr;</span>
        <span class="pipeline-step" style="background:#E0F7FA">6. Mitigate<br><small>Commands</small></span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model comparison
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Comparison</div>',
                unsafe_allow_html=True)
    model_results = load_model_results()
    comp_fig = make_model_comparison(model_results)
    st.plotly_chart(comp_fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
