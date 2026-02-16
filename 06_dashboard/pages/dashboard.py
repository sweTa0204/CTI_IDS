"""Dashboard page — Model performance overview with all 7 models."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_all_model_results, THRESHOLD
from src.charts import model_comparison_bar, confusion_matrix_heatmap

# Header
st.markdown("""
<div class="header-banner">
    <h2>XAI-Powered DoS Detection & Mitigation System</h2>
    <p>Explainable AI for Network Security — Real-time detection, explanation, and automated response</p>
</div>
""", unsafe_allow_html=True)

# ── Model Performance Metrics (XGBoost — selected model) ──
model_results = load_all_model_results()
xgb = model_results.get("XGBoost", {})

st.subheader("Selected Model: XGBoost (Best F1 Score)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{xgb.get('accuracy', 0):.2f}%")
c2.metric("Precision", f"{xgb.get('precision', 0):.2f}%")
c3.metric("Recall", f"{xgb.get('recall', 0):.2f}%")
c4.metric("F1 Score", f"{xgb.get('f1', 0):.2f}%")
c5.metric("Threshold", f"{xgb.get('threshold', 0):.4f}")

st.divider()

# ── Confusion Matrix + Detection Summary (from benchmark test) ──
col_cm, col_info = st.columns([1.2, 1])

with col_cm:
    cm_fig = confusion_matrix_heatmap(
        tp=xgb.get("tp", 3535), tn=xgb.get("tn", 36791),
        fp=xgb.get("fp", 209), fn=xgb.get("fn", 554),
        title="XGBoost — Benchmark Test (41,089 samples)",
    )
    st.plotly_chart(cm_fig, key="dash_cm")

with col_info:
    st.markdown("**Benchmark Test Dataset**")
    st.markdown("""
    - **Source:** UNSW-NB15 Official Testing Set
    - **Samples:** 41,089 (37,000 Normal + 4,089 DoS)
    - **Class ratio:** ~90% Normal / ~10% DoS (real-world imbalance)
    - **Threshold:** Optimized for F1 on validation set (0.8517)
    """)
    st.markdown("**Why XGBoost was selected:**")
    st.markdown("""
    - Highest F1 score among all 7 models
    - Best precision-recall trade-off on imbalanced data
    - Compatible with SHAP TreeExplainer (exact Shapley values)
    - Fast inference (~0.01s for 41k records)
    """)

st.divider()

# ── All 7 Models Comparison ──
st.subheader("Model Comparison — All 7 Trained Models")

comp_fig = model_comparison_bar(model_results)
st.plotly_chart(comp_fig, key="dash_comp")

# Comparison table
comp_df = pd.DataFrame(model_results).T
comp_df.index.name = "Model"
display_cols = ["f1", "accuracy", "precision", "recall", "threshold"]
if "cv_f1" in comp_df.columns:
    display_cols.insert(0, "cv_f1")

comp_df_display = comp_df[display_cols].copy()
col_labels = {
    "cv_f1": "CV F1 (%)", "f1": "Benchmark F1 (%)",
    "accuracy": "Accuracy (%)", "precision": "Precision (%)",
    "recall": "Recall (%)", "threshold": "Opt. Threshold",
}
comp_df_display = comp_df_display.rename(columns=col_labels)

# Pre-format values (Streamlit st.dataframe doesn't always apply Styler formats)
for c in comp_df_display.columns:
    if c == "Opt. Threshold":
        comp_df_display[c] = comp_df_display[c].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    else:
        comp_df_display[c] = comp_df_display[c].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—")

st.dataframe(comp_df_display, height=300)

st.divider()

# ── Pipeline Diagram ──
st.subheader("Detection Pipeline")
st.markdown("""
<div style="text-align:center; padding:1rem 0">
    <span class="pipeline-step">1. Upload CSV</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">2. Feature Extraction</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">3. Encoding & Scaling</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">4. XGBoost Detection</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">5. SHAP Explanation</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">6. Attack Classification</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">7. Mitigation</span>
</div>
""", unsafe_allow_html=True)

st.caption("The system processes network traffic through a 7-step pipeline: "
           "raw CSV data is preprocessed, classified by XGBoost, explained by SHAP, "
           "and matched with appropriate mitigation commands.")
