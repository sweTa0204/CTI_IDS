"""Analyze page — Upload CSV → Pipeline → Detection → SHAP → Mitigation."""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import FEATURE_NAMES, load_sample_data
from src.pipeline import (
    preprocess_raw_csv, preprocess_cic_csv, detect_csv_type,
    run_detection, MITIGATION_COMMANDS,
)
from src.charts import (
    attack_donut, severity_bar, shap_waterfall,
    results_confusion_matrix,
)

# Header
st.markdown("""
<div class="header-banner">
    <h2>Analyze Network Traffic</h2>
    <p>Upload a CSV file or use the sample test dataset to run the full detection pipeline</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1: DATA INPUT
# ═══════════════════════════════════════════════════════════════
st.subheader("Step 1 — Data Input")

col_upload, col_or, col_sample = st.columns([5, 1, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        help="Supports raw UNSW-NB15, CIC-IDS2017/DDoS2019, or preprocessed (10 features). "
             "If labels are present, accuracy metrics will be shown.",
    )

with col_or:
    st.markdown(
        '<div style="margin-top:2.6rem;text-align:center;color:#86868B;font-weight:600;'
        'font-size:0.85rem;letter-spacing:0.5px">OR</div>',
        unsafe_allow_html=True,
    )

with col_sample:
    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    use_sample = st.button("Load Sample Test Data", type="primary",
                           help="UNSW-NB15 test set: 41,089 samples with ground truth labels")

# ═══════════════════════════════════════════════════════════════
# STEP 2: PREPROCESSING
# ═══════════════════════════════════════════════════════════════
if use_sample:
    X_data, y_labels = load_sample_data()
    st.session_state["X_data"] = X_data
    st.session_state["y_labels"] = y_labels
    st.session_state["data_ready"] = True
    st.session_state["analyzed"] = False
    st.success(f"Sample data loaded: {len(X_data):,} records, 10 features, with ground truth labels.")

elif uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    csv_type = detect_csv_type(df_raw)

    if csv_type == "raw":
        st.subheader("Step 2 — Preprocessing Pipeline")
        with st.status("Processing raw CSV through pipeline...", expanded=True) as status:
            X_data, y_labels = preprocess_raw_csv(df_raw, status_container=st)
            status.update(
                label=f"Preprocessing complete — {len(X_data):,} records ready",
                state="complete", expanded=False,
            )
        st.session_state["X_data"] = X_data
        st.session_state["y_labels"] = y_labels
        st.session_state["data_ready"] = True
        st.session_state["analyzed"] = False

    elif csv_type == "cic":
        st.subheader("Step 2 — CIC Adapter Pipeline")
        with st.status("Processing CIC-format CSV through adapter...", expanded=True) as status:
            X_data, y_labels = preprocess_cic_csv(df_raw, status_container=st)
            status.update(
                label=f"CIC preprocessing complete — {len(X_data):,} records ready",
                state="complete", expanded=False,
            )
        st.session_state["X_data"] = X_data
        st.session_state["y_labels"] = y_labels
        st.session_state["data_ready"] = True
        st.session_state["analyzed"] = False

    elif csv_type == "preprocessed":
        if len(df_raw.columns) == 10 and list(df_raw.columns) != FEATURE_NAMES:
            df_raw.columns = FEATURE_NAMES
        X_data = df_raw[FEATURE_NAMES] if all(f in df_raw.columns for f in FEATURE_NAMES) else df_raw
        st.session_state["X_data"] = X_data
        st.session_state["y_labels"] = None
        st.session_state["data_ready"] = True
        st.session_state["analyzed"] = False
        st.success(f"Preprocessed data loaded: {len(X_data):,} records, 10 features.")

    else:
        st.error(f"Unrecognized CSV format ({len(df_raw.columns)} columns). "
                 "Expected raw UNSW-NB15, CIC-IDS2017/DDoS2019, "
                 "or preprocessed (10 features).")

# ═══════════════════════════════════════════════════════════════
# STEP 3: DATA PREVIEW + RUN DETECTION
# ═══════════════════════════════════════════════════════════════
if st.session_state.get("data_ready"):
    X_data = st.session_state["X_data"]
    y_labels = st.session_state.get("y_labels")

    st.divider()
    st.subheader("Data Preview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(X_data):,}")
    c2.metric("Features", f"{len(X_data.columns)}")
    c3.metric("Labels Available", "Yes" if y_labels is not None else "No (prediction only)")

    with st.expander("View first 5 rows", expanded=False):
        st.dataframe(X_data.head(), height=200)

    st.divider()
    st.subheader("Step 3 — Run Detection")

    max_records = st.slider(
        "Number of records to analyze",
        min_value=100,
        max_value=min(len(X_data), 41089),
        value=min(len(X_data), 5000),
        step=100,
    )

    # Clear stale results when slider changes so charts always match the current run
    if max_records != st.session_state.get("_last_max_records"):
        st.session_state["analyzed"] = False
        st.session_state["_last_max_records"] = max_records

    run_btn = st.button("Run Detection Pipeline", type="primary")

    if run_btn:
        X_subset = X_data.iloc[:max_records]
        y_subset = y_labels[:max_records] if y_labels is not None else None

        # Check cache — skip recomputation for identical input
        cache_key = f"{max_records}_{id(X_data)}"
        if st.session_state.get("_results_cache_key") == cache_key:
            results = st.session_state["results"]
            st.toast(f"Loaded cached results for {max_records:,} records.", icon="⚡")
        else:
            with st.status(f"Running detection on {max_records:,} records...", expanded=True) as status:
                results = run_detection(X_subset, y_subset, status_container=st)
                status.update(
                    label=f"Detection complete — {max_records:,} records analyzed",
                    state="complete", expanded=False,
                )
            st.session_state["_results_cache_key"] = cache_key

        st.session_state["results"] = results
        st.session_state["analyzed"] = True

# ═══════════════════════════════════════════════════════════════
# STEP 4: RESULTS
# ═══════════════════════════════════════════════════════════════
if st.session_state.get("analyzed"):
    results = st.session_state["results"]
    has_labels = results[0]["actual"] is not None if results else False

    total = len(results)
    dos_count = sum(1 for r in results if r["prediction"] == "DoS")
    normal_count = total - dos_count

    st.divider()
    st.subheader("Detection Results")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Analyzed", f"{total:,}")
    c2.metric("DoS Detected", f"{dos_count:,}")
    c3.metric("Normal Traffic", f"{normal_count:,}")

    if has_labels:
        tp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "DoS")
        fp = sum(1 for r in results if r["prediction"] == "DoS" and r["actual"] == "Normal")
        fn = sum(1 for r in results if r["prediction"] == "Normal" and r["actual"] == "DoS")
        tn = sum(1 for r in results if r["prediction"] == "Normal" and r["actual"] == "Normal")
        accuracy = (tp + tn) / total * 100 if total > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        c4, c5, c6, c7 = st.columns(4)
        c4.metric("Accuracy", f"{accuracy:.2f}%")
        c5.metric("Precision", f"{precision:.2f}%")
        c6.metric("Recall", f"{recall:.2f}%")
        c7.metric("F1 Score", f"{f1:.2f}%")

        # Explain zero metrics when the analyzed subset lacks one class
        actual_dos = tp + fn
        actual_normal = tn + fp
        if actual_dos == 0 and dos_count > 0:
            st.info(
                f"**Why are Precision, Recall, and F1 at 0%?** "
                f"The first {total:,} records in this dataset contain **no actual attack traffic** "
                f"(all {total:,} ground-truth labels are Normal). The {dos_count:,} DoS detections "
                f"are false positives against this benign-only subset. "
                f"Increase the record count or shuffle the data to include actual attack samples "
                f"for meaningful metrics."
            )
        elif actual_normal == 0 and normal_count > 0:
            st.info(
                f"**Why is Precision at 100% but accuracy seems off?** "
                f"The first {total:,} records contain **only attack traffic** "
                f"(no benign samples). Increase the record count to include a mix of both classes."
            )

    # Charts in tabs
    chart_tabs = ["Overview"]
    if has_labels:
        chart_tabs.append("Confusion Matrix")

    tabs = st.tabs(chart_tabs)

    with tabs[0]:
        st.caption(f"Charts below reflect records 0–{total-1:,} "
                   f"({total:,} analyzed, {dos_count:,} DoS detected)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(attack_donut(results), key=f"donut_{total}_{dos_count}")
        with col_b:
            st.plotly_chart(severity_bar(results), key=f"sev_{total}_{dos_count}")

    if has_labels:
        with tabs[1]:
            st.plotly_chart(results_confusion_matrix(results), key=f"cm_{total}_{dos_count}")

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # STEP 5: INDIVIDUAL RECORD ANALYSIS
    # ═══════════════════════════════════════════════════════════
    st.subheader("Step 5 — Individual Record Analysis (SHAP Explanation)")

    # Build options list
    dos_indices = [r["index"] for r in results if r["prediction"] == "DoS"]
    all_indices = [r["index"] for r in results]
    default_idx = dos_indices[0] if dos_indices else 0

    selected = st.selectbox(
        "Select a record to inspect",
        options=all_indices,
        index=all_indices.index(default_idx) if default_idx in all_indices else 0,
        format_func=lambda x: (
            f"Record #{x} — "
            f"{next(r['prediction'] for r in results if r['index']==x)} "
            f"({next(r['confidence'] for r in results if r['index']==x):.1f}% confidence)"
        ),
    )

    rec = next(r for r in results if r["index"] == selected)

    # Detection + Classification summary
    det_col, cls_col = st.columns(2)

    with det_col:
        pred = rec["prediction"]
        badge = "badge-dos" if pred == "DoS" else "badge-normal"
        st.markdown(f"""
        **Detection Result**

        Prediction: <span class="badge {badge}">{pred}</span>
        &nbsp;&nbsp; Confidence: **{rec['confidence']:.1f}%** &nbsp;&nbsp; P(DoS): **{rec['p_dos']:.1f}%**
        """, unsafe_allow_html=True)

        if rec["actual"]:
            correct = "Yes" if rec["prediction"] == rec["actual"] else "No"
            st.markdown(f"Actual: **{rec['actual']}** &nbsp;&nbsp; Correct: **{correct}**")

    with cls_col:
        if rec["prediction"] == "DoS":
            sev = rec["severity"] or "—"
            sev_badge = f"badge-{sev.lower()}" if sev != "—" else ""
            st.markdown(f"""
            **Attack Classification**

            Type: **{rec['attack_type']}**
            &nbsp;&nbsp; Severity: <span class="badge {sev_badge}">{sev}</span>

            Top contributing features: `{', '.join(rec['top_features'])}`
            """, unsafe_allow_html=True)
        else:
            st.markdown("**Classification:** Normal traffic — no attack indicators detected.")

    # SHAP chart
    st.plotly_chart(shap_waterfall(rec["shap_values"]), key="analyze_shap")

    # Feature contribution table
    with st.expander("Feature Details", expanded=False):
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
        st.dataframe(feat_df.drop(columns=["|Impact|"]), hide_index=True, height=300)

    # ═══════════════════════════════════════════════════════════
    # STEP 6: MITIGATION
    # ═══════════════════════════════════════════════════════════
    if rec["prediction"] == "DoS":
        st.divider()
        st.subheader("Step 6 — Recommended Mitigation")

        attack_type = rec["attack_type"] or "Volumetric Flood"
        st.markdown(f"**Attack Type:** {attack_type} &nbsp;&nbsp; "
                    f"**Severity:** {rec['severity']}")
        st.markdown("**Recommended iptables / sysctl commands:**")

        commands = MITIGATION_COMMANDS.get(attack_type, MITIGATION_COMMANDS["Volumetric Flood"])
        for cmd, comment in commands:
            st.markdown(
                f'<div class="cmd-block">$ {cmd} '
                f'<span style="color:#8E8E93"># {comment}</span></div>',
                unsafe_allow_html=True,
            )

        st.caption("These commands are generated based on the SHAP-identified attack pattern. "
                   "Review and adapt before deploying in production environments.")
