"""About page — Research context and methodology."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_all_model_results, FEATURE_NAMES
from src.charts import model_comparison_bar

st.markdown("""
<div class="header-banner">
    <h2>About This System</h2>
    <p>Research context, methodology, and technical details</p>
</div>
""", unsafe_allow_html=True)

# ── Project Overview ──
st.subheader("Project Overview")
st.markdown("""
This system is a research project that combines Machine Learning with
**Explainable AI (XAI)** to detect Denial-of-Service (DoS) attacks in network
traffic and generate automated, transparent mitigation responses.

**Research Objectives:**
1. **Data Preparation** — Collect, clean, and engineer features from the UNSW-NB15 dataset
2. **Model Training** — Train and benchmark 7 ML models; select the best performer
3. **XAI Integration** — Apply SHAP TreeExplainer for mathematically exact feature explanations
4. **Mitigation Framework** — Use XAI output to classify attack types and generate response commands
""")

st.divider()

# ── Dataset ──
st.subheader("Dataset: UNSW-NB15")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    **Training Set:**
    - 24,528 samples (balanced: 12,264 DoS + 12,264 Normal)
    - 5-fold stratified cross-validation
    - Threshold optimized on validation set
    """)
with c2:
    st.markdown("""
    **Benchmark Test Set:**
    - 41,089 samples (imbalanced: 37,000 Normal + 4,089 DoS)
    - Simulates real-world class distribution (~90/10)
    - Completely unseen during training
    """)

st.markdown("**10 Selected Features:**")
feature_descriptions = {
    "rate": "Connection rate (packets/second)",
    "sload": "Source bits per second",
    "sbytes": "Source to destination bytes",
    "dload": "Destination bits per second",
    "proto": "Protocol type (encoded)",
    "dtcpb": "Destination TCP base sequence number",
    "stcpb": "Source TCP base sequence number",
    "dmean": "Mean of packet size from destination",
    "tcprtt": "TCP round trip time",
    "dur": "Connection duration",
}
feat_cols = st.columns(2)
for i, (feat, desc) in enumerate(feature_descriptions.items()):
    feat_cols[i % 2].markdown(f"- **`{feat}`** — {desc}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SYSTEM ARCHITECTURE — 7 Pipeline Steps
# ══════════════════════════════════════════════════════════════
st.subheader("System Architecture — 7-Step Detection Pipeline")

st.markdown("""
<div style="text-align:center; padding:1.5rem 0">
    <span class="pipeline-step" style="background:#E0F2F1;border-color:#80CBC4">1. Data Input<br><small>CSV Upload</small></span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step" style="background:#E0F2F1;border-color:#80CBC4">2. Feature Extraction<br><small>10 features</small></span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step" style="background:#E8F5E9;border-color:#A5D6A7">3. Encode & Scale<br><small>LabelEncoder + StandardScaler</small></span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step" style="background:#FFF3E0;border-color:#FFCC80">4. XGBoost<br><small>Detection</small></span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step" style="background:#FCE4EC;border-color:#F48FB1">5. SHAP<br><small>Explanation</small></span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step" style="background:#F3E5F5;border-color:#CE93D8">6. Classify<br><small>Attack Type</small></span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step" style="background:#E0F2F1;border-color:#80CBC4">7. Mitigate<br><small>iptables Commands</small></span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
| Step | Component | Description |
|:---:|---|---|
| **1** | **Data Input** | Upload raw UNSW-NB15 CSV (42+ columns) or preprocessed data (10 features) |
| **2** | **Feature Extraction** | Select the 10 most discriminative features from the raw dataset |
| **3** | **Encode & Scale** | Encode protocol type with LabelEncoder; normalize all features with StandardScaler (mean=0, std=1) |
| **4** | **XGBoost Detection** | Binary classification — each record receives P(DoS) and P(Normal); threshold at 0.8517 |
| **5** | **SHAP Explanation** | TreeExplainer computes exact Shapley values showing each feature's contribution toward DoS or Normal |
| **6** | **Attack Classification** | Top positive-SHAP features (pushing toward DoS) are matched to one of 4 attack types |
| **7** | **Mitigation** | Attack-specific iptables/sysctl commands are generated for immediate network defense |
""")

st.divider()

# ══════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
st.subheader("Model Comparison — All 7 Trained Models")
model_results = load_all_model_results()
comp_fig = model_comparison_bar(model_results)
st.plotly_chart(comp_fig, key="about_comp")

st.markdown("""
**Why XGBoost was selected:**
- Highest F1 score (90.26%) on the imbalanced benchmark test
- Gradient boosting handles tabular features better than sequence models (LSTM, CNN)
- Compatible with SHAP TreeExplainer — provides exact (not approximate) Shapley values
- Fast inference: ~0.01 seconds for 41,000 records
""")

st.divider()

# ══════════════════════════════════════════════════════════════
# SHAP EXPLANATION
# ══════════════════════════════════════════════════════════════
st.subheader("Step 5 — SHAP Explanation (Explainable AI)")

st.markdown("""
**SHAP (SHapley Additive exPlanations)** uses cooperative game theory to compute
each feature's exact contribution to every prediction. For each record, SHAP
assigns a signed value to every feature:

- **Positive SHAP** = pushes the prediction **toward DoS**
- **Negative SHAP** = pushes the prediction **toward Normal**

Only features with **positive SHAP values** are selected as **top contributors**
for DoS detections — these are the features that actually caused the model to
flag the traffic as an attack. Features with negative SHAP are excluded because
they push away from the DoS prediction.
""")

st.divider()

# ══════════════════════════════════════════════════════════════
# ATTACK CLASSIFICATION
# ══════════════════════════════════════════════════════════════
st.subheader("Step 6 — Attack Classification")

st.markdown("""
Based on which features have the highest positive SHAP values (top contributors
pushing toward DoS), each detected attack is classified into one of **4 types**:
""")

# ── 1. Volumetric Flood ──
st.markdown("""
**1. Volumetric Flood** — High-volume traffic flood that overwhelms
network bandwidth and resources.

Top predictors: `rate`, `sload`, `sbytes`
""")

# ── 2. Protocol Exploit ──
st.markdown("""
**2. Protocol Exploit** — Attack that exploits weaknesses in network protocols
such as TCP SYN floods or malformed packet attacks.

Top predictors: `proto`, `tcprtt`, `stcpb`, `dtcpb`
""")

# ── 3. Slowloris ──
st.markdown("""
**3. Slowloris** — Slow, persistent attack that keeps many connections open
for a long time to exhaust server resources with minimal bandwidth.

Top predictors: `dur`, `dmean`
""")

# ── 4. Amplification ──
st.markdown("""
**4. Amplification** — Attack where the response is significantly larger
than the request, using reflection techniques like DNS amplification.

Top predictors: `dload`
""")

st.divider()

# ══════════════════════════════════════════════════════════════
# MITIGATION COMMANDS
# ══════════════════════════════════════════════════════════════
st.subheader("Step 7 — Mitigation Commands")

st.markdown("For each classified attack type, the system generates specific "
            "**iptables** and **sysctl** commands for immediate network defense:")

# ── Volumetric Flood ──
st.markdown("**Volumetric Flood** — Rate-limit incoming SYN packets and apply traffic shaping:")
st.markdown("""
<div class="cmd-block">$ iptables -A INPUT -p tcp --syn -m limit --limit 10/s --limit-burst 20 -j ACCEPT <span style="color:#8E8E93"># Allow max 10 SYN/sec</span></div>
<div class="cmd-block">$ iptables -A INPUT -p tcp --syn -j DROP <span style="color:#8E8E93"># Drop excess SYN packets</span></div>
<div class="cmd-block">$ tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 400ms <span style="color:#8E8E93"># Cap bandwidth to 100Mbit</span></div>
""", unsafe_allow_html=True)

# ── Protocol Exploit ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Protocol Exploit** — Enable SYN cookies and drop malformed TCP packets:")
st.markdown("""
<div class="cmd-block">$ echo 1 > /proc/sys/net/ipv4/tcp_syncookies <span style="color:#8E8E93"># Enable SYN cookie protection</span></div>
<div class="cmd-block">$ iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT <span style="color:#8E8E93"># Strict SYN rate limit</span></div>
<div class="cmd-block">$ iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP <span style="color:#8E8E93"># Drop null flag packets</span></div>
<div class="cmd-block">$ iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP <span style="color:#8E8E93"># Drop XMAS scan packets</span></div>
""", unsafe_allow_html=True)

# ── Slowloris ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Slowloris** — Reduce TCP timeouts and limit concurrent connections per IP:")
st.markdown("""
<div class="cmd-block">$ sysctl -w net.ipv4.tcp_fin_timeout=30 <span style="color:#8E8E93"># Reduce FIN wait timeout</span></div>
<div class="cmd-block">$ sysctl -w net.ipv4.tcp_keepalive_time=300 <span style="color:#8E8E93"># Shorten keepalive interval</span></div>
<div class="cmd-block">$ iptables -A INPUT -p tcp -m connlimit --connlimit-above 10 -j DROP <span style="color:#8E8E93"># Max 10 connections per IP</span></div>
""", unsafe_allow_html=True)

# ── Amplification ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Amplification** — Drop oversized UDP responses and rate-limit UDP traffic:")
st.markdown("""
<div class="cmd-block">$ iptables -A INPUT -p udp --sport 53 -m length --length 512: -j DROP <span style="color:#8E8E93"># Block large DNS replies</span></div>
<div class="cmd-block">$ iptables -A INPUT -p udp -m limit --limit 100/s -j ACCEPT <span style="color:#8E8E93"># Rate-limit UDP traffic</span></div>
<div class="cmd-block">$ iptables -A INPUT -p udp -j DROP <span style="color:#8E8E93"># Drop excess UDP packets</span></div>
""", unsafe_allow_html=True)

st.caption("These commands are templates generated based on the SHAP-identified attack pattern. "
           "Review and adapt parameters before deploying in production environments.")

st.divider()

# ── Technology Stack ──
st.subheader("Technology Stack")
st.markdown("""
| Component | Technology |
|---|---|
| **ML Models** | XGBoost, Random Forest, SVM, MLP, Logistic Regression (scikit-learn) |
| **Deep Learning** | LSTM, 1D-CNN (TensorFlow/Keras) |
| **XAI** | SHAP TreeExplainer |
| **Dashboard** | Streamlit + Plotly |
| **Dataset** | UNSW-NB15 (University of New South Wales) |
| **Language** | Python 3.13 |
""")
