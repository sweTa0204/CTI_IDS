"""Preprocessing and detection pipeline."""

import numpy as np
import pandas as pd
import streamlit as st

from .models import (
    FEATURE_NAMES, THRESHOLD,
    load_model, load_scaler, load_encoder, load_shap_explainer,
)


def preprocess_raw_csv(df, status_container=None):
    """Preprocess a raw UNSW-NB15 CSV through the full pipeline.

    If status_container is provided (an st.status context), writes
    progress steps into it.
    """
    def log(msg):
        if status_container:
            status_container.write(msg)

    original_count = len(df)

    # Step 1 — Identify labels if present
    y_labels = None
    label_col = None
    for c in ["attack_cat", "label"]:
        if c in df.columns:
            label_col = c
            break

    if label_col == "attack_cat":
        mask = df["attack_cat"].str.strip().isin(["DoS", "Normal", ""])
        df = df[mask | df["attack_cat"].isna()].copy()
        if "label" in df.columns:
            y_labels = df["label"].values
        else:
            y_labels = (df["attack_cat"].str.strip() == "DoS").astype(int).values
        log(f"**Step 1/5 — Filter relevant traffic:** {original_count:,} → {len(df):,} records")
    elif label_col == "label":
        y_labels = df["label"].values
        log(f"**Step 1/5 — Labels detected:** {len(df):,} records with ground truth")
    else:
        log(f"**Step 1/5 — No labels found:** {len(df):,} records (prediction-only mode)")

    # Step 2 — Select features
    available = [f for f in FEATURE_NAMES if f in df.columns]
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    X = df[available].copy()
    for m in missing:
        X[m] = 0
    X = X[FEATURE_NAMES]
    log(f"**Step 2/5 — Feature selection:** {len(available)}/10 features extracted"
        + (f" *(missing: {missing})*" if missing else ""))

    # Step 3 — Encode protocol
    encoder = load_encoder()
    if X["proto"].dtype == object:
        known = set(encoder.classes_)
        X["proto"] = X["proto"].apply(lambda v: v if v in known else "tcp")
        X["proto"] = encoder.transform(X["proto"])
        log("**Step 3/5 — Protocol encoding:** Categorical → numeric (LabelEncoder)")
    else:
        log("**Step 3/5 — Protocol encoding:** Already numeric, skipped")

    # Step 4 — Fill missing values
    n_missing = int(X.isnull().sum().sum())
    X = X.fillna(X.median())
    log(f"**Step 4/5 — Missing values:** {n_missing} found and imputed with median")

    # Step 5 — Scale
    scaler = load_scaler()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_NAMES, index=X.index)
    log("**Step 5/5 — Feature scaling:** StandardScaler applied (mean=0, std=1)")

    return X_scaled, y_labels


def preprocess_cic_csv(df, status_container=None):
    """Preprocess a CIC-format CSV (CIC-IDS2017, CIC-DDoS2019) through adapter.

    Maps CICFlowMeter features to the 10 model features used by our XGBoost
    DoS detector, sets unavailable features (stcpb, dtcpb, tcprtt) to neutral
    values, and scales with the saved UNSW-NB15 StandardScaler.

    Feature mapping:
        rate   ← Flow Packets/s
        sload  ← Total Length of Fwd Packets × 8 / dur  (bits/sec)
        sbytes ← Total Length of Fwd Packets
        dload  ← Total Length of Bwd Packets × 8 / dur  (bits/sec)
        proto  ← Protocol column → LabelEncoder  (or TCP default)
        dtcpb  ← training mean  (neutral, scales to 0)
        stcpb  ← training mean  (neutral, scales to 0)
        dmean  ← Bwd Packet Length Mean
        tcprtt ← training mean  (neutral, scales to 0)
        dur    ← Flow Duration / 1e6  (µs → seconds)
    """
    def log(msg):
        if status_container:
            status_container.write(msg)

    df = df.copy()
    df.columns = df.columns.str.strip()
    original_count = len(df)

    # ── Step 1: Extract labels ──────────────────────────────────────
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
        log(f"**Step 1/5 — Labels detected:** {original_count:,} records — "
            f"{n_attack:,} attack, {n_benign:,} benign. Types: {type_str}")
    else:
        log(f"**Step 1/5 — No labels found:** {original_count:,} records "
            f"(prediction-only mode)")

    # ── Step 2: Map CIC features → 10 model features ───────────────
    log("**Step 2/5 — Feature mapping:** CICFlowMeter → 10 model features")

    dur_us = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0).values
    dur = np.maximum(dur_us / 1e6, 0.0)                    # µs → seconds

    rate = pd.to_numeric(df["Flow Packets/s"], errors="coerce").fillna(0).values

    sbytes = pd.to_numeric(
        df["Total Length of Fwd Packets"], errors="coerce"
    ).fillna(0).values.astype(float)

    bwd_bytes = pd.to_numeric(
        df["Total Length of Bwd Packets"], errors="coerce"
    ).fillna(0).values.astype(float)

    # sload / dload in bits-per-second (matches UNSW-NB15 Argus definition)
    with np.errstate(divide="ignore", invalid="ignore"):
        sload = np.where(dur > 0, sbytes * 8.0 / dur, 0.0)
        dload = np.where(dur > 0, bwd_bytes * 8.0 / dur, 0.0)

    dmean = pd.to_numeric(
        df["Bwd Packet Length Mean"], errors="coerce"
    ).fillna(0).values.astype(float)

    # ── Step 3: Protocol encoding ───────────────────────────────────
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
        log("**Step 3/5 — Protocol encoding:** Protocol column → LabelEncoder")
    else:
        tcp_val = float(encoder.transform(["tcp"])[0])
        proto = np.full(len(df), tcp_val)
        log("**Step 3/5 — Protocol encoding:** No Protocol column — defaulting "
            "to TCP (most CIC traffic is HTTP-based)")

    # ── Step 4: Unavailable features → neutral ──────────────────────
    scaler = load_scaler()
    dtcpb  = np.full(len(df), scaler.mean_[5])   # index 5 in FEATURE_NAMES
    stcpb  = np.full(len(df), scaler.mean_[6])   # index 6 in FEATURE_NAMES
    tcprtt = np.full(len(df), scaler.mean_[8])    # index 8 in FEATURE_NAMES
    log("**Step 4/5 — Unavailable features:** stcpb, dtcpb, tcprtt → training "
        "mean (neutral after scaling)")

    # ── Step 5: Assemble + scale ────────────────────────────────────
    # Order: rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur
    X_raw = np.column_stack([
        rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur,
    ])
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = pd.DataFrame(
        scaler.transform(X_raw), columns=FEATURE_NAMES, index=df.index,
    )
    log("**Step 5/5 — Feature scaling:** StandardScaler applied (mean=0, std=1)")

    return X_scaled, y_labels


def detect_csv_type(df):
    """Determine CSV type: raw UNSW-NB15, CIC (CICFlowMeter), or preprocessed."""
    cols_stripped = set(c.strip() for c in df.columns)
    # CIC format (CIC-IDS2017, CIC-DDoS2019): CICFlowMeter output with ~80 cols
    if "Flow Duration" in cols_stripped and "Total Fwd Packets" in cols_stripped:
        return "cic"
    # Raw UNSW-NB15 (42+ columns)
    if len(df.columns) > 15:
        return "raw"
    # Preprocessed (10 scaled features)
    feature_set = set(FEATURE_NAMES)
    if feature_set.issubset(set(df.columns)) or len(df.columns) == 10:
        return "preprocessed"
    return "unknown"


def classify_attack(top_features):
    """Classify attack type based on SHAP top features."""
    volumetric = {"rate", "sload", "sbytes"}
    protocol = {"proto", "tcprtt", "stcpb", "dtcpb"}
    slowloris = {"dur", "dmean"}
    amplification = {"dload"}

    top_set = set(top_features[:3])
    scores = {
        "Volumetric Flood": len(top_set & volumetric),
        "Protocol Exploit": len(top_set & protocol),
        "Slowloris": len(top_set & slowloris),
        "Amplification": len(top_set & amplification),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Volumetric Flood"


def calc_severity(confidence):
    if confidence >= 0.95:
        return "CRITICAL"
    elif confidence >= 0.90:
        return "HIGH"
    elif confidence >= 0.75:
        return "MEDIUM"
    elif confidence >= 0.60:
        return "LOW"
    return None


MITIGATION_COMMANDS = {
    "Volumetric Flood": [
        ("iptables -A INPUT -p tcp --syn -m limit --limit 10/s --limit-burst 20 -j ACCEPT", "Allow max 10 SYN/sec"),
        ("iptables -A INPUT -p tcp --syn -j DROP", "Drop excess SYN packets"),
        ("tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 400ms", "Cap bandwidth to 100Mbit"),
    ],
    "Protocol Exploit": [
        ("echo 1 > /proc/sys/net/ipv4/tcp_syncookies", "Enable SYN cookie protection"),
        ("iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT", "Strict SYN rate limit"),
        ("iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP", "Drop null flag packets"),
        ("iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP", "Drop XMAS scan packets"),
    ],
    "Slowloris": [
        ("sysctl -w net.ipv4.tcp_fin_timeout=30", "Reduce FIN wait timeout"),
        ("sysctl -w net.ipv4.tcp_keepalive_time=300", "Shorten keepalive interval"),
        ("iptables -A INPUT -p tcp -m connlimit --connlimit-above 10 -j DROP", "Max 10 connections per IP"),
    ],
    "Amplification": [
        ("iptables -A INPUT -p udp --sport 53 -m length --length 512: -j DROP", "Block large DNS replies"),
        ("iptables -A INPUT -p udp -m limit --limit 100/s -j ACCEPT", "Rate-limit UDP traffic"),
        ("iptables -A INPUT -p udp -j DROP", "Drop excess UDP packets"),
    ],
}


def _top_features_for_row(shap_row, is_dos):
    """Return top-3 feature names pushing toward the predicted class."""
    names = np.array(FEATURE_NAMES)
    if is_dos:
        mask = shap_row > 0
        if mask.any():
            idx = np.where(mask)[0]
            order = idx[np.argsort(-shap_row[idx])]
            return names[order[:3]].tolist()
    else:
        mask = shap_row < 0
        if mask.any():
            idx = np.where(mask)[0]
            order = idx[np.argsort(-np.abs(shap_row[idx]))]
            return names[order[:3]].tolist()
    # Fallback: sort by absolute value
    order = np.argsort(-np.abs(shap_row))
    return names[order[:3]].tolist()


def run_detection(X_df, y_labels=None, status_container=None):
    """Run XGBoost detection + SHAP on all records (vectorized)."""
    def log(msg):
        if status_container:
            status_container.write(msg)

    model = load_model()
    explainer = load_shap_explainer(model)

    X_arr = X_df.values if hasattr(X_df, "values") else np.array(X_df)
    n = len(X_arr)

    # Step 1: XGBoost inference (vectorized, ~0.01s)
    log(f"**Step 1/3 — XGBoost inference** on {n:,} records...")
    probas = model.predict_proba(X_arr)
    p_dos_arr = probas[:, 1]
    is_dos_arr = p_dos_arr >= THRESHOLD

    # Step 2: SHAP explanations (vectorized C backend, main cost)
    log(f"**Step 2/3 — Computing SHAP explanations** for {n:,} records...")
    shap_values = explainer.shap_values(X_arr)

    # Step 3: Build results (vectorized where possible)
    log(f"**Step 3/3 — Classifying attacks** for {n:,} records...")
    confidence_arr = np.where(is_dos_arr, p_dos_arr, 1 - p_dos_arr)
    confidence_pct = np.round(confidence_arr * 100, 2)
    p_dos_pct = np.round(p_dos_arr * 100, 2)

    # Vectorize severity: map confidence to severity strings
    severity_arr = np.full(n, None, dtype=object)
    dos_mask = is_dos_arr
    severity_arr[dos_mask & (p_dos_arr >= 0.95)] = "CRITICAL"
    severity_arr[dos_mask & (p_dos_arr >= 0.90) & (p_dos_arr < 0.95)] = "HIGH"
    severity_arr[dos_mask & (p_dos_arr >= 0.75) & (p_dos_arr < 0.90)] = "MEDIUM"
    severity_arr[dos_mask & (p_dos_arr >= 0.60) & (p_dos_arr < 0.75)] = "LOW"

    # Actuals (vectorized)
    actual_arr = np.full(n, None, dtype=object)
    if y_labels is not None:
        y = np.asarray(y_labels[:n], dtype=int)
        actual_arr[:len(y)] = np.where(y == 1, "DoS", "Normal")

    # Build results list
    results = []
    for i in range(n):
        is_dos = bool(is_dos_arr[i])
        sv_row = shap_values[i]
        top3 = _top_features_for_row(sv_row, is_dos)
        sv_dict = {FEATURE_NAMES[j]: float(sv_row[j]) for j in range(10)}

        results.append({
            "index": i,
            "prediction": "DoS" if is_dos else "Normal",
            "confidence": float(confidence_pct[i]),
            "p_dos": float(p_dos_pct[i]),
            "attack_type": classify_attack(top3) if is_dos else None,
            "severity": severity_arr[i],
            "top_features": top3,
            "shap_values": sv_dict,
            "actual": actual_arr[i],
        })

    return results
