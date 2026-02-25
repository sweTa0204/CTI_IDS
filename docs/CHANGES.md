# CTI_IDS — Project-Level Changes Summary

## Overview

This document summarizes all changes and new additions made to the XAI-Powered DoS Detection & Mitigation System. Each directory with changes has its own `CHANGES.md` with detailed file-by-file documentation.

---

## Directory-Level Change Index

| Directory | Type | Summary |
|-----------|------|---------|
| `04_xai_integration/` | Modified | Directional SHAP feature selection in `shap_explainer.py` |
| `05_mitigation_framework/` | Modified | Refined Slowloris detection thresholds in `attack_classifier.py` |
| `06_complete_testing/` | Modified | Directional top-feature selection in both testing scripts |
| `06_dashboard/` | **New** | Complete modular Streamlit dashboard (app + 3 pages + src modules) |
| `datasets/` | **New** | Reference datasets (UNSW-NB15 test set + CIC-IDS2017 Friday DDoS) |
| `03_model_training/.../decisiontree/` | **New** | Decision Tree model training, results, and comparison charts |

---

## New Root-Level Files

### `dashboard.py` — Standalone Dashboard (Legacy)

A single-file Streamlit dashboard (1,078 lines) that was the initial prototype before the modular `06_dashboard/` was created. Contains the complete pipeline in one file:
- Page config and custom CSS
- Model/scaler/encoder loading
- Full preprocessing pipeline (raw UNSW-NB15 + CIC adapter)
- XGBoost detection + SHAP explanation
- Attack classification and mitigation
- Plotly charts (donut, severity bar, confusion matrix, SHAP waterfall)
- Multi-page navigation (Dashboard, Analyze, About)

**Status:** Superseded by `06_dashboard/` (modular version). Kept for reference but the modular dashboard is the primary interface.

**Run:** `streamlit run dashboard.py` (from CTI_IDS directory)

### `pcap_to_features.py` — PCAP to Model Features Extractor

A standalone utility script (371 lines) that converts raw PCAP files from live network captures into the 10-feature CSV format required by the XGBoost model.

**What it does:**
1. Reads a PCAP file using Scapy
2. Groups packets into bidirectional TCP/UDP flows (5-tuple: src_ip, src_port, dst_ip, dst_port, proto)
3. Implements idle-timeout flow splitting (5 second timeout) to prevent port-reuse from merging unrelated flows
4. Computes the 10 model features from each flow:
   - `rate` — packets/second (total_pkts / duration)
   - `sload` — source bytes/second (fwd_bytes / duration)
   - `sbytes` — total forward bytes
   - `dload` — destination bytes/second (bwd_bytes / duration)
   - `proto` — protocol number → name → LabelEncoder value
   - `dtcpb` — destination TCP base sequence number
   - `stcpb` — source TCP base sequence number
   - `dmean` — mean backward packet size
   - `tcprtt` — TCP round-trip time (SYN → SYN-ACK delta)
   - `dur` — flow duration in seconds
5. Scales features with the saved StandardScaler
6. Outputs two CSV files:
   - `live_traffic_scaled.csv` — 10 columns, scaled, ready for dashboard upload
   - `live_traffic_raw.csv` — raw values + metadata (src/dst IPs, packet counts, timestamps)

**Key design: `Flow` class** uses `__slots__` for memory efficiency and tracks SYN/SYN-ACK timestamps for TCP RTT calculation.

**Usage:** `python pcap_to_features.py capture.pcap`

### `review_notes_spm_adaboost.md` — Review Notes

Study notes on Sequential Pattern Mining (SPM) and AdaBoost, covering:
- SPM algorithms (AprioriAll, GSP, PrefixSpan) and their application to network security
- AdaBoost (Adaptive Boosting) — ensemble method, comparison with XGBoost
- How these relate to the project's detection approach

---

## Cross-Cutting Change: Directional SHAP Feature Selection

The most significant behavioral change affects 4 files across 4 directories. Previously, "top features" for any prediction were selected by **absolute SHAP value** — the 3 features with the largest magnitude regardless of sign. This was incorrect because:

- For a DoS prediction, a feature with large **negative** SHAP pushes **away from DoS** (toward Normal). Listing it as a "top contributor to DoS" is misleading.
- The attack classifier uses top features to determine attack type. Including wrong-direction features leads to incorrect classifications.

**The fix:** All 4 locations now use **directional selection**:
- DoS predictions → only **positive SHAP** features (pushing toward DoS)
- Normal predictions → only **negative SHAP** features (pushing toward Normal)
- Fallback to absolute sorting if no features match

**Files changed:**
1. `04_xai_integration/shap_explainer.py` — `SHAPExplainer.explain_single()`
2. `05_mitigation_framework/attack_classifier.py` — `_calc_slowloris_score()` thresholds
3. `06_complete_testing/demo_single_sample.py` — `demo_pipeline()` top features
4. `06_complete_testing/run_complete_test.py` — `explain_single_with_threshold()` top features
5. `06_dashboard/src/pipeline.py` — `_top_features_for_row()` (new, already directional)

---

## New: CIC-IDS2017/CIC-DDoS2019 Adapter

The dashboard now supports uploading CIC-format CSV files (from CICFlowMeter), not just UNSW-NB15. The adapter:
1. Auto-detects CIC format by checking for "Flow Duration" and "Total Fwd Packets" columns
2. Maps 7 CICFlowMeter features → model features (using bits/sec for sload/dload)
3. Sets 3 unavailable features (stcpb, dtcpb, tcprtt) to training mean (scales to 0 = neutral)
4. Encodes protocol numbers to names using a lookup table
5. Scales with the UNSW-NB15 StandardScaler

Implemented in:
- `06_dashboard/src/pipeline.py` — `preprocess_cic_csv()` and updated `detect_csv_type()`
- `06_dashboard/pages/analyze.py` — CIC branch in upload handler
- `dashboard.py` — Same changes in standalone version

---

## New: Streamlit Dashboard (`06_dashboard/`)

A complete modular Streamlit dashboard with 3 pages:
1. **Dashboard** — Model performance overview (7 models, confusion matrix, comparison chart)
2. **Analyze** — Upload CSV → Preprocess → Detect → Explain → Mitigate (the core feature)
3. **About** — Research context, architecture, methodology

See `06_dashboard/CHANGES.md` for full file-by-file documentation.

---

## New: Reference Datasets (`datasets/`)

Contains copies of test datasets for easy browsing from the Analyze page:
- `UNSW_NB15_test_scaled.csv` (7.9 MB, 41,089 records, preprocessed)
- `CIC-IDS2017_Friday_DDos.csv` (74 MB, 225,745 records, CICFlowMeter format)

See `datasets/CHANGES.md` for full documentation including cross-dataset performance notes.
