# 06_dashboard — Streamlit Interactive Dashboard

## Overview

This directory contains the **modular Streamlit dashboard** that serves as the visual interface for the entire XAI-Powered DoS Detection & Mitigation System. It ties together all project components (data preparation, model training, SHAP explainability, attack classification, and mitigation) into a single interactive web application.

**Run command:** `streamlit run app.py` (from this directory)
**URL:** http://localhost:8501

---

## Directory Structure

```
06_dashboard/
├── app.py                  # Entry point — page config, CSS loading, navigation
├── .streamlit/
│   └── config.toml         # Streamlit theme (light base, teal accent, sans-serif)
├── assets/
│   └── style.css           # Custom CSS — Apple/Google-inspired design
├── pages/
│   ├── dashboard.py        # Page 1: Model performance overview (7 models)
│   ├── analyze.py          # Page 2: Upload CSV → Detect → Explain → Mitigate
│   └── about.py            # Page 3: Research context, methodology, architecture
├── src/
│   ├── __init__.py         # Package marker
│   ├── models.py           # Model/scaler/encoder loading + results aggregation
│   ├── pipeline.py         # Preprocessing pipelines + detection engine
│   └── charts.py           # Plotly chart helpers (donut, bar, heatmap, SHAP)
└── CHANGES.md              # This file
```

---

## File-by-File Documentation

### `app.py` — Entry Point

- Configures Streamlit page settings (title, icon, wide layout)
- Loads custom CSS from `assets/style.css`
- Defines top navigation with 3 pages: Dashboard, Analyze, About
- Uses `st.navigation()` with `position="top"` for flat horizontal nav

### `.streamlit/config.toml` — Theme Configuration

- Light base theme with teal primary color (`#00897B`)
- White background with light gray secondary (`#F5F5F7`)
- Dark text color (`#1D1D1F`) for high contrast
- Usage stats collection disabled

### `assets/style.css` — Custom Styling

Publication-grade CSS inspired by Apple/Google design language:
- **Inter font** imported from Google Fonts
- **Metric cards** with 12px rounded corners, subtle shadows, uppercase labels
- **Severity badges** (CRITICAL=red, HIGH=orange, MEDIUM=yellow, LOW=green)
- **Terminal-style command blocks** (dark background, green monospace text)
- **Pipeline flow arrows** with styled step boxes
- **Header banners** with dark gradient backgrounds
- Hides Streamlit branding (footer, toolbar, deploy button)

### `pages/dashboard.py` — Model Performance Overview

Displays at-a-glance metrics for the selected model (XGBoost):
- **5 metric cards:** Accuracy, Precision, Recall, F1 Score, Threshold
- **Confusion matrix heatmap** from benchmark test (41,089 samples)
- **Model comparison bar chart** showing all 7 trained models side by side
- **Comparison data table** with CV F1, Benchmark F1, Accuracy, Precision, Recall, Threshold
- **Pipeline diagram** showing the 7-step detection flow

### `pages/analyze.py` — Core Analysis Page

This is the main feature page. End-to-end flow:

1. **Step 1 — Data Input:**
   - File uploader accepting CSV files
   - "Load Sample Test Data" button (UNSW-NB15 test set, 41,089 samples)

2. **Step 2 — Preprocessing:**
   - Auto-detects CSV format using `detect_csv_type()`:
     - **Raw UNSW-NB15** (42+ columns): Full 5-step pipeline (filter → select features → encode protocol → fill missing → scale)
     - **CIC-IDS2017 / CIC-DDoS2019** (~80 columns): CIC adapter pipeline (extract labels → map 7 features + 3 neutral → encode protocol → scale)
     - **Preprocessed** (10 columns): Direct use
   - Shows progress via `st.status()` for each preprocessing step

3. **Step 3 — Run Detection:**
   - Adjustable record count slider (100 to max records)
   - Caching to avoid recomputation for identical inputs
   - Runs XGBoost inference + SHAP explanations + attack classification

4. **Step 4 — Detection Results:**
   - Summary metrics: Total Analyzed, DoS Detected, Normal Traffic
   - Accuracy, Precision, Recall, F1 Score (when ground-truth labels exist)
   - **Info box explanation** when metrics are zero (e.g., CIC dataset's first N records are all benign)
   - **Overview tab:** Detection Distribution donut chart + Severity bar chart
   - **Confusion Matrix tab** (when labels available)

5. **Step 5 — Individual Record Analysis:**
   - Dropdown to select any record (defaults to first DoS detection)
   - Detection result badge (DoS/Normal) with confidence and P(DoS)
   - Attack classification (type + severity) for DoS records
   - **SHAP waterfall chart** showing feature contributions
   - Feature details table with SHAP values and direction

6. **Step 6 — Recommended Mitigation:**
   - Attack-type-specific iptables/sysctl commands
   - Terminal-styled command blocks with comments

### `pages/about.py` — Research Context

Static documentation page containing:
- Project overview and 4 research objectives
- Dataset description (UNSW-NB15 training/test splits)
- 10 selected features with descriptions
- System architecture — 7-step pipeline with styled flow diagram and table
- Model comparison chart and selection rationale
- SHAP explanation methodology
- Attack classification types (4 categories with top predictors)
- Mitigation command templates for all 4 attack types
- Technology stack table

### `src/models.py` — Model Loading & Results

- Defines paths relative to project root (`PROJECT_ROOT` → `03_model_training/proper_training/`)
- `FEATURE_NAMES`: The 10 model features in order
- `THRESHOLD`: 0.8517 (optimized on validation set)
- `load_model()`: Loads `xgboost_model.pkl` (cached with `@st.cache_resource`)
- `load_scaler()`: Loads `feature_scaler.pkl` (StandardScaler)
- `load_encoder()`: Loads `proto_encoder.pkl` (LabelEncoder, 132 classes)
- `load_shap_explainer()`: Creates SHAP TreeExplainer from model
- `load_sample_data()`: Loads X_test_scaled.csv + y_test.csv (cached with `@st.cache_data`)
- `load_all_model_results()`: Aggregates results from all 7 models (XGBoost, RF, SVM, MLP, LR, LSTM, 1D-CNN)

### `src/pipeline.py` — Preprocessing & Detection Engine

**`detect_csv_type(df)`** — Auto-detection logic:
- CIC format: Checks for "Flow Duration" AND "Total Fwd Packets" in column names
- Raw UNSW-NB15: More than 15 columns
- Preprocessed: Exactly 10 columns matching FEATURE_NAMES
- Returns: "cic", "raw", "preprocessed", or "unknown"

**`preprocess_raw_csv(df, status_container)`** — UNSW-NB15 pipeline:
1. Filter relevant traffic (DoS + Normal only)
2. Select 10 features (fill missing with zeros)
3. Encode protocol column (LabelEncoder, unknown → "tcp")
4. Fill NaN values with median
5. Scale with StandardScaler

**`preprocess_cic_csv(df, status_container)`** — CIC adapter pipeline:
1. Extract labels (BENIGN → 0, anything else → 1)
2. Map CICFlowMeter features → 10 model features:
   - `rate` ← Flow Packets/s
   - `sload` ← Total Length of Fwd Packets × 8 / duration (bits/sec)
   - `sbytes` ← Total Length of Fwd Packets
   - `dload` ← Total Length of Bwd Packets × 8 / duration (bits/sec)
   - `dmean` ← Bwd Packet Length Mean
   - `dur` ← Flow Duration / 1,000,000 (microseconds → seconds)
   - `proto` ← Protocol column → name → LabelEncoder (or TCP default)
   - `stcpb`, `dtcpb`, `tcprtt` ← Training mean (neutral after scaling → 0)
3. Encode protocol numbers to names using lookup table
4. Set unavailable features (stcpb, dtcpb, tcprtt) to training mean
5. Assemble + scale with StandardScaler

**`run_detection(X_df, y_labels, status_container)`** — Vectorized detection:
1. XGBoost `predict_proba()` on all records
2. SHAP `shap_values()` for all records
3. For each record: classify attack type, calculate severity, extract top features

**`classify_attack(top_features)`** — Maps top SHAP features to 4 attack types:
- Volumetric Flood: rate, sload, sbytes
- Protocol Exploit: proto, tcprtt, stcpb, dtcpb
- Slowloris: dur, dmean
- Amplification: dload

**`calc_severity(confidence)`** — Maps prediction confidence to severity:
- ≥95%: CRITICAL, ≥90%: HIGH, ≥75%: MEDIUM, ≥60%: LOW

**`MITIGATION_COMMANDS`** — Dictionary mapping each attack type to iptables/sysctl commands with comments.

### `src/charts.py` — Plotly Chart Helpers

All charts use a clean white theme with Inter font.

**`model_comparison_bar(model_results)`** — Grouped bar chart comparing all models on F1, Accuracy, Precision, Recall.

**`confusion_matrix_heatmap(tp, tn, fp, fn, title)`** — Publication-grade heatmap with blue color scale, cell annotations (TP/TN/FP/FN with counts).

**`attack_donut(results)`** — Donut chart of detection distribution:
- Slices: Normal (green) + attack types (red/orange/yellow/purple)
- Percentages displayed inside slices (white text)
- Horizontal legend below chart with category names, counts, and percentages
- Title pinned to top (`y=0.98`) to prevent label overlap

**`severity_bar(results)`** — Bar chart of severity level distribution (CRITICAL/HIGH/MEDIUM/LOW) among DoS detections.

**`shap_waterfall(shap_vals)`** — Horizontal bar chart showing SHAP feature contributions:
- Red bars: pushes toward DoS (positive SHAP)
- Blue bars: pushes toward Normal (negative SHAP)
- Sorted by absolute value, legend annotation at bottom

**`results_confusion_matrix(results)`** — Builds confusion matrix from detection results list.

---

## Key Design Decisions

1. **CIC Adapter** — Maps CICFlowMeter features to UNSW-NB15 model features using bits-per-second for sload/dload (matching UNSW-NB15's Argus definition). 3 unavailable features (stcpb, dtcpb, tcprtt) set to training mean → scale to 0 (neutral).

2. **Donut Chart Labels** — Moved from on-chart text (which overlapped with title) to inside-slice percentages + horizontal legend below. Prevents overlap at any label values.

3. **Zero Metrics Info Box** — When precision/recall/F1 are all 0%, an explanatory info box appears (e.g., CIC dataset's first N records may all be benign).

4. **Directional SHAP Selection** — Top features for DoS predictions use only positive SHAP values (features pushing toward DoS). This ensures attack classification is based on actual attack indicators, not overall feature importance.

5. **Caching Strategy** — Model, scaler, encoder use `@st.cache_resource` (singleton). Sample data uses `@st.cache_data`. Detection results use a manual cache key to avoid recomputation.

---

## Cross-Dataset Support

| Dataset | Format | Columns | Detection | Auto-Detected By |
|---------|--------|---------|-----------|-----------------|
| UNSW-NB15 (raw) | CSV, 42+ cols | Full feature set | `preprocess_raw_csv()` | Column count > 15 |
| CIC-IDS2017 | CSV, ~80 cols | CICFlowMeter output | `preprocess_cic_csv()` | "Flow Duration" + "Total Fwd Packets" |
| CIC-DDoS2019 | CSV, ~80 cols | CICFlowMeter output | `preprocess_cic_csv()` | "Flow Duration" + "Total Fwd Packets" |
| Preprocessed | CSV, 10 cols | Scaled features | Direct use | 10 columns matching FEATURE_NAMES |
