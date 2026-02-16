# datasets — Reference Datasets for Dashboard Testing

## Overview

This directory contains copies of reference datasets used by the dashboard's Analyze page. Having them in a single location makes it easy to browse and upload files for testing without navigating to their original locations in the project tree.

---

## Files

### `UNSW_NB15_test_scaled.csv` (7.9 MB)

- **Source:** Copied from `03_model_training/proper_training/data/X_test_scaled.csv`
- **Records:** 41,089
- **Columns:** 10 (the model's feature set, already scaled)
- **Features:** rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur
- **Scaling:** StandardScaler applied (mean=0, std=1)
- **Labels:** Available via `03_model_training/proper_training/data/y_test.csv` (loaded automatically by the "Load Sample Test Data" button)
- **Class Distribution:** 37,000 Normal + 4,089 DoS (real-world imbalance ~90/10)
- **Usage:** This is the official UNSW-NB15 benchmark test set. Upload to the Analyze page as a "preprocessed" CSV, or use the "Load Sample Test Data" button which loads this file with labels.

### `CIC-IDS2017_Friday_DDos.csv` (74 MB)

- **Source:** Copied from `CIC-IDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- **Original Dataset:** CIC-IDS2017 from the Canadian Institute for Cybersecurity (downloaded from Kaggle: `chethuhn/network-intrusion-dataset`)
- **Records:** 225,745
- **Columns:** 79 (CICFlowMeter output format)
- **Labels:** "DDoS" (128,027 records) and "BENIGN" (97,718 records)
- **Label Column:** `Label` (text: "BENIGN" or "DDoS")
- **Class Distribution:** ~57% DDoS / ~43% BENIGN
- **Important Note:** Records are ordered chronologically — benign traffic comes first, then DDoS starts. The first ~97,718 records are all BENIGN. To get a mix of both classes, either analyze all records or use a large record count (>100,000).
- **Usage:** Upload to the Analyze page. The CIC adapter pipeline automatically detects the format and maps CICFlowMeter features to the 10 model features.

---

## How the Dashboard Uses These Files

1. **UNSW-NB15 test set:** The "Load Sample Test Data" button loads this directly from `03_model_training/proper_training/data/` (not from this directory). This copy is for manual upload testing via the file uploader.

2. **CIC-IDS2017:** Upload via the file uploader on the Analyze page. The dashboard auto-detects it as CIC format and runs the CIC adapter pipeline:
   - Maps 7 directly available features (rate, sload, sbytes, dload, dmean, dur, proto)
   - Sets 3 unavailable features (stcpb, dtcpb, tcprtt) to training mean (neutral)
   - Scales all 10 features with the UNSW-NB15 StandardScaler

---

## Cross-Dataset Performance Notes

| Metric | UNSW-NB15 (in-distribution) | CIC-IDS2017 (cross-dataset) |
|--------|----------------------------|---------------------------|
| Accuracy | 88.38% | 88.40%* |
| Precision | 99.24% | 85.3% |
| Recall | 86.45% | 32.8% |
| F1 Score | 92.41% | 47.4% |

*CIC accuracy is high due to class imbalance — most records are benign and correctly classified as Normal. The lower recall indicates that the model (trained on UNSW-NB15) misses many CIC DDoS patterns, which is expected for cross-dataset evaluation.

**Why lower performance on CIC?** The model was trained on UNSW-NB15 features. CIC-IDS2017 uses a different feature extraction tool (CICFlowMeter vs Argus), so feature distributions differ. Three features (stcpb, dtcpb, tcprtt) are unavailable in CIC data and are set to neutral values, reducing the model's discriminative power.
