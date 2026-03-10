# Detection to Defense: An XAI-Powered DoS Detection System with Implementable Mitigation Protocols

B.Tech. Project — Department of Computer Science and Engineering, CHRIST (Deemed to be University), Bangalore

## Project Overview

This project presents an end-to-end system for detecting Denial-of-Service (DoS) attacks using an XGBoost classifier enhanced with SHAP-based Explainable AI. The system goes beyond detection by classifying attack types and generating actionable mitigation commands (iptables/sysctl) tailored to the identified threat pattern.

**Key results:**
- XGBoost model with 97.39% F1-score on UNSW-NB15 test set (optimized threshold: 0.8517)
- 10-feature pipeline with 76% dimensionality reduction from the original 42 features
- SHAP explanations for every detection, mapping to 4 attack categories with severity levels
- Cross-dataset validation on CIC-IDS2017 and CIC-DDoS2019
- Interactive Streamlit dashboard for real-time analysis

## Live Dashboard

The dashboard is deployed on Streamlit Community Cloud and accessible to anyone:

**[Launch Dashboard](https://cti-ids.streamlit.app)** *(link will be active after deployment)*

Upload your own network traffic CSV or use the built-in UNSW-NB15 test set (41,089 samples) to run the full pipeline: detection, SHAP explanation, attack classification, and mitigation recommendations.

## Repository Structure

```
CTI_IDS/
├── 01_data_preparation/       # Dataset extraction and preprocessing
├── 03_model_training/         # Model training, evaluation, and artifacts
│   └── proper_training/
│       ├── data/              # Scalers, encoders, train/test splits
│       ├── models/            # XGBoost, LSTM, 1D-CNN, RF, SVM, MLP, LR, DT
│       └── results/           # Benchmark results and comparisons
├── 04_xai_integration/        # SHAP integration and analysis
├── 05_mitigation_framework/   # Attack classification and mitigation rules
├── 06_dashboard/              # Streamlit web application
│   ├── app.py                 # Entry point
│   ├── pages/                 # Dashboard, Analyze, About pages
│   ├── src/                   # Models, pipeline, charts modules
│   └── assets/                # CSS styling
├── docs/                      # Project documentation
├── tools/                     # Utility scripts
├── figures/                   # Publication figures
└── requirements.txt           # Python dependencies
```

## Quick Start

```bash
git clone https://github.com/AkashMadanu/CTI_IDS.git
cd CTI_IDS
pip install -r requirements.txt
cd 06_dashboard
streamlit run app.py
```

## Team

This project was developed collaboratively by:

- **Sweta Sharma** (2260452)
- **Madanu Akash** (2260393)
- **Devika K P** (2260359)

### Guidance

We are grateful to our project guide **Dr. Daniel D**, Associate Professor, Department of Computer Science and Engineering, CHRIST (Deemed to be University), for his consistent guidance, technical insights, and encouragement throughout this project. His feedback shaped the direction of our work in meaningful ways.

We also thank the faculty and staff of the Department of Computer Science and Engineering for their support during the course of this project.

## Academic Year

2025–2026
