# 🔍 SHAP & LIME Integration - Explainable AI for DoS Detection

## Purpose
Apply SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to understand WHY our models make specific predictions - critical for CTI (Cyber Threat Intelligence) applications.

---

## 📁 Directory Structure

```
02_SHAP_LIME_Integration/
├── README.md                     # This file
├── scripts/                      # Implementation scripts
│   ├── shap_analysis.py          # SHAP implementation
│   ├── lime_analysis.py          # LIME implementation
│   └── comparative_analysis.py   # Compare SHAP vs LIME
├── shap_analysis/                # SHAP outputs
│   ├── xgboost/                  # XGBoost SHAP results
│   ├── random_forest/            # Random Forest SHAP results
│   └── summary_plots/            # Combined visualizations
├── lime_analysis/                # LIME outputs
│   ├── xgboost/                  # XGBoost LIME results
│   ├── random_forest/            # Random Forest LIME results
│   └── sample_explanations/      # Individual sample explanations
├── comparative_analysis/         # SHAP vs LIME comparison
│   ├── feature_rankings.csv      # Feature importance comparison
│   └── agreement_analysis.md     # Where methods agree/disagree
├── results/                      # Final outputs
│   └── xai_report.pdf            # Publication-ready report
└── documentation/                # Analysis documents
    └── methodology.md            # Detailed methodology
```

---

## 🎯 Objectives

### SHAP Analysis
1. **Global Interpretability**
   - Feature importance across all predictions
   - Summary plots showing feature impact distribution
   - Identify most influential features for DoS detection

2. **Local Interpretability**
   - Individual prediction explanations
   - Waterfall plots for specific samples
   - Force plots for prediction breakdown

### LIME Analysis
1. **Local Explanations**
   - Model-agnostic interpretability
   - Sample-specific feature contributions
   - Visual explanation of individual predictions

2. **Comparative Analysis**
   - Compare SHAP and LIME feature rankings
   - Identify agreement and disagreement
   - Understand when each method is most useful

---

## 📋 Task Checklist

### SHAP Implementation
- [ ] Install SHAP library (`pip install shap`)
- [ ] Load trained XGBoost model
- [ ] Create SHAP TreeExplainer
- [ ] Calculate SHAP values for test set
- [ ] Generate Summary Plot
- [ ] Generate Feature Importance Plot
- [ ] Generate Waterfall Plots (top 3 samples)
- [ ] Generate Force Plots
- [ ] Repeat for Random Forest model
- [ ] Document insights

### LIME Implementation
- [ ] Install LIME library (`pip install lime`)
- [ ] Load trained models
- [ ] Create LIME TabularExplainer
- [ ] Generate explanations for DoS predictions
- [ ] Generate explanations for Normal predictions
- [ ] Create visual explanations
- [ ] Document feature contributions

### Comparative Analysis
- [ ] Extract SHAP feature rankings
- [ ] Extract LIME feature rankings
- [ ] Calculate agreement metrics
- [ ] Identify consistent important features
- [ ] Document discrepancies
- [ ] Write comparative report

### Documentation
- [ ] Methodology explanation
- [ ] Results interpretation
- [ ] CTI implications
- [ ] Research contributions

---

## 📊 10 Selected Features

The XAI analysis will focus on these 10 features:

| # | Feature | Description | Category |
|---|---------|-------------|----------|
| 1 | dur | Connection duration | Time-based |
| 2 | proto | Protocol type (encoded) | Protocol |
| 3 | sbytes | Source to dest bytes | Volume |
| 4 | dload | Dest bits per second | Rate |
| 5 | sload | Source bits per second | Rate |
| 6 | stcpb | Source TCP base seq # | Protocol |
| 7 | dtcpb | Dest TCP base seq # | Protocol |
| 8 | rate | Packets per second | Rate |
| 9 | dmean | Mean dest packet size | Statistics |
| 10 | tcprtt | TCP round-trip time | Time-based |

---

## 🔗 Previous XAI Work

We have existing XAI implementation that needs validation:
- **Location:** `../../05_XAI_integration/`
- **SHAP Results:** `../../05_XAI_integration/SHAP_analysis/`
- **LIME Results:** `../../05_XAI_integration/LIME_analysis/`

### Action: Review and Validate Existing Work
- [ ] Review existing SHAP implementation
- [ ] Verify SHAP results correctness
- [ ] Review existing LIME implementation
- [ ] Verify LIME results correctness
- [ ] Decide: Update existing or create new

---

## 📚 Reference Papers

### XAI in Network Intrusion Detection
1. **MDPI Applied Sciences 2025** - SHAP vs LIME on UNSW-NB15
2. **IEEE Access 2024** - E-XAI Framework for DDoS Detection
3. **IEEE Networking Letters 2022** - XAI with XGBoost for Network Analysis

### Key Concepts
- **SHAP:** Based on Shapley values from game theory
- **LIME:** Creates local linear surrogate models
- **Global vs Local:** Understanding overall vs individual predictions

---

## 📈 Expected Outputs

### SHAP Outputs
1. **Summary Plot** - Feature importance with distribution
2. **Bar Plot** - Mean absolute SHAP values
3. **Waterfall Plots** - Individual prediction breakdown
4. **Force Plots** - Interactive prediction explanation

### LIME Outputs
1. **Feature Contribution Bars** - Per-sample explanations
2. **Probability Plots** - Class probability breakdown
3. **HTML Reports** - Interactive explanations

### Comparative Outputs
1. **Feature Ranking Table** - Side-by-side comparison
2. **Agreement Metrics** - Correlation between methods
3. **Visual Comparison** - Combined plots

---

## 💡 CTI Implications

Understanding **WHY** a model classifies traffic as DoS attack is crucial for:
- **Incident Response:** Know which features triggered detection
- **Threat Analysis:** Understand attack characteristics
- **Defense Improvement:** Focus on most impactful features
- **Trust Building:** Explainable decisions for security teams

---

## 📅 Progress Tracking

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Directory setup | ✅ Complete | Dec 12, 2025 | |
| Review existing XAI | ⏳ Pending | | Check 05_XAI_integration/ |
| SHAP implementation | ⏳ Pending | | |
| LIME implementation | ⏳ Pending | | |
| Comparative analysis | ⏳ Pending | | |
| Documentation | ⏳ Pending | | |

---

**Status:** 🚀 Ready to Start
