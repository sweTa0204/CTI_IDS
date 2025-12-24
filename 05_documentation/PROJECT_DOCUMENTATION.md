# DoS Detection Research Project Documentation

## Project Overview
**Title**: From Detection to Defense: An XAI Powered DoS Prevention System with Implementable Mitigation Protocols
**Dataset**: UNSW-NB15 Network Intrusion Detection Dataset
**Research Focus**: DoS Attack Detection using Machine Learning with Explainable AI (XAI)
**Methodology**: ADASYN Resampling with SHAP Analysis

## Project Structure
```
DoS_Detection_Research/
├── PROJECT_DOCUMENTATION.md          # Main project documentation
├── documentation/                    # Step-by-step documentation files
│   ├── step_01_dos_extraction.md     # Step 1 documentation
│   ├── step_02_feature_selection.md  # Step 2 documentation
│   ├── step_03_adasyn_resampling.md  # Step 3 documentation
│   ├── step_04_model_training.md     # Step 4 documentation
│   └── step_05_xai_analysis.md       # Step 5 documentation
├── data/                             # Processed datasets
├── models/                           # Trained machine learning models
├── results/                          # Analysis results and visualizations
└── scripts/                          # Python scripts for each step
```

## Dataset Information
**Training Set**: UNSW_NB15_training-set.csv
- Total Records: 82,332
- Total Features: 45
- DoS Attack Records: 4,089
- Normal Records: 78,243

**Testing Set**: UNSW_NB15_testing-set.csv
- Total Records: 175,343
- Features: 45 (same as training)

## Research Methodology

### Step 1: Basic DoS Extraction and Analysis
**Objective**: Extract all DoS attack records from the training dataset and perform initial analysis
**Input**: UNSW_NB15_training-set.csv
**Output**: 
- dos_attacks.csv (4,089 records with 45 features)
- Basic statistical analysis report
**Key Tasks**:
- Load and examine dataset structure
- Filter DoS attack records (attack_cat == 'DoS')
- Analyze DoS attack distribution
- Validate data quality
**Status**: PENDING

### Step 2: Smart Feature Selection
**Objective**: Select 15-20 most important features to avoid feature explosion
**Input**: dos_attacks.csv (45 features)
**Output**: 
- selected_features.csv (15-20 features)
- feature_selection_report.txt
**Key Tasks**:
- Calculate feature correlation matrix
- Perform statistical importance analysis
- Remove redundant and low-variance features
- Validate selected features
**Status**: PENDING

### Step 3: ADASYN Resampling
**Objective**: Generate synthetic DoS samples to balance the dataset
**Input**: selected_features.csv
**Output**: 
- balanced_dataset.csv (approximately 12,000 DoS samples)
- resampling_report.txt
**Key Tasks**:
- Prepare data for ADASYN algorithm
- Generate synthetic minority class samples
- Validate synthetic data quality
- Create balanced training dataset
**Status**: PENDING

### Step 4: Model Training and Evaluation
**Objective**: Train multiple ML models and evaluate performance
**Input**: balanced_dataset.csv
**Output**: 
- Trained models (Random Forest, CNN, KNN, LightGBM)
- performance_comparison.csv
- model_evaluation_report.txt
**Key Tasks**:
- Train four different ML algorithms
- Evaluate on original UNSW-NB15 test set
- Compare model performance metrics
- Select best performing model
**Status**: PENDING

### Step 5: XAI Analysis with SHAP
**Objective**: Generate explainable AI insights for DoS detection
**Input**: Best performing trained model
**Output**: 
- SHAP visualizations
- feature_importance_analysis.txt
- research_conclusions.txt
**Key Tasks**:
- Generate SHAP values for model predictions
- Create feature importance visualizations
- Analyze model decision patterns
- Document research findings
**Status**: PENDING

## Progress Tracking

### Completed Steps
- Project setup and documentation creation

### Current Step
**Step 1: Basic DoS Extraction and Analysis**

### Next Steps
1. Step 1: Basic DoS Extraction and Analysis
2. Step 2: Smart Feature Selection
3. Step 3: ADASYN Resampling
4. Step 4: Model Training and Evaluation
5. Step 5: XAI Analysis with SHAP

## Key Design Decisions

### Feature Handling Strategy
- Maintain original 45 features from UNSW-NB15 dataset
- Avoid feature expansion through one-hot encoding
- Use statistical methods for feature selection
- Target 15-20 most important features

### Resampling Strategy
- Use ADASYN (Adaptive Synthetic Sampling) for minority class oversampling
- Target balanced dataset with approximately 12,000 DoS samples
- Maintain original test set for unbiased evaluation

### Model Selection
- Random Forest: Ensemble method for robust performance
- Convolutional Neural Network: Deep learning approach
- K-Nearest Neighbors: Instance-based learning
- LightGBM: Gradient boosting for efficiency

### Evaluation Strategy
- Use original UNSW-NB15 test set for final evaluation
- Focus on precision, recall, F1-score for DoS detection
- Implement SHAP for model interpretability

## Research Questions
1. How effective is ADASYN resampling for DoS attack detection?
2. Which machine learning algorithm performs best for DoS detection?
3. What are the most important features for DoS attack identification?
4. How can XAI techniques improve understanding of DoS detection models?

## Expected Outcomes
- Balanced dataset for improved DoS detection training
- Comparative analysis of four ML algorithms
- Feature importance insights through SHAP analysis
- Actionable recommendations for DoS prevention systems

## Notes and Observations
- Previous attempt resulted in feature explosion from 45 to 188 features
- Current approach focuses on feature selection rather than expansion
- Emphasis on simplicity and interpretability
- Documentation updated after each completed step

---

**Last Updated**: August 31, 2025
**Project Status**: Step 1 - Ready to Begin
**Next Action**: Execute DoS extraction and analysis script
