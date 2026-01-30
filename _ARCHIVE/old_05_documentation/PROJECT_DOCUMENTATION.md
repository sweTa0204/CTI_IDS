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
**Original UNSW-NB15 Training Set**: UNSW_NB15_training-set.csv
- Total Records: 82,332
- Total Features: 45
- Attack Categories: 10 (Normal + 9 attack types)
- DoS Attack Records: 4,089 (4.97%)
- Normal Traffic Records: 37,000 (44.94%)
- Other Attack Records: 41,243 (remaining 8 attack types)

**Our Balanced DoS Detection Dataset**: dos_detection_dataset.csv
- Total Records: 8,178
- DoS Attacks: 4,089 (50%) - All available DoS attacks
- Normal Traffic: 4,089 (50%) - Randomly sampled from 37,000 available
- Features: 45 (same as original)
- Purpose: Binary classification (DoS vs Normal)

**Testing Set**: UNSW_NB15_testing-set.csv
- Total Records: 175,343
- Features: 45 (same as training)

## Research Methodology

### Step 1: DoS Detection Dataset Creation
**Objective**: Create a balanced binary classification dataset for DoS attack detection
**Input**: UNSW_NB15_training-set.csv (82,332 records, 10 attack categories)
**Output**: 
- dos_detection_dataset.csv (8,178 records: 4,089 DoS + 4,089 Normal)
- feature_info.csv (metadata for all 42 input features)
- step1_dos_detection_extraction_report.txt (comprehensive analysis)
**Key Tasks**:
- Extract all available DoS attacks (4,089 records)
- Randomly sample Normal traffic (4,089 from 37,000 available)
- Create perfectly balanced 50/50 dataset
- Validate data quality and sampling diversity
- Document sampling strategy and rationale
**Key Insights**:
- Solved class imbalance problem (avoided 90% Normal, 10% DoS bias)
- Achieved good protocol diversity (76% TCP, 21% UDP, 3% ARP)
- Achieved good service diversity (74% generic, 11% HTTP, 8% DNS, 5% FTP, 1% SMTP)
- Random sampling worked well without complex stratification
**Status**: COMPLETED
**Completion Date**: August 31, 2025

### Step 2: Feature Engineering (6 Sub-steps)
**Objective**: Transform 42 raw features into 12-15 high-quality features for DoS detection
**Input**: dos_detection_dataset.csv (8,178 records, 42 input features + target)
**Output**: 
- final_features_dataset.csv (12-15 selected features)
- feature_engineering_report.txt (comprehensive analysis)
**Sub-Steps**:
- **2.1 Data Cleanup**: Remove unnecessary columns (id), choose target variable
- **2.2 Categorical Encoding**: Convert text features (proto, service, state) to numeric
- **2.3 Correlation Analysis**: Remove highly correlated redundant features  
- **2.4 Variance Analysis**: Remove low-variance uninformative features
- **2.5 Statistical Testing**: Test DoS vs Normal discrimination power (ANOVA F-tests, mutual information)
- **2.6 Final Selection**: Combine all analyses to select 12-15 best features
**Status**: PENDING

### Step 3: ADASYN Resampling
**Objective**: Generate synthetic samples to enhance model training (if needed)
**Input**: final_features_dataset.csv (12-15 features, balanced 8,178 records)
**Output**: 
- enhanced_dataset.csv (potentially ~12,000 samples with synthetic data)
- adasyn_analysis_report.txt
**Key Tasks**:
- Evaluate if current balance (50/50) is sufficient
- Apply ADASYN if additional synthetic samples would improve performance
- Validate synthetic data quality and realism
- Create final training dataset
**Status**: PENDING

### Step 4: Model Training and Evaluation
**Objective**: Train multiple ML models and evaluate DoS detection performance
**Input**: enhanced_dataset.csv (final training data)
**Output**: 
- Trained models (Random Forest, CNN, KNN, LightGBM)
- performance_comparison.csv
- model_evaluation_report.txt
**Key Tasks**:
- Train four different ML algorithms on balanced DoS vs Normal data
- Evaluate on UNSW-NB15 test set (DoS vs Normal subset)
- Compare model performance metrics (precision, recall, F1-score)
- Focus on DoS detection accuracy and false positive rates
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

### 🚀 **MOTIVATION PROGRESS BAR**
```
XAI Powered DoS Prevention System Development
=============================================

[████████████████████████████████████████] Step 1: Dataset Creation (COMPLETED ✅)
[                                        ] Step 2: Feature Engineering (READY 🎯)
[                                        ] Step 3: ADASYN Enhancement (PENDING ⏳)
[                                        ] Step 4: Model Training (PENDING ⏳)
[                                        ] Step 5: XAI Analysis (PENDING ⏳)

Overall Progress: 20% Complete (1/5 major steps)
```

### Completed Steps
- ✅ **Project setup and documentation creation**
- ✅ **Step 1: DoS Detection Dataset Creation** (COMPLETED - August 31, 2025)
  - Balanced dataset: 8,178 records (4,089 DoS + 4,089 Normal)
  - Perfect 50/50 distribution for unbiased training
  - Documentation: `documentation/step_01_dos_extraction.md`
  - Cross-validation completed and verified

### Current Step
**🎯 Step 2: Feature Engineering (6 Sub-steps)**
- Ready to begin with balanced dataset (8,178 records: 50% DoS, 50% Normal)
- **Goal**: Transform 42 raw features → 12-15 optimized features
- **Time estimate**: ~52 minutes total
- **Documentation**: `documentation/step_02_feature_engineering.md`

### Step 2 Sub-Progress
```
FEATURE ENGINEERING ROADMAP:
[                                        ] 2.1: Data Cleanup (READY 🎯)
[                                        ] 2.2: Categorical Encoding (PENDING ⏳)
[                                        ] 2.3: Correlation Analysis (PENDING ⏳)
[                                        ] 2.4: Variance Analysis (PENDING ⏳)
[                                        ] 2.5: Statistical Testing (PENDING ⏳)
[                                        ] 2.6: Final Selection (PENDING ⏳)

Step 2 Progress: 0% Complete (0/6 sub-steps)
```

### Next Immediate Action
- 🚀 **Step 2.1: Data Cleanup** 
  - Remove unnecessary columns ('id')
  - Choose target variable (attack_cat vs label)
  - Prepare clean feature matrix
  - Expected time: ~5 minutes

### Upcoming Steps
1. **Step 2.2**: Categorical Encoding (convert text to numeric)
2. **Step 2.3**: Correlation Analysis (remove redundant features)
3. **Step 2.4**: Variance Analysis (remove uninformative features)
4. **Step 2.5**: Statistical Testing (test DoS vs Normal discrimination)
5. **Step 2.6**: Final Selection (choose 12-15 best features)
6. **Step 3**: ADASYN Resampling (enhance dataset if needed)
7. **Step 4**: Model Training and Evaluation
8. **Step 5**: XAI Analysis with SHAP

## Key Design Decisions

### Dataset Balancing Strategy
- **Problem Identified**: Original approach used only DoS attacks (4,089 records), no Normal traffic
- **Solution Implemented**: Balanced binary classification dataset
  - DoS attacks: 4,089 (all available)
  - Normal traffic: 4,089 (randomly sampled from 37,000 available)
  - Result: Perfect 50/50 balance eliminates class imbalance bias
- **Sampling Method**: Pure random sampling (assessed as adequate due to good diversity achieved)

### Feature Engineering Strategy
- Start with 42 input features (excluding id, attack_cat, label)
- Target 12-15 final features through systematic selection
- Avoid feature explosion through intelligent selection rather than expansion
- Focus on features that best discriminate DoS from Normal traffic

### Binary Classification Focus
- **Scope**: DoS detection vs Normal traffic (not multi-class attack detection)
- **Rationale**: Specialized binary models typically outperform multi-class for specific threats
- **Real-world application**: Clear security decision (block/allow traffic)

### Resampling Strategy
- Current dataset already balanced (50/50)
- ADASYN may be used for enhancement rather than balancing
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
1. **How effective is balanced dataset approach for DoS attack detection?**
   - Compare 50/50 balanced vs imbalanced (90% Normal, 10% DoS) performance
2. **Which machine learning algorithm performs best for binary DoS detection?**
   - Comparative analysis of Random Forest, CNN, KNN, and LightGBM
3. **What are the most important network features for distinguishing DoS from Normal traffic?**
   - Statistical and ML-based feature importance analysis
4. **How can XAI techniques improve understanding and trust in DoS detection models?**
   - SHAP analysis for model interpretability and decision explanation
5. **What is the optimal feature set size for DoS detection?**
   - Balance between model performance and complexity (targeting 12-15 features)

## Expected Outcomes
- **Balanced DoS detection dataset** that eliminates class imbalance bias
- **Optimized feature set** (12-15 features) for efficient and effective DoS detection
- **Comparative analysis** of four ML algorithms on binary DoS classification
- **Feature importance insights** through statistical testing and SHAP analysis
- **Actionable recommendations** for real-world DoS prevention system implementation
- **Explainable AI model** that provides interpretable DoS detection decisions

## Notes and Observations
- **Corrected Approach**: Initially extracted only DoS attacks; corrected to balanced DoS + Normal dataset
- **Class Imbalance Solution**: 50/50 balance prevents model bias toward majority class
- **Sampling Assessment**: Random sampling achieved good diversity (protocols, services)
- **Feature Strategy**: Focus on selection rather than expansion to avoid curse of dimensionality
- **Documentation**: Comprehensive documentation after each step for reproducibility

---

**Last Updated**: August 31, 2025
**Project Status**: Step 1 COMPLETED - Ready for Step 2 Feature Engineering
**Next Action**: Step 2.1 Data Cleanup (remove id column, choose target variable, prepare clean features)
