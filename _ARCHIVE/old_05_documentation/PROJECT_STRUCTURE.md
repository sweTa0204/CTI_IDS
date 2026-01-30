# DoS Detection Project - Organized Structure

## Project Overview
This project focuses on detecting Denial of Service (DoS) attacks using machine learning techniques. The project has been organized into phases for better workflow management and documentation.

## Folder Structure

### 01_data_preparation/
Contains all data preprocessing scripts and datasets:
- **data/**: All preprocessed datasets (cleaned, encoded, decorrelated, scaled)
- **scripts/**: Data preparation and extraction scripts (including DoS detection extraction)
- **Main Dataset**: `final_scaled_dataset.csv` (ready for model training)

### 02_feature_engineering/
Feature analysis and engineering phase:
- **scripts/**: Correlation analysis, variance analysis, statistical testing scripts
- **results/**: Analysis reports and visualizations

### 03_model_training/
Model development and training phase:
- **scripts/**: (Empty - Ready for ML model implementation)
- **STATUS**: Not started yet - awaiting model training scripts

### 04_validation_results/
Validation and final results:
- Comprehensive reports, visualizations, and analysis summaries
- XAI compatibility analysis
- Project verification and comparison scripts

### 05_documentation/
All project documentation:
- Research documentation
- Step-by-step guides
- Project overviews

### deprecated_adasyn/
**DO NOT USE** - Contains ADASYN-related files that were excluded from the project:
- **data/**: ADASYN-enhanced dataset (incompatible)
- **scripts/**: ADASYN generation and validation scripts
- **results/**: ADASYN analysis results

## Current Status
- ✅ Data Preparation: Complete
- ✅ Feature Engineering: Complete  
- ❌ Model Training: **NOT STARTED** (Next Step)
- ⏳ Final Validation: Pending

## Working Dataset
The main dataset for model training is: `working_dataset.csv` (symbolic link to the final scaled dataset)

## Next Steps
1. Implement machine learning models using the final scaled dataset
2. Train and evaluate multiple algorithms
3. Select best performing model
4. Document final results and conclusions
