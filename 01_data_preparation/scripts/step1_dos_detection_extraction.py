#!/usr/bin/env python3
"""
DoS Detection Research - Step 1: DoS Detection Dataset Creation
==============================================================

This script creates a balanced dataset for DoS attack detection by extracting
both DoS attacks and Normal traffic from the UNSW-NB15 dataset.

Purpose: Create a binary classification dataset (DoS vs Normal)
Author: DoS Detection Research Team
Date: August 31, 2025
Version: 1.0 (Fresh Start)

Input: UNSW_NB15_training-set.csv
Output: dos_detection_dataset.csv (DoS attacks + Normal traffic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_unsw_dataset():
    """Load the UNSW-NB15 training dataset."""
    print("=" * 80)
    print("LOADING UNSW-NB15 DATASET")
    print("=" * 80)
    
    input_file = r"d:\Edu\Final Project\projectCodeing\datasetsfinalproject\UNSW_NB15_training-set.csv"
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded: {input_file}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Total records: {df.shape[0]:,}")
        print(f"✓ Total features: {df.shape[1]}")
        
        # Display column information
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

def analyze_attack_distribution(df):
    """Analyze the distribution of attack categories in the dataset."""
    print("\n" + "=" * 80)
    print("ATTACK CATEGORY ANALYSIS")
    print("=" * 80)
    
    if 'attack_cat' not in df.columns:
        print("✗ 'attack_cat' column not found!")
        return None
    
    # Attack category distribution
    attack_dist = df['attack_cat'].value_counts()
    print("Attack Category Distribution:")
    print("-" * 40)
    
    total_records = len(df)
    for category, count in attack_dist.items():
        percentage = (count / total_records) * 100
        print(f"{category:<15}: {count:>7,} ({percentage:>6.2f}%)")
    
    # Label distribution
    if 'label' in df.columns:
        label_dist = df['label'].value_counts()
        print(f"\nLabel Distribution:")
        print("-" * 20)
        for label, count in label_dist.items():
            label_name = "Normal" if label == 0 else "Attack"
            percentage = (count / total_records) * 100
            print(f"{label_name:<10}: {count:>7,} ({percentage:>6.2f}%)")
    
    return attack_dist

def extract_dos_and_normal(df):
    """Extract DoS attacks and Normal traffic for binary classification."""
    print("\n" + "=" * 80)
    print("EXTRACTING DoS ATTACKS AND NORMAL TRAFFIC")
    print("=" * 80)
    
    # Extract DoS attacks
    dos_attacks = df[df['attack_cat'] == 'DoS'].copy()
    dos_count = len(dos_attacks)
    print(f"✓ DoS attacks found: {dos_count:,}")
    
    # Extract Normal traffic
    normal_traffic = df[df['attack_cat'] == 'Normal'].copy()
    normal_count = len(normal_traffic)
    print(f"✓ Normal traffic found: {normal_count:,}")
    
    # Determine balanced dataset size
    min_count = min(dos_count, normal_count)
    print(f"\nBalancing Strategy:")
    print(f"- DoS attacks available: {dos_count:,}")
    print(f"- Normal traffic available: {normal_count:,}")
    print(f"- Balanced size (each class): {min_count:,}")
    
    # Sample equal amounts from each class
    if dos_count > min_count:
        dos_attacks = dos_attacks.sample(n=min_count, random_state=42)
        print(f"✓ DoS attacks sampled down to: {len(dos_attacks):,}")
    
    if normal_count > min_count:
        normal_traffic = normal_traffic.sample(n=min_count, random_state=42)
        print(f"✓ Normal traffic sampled down to: {len(normal_traffic):,}")
    
    # Combine datasets
    balanced_dataset = pd.concat([dos_attacks, normal_traffic], ignore_index=True)
    
    # Shuffle the combined dataset
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Balanced dataset created:")
    print(f"  - Total records: {len(balanced_dataset):,}")
    print(f"  - DoS attacks: {(balanced_dataset['attack_cat'] == 'DoS').sum():,}")
    print(f"  - Normal traffic: {(balanced_dataset['attack_cat'] == 'Normal').sum():,}")
    print(f"  - Features: {balanced_dataset.shape[1]}")
    
    return balanced_dataset

def analyze_dataset_quality(df):
    """Analyze the quality of the extracted dataset."""
    print("\n" + "=" * 80)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 80)
    
    # Check for missing values
    print("1. Missing Values Analysis:")
    print("-" * 30)
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing == 0:
        print("✓ No missing values found")
    else:
        print(f"⚠ Missing values found: {total_missing:,} total")
        missing_cols = missing_values[missing_values > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({percentage:.2f}%)")
    
    # Check for duplicates
    print("\n2. Duplicate Records Analysis:")
    print("-" * 30)
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print("✓ No duplicate records found")
    else:
        percentage = (duplicates / len(df)) * 100
        print(f"⚠ Duplicate records: {duplicates:,} ({percentage:.2f}%)")
    
    # Data types analysis
    print("\n3. Data Types Analysis:")
    print("-" * 30)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Target balance verification
    print("\n4. Target Balance Verification:")
    print("-" * 30)
    target_dist = df['attack_cat'].value_counts()
    for category, count in target_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count:,} ({percentage:.2f}%)")
    
    label_dist = df['label'].value_counts()
    for label, count in label_dist.items():
        label_name = "Normal (0)" if label == 0 else "Attack (1)"
        percentage = (count / len(df)) * 100
        print(f"  {label_name}: {count:,} ({percentage:.2f}%)")

def create_feature_info_summary(df):
    """Create a summary of feature information."""
    print("\n" + "=" * 80)
    print("FEATURE INFORMATION SUMMARY")
    print("=" * 80)
    
    feature_info = []
    
    # Exclude ID and target columns for feature analysis
    feature_columns = [col for col in df.columns if col not in ['id', 'attack_cat', 'label']]
    
    print(f"Analyzing {len(feature_columns)} features...")
    print(f"{'Feature':<20} {'Type':<10} {'Unique':<8} {'Missing':<8} {'Range/Categories'}")
    print("-" * 80)
    
    for col in feature_columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        
        # Get range or categories
        if df[col].dtype in ['int64', 'float64']:
            min_val = df[col].min()
            max_val = df[col].max()
            range_info = f"[{min_val:.2f}, {max_val:.2f}]"
        else:
            # Show top categories for categorical
            top_categories = df[col].value_counts().head(3)
            range_info = ", ".join([f"{cat}({count})" for cat, count in top_categories.items()])
            if len(range_info) > 30:
                range_info = range_info[:27] + "..."
        
        print(f"{col:<20} {dtype:<10} {unique_count:<8} {missing_count:<8} {range_info}")
        
        feature_info.append({
            'feature': col,
            'data_type': dtype,
            'unique_values': unique_count,
            'missing_values': missing_count,
            'missing_percentage': (missing_count / len(df)) * 100
        })
    
    # Save feature info to CSV
    feature_info_df = pd.DataFrame(feature_info)
    feature_info_df.to_csv('../data/feature_info.csv', index=False)
    print(f"\n✓ Feature information saved to: feature_info.csv")
    
    return feature_info_df

def save_dataset(df):
    """Save the balanced DoS detection dataset."""
    print("\n" + "=" * 80)
    print("SAVING DoS DETECTION DATASET")
    print("=" * 80)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Save the balanced dataset
    output_file = '../data/dos_detection_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Dataset saved to: {output_file}")
    print(f"✓ Records saved: {len(df):,}")
    print(f"✓ Features saved: {df.shape[1]}")
    print(f"✓ File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

def create_analysis_report(df, feature_info_df):
    """Create a comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("CREATING ANALYSIS REPORT")
    print("=" * 80)
    
    # Create results directory
    os.makedirs('../results', exist_ok=True)
    
    report = f"""
DoS Detection Dataset Creation Report
====================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: UNSW-NB15 Training Set
Purpose: Binary DoS Detection (DoS vs Normal)

DATASET SUMMARY:
- Total Records: {len(df):,}
- DoS Attacks: {(df['attack_cat'] == 'DoS').sum():,}
- Normal Traffic: {(df['attack_cat'] == 'Normal').sum():,}
- Total Features: {df.shape[1]}
- Balance Ratio: 50/50 (perfectly balanced)

TARGET DISTRIBUTION:
"""
    
    # Add target distribution
    target_dist = df['attack_cat'].value_counts()
    for category, count in target_dist.items():
        percentage = (count / len(df)) * 100
        report += f"- {category}: {count:,} ({percentage:.2f}%)\n"
    
    report += f"\nLABEL DISTRIBUTION:\n"
    label_dist = df['label'].value_counts()
    for label, count in label_dist.items():
        label_name = "Normal (0)" if label == 0 else "Attack (1)"
        percentage = (count / len(df)) * 100
        report += f"- {label_name}: {count:,} ({percentage:.2f}%)\n"
    
    # Add feature summary
    numeric_features = feature_info_df[feature_info_df['data_type'].isin(['int64', 'float64'])]
    categorical_features = feature_info_df[~feature_info_df['data_type'].isin(['int64', 'float64'])]
    
    report += f"""
FEATURE SUMMARY:
- Total Features: {len(feature_info_df)}
- Numeric Features: {len(numeric_features)}
- Categorical Features: {len(categorical_features)}
- Features with Missing Values: {(feature_info_df['missing_values'] > 0).sum()}

DATA QUALITY:
- Missing Values: {df.isnull().sum().sum():,} total
- Duplicate Records: {df.duplicated().sum():,}
- Complete Records: {len(df) - df.isnull().any(axis=1).sum():,}

NEXT STEPS:
1. Step 2.1: Data Cleanup (remove missing values, handle duplicates)
2. Step 2.2: Categorical Encoding (encode non-numeric features)
3. Step 2.3: Correlation Analysis (remove highly correlated features)
4. Step 2.4: Variance Analysis (remove low-variance features)
5. Step 2.5: Statistical Testing (ANOVA F-tests, mutual information)
6. Step 2.6: Final Feature Selection

BALANCED DATASET BENEFITS:
- Equal representation of both classes
- Prevents model bias toward majority class
- Enables accurate performance metrics
- Suitable for binary classification
"""
    
    # Save report
    with open('../results/step1_dos_detection_extraction_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Analysis report saved to: step1_dos_detection_extraction_report.txt")

def main():
    """Main execution function for Step 1: DoS Detection Dataset Creation."""
    print("DoS Detection Research - Step 1: DoS Detection Dataset Creation")
    print("=" * 80)
    print("Creating a balanced dataset with DoS attacks and Normal traffic")
    print("for binary classification DoS detection research.")
    print("=" * 80)
    
    try:
        # Load UNSW-NB15 dataset
        df = load_unsw_dataset()
        if df is None:
            return
        
        # Analyze attack distribution
        attack_dist = analyze_attack_distribution(df)
        if attack_dist is None:
            return
        
        # Extract DoS attacks and Normal traffic
        balanced_dataset = extract_dos_and_normal(df)
        
        # Analyze dataset quality
        analyze_dataset_quality(balanced_dataset)
        
        # Create feature information summary
        feature_info_df = create_feature_info_summary(balanced_dataset)
        
        # Save the dataset
        save_dataset(balanced_dataset)
        
        # Create analysis report
        create_analysis_report(balanced_dataset, feature_info_df)
        
        print("\n" + "=" * 80)
        print("STEP 1 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✓ Balanced DoS detection dataset created")
        print("✓ Quality analysis completed")
        print("✓ Feature information documented")
        print("✓ Ready for Step 2: Feature Selection")
        print("=" * 80)
        
        return balanced_dataset
        
    except Exception as e:
        print(f"\nERROR in Step 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    dataset = main()
