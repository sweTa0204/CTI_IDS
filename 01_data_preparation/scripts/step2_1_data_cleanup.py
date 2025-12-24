#!/usr/bin/env python3
"""
Step 2.1: Data Cleanup
======================
Removes administrative clutter and prepares clean data structure for feature engineering.

Goals:
1. Remove 'id' column (just row numbers, not useful for DoS detection)
2. Remove 'attack_cat' column (keep 'label' as target - same information)
3. Organize clean separation of input features vs target variable

Input:  dos_detection_dataset.csv (8,178 records × 45 columns)
Output: cleaned_dataset.csv (8,178 records × 43 columns)
        42 network features + 1 target variable
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def main():
    print("=" * 60)
    print("🧹 STEP 2.1: DATA CLEANUP")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the balanced dataset
    input_file = '../data/dos_detection_dataset.csv'
    print(f"📂 Loading dataset: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print()
    print("📊 BEFORE CLEANUP:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Column names: {list(df.columns)}")
    print()
    
    # Check current data balance
    print("🎯 Target Variable Analysis:")
    if 'label' in df.columns:
        print("   'label' distribution:")
        print(df['label'].value_counts().to_string())
    
    if 'attack_cat' in df.columns:
        print("   'attack_cat' distribution:")
        print(df['attack_cat'].value_counts().to_string())
    print()
    
    # Step 1: Remove 'id' column
    print("🗑️  STEP 1: Removing 'id' column...")
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        print("   ✅ 'id' column removed successfully")
    else:
        print("   ⚠️  'id' column not found (already removed?)")
    
    # Step 2: Remove 'attack_cat' column (keep 'label')
    print("🗑️  STEP 2: Removing 'attack_cat' column...")
    if 'attack_cat' in df.columns:
        df = df.drop('attack_cat', axis=1)
        print("   ✅ 'attack_cat' column removed successfully")
        print("   ✅ Keeping 'label' as target variable")
    else:
        print("   ⚠️  'attack_cat' column not found (already removed?)")
    
    print()
    print("📊 AFTER CLEANUP:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Column names: {list(df.columns)}")
    print()
    
    # Verify data integrity
    print("🔍 DATA INTEGRITY CHECK:")
    print(f"   Records: {len(df)} (should be 8,178)")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Check target variable balance
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"   Target balance: {label_counts.to_dict()}")
        balance_ratio = min(label_counts) / max(label_counts)
        print(f"   Balance ratio: {balance_ratio:.3f} (should be 1.000 for perfect balance)")
    
    # Separate features and target
    print()
    print("📋 FEATURE ORGANIZATION:")
    
    if 'label' in df.columns:
        features = df.drop('label', axis=1)
        target = df['label']
        
        print(f"   Input features: {len(features.columns)} columns")
        print(f"   Target variable: 1 column ('label')")
        print(f"   Feature names: {list(features.columns)}")
        
        # Check data types
        print()
        print("📊 DATA TYPES ANALYSIS:")
        text_features = features.select_dtypes(include=['object']).columns
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        print(f"   Text features: {len(text_features)} columns")
        if len(text_features) > 0:
            print(f"   Text columns: {list(text_features)}")
        
        print(f"   Numeric features: {len(numeric_features)} columns")
        print(f"   Numeric columns: {list(numeric_features)[:10]}{'...' if len(numeric_features) > 10 else ''}")
    
    # Save cleaned dataset
    output_file = '../data/cleaned_dataset.csv'
    print()
    print(f"💾 SAVING CLEANED DATASET:")
    print(f"   Output file: {output_file}")
    
    try:
        df.to_csv(output_file, index=False)
        print(f"   ✅ Cleaned dataset saved successfully")
        
        # Verify saved file
        saved_df = pd.read_csv(output_file)
        print(f"   ✅ Verification: {saved_df.shape} (matches expected)")
        
    except Exception as e:
        print(f"   ❌ Error saving dataset: {e}")
        return
    
    # Summary
    print()
    print("=" * 60)
    print("📈 STEP 2.1 SUMMARY")
    print("=" * 60)
    print(f"✅ Administrative cleanup completed successfully")
    print(f"✅ Dataset structure organized for ML pipeline")
    print()
    print("📊 TRANSFORMATION SUMMARY:")
    print(f"   Input:  dos_detection_dataset.csv (8,178 × 45)")
    print(f"   Output: cleaned_dataset.csv (8,178 × {len(df.columns)})")
    print(f"   Removed: 2 administrative columns ('id', 'attack_cat')")
    print(f"   Result: {len(df.columns)-1} input features + 1 target variable")
    print()
    print("🎯 READY FOR STEP 2.2: Categorical Encoding")
    print(f"   Text features to encode: {len(text_features)} columns")
    print(f"   Text columns: {list(text_features)}")
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
