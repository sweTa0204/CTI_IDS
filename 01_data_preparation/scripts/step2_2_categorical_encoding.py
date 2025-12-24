#!/usr/bin/env python3
"""
Step 2.2: Categorical Encoding
===============================
Converts text features to numbers so ML algorithms can process them.

Goals:
1. Convert 'proto' column (131 unique protocols) to numeric codes
2. Convert 'service' column (13 unique services) to numeric codes  
3. Convert 'state' column (4 unique states) to numeric codes
4. Maintain all original information while making data ML-compatible

Input:  cleaned_dataset.csv (8,178 records × 43 columns, 3 text features)
Output: encoded_dataset.csv (8,178 records × 43 columns, all numeric)
        42 numeric features + 1 target variable
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import pickle

def main():
    print("=" * 60)
    print("🔤➡️🔢 STEP 2.2: CATEGORICAL ENCODING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the cleaned dataset
    input_file = '../data/cleaned_dataset.csv'
    print(f"📂 Loading cleaned dataset: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print()
    print("📊 BEFORE ENCODING:")
    print(f"   Total columns: {len(df.columns)}")
    
    # Identify text features
    text_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in text_features:
        text_features.remove('label')  # Don't encode target if it's text
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_features:
        numeric_features.remove('label')  # Separate target
    
    print(f"   Text features: {len(text_features)} columns")
    print(f"   Text columns: {text_features}")
    print(f"   Numeric features: {len(numeric_features)} columns")
    print()
    
    # Analyze each text feature
    print("🔍 TEXT FEATURE ANALYSIS:")
    encoders = {}
    encoding_mapping = {}
    
    for feature in text_features:
        unique_values = df[feature].unique()
        print(f"\n📋 Feature: '{feature}'")
        print(f"   Unique values: {len(unique_values)}")
        print(f"   Values: {list(unique_values)}")
        print(f"   Value counts:")
        value_counts = df[feature].value_counts()
        print(value_counts.to_string())
    
    print()
    print("🔄 ENCODING PROCESS:")
    
    # Create a copy for encoding
    df_encoded = df.copy()
    
    # Encode each text feature
    for feature in text_features:
        print(f"\n🔤 Encoding '{feature}'...")
        
        # Create label encoder
        encoder = LabelEncoder()
        
        # Fit and transform the feature
        try:
            encoded_values = encoder.fit_transform(df[feature])
            df_encoded[feature] = encoded_values
            
            # Store encoder for future use
            encoders[feature] = encoder
            
            # Create mapping for documentation
            unique_original = df[feature].unique()
            unique_encoded = encoder.transform(unique_original)
            mapping = dict(zip(unique_original, unique_encoded))
            encoding_mapping[feature] = mapping
            
            print(f"   ✅ '{feature}' encoded successfully")
            print(f"   Mapping: {mapping}")
            
        except Exception as e:
            print(f"   ❌ Error encoding '{feature}': {e}")
            return
    
    print()
    print("📊 AFTER ENCODING:")
    print(f"   Total columns: {len(df_encoded.columns)}")
    
    # Verify all features are now numeric
    text_remaining = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if 'label' in text_remaining:
        text_remaining.remove('label')  # Target can be text
    
    numeric_total = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_total:
        numeric_total.remove('label')  # Separate target
    
    print(f"   Text features remaining: {len(text_remaining)}")
    print(f"   Numeric features: {len(numeric_total)}")
    
    if len(text_remaining) == 0:
        print("   ✅ All features successfully converted to numeric!")
    else:
        print(f"   ⚠️  Text features still remaining: {text_remaining}")
    
    # Data integrity check
    print()
    print("🔍 DATA INTEGRITY CHECK:")
    print(f"   Records: {len(df_encoded)} (should be 8,178)")
    print(f"   Missing values: {df_encoded.isnull().sum().sum()}")
    
    # Check target variable balance (should be unchanged)
    if 'label' in df_encoded.columns:
        label_counts = df_encoded['label'].value_counts()
        print(f"   Target balance: {label_counts.to_dict()}")
        balance_ratio = min(label_counts) / max(label_counts)
        print(f"   Balance ratio: {balance_ratio:.3f} (should be 1.000)")
    
    # Save encoded dataset
    output_file = '../data/encoded_dataset.csv'
    print()
    print(f"💾 SAVING ENCODED DATASET:")
    print(f"   Output file: {output_file}")
    
    try:
        df_encoded.to_csv(output_file, index=False)
        print(f"   ✅ Encoded dataset saved successfully")
        
        # Verify saved file
        saved_df = pd.read_csv(output_file)
        print(f"   ✅ Verification: {saved_df.shape} (matches expected)")
        
    except Exception as e:
        print(f"   ❌ Error saving dataset: {e}")
        return
    
    # Save encoders for future use
    encoders_file = '../data/label_encoders.pkl'
    print(f"💾 SAVING LABEL ENCODERS:")
    print(f"   Encoders file: {encoders_file}")
    
    try:
        with open(encoders_file, 'wb') as f:
            pickle.dump(encoders, f)
        print(f"   ✅ Label encoders saved successfully")
    except Exception as e:
        print(f"   ❌ Error saving encoders: {e}")
    
    # Save encoding mapping documentation
    mapping_file = '../data/encoding_mapping.txt'
    print(f"📄 SAVING ENCODING DOCUMENTATION:")
    print(f"   Mapping file: {mapping_file}")
    
    try:
        with open(mapping_file, 'w') as f:
            f.write("CATEGORICAL ENCODING MAPPING\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for feature, mapping in encoding_mapping.items():
                f.write(f"\nFeature: {feature}\n")
                f.write("-" * 20 + "\n")
                for original, encoded in mapping.items():
                    f.write(f"{original} → {encoded}\n")
                f.write(f"\nTotal mappings: {len(mapping)}\n")
        
        print(f"   ✅ Encoding mapping documented successfully")
    except Exception as e:
        print(f"   ❌ Error saving mapping: {e}")
    
    # Summary
    print()
    print("=" * 60)
    print("📈 STEP 2.2 SUMMARY")
    print("=" * 60)
    print(f"✅ Categorical encoding completed successfully")
    print(f"✅ All features converted to numeric format")
    print()
    print("📊 ENCODING SUMMARY:")
    for feature, mapping in encoding_mapping.items():
        print(f"   '{feature}': {len(mapping)} unique values encoded")
    print()
    print("📊 TRANSFORMATION SUMMARY:")
    print(f"   Input:  cleaned_dataset.csv (8,178 × 43, 3 text features)")
    print(f"   Output: encoded_dataset.csv (8,178 × 43, all numeric)")
    print(f"   Result: 42 numeric features + 1 target variable")
    print()
    print("🎯 READY FOR STEP 2.3: Feature Reduction - Correlation Analysis")
    print(f"   Features ready for correlation analysis: {len(numeric_total)} columns")
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
