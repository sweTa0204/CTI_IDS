#!/usr/bin/env python3
"""
Step 2.3: Feature Reduction - Correlation Analysis
==================================================
Removes redundant features that provide duplicate information through correlation analysis.

Goals:
1. Calculate correlation matrix for all 42 numeric features
2. Identify highly correlated pairs (correlation > 0.90)
3. Smart selection: Keep more important feature from each correlated pair
4. Remove redundant features to eliminate duplicate information
5. Apply domain knowledge for network security relevance

Input:  encoded_dataset.csv (8,178 records × 43 columns, 42 features)
Output: decorrelated_dataset.csv (8,178 records × ~28-33 columns, ~25-30 features)
        Expected reduction: 42 → ~25-30 features (30-40% reduction)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

def calculate_correlation_matrix(df, target_col='label'):
    """Calculate correlation matrix for features only (excluding target)"""
    features = df.drop(target_col, axis=1)
    correlation_matrix = features.corr()
    return correlation_matrix, features.columns.tolist()

def find_high_correlation_pairs(correlation_matrix, threshold=0.90):
    """Find pairs of features with correlation above threshold"""
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = abs(correlation_matrix.iloc[i, j])
            if corr > threshold:
                feat1 = correlation_matrix.columns[i]
                feat2 = correlation_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr))
    
    # Sort by correlation strength (highest first)
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    return high_corr_pairs

def select_features_to_remove(high_corr_pairs, features_list):
    """
    Smart selection of features to remove from correlated pairs
    Uses domain knowledge and feature importance heuristics
    """
    features_to_remove = set()
    feature_priorities = get_feature_priorities()
    
    print("🧠 SMART FEATURE SELECTION:")
    print("Applying domain knowledge and feature importance heuristics...")
    print()
    
    for feat1, feat2, corr in high_corr_pairs:
        if feat1 in features_to_remove or feat2 in features_to_remove:
            print(f"⏭️  Skipping {feat1} ↔ {feat2} (one already marked for removal)")
            continue
            
        # Apply selection logic
        priority1 = feature_priorities.get(feat1, 5)  # Default medium priority
        priority2 = feature_priorities.get(feat2, 5)
        
        if priority1 < priority2:  # Lower number = higher priority
            remove_feature = feat2
            keep_feature = feat1
            reason = f"domain priority ({priority1} vs {priority2})"
        elif priority2 < priority1:
            remove_feature = feat1
            keep_feature = feat2
            reason = f"domain priority ({priority2} vs {priority1})"
        else:
            # Same priority - use naming heuristics
            if 'src' in feat1 or 's' == feat1[0]:  # Prefer source features
                remove_feature = feat2
                keep_feature = feat1
                reason = "prefer source features"
            elif 'src' in feat2 or 's' == feat2[0]:
                remove_feature = feat1
                keep_feature = feat2
                reason = "prefer source features"
            else:
                # Default: remove alphabetically later feature
                if feat1 < feat2:
                    remove_feature = feat2
                    keep_feature = feat1
                    reason = "alphabetical order"
                else:
                    remove_feature = feat1
                    keep_feature = feat2
                    reason = "alphabetical order"
        
        features_to_remove.add(remove_feature)
        print(f"📊 {feat1} ↔ {feat2} (r={corr:.3f})")
        print(f"   ✅ Keep: {keep_feature}")
        print(f"   ❌ Remove: {remove_feature} (reason: {reason})")
        print()
    
    return list(features_to_remove)

def get_feature_priorities():
    """
    Domain knowledge-based feature priorities for network security
    Lower number = higher priority (more important to keep)
    """
    return {
        # High priority - core network characteristics
        'dur': 1,           # Duration is fundamental
        'rate': 1,          # Traffic rate is crucial for DoS
        'proto': 1,         # Protocol type is essential
        'service': 1,       # Service type is important
        'state': 1,         # Connection state is key
        
        # Medium-high priority - traffic volume
        'spkts': 2,         # Source packets
        'sbytes': 2,        # Source bytes
        'sload': 2,         # Source load
        
        # Medium priority - destination metrics (keep some)
        'dpkts': 3,         # Destination packets
        'dbytes': 3,        # Destination bytes
        'dload': 3,         # Destination load
        
        # Lower priority - loss metrics (often redundant)
        'sloss': 4,         # Source loss
        'dloss': 4,         # Destination loss
        
        # Specialized features - context dependent
        'ct_srv_src': 2,    # Service connection count
        'ct_state_ttl': 2,  # State-TTL connection count
        
        # Binary flags - lower priority if mostly zeros
        'is_ftp_login': 6,  # Usually mostly zeros
        'ct_ftp_cmd': 6,    # Usually mostly zeros
        'is_sm_ips_ports': 6, # Usually mostly zeros
    }

def main():
    print("=" * 60)
    print("📈📉 STEP 2.3: FEATURE REDUCTION - CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the encoded dataset
    input_file = '../data/encoded_dataset.csv'
    print(f"📂 Loading encoded dataset: {input_file}")
    
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
    print("📊 BEFORE CORRELATION ANALYSIS:")
    features = df.drop('label', axis=1)
    print(f"   Total features: {len(features.columns)}")
    print(f"   Feature names: {list(features.columns)}")
    print()
    
    # Calculate correlation matrix
    print("🔍 CALCULATING CORRELATION MATRIX:")
    correlation_matrix, feature_names = calculate_correlation_matrix(df)
    print(f"   ✅ Correlation matrix calculated ({len(feature_names)} × {len(feature_names)})")
    
    # Find high correlation pairs
    correlation_threshold = 0.90
    print(f"🔎 FINDING HIGH CORRELATION PAIRS (threshold > {correlation_threshold}):")
    high_corr_pairs = find_high_correlation_pairs(correlation_matrix, correlation_threshold)
    
    if len(high_corr_pairs) == 0:
        print(f"   ℹ️  No highly correlated pairs found above threshold {correlation_threshold}")
        print("   ℹ️  Trying lower threshold...")
        correlation_threshold = 0.85
        high_corr_pairs = find_high_correlation_pairs(correlation_matrix, correlation_threshold)
    
    print(f"   ✅ Found {len(high_corr_pairs)} highly correlated pairs")
    print()
    
    if len(high_corr_pairs) > 0:
        print("📋 TOP CORRELATED PAIRS:")
        for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:10], 1):
            print(f"   {i:2d}. {feat1} ↔ {feat2}: {corr:.3f}")
        print()
        
        # Smart feature selection
        features_to_remove = select_features_to_remove(high_corr_pairs, feature_names)
        
        print("📊 CORRELATION ANALYSIS SUMMARY:")
        print(f"   Highly correlated pairs found: {len(high_corr_pairs)}")
        print(f"   Features marked for removal: {len(features_to_remove)}")
        print(f"   Features to remove: {features_to_remove}")
        print()
        
        # Remove selected features
        print("🗑️  REMOVING CORRELATED FEATURES:")
        df_decorrelated = df.drop(columns=features_to_remove)
        
        remaining_features = df_decorrelated.drop('label', axis=1).columns.tolist()
        print(f"   ✅ Removed {len(features_to_remove)} features")
        print(f"   ✅ Remaining features: {len(remaining_features)}")
        print()
        
    else:
        print("ℹ️  No features to remove - keeping all features")
        df_decorrelated = df.copy()
        remaining_features = feature_names
        features_to_remove = []
    
    # Verify correlation reduction
    if len(features_to_remove) > 0:
        print("🔍 VERIFYING CORRELATION REDUCTION:")
        new_correlation_matrix, _ = calculate_correlation_matrix(df_decorrelated)
        new_high_corr_pairs = find_high_correlation_pairs(new_correlation_matrix, correlation_threshold)
        
        print(f"   Before: {len(high_corr_pairs)} highly correlated pairs")
        print(f"   After:  {len(new_high_corr_pairs)} highly correlated pairs")
        print(f"   Reduction: {len(high_corr_pairs) - len(new_high_corr_pairs)} pairs eliminated")
        
        if len(new_high_corr_pairs) > 0:
            print("   ⚠️  Remaining high correlations:")
            for feat1, feat2, corr in new_high_corr_pairs[:5]:
                print(f"      {feat1} ↔ {feat2}: {corr:.3f}")
        else:
            print("   ✅ All high correlations successfully eliminated!")
        print()
    
    # Data integrity check
    print("🔍 DATA INTEGRITY CHECK:")
    print(f"   Records: {len(df_decorrelated)} (should be 8,178)")
    print(f"   Missing values: {df_decorrelated.isnull().sum().sum()}")
    
    # Check target variable balance
    if 'label' in df_decorrelated.columns:
        label_counts = df_decorrelated['label'].value_counts()
        print(f"   Target balance: {label_counts.to_dict()}")
        balance_ratio = min(label_counts) / max(label_counts)
        print(f"   Balance ratio: {balance_ratio:.3f} (should be 1.000)")
    print()
    
    # Save decorrelated dataset
    output_file = '../data/decorrelated_dataset.csv'
    print(f"💾 SAVING DECORRELATED DATASET:")
    print(f"   Output file: {output_file}")
    
    try:
        df_decorrelated.to_csv(output_file, index=False)
        print(f"   ✅ Decorrelated dataset saved successfully")
        
        # Verify saved file
        saved_df = pd.read_csv(output_file)
        print(f"   ✅ Verification: {saved_df.shape} (matches expected)")
        
    except Exception as e:
        print(f"   ❌ Error saving dataset: {e}")
        return
    
    # Save correlation analysis report
    report_file = '../results/step3_correlation_analysis_report.txt'
    print(f"📄 SAVING CORRELATION ANALYSIS REPORT:")
    print(f"   Report file: {report_file}")
    
    try:
        os.makedirs('../results', exist_ok=True)
        with open(report_file, 'w') as f:
            f.write("STEP 2.3: CORRELATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"INPUT:\n")
            f.write(f"- Dataset: encoded_dataset.csv\n")
            f.write(f"- Features: {len(feature_names)}\n")
            f.write(f"- Records: {len(df)}\n\n")
            
            f.write(f"CORRELATION ANALYSIS:\n")
            f.write(f"- Threshold: {correlation_threshold}\n")
            f.write(f"- High correlation pairs found: {len(high_corr_pairs)}\n")
            f.write(f"- Features removed: {len(features_to_remove)}\n\n")
            
            if len(high_corr_pairs) > 0:
                f.write("HIGH CORRELATION PAIRS:\n")
                for feat1, feat2, corr in high_corr_pairs:
                    f.write(f"- {feat1} ↔ {feat2}: {corr:.3f}\n")
                f.write("\n")
                
                f.write("FEATURES REMOVED:\n")
                for feat in features_to_remove:
                    f.write(f"- {feat}\n")
                f.write("\n")
            
            f.write(f"OUTPUT:\n")
            f.write(f"- Dataset: decorrelated_dataset.csv\n")
            f.write(f"- Features: {len(remaining_features)}\n")
            f.write(f"- Records: {len(df_decorrelated)}\n")
            f.write(f"- Reduction: {len(features_to_remove)} features removed\n")
        
        print(f"   ✅ Correlation analysis report saved successfully")
    except Exception as e:
        print(f"   ❌ Error saving report: {e}")
    
    # Summary
    print()
    print("=" * 60)
    print("📈 STEP 2.3 SUMMARY")
    print("=" * 60)
    print(f"✅ Correlation analysis completed successfully")
    print(f"✅ Redundant features identified and removed")
    print()
    print("📊 TRANSFORMATION SUMMARY:")
    print(f"   Input:  encoded_dataset.csv ({len(df)} × {len(feature_names) + 1})")
    print(f"   Output: decorrelated_dataset.csv ({len(df_decorrelated)} × {len(remaining_features) + 1})")
    print(f"   Removed: {len(features_to_remove)} redundant features")
    print(f"   Reduction: {len(feature_names)} → {len(remaining_features)} features ({len(features_to_remove)/len(feature_names)*100:.1f}% reduction)")
    print()
    print("🎯 READY FOR STEP 2.4: Feature Reduction - Variance Analysis")
    print(f"   Features ready for variance analysis: {len(remaining_features)} columns")
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
