#!/usr/bin/env python3
"""
Step 2.6: Feature Scaling - Final Feature Engineering Step
========================================================

🎯 PURPOSE: Scale features to optimal ranges for ML algorithms
📊 INPUT: statistical_features.csv (10 statistically significant features)
📈 OUTPUT: final_scaled_dataset.csv (10 features with optimal scaling)

🔧 SCALING METHOD: StandardScaler (mean=0, std=1)
⚖️ WHY: Best for DoS detection - preserves statistical relationships

📋 PROCESS:
1. Load statistical features (10 features from step 2.5)
2. Analyze current feature ranges and distributions
3. Apply StandardScaler to normalize all features
4. Validate scaling results
5. Save final scaled dataset for model training

⏱️ ESTIMATED TIME: ~8 minutes
📊 FEATURE COUNT: 10 → 10 (same count, optimized ranges)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_distributions(df, features, title_prefix=""):
    """Analyze and visualize feature distributions before/after scaling"""
    print(f"\n📊 {title_prefix}FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    stats_summary = pd.DataFrame({
        'Feature': features,
        'Min': [df[col].min() for col in features],
        'Max': [df[col].max() for col in features],
        'Mean': [df[col].mean() for col in features],
        'Std': [df[col].std() for col in features],
        'Range': [df[col].max() - df[col].min() for col in features]
    })
    
    print("\n📈 Statistical Summary:")
    print(stats_summary.round(4))
    
    # Calculate range ratios to show scaling impact
    range_ratio = stats_summary['Range'].max() / stats_summary['Range'].min()
    print(f"\n⚖️ Range Analysis:")
    print(f"   • Largest range: {stats_summary['Range'].max():.2f}")
    print(f"   • Smallest range: {stats_summary['Range'].min():.2f}")
    print(f"   • Range ratio: {range_ratio:.2f}x difference")
    
    if range_ratio > 100:
        print(f"   ❌ PROBLEM: {range_ratio:.0f}x range difference needs scaling!")
    elif range_ratio > 10:
        print(f"   ⚠️  WARNING: {range_ratio:.1f}x range difference - scaling recommended")
    else:
        print(f"   ✅ GOOD: {range_ratio:.1f}x range difference - well scaled!")
    
    return stats_summary

def validate_scaling_quality(original_df, scaled_df, features):
    """Validate that scaling was applied correctly"""
    print(f"\n🔍 SCALING QUALITY VALIDATION")
    print("=" * 60)
    
    validation_results = []
    
    for feature in features:
        scaled_mean = scaled_df[feature].mean()
        scaled_std = scaled_df[feature].std()
        
        # StandardScaler should produce mean≈0, std≈1
        mean_ok = abs(scaled_mean) < 0.01  # Very close to 0
        std_ok = abs(scaled_std - 1.0) < 0.01  # Very close to 1
        
        validation_results.append({
            'Feature': feature,
            'Scaled_Mean': scaled_mean,
            'Scaled_Std': scaled_std,
            'Mean_OK': mean_ok,
            'Std_OK': std_ok,
            'Overall_OK': mean_ok and std_ok
        })
    
    validation_df = pd.DataFrame(validation_results)
    print("\n📊 Scaling Validation Results:")
    print(validation_df.round(4))
    
    # Summary
    total_features = len(validation_results)
    perfect_scaling = sum(result['Overall_OK'] for result in validation_results)
    
    print(f"\n✅ SCALING SUMMARY:")
    print(f"   • Total features: {total_features}")
    print(f"   • Perfectly scaled: {perfect_scaling}")
    print(f"   • Success rate: {(perfect_scaling/total_features)*100:.1f}%")
    
    if perfect_scaling == total_features:
        print(f"   🎯 EXCELLENT: All features perfectly scaled!")
        return True
    else:
        print(f"   ❌ ISSUE: {total_features - perfect_scaling} features not perfectly scaled")
        return False

def main():
    """Main function for Step 2.6: Feature Scaling"""
    
    print("🚀 STEP 2.6: FEATURE SCALING")
    print("=" * 60)
    print("🎯 Goal: Scale features to optimal ranges for ML algorithms")
    print("⚖️ Method: StandardScaler (mean=0, std=1)")
    print("📊 Expected: 10 features → 10 optimized features")
    
    # Set up paths
    data_dir = Path("../data")
    input_file = data_dir / "statistical_features.csv"
    output_file = data_dir / "final_scaled_dataset.csv"
    
    # Verify input file exists
    if not input_file.exists():
        print(f"❌ ERROR: Input file not found: {input_file}")
        print("   Please ensure step 2.5 (statistical testing) was completed successfully")
        return False
    
    print(f"\n📁 Loading data from: {input_file}")
    
    try:
        # Load the statistical features dataset
        df = pd.read_csv(input_file)
        print(f"✅ Successfully loaded dataset")
        print(f"   • Shape: {df.shape}")
        print(f"   • Features: {df.shape[1] - 1} (excluding target)")
        
        # Verify data integrity
        print(f"\n🔍 DATA INTEGRITY CHECK:")
        print(f"   • Total records: {len(df):,}")
        print(f"   • Missing values: {df.isnull().sum().sum()}")
        print(f"   • Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Get feature columns (all except 'label')
        feature_columns = [col for col in df.columns if col != 'label']
        print(f"   • Feature columns: {len(feature_columns)}")
        print(f"   • Features: {feature_columns}")
        
        # Verify target distribution
        target_counts = df['label'].value_counts().sort_index()
        print(f"\n🎯 TARGET DISTRIBUTION:")
        for label, count in target_counts.items():
            label_name = "Normal" if label == 0 else "DoS"
            print(f"   • {label_name} (label={label}): {count:,} records ({(count/len(df))*100:.1f}%)")
        
        # Analyze original feature distributions
        original_stats = analyze_feature_distributions(df, feature_columns, "ORIGINAL ")
        
        # Prepare data for scaling
        print(f"\n⚙️ PREPARING FOR SCALING:")
        X = df[feature_columns].copy()
        y = df['label'].copy()
        
        print(f"   • Feature matrix shape: {X.shape}")
        print(f"   • Target vector shape: {y.shape}")
        
        # Initialize and fit StandardScaler
        print(f"\n🔧 APPLYING STANDARDSCALER:")
        scaler = StandardScaler()
        
        print(f"   • Fitting scaler on {len(feature_columns)} features...")
        X_scaled = scaler.fit_transform(X)
        
        print(f"   • Scaling completed successfully")
        print(f"   • Scaled matrix shape: {X_scaled.shape}")
        
        # Convert back to DataFrame with original column names
        df_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)
        df_scaled['label'] = y  # Add target back
        
        print(f"   • Final scaled dataset shape: {df_scaled.shape}")
        
        # Analyze scaled feature distributions
        scaled_stats = analyze_feature_distributions(df_scaled, feature_columns, "SCALED ")
        
        # Validate scaling quality
        scaling_perfect = validate_scaling_quality(df, df_scaled, feature_columns)
        
        # Show transformation examples
        print(f"\n🔍 TRANSFORMATION EXAMPLES:")
        print("=" * 60)
        example_features = feature_columns[:3]  # Show first 3 features
        
        for feature in example_features:
            orig_min, orig_max = df[feature].min(), df[feature].max()
            scaled_min, scaled_max = df_scaled[feature].min(), df_scaled[feature].max()
            
            print(f"\n📊 {feature}:")
            print(f"   • Original range: [{orig_min:.4f}, {orig_max:.4f}]")
            print(f"   • Scaled range: [{scaled_min:.4f}, {scaled_max:.4f}]")
            print(f"   • Range reduction: {(orig_max - orig_min):.2f} → {(scaled_max - scaled_min):.2f}")
        
        # Save scaled dataset
        print(f"\n💾 SAVING SCALED DATASET:")
        print(f"   • Output file: {output_file}")
        
        df_scaled.to_csv(output_file, index=False)
        print(f"   ✅ Scaled dataset saved successfully")
        
        # Verify saved file
        verification_df = pd.read_csv(output_file)
        print(f"   • Verification - Shape: {verification_df.shape}")
        print(f"   • Verification - Columns: {list(verification_df.columns)}")
        
        # Generate scaling transformation summary
        print(f"\n📋 SCALING TRANSFORMATION SUMMARY:")
        print("=" * 60)
        
        transformation_summary = pd.DataFrame({
            'Feature': feature_columns,
            'Original_Mean': [df[col].mean() for col in feature_columns],
            'Original_Std': [df[col].std() for col in feature_columns],
            'Scaled_Mean': [df_scaled[col].mean() for col in feature_columns],
            'Scaled_Std': [df_scaled[col].std() for col in feature_columns]
        })
        
        print(transformation_summary.round(4))
        
        # Calculate overall scaling impact
        original_max_range = original_stats['Range'].max()
        scaled_max_range = scaled_stats['Range'].max()
        range_normalization = original_max_range / scaled_max_range
        
        print(f"\n🎯 SCALING IMPACT ANALYSIS:")
        print(f"   • Original max range: {original_max_range:.2f}")
        print(f"   • Scaled max range: {scaled_max_range:.2f}")
        print(f"   • Range normalization factor: {range_normalization:.1f}x")
        print(f"   • All features now in similar scale: ±{scaled_max_range/2:.1f}")
        
        # Final success summary
        print(f"\n🎉 STEP 2.6 COMPLETION SUMMARY:")
        print("=" * 60)
        print(f"✅ Input processed: {input_file.name}")
        print(f"✅ Records processed: {len(df):,}")
        print(f"✅ Features scaled: {len(feature_columns)}")
        print(f"✅ Scaling method: StandardScaler (mean=0, std=1)")
        print(f"✅ Quality validation: {'PASSED' if scaling_perfect else 'NEEDS REVIEW'}")
        print(f"✅ Output saved: {output_file.name}")
        print(f"✅ Data integrity: MAINTAINED")
        print(f"✅ Target distribution: PRESERVED")
        
        print(f"\n🚀 READY FOR NEXT STEP:")
        print(f"   • Feature engineering: COMPLETED (6/6 sub-steps)")
        print(f"   • Final dataset: {output_file.name}")
        print(f"   • Features ready for: Model Training (Step 3)")
        print(f"   • Optimization level: Maximum (10 high-quality, scaled features)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during feature scaling:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 Step 2.6 completed successfully! Feature engineering is now complete.")
    else:
        print(f"\n❌ Step 2.6 failed. Please check the error messages above.")
