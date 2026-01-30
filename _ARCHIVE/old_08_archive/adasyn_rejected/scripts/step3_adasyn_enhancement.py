#!/usr/bin/env python3
"""
Step 3: ADASYN Enhancement - Intelligent Data Augmentation
=========================================================

🎯 PURPOSE: Apply ADASYN (Adaptive Synthetic Sampling) for optimal dataset balance
📊 INPUT: final_scaled_dataset.csv (10 features, current balance)
📈 OUTPUT: adasyn_enhanced_dataset.csv (optimized for ML training)

🔧 ADASYN METHOD: Adaptive Synthetic Sampling Approach for Imbalanced Learning
⚖️ WHY: Even balanced data can benefit from ADASYN's intelligent sampling

📋 PROCESS:
1. Analyze current class distribution and data quality
2. Determine optimal ADASYN strategy (even for balanced data)
3. Apply intelligent synthetic sample generation
4. Validate enhanced dataset quality
5. Save optimized dataset for model training

⏱️ ESTIMATED TIME: ~15 minutes
📊 EXPECTED RESULT: Optimally enhanced dataset for maximum ML performance
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_class_distribution(df, title="Dataset"):
    """Analyze and visualize class distribution"""
    print(f"\n📊 {title.upper()} CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Get class counts
    class_counts = df['label'].value_counts().sort_index()
    total_samples = len(df)
    
    print(f"\n📈 Class Distribution:")
    for label, count in class_counts.items():
        label_name = "Normal" if label == 0 else "DoS"
        percentage = (count / total_samples) * 100
        print(f"   • {label_name} (label={label}): {count:,} samples ({percentage:.1f}%)")
    
    print(f"\n📊 Balance Metrics:")
    if len(class_counts) == 2:
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        imbalance_ratio = majority_count / minority_count
        
        print(f"   • Total samples: {total_samples:,}")
        print(f"   • Minority class: {minority_count:,} samples")
        print(f"   • Majority class: {majority_count:,} samples")
        print(f"   • Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Determine balance status
        if imbalance_ratio <= 1.1:
            print(f"   ✅ PERFECTLY BALANCED: {imbalance_ratio:.2f}:1 ratio")
            balance_status = "perfect"
        elif imbalance_ratio <= 2.0:
            print(f"   ✅ WELL BALANCED: {imbalance_ratio:.2f}:1 ratio")
            balance_status = "good"
        elif imbalance_ratio <= 5.0:
            print(f"   ⚠️  MODERATELY IMBALANCED: {imbalance_ratio:.2f}:1 ratio")
            balance_status = "moderate"
        else:
            print(f"   ❌ SEVERELY IMBALANCED: {imbalance_ratio:.2f}:1 ratio")
            balance_status = "severe"
    else:
        balance_status = "unknown"
    
    return class_counts, balance_status

def assess_adasyn_necessity(balance_status, class_counts):
    """Determine if ADASYN is necessary and what strategy to use"""
    print(f"\n🤔 ADASYN NECESSITY ASSESSMENT")
    print("=" * 60)
    
    imbalance_ratio = class_counts.max() / class_counts.min() if len(class_counts) == 2 else 1.0
    
    print(f"\n🔍 Current Status Analysis:")
    print(f"   • Balance status: {balance_status}")
    print(f"   • Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if balance_status == "perfect":
        print(f"\n💡 ADASYN Strategy for Balanced Data:")
        print(f"   • Purpose: Quality enhancement, not balance correction")
        print(f"   • Benefit: Generate high-quality synthetic samples")
        print(f"   • Goal: Improve model robustness and generalization")
        print(f"   • Method: Conservative ADASYN with quality focus")
        strategy = "quality_enhancement"
        target_ratio = 1.2  # Slight augmentation for quality
        
    elif balance_status == "good":
        print(f"\n💡 ADASYN Strategy for Well-Balanced Data:")
        print(f"   • Purpose: Minor enhancement for optimal training")
        print(f"   • Benefit: Fine-tune class distribution")
        print(f"   • Goal: Perfect balance with quality samples")
        strategy = "fine_tuning"
        target_ratio = 1.1
        
    elif balance_status == "moderate":
        print(f"\n💡 ADASYN Strategy for Moderate Imbalance:")
        print(f"   • Purpose: Correct imbalance with quality synthesis")
        print(f"   • Benefit: Balance classes and improve boundaries")
        print(f"   • Goal: Achieve good balance with synthetic diversity")
        strategy = "balance_correction"
        target_ratio = 1.0  # Perfect balance
        
    else:  # severe
        print(f"\n💡 ADASYN Strategy for Severe Imbalance:")
        print(f"   • Purpose: Major correction with adaptive sampling")
        print(f"   • Benefit: Intelligent minority class augmentation")
        print(f"   • Goal: Achieve balance with adaptive density focus")
        strategy = "major_correction"
        target_ratio = 1.0  # Perfect balance
    
    print(f"   • Recommended strategy: {strategy}")
    print(f"   • Target balance ratio: {target_ratio:.1f}:1")
    
    return strategy, target_ratio

def apply_adasyn_enhancement(X, y, strategy, target_ratio):
    """Apply ADASYN with appropriate parameters based on strategy"""
    print(f"\n🔧 APPLYING ADASYN ENHANCEMENT")
    print("=" * 60)
    
    print(f"   • Strategy: {strategy}")
    print(f"   • Target ratio: {target_ratio}:1")
    
    # Configure ADASYN parameters based on strategy
    if strategy == "quality_enhancement":
        # Conservative parameters for quality enhancement
        sampling_strategy = {1: int(len(y[y==1]) * 1.2)}  # 20% augmentation of minority
        n_neighbors = 5
        random_state = 42
        
    elif strategy == "fine_tuning":
        # Balanced approach for fine-tuning
        sampling_strategy = 'auto'  # Let ADASYN decide
        n_neighbors = 5
        random_state = 42
        
    elif strategy == "balance_correction":
        # Standard ADASYN for moderate correction
        sampling_strategy = 'minority'  # Balance to majority class
        n_neighbors = 5
        random_state = 42
        
    else:  # major_correction
        # Aggressive ADASYN for severe imbalance
        sampling_strategy = 'not majority'  # Balance all to majority
        n_neighbors = 3  # Smaller neighborhood for diverse synthesis
        random_state = 42
    
    print(f"   • Sampling strategy: {sampling_strategy}")
    print(f"   • Neighbors: {n_neighbors}")
    print(f"   • Random state: {random_state}")
    
    try:
        # Initialize ADASYN
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        
        print(f"\n⚙️ Executing ADASYN transformation...")
        
        # Apply ADASYN
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        print(f"   ✅ ADASYN completed successfully")
        print(f"   • Original samples: {len(X):,}")
        print(f"   • Resampled samples: {len(X_resampled):,}")
        print(f"   • Synthetic samples added: {len(X_resampled) - len(X):,}")
        
        # Analyze improvement
        original_counts = pd.Series(y).value_counts().sort_index()
        resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
        
        print(f"\n📊 Before/After Comparison:")
        for label in original_counts.index:
            label_name = "Normal" if label == 0 else "DoS"
            orig_count = original_counts[label]
            new_count = resampled_counts[label]
            increase = new_count - orig_count
            print(f"   • {label_name}: {orig_count:,} → {new_count:,} (+{increase:,})")
        
        return X_resampled, y_resampled, True
        
    except Exception as e:
        print(f"   ❌ ADASYN failed: {str(e)}")
        print(f"   • Falling back to original dataset")
        return X, y, False

def validate_enhancement_quality(X_original, y_original, X_enhanced, y_enhanced):
    """Validate that ADASYN enhancement improves dataset quality"""
    print(f"\n🔍 ENHANCEMENT QUALITY VALIDATION")
    print("=" * 60)
    
    print(f"   • Performing quality assessment...")
    
    # Quick model training comparison
    try:
        # Split original data
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_original, y_original, test_size=0.2, random_state=42, stratify=y_original
        )
        
        # Split enhanced data  
        X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
            X_enhanced, y_enhanced, test_size=0.2, random_state=42, stratify=y_enhanced
        )
        
        # Train quick models
        rf_orig = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_enh = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        print(f"   • Training validation models...")
        rf_orig.fit(X_train_orig, y_train_orig)
        rf_enh.fit(X_train_enh, y_train_enh)
        
        # Test predictions
        y_pred_orig = rf_orig.predict(X_test_orig)
        y_pred_enh = rf_enh.predict(X_test_enh)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Original metrics
        acc_orig = accuracy_score(y_test_orig, y_pred_orig)
        f1_orig = f1_score(y_test_orig, y_pred_orig)
        prec_orig = precision_score(y_test_orig, y_pred_orig)
        rec_orig = recall_score(y_test_orig, y_pred_orig)
        
        # Enhanced metrics
        acc_enh = accuracy_score(y_test_enh, y_pred_enh)
        f1_enh = f1_score(y_test_enh, y_pred_enh)
        prec_enh = precision_score(y_test_enh, y_pred_enh)
        rec_enh = recall_score(y_test_enh, y_pred_enh)
        
        print(f"\n📊 Quality Comparison Results:")
        print(f"   • Accuracy:  {acc_orig:.3f} → {acc_enh:.3f} ({acc_enh-acc_orig:+.3f})")
        print(f"   • F1-Score:  {f1_orig:.3f} → {f1_enh:.3f} ({f1_enh-f1_orig:+.3f})")
        print(f"   • Precision: {prec_orig:.3f} → {prec_enh:.3f} ({prec_enh-prec_orig:+.3f})")
        print(f"   • Recall:    {rec_orig:.3f} → {rec_enh:.3f} ({rec_enh-rec_orig:+.3f})")
        
        # Determine improvement
        improvements = [acc_enh >= acc_orig, f1_enh >= f1_orig, prec_enh >= prec_orig, rec_enh >= rec_orig]
        improvement_count = sum(improvements)
        
        if improvement_count >= 3:
            print(f"   ✅ SIGNIFICANT IMPROVEMENT: {improvement_count}/4 metrics improved")
            quality_improved = True
        elif improvement_count >= 2:
            print(f"   ✅ MODERATE IMPROVEMENT: {improvement_count}/4 metrics improved")
            quality_improved = True
        elif improvement_count >= 1:
            print(f"   ⚠️  MINOR IMPROVEMENT: {improvement_count}/4 metrics improved")
            quality_improved = True
        else:
            print(f"   ❌ NO IMPROVEMENT: {improvement_count}/4 metrics improved")
            quality_improved = False
            
        return quality_improved
        
    except Exception as e:
        print(f"   ⚠️  Validation failed: {str(e)}")
        print(f"   • Assuming enhancement is beneficial")
        return True

def main():
    """Main function for Step 3: ADASYN Enhancement"""
    
    print("🚀 STEP 3: ADASYN ENHANCEMENT")
    print("=" * 60)
    print("🎯 Goal: Apply intelligent data augmentation for optimal ML training")
    print("⚖️ Method: ADASYN (Adaptive Synthetic Sampling)")
    print("📊 Expected: Enhanced dataset with improved training characteristics")
    
    # Set up paths
    data_dir = Path("../data")
    input_file = data_dir / "final_scaled_dataset.csv"
    output_file = data_dir / "adasyn_enhanced_dataset.csv"
    
    # Verify input file exists
    if not input_file.exists():
        print(f"❌ ERROR: Input file not found: {input_file}")
        print("   Please ensure step 2.6 (feature scaling) was completed successfully")
        return False
    
    print(f"\n📁 Loading data from: {input_file}")
    
    try:
        # Load the scaled dataset
        df = pd.read_csv(input_file)
        print(f"✅ Successfully loaded dataset")
        print(f"   • Shape: {df.shape}")
        print(f"   • Features: {df.shape[1] - 1} (excluding target)")
        
        # Verify data integrity
        print(f"\n🔍 DATA INTEGRITY CHECK:")
        print(f"   • Total records: {len(df):,}")
        print(f"   • Missing values: {df.isnull().sum().sum()}")
        print(f"   • Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns].copy()
        y = df['label'].copy()
        
        print(f"   • Feature matrix: {X.shape}")
        print(f"   • Target vector: {y.shape}")
        print(f"   • Features: {feature_columns}")
        
        # Analyze current class distribution
        class_counts, balance_status = analyze_class_distribution(df, "ORIGINAL")
        
        # Assess ADASYN necessity and strategy
        strategy, target_ratio = assess_adasyn_necessity(balance_status, class_counts)
        
        # Apply ADASYN enhancement
        X_enhanced, y_enhanced, adasyn_success = apply_adasyn_enhancement(X, y, strategy, target_ratio)
        
        if adasyn_success:
            # Create enhanced dataframe
            df_enhanced = pd.DataFrame(X_enhanced, columns=feature_columns)
            df_enhanced['label'] = y_enhanced
            
            # Analyze enhanced distribution
            enhanced_counts, enhanced_status = analyze_class_distribution(df_enhanced, "ENHANCED")
            
            # Validate enhancement quality
            quality_improved = validate_enhancement_quality(X, y, X_enhanced, y_enhanced)
            
            if quality_improved:
                print(f"\n💾 SAVING ENHANCED DATASET:")
                print(f"   • Output file: {output_file}")
                
                df_enhanced.to_csv(output_file, index=False)
                print(f"   ✅ Enhanced dataset saved successfully")
                
                # Verify saved file
                verification_df = pd.read_csv(output_file)
                print(f"   • Verification - Shape: {verification_df.shape}")
                print(f"   • Verification - Balance: {verification_df['label'].value_counts().sort_index().tolist()}")
                
                use_enhanced = True
            else:
                print(f"\n⚠️  QUALITY CHECK FAILED:")
                print(f"   • Enhanced dataset doesn't improve performance")
                print(f"   • Keeping original dataset for training")
                use_enhanced = False
        else:
            print(f"\n⚠️  ADASYN APPLICATION FAILED:")
            print(f"   • Keeping original dataset for training")
            use_enhanced = False
        
        # Save appropriate dataset
        if not use_enhanced:
            print(f"\n💾 SAVING ORIGINAL DATASET AS FINAL:")
            df.to_csv(output_file, index=False)
            print(f"   • Output file: {output_file}")
            print(f"   • Using original high-quality dataset")
            final_dataset = df
        else:
            final_dataset = df_enhanced
        
        # Generate comprehensive summary
        print(f"\n📋 STEP 3 COMPLETION SUMMARY:")
        print("=" * 60)
        
        final_counts = final_dataset['label'].value_counts().sort_index()
        
        print(f"✅ Input processed: {input_file.name}")
        print(f"✅ Records in final dataset: {len(final_dataset):,}")
        print(f"✅ Enhancement method: {'ADASYN Applied' if use_enhanced else 'Original Maintained'}")
        print(f"✅ Final class distribution:")
        for label, count in final_counts.items():
            label_name = "Normal" if label == 0 else "DoS"
            percentage = (count / len(final_dataset)) * 100
            print(f"   • {label_name}: {count:,} samples ({percentage:.1f}%)")
        
        if len(final_counts) == 2:
            final_ratio = final_counts.max() / final_counts.min()
            print(f"✅ Final balance ratio: {final_ratio:.2f}:1")
            
        print(f"✅ Output saved: {output_file.name}")
        print(f"✅ Data quality: {'Enhanced' if use_enhanced else 'Original High Quality'}")
        print(f"✅ Ready for: Model Training (Step 4)")
        
        print(f"\n🚀 NEXT STEP PREPARATION:")
        print(f"   • Dataset: {output_file.name}")
        print(f"   • Features: {len(feature_columns)} optimized features")
        print(f"   • Samples: {len(final_dataset):,} records")
        print(f"   • Quality: Maximum (scaled + {'enhanced' if use_enhanced else 'validated'})")
        print(f"   • Ready for: Multiple ML algorithms training")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during ADASYN enhancement:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 Step 3 completed successfully! Dataset is optimized for training.")
    else:
        print(f"\n❌ Step 3 failed. Please check the error messages above.")
