#!/usr/bin/env python3
"""
Step 2.4: Feature Reduction - Variance Analysis
Removes features with low variance (uninformative features)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_variance(df, feature_columns):
    """Analyze variance of features and identify low-variance ones"""
    variance_info = []
    
    for feature in feature_columns:
        variance = df[feature].var()
        unique_count = df[feature].nunique()
        total_count = len(df[feature])
        unique_ratio = unique_count / total_count
        
        # Calculate value distribution
        value_counts = df[feature].value_counts()
        most_common_ratio = value_counts.iloc[0] / total_count if len(value_counts) > 0 else 0
        
        variance_info.append({
            'feature': feature,
            'variance': variance,
            'unique_count': unique_count,
            'unique_ratio': unique_ratio,
            'most_common_ratio': most_common_ratio,
            'min_value': df[feature].min(),
            'max_value': df[feature].max(),
            'mean_value': df[feature].mean(),
            'std_value': df[feature].std()
        })
    
    return pd.DataFrame(variance_info).sort_values('variance')

def identify_low_variance_features(variance_df, quasi_constant_threshold=0.95):
    """Identify features to remove based on variance analysis"""
    
    # Features with zero or near-zero variance
    zero_variance = variance_df[variance_df['variance'] < 1e-10]['feature'].tolist()
    
    # Quasi-constant features (95%+ same value)
    quasi_constant = variance_df[variance_df['most_common_ratio'] >= quasi_constant_threshold]['feature'].tolist()
    
    # Very low unique ratio (less than 1% unique values)
    low_unique = variance_df[variance_df['unique_ratio'] < 0.01]['feature'].tolist()
    
    # Combine all low-variance features
    all_low_variance = list(set(zero_variance + quasi_constant + low_unique))
    
    return {
        'zero_variance': zero_variance,
        'quasi_constant': quasi_constant,
        'low_unique': low_unique,
        'all_low_variance': all_low_variance
    }

def create_variance_visualization(variance_df, output_dir='../results'):
    """Create visualizations for variance analysis"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Variance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Variance distribution
    axes[0, 0].hist(variance_df['variance'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Distribution of Feature Variances')
    axes[0, 0].set_yscale('log')
    
    # 2. Unique ratio distribution
    axes[0, 1].hist(variance_df['unique_ratio'], bins=30, alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('Unique Value Ratio')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Distribution of Unique Value Ratios')
    
    # 3. Most common value ratio
    axes[1, 0].hist(variance_df['most_common_ratio'], bins=30, alpha=0.7, color='salmon')
    axes[1, 0].set_xlabel('Most Common Value Ratio')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Distribution of Most Common Value Ratios')
    axes[1, 0].axvline(x=0.95, color='red', linestyle='--', label='95% threshold')
    axes[1, 0].legend()
    
    # 4. Variance vs Unique Ratio scatter
    scatter = axes[1, 1].scatter(variance_df['unique_ratio'], variance_df['variance'], 
                                alpha=0.6, c=variance_df['most_common_ratio'], 
                                cmap='viridis', s=50)
    axes[1, 1].set_xlabel('Unique Value Ratio')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title('Variance vs Unique Ratio')
    axes[1, 1].set_yscale('log')
    plt.colorbar(scatter, ax=axes[1, 1], label='Most Common Ratio')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Variance analysis visualization saved to: {output_dir}/variance_analysis.png")

def main():
    print('=' * 60)
    print('📊 STEP 2.4: VARIANCE ANALYSIS')
    print('=' * 60)
    print('Removing features with low variance (uninformative features)...')
    print()

    # Load the corrected decorrelated dataset
    print('📂 Loading corrected decorrelated dataset...')
    df = pd.read_csv('../data/decorrelated_dataset_corrected.csv')
    
    print(f'Dataset shape: {df.shape}')
    print(f'Features: {df.shape[1] - 1} (excluding label)')
    print()

    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']

    print('🔍 Analyzing feature variances...')
    variance_df = analyze_feature_variance(df, feature_columns)
    
    # Display variance summary
    print('\n📊 VARIANCE SUMMARY (sorted by variance):')
    print('=' * 80)
    print(f"{'Feature':<20} {'Variance':<15} {'Unique%':<10} {'MostCommon%':<12} {'Range':<15}")
    print('=' * 80)
    
    for _, row in variance_df.iterrows():
        feature = row['feature']
        variance = row['variance']
        unique_pct = row['unique_ratio'] * 100
        common_pct = row['most_common_ratio'] * 100
        value_range = f"{row['min_value']:.2f}-{row['max_value']:.2f}"
        
        print(f"{feature:<20} {variance:<15.6f} {unique_pct:<10.1f} {common_pct:<12.1f} {value_range:<15}")
    
    # Identify low-variance features
    print('\n🔍 Identifying low-variance features...')
    low_variance_info = identify_low_variance_features(variance_df, quasi_constant_threshold=0.95)
    
    # Display findings
    print(f"\n📋 LOW-VARIANCE FEATURE ANALYSIS:")
    print(f"   Zero variance features: {len(low_variance_info['zero_variance'])}")
    if low_variance_info['zero_variance']:
        for feat in low_variance_info['zero_variance']:
            print(f"      - {feat}")
    
    print(f"   Quasi-constant features (≥95% same value): {len(low_variance_info['quasi_constant'])}")
    if low_variance_info['quasi_constant']:
        for feat in low_variance_info['quasi_constant']:
            common_ratio = variance_df[variance_df['feature'] == feat]['most_common_ratio'].iloc[0]
            print(f"      - {feat} ({common_ratio*100:.1f}% same value)")
    
    print(f"   Low unique features (<1% unique): {len(low_variance_info['low_unique'])}")
    if low_variance_info['low_unique']:
        for feat in low_variance_info['low_unique']:
            unique_ratio = variance_df[variance_df['feature'] == feat]['unique_ratio'].iloc[0]
            print(f"      - {feat} ({unique_ratio*100:.1f}% unique)")
    
    # Features to remove
    features_to_remove = low_variance_info['all_low_variance']
    print(f"\n🗑️ TOTAL FEATURES TO REMOVE: {len(features_to_remove)}")
    
    if features_to_remove:
        print("Features being removed:")
        for i, feature in enumerate(features_to_remove, 1):
            print(f"  {i}. {feature}")
    else:
        print("✅ No low-variance features found! All features have good variance.")
    
    # Create variance-cleaned dataset
    print(f'\n📊 Creating variance-cleaned dataset...')
    remaining_features = [col for col in feature_columns if col not in features_to_remove]
    df_variance_cleaned = df[remaining_features + ['label']].copy()
    
    print(f'Original features: {len(feature_columns)}')
    print(f'Features after variance filtering: {len(remaining_features)}')
    
    if len(features_to_remove) > 0:
        print(f'Reduction: {len(features_to_remove)} features ({len(features_to_remove)/len(feature_columns)*100:.1f}%)')
    else:
        print('Reduction: 0 features (0.0%) - All features passed variance threshold')
    
    # Verify no zero-variance features remain using sklearn
    print('\n🔍 Verifying variance filtering with sklearn VarianceThreshold...')
    variance_selector = VarianceThreshold(threshold=0.0)
    X_variance_filtered = variance_selector.fit_transform(df_variance_cleaned[remaining_features])
    
    features_removed_by_sklearn = len(remaining_features) - X_variance_filtered.shape[1]
    
    if features_removed_by_sklearn == 0:
        print('✅ SUCCESS: No additional zero-variance features detected by sklearn!')
    else:
        print(f'⚠️  WARNING: sklearn detected {features_removed_by_sklearn} additional zero-variance features')
    
    # Save variance-cleaned dataset
    output_path = '../data/variance_cleaned_dataset.csv'
    df_variance_cleaned.to_csv(output_path, index=False)
    print(f'\n💾 Variance-cleaned dataset saved to: {output_path}')
    
    # Create visualization
    print('\n📊 Creating variance analysis visualization...')
    create_variance_visualization(variance_df)
    
    # Verify data integrity
    print('\n🔍 Data integrity check...')
    print(f'Original dataset shape: {df.shape}')
    print(f'Variance-cleaned dataset shape: {df_variance_cleaned.shape}')
    print(f'Records preserved: {len(df_variance_cleaned) == len(df)}')
    
    # Check label distribution
    original_distribution = df['label'].value_counts().sort_index()
    new_distribution = df_variance_cleaned['label'].value_counts().sort_index()
    print(f'Label distribution preserved: {original_distribution.equals(new_distribution)}')
    
    # Feature progression summary
    print('\n📈 FEATURE PROGRESSION SUMMARY:')
    print(f'   Step 2.1 (Cleanup): 42 features')
    print(f'   Step 2.2 (Encoding): 42 features')
    print(f'   Step 2.3 (Correlation): 34 features')
    print(f'   Step 2.4 (Variance): {len(remaining_features)} features')
    print(f'   Remaining for Step 2.5: {len(remaining_features)} features')
    
    print('\n✅ Variance analysis completed successfully!')
    print('💡 Next step: Statistical testing to select most discriminative features')

if __name__ == "__main__":
    main()
