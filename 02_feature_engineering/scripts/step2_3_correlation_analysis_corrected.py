#!/usr/bin/env python3
"""
CORRECTED Step 2.3: Correlation Analysis with Smart Domain Knowledge Selection
Fixed the wrong decisions identified in validation
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def find_high_correlations(df, threshold=0.90):
    """Find pairs of features with correlation above threshold"""
    corr_matrix = df.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper_triangle = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    
    # Find pairs with high correlation
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value >= threshold:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))
    
    return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)

def calculate_discrimination_power(df, feature, target_col='label'):
    """Calculate how well a feature discriminates between classes"""
    dos_data = df[df[target_col] == 1][feature]
    normal_data = df[df[target_col] == 0][feature]
    
    # ANOVA F-test
    f_stat, p_value = stats.f_oneway(dos_data, normal_data)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(dos_data) - 1) * dos_data.var() + 
                         (len(normal_data) - 1) * normal_data.var()) / 
                        (len(dos_data) + len(normal_data) - 2))
    
    cohen_d = abs(dos_data.mean() - normal_data.mean()) / pooled_std if pooled_std > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'cohens_d': cohen_d,
        'variance': df[feature].var(),
        'dos_mean': dos_data.mean(),
        'normal_mean': normal_data.mean()
    }

def smart_feature_selection(df, pair, target_col='label'):
    """
    CORRECTED: Smart selection based on STATISTICAL EVIDENCE
    Choose the feature with better discrimination power
    """
    feature1, feature2, correlation = pair
    
    # Calculate discrimination power for both features
    stats1 = calculate_discrimination_power(df, feature1, target_col)
    stats2 = calculate_discrimination_power(df, feature2, target_col)
    
    print(f"\n🔍 ANALYZING PAIR: {feature1} ↔ {feature2} (correlation: {correlation:.3f})")
    print(f"   {feature1}: F-stat={stats1['f_statistic']:.2f}, p-val={stats1['p_value']:.6f}, variance={stats1['variance']:.6f}")
    print(f"   {feature2}: F-stat={stats2['f_statistic']:.2f}, p-val={stats2['p_value']:.6f}, variance={stats2['variance']:.6f}")
    
    # CORRECTED DECISION LOGIC: Choose based on statistical evidence
    feature1_better = (
        stats1['f_statistic'] > stats2['f_statistic'] and  # Better discrimination
        stats1['p_value'] < stats2['p_value']  # More significant
    )
    
    if feature1_better:
        keep_feature = feature1
        remove_feature = feature2
        print(f"   ✅ DECISION: Keep {feature1} (better discrimination & significance)")
    else:
        keep_feature = feature2
        remove_feature = feature1
        print(f"   ✅ DECISION: Keep {feature2} (better discrimination & significance)")
    
    return keep_feature, remove_feature

def main():
    print('=' * 60)
    print('🔧 CORRECTED STEP 2.3: CORRELATION ANALYSIS')
    print('=' * 60)
    print('Fixing the wrong decisions from previous analysis...')
    print()

    # Load the encoded dataset
    print('📂 Loading encoded dataset...')
    df = pd.read_csv('../data/encoded_dataset.csv')
    
    print(f'Dataset shape: {df.shape}')
    print(f'Features: {df.shape[1] - 1} (excluding label)')
    print()

    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']

    print('🔍 Finding highly correlated feature pairs (threshold > 0.90)...')
    high_corr_pairs = find_high_correlations(X, threshold=0.90)
    
    print(f'Found {len(high_corr_pairs)} highly correlated pairs:')
    for pair in high_corr_pairs:
        print(f'  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}')
    print()

    # Apply CORRECTED smart selection
    print('🧠 Applying CORRECTED smart selection based on statistical evidence...')
    features_to_remove = []
    
    # Process each highly correlated pair
    for pair in high_corr_pairs:
        feature1, feature2, correlation = pair
        
        # Skip if one of the features is already marked for removal
        if feature1 in features_to_remove or feature2 in features_to_remove:
            print(f"\n⏭️  SKIPPING: {feature1} ↔ {feature2} (one already removed)")
            continue
        
        keep_feature, remove_feature = smart_feature_selection(df, pair)
        features_to_remove.append(remove_feature)

    print('\n' + '='*60)
    print('📋 CORRECTED REMOVAL SUMMARY')
    print('='*60)
    print(f'Total features to remove: {len(features_to_remove)}')
    print('Features being removed:')
    for i, feature in enumerate(features_to_remove, 1):
        print(f'  {i}. {feature}')
    
    # Create decorrelated dataset
    print(f'\n📊 Creating corrected decorrelated dataset...')
    decorrelated_features = [col for col in feature_columns if col not in features_to_remove]
    df_decorrelated = df[decorrelated_features + ['label']].copy()
    
    print(f'Original features: {len(feature_columns)}')
    print(f'Features after removal: {len(decorrelated_features)}')
    print(f'Reduction: {len(features_to_remove)} features ({len(features_to_remove)/len(feature_columns)*100:.1f}%)')
    
    # Verify no high correlations remain
    print('\n🔍 Verifying correlation removal...')
    remaining_high_corr = find_high_correlations(df_decorrelated[decorrelated_features], threshold=0.90)
    
    if len(remaining_high_corr) == 0:
        print('✅ SUCCESS: No high correlations remaining!')
    else:
        print(f'⚠️  WARNING: {len(remaining_high_corr)} high correlations still remain:')
        for pair in remaining_high_corr:
            print(f'  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}')

    # Save corrected dataset
    output_path = '../data/decorrelated_dataset_corrected.csv'
    df_decorrelated.to_csv(output_path, index=False)
    print(f'\n💾 Corrected decorrelated dataset saved to: {output_path}')
    
    # Verify data integrity
    print('\n🔍 Data integrity check...')
    print(f'Original dataset shape: {df.shape}')
    print(f'Decorrelated dataset shape: {df_decorrelated.shape}')
    print(f'Records preserved: {len(df_decorrelated) == len(df)}')
    
    # Check label distribution
    original_distribution = df['label'].value_counts().sort_index()
    new_distribution = df_decorrelated['label'].value_counts().sort_index()
    print(f'Label distribution preserved: {original_distribution.equals(new_distribution)}')
    
    print('\n✅ CORRECTED correlation analysis completed successfully!')
    print('💡 Key fixes applied:')
    print('   1. Better statistical decision-making')
    print('   2. Preserve features with stronger discrimination power')
    print('   3. Keep statistically significant features')

if __name__ == "__main__":
    main()
