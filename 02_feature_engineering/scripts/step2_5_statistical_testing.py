#!/usr/bin/env python3
"""
Step 2.5: Feature Reduction - Statistical Testing
Selects only statistically significant features that best distinguish DoS from Normal traffic
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_statistical_metrics(df, feature_columns, target_col='label'):
    """Calculate comprehensive statistical metrics for feature selection"""
    
    X = df[feature_columns]
    y = df[target_col]
    
    # Separate DoS and Normal data
    dos_data = df[df[target_col] == 1]
    normal_data = df[df[target_col] == 0]
    
    statistical_metrics = []
    
    for feature in feature_columns:
        # ANOVA F-test
        f_stat, p_value = stats.f_oneway(dos_data[feature], normal_data[feature])
        
        # Effect size (Cohen's d)
        dos_mean = dos_data[feature].mean()
        normal_mean = normal_data[feature].mean()
        pooled_std = np.sqrt(((len(dos_data) - 1) * dos_data[feature].var() + 
                             (len(normal_data) - 1) * normal_data[feature].var()) / 
                            (len(dos_data) + len(normal_data) - 2))
        
        cohens_d = abs(dos_mean - normal_mean) / pooled_std if pooled_std > 0 else 0
        
        # Additional metrics
        variance = df[feature].var()
        dos_std = dos_data[feature].std()
        normal_std = normal_data[feature].std()
        
        # Statistical significance levels
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "n.s."
        
        # Effect size interpretation
        if cohens_d < 0.2:
            effect_size = "Small"
        elif cohens_d < 0.5:
            effect_size = "Medium"
        elif cohens_d < 0.8:
            effect_size = "Large"
        else:
            effect_size = "Very Large"
        
        statistical_metrics.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significance': significance,
            'effect_size': effect_size,
            'dos_mean': dos_mean,
            'normal_mean': normal_mean,
            'dos_std': dos_std,
            'normal_std': normal_std,
            'variance': variance
        })
    
    return pd.DataFrame(statistical_metrics)

def calculate_mutual_information(df, feature_columns, target_col='label'):
    """Calculate mutual information scores"""
    X = df[feature_columns]
    y = df[target_col]
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': feature_columns,
        'mutual_info': mi_scores
    })
    
    return mi_df.sort_values('mutual_info', ascending=False)

def evaluate_feature_importance(df, feature_columns, target_col='label'):
    """Evaluate feature importance using Random Forest"""
    X = df[feature_columns]
    y = df[target_col]
    
    # Train Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    return importance_df

def select_top_features(stats_df, mi_df, rf_df, target_features=10):
    """Select top features based on multiple criteria"""
    
    # Merge all metrics
    combined_df = stats_df.merge(mi_df, on='feature').merge(rf_df, on='feature')
    
    # Normalize scores to 0-1 range for combination
    combined_df['f_stat_norm'] = (combined_df['f_statistic'] - combined_df['f_statistic'].min()) / \
                                 (combined_df['f_statistic'].max() - combined_df['f_statistic'].min())
    
    combined_df['mi_norm'] = (combined_df['mutual_info'] - combined_df['mutual_info'].min()) / \
                             (combined_df['mutual_info'].max() - combined_df['mutual_info'].min())
    
    combined_df['rf_norm'] = (combined_df['rf_importance'] - combined_df['rf_importance'].min()) / \
                             (combined_df['rf_importance'].max() - combined_df['rf_importance'].min())
    
    # Combined score (weighted average)
    combined_df['combined_score'] = (
        0.4 * combined_df['f_stat_norm'] +  # ANOVA F-test (40%)
        0.3 * combined_df['mi_norm'] +      # Mutual Information (30%)
        0.3 * combined_df['rf_norm']        # Random Forest Importance (30%)
    )
    
    # Sort by combined score
    combined_df = combined_df.sort_values('combined_score', ascending=False)
    
    # Apply significance filter (must be p < 0.05)
    significant_features = combined_df[combined_df['p_value'] < 0.05]
    
    # Select top features
    selected_features = significant_features.head(target_features)
    
    return combined_df, selected_features

def create_statistical_visualization(stats_df, selected_features, output_dir='../results'):
    """Create visualizations for statistical analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Feature Selection Analysis', fontsize=16, fontweight='bold')
    
    # 1. F-statistic distribution
    axes[0, 0].hist(stats_df['f_statistic'], bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('F-statistic')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Distribution of F-statistics')
    axes[0, 0].set_yscale('log')
    
    # 2. P-value distribution
    axes[0, 1].hist(stats_df['p_value'], bins=20, alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('P-value')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Distribution of P-values')
    axes[0, 1].axvline(x=0.05, color='red', linestyle='--', label='p=0.05 threshold')
    axes[0, 1].legend()
    
    # 3. Effect size distribution
    axes[0, 2].hist(stats_df['cohens_d'], bins=20, alpha=0.7, color='salmon')
    axes[0, 2].set_xlabel("Cohen's d (Effect Size)")
    axes[0, 2].set_ylabel('Number of Features')
    axes[0, 2].set_title('Distribution of Effect Sizes')
    
    # 4. F-statistic vs P-value scatter
    scatter = axes[1, 0].scatter(stats_df['p_value'], stats_df['f_statistic'], 
                                alpha=0.6, c=stats_df['cohens_d'], cmap='viridis', s=50)
    axes[1, 0].set_xlabel('P-value')
    axes[1, 0].set_ylabel('F-statistic')
    axes[1, 0].set_title('F-statistic vs P-value')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].axvline(x=0.05, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(scatter, ax=axes[1, 0], label="Cohen's d")
    
    # 5. Selected features F-statistics
    selected_f_stats = selected_features['f_statistic'].head(10)
    selected_names = selected_features['feature'].head(10)
    
    axes[1, 1].barh(range(len(selected_f_stats)), selected_f_stats, color='gold', alpha=0.8)
    axes[1, 1].set_yticks(range(len(selected_names)))
    axes[1, 1].set_yticklabels(selected_names)
    axes[1, 1].set_xlabel('F-statistic')
    axes[1, 1].set_title('Top Selected Features (F-statistics)')
    axes[1, 1].invert_yaxis()
    
    # 6. Effect sizes of selected features
    selected_effects = selected_features['cohens_d'].head(10)
    
    axes[1, 2].barh(range(len(selected_effects)), selected_effects, color='lightcoral', alpha=0.8)
    axes[1, 2].set_yticks(range(len(selected_names)))
    axes[1, 2].set_yticklabels(selected_names)
    axes[1, 2].set_xlabel("Cohen's d (Effect Size)")
    axes[1, 2].set_title('Top Selected Features (Effect Sizes)')
    axes[1, 2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Statistical analysis visualization saved to: {output_dir}/statistical_analysis.png")

def main():
    print('=' * 60)
    print('📋 STEP 2.5: STATISTICAL TESTING')
    print('=' * 60)
    print('Selecting statistically significant features that best distinguish DoS from Normal...')
    print()

    # Load the variance-cleaned dataset
    print('📂 Loading variance-cleaned dataset...')
    df = pd.read_csv('../data/variance_cleaned_dataset.csv')
    
    print(f'Dataset shape: {df.shape}')
    print(f'Features: {df.shape[1] - 1} (excluding label)')
    print()

    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']

    print('🔍 Calculating statistical metrics for all features...')
    
    # Calculate statistical metrics
    stats_df = calculate_statistical_metrics(df, feature_columns)
    
    # Calculate mutual information
    print('🧠 Calculating mutual information scores...')
    mi_df = calculate_mutual_information(df, feature_columns)
    
    # Calculate Random Forest feature importance
    print('🌲 Evaluating feature importance with Random Forest...')
    rf_df = evaluate_feature_importance(df, feature_columns)
    
    # Display comprehensive analysis
    print('\n📊 COMPREHENSIVE STATISTICAL ANALYSIS:')
    print('=' * 100)
    print(f"{'Feature':<20} {'F-stat':<10} {'p-value':<12} {'Cohen-d':<10} {'Sig':<5} {'Effect':<12} {'MI':<8} {'RF-Imp':<8}")
    print('=' * 100)
    
    # Merge for display
    display_df = stats_df.merge(mi_df, on='feature').merge(rf_df, on='feature')
    display_df = display_df.sort_values('f_statistic', ascending=False)
    
    for _, row in display_df.iterrows():
        print(f"{row['feature']:<20} {row['f_statistic']:<10.2f} {row['p_value']:<12.2e} "
              f"{row['cohens_d']:<10.3f} {row['significance']:<5} {row['effect_size']:<12} "
              f"{row['mutual_info']:<8.3f} {row['rf_importance']:<8.3f}")
    
    # Feature selection
    print(f'\n🎯 Selecting top 10 features based on combined criteria...')
    combined_df, selected_features = select_top_features(stats_df, mi_df, rf_df, target_features=10)
    
    # Display selected features
    print(f'\n✅ TOP 10 SELECTED FEATURES:')
    print('=' * 90)
    print(f"{'Rank':<5} {'Feature':<20} {'Combined':<10} {'F-stat':<10} {'p-value':<12} {'Cohen-d':<10}")
    print('=' * 90)
    
    for i, (_, row) in enumerate(selected_features.iterrows(), 1):
        print(f"{i:<5} {row['feature']:<20} {row['combined_score']:<10.3f} "
              f"{row['f_statistic']:<10.2f} {row['p_value']:<12.2e} {row['cohens_d']:<10.3f}")
    
    # Check statistical significance
    significant_count = len(stats_df[stats_df['p_value'] < 0.05])
    print(f'\n📈 SIGNIFICANCE ANALYSIS:')
    print(f'   Features with p < 0.05: {significant_count}/{len(feature_columns)} ({significant_count/len(feature_columns)*100:.1f}%)')
    print(f'   Features with p < 0.01: {len(stats_df[stats_df["p_value"] < 0.01])}/{len(feature_columns)}')
    print(f'   Features with p < 0.001: {len(stats_df[stats_df["p_value"] < 0.001])}/{len(feature_columns)}')
    
    # Effect size analysis
    large_effect = len(stats_df[stats_df['cohens_d'] >= 0.8])
    medium_effect = len(stats_df[(stats_df['cohens_d'] >= 0.5) & (stats_df['cohens_d'] < 0.8)])
    small_effect = len(stats_df[(stats_df['cohens_d'] >= 0.2) & (stats_df['cohens_d'] < 0.5)])
    
    print(f'\n📊 EFFECT SIZE DISTRIBUTION:')
    print(f'   Very Large/Large effect (d ≥ 0.8): {large_effect} features')
    print(f'   Medium effect (0.5 ≤ d < 0.8): {medium_effect} features')
    print(f'   Small effect (0.2 ≤ d < 0.5): {small_effect} features')
    
    # Create final statistical dataset
    print(f'\n📊 Creating final statistical features dataset...')
    selected_feature_names = selected_features['feature'].tolist()
    df_statistical = df[selected_feature_names + ['label']].copy()
    
    print(f'Original features: {len(feature_columns)}')
    print(f'Features after statistical selection: {len(selected_feature_names)}')
    print(f'Reduction: {len(feature_columns) - len(selected_feature_names)} features ({(len(feature_columns) - len(selected_feature_names))/len(feature_columns)*100:.1f}%)')
    
    # Save statistical features dataset
    output_path = '../data/statistical_features.csv'
    df_statistical.to_csv(output_path, index=False)
    print(f'\n💾 Statistical features dataset saved to: {output_path}')
    
    # Create visualization
    print('\n📊 Creating statistical analysis visualization...')
    create_statistical_visualization(stats_df, selected_features)
    
    # Verify data integrity
    print('\n🔍 Data integrity check...')
    print(f'Original dataset shape: {df.shape}')
    print(f'Statistical dataset shape: {df_statistical.shape}')
    print(f'Records preserved: {len(df_statistical) == len(df)}')
    
    # Check label distribution
    original_distribution = df['label'].value_counts().sort_index()
    new_distribution = df_statistical['label'].value_counts().sort_index()
    print(f'Label distribution preserved: {original_distribution.equals(new_distribution)}')
    
    # Feature progression summary
    print('\n📈 COMPLETE FEATURE PROGRESSION SUMMARY:')
    print(f'   Step 2.1 (Cleanup): 42 features')
    print(f'   Step 2.2 (Encoding): 42 features')
    print(f'   Step 2.3 (Correlation): 34 features')
    print(f'   Step 2.4 (Variance): 18 features')
    print(f'   Step 2.5 (Statistical): {len(selected_feature_names)} features')
    print(f'   Ready for Step 2.6: {len(selected_feature_names)} features')
    
    print('\n✅ Statistical testing completed successfully!')
    print('💡 Final feature set optimized for DoS detection performance')
    print('🚀 Ready for Step 2.6: Feature Scaling!')

if __name__ == "__main__":
    main()
