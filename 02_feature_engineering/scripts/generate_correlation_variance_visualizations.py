#!/usr/bin/env python3
"""
Correlation and Variance Analysis Visualizations
Generate comprehensive visualizations for correlation matrix and variance analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Rectangle

# Set style for professional plots
plt.style.use('default')
sns.set_palette("RdYlBu_r")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

def load_dataset():
    """Load the original UNSW-NB15 training dataset"""
    try:
        print("📊 Loading UNSW-NB15 training dataset...")
        df = pd.read_csv('../datasetsfinalproject/UNSW_NB15_training-set.csv')
        print(f"✅ Dataset loaded: {len(df)} records, {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print("❌ Dataset not found. Trying alternative path...")
        try:
            df = pd.read_csv('data/dos_detection_dataset.csv')
            print(f"✅ Alternative dataset loaded: {len(df)} records")
            return df
        except FileNotFoundError:
            print("❌ No dataset found!")
            return None

def prepare_numeric_data(df):
    """Prepare numeric data for correlation and variance analysis"""
    print("🔧 Preparing numeric data...")
    
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID column if present
    if 'id' in numeric_columns:
        numeric_columns.remove('id')
    
    # Remove label columns for correlation analysis
    label_cols = ['label', 'attack_cat']
    for col in label_cols:
        if col in numeric_columns:
            numeric_columns.remove(col)
    
    print(f"📈 Selected {len(numeric_columns)} numeric features for analysis")
    return df[numeric_columns], numeric_columns

def create_correlation_heatmap(df_numeric, feature_names):
    """Create comprehensive correlation heatmap"""
    print("🎨 Creating correlation heatmap...")
    
    # Calculate correlation matrix
    correlation_matrix = df_numeric.corr()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main correlation heatmap
    ax1 = plt.subplot(2, 2, (1, 2))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=False,
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.1,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
    
    ax1.set_title('Feature Correlation Matrix\n(Lower Triangle Only)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Features', fontweight='bold')
    ax1.set_ylabel('Features', fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # High correlation pairs analysis
    ax2 = plt.subplot(2, 2, 3)
    
    # Find high correlation pairs (> 0.7 or < -0.7)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Plot top 15 high correlations
        top_correlations = high_corr_df.head(15)
        colors = ['red' if x < 0 else 'green' for x in top_correlations['Correlation']]
        
        bars = ax2.barh(range(len(top_correlations)), top_correlations['Correlation'], 
                       color=colors, alpha=0.7, edgecolor='black')
        
        ax2.set_yticks(range(len(top_correlations)))
        ax2.set_yticklabels([f"{row['Feature_1']} - {row['Feature_2']}" 
                            for _, row in top_correlations.iterrows()], fontsize=9)
        ax2.set_xlabel('Correlation Coefficient', fontweight='bold')
        ax2.set_title('Top 15 High Correlation Pairs\n(|r| > 0.7)', 
                     fontweight='bold', fontsize=14)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_correlations['Correlation'])):
            ax2.text(val + (0.02 if val > 0 else -0.02), i, f'{val:.3f}', 
                    va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No high correlations found\n(|r| > 0.7)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('High Correlation Analysis', fontweight='bold')
    
    # Correlation distribution
    ax3 = plt.subplot(2, 2, 4)
    
    # Get upper triangle correlations (excluding diagonal)
    upper_tri_corr = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlations_flat = upper_tri_corr.stack().values
    
    # Create histogram
    ax3.hist(correlations_flat, bins=50, alpha=0.7, color='skyblue', 
            edgecolor='black', density=True)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Correlation Coefficient', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Distribution of Feature Correlations', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_corr = np.mean(np.abs(correlations_flat))
    median_corr = np.median(np.abs(correlations_flat))
    ax3.text(0.02, 0.98, f'Mean |r|: {mean_corr:.3f}\nMedian |r|: {median_corr:.3f}', 
            transform=ax3.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_variance_analysis(df_numeric, feature_names):
    """Create comprehensive variance analysis visualization"""
    print("📊 Creating variance analysis...")
    
    # Calculate variance for each feature
    variances = df_numeric.var().sort_values(ascending=False)
    
    # Calculate coefficient of variation (CV = std/mean)
    cv_data = []
    for col in df_numeric.columns:
        if df_numeric[col].mean() != 0:
            cv = df_numeric[col].std() / abs(df_numeric[col].mean())
            cv_data.append({'Feature': col, 'CV': cv, 'Variance': df_numeric[col].var()})
    
    cv_df = pd.DataFrame(cv_data).sort_values('CV', ascending=False)
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    
    # Feature variance plot
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Select top 20 features by variance
    top_variance = variances.head(20)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_variance)))
    
    bars = ax1.bar(range(len(top_variance)), top_variance.values, 
                   color=colors, alpha=0.8, edgecolor='black')
    
    ax1.set_xticks(range(len(top_variance)))
    ax1.set_xticklabels(top_variance.index, rotation=45, ha='right')
    ax1.set_ylabel('Variance', fontweight='bold')
    ax1.set_title('Top 20 Features by Variance', fontweight='bold', fontsize=16)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, top_variance.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(top_variance.values) * 0.01,
                f'{val:.2e}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Coefficient of Variation plot
    ax2 = plt.subplot(2, 3, 3)
    
    top_cv = cv_df.head(15)
    colors_cv = ['red' if cv > 2 else 'orange' if cv > 1 else 'green' for cv in top_cv['CV']]
    
    bars_cv = ax2.barh(range(len(top_cv)), top_cv['CV'], 
                       color=colors_cv, alpha=0.7, edgecolor='black')
    
    ax2.set_yticks(range(len(top_cv)))
    ax2.set_yticklabels(top_cv['Feature'], fontsize=10)
    ax2.set_xlabel('Coefficient of Variation', fontweight='bold')
    ax2.set_title('Top 15 Features by\nCoefficient of Variation', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_cv, top_cv['CV'])):
        ax2.text(val + 0.05, i, f'{val:.2f}', va='center', ha='left', fontweight='bold', fontsize=9)
    
    # Variance distribution
    ax3 = plt.subplot(2, 3, 4)
    
    log_variances = np.log10(variances.values + 1e-10)  # Add small value to avoid log(0)
    ax3.hist(log_variances, bins=30, alpha=0.7, color='lightcoral', 
            edgecolor='black', density=True)
    ax3.set_xlabel('Log₁₀(Variance)', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Distribution of Feature Variances\n(Log Scale)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Low variance features (potential candidates for removal)
    ax4 = plt.subplot(2, 3, 5)
    
    low_variance = variances.tail(15)
    bars_low = ax4.barh(range(len(low_variance)), low_variance.values, 
                        color='lightblue', alpha=0.7, edgecolor='black')
    
    ax4.set_yticks(range(len(low_variance)))
    ax4.set_yticklabels(low_variance.index, fontsize=10)
    ax4.set_xlabel('Variance', fontweight='bold')
    ax4.set_title('Bottom 15 Features by Variance\n(Low Variability)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_low, low_variance.values)):
        ax4.text(val + max(low_variance.values) * 0.02, i, f'{val:.2e}', 
                va='center', ha='left', fontweight='bold', fontsize=8)
    
    # Variance vs Mean scatter plot
    ax5 = plt.subplot(2, 3, 6)
    
    means = df_numeric.mean()
    scatter = ax5.scatter(means.values, variances.values, alpha=0.6, 
                         c=cv_df['CV'], cmap='viridis', s=50, edgecolors='black')
    
    ax5.set_xlabel('Feature Mean', fontweight='bold')
    ax5.set_ylabel('Feature Variance', fontweight='bold')
    ax5.set_title('Variance vs Mean Relationship', fontweight='bold')
    ax5.set_xscale('symlog')
    ax5.set_yscale('symlog')
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Coefficient of Variation', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_statistics():
    """Create summary statistics visualization"""
    print("📈 Creating summary statistics...")
    
    df = load_dataset()
    if df is None:
        return None
    
    df_numeric, feature_names = prepare_numeric_data(df)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Feature count by data type
    data_types = df.dtypes.value_counts()
    ax1.pie(data_types.values, labels=data_types.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Feature Distribution by Data Type', fontweight='bold', fontsize=14)
    
    # Missing values analysis
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        ax2.bar(range(len(missing_data)), missing_data.values, color='red', alpha=0.7)
        ax2.set_xticks(range(len(missing_data)))
        ax2.set_xticklabels(missing_data.index, rotation=45, ha='right')
        ax2.set_ylabel('Missing Values Count', fontweight='bold')
        ax2.set_title('Missing Values by Feature', fontweight='bold', fontsize=14)
    else:
        ax2.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16, fontweight='bold')
        ax2.set_title('Missing Values Analysis', fontweight='bold', fontsize=14)
    
    # Feature statistics summary
    stats_summary = df_numeric.describe()
    
    # Display key statistics
    stats_text = f"""
Dataset Summary Statistics:
• Total Records: {len(df):,}
• Total Features: {len(df.columns)}
• Numeric Features: {len(df_numeric.columns)}
• Mean Variance: {df_numeric.var().mean():.2e}
• Median Variance: {df_numeric.var().median():.2e}
• High Variance Features (>1e6): {sum(df_numeric.var() > 1e6)}
• Low Variance Features (<1e-6): {sum(df_numeric.var() < 1e-6)}
    """
    
    ax3.text(0.05, 0.95, stats_text.strip(), transform=ax3.transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Dataset Overview', fontweight='bold', fontsize=14)
    
    # Feature value ranges
    feature_ranges = []
    for col in df_numeric.columns[:10]:  # Top 10 features
        min_val = df_numeric[col].min()
        max_val = df_numeric[col].max()
        range_val = max_val - min_val
        feature_ranges.append({'Feature': col, 'Range': range_val, 'Min': min_val, 'Max': max_val})
    
    range_df = pd.DataFrame(feature_ranges).sort_values('Range', ascending=True)
    
    ax4.barh(range(len(range_df)), range_df['Range'], color='green', alpha=0.7)
    ax4.set_yticks(range(len(range_df)))
    ax4.set_yticklabels(range_df['Feature'])
    ax4.set_xlabel('Value Range (Max - Min)', fontweight='bold')
    ax4.set_title('Feature Value Ranges\n(Top 10 Features)', fontweight='bold', fontsize=14)
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def main():
    """Generate correlation and variance visualizations"""
    print("🎯 Generating Correlation and Variance Visualizations")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load and prepare data
    df = load_dataset()
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    df_numeric, feature_names = prepare_numeric_data(df)
    
    # Generate correlation heatmap
    print("\n1️⃣ Generating correlation analysis...")
    correlation_fig = create_correlation_heatmap(df_numeric, feature_names)
    correlation_output = f'{results_dir}/correlation_analysis.png'
    correlation_fig.savefig(correlation_output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(correlation_fig)
    print(f"✅ Correlation analysis saved: {correlation_output}")
    
    # Generate variance analysis
    print("\n2️⃣ Generating variance analysis...")
    variance_fig = create_variance_analysis(df_numeric, feature_names)
    variance_output = f'{results_dir}/variance_analysis.png'
    variance_fig.savefig(variance_output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(variance_fig)
    print(f"✅ Variance analysis saved: {variance_output}")
    
    # Generate summary statistics
    print("\n3️⃣ Generating summary statistics...")
    summary_fig = create_summary_statistics()
    if summary_fig:
        summary_output = f'{results_dir}/dataset_summary_statistics.png'
        summary_fig.savefig(summary_output, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(summary_fig)
        print(f"✅ Summary statistics saved: {summary_output}")
    
    print(f"\n🎉 All visualizations completed!")
    print("=" * 40)
    print("📊 Generated Files:")
    print(f"   • {correlation_output}")
    print(f"   • {variance_output}")
    print(f"   • {summary_output}")
    print("\n📈 Visualizations Include:")
    print("   • Feature correlation heatmap")
    print("   • High correlation pairs analysis")
    print("   • Correlation distribution")
    print("   • Feature variance analysis")
    print("   • Coefficient of variation")
    print("   • Low variance features")
    print("   • Variance vs mean relationships")
    print("   • Dataset summary statistics")

if __name__ == "__main__":
    main()
