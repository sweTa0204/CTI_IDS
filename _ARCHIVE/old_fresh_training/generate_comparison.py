#!/usr/bin/env python3
"""
Model Comparison Chart - All 5 Models
=====================================
Generates a comprehensive comparison chart of all trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_all_results():
    """Load results from all models"""
    models = ['xgboost', 'random_forest', 'svm', 'mlp', 'logistic_regression']
    display_names = ['XGBoost', 'Random Forest', 'SVM', 'MLP', 'Logistic Reg.']

    results = []
    for model, display_name in zip(models, display_names):
        path = Path(f'{model}/training_results.json')
        with open(path, 'r') as f:
            data = json.load(f)

        metrics = data['performance_metrics']
        results.append({
            'Model': display_name,
            'Accuracy': metrics['accuracy'] * 100,
            'Precision': metrics['precision'] * 100,
            'Recall': metrics['recall'] * 100,
            'F1-Score': metrics['f1_score'] * 100,
            'ROC-AUC': metrics['roc_auc'] * 100
        })

    return pd.DataFrame(results)

def generate_comparison_chart(df):
    """Generate comprehensive comparison chart"""

    fig = plt.figure(figsize=(16, 12))

    # 1. Bar chart comparing all metrics
    ax1 = fig.add_subplot(2, 2, 1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(df))
    width = 0.15

    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax1.bar(x + i*width, df[metric], width, label=metric, color=color)

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(df['Model'], fontsize=10)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim([70, 105])
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5)

    # 2. F1-Score comparison (main metric)
    ax2 = fig.add_subplot(2, 2, 2)
    colors_f1 = ['#3498db', '#27ae60', '#9b59b6', '#e67e22', '#e74c3c']
    bars = ax2.barh(df['Model'], df['F1-Score'], color=colors_f1)

    for bar, val in zip(bars, df['F1-Score']):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')

    ax2.set_xlabel('F1-Score (%)', fontsize=12)
    ax2.set_title('Model Ranking by F1-Score', fontsize=14, fontweight='bold')
    ax2.set_xlim([70, 102])
    ax2.axvline(x=90, color='gray', linestyle='--', alpha=0.5)

    # 3. Radar chart
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors_radar = ['#3498db', '#27ae60', '#9b59b6', '#e67e22', '#e74c3c']

    for idx, row in df.iterrows():
        values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']]
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors_radar[idx])
        ax3.fill(angles, values, alpha=0.1, color=colors_radar[idx])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim([70, 100])
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax3.set_title('Model Performance Radar', fontsize=14, fontweight='bold', pad=20)

    # 4. Summary table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Create ranking
    df_sorted = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    summary_text = """
    MODEL PERFORMANCE SUMMARY
    ═════════════════════════════════════════════════

    🏆 RANKING BY F1-SCORE:

    1st: {} ({:.2f}%)
    2nd: {} ({:.2f}%)
    3rd: {} ({:.2f}%)
    4th: {} ({:.2f}%)
    5th: {} ({:.2f}%)

    ═════════════════════════════════════════════════

    📊 KEY INSIGHTS:

    • Best Overall: {} (Highest F1-Score)
    • Best Precision: {} ({:.1f}%)
    • Best Recall: {} ({:.1f}%)
    • Best ROC-AUC: {} ({:.1f}%)

    ═════════════════════════════════════════════════

    ✅ All models trained with random_state=42
    ✅ 80/20 train-test split with stratification
    ✅ 5-fold cross-validation performed
    """.format(
        df_sorted.iloc[0]['Model'], df_sorted.iloc[0]['F1-Score'],
        df_sorted.iloc[1]['Model'], df_sorted.iloc[1]['F1-Score'],
        df_sorted.iloc[2]['Model'], df_sorted.iloc[2]['F1-Score'],
        df_sorted.iloc[3]['Model'], df_sorted.iloc[3]['F1-Score'],
        df_sorted.iloc[4]['Model'], df_sorted.iloc[4]['F1-Score'],
        df_sorted.iloc[0]['Model'],
        df.loc[df['Precision'].idxmax(), 'Model'], df['Precision'].max(),
        df.loc[df['Recall'].idxmax(), 'Model'], df['Recall'].max(),
        df.loc[df['ROC-AUC'].idxmax(), 'Model'], df['ROC-AUC'].max()
    )

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout(pad=2.0)
    plt.savefig('model_comparison_complete.png', bbox_inches='tight', facecolor='white')
    plt.close()

    print("Saved: model_comparison_complete.png")

def save_comparison_results(df):
    """Save comparison results to JSON"""
    results = {
        "comparison_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": df.to_dict('records'),
        "ranking": df.sort_values('F1-Score', ascending=False)['Model'].tolist(),
        "best_model": df.loc[df['F1-Score'].idxmax(), 'Model'],
        "best_f1_score": float(df['F1-Score'].max())
    }

    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Saved: comparison_results.json")

def main():
    print("="*60)
    print("GENERATING MODEL COMPARISON CHART")
    print("="*60)

    df = load_all_results()

    print("\nModel Performance Summary:")
    print(df.to_string(index=False))

    generate_comparison_chart(df)
    save_comparison_results(df)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)

    # Print ranking
    df_sorted = df.sort_values('F1-Score', ascending=False)
    print("\n🏆 FINAL RANKING (by F1-Score):")
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"   {i}. {row['Model']}: {row['F1-Score']:.2f}%")

if __name__ == "__main__":
    main()
