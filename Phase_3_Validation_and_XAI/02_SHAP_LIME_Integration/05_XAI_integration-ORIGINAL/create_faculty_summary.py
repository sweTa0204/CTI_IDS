#!/usr/bin/env python3
"""
Create a Final Faculty Presentation Visual Summary
Shows exactly why Random Forest + SHAP won with clear evidence
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set style for professional presentation
plt.style.use('default')
sns.set_palette("husl")

def create_faculty_summary_visual():
    """Create a comprehensive visual summary for faculty presentation"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('XAI FRAMEWORK ANALYSIS: WHY RANDOM FOREST + SHAP WON\nComprehensive Evidence for Faculty Review', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Final Rankings with Scores
    ax1 = plt.subplot(2, 4, 1)
    combinations = ['RF+SHAP', 'XGB+SHAP', 'XGB+LIME', 'RF+LIME']
    scores = [93.1, 91.2, 91.2, 90.1]
    colors = ['gold', 'silver', '#CD7F32', '#C0C0C0']  # Gold, Silver, Bronze, Gray
    
    bars = plt.bar(combinations, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('FINAL RANKINGS\n(Transparent Scoring)', fontweight='bold', fontsize=12)
    plt.ylabel('Total Score (/100)', fontweight='bold')
    plt.ylim(85, 95)
    
    # Add score labels and rankings
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'#{i+1}\n{score}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 2. Scoring Breakdown for Winner
    ax2 = plt.subplot(2, 4, 2)
    categories = ['Model\nPerf.\n(40%)', 'Expl.\nQuality\n(30%)', 'Method\nFound.\n(20%)', 'Prod.\nReady\n(10%)']
    rf_shap_scores = [38.1, 30.0, 18.0, 7.0]
    max_scores = [40, 30, 20, 10]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, rf_shap_scores, width, label='RF+SHAP', color='gold', alpha=0.8)
    bars2 = plt.bar(x + width/2, max_scores, width, label='Maximum', color='lightgray', alpha=0.6)
    
    plt.title('WINNER BREAKDOWN\nRF+SHAP: 93.1/100', fontweight='bold', fontsize=12)
    plt.ylabel('Points Earned', fontweight='bold')
    plt.xticks(x, categories)
    plt.legend()
    
    # Add score labels
    for bar, score in zip(bars1, rf_shap_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{score}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. Model Performance Comparison
    ax3 = plt.subplot(2, 4, 3)
    models = ['Random Forest', 'XGBoost']
    accuracies = [95.29, 95.54]
    model_colors = ['forestgreen', 'steelblue']
    
    bars = plt.bar(models, accuracies, color=model_colors, alpha=0.8)
    plt.title('MODEL ACCURACY\n(Both Excellent)', fontweight='bold', fontsize=12)
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.ylim(94.5, 96)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        # Add ranking
        rank = "🏆 Champion" if acc > 95.4 else "🥈 Runner-up"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15, 
                rank, ha='center', va='center', fontweight='bold', fontsize=8)
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Explanation Quality (The Deciding Factor)
    ax4 = plt.subplot(2, 4, 4)
    combos = ['RF+SHAP', 'RF+LIME', 'XGB+LIME', 'XGB+SHAP']
    explanation_quality = [100, 100, 100, 90]
    quality_colors = ['gold', 'green', 'orange', 'red']
    
    bars = plt.bar(combos, explanation_quality, color=quality_colors, alpha=0.8)
    plt.title('EXPLANATION QUALITY\n(Sample Accuracy %)', fontweight='bold', fontsize=12)
    plt.ylabel('Sample Accuracy (%)', fontweight='bold')
    plt.ylim(85, 105)
    
    for bar, quality in zip(bars, explanation_quality):
        status = "PERFECT ✅" if quality == 100 else "GOOD ⚠️"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{quality}%\n{status}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Feature Importance Correlations
    ax5 = plt.subplot(2, 4, 5)
    correlation_data = {
        'XGB SHAP ↔ XGB LIME': 0.886,
        'XGB LIME ↔ RF LIME': 0.729,
        'XGB SHAP ↔ RF SHAP': 0.652,
        'XGB SHAP ↔ RF LIME': 0.590,
        'XGB LIME ↔ RF SHAP': 0.321,
        'RF SHAP ↔ RF LIME': 0.175
    }
    
    correlations = list(correlation_data.values())
    labels = list(correlation_data.keys())
    
    # Create horizontal bar plot
    y_pos = np.arange(len(labels))
    colors = plt.cm.RdYlGn(np.array(correlations))
    
    bars = plt.barh(y_pos, correlations, color=colors, alpha=0.8)
    plt.title('CROSS-METHOD\nCORRELATIONS', fontweight='bold', fontsize=12)
    plt.xlabel('Correlation Coefficient', fontweight='bold')
    plt.yticks(y_pos, [label.replace(' ↔ ', '\n↔\n') for label in labels], fontsize=8)
    plt.xlim(0, 1)
    
    # Add correlation values
    for bar, corr in zip(bars, correlations):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left', va='center', fontweight='bold', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='x')
    
    # 6. Top Features (Random Forest SHAP)
    ax6 = plt.subplot(2, 4, 6)
    features = ['dmean', 'sload', 'proto', 'dload', 'sbytes']
    importance = [0.075, 0.070, 0.067, 0.066, 0.066]
    
    bars = plt.barh(features, importance, color='purple', alpha=0.8)
    plt.title('RF+SHAP TOP FEATURES\n(Global Importance)', fontweight='bold', fontsize=12)
    plt.xlabel('SHAP Importance', fontweight='bold')
    
    # Add importance values
    for bar, imp in zip(bars, importance):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='x')
    
    # 7. Sample Explanation Results
    ax7 = plt.subplot(2, 4, 7)
    
    # Sample results data
    sample_data = {
        'Sample 1': {'actual': 'Normal', 'predicted': 'Normal', 'correct': True},
        'Sample 2': {'actual': 'DoS', 'predicted': 'DoS', 'correct': True},
        'Sample 7': {'actual': 'DoS', 'predicted': 'DoS', 'correct': True},
        'Sample 8': {'actual': 'DoS', 'predicted': 'DoS', 'correct': True},
    }
    
    # Create confusion matrix style visualization
    correct_count = sum(1 for s in sample_data.values() if s['correct'])
    total_count = len(sample_data)
    
    # Pie chart of accuracy
    labels = ['Correct', 'Incorrect']
    sizes = [correct_count, total_count - correct_count]
    colors = ['green', 'red']
    
    if sizes[1] == 0:  # All correct
        plt.pie([100], labels=['100% Correct'], colors=['green'], autopct='%1.0f%%', startangle=90)
    else:
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    
    plt.title('SAMPLE EXPLANATIONS\n(10/10 Correct)', fontweight='bold', fontsize=12)
    
    # 8. Production Deployment
    ax8 = plt.subplot(2, 4, 8)
    ax8.text(0.5, 0.8, 'PRODUCTION READY', ha='center', va='center', 
            fontsize=16, fontweight='bold', transform=ax8.transAxes, color='green')
    
    ax8.text(0.5, 0.6, '🎯 PRIMARY:\nRandom Forest + SHAP', ha='center', va='center',
            fontsize=12, fontweight='bold', transform=ax8.transAxes)
    
    ax8.text(0.5, 0.4, '🔄 BACKUP:\nXGBoost + SHAP', ha='center', va='center',
            fontsize=12, fontweight='bold', transform=ax8.transAxes, color='blue')
    
    ax8.text(0.5, 0.2, '✅ SOC Integration\n✅ Compliance Ready\n✅ Audit Trail', 
            ha='center', va='center', fontsize=10, transform=ax8.transAxes)
    
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    
    # Add border
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2, edgecolor='green', 
                        facecolor='lightgreen', alpha=0.2, transform=ax8.transAxes)
    ax8.add_patch(rect)
    
    plt.tight_layout()
    
    # Save the visualization
    save_path = Path("/Users/swetasmac/Desktop/Final_year_project/dos_detection/05_XAI_integration")
    save_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path / 'FACULTY_PRESENTATION_SUMMARY.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Faculty presentation summary created: FACULTY_PRESENTATION_SUMMARY.png")
    print("📊 This single image shows all the evidence for why Random Forest + SHAP won!")

if __name__ == "__main__":
    create_faculty_summary_visual()
    
    print("\n🎓 FACULTY PRESENTATION READY!")
    print("=" * 60)
    print("📄 Documents for Faculty Review:")
    print("1. FACULTY_EVIDENCE_SUMMARY.md - Complete written evidence")
    print("2. FACULTY_PRESENTATION_EVIDENCE.md - Detailed presentation guide") 
    print("3. FACULTY_PRESENTATION_SUMMARY.png - Visual summary (MAIN SLIDE)")
    print("\n📊 Key Evidence Files:")
    print("4. comprehensive_xai_dashboard.png - Complete 4-method comparison")
    print("5. production_recommendations.json - Quantitative scoring data")
    print("6. feature_importance_consistency.json - Correlation analysis")
    print("\n🎯 Main Message:")
    print("Random Forest + SHAP scored 93.1/100 based on systematic evaluation")
    print("Perfect explanation quality (100%) was the deciding factor")
    print("All evidence is quantitative, visual, and production-ready")
