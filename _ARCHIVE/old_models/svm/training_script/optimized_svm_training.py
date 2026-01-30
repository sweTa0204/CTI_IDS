#!/usr/bin/env python3
"""
OPTIMIZED SVM TRAINING - FAST EXECUTION
======================================

SVM Model Training for DoS Detection
Optimized for speed with effective parameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import joblib
import json
import time
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("⚡ SVM OPTIMIZED TRAINING")
    print("=" * 40)
    
    # Step 1: Load Data
    print("\n📊 STEP 1: LOADING DATA")
    print("=" * 30)
    
    data_path = "../../../../../../Final_year_project/dos_detection/working_dataset.csv"
    df = pd.read_csv(data_path)
    
    feature_names = [col for col in df.columns if col != 'label']
    X = df[feature_names]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test")
    
    # Step 2: Feature Scaling (Critical for SVM)
    print("\n📏 STEP 2: FEATURE SCALING")
    print("=" * 30)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Features scaled with StandardScaler")
    
    # Step 3: Quick Hyperparameter Tuning (Simplified for speed)
    print("\n🔧 STEP 3: OPTIMIZED PARAMETER TUNING")
    print("=" * 30)
    
    # Simplified parameter grid for faster execution
    param_grid = {
        'C': [1.0, 10.0],  # Reduced from 5 to 2 values
        'kernel': ['rbf', 'linear'],  # Most effective kernels
        'gamma': ['scale', 'auto']  # Only 2 gamma values
    }
    
    total_combinations = 2 * 2 * 2  # 8 combinations
    print(f"📊 Parameter combinations: {total_combinations}")
    print(f"🎯 Optimized for speed and effectiveness")
    
    svm_model = SVC(random_state=42, probability=True)
    
    print(f"\n⚙️ Starting optimized Grid Search...")
    tuning_start = time.time()
    
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=3,  # 3-fold CV
        scoring='f1',
        n_jobs=-1,
        verbose=0  # Reduce output for speed
    )
    
    grid_search.fit(X_train_scaled, y_train)
    tuning_time = time.time() - tuning_start
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"✅ Tuning completed in {tuning_time:.2f} seconds")
    print(f"🏆 Best CV F1-Score: {best_score:.4f}")
    print(f"🎯 Best parameters: {best_params}")
    
    # Step 4: Train Final Model
    print("\n🚀 STEP 4: FINAL MODEL TRAINING")
    print("=" * 30)
    
    start_time = time.time()
    final_model = SVC(**best_params, random_state=42, probability=True)
    final_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Model trained in {training_time:.2f} seconds")
    print(f"📊 Kernel: {final_model.kernel}")
    print(f"📊 C: {final_model.C}")
    
    # Step 5: Evaluate Performance
    print("\n📈 STEP 5: PERFORMANCE EVALUATION")
    print("=" * 30)
    
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"               Normal  DoS")
    print(f"Actual Normal    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       DoS       {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_f1_score': best_score
    }
    
    # Step 6: Create Visualization
    print("\n📊 STEP 6: CREATING VISUALIZATION")
    print("=" * 30)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SVM - DoS Detection Performance', fontsize=14, fontweight='bold')
    
    # Confusion Matrix
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Normal', 'DoS'])
    ax1.set_yticklabels(['Normal', 'DoS'])
    
    # ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_plot = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc_plot:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")
    
    # Performance Metrics
    ax3 = axes[1, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    
    bars = ax3.bar(metric_names, values, color=['orange', 'darkorange', 'peru', 'chocolate', 'sandybrown'])
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Model Comparison (if we have previous results)
    ax4 = axes[1, 1]
    model_names = ['Random Forest', 'XGBoost', 'Logistic Reg', 'SVM']
    model_accuracies = [95.29, 95.54, 78.18, accuracy*100]  # Previous results + current
    
    bars = ax4.bar(model_names, model_accuracies, 
                   color=['skyblue', 'lightgreen', 'mediumpurple', 'orange'])
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Model Comparison')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, model_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('../results', exist_ok=True)
    plot_path = '../results/svm_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {plot_path}")
    plt.close()
    
    # Step 7: Save Model and Results
    print("\n💾 STEP 7: SAVING RESULTS")
    print("=" * 30)
    
    try:
        # Create directories
        os.makedirs('../saved_model', exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        os.makedirs('../documentation', exist_ok=True)
        
        # Save model and scaler
        joblib.dump(final_model, '../saved_model/svm_model.pkl')
        joblib.dump(scaler, '../saved_model/feature_scaler.pkl')
        
        # Save results
        results = {
            'model_name': 'SVM',
            'training_date': str(datetime.now()),
            'best_parameters': best_params,
            'performance_metrics': metrics,
            'training_time_seconds': training_time,
            'tuning_time_seconds': tuning_time
        }
        
        with open('../results/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        with open('../saved_model/feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        print(f"✅ Model and results saved successfully")
        
    except Exception as e:
        print(f"❌ Error saving: {str(e)}")
        return False
    
    # Step 8: Generate Summary Report
    print("\n📋 STEP 8: FINAL SUMMARY")
    print("=" * 30)
    
    report = f"""# SVM TRAINING COMPLETION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## PERFORMANCE RESULTS
- Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
- F1-Score: {f1:.4f} ({f1*100:.2f}%)
- ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)
- Training Time: {training_time:.2f} seconds
- Tuning Time: {tuning_time:.2f} seconds

## BEST PARAMETERS
- Kernel: {best_params['kernel']}
- C: {best_params['C']}
- Gamma: {best_params['gamma']}

## 4-MODEL COMPARISON COMPLETE
1. XGBoost: 95.54% (Winner)
2. Random Forest: 95.29%
3. SVM: {accuracy*100:.2f}%
4. Logistic Regression: 78.18%

## NEXT STEPS
- All 4 models trained and compared
- Ready for Layer 2: XAI/SHAP analysis
- Model selection for production deployment

---
SVM Training Successfully Completed
"""
    
    with open('../documentation/training_report.md', 'w') as f:
        f.write(report)
    
    print(f"🎉 SVM TRAINING COMPLETED!")
    print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Final F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"⏱️  Total Time: {training_time + tuning_time:.2f} seconds")
    
    print(f"\n🏆 4-MODEL COMPARISON COMPLETE:")
    print(f"   1. XGBoost: 95.54% (Best)")
    print(f"   2. Random Forest: 95.29%")
    print(f"   3. SVM: {accuracy*100:.2f}%")
    print(f"   4. Logistic Regression: 78.18%")
    
    print(f"\n🚀 READY FOR LAYER 2: XAI ANALYSIS")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ SVM training pipeline completed successfully!")
        print(f"🎯 4-model comparison framework complete!")
    else:
        print(f"\n❌ Training encountered issues.")
