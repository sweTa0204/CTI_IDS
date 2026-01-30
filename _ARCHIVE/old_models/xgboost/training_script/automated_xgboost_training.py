#!/usr/bin/env python3
"""
AUTOMATED XGBOOST TRAINING - COMPLETE PIPELINE
============================================

XGBoost Model Training for DoS Detection
Layer 1: Training + Performance Evaluation
XAI Analysis: Layer 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import joblib
import json
import time
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("🚀 XGBOOST AUTOMATED TRAINING")
    print("=" * 50)
    
    # Step 1: Load Data
    print("\n📊 STEP 1: LOADING DATA")
    print("=" * 40)
    
    data_path = "../../../../../../Final_year_project/dos_detection/working_dataset.csv"
    df = pd.read_csv(data_path)
    
    feature_names = [col for col in df.columns if col != 'label']
    X = df[feature_names]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data loaded successfully")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"📊 Features: {feature_names}")
    
    # Step 2: Hyperparameter Tuning
    print("\n🔧 STEP 2: HYPERPARAMETER OPTIMIZATION")
    print("=" * 40)
    
    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print(f"🔍 Parameter grid defined:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"📊 Total parameter combinations: {total_combinations}")
    
    # Perform Grid Search
    print(f"\n⚙️ Starting Grid Search (3-fold CV for efficiency)...")
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,  # 3-fold for faster execution
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    tuning_start = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - tuning_start
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"✅ Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"🏆 Best CV F1-Score: {best_score:.4f}")
    print(f"🎯 Best parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Step 3: Train Final Model
    print("\n🚀 STEP 3: TRAINING FINAL MODEL")
    print("=" * 40)
    
    # Train final model with best parameters
    final_params = {
        **best_params,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    start_time = time.time()
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Final model training completed in {training_time:.2f} seconds")
    print(f"🌳 Number of estimators: {final_model.n_estimators}")
    print(f"📊 Max depth: {final_model.max_depth}")
    
    # Step 4: Evaluate Performance
    print("\n📈 STEP 4: PERFORMANCE EVALUATION")
    print("=" * 40)
    
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
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
    print(f"                 Predicted")
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
    
    # Step 5: Feature Importance Analysis
    print("\n🔍 STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    importance = final_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:15s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")
    
    # Step 6: Create Visualizations
    print("\n📊 STEP 6: CREATING VISUALIZATIONS")
    print("=" * 40)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XGBoost - DoS Detection Performance', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Normal', 'DoS'])
    ax1.set_yticklabels(['Normal', 'DoS'])
    
    # ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_plot = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='green', lw=2, 
            label=f'ROC curve (AUC = {roc_auc_plot:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")
    
    # Feature Importance
    ax3 = axes[1, 0]
    top_features = feature_importance_df.head(8)
    bars = ax3.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 8 Feature Importance')
    ax3.invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    # Performance Metrics
    ax4 = axes[1, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    bars = ax4.bar(metric_names, values, color=['lightgreen', 'green', 'darkgreen', 'lime', 'forestgreen'])
    ax4.set_ylim([0, 1])
    ax4.set_ylabel('Score')
    ax4.set_title('Performance Metrics Summary')
    ax4.set_xticklabels(metric_names, rotation=45)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create results directory if needed
    os.makedirs('../results', exist_ok=True)
    plot_path = '../results/xgboost_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance visualization saved: {plot_path}")
    plt.close()
    
    # Step 7: Save Model and Results
    print("\n💾 STEP 7: SAVING MODEL AND RESULTS")
    print("=" * 40)
    
    try:
        # Create directories
        os.makedirs('../saved_model', exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        os.makedirs('../documentation', exist_ok=True)
        
        # Save model
        model_path = '../saved_model/xgboost_model.pkl'
        joblib.dump(final_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save results
        results = {
            'model_name': 'XGBoost',
            'training_date': str(datetime.now()),
            'best_parameters': final_params,
            'performance_metrics': metrics,
            'training_time_seconds': training_time,
            'tuning_time_seconds': tuning_time
        }
        results_path = '../results/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Training results saved: {results_path}")
        
        # Save feature names
        feature_path = '../saved_model/feature_names.json'
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"✅ Feature names saved: {feature_path}")
        
    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
        return False
    
    # Step 8: Generate Report
    print("\n📋 STEP 8: GENERATING TRAINING REPORT")
    print("=" * 40)
    
    report = f"""# XGBOOST TRAINING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL CONFIGURATION
- Algorithm: XGBoost Classifier
- Features: 10 network traffic features
- Training Time: {training_time:.2f} seconds
- Hyperparameter Tuning Time: {tuning_time:.2f} seconds
- Cross-Validation: 3-fold
- Class Balance: Balanced (50-50)

## BEST HYPERPARAMETERS
- n_estimators: {final_params['n_estimators']}
- max_depth: {final_params['max_depth']}
- learning_rate: {final_params['learning_rate']}
- subsample: {final_params['subsample']}
- colsample_bytree: {final_params['colsample_bytree']}

## PERFORMANCE METRICS
- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
- ROC-AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)
- CV F1-Score: {metrics['cv_f1_score']:.4f} ({metrics['cv_f1_score']*100:.2f}%)

## TOP 5 IMPORTANT FEATURES FOR DoS DETECTION
"""
    for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
        report += f"{i}. {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.1f}%)\n"
    
    report += f"""
## CONFUSION MATRIX ANALYSIS
- True Negatives (Normal correctly classified): {cm[0,0]}
- False Positives (Normal misclassified as DoS): {cm[0,1]}
- False Negatives (DoS misclassified as Normal): {cm[1,0]}
- True Positives (DoS correctly classified): {cm[1,1]}

## MODEL COMPARISON WITH RANDOM FOREST
### Performance Comparison:
- XGBoost Accuracy: {metrics['accuracy']:.4f}
- XGBoost F1-Score: {metrics['f1_score']:.4f}
- XGBoost ROC-AUC: {metrics['roc_auc']:.4f}

## ANALYSIS SUMMARY
✅ Hyperparameter Tuning: COMPLETED
✅ Training: COMPLETED
✅ Evaluation: COMPLETED  
✅ Model Saved: COMPLETED
✅ Visualizations: COMPLETED
⏳ XAI Analysis: PENDING (Layer 2)

## NEXT STEPS
1. Train Logistic Regression model
2. Train SVM model
3. Compare all 4 models
4. Proceed to Layer 2: XAI/SHAP analysis

---
XGBoost Training Successfully Completed
"""
    
    report_path = '../documentation/training_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Training report saved: {report_path}")
    
    # Final Summary
    print(f"\n🎉 XGBOOST TRAINING COMPLETED!")
    print("=" * 50)
    print(f"🎯 Final Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"🎯 Final F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"🎯 Final ROC-AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print(f"🔧 Tuning Time: {tuning_time:.2f} seconds")
    print(f"💾 Model saved and ready for deployment")
    print(f"📊 All results and visualizations generated")
    
    print(f"\n📋 PROGRESS UPDATE:")
    print(f"   1. ✅ Random Forest: COMPLETED (95.29% accuracy)")
    print(f"   2. ✅ XGBoost: COMPLETED ({metrics['accuracy']*100:.2f}% accuracy)")
    print(f"   3. ⏳ Logistic Regression: Ready to train")
    print(f"   4. ⏳ SVM: Ready to train")
    print(f"   5. ⏳ Layer 2: XAI analysis")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ XGBoost training pipeline completed successfully!")
    else:
        print(f"\n❌ Training encountered issues. Check logs above.")
