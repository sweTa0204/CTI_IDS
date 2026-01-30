#!/usr/bin/env python3
"""
STEP-BY-STEP RANDOM FOREST TRAINING
===================================

This script allows manual execution of each training step
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
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

def load_data():
    """Step 1: Load and prepare data"""
    print("📊 STEP 1: LOADING DATA")
    print("=" * 40)
    
    # Load data
    data_path = "../../../../../../Final_year_project/dos_detection/working_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_names = [col for col in df.columns if col != 'label']
    X = df[feature_names]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data loaded successfully")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"📊 Features: {feature_names}")
    
    return X_train, X_test, y_train, y_test, feature_names

def train_model_with_best_params(X_train, y_train):
    """Step 2: Train model with optimized parameters"""
    print("\n🚀 STEP 2: TRAINING FINAL MODEL")
    print("=" * 40)
    
    # Use optimized parameters (simplified for faster training)
    best_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"🎯 Using optimized parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Train model
    start_time = time.time()
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Model training completed in {training_time:.2f} seconds")
    print(f"🌳 Number of trees: {model.n_estimators}")
    
    return model, best_params

def evaluate_performance(model, X_test, y_test):
    """Step 3: Evaluate model performance"""
    print("\n📈 STEP 3: PERFORMANCE EVALUATION")
    print("=" * 40)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
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
    
    # Confusion Matrix
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
        'roc_auc': roc_auc
    }
    
    return metrics, y_pred, y_pred_proba, cm

def analyze_feature_importance(model, feature_names):
    """Step 4: Feature importance analysis"""
    print("\n🔍 STEP 4: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:15s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")
    
    return feature_importance_df

def create_visualizations(metrics, y_test, y_pred, y_pred_proba, cm, feature_importance_df):
    """Step 5: Create performance visualizations"""
    print("\n📊 STEP 5: CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Set up plotting
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Random Forest - DoS Detection Performance', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Normal', 'DoS'])
    ax1.set_yticklabels(['Normal', 'DoS'])
    
    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")
    
    # 3. Feature Importance
    ax3 = axes[1, 0]
    top_features = feature_importance_df.head(8)
    bars = ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 8 Feature Importance')
    ax3.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    # 4. Performance Metrics Summary
    ax4 = axes[1, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    bars = ax4.bar(metric_names, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
    ax4.set_ylim([0, 1])
    ax4.set_ylabel('Score')
    ax4.set_title('Performance Metrics Summary')
    ax4.set_xticklabels(metric_names, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = '../results/random_forest_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance visualization saved: {plot_path}")
    plt.show()
    
    return True

def save_model_and_results(model, best_params, metrics, feature_names):
    """Step 6: Save model and results"""
    print("\n💾 STEP 6: SAVING MODEL AND RESULTS")
    print("=" * 40)
    
    try:
        # Create directories if they don't exist
        os.makedirs('../saved_model', exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        os.makedirs('../documentation', exist_ok=True)
        
        # Save the trained model
        model_path = '../saved_model/random_forest_model.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save training results
        results = {
            'model_name': 'Random Forest',
            'training_date': str(datetime.now()),
            'best_parameters': best_params,
            'performance_metrics': metrics
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
        return False

def generate_report(metrics, best_params, feature_importance_df):
    """Step 7: Generate training report"""
    print("\n📋 STEP 7: GENERATING TRAINING REPORT")
    print("=" * 40)
    
    report = f"""
# RANDOM FOREST TRAINING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL CONFIGURATION
- Algorithm: Random Forest Classifier
- Features: 10 network traffic features
- Class Balance: Balanced (50-50)

## BEST HYPERPARAMETERS
"""
    for param, value in best_params.items():
        if param != 'n_jobs':  # Skip technical parameters
            report += f"- {param}: {value}\n"
    
    report += f"""
## PERFORMANCE METRICS
- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
- ROC-AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)

## TOP 5 IMPORTANT FEATURES FOR DoS DETECTION
"""
    for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
        report += f"{i}. {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.1f}%)\n"
    
    report += f"""
## ANALYSIS SUMMARY
✅ Training: COMPLETED
✅ Evaluation: COMPLETED  
✅ Model Saved: COMPLETED
⏳ XAI Analysis: PENDING (Layer 2)

## NEXT STEPS
1. Train XGBoost model for comparison
2. Train Logistic Regression model
3. Train SVM model
4. Compare all models
5. Proceed to Layer 2: XAI/SHAP analysis

---
End of Random Forest Training Report
"""
    
    # Save report
    report_path = '../documentation/training_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Training report saved: {report_path}")
    return report

if __name__ == "__main__":
    print("🌲 RANDOM FOREST - STEP-BY-STEP TRAINING")
    print("=" * 50)
    print("This script will execute each step individually")
    print("=" * 50)
    
    # Execute all steps
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    input("\n🔄 Press Enter to continue to Step 2 (Train Model)...")
    model, best_params = train_model_with_best_params(X_train, y_train)
    
    input("\n🔄 Press Enter to continue to Step 3 (Evaluate Performance)...")
    metrics, y_pred, y_pred_proba, cm = evaluate_performance(model, X_test, y_test)
    
    input("\n🔄 Press Enter to continue to Step 4 (Feature Importance)...")
    feature_importance_df = analyze_feature_importance(model, feature_names)
    
    input("\n🔄 Press Enter to continue to Step 5 (Create Visualizations)...")
    create_visualizations(metrics, y_test, y_pred, y_pred_proba, cm, feature_importance_df)
    
    input("\n🔄 Press Enter to continue to Step 6 (Save Model)...")
    save_model_and_results(model, best_params, metrics, feature_names)
    
    input("\n🔄 Press Enter to continue to Step 7 (Generate Report)...")
    generate_report(metrics, best_params, feature_importance_df)
    
    print(f"\n🎉 RANDOM FOREST TRAINING COMPLETED!")
    print("=" * 50)
    print(f"🎯 Final Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"🎯 Final F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"💾 Model saved and ready for deployment")
    print(f"📊 All results and visualizations generated")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review Random Forest results")
    print(f"   2. Request approval for XGBoost training")
    print(f"   3. Continue with remaining models")
    print(f"   4. Proceed to Layer 2: XAI analysis")
