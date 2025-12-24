#!/usr/bin/env python3
"""
OPTIMIZED MLP (MULTI-LAYER PERCEPTRON) TRAINING
==============================================

Neural Network Model Training for DoS Detection
Completes 5-Model Comparison Framework
Layer 1: Training + Performance Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
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
    print("🧠 MLP (NEURAL NETWORK) TRAINING")
    print("=" * 45)
    print("Completing 5-Model Comparison Framework")
    print("=" * 45)
    
    # Step 1: Load Data
    print("\n📊 STEP 1: LOADING DATA")
    print("=" * 35)
    
    data_path = "../../../../../../Final_year_project/dos_detection/working_dataset.csv"
    df = pd.read_csv(data_path)
    
    feature_names = [col for col in df.columns if col != 'label']
    X = df[feature_names]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test")
    print(f"📊 Features: {len(feature_names)} network traffic features")
    
    # Step 2: Feature Scaling (Critical for Neural Networks)
    print("\n📏 STEP 2: FEATURE SCALING")
    print("=" * 35)
    
    print("🔧 Applying StandardScaler (essential for neural networks)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check scaling
    print(f"✅ Features scaled to mean≈0, std≈1")
    print(f"📊 Training data range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    # Step 3: Neural Network Architecture Design
    print("\n🧠 STEP 3: NEURAL NETWORK ARCHITECTURE")
    print("=" * 35)
    
    print("🏗️ Designing optimized MLP architecture for DoS detection:")
    print("   • Input Layer: 10 features (network traffic)")
    print("   • Hidden Layers: Multiple configurations tested")
    print("   • Output Layer: 2 classes (Normal/DoS)")
    print("   • Activation: ReLU (proven effective)")
    print("   • Solver: Adam (adaptive learning rate)")
    
    # Step 4: Hyperparameter Optimization
    print("\n🔧 STEP 4: NEURAL NETWORK OPTIMIZATION")
    print("=" * 35)
    
    # Optimized parameter grid for neural networks
    param_grid = {
        'hidden_layer_sizes': [
            (50,),          # Single layer
            (100,),         # Single layer (larger)
            (100, 50),      # Two layers
            (150, 75, 25)   # Three layers (deep)
        ],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
        'learning_rate_init': [0.001, 0.01]
    }
    
    total_combinations = 4 * 2 * 3 * 2  # 48 combinations
    print(f"🔍 Neural network configurations: {total_combinations}")
    print(f"📊 Architecture variations:")
    for i, layers in enumerate(param_grid['hidden_layer_sizes'], 1):
        print(f"   {i}. Hidden layers: {layers}")
    
    mlp_model = MLPClassifier(
        max_iter=500,  # Sufficient for convergence
        random_state=42,
        early_stopping=True,  # Prevent overfitting
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    print(f"\n⚙️ Starting neural network optimization...")
    tuning_start = time.time()
    
    grid_search = GridSearchCV(
        estimator=mlp_model,
        param_grid=param_grid,
        cv=3,  # 3-fold CV for efficiency
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    tuning_time = time.time() - tuning_start
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n✅ Neural network optimization completed in {tuning_time:.2f} seconds")
    print(f"🏆 Best CV F1-Score: {best_score:.4f}")
    print(f"🧠 Optimal architecture:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Step 5: Train Final Neural Network
    print("\n🚀 STEP 5: FINAL NEURAL NETWORK TRAINING")
    print("=" * 35)
    
    final_params = {
        **best_params,
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
    }
    
    start_time = time.time()
    final_model = MLPClassifier(**final_params)
    final_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Neural network trained in {training_time:.2f} seconds")
    print(f"🧠 Architecture: {final_model.hidden_layer_sizes}")
    print(f"📊 Layers: {len(final_model.hidden_layer_sizes) + 2} total (input + hidden + output)")
    print(f"🔥 Activation: {final_model.activation}")
    print(f"⚡ Solver: {final_model.solver}")
    print(f"📈 Iterations: {final_model.n_iter_}")
    
    # Step 6: Performance Evaluation
    print("\n📈 STEP 6: NEURAL NETWORK PERFORMANCE")
    print("=" * 35)
    
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 NEURAL NETWORK PERFORMANCE:")
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
    
    # Step 7: Neural Network Analysis
    print("\n🧠 STEP 7: NEURAL NETWORK ANALYSIS")
    print("=" * 35)
    
    # Analyze network structure
    n_layers = len(final_model.coefs_)
    total_params = sum(coef.size for coef in final_model.coefs_) + sum(intercept.size for intercept in final_model.intercepts_)
    
    print(f"🏗️ NETWORK ARCHITECTURE ANALYSIS:")
    print(f"   Total layers: {n_layers} (hidden layers)")
    print(f"   Total parameters: {total_params}")
    print(f"   Input → Hidden: {final_model.coefs_[0].shape}")
    
    if len(final_model.coefs_) > 1:
        for i in range(1, len(final_model.coefs_)):
            if i == len(final_model.coefs_) - 1:
                print(f"   Hidden → Output: {final_model.coefs_[i].shape}")
            else:
                print(f"   Hidden{i} → Hidden{i+1}: {final_model.coefs_[i].shape}")
    
    print(f"\n🔍 TRAINING CONVERGENCE:")
    print(f"   Converged in: {final_model.n_iter_} iterations")
    print(f"   Early stopping: {'Yes' if final_model.n_iter_ < 500 else 'No'}")
    
    # Step 8: Create Visualizations
    print("\n📊 STEP 8: CREATING VISUALIZATIONS")
    print("=" * 35)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MLP Neural Network - DoS Detection Performance', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Normal', 'DoS'])
    ax1.set_yticklabels(['Normal', 'DoS'])
    
    # ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_plot = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='red', lw=2, 
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
    
    bars = ax3.bar(metric_names, values, color=['crimson', 'red', 'darkred', 'indianred', 'lightcoral'])
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 5-Model Comparison
    ax4 = axes[1, 1]
    model_names = ['XGBoost', 'Random Forest', 'MLP', 'SVM', 'Logistic Reg']
    model_accuracies = [95.54, 95.29, accuracy*100, 90.04, 78.18]
    
    colors = ['lightgreen', 'skyblue', 'red', 'orange', 'mediumpurple']
    bars = ax4.bar(model_names, model_accuracies, color=colors)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('5-Model Comparison')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, model_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('../results', exist_ok=True)
    plot_path = '../results/mlp_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Neural network visualization saved: {plot_path}")
    plt.close()
    
    # Step 9: Save Model and Results
    print("\n💾 STEP 9: SAVING NEURAL NETWORK")
    print("=" * 35)
    
    try:
        # Create directories
        os.makedirs('../saved_model', exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        os.makedirs('../documentation', exist_ok=True)
        
        # Save model and scaler
        joblib.dump(final_model, '../saved_model/mlp_model.pkl')
        joblib.dump(scaler, '../saved_model/feature_scaler.pkl')
        
        # Save results
        results = {
            'model_name': 'MLP (Multi-Layer Perceptron)',
            'training_date': str(datetime.now()),
            'best_parameters': final_params,
            'performance_metrics': metrics,
            'training_time_seconds': training_time,
            'tuning_time_seconds': tuning_time,
            'network_architecture': {
                'hidden_layers': str(final_model.hidden_layer_sizes),
                'total_parameters': total_params,
                'activation_function': final_model.activation,
                'solver': final_model.solver,
                'iterations_to_convergence': int(final_model.n_iter_)
            }
        }
        
        with open('../results/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        with open('../saved_model/feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        print(f"✅ Neural network model and results saved")
        
    except Exception as e:
        print(f"❌ Error saving: {str(e)}")
        return False
    
    # Step 10: Generate Comprehensive Report
    print("\n📋 STEP 10: NEURAL NETWORK REPORT")
    print("=" * 35)
    
    report = f"""# MLP NEURAL NETWORK TRAINING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## NEURAL NETWORK CONFIGURATION
- Algorithm: Multi-Layer Perceptron (MLP)
- Architecture: {final_model.hidden_layer_sizes}
- Total Parameters: {total_params}
- Activation Function: {final_model.activation}
- Solver: {final_model.solver}
- Training Time: {training_time:.2f} seconds
- Tuning Time: {tuning_time:.2f} seconds
- Convergence: {final_model.n_iter_} iterations

## PERFORMANCE METRICS
- Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
- Precision: {precision:.4f} ({precision*100:.2f}%)
- Recall: {recall:.4f} ({recall*100:.2f}%)
- F1-Score: {f1:.4f} ({f1*100:.2f}%)
- ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)
- CV F1-Score: {best_score:.4f} ({best_score*100:.2f}%)

## NEURAL NETWORK INSIGHTS
- Hidden Layer Configuration: {final_model.hidden_layer_sizes}
- Network Depth: {len(final_model.hidden_layer_sizes)} hidden layers
- Parameter Count: {total_params} (learnable weights + biases)
- Early Stopping: {'Activated' if final_model.n_iter_ < 500 else 'Not needed'}

## 5-MODEL COMPARISON COMPLETE
1. XGBoost: 95.54% (Tree-based leader)
2. Random Forest: 95.29% (Tree-based strong)
3. MLP: {accuracy*100:.2f}% (Neural network)
4. SVM: 90.04% (Kernel method)
5. Logistic Regression: 78.18% (Linear baseline)

## NEURAL NETWORK ANALYSIS
- Performance Position: {'3rd' if accuracy > 0.90 else '4th or 5th'} place in 5-model comparison
- Neural vs Tree-based: {'Competitive' if accuracy > 0.93 else 'Lower performance'} compared to XGBoost/Random Forest
- Neural vs Traditional: {'Superior' if accuracy > 0.85 else 'Similar'} to SVM and Logistic Regression
- Complexity Trade-off: Higher complexity than traditional ML with {'justified' if accuracy > 0.92 else 'limited'} performance gain

## RESEARCH CONTRIBUTIONS
- Completes comprehensive ML paradigm comparison
- Demonstrates neural network performance on tabular cybersecurity data
- Establishes tree-based model superiority for DoS detection
- Provides neural network baseline for future research

## NEXT STEPS
- Layer 2: XAI/SHAP analysis for top performing models
- Production deployment with optimal model selection
- Research documentation for publication

---
MLP Neural Network Training Successfully Completed
5-Model Comparison Framework Complete
"""
    
    with open('../documentation/training_report.md', 'w') as f:
        f.write(report)
    
    print(f"✅ Comprehensive neural network report saved")
    
    # Final Summary
    print(f"\n🎉 MLP NEURAL NETWORK TRAINING COMPLETED!")
    print("=" * 50)
    print(f"🧠 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🧠 Final F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"🧠 Final ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"⏱️  Total Time: {training_time + tuning_time:.2f} seconds")
    print(f"🏗️  Architecture: {final_model.hidden_layer_sizes}")
    print(f"📊 Parameters: {total_params}")
    
    print(f"\n🏆 COMPLETE 5-MODEL RANKING:")
    print(f"   1. 🥇 XGBoost: 95.54% (Tree-based champion)")
    print(f"   2. 🥈 Random Forest: 95.29% (Tree-based strong)")
    print(f"   3. 🥉 MLP: {accuracy*100:.2f}% (Neural network)")
    print(f"   4. 🏅 SVM: 90.04% (Kernel method)")
    print(f"   5. 🏅 Logistic Regression: 78.18% (Linear baseline)")
    
    print(f"\n🚀 RESEARCH MILESTONE: 5-MODEL COMPARISON COMPLETE")
    print(f"✅ All major ML paradigms covered:")
    print(f"   • Tree-based: Random Forest, XGBoost")
    print(f"   • Neural Networks: MLP")
    print(f"   • Kernel Methods: SVM")
    print(f"   • Linear Models: Logistic Regression")
    
    print(f"\n🎯 READY FOR LAYER 2: XAI ANALYSIS")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ MLP training completed successfully!")
        print(f"🎯 5-model comparison framework complete!")
        print(f"🚀 Ready for explainable AI implementation!")
    else:
        print(f"\n❌ Neural network training encountered issues.")
