#!/usr/bin/env python3
"""
LAYER 1: RANDOM FOREST TRAINING FOR DoS DETECTION
================================================

Model: Random Forest Classifier
Focus: Training + Performance Evaluation (XAI in Layer 2)
Objective: Establish baseline performance for DoS detection

Author: DoS Detection Research Team
Date: September 17, 2025
Layer: 1 (Training Focus)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
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

class RandomForestTrainer:
    """
    Random Forest training pipeline for DoS detection
    """
    
    def __init__(self):
        """Initialize the trainer"""
        self.model = None
        self.best_params = None
        self.training_results = {
            'model_name': 'Random Forest',
            'training_date': str(datetime.now()),
            'training_duration': None,
            'best_parameters': None,
            'performance_metrics': {},
            'cross_validation': {},
            'feature_importance': {}
        }
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load the preprocessed data"""
        print("📊 LOADING TRAINING DATA")
        print("=" * 50)
        
        try:
            # Load data (assuming setup script was run)
            data_path = "../../../../../../Final_year_project/dos_detection/working_dataset.csv"
            if os.path.exists(data_path):
                # Load and split data
                df = pd.read_csv(data_path)
                self.feature_names = [col for col in df.columns if col != 'label']
                X = df[self.feature_names]
                y = df['label']
                
                # Use the same split as setup (80-20, random_state=42)
                from sklearn.model_selection import train_test_split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                print(f"✅ Data loaded successfully")
                print(f"📊 Training samples: {len(self.X_train)}")
                print(f"📊 Test samples: {len(self.X_test)}")
                print(f"📊 Features: {len(self.feature_names)}")
                print(f"📊 Feature names: {self.feature_names}")
                
                # Check class distribution
                train_dist = self.y_train.value_counts()
                test_dist = self.y_test.value_counts()
                print(f"\n🎯 Training distribution: Normal={train_dist[0]}, DoS={train_dist[1]}")
                print(f"🎯 Test distribution: Normal={test_dist[0]}, DoS={test_dist[1]}")
                
                return True
                
            else:
                print(f"❌ Data file not found: {data_path}")
                print("💡 Please run setup script first: python3 ../../scripts/01_setup_and_data_prep.py")
                return False
                
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using GridSearchCV"""
        print(f"\n🔧 HYPERPARAMETER TUNING")
        print("=" * 50)
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        print(f"🔍 Parameter grid defined:")
        for param, values in param_grid.items():
            print(f"   {param}: {values}")
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid Search with Cross-Validation
        print(f"\n⚙️ Starting Grid Search (5-fold CV)...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',  # Use F1 score for balanced evaluation
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        tuning_time = time.time() - start_time
        
        # Store best parameters
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"\n✅ Hyperparameter tuning completed!")
        print(f"⏱️ Tuning time: {tuning_time:.2f} seconds")
        print(f"🏆 Best CV score (F1): {grid_search.best_score_:.4f}")
        print(f"\n🎯 Best parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        # Store results
        self.training_results['best_parameters'] = self.best_params
        self.training_results['cv_best_score'] = grid_search.best_score_
        self.training_results['tuning_time'] = tuning_time
        
        return True
    
    def train_final_model(self):
        """Train the final model with best parameters"""
        print(f"\n🚀 TRAINING FINAL MODEL")
        print("=" * 50)
        
        start_time = time.time()
        
        # Train the model (already done in grid search, but let's be explicit)
        print(f"⚙️ Training Random Forest with best parameters...")
        self.model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        print(f"✅ Model training completed!")
        print(f"⏱️ Training time: {training_time:.2f} seconds")
        print(f"🌳 Number of trees: {self.model.n_estimators}")
        print(f"📊 Max depth: {self.model.max_depth}")
        
        self.training_results['final_training_time'] = training_time
        self.training_results['training_duration'] = training_time
        
        return True
    
    def evaluate_performance(self):
        """Evaluate model performance on test set"""
        print(f"\n📈 PERFORMANCE EVALUATION")
        print("=" * 50)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"🎯 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        
        # Store metrics
        self.training_results['performance_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Normal', 'DoS Attack']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n🎯 CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"               Normal  DoS")
        print(f"Actual Normal    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       DoS       {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return y_pred, y_pred_proba
    
    def cross_validation_analysis(self):
        """Perform detailed cross-validation analysis"""
        print(f"\n🔄 CROSS-VALIDATION ANALYSIS")
        print("=" * 50)
        
        # 5-fold stratified cross-validation
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Calculate different metrics
        cv_accuracy = cross_val_score(self.model, self.X_train, self.y_train, 
                                    cv=cv_folds, scoring='accuracy')
        cv_precision = cross_val_score(self.model, self.X_train, self.y_train, 
                                     cv=cv_folds, scoring='precision')
        cv_recall = cross_val_score(self.model, self.X_train, self.y_train, 
                                   cv=cv_folds, scoring='recall')
        cv_f1 = cross_val_score(self.model, self.X_train, self.y_train, 
                               cv=cv_folds, scoring='f1')
        
        print(f"📊 5-FOLD CROSS-VALIDATION RESULTS:")
        print(f"   Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"   Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        print(f"   Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        print(f"   F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        # Store CV results
        self.training_results['cross_validation'] = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_precision_mean': cv_precision.mean(),
            'cv_precision_std': cv_precision.std(),
            'cv_recall_mean': cv_recall.mean(),
            'cv_recall_std': cv_recall.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std()
        }
        
        print(f"\n✅ Model shows {'good' if cv_f1.std() < 0.05 else 'moderate'} stability across folds")
        
        return True
    
    def feature_importance_analysis(self):
        """Analyze feature importance (basic for Layer 1)"""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Get feature importance from Random Forest
        importance = self.model.feature_importances_
        
        # Create importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"🏆 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:15s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")
        
        # Store feature importance
        self.training_results['feature_importance'] = {
            'method': 'Random Forest built-in importance',
            'top_features': feature_importance_df.head(10).to_dict('records')
        }
        
        return feature_importance_df
    
    def create_visualizations(self, y_pred, y_pred_proba, feature_importance_df):
        """Create performance visualizations"""
        print(f"\n📊 CREATING VISUALIZATIONS")
        print("=" * 50)
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest - DoS Detection Performance', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_xticklabels(['Normal', 'DoS'])
        ax1.set_yticklabels(['Normal', 'DoS'])
        
        # 2. ROC Curve
        ax2 = axes[0, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
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
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [
            self.training_results['performance_metrics']['accuracy'],
            self.training_results['performance_metrics']['precision'],
            self.training_results['performance_metrics']['recall'],
            self.training_results['performance_metrics']['f1_score'],
            self.training_results['performance_metrics']['roc_auc']
        ]
        
        bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Metrics Summary')
        ax4.set_xticklabels(metrics, rotation=45)
        
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
    
    def save_model_and_results(self):
        """Save the trained model and results"""
        print(f"\n💾 SAVING MODEL AND RESULTS")
        print("=" * 50)
        
        try:
            # Save the trained model
            model_path = '../saved_model/random_forest_model.pkl'
            joblib.dump(self.model, model_path)
            print(f"✅ Model saved: {model_path}")
            
            # Save training results
            results_path = '../results/training_results.json'
            with open(results_path, 'w') as f:
                json.dump(self.training_results, f, indent=4)
            print(f"✅ Training results saved: {results_path}")
            
            # Save feature names
            feature_path = '../saved_model/feature_names.json'
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f)
            print(f"✅ Feature names saved: {feature_path}")
            
            # Save model parameters
            params_path = '../saved_model/model_parameters.json'
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            print(f"✅ Model parameters saved: {params_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving files: {str(e)}")
            return False
    
    def generate_training_report(self):
        """Generate a comprehensive training report"""
        print(f"\n📋 GENERATING TRAINING REPORT")
        print("=" * 50)
        
        report = f"""
# RANDOM FOREST TRAINING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL CONFIGURATION
- Algorithm: Random Forest Classifier
- Training Samples: {len(self.X_train)}
- Test Samples: {len(self.X_test)}
- Features: {len(self.feature_names)}
- Class Balance: Balanced (50-50)

## BEST HYPERPARAMETERS
"""
        for param, value in self.best_params.items():
            report += f"- {param}: {value}\n"
        
        report += f"""
## PERFORMANCE METRICS
- Accuracy: {self.training_results['performance_metrics']['accuracy']:.4f} ({self.training_results['performance_metrics']['accuracy']*100:.2f}%)
- Precision: {self.training_results['performance_metrics']['precision']:.4f} ({self.training_results['performance_metrics']['precision']*100:.2f}%)
- Recall: {self.training_results['performance_metrics']['recall']:.4f} ({self.training_results['performance_metrics']['recall']*100:.2f}%)
- F1-Score: {self.training_results['performance_metrics']['f1_score']:.4f} ({self.training_results['performance_metrics']['f1_score']*100:.2f}%)
- ROC-AUC: {self.training_results['performance_metrics']['roc_auc']:.4f} ({self.training_results['performance_metrics']['roc_auc']*100:.2f}%)

## CROSS-VALIDATION RESULTS (5-Fold)
- CV Accuracy: {self.training_results['cross_validation']['cv_accuracy_mean']:.4f} ± {self.training_results['cross_validation']['cv_accuracy_std']:.4f}
- CV Precision: {self.training_results['cross_validation']['cv_precision_mean']:.4f} ± {self.training_results['cross_validation']['cv_precision_std']:.4f}
- CV Recall: {self.training_results['cross_validation']['cv_recall_mean']:.4f} ± {self.training_results['cross_validation']['cv_recall_std']:.4f}
- CV F1-Score: {self.training_results['cross_validation']['cv_f1_mean']:.4f} ± {self.training_results['cross_validation']['cv_f1_std']:.4f}

## TOP 5 IMPORTANT FEATURES
"""
        for i, feature_info in enumerate(self.training_results['feature_importance']['top_features'][:5], 1):
            report += f"{i}. {feature_info['feature']}: {feature_info['importance']:.4f}\n"
        
        report += f"""
## TRAINING SUMMARY
- Hyperparameter tuning time: {self.training_results.get('tuning_time', 0):.2f} seconds
- Final training time: {self.training_results.get('final_training_time', 0):.2f} seconds
- Total training duration: {self.training_results.get('training_duration', 0):.2f} seconds

## MODEL STATUS
✅ Training: COMPLETED
✅ Evaluation: COMPLETED
✅ Model Saved: COMPLETED
⏳ XAI Analysis: PENDING (Layer 2)

## NEXT STEPS
1. Review performance metrics
2. Compare with other models (XGBoost, Logistic Regression, SVM)
3. Proceed to Layer 2: XAI/SHAP analysis for best performing models
4. Generate final model selection report

---
End of Random Forest Training Report
"""
        
        # Save report
        report_path = '../documentation/training_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Training report saved: {report_path}")
        return report
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("🌲 RANDOM FOREST TRAINING - LAYER 1")
        print("=" * 60)
        print("Focus: Training + Performance Evaluation")
        print("XAI Analysis: Layer 2")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Load data
        if not self.load_data():
            print("❌ Training failed: Could not load data")
            return False
        
        # Step 2: Hyperparameter tuning
        if not self.hyperparameter_tuning():
            print("❌ Training failed: Hyperparameter tuning failed")
            return False
        
        # Step 3: Train final model
        if not self.train_final_model():
            print("❌ Training failed: Model training failed")
            return False
        
        # Step 4: Evaluate performance
        y_pred, y_pred_proba = self.evaluate_performance()
        
        # Step 5: Cross-validation analysis
        if not self.cross_validation_analysis():
            print("❌ Training failed: Cross-validation failed")
            return False
        
        # Step 6: Feature importance analysis
        feature_importance_df = self.feature_importance_analysis()
        
        # Step 7: Create visualizations
        if not self.create_visualizations(y_pred, y_pred_proba, feature_importance_df):
            print("❌ Training failed: Visualization creation failed")
            return False
        
        # Step 8: Save model and results
        if not self.save_model_and_results():
            print("❌ Training failed: Could not save results")
            return False
        
        # Step 9: Generate report
        self.generate_training_report()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 RANDOM FOREST TRAINING COMPLETED!")
        print("=" * 60)
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"🎯 Final F1-Score: {self.training_results['performance_metrics']['f1_score']:.4f}")
        print(f"🎯 Final Accuracy: {self.training_results['performance_metrics']['accuracy']:.4f}")
        print(f"💾 Model saved and ready for deployment")
        print(f"📊 Results and visualizations generated")
        print(f"📋 Complete training report available")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Review Random Forest performance")
        print(f"   2. Train next model: XGBoost")
        print(f"   3. Compare all models in Layer 1")
        print(f"   4. Proceed to Layer 2: XAI analysis")
        
        return True

def main():
    """Main execution function"""
    # Initialize trainer
    trainer = RandomForestTrainer()
    
    # Run complete training
    success = trainer.run_complete_training()
    
    if success:
        print(f"\n✅ Random Forest training completed successfully!")
        print(f"🚀 Ready for next model training!")
    else:
        print(f"\n❌ Random Forest training failed!")

if __name__ == "__main__":
    main()
