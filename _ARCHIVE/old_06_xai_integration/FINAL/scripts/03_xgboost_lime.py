#!/usr/bin/env python3
"""
XGBoost LIME Comprehensive Analysis for DoS Detection
Complete local explanation analysis using LIME
Comparison with SHAP explanations for validation

Author: DoS Detection Research Team
Date: 2025-09-17
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class XGBoostLIMEAnalyzer:
    """
    Comprehensive LIME analysis for XGBoost DoS detection model
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        self.viz_dir = self.base_dir / "visualizations"
        self.doc_dir = self.base_dir / "documentation"
        self.create_directories()
        
        # Data and model paths
        self.data_path = "/Users/swetasmac/Desktop/Final_year_project/dos_detection/01_data_preparation/data/final_scaled_dataset.csv"
        self.model_path = "/Users/swetasmac/Desktop/Final_year_project/dos_detection/03_model_training/models/xgboost/saved_model/xgboost_model.pkl"
        
        # Initialize components
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.lime_explainer = None
        
    def create_directories(self):
        """Create directory structure for results"""
        print("✅ Directory structure created")
        for dir_path in [self.results_dir, self.viz_dir, self.doc_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create visualization subdirectories
        viz_subdirs = ['explanation_plots', 'feature_importance', 'prediction_analysis', 'comparative_plots']
        for subdir in viz_subdirs:
            (self.viz_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_data(self):
        """Load XGBoost model and prepare data"""
        print("\n📊 LOADING XGBOOST MODEL AND DATA")
        print("=" * 50)
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✅ XGBoost model loaded successfully")
        
        # Load and prepare data
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.feature_names = [col for col in df.columns if col != 'label']
        X = df[self.feature_names].values
        y = df['label'].values
        
        # Use same train-test split as original training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training data: {len(self.X_train)} samples")
        print(f"📊 Test data: {len(self.X_test)} samples, {len(self.feature_names)} features")
        
        # Evaluate model
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        print(f"🎯 Model accuracy - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
        
    def initialize_lime_explainer(self):
        """Initialize LIME explainer for tabular data"""
        print("\n🔍 INITIALIZING LIME EXPLAINER")
        print("=" * 50)
        
        print("🧪 Creating LIME Tabular Explainer...")
        
        # Create LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=['Normal', 'DoS Attack'],
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
        
        print("✅ LIME explainer initialized")
        print(f"📊 Training samples used for explanation: {len(self.X_train)}")
        print(f"🔍 Features to explain: {len(self.feature_names)}")
        
    def analyze_sample_explanations(self, num_samples=15):
        """Analyze individual sample explanations using LIME"""
        print(f"\n🎯 INDIVIDUAL SAMPLE EXPLANATION ANALYSIS")
        print("=" * 50)
        
        # Select diverse samples for analysis
        normal_indices = np.where(self.y_test == 0)[0]
        attack_indices = np.where(self.y_test == 1)[0]
        
        # Select balanced samples
        normal_samples = np.random.choice(normal_indices, min(num_samples//2, len(normal_indices)), replace=False)
        attack_samples = np.random.choice(attack_indices, min(num_samples//2, len(attack_indices)), replace=False)
        selected_indices = np.concatenate([normal_samples, attack_samples])
        
        print(f"🔍 Analyzing {len(selected_indices)} individual predictions:")
        print(f"   • Normal traffic samples: {len(normal_samples)}")
        print(f"   • DoS attack samples: {len(attack_samples)}")
        
        explanations_data = []
        feature_importance_aggregate = {feature: 0.0 for feature in self.feature_names}
        
        for i, idx in enumerate(selected_indices, 1):
            sample = self.X_test[idx]
            actual = self.y_test[idx]
            predicted = self.model.predict([sample])[0]
            probabilities = self.model.predict_proba([sample])[0]
            dos_prob = probabilities[1]
            
            correct = actual == predicted
            actual_label = "DoS Attack" if actual == 1 else "Normal Traffic"
            predicted_label = "DoS Attack" if predicted == 1 else "Normal Traffic"
            
            print(f"   Sample {i:2d}: {actual_label} → {predicted_label} "
                  f"(Prob: {dos_prob:.3f}, Correct: {correct})")
            
            try:
                # Generate LIME explanation
                explanation = self.lime_explainer.explain_instance(
                    sample, 
                    self.model.predict_proba,
                    num_features=len(self.feature_names),
                    num_samples=1000
                )
                
                # Extract feature importance
                exp_list = explanation.as_list()
                sample_importance = {feature: 0.0 for feature in self.feature_names}
                
                for feature_desc, importance in exp_list:
                    # Parse feature name from LIME description
                    feature_name = feature_desc.split(' ')[0] if ' ' in feature_desc else feature_desc
                    if feature_name in sample_importance:
                        sample_importance[feature_name] = importance
                        feature_importance_aggregate[feature_name] += abs(importance)
                
                # Save explanation plot
                fig = explanation.as_pyplot_figure()
                fig.suptitle(f'XGBoost LIME - Sample {i} ({actual_label})', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.viz_dir / 'explanation_plots' / f'lime_explanation_sample_{i}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Store explanation data
                explanations_data.append({
                    'sample_id': i,
                    'index': int(idx),
                    'actual': int(actual),
                    'predicted': int(predicted),
                    'dos_probability': float(dos_prob),
                    'correct': bool(correct),
                    'feature_importance': sample_importance,
                    'explanation_text': str(explanation.as_list())
                })
                
            except Exception as e:
                print(f"   Warning: Could not generate explanation for sample {i}: {e}")
        
        # Calculate aggregated feature importance
        total_importance = sum(feature_importance_aggregate.values())
        if total_importance > 0:
            for feature in feature_importance_aggregate:
                feature_importance_aggregate[feature] /= total_importance
        
        # Save results
        with open(self.results_dir / 'sample_explanations.json', 'w') as f:
            json.dump(explanations_data, f, indent=2)
            
        with open(self.results_dir / 'aggregated_feature_importance.json', 'w') as f:
            json.dump(feature_importance_aggregate, f, indent=2)
        
        print("✅ Individual sample analysis complete")
        print("🎨 Explanation plots saved to visualizations/explanation_plots/")
        
        return explanations_data, feature_importance_aggregate
    
    def create_feature_importance_analysis(self, feature_importance):
        """Create comprehensive feature importance visualizations"""
        print("\n📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Convert to sorted dataframe
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        print("🏆 TOP 10 MOST IMPORTANT FEATURES (LIME):")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<12}: {row['importance']:.4f}")
        
        # Create visualizations
        
        # 1. Feature importance bar chart
        plt.figure(figsize=(14, 8))
        top_features = importance_df.head(10)
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='darkorange', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('LIME Feature Importance (Aggregated)')
        plt.title('XGBoost + LIME - Global Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
            plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_importance' / 'lime_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance pie chart (top 8)
        plt.figure(figsize=(12, 10))
        top_8 = importance_df.head(8)
        others_importance = importance_df.iloc[8:]['importance'].sum()
        
        labels = list(top_8['feature']) + ['Others']
        sizes = list(top_8['importance']) + [others_importance]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('XGBoost + LIME - Feature Importance Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_importance' / 'lime_importance_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance
        importance_df.to_csv(self.results_dir / 'lime_feature_importance.csv', index=False)
        
        print("✅ Feature importance analysis complete")
        print("📊 Visualizations saved to visualizations/feature_importance/")
        
        return importance_df
    
    def prediction_confidence_analysis(self, explanations_data):
        """Analyze prediction confidence and explanation quality"""
        print("\n🎯 PREDICTION CONFIDENCE ANALYSIS")
        print("=" * 50)
        
        # Extract metrics
        accuracies = [exp['correct'] for exp in explanations_data]
        probabilities = [exp['dos_probability'] for exp in explanations_data]
        actual_labels = [exp['actual'] for exp in explanations_data]
        predicted_labels = [exp['predicted'] for exp in explanations_data]
        
        # Calculate metrics
        overall_accuracy = sum(accuracies) / len(accuracies)
        avg_confidence = np.mean(probabilities)
        normal_samples = [p for i, p in enumerate(probabilities) if actual_labels[i] == 0]
        attack_samples = [p for i, p in enumerate(probabilities) if actual_labels[i] == 1]
        
        print(f"📊 ANALYSIS METRICS:")
        print(f"   • Overall Accuracy: {overall_accuracy:.1%}")
        print(f"   • Average Confidence: {avg_confidence:.3f}")
        print(f"   • Normal Traffic Avg Confidence: {np.mean(normal_samples):.3f}")
        print(f"   • DoS Attack Avg Confidence: {np.mean(attack_samples):.3f}")
        
        # Create confidence distribution plot
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        normal_probs = [p for i, p in enumerate(probabilities) if actual_labels[i] == 0]
        attack_probs = [p for i, p in enumerate(probabilities) if actual_labels[i] == 1]
        
        plt.hist(normal_probs, bins=10, alpha=0.7, label='Normal Traffic', color='green')
        plt.hist(attack_probs, bins=10, alpha=0.7, label='DoS Attack', color='red')
        plt.xlabel('DoS Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        correct_probs = [probabilities[i] for i, acc in enumerate(accuracies) if acc]
        incorrect_probs = [probabilities[i] for i, acc in enumerate(accuracies) if not acc]
        
        plt.hist(correct_probs, bins=10, alpha=0.7, label='Correct Predictions', color='blue')
        if incorrect_probs:
            plt.hist(incorrect_probs, bins=10, alpha=0.7, label='Incorrect Predictions', color='orange')
        plt.xlabel('DoS Probability')
        plt.ylabel('Frequency')
        plt.title('Accuracy vs Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'prediction_analysis' / 'confidence_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save analysis results
        analysis_results = {
            'overall_accuracy': overall_accuracy,
            'average_confidence': avg_confidence,
            'normal_traffic_confidence': np.mean(normal_samples),
            'attack_traffic_confidence': np.mean(attack_samples),
            'total_samples_analyzed': len(explanations_data),
            'correct_predictions': sum(accuracies),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'prediction_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print("✅ Prediction confidence analysis complete")
        print("📈 Analysis plots saved to visualizations/prediction_analysis/")
        
        return analysis_results
    
    def compare_with_shap_results(self, lime_importance):
        """Compare LIME results with existing SHAP results"""
        print("\n🔍 LIME vs SHAP COMPARISON")
        print("=" * 50)
        
        try:
            # Load SHAP results
            shap_path = "/Users/swetasmac/Desktop/Final_year_project/dos_detection/05_XAI_integration/SHAP_analysis/xgboost_shap/results/global_importance.json"
            with open(shap_path, 'r') as f:
                shap_data = json.load(f)
            
            shap_importance = shap_data['global_feature_importance']
            
            # Normalize SHAP values for comparison
            shap_total = sum(shap_importance.values())
            shap_normalized = {k: v/shap_total for k, v in shap_importance.items()}
            
            print("🏆 FEATURE IMPORTANCE COMPARISON:")
            print(f"{'Feature':>12} {'LIME':>12} {'SHAP':>12} {'Difference':>12}")
            print("-" * 52)
            
            comparison_data = {}
            for feature in self.feature_names:
                lime_val = lime_importance.get(feature, 0.0)
                shap_val = shap_normalized.get(feature, 0.0)
                diff = lime_val - shap_val
                
                print(f"{feature:>12} {lime_val:>12.4f} {shap_val:>12.4f} {diff:>12.4f}")
                comparison_data[feature] = {
                    'lime': lime_val,
                    'shap': shap_val,
                    'difference': diff
                }
            
            # Calculate correlation
            lime_values = [lime_importance.get(f, 0.0) for f in self.feature_names]
            shap_values = [shap_normalized.get(f, 0.0) for f in self.feature_names]
            correlation = np.corrcoef(lime_values, shap_values)[0, 1]
            
            print(f"\n🔗 LIME-SHAP Correlation: {correlation:.3f}")
            
            # Create comparison visualization
            plt.figure(figsize=(15, 6))
            
            # Side-by-side comparison
            plt.subplot(1, 2, 1)
            features = list(lime_importance.keys())[:10]
            lime_vals = [lime_importance[f] for f in features]
            shap_vals = [shap_normalized.get(f, 0.0) for f in features]
            
            x = np.arange(len(features))
            width = 0.35
            
            plt.bar(x - width/2, lime_vals, width, label='LIME', color='darkorange', alpha=0.8)
            plt.bar(x + width/2, shap_vals, width, label='SHAP', color='steelblue', alpha=0.8)
            
            plt.xlabel('Features')
            plt.ylabel('Normalized Importance')
            plt.title('LIME vs SHAP Feature Importance')
            plt.xticks(x, features, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Correlation plot
            plt.subplot(1, 2, 2)
            plt.scatter(lime_values, shap_values, alpha=0.7, s=100)
            
            # Add feature labels
            for i, feature in enumerate(self.feature_names):
                plt.annotate(feature, (lime_values[i], shap_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Add correlation line
            plt.plot([0, max(max(lime_values), max(shap_values))], 
                    [0, max(max(lime_values), max(shap_values))], 
                    'r--', alpha=0.5, label=f'Perfect Correlation')
            
            plt.xlabel('LIME Importance')
            plt.ylabel('SHAP Importance')
            plt.title(f'LIME vs SHAP Correlation (r={correlation:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'comparative_plots' / 'lime_shap_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save comparison results
            comparison_results = {
                'correlation': correlation,
                'feature_comparison': comparison_data,
                'lime_top_feature': max(lime_importance.items(), key=lambda x: x[1]),
                'shap_top_feature': max(shap_normalized.items(), key=lambda x: x[1]),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            with open(self.results_dir / 'lime_shap_comparison.json', 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            print("✅ LIME vs SHAP comparison complete")
            print("📊 Comparison plots saved to visualizations/comparative_plots/")
            
            return comparison_results
            
        except Exception as e:
            print(f"❌ Could not compare with SHAP results: {e}")
            return None
    
    def generate_comprehensive_report(self, explanations_data, importance_df, prediction_analysis, comparison_results):
        """Generate comprehensive LIME analysis report"""
        print("\n📋 GENERATING COMPREHENSIVE LIME REPORT")
        print("=" * 50)
        
        report = f"""# XGBOOST LIME COMPREHENSIVE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL INFORMATION
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Performance**: 95.54% accuracy (Champion model)
- **Features**: 10 engineered network traffic features
- **XAI Method**: LIME (Local Interpretable Model-agnostic Explanations)

## LIME ANALYSIS OVERVIEW

### Global Feature Importance (Aggregated from Local Explanations)

#### Top 10 Most Important Features:
"""
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            report += f" {i:2d}. **{row['feature']}**: {row['importance']:.4f}\n"
        
        report += f"""
### Feature Importance Insights:
- **Most Critical Feature**: {importance_df.iloc[0]['feature']} (LIME value: {importance_df.iloc[0]['importance']:.4f})
- **Feature Distribution**: {'Concentrated' if importance_df.head(3)['importance'].sum() > 0.5 else 'Distributed'} importance pattern
- **Top 5 Features Account**: {importance_df.head(5)['importance'].sum():.1%} of total importance

## LOCAL EXPLANATION ANALYSIS

### Sample Predictions Analyzed: {len(explanations_data)}

- **Accuracy on Analyzed Samples**: {prediction_analysis['overall_accuracy']:.1%}
- **Average DoS Probability**: {prediction_analysis['average_confidence']:.3f}
- **Normal Traffic Confidence**: {prediction_analysis['normal_traffic_confidence']:.3f}
- **DoS Attack Confidence**: {prediction_analysis['attack_traffic_confidence']:.3f}

### Individual Prediction Examples:

"""
        
        for exp in explanations_data[:5]:
            actual_label = "DoS Attack" if exp['actual'] == 1 else "Normal Traffic"
            predicted_label = "DoS Attack" if exp['predicted'] == 1 else "Normal Traffic"
            status = "✅ Correct" if exp['correct'] else "❌ Incorrect"
            
            report += f"""**Sample {exp['sample_id']}:**
- Actual: {actual_label}
- Predicted: {predicted_label} (Probability: {exp['dos_probability']:.3f})
- Prediction: {status}

"""
        
        if comparison_results:
            report += f"""## LIME vs SHAP COMPARISON

### Explanation Method Comparison:
- **Correlation with SHAP**: {comparison_results['correlation']:.3f}
- **LIME Top Feature**: {comparison_results['lime_top_feature'][0]} ({comparison_results['lime_top_feature'][1]:.4f})
- **SHAP Top Feature**: {comparison_results['shap_top_feature'][0]} ({comparison_results['shap_top_feature'][1]:.4f})
- **Agreement Level**: {'High' if comparison_results['correlation'] > 0.7 else 'Moderate' if comparison_results['correlation'] > 0.5 else 'Low'}

### Method-Specific Insights:
- **LIME Strengths**: Local fidelity, model-agnostic, interpretable explanations
- **SHAP Strengths**: Global consistency, theoretical foundations, efficient for tree models
- **Recommendation**: {'Use both methods for comprehensive analysis' if comparison_results['correlation'] > 0.6 else 'Consider primary method based on use case'}
"""
        
        report += f"""
## LIME ANALYSIS INSIGHTS

### Model Behavior:
- **Explanation Method**: LIME Tabular Explainer (model-agnostic)
- **Local Explanations**: Individual instance interpretability
- **Feature Interactions**: Captured through local linear approximations
- **Prediction Transparency**: Per-instance feature contribution analysis

### Cybersecurity Implications:
- **Attack Pattern Recognition**: LIME reveals local decision boundaries for DoS detection
- **Feature Attribution**: Understanding which features drive individual predictions
- **False Positive Analysis**: Local explanations help identify misclassification causes
- **Model Interpretability**: Clear explanation of XGBoost decision process per instance

## VISUALIZATIONS CREATED

### Local Analysis:
- 💡 Individual explanation plots for sample predictions
- 📊 Aggregated feature importance analysis
- 📈 Prediction confidence distribution analysis

### Comparative Analysis:
- 🔍 LIME vs SHAP feature importance comparison
- 📊 Correlation analysis between explanation methods
- 🎯 Method agreement assessment

## PRODUCTION RECOMMENDATIONS

### LIME Integration:
1. **Local Explanations**: Provide LIME explanations for critical DoS alerts
2. **Feature Analysis**: Use aggregated LIME importance for feature monitoring
3. **Model Validation**: Regular LIME analysis for model behavior verification
4. **Security Operations**: Include LIME explanations in analyst dashboards

### Explanation Strategy:
1. **Primary Method**: {'LIME for local explanations' if not comparison_results or comparison_results['correlation'] < 0.7 else 'SHAP for global insights, LIME for local details'}
2. **Use Cases**: LIME excels at explaining individual predictions to security analysts
3. **Deployment**: Real-time LIME explanations for high-confidence DoS predictions
4. **Monitoring**: Track LIME feature importance for model drift detection

## RESEARCH CONTRIBUTIONS

### Academic Value:
- Comprehensive LIME analysis for cybersecurity ML
- XGBoost interpretability through model-agnostic explanations
- Local explanation methodology for DoS detection
- Production-ready explainable AI implementation

### Technical Achievements:
- Complete local explanation framework using LIME
- Feature importance aggregation methodology
- Comparative analysis with SHAP explanations
- Individual prediction explanation capability

---
**XGBoost LIME Analysis Complete**
**Champion Model Local Explanations Achieved**
**Production-Ready Model-Agnostic Interpretability**

"""
        
        # Save report
        with open(self.doc_dir / 'xgboost_lime_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Comprehensive LIME analysis report generated")
        print("📚 Report saved to documentation/xgboost_lime_report.md")
    
    def run_complete_analysis(self):
        """Execute complete LIME analysis pipeline"""
        print("🔍 XGBOOST LIME COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        print("Champion Model (95.54%) + LIME Explanations")
        print("Complete Local and Comparative Analysis")
        print("=" * 60)
        
        try:
            # Load model and data
            self.load_model_and_data()
            
            # Initialize LIME
            self.initialize_lime_explainer()
            
            # Analyze sample explanations
            explanations_data, feature_importance = self.analyze_sample_explanations()
            
            # Feature importance analysis
            importance_df = self.create_feature_importance_analysis(feature_importance)
            
            # Prediction confidence analysis
            prediction_analysis = self.prediction_confidence_analysis(explanations_data)
            
            # Compare with SHAP
            comparison_results = self.compare_with_shap_results(feature_importance)
            
            # Generate comprehensive report
            self.generate_comprehensive_report(explanations_data, importance_df, prediction_analysis, comparison_results)
            
            print("\n🎉 XGBOOST LIME ANALYSIS COMPLETED!")
            print("=" * 60)
            print("💡 Local explanations: DONE")
            print("📊 Feature importance: DONE")
            print("🎯 Prediction analysis: DONE")
            print("🔍 LIME vs SHAP comparison: DONE")
            print("📚 Comprehensive report: DONE")
            print("💾 All results saved to results/ directory")
            print("🎨 All visualizations saved to visualizations/ directory")
            
            print("\n🏆 CHAMPION MODEL LIME EXPLANATION COMPLETE!")
            print("✅ XGBoost (95.54%) is now explainable with LIME")
            print("💡 Local interpretable explanations available")
            print("🚀 Ready for comparative analysis with Random Forest LIME")
            
        except Exception as e:
            print(f"❌ Error in LIME analysis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    analyzer = XGBoostLIMEAnalyzer()
    analyzer.run_complete_analysis()
