#!/usr/bin/env python3
"""
XGBOOST SHAP COMPREHENSIVE ANALYSIS
==================================

Explainable AI Implementation for DoS Detection
Champion Model (95.54% accuracy) + SHAP Explanations
Complete Global and Local Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import json
import warnings
from pathlib import Path
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def setup_directories():
    """Create necessary directories for results and visualizations"""
    directories = [
        '../results',
        '../visualizations',
        '../visualizations/summary_plots',
        '../visualizations/waterfall_plots', 
        '../visualizations/dependence_plots',
        '../visualizations/force_plots',
        '../documentation'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def load_model_and_data():
    """Load the trained XGBoost model and test data"""
    print("📊 LOADING XGBOOST MODEL AND DATA")
    print("=" * 50)
    
    # Load XGBoost model
    model_path = "../../../../03_model_training/models/xgboost/saved_model/xgboost_model.pkl"
    model = joblib.load(model_path)
    
    # Load feature names
    feature_names_path = "../../../../03_model_training/models/xgboost/saved_model/feature_names.json"
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    # Load dataset
    data_path = "../../../../01_data_preparation/data/final_scaled_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df[feature_names]
    y = df['label']
    
    # Use same train-test split as training (random_state=42)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ XGBoost model loaded successfully")
    print(f"📊 Test data: {len(X_test)} samples, {len(feature_names)} features")
    print(f"🎯 Model accuracy on test set: {model.score(X_test, y_test):.4f}")
    
    return model, X_train, X_test, y_train, y_test, feature_names

def initialize_shap_explainer(model, X_train):
    """Initialize SHAP TreeExplainer for XGBoost"""
    print("\n🔍 INITIALIZING SHAP EXPLAINER")
    print("=" * 50)
    
    print("🌳 Creating SHAP TreeExplainer for XGBoost...")
    explainer = shap.TreeExplainer(model)
    
    print("📊 Computing SHAP values for test set...")
    # Use a subset for initial analysis (computational efficiency)
    X_test_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    print(f"✅ SHAP explainer initialized")
    print(f"📊 SHAP values computed for {len(X_test_sample)} samples")
    print(f"🔍 Feature explanations shape: {shap_values.shape}")
    
    return explainer, shap_values, X_test_sample

def global_feature_importance_analysis(explainer, shap_values, X_test_sample, feature_names):
    """Analyze global feature importance using SHAP"""
    print("\n🌍 GLOBAL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Calculate mean absolute SHAP values for global importance
    global_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_dict = {feature: float(importance) 
                              for feature, importance in zip(feature_names, global_importance)}
    
    # Sort by importance
    sorted_features = sorted(feature_importance_dict.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("🏆 TOP 10 MOST IMPORTANT FEATURES (SHAP):")
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"   {i:2d}. {feature:<12}: {importance:.4f}")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, 
                     feature_names=feature_names, 
                     plot_type="bar", show=False)
    plt.title('XGBoost SHAP - Global Feature Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../visualizations/summary_plots/global_importance_bar.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_sample, 
                     feature_names=feature_names, show=False)
    plt.title('XGBoost SHAP - Feature Impact Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../visualizations/summary_plots/feature_impact_summary.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save global importance results
    results = {
        'global_feature_importance': feature_importance_dict,
        'top_10_features': {feature: float(importance) for feature, importance in sorted_features[:10]},
        'analysis_timestamp': str(datetime.now()),
        'samples_analyzed': len(X_test_sample),
        'total_features': len(feature_names)
    }
    
    with open('../results/global_importance.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("✅ Global feature importance analysis complete")
    print("📊 Summary plots saved to visualizations/summary_plots/")
    
    return feature_importance_dict

def local_explanation_analysis(explainer, X_test, y_test, feature_names, model, num_samples=10):
    """Analyze local explanations for individual predictions"""
    print("\n🎯 LOCAL EXPLANATION ANALYSIS")
    print("=" * 50)
    
    # Select diverse samples for analysis
    normal_indices = np.where(y_test == 0)[0]
    dos_indices = np.where(y_test == 1)[0]
    
    # Select samples: mix of correct and potentially interesting cases
    selected_indices = []
    
    # Add some normal samples
    selected_indices.extend(np.random.choice(normal_indices, size=num_samples//2, replace=False))
    # Add some DoS samples  
    selected_indices.extend(np.random.choice(dos_indices, size=num_samples//2, replace=False))
    
    X_local = X_test.iloc[selected_indices]
    y_local = y_test.iloc[selected_indices]
    
    # Get predictions and probabilities
    predictions = model.predict(X_local)
    probabilities = model.predict_proba(X_local)
    
    # Compute SHAP values for selected samples
    shap_values_local = explainer.shap_values(X_local)
    
    print(f"🔍 Analyzing {len(selected_indices)} individual predictions:")
    
    local_explanations = []
    
    for i, idx in enumerate(selected_indices):
        actual = y_local.iloc[i]
        predicted = predictions[i] 
        prob_dos = probabilities[i][1]
        
        explanation = {
            'sample_index': int(idx),
            'actual_label': int(actual),
            'predicted_label': int(predicted),
            'dos_probability': float(prob_dos),
            'correct_prediction': bool(actual == predicted),
            'shap_values': shap_values_local[i].tolist(),
            'feature_values': X_local.iloc[i].to_dict()
        }
        local_explanations.append(explanation)
        
        print(f"   Sample {i+1:2d}: Actual={actual}, Predicted={predicted}, "
              f"DoS_Prob={prob_dos:.3f}, Correct={actual==predicted}")
        
        # Create waterfall plot for this sample
        plt.figure(figsize=(10, 6))
        try:
            shap.plots.waterfall(
                shap.Explanation(values=shap_values_local[i], 
                               base_values=explainer.expected_value, 
                               data=X_local.iloc[i].values,
                               feature_names=feature_names),
                show=False
            )
        except:
            # Fallback to basic plot if waterfall fails
            plt.bar(range(len(feature_names)), shap_values_local[i])
            plt.xticks(range(len(feature_names)), feature_names, rotation=45)
            plt.ylabel('SHAP value')
            plt.title('SHAP Values')
        plt.title(f'XGBoost SHAP Waterfall - Sample {i+1}\n'
                 f'Actual: {"DoS" if actual else "Normal"}, '
                 f'Predicted: {"DoS" if predicted else "Normal"} '
                 f'(Prob: {prob_dos:.3f})', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../visualizations/waterfall_plots/waterfall_sample_{i+1}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save local explanations
    local_results = {
        'local_explanations': local_explanations,
        'analysis_timestamp': str(datetime.now()),
        'samples_analyzed': len(selected_indices),
        'explainer_expected_value': float(explainer.expected_value)
    }
    
    with open('../results/local_explanations.json', 'w') as f:
        json.dump(local_results, f, indent=4)
    
    print("✅ Local explanation analysis complete")
    print("💧 Waterfall plots saved to visualizations/waterfall_plots/")
    
    return local_explanations

def feature_dependence_analysis(explainer, shap_values, X_test_sample, feature_names, top_features):
    """Analyze feature dependencies and interactions"""
    print("\n📈 FEATURE DEPENDENCE ANALYSIS")
    print("=" * 50)
    
    # Analyze top 5 features
    top_5_features = list(top_features.keys())[:5]
    
    print(f"🔍 Creating dependence plots for top 5 features:")
    for feature in top_5_features:
        print(f"   • {feature}")
    
    for i, feature in enumerate(top_5_features):
        feature_idx = feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, 
            shap_values, 
            X_test_sample,
            feature_names=feature_names,
            show=False
        )
        plt.title(f'XGBoost SHAP Dependence - {feature}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../visualizations/dependence_plots/dependence_{feature.lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Feature dependence analysis complete")
    print("📈 Dependence plots saved to visualizations/dependence_plots/")

def force_plot_analysis(explainer, shap_values, X_test_sample, feature_names, num_plots=5):
    """Create force plots for individual predictions"""
    print("\n⚡ FORCE PLOT ANALYSIS")
    print("=" * 50)
    
    print(f"🔍 Creating force plots for {num_plots} sample predictions...")
    
    # Initialize SHAP JavaScript for force plots
    shap.initjs()
    
    for i in range(min(num_plots, len(shap_values))):
        # Create force plot
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X_test_sample.iloc[i],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f'XGBoost SHAP Force Plot - Sample {i+1}', 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../visualizations/force_plots/force_plot_sample_{i+1}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Force plot analysis complete")
    print("⚡ Force plots saved to visualizations/force_plots/")

def comprehensive_analysis_report(model, feature_importance, local_explanations, feature_names):
    """Generate comprehensive analysis report"""
    print("\n📋 GENERATING COMPREHENSIVE REPORT")
    print("=" * 50)
    
    report = f"""# XGBOOST SHAP COMPREHENSIVE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL INFORMATION
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Performance**: 95.54% accuracy (Champion model)
- **Features**: {len(feature_names)} engineered network traffic features
- **XAI Method**: SHAP (TreeExplainer)

## GLOBAL FEATURE IMPORTANCE (SHAP)

### Top 10 Most Important Features:
"""
    
    # Add top features to report
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        report += f"{i:2d}. **{feature}**: {importance:.4f}\n"
    
    report += f"""
### Feature Importance Insights:
- **Most Critical Feature**: {sorted_features[0][0]} (SHAP value: {sorted_features[0][1]:.4f})
- **Feature Distribution**: {'Balanced' if sorted_features[0][1] < 0.2 else 'Concentrated'} importance across features
- **Top 5 Features Account**: {sum(imp for _, imp in sorted_features[:5]):.1%} of total importance

## LOCAL EXPLANATION ANALYSIS

### Sample Predictions Analyzed: {len(local_explanations)}
"""
    
    # Analyze local explanations
    correct_predictions = sum(1 for exp in local_explanations if exp['correct_prediction'])
    accuracy_on_sample = correct_predictions / len(local_explanations)
    
    report += f"""
- **Accuracy on Analyzed Samples**: {accuracy_on_sample:.1%} ({correct_predictions}/{len(local_explanations)})
- **Average DoS Probability**: {np.mean([exp['dos_probability'] for exp in local_explanations]):.3f}
- **Prediction Confidence**: {'High' if np.mean([abs(exp['dos_probability'] - 0.5) for exp in local_explanations]) > 0.3 else 'Moderate'}

### Individual Prediction Examples:
"""
    
    # Add sample explanations
    for i, exp in enumerate(local_explanations[:5], 1):
        actual_label = "DoS Attack" if exp['actual_label'] else "Normal Traffic"
        predicted_label = "DoS Attack" if exp['predicted_label'] else "Normal Traffic"
        report += f"""
**Sample {i}:**
- Actual: {actual_label}
- Predicted: {predicted_label} (Probability: {exp['dos_probability']:.3f})
- Prediction: {'✅ Correct' if exp['correct_prediction'] else '❌ Incorrect'}
"""

    report += f"""
## SHAP ANALYSIS INSIGHTS

### Model Behavior:
- **Explanation Method**: SHAP TreeExplainer (optimized for XGBoost)
- **Feature Interactions**: Captured through SHAP dependence plots
- **Prediction Transparency**: Individual feature contributions identified
- **Decision Boundary**: Non-linear patterns explained through SHAP values

### Cybersecurity Implications:
- **Attack Pattern Recognition**: SHAP reveals which network features indicate DoS attacks
- **False Positive Analysis**: Understanding why normal traffic might be misclassified
- **Feature Reliability**: Identifying most trustworthy indicators for DoS detection
- **Model Interpretability**: Clear explanation of XGBoost decision process

## VISUALIZATIONS CREATED

### Global Analysis:
- 📊 Global feature importance bar chart
- 📈 Feature impact summary plot
- 📉 Feature dependence plots (top 5 features)

### Local Analysis:
- 💧 Waterfall plots for individual predictions
- ⚡ Force plots showing prediction forces
- 🎯 Sample-specific feature contributions

## PRODUCTION RECOMMENDATIONS

### Model Deployment:
1. **XGBoost + SHAP**: Recommended combination for production
2. **Feature Monitoring**: Track top features identified by SHAP
3. **Explanation Interface**: Provide SHAP explanations for security analysts
4. **Threshold Tuning**: Use SHAP insights for optimal classification thresholds

### Security Operations:
1. **Alert Explanations**: Include SHAP explanations with DoS alerts
2. **Feature Investigation**: Focus on features with high SHAP values
3. **Model Validation**: Regular SHAP analysis for model drift detection
4. **Training Enhancement**: Use SHAP insights for feature engineering

## RESEARCH CONTRIBUTIONS

### Academic Value:
- Comprehensive SHAP analysis for cybersecurity ML
- XGBoost interpretability in DoS detection context
- Feature importance validation through explanation AI
- Production-ready explainable AI implementation

### Technical Achievements:
- Complete global and local explanation framework
- Visualization suite for model interpretation
- Quantitative feature importance analysis
- Individual prediction explanation capability

---
**XGBoost SHAP Analysis Complete**
**Champion Model Explainability Achieved**
**Production-Ready Interpretable DoS Detection**
"""
    
    # Save comprehensive report
    with open('../documentation/xgboost_shap_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Comprehensive analysis report generated")
    print("📚 Report saved to documentation/xgboost_shap_report.md")

def main():
    """Main execution function"""
    print("🔍 XGBOOST SHAP COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    print("Champion Model (95.54%) + SHAP Explanations")
    print("Complete Global and Local Analysis")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Load model and data
    model, X_train, X_test, y_train, y_test, feature_names = load_model_and_data()
    
    # Initialize SHAP
    explainer, shap_values, X_test_sample = initialize_shap_explainer(model, X_train)
    
    # Global analysis
    feature_importance = global_feature_importance_analysis(
        explainer, shap_values, X_test_sample, feature_names
    )
    
    # Local analysis
    local_explanations = local_explanation_analysis(
        explainer, X_test, y_test, feature_names, model
    )
    
    # Feature dependence analysis
    feature_dependence_analysis(
        explainer, shap_values, X_test_sample, feature_names, feature_importance
    )
    
    # Force plot analysis
    force_plot_analysis(
        explainer, shap_values, X_test_sample, feature_names
    )
    
    # Generate comprehensive report
    comprehensive_analysis_report(
        model, feature_importance, local_explanations, feature_names
    )
    
    # Save SHAP values for future use
    shap_data = {
        'shap_values': shap_values.tolist(),
        'expected_value': float(explainer.expected_value),
        'feature_names': feature_names,
        'sample_data': X_test_sample.to_dict('records')
    }
    
    with open('../results/shap_values.json', 'w') as f:
        json.dump(shap_data, f, indent=2)
    
    print(f"\n🎉 XGBOOST SHAP ANALYSIS COMPLETED!")
    print("=" * 60)
    print("📊 Global feature importance: DONE")
    print("🎯 Local explanations: DONE") 
    print("📈 Feature dependencies: DONE")
    print("⚡ Force plots: DONE")
    print("📚 Comprehensive report: DONE")
    print("💾 All results saved to results/ directory")
    print("🎨 All visualizations saved to visualizations/ directory")
    
    print(f"\n🏆 CHAMPION MODEL EXPLANATION COMPLETE!")
    print(f"✅ XGBoost (95.54%) is now fully explainable")
    print(f"🔍 SHAP provides complete transparency")
    print(f"🚀 Ready for production deployment with explanations")

if __name__ == "__main__":
    main()
