#!/usr/bin/env python3
"""
ADASYN Synthetic Data Validation Tool - Comprehensive Analysis
============================================================

🎯 PURPOSE: Comprehensive validation of ADASYN-generated synthetic data
📊 INPUT: Original data (final_scaled_dataset.csv) + Enhanced data (adasyn_enhanced_dataset.csv)
📈 OUTPUT: Complete validation report with quality score and recommendations

🔧 VALIDATION TIERS:
1. Statistical Distribution Validation (30 points)
2. Correlation Structure Validation (25 points)  
3. Domain Constraint Validation (20 points)
4. ML Performance Validation (20 points)
5. Visual Validation (5 points)

⏱️ ESTIMATED TIME: ~20 minutes
📊 OUTPUT: Comprehensive validation report + quality score (0-100)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, mannwhitneyu, anderson_ksamp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json

class ADASYNValidator:
    """Comprehensive ADASYN synthetic data validation class"""
    
    def __init__(self, original_file, enhanced_file):
        """Initialize validator with data files"""
        self.original_file = original_file
        self.enhanced_file = enhanced_file
        self.validation_results = {}
        self.quality_score = 0
        
    def load_and_prepare_data(self):
        """Load and prepare data for validation"""
        print("🔍 LOADING DATA FOR VALIDATION")
        print("=" * 60)
        
        # Load datasets
        self.original_df = pd.read_csv(self.original_file)
        self.enhanced_df = pd.read_csv(self.enhanced_file)
        
        print(f"✅ Original dataset: {self.original_df.shape}")
        print(f"✅ Enhanced dataset: {self.enhanced_df.shape}")
        
        # Extract features (exclude label)
        self.features = [col for col in self.original_df.columns if col != 'label']
        print(f"✅ Features to validate: {len(self.features)}")
        
        # Separate original and synthetic data
        original_size = len(self.original_df)
        self.original_features = self.original_df[self.features]
        self.enhanced_features = self.enhanced_df[self.features]
        
        # Extract only synthetic samples for specific validation
        self.synthetic_features = self.enhanced_features.iloc[original_size:]
        
        print(f"✅ Original samples: {len(self.original_features):,}")
        print(f"✅ Enhanced samples: {len(self.enhanced_features):,}")
        print(f"✅ Synthetic samples: {len(self.synthetic_features):,}")
        
    def validate_statistical_distributions(self):
        """Tier 1: Statistical distribution validation (30 points)"""
        print(f"\n📊 TIER 1: STATISTICAL DISTRIBUTION VALIDATION")
        print("=" * 60)
        
        tier1_results = {}
        tier1_score = 0
        
        # Test each feature's distribution
        feature_tests = {}
        for feature in self.features:
            orig_data = self.original_features[feature].dropna()
            synth_data = self.synthetic_features[feature].dropna()
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(orig_data, synth_data)
            
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(orig_data, synth_data, alternative='two-sided')
            
            # Descriptive statistics comparison
            orig_mean, orig_std = orig_data.mean(), orig_data.std()
            synth_mean, synth_std = synth_data.mean(), synth_data.std()
            
            mean_diff_pct = abs(orig_mean - synth_mean) / abs(orig_mean) * 100 if orig_mean != 0 else 0
            std_ratio = synth_std / orig_std if orig_std != 0 else 1
            
            feature_tests[feature] = {
                'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 'pass': ks_p > 0.05},
                'mw_test': {'statistic': mw_stat, 'p_value': mw_p, 'pass': mw_p > 0.05},
                'mean_diff_pct': mean_diff_pct,
                'std_ratio': std_ratio,
                'stats_good': mean_diff_pct < 10 and 0.8 < std_ratio < 1.2
            }
        
        # Calculate Tier 1 score
        ks_pass_rate = sum(1 for f in feature_tests.values() if f['ks_test']['pass']) / len(self.features)
        mw_pass_rate = sum(1 for f in feature_tests.values() if f['mw_test']['pass']) / len(self.features)
        stats_pass_rate = sum(1 for f in feature_tests.values() if f['stats_good']) / len(self.features)
        
        tier1_score = (ks_pass_rate * 10) + (mw_pass_rate * 10) + (stats_pass_rate * 10)
        
        # Summary
        print(f"📈 Distribution Test Results:")
        print(f"   • K-S test pass rate: {ks_pass_rate:.1%} ({ks_pass_rate * 10:.1f}/10 points)")
        print(f"   • Mann-Whitney pass rate: {mw_pass_rate:.1%} ({mw_pass_rate * 10:.1f}/10 points)")
        print(f"   • Descriptive stats similarity: {stats_pass_rate:.1%} ({stats_pass_rate * 10:.1f}/10 points)")
        print(f"🎯 Tier 1 Score: {tier1_score:.1f}/30 points")
        
        tier1_results = {
            'feature_tests': feature_tests,
            'ks_pass_rate': ks_pass_rate,
            'mw_pass_rate': mw_pass_rate,
            'stats_pass_rate': stats_pass_rate,
            'score': tier1_score
        }
        
        self.validation_results['tier1_distributions'] = tier1_results
        return tier1_score
    
    def validate_correlation_structure(self):
        """Tier 2: Correlation structure validation (25 points)"""
        print(f"\n📊 TIER 2: CORRELATION STRUCTURE VALIDATION")
        print("=" * 60)
        
        # Calculate correlation matrices
        orig_corr = self.original_features.corr()
        synth_corr = self.synthetic_features.corr()
        enhanced_corr = self.enhanced_features.corr()
        
        # Correlation preservation analysis
        corr_diff_synth = abs(orig_corr - synth_corr)
        corr_diff_enhanced = abs(orig_corr - enhanced_corr)
        
        # Summary metrics
        mean_diff_synth = corr_diff_synth.mean().mean()
        max_diff_synth = corr_diff_synth.max().max()
        
        mean_diff_enhanced = corr_diff_enhanced.mean().mean()
        max_diff_enhanced = corr_diff_enhanced.max().max()
        
        # Preservation percentages
        synth_preservation = (1 - mean_diff_synth) * 100
        enhanced_preservation = (1 - mean_diff_enhanced) * 100
        
        # Calculate Tier 2 score
        # Score based on enhanced dataset correlation preservation
        if enhanced_preservation >= 90:
            corr_score = 15
        elif enhanced_preservation >= 80:
            corr_score = 12
        elif enhanced_preservation >= 70:
            corr_score = 8
        else:
            corr_score = 4
        
        # Bonus for synthetic-only preservation
        if synth_preservation >= 80:
            synth_bonus = 10
        elif synth_preservation >= 60:
            synth_bonus = 7
        else:
            synth_bonus = 3
        
        tier2_score = corr_score + synth_bonus
        
        print(f"📈 Correlation Analysis Results:")
        print(f"   • Synthetic samples preservation: {synth_preservation:.1f}% ({synth_bonus}/10 points)")
        print(f"   • Enhanced dataset preservation: {enhanced_preservation:.1f}% ({corr_score}/15 points)")
        print(f"   • Mean correlation difference: {mean_diff_enhanced:.3f}")
        print(f"   • Max correlation difference: {max_diff_enhanced:.3f}")
        print(f"🎯 Tier 2 Score: {tier2_score:.1f}/25 points")
        
        tier2_results = {
            'synthetic_preservation': synth_preservation,
            'enhanced_preservation': enhanced_preservation,
            'mean_difference': mean_diff_enhanced,
            'max_difference': max_diff_enhanced,
            'score': tier2_score
        }
        
        self.validation_results['tier2_correlations'] = tier2_results
        return tier2_score
    
    def validate_domain_constraints(self):
        """Tier 3: Domain-specific constraint validation (20 points)"""
        print(f"\n📊 TIER 3: DOMAIN CONSTRAINT VALIDATION")
        print("=" * 60)
        
        violations = []
        constraint_score = 20  # Start with full points, deduct for violations
        
        # Check synthetic data specifically
        synth_data = self.synthetic_features
        
        # Protocol constraints
        if 'proto' in synth_data.columns:
            proto_violations = (~synth_data['proto'].between(0, 2)).sum()
            if proto_violations > 0:
                violations.append(f"Protocol violations: {proto_violations} samples")
                constraint_score -= 3
        
        # Positive value constraints
        positive_features = ['sbytes', 'dbytes', 'sload', 'dload', 'rate']
        for feature in positive_features:
            if feature in synth_data.columns:
                negative_count = (synth_data[feature] < 0).sum()
                if negative_count > 0:
                    violations.append(f"Negative {feature}: {negative_count} samples")
                    constraint_score -= 2
        
        # Range violations (values outside original data bounds)
        for feature in self.features:
            orig_min, orig_max = self.original_features[feature].min(), self.original_features[feature].max()
            out_of_bounds = ((synth_data[feature] < orig_min) | (synth_data[feature] > orig_max)).sum()
            if out_of_bounds > len(synth_data) * 0.05:  # More than 5% out of bounds
                violations.append(f"{feature} out of bounds: {out_of_bounds} samples ({out_of_bounds/len(synth_data):.1%})")
                constraint_score -= 2
        
        # Logical consistency checks
        if 'sbytes' in synth_data.columns and 'rate' in synth_data.columns:
            # Check for unrealistic rate-bytes combinations
            high_rate_threshold = synth_data['rate'].quantile(0.9)
            low_bytes_threshold = synth_data['sbytes'].quantile(0.1)
            inconsistent = ((synth_data['rate'] > high_rate_threshold) & 
                          (synth_data['sbytes'] < low_bytes_threshold)).sum()
            if inconsistent > len(synth_data) * 0.05:
                violations.append(f"Rate-bytes inconsistency: {inconsistent} samples")
                constraint_score -= 2
        
        # Ensure score doesn't go below 0
        constraint_score = max(0, constraint_score)
        
        print(f"📈 Domain Constraint Results:")
        print(f"   • Total violations found: {len(violations)}")
        for violation in violations:
            print(f"     - {violation}")
        print(f"   • Constraint compliance: {100 - (20 - constraint_score) * 5:.1f}%")
        print(f"🎯 Tier 3 Score: {constraint_score:.1f}/20 points")
        
        tier3_results = {
            'violations': violations,
            'violation_count': len(violations),
            'compliance_percentage': 100 - (20 - constraint_score) * 5,
            'score': constraint_score
        }
        
        self.validation_results['tier3_constraints'] = tier3_results
        return constraint_score
    
    def validate_ml_performance(self):
        """Tier 4: Machine learning performance validation (20 points)"""
        print(f"\n📊 TIER 4: ML PERFORMANCE VALIDATION")
        print("=" * 60)
        
        # Prepare data for ML testing
        X_orig = self.original_df[self.features]
        y_orig = self.original_df['label']
        
        X_enhanced = self.enhanced_df[self.features]
        y_enhanced = self.enhanced_df['label']
        
        # Split data for testing
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
        )
        
        X_enh_train, X_enh_test, y_enh_train, y_enh_test = train_test_split(
            X_enhanced, y_enhanced, test_size=0.2, random_state=42, stratify=y_enhanced
        )
        
        # Train models
        rf_orig = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_enhanced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        print(f"   • Training baseline model (original data)...")
        rf_orig.fit(X_orig_train, y_orig_train)
        
        print(f"   • Training enhanced model (ADASYN data)...")
        rf_enhanced.fit(X_enh_train, y_enh_train)
        
        # Evaluate on original test set (fair comparison)
        y_pred_orig = rf_orig.predict(X_orig_test)
        y_pred_enhanced = rf_enhanced.predict(X_orig_test)
        
        # Calculate metrics
        metrics_orig = {
            'accuracy': accuracy_score(y_orig_test, y_pred_orig),
            'f1': f1_score(y_orig_test, y_pred_orig),
            'precision': precision_score(y_orig_test, y_pred_orig),
            'recall': recall_score(y_orig_test, y_pred_orig)
        }
        
        metrics_enhanced = {
            'accuracy': accuracy_score(y_orig_test, y_pred_enhanced),
            'f1': f1_score(y_orig_test, y_pred_enhanced),
            'precision': precision_score(y_orig_test, y_pred_enhanced),
            'recall': recall_score(y_orig_test, y_pred_enhanced)
        }
        
        # Calculate improvements
        improvements = {
            metric: metrics_enhanced[metric] - metrics_orig[metric]
            for metric in metrics_orig.keys()
        }
        
        # Score calculation
        performance_score = 0
        
        # Performance improvement scoring (15 points max)
        improved_metrics = sum(1 for imp in improvements.values() if imp >= 0)
        significant_improvements = sum(1 for imp in improvements.values() if imp >= 0.01)
        
        performance_score += min(15, improved_metrics * 3 + significant_improvements * 2)
        
        # Overfitting check (5 points max)
        # Cross-validation consistency
        cv_scores_orig = cross_val_score(rf_orig, X_orig, y_orig, cv=3, scoring='f1')
        cv_scores_enhanced = cross_val_score(rf_enhanced, X_enhanced, y_enhanced, cv=3, scoring='f1')
        
        cv_improvement = cv_scores_enhanced.mean() - cv_scores_orig.mean()
        if cv_improvement >= 0:
            overfitting_score = 5
        elif cv_improvement >= -0.02:
            overfitting_score = 3
        else:
            overfitting_score = 1
        
        performance_score += overfitting_score
        
        print(f"📈 ML Performance Results:")
        print(f"   • Original model performance:")
        for metric, value in metrics_orig.items():
            print(f"     - {metric.capitalize()}: {value:.3f}")
        
        print(f"   • Enhanced model performance:")
        for metric, value in metrics_enhanced.items():
            improvement = improvements[metric]
            arrow = "↗️" if improvement > 0 else "↘️" if improvement < 0 else "➡️"
            print(f"     - {metric.capitalize()}: {value:.3f} ({improvement:+.3f}) {arrow}")
        
        print(f"   • Cross-validation F1: {cv_scores_orig.mean():.3f} → {cv_scores_enhanced.mean():.3f}")
        print(f"🎯 Tier 4 Score: {performance_score:.1f}/20 points")
        
        tier4_results = {
            'original_metrics': metrics_orig,
            'enhanced_metrics': metrics_enhanced,
            'improvements': improvements,
            'cv_original': cv_scores_orig.mean(),
            'cv_enhanced': cv_scores_enhanced.mean(),
            'score': performance_score
        }
        
        self.validation_results['tier4_performance'] = tier4_results
        return performance_score
    
    def validate_visual_structure(self):
        """Tier 5: Visual and structural validation (5 points)"""
        print(f"\n📊 TIER 5: VISUAL STRUCTURE VALIDATION")
        print("=" * 60)
        
        # PCA analysis
        pca = PCA(n_components=2, random_state=42)
        
        # Fit on original data
        orig_pca = pca.fit_transform(self.original_features)
        synth_pca = pca.transform(self.synthetic_features)
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        
        # Check if synthetic samples are in reasonable PCA space
        orig_pca_bounds = {
            'pc1_min': orig_pca[:, 0].min(), 'pc1_max': orig_pca[:, 0].max(),
            'pc2_min': orig_pca[:, 1].min(), 'pc2_max': orig_pca[:, 1].max()
        }
        
        # Allow 10% expansion of bounds for synthetic data
        expansion = 0.1
        synth_in_bounds = (
            (synth_pca[:, 0] >= orig_pca_bounds['pc1_min'] * (1 + expansion)) &
            (synth_pca[:, 0] <= orig_pca_bounds['pc1_max'] * (1 + expansion)) &
            (synth_pca[:, 1] >= orig_pca_bounds['pc2_min'] * (1 + expansion)) &
            (synth_pca[:, 1] <= orig_pca_bounds['pc2_max'] * (1 + expansion))
        ).mean()
        
        # Score calculation
        if synth_in_bounds >= 0.95:
            visual_score = 5
        elif synth_in_bounds >= 0.85:
            visual_score = 4
        elif synth_in_bounds >= 0.75:
            visual_score = 3
        else:
            visual_score = 2
        
        print(f"📈 Visual Structure Results:")
        print(f"   • PCA explained variance: {explained_var:.1%}")
        print(f"   • Synthetic samples in reasonable bounds: {synth_in_bounds:.1%}")
        print(f"🎯 Tier 5 Score: {visual_score:.1f}/5 points")
        
        tier5_results = {
            'pca_explained_variance': explained_var,
            'synthetic_in_bounds': synth_in_bounds,
            'score': visual_score
        }
        
        self.validation_results['tier5_visual'] = tier5_results
        return visual_score
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report"""
        print(f"\n🎉 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 60)
        
        # Calculate total score
        total_score = sum([
            self.validation_results['tier1_distributions']['score'],
            self.validation_results['tier2_correlations']['score'],
            self.validation_results['tier3_constraints']['score'],
            self.validation_results['tier4_performance']['score'],
            self.validation_results['tier5_visual']['score']
        ])
        
        # Quality assessment
        if total_score >= 90:
            quality_grade = "EXCELLENT"
            recommendation = "✅ USE synthetic data - exceptional quality"
        elif total_score >= 80:
            quality_grade = "GOOD"
            recommendation = "✅ USE synthetic data - good quality"
        elif total_score >= 70:
            quality_grade = "ACCEPTABLE"
            recommendation = "⚠️ USE with caution - monitor performance"
        elif total_score >= 60:
            quality_grade = "NEEDS IMPROVEMENT"
            recommendation = "❌ IMPROVE synthetic data generation"
        else:
            quality_grade = "POOR"
            recommendation = "❌ REJECT synthetic data - use original only"
        
        print(f"📊 FINAL VALIDATION SCORES:")
        print(f"   • Tier 1 (Distributions): {self.validation_results['tier1_distributions']['score']:.1f}/30")
        print(f"   • Tier 2 (Correlations): {self.validation_results['tier2_correlations']['score']:.1f}/25")
        print(f"   • Tier 3 (Constraints): {self.validation_results['tier3_constraints']['score']:.1f}/20")
        print(f"   • Tier 4 (ML Performance): {self.validation_results['tier4_performance']['score']:.1f}/20")
        print(f"   • Tier 5 (Visual): {self.validation_results['tier5_visual']['score']:.1f}/5")
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   • Total Score: {total_score:.1f}/100")
        print(f"   • Quality Grade: {quality_grade}")
        print(f"   • Recommendation: {recommendation}")
        
        # Save detailed results
        self.validation_results['summary'] = {
            'total_score': total_score,
            'quality_grade': quality_grade,
            'recommendation': recommendation
        }
        
        return total_score, quality_grade, recommendation

def main():
    """Main validation function"""
    
    print("🔍 ADASYN SYNTHETIC DATA VALIDATION")
    print("=" * 60)
    print("🎯 Goal: Comprehensive validation of ADASYN-generated synthetic data")
    print("📊 Method: 5-tier validation framework (100-point scale)")
    print("🏆 Output: Quality score and usage recommendation")
    
    # Set up paths
    data_dir = Path("../data")
    original_file = data_dir / "final_scaled_dataset.csv"
    enhanced_file = data_dir / "adasyn_enhanced_dataset.csv"
    
    # Verify files exist
    if not original_file.exists():
        print(f"❌ ERROR: Original file not found: {original_file}")
        return False
    
    if not enhanced_file.exists():
        print(f"❌ ERROR: Enhanced file not found: {enhanced_file}")
        return False
    
    try:
        # Initialize validator
        validator = ADASYNValidator(original_file, enhanced_file)
        
        # Load and prepare data
        validator.load_and_prepare_data()
        
        # Run all validation tiers
        tier1_score = validator.validate_statistical_distributions()
        tier2_score = validator.validate_correlation_structure()
        tier3_score = validator.validate_domain_constraints()
        tier4_score = validator.validate_ml_performance()
        tier5_score = validator.validate_visual_structure()
        
        # Generate comprehensive report
        total_score, quality_grade, recommendation = validator.generate_comprehensive_report()
        
        # Save validation results
        results_dir = Path("../results")
        results_file = results_dir / "adasyn_validation_results.json"
        
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_types(validator.validation_results), f, indent=2)
        
        print(f"\n💾 VALIDATION RESULTS SAVED:")
        print(f"   • Detailed results: {results_file}")
        print(f"   • Validation complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during validation:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 ADASYN validation completed successfully!")
    else:
        print(f"\n❌ ADASYN validation failed. Please check the error messages above.")
