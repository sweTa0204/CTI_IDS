#!/usr/bin/env python3
"""
DoS Detection Model Training - Day 1: Setup & Data Preparation
==============================================================

This script sets up the environment and prepares data for model training.
This is the foundation for our comprehensive model comparison study.

Author: DoS Detection Research Team
Date: September 17, 2025
Phase: Model Training - Day 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class DoSModelTrainingSetup:
    """
    Setup class for DoS detection model training pipeline
    """
    
    def __init__(self, data_path="../working_dataset.csv"):
        """
        Initialize the training setup
        
        Args:
            data_path (str): Path to the working dataset
        """
        self.data_path = data_path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results_log = {
            'setup_date': str(datetime.now()),
            'dataset_info': {},
            'train_test_split': {},
            'models_to_compare': [
                'Random Forest (XAI Optimized)',
                'XGBoost (SHAP Ready)',
                'Logistic Regression (XAI Baseline)',
                'Support Vector Machine'
            ],
            'xai_objectives': [
                'SHAP analysis for model interpretability',
                'Feature importance ranking',
                'Decision explanation capability',
                'Explainable AI integration'
            ]
        }
        
    def load_and_inspect_data(self):
        """
        Load the dataset and perform initial inspection
        """
        print("🔍 LOADING AND INSPECTING DATASET")
        print("=" * 50)
        
        try:
            # Load dataset
            self.dataset = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully from: {self.data_path}")
            
            # Basic information
            print(f"\n📊 Dataset Shape: {self.dataset.shape}")
            print(f"📊 Features: {self.dataset.shape[1] - 1}")  # Excluding label
            print(f"📊 Samples: {self.dataset.shape[0]}")
            
            # Feature names (excluding label)
            self.feature_names = [col for col in self.dataset.columns if col != 'label']
            print(f"\n🏷️  Feature Names: {self.feature_names}")
            
            # Target distribution
            label_counts = self.dataset['label'].value_counts()
            label_percentages = self.dataset['label'].value_counts(normalize=True) * 100
            
            print(f"\n🎯 Class Distribution:")
            print(f"   Normal Traffic (0): {label_counts[0]} samples ({label_percentages[0]:.1f}%)")
            print(f"   DoS Attacks (1): {label_counts[1]} samples ({label_percentages[1]:.1f}%)")
            
            # Check for missing values
            missing_values = self.dataset.isnull().sum().sum()
            print(f"\n🔍 Missing Values: {missing_values}")
            
            # Data types
            print(f"\n📋 Data Types:")
            for col in self.dataset.columns:
                print(f"   {col}: {self.dataset[col].dtype}")
            
            # Save dataset info to results log
            self.results_log['dataset_info'] = {
                'shape': self.dataset.shape,
                'features': len(self.feature_names),
                'samples': self.dataset.shape[0],
                'class_distribution': {
                    'normal': int(label_counts[0]),
                    'dos_attacks': int(label_counts[1]),
                    'balance': 'Perfect' if abs(label_counts[0] - label_counts[1]) <= 1 else 'Imbalanced'
                },
                'missing_values': int(missing_values),
                'feature_names': self.feature_names
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create stratified train-test split
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        """
        print(f"\n🔄 CREATING TRAIN-TEST SPLIT")
        print("=" * 50)
        
        try:
            # Separate features and target
            X = self.dataset[self.feature_names]
            y = self.dataset['label']
            
            # Create stratified split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"✅ Train-Test Split Created Successfully")
            print(f"\n📊 Split Information:")
            print(f"   Training Set: {self.X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
            print(f"   Test Set: {self.X_test.shape[0]} samples ({test_size*100:.0f}%)")
            
            # Check class distribution in splits
            train_dist = self.y_train.value_counts()
            test_dist = self.y_test.value_counts()
            
            print(f"\n🎯 Training Set Distribution:")
            print(f"   Normal (0): {train_dist[0]} samples ({train_dist[0]/len(self.y_train)*100:.1f}%)")
            print(f"   DoS (1): {train_dist[1]} samples ({train_dist[1]/len(self.y_train)*100:.1f}%)")
            
            print(f"\n🎯 Test Set Distribution:")
            print(f"   Normal (0): {test_dist[0]} samples ({test_dist[0]/len(self.y_test)*100:.1f}%)")
            print(f"   DoS (1): {test_dist[1]} samples ({test_dist[1]/len(self.y_test)*100:.1f}%)")
            
            # Save split info to results log
            self.results_log['train_test_split'] = {
                'test_size': test_size,
                'random_state': random_state,
                'train_samples': int(self.X_train.shape[0]),
                'test_samples': int(self.X_test.shape[0]),
                'train_distribution': {
                    'normal': int(train_dist[0]),
                    'dos_attacks': int(train_dist[1])
                },
                'test_distribution': {
                    'normal': int(test_dist[0]),
                    'dos_attacks': int(test_dist[1])
                }
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating train-test split: {str(e)}")
            return False
    
    def create_data_visualization(self, save_plots=True):
        """
        Create visualizations of the dataset
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        print(f"\n📈 CREATING DATA VISUALIZATIONS")
        print("=" * 50)
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('DoS Detection Dataset - Exploratory Analysis', fontsize=16, fontweight='bold')
            
            # 1. Class Distribution
            ax1 = axes[0, 0]
            class_counts = self.dataset['label'].value_counts()
            labels = ['Normal Traffic', 'DoS Attacks']
            colors = ['#2E86C1', '#E74C3C']
            ax1.pie(class_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
            ax1.set_title('Class Distribution', fontweight='bold')
            
            # 2. Feature Correlation Heatmap (sample of features)
            ax2 = axes[0, 1]
            # Select top 8 features for heatmap (to keep it readable)
            sample_features = self.feature_names[:8] + ['label']
            correlation_matrix = self.dataset[sample_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax2, fmt='.2f')
            ax2.set_title('Feature Correlation Matrix (Sample)', fontweight='bold')
            
            # 3. Train-Test Split Visualization
            ax3 = axes[1, 0]
            split_data = {
                'Train Normal': self.y_train.value_counts()[0],
                'Train DoS': self.y_train.value_counts()[1],
                'Test Normal': self.y_test.value_counts()[0],
                'Test DoS': self.y_test.value_counts()[1]
            }
            bars = ax3.bar(split_data.keys(), split_data.values(), 
                          color=['#3498DB', '#E74C3C', '#5DADE2', '#F1948A'])
            ax3.set_title('Train-Test Split Distribution', fontweight='bold')
            ax3.set_ylabel('Number of Samples')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # 4. Feature Distribution (sample features)
            ax4 = axes[1, 1]
            # Plot distribution of first 3 features
            sample_features_plot = self.feature_names[:3]
            for i, feature in enumerate(sample_features_plot):
                ax4.hist(self.dataset[feature], bins=30, alpha=0.6, label=f'{feature}')
            ax4.set_title('Sample Feature Distributions', fontweight='bold')
            ax4.set_xlabel('Feature Values (Scaled)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            
            plt.tight_layout()
            
            if save_plots:
                plot_filename = '../04_validation_results/01_dataset_analysis.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"✅ Visualization saved to: {plot_filename}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")
            return False
    
    def save_setup_results(self):
        """
        Save setup results and configuration
        """
        print(f"\n💾 SAVING SETUP RESULTS")
        print("=" * 50)
        
        try:
            # Save results log
            results_filename = '../04_validation_results/01_setup_results.json'
            with open(results_filename, 'w') as f:
                json.dump(self.results_log, f, indent=4)
            
            print(f"✅ Setup results saved to: {results_filename}")
            
            # Save train-test split data for later use
            np.save('../03_model_training/X_train.npy', self.X_train.values)
            np.save('../03_model_training/X_test.npy', self.X_test.values)
            np.save('../03_model_training/y_train.npy', self.y_train.values)
            np.save('../03_model_training/y_test.npy', self.y_test.values)
            
            # Save feature names
            with open('../03_model_training/feature_names.json', 'w') as f:
                json.dump(self.feature_names, f)
            
            print(f"✅ Train-test split data saved for model training")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving setup results: {str(e)}")
            return False
    
    def run_complete_setup(self):
        """
        Run the complete setup process
        """
        print("🚀 DoS DETECTION MODEL TRAINING - XAI-ALIGNED SETUP")
        print("=" * 60)
        print("This script prepares data for XAI-integrated model comparison")
        print("Models: Random Forest (XAI-optimized), XGBoost (SHAP), Logistic Regression, SVM")
        print("XAI Focus: SHAP analysis, feature importance, explainable decisions")
        print("=" * 60)
        
        # Step 1: Load and inspect data
        if not self.load_and_inspect_data():
            return False
        
        # Step 2: Create train-test split
        if not self.create_train_test_split():
            return False
        
        # Step 3: Create visualizations
        if not self.create_data_visualization():
            return False
        
        # Step 4: Save results
        if not self.save_setup_results():
            return False
        
        print(f"\n🎉 SETUP COMPLETE!")
        print("=" * 50)
        print("✅ Dataset loaded and analyzed")
        print("✅ Train-test split created")
        print("✅ Visualizations generated")
        print("✅ Results saved")
        print("\n📋 Next Steps:")
        print("   1. Run Day 2: XAI Baseline Models (02_xai_baseline_models.py)")
        print("   2. Compare Random Forest and Logistic Regression with XAI")
        print("   3. Continue with XGBoost SHAP integration")
        print("   4. Focus on explainable AI throughout the process")
        
        return True

def main():
    """
    Main execution function
    """
    # Initialize setup
    setup = DoSModelTrainingSetup()
    
    # Run complete setup
    success = setup.run_complete_setup()
    
    if success:
        print(f"\n✅ Setup completed successfully!")
        print(f"🚀 Ready to start model training!")
    else:
        print(f"\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
