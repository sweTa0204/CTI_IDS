#!/usr/bin/env python3
"""
Fast Benchmarking Script for DoS Detection Model
Tests trained XGBoost model on UNSW-NB15 testing dataset
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def load_trained_model():
    """Load the trained XGBoost model and feature names"""
    print("🔄 Loading trained XGBoost model...")
    
    # Load model
    with open('03_model_training/models/xgboost/saved_model/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    with open('03_model_training/models/xgboost/saved_model/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    print(f"✅ Model loaded with {len(feature_names)} features")
    return model, feature_names

def encode_categorical_features(df):
    """Encode categorical features to match training data encoding"""
    print("🔄 Encoding categorical features...")
    
    # Load the original DoS dataset to get protocol mapping
    dos_df = pd.read_csv('01_data_preparation/data/dos_detection_dataset.csv')
    encoded_df = pd.read_csv('01_data_preparation/data/encoded_dataset.csv')
    
    # Create protocol mapping from original to encoded values
    proto_mapping = {}
    for i, proto in enumerate(dos_df['proto'].unique()):
        # Find corresponding encoded value
        mask = dos_df['proto'] == proto
        if mask.any():
            encoded_value = encoded_df.loc[dos_df[mask].index[0], 'proto']
            proto_mapping[proto] = encoded_value
    
    # Apply protocol encoding
    df_encoded = df.copy()
    df_encoded['proto'] = df_encoded['proto'].map(proto_mapping)
    
    # Handle missing protocols (assign default value)
    missing_protocols = df_encoded['proto'].isna().sum()
    if missing_protocols > 0:
        print(f"⚠️  Found {missing_protocols} unmapped protocols, assigning default value")
        df_encoded['proto'] = df_encoded['proto'].fillna(0)
    
    print("✅ Categorical encoding completed")
    return df_encoded

def load_and_prepare_test_data(feature_names):
    """Load testing dataset and prepare for prediction"""
    print("🔄 Loading testing dataset...")
    
    # Load test data
    test_df = pd.read_csv('01_data_preparation/data/UNSW_NB15_testing-set.csv')
    print(f"📊 Test dataset shape: {test_df.shape}")
    
    # Extract DoS vs Normal samples
    dos_mask = test_df['attack_cat'] == 'DoS'
    normal_mask = test_df['attack_cat'] == 'Normal'
    
    # Create binary classification dataset
    binary_test = test_df[dos_mask | normal_mask].copy()
    print(f"📊 Binary test data shape: {binary_test.shape}")
    print(f"   - DoS samples: {dos_mask.sum()}")
    print(f"   - Normal samples: {normal_mask.sum()}")
    
    # Create binary labels (0: Normal, 1: DoS)
    binary_test['binary_label'] = (binary_test['attack_cat'] == 'DoS').astype(int)
    
    # Encode categorical features before feature selection
    binary_test_encoded = encode_categorical_features(binary_test)
    
    # Select only the features used in training
    available_features = [f for f in feature_names if f in binary_test_encoded.columns]
    missing_features = [f for f in feature_names if f not in binary_test_encoded.columns]
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
    
    print(f"✅ Using {len(available_features)} features: {available_features}")
    
    # Extract features and labels
    X_test = binary_test_encoded[available_features]
    y_test = binary_test_encoded['binary_label']
    
    return X_test, y_test, binary_test_encoded

def scale_features(X_test, feature_names):
    """Apply the same scaling as used in training"""
    print("🔄 Scaling features...")
    
    # Load training data to get scaling parameters
    train_df = pd.read_csv('01_data_preparation/data/final_scaled_dataset.csv')
    
    # Get the same features from training data
    X_train_sample = train_df[feature_names].head(1000)  # Use sample for efficiency
    
    # Create and fit scaler on training sample
    scaler = StandardScaler()
    scaler.fit(X_train_sample)
    
    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled successfully")
    return X_test_scaled

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data"""
    print("🔄 Making predictions...")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    
    print(f"⏱️  Prediction time: {prediction_time:.2f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'prediction_time': prediction_time,
        'total_samples': len(y_test)
    }

def display_results(results, training_results=None):
    """Display benchmarking results"""
    print("\n" + "="*60)
    print("🎯 EXTERNAL BENCHMARKING RESULTS")
    print("="*60)
    
    print(f"\n📊 Test Dataset Performance:")
    print(f"   • Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   • Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   • Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   • F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"   • ROC-AUC:   {results['roc_auc']:.4f} ({results['roc_auc']*100:.2f}%)")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"   • Prediction Time: {results['prediction_time']:.2f} seconds")
    print(f"   • Total Samples:   {results['total_samples']:,}")
    print(f"   • Predictions/sec: {results['total_samples']/results['prediction_time']:.0f}")
    
    print(f"\n📈 Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"   • True Negatives:  {cm[0,0]:,}")
    print(f"   • False Positives: {cm[0,1]:,}")
    print(f"   • False Negatives: {cm[1,0]:,}")
    print(f"   • True Positives:  {cm[1,1]:,}")
    
    if training_results:
        print(f"\n🔄 Training vs Testing Comparison:")
        print(f"   • Training Accuracy: {training_results['accuracy']:.4f}")
        print(f"   • Testing Accuracy:  {results['accuracy']:.4f}")
        accuracy_diff = results['accuracy'] - training_results['accuracy']
        print(f"   • Difference:        {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
        
        if abs(accuracy_diff) < 0.02:
            print("   ✅ Excellent generalization (< 2% difference)")
        elif abs(accuracy_diff) < 0.05:
            print("   ⚠️  Good generalization (< 5% difference)")
        else:
            print("   ❌ Poor generalization (> 5% difference)")

def load_training_results():
    """Load training results for comparison"""
    try:
        with open('03_model_training/models/xgboost/results/training_results.json', 'r') as f:
            return json.load(f)['performance_metrics']
    except:
        return None

def main():
    """Main benchmarking function"""
    print("🚀 Starting Fast DoS Detection Benchmarking")
    print("="*60)
    
    try:
        # Load model
        model, feature_names = load_trained_model()
        
        # Load and prepare test data
        X_test, y_test, test_df = load_and_prepare_test_data(feature_names)
        
        # Scale features
        X_test_scaled = scale_features(X_test, feature_names)
        
        # Evaluate model
        results = evaluate_model(model, X_test_scaled, y_test)
        
        # Load training results for comparison
        training_results = load_training_results()
        
        # Display results
        display_results(results, training_results)
        
        print("\n✅ Benchmarking completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
