#!/usr/bin/env python3
"""
Fixed Benchmarking Script for DoS Detection Model
Properly preprocesses test data to match training data format
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

def create_protocol_mapping():
    """Create protocol mapping from original DoS dataset"""
    dos_df = pd.read_csv('01_data_preparation/data/dos_detection_dataset.csv')
    encoded_df = pd.read_csv('01_data_preparation/data/encoded_dataset.csv')
    
    # Create mapping by matching indices
    proto_mapping = {}
    unique_protocols = dos_df['proto'].unique()
    
    for proto in unique_protocols:
        # Find first occurrence in original data
        idx = dos_df[dos_df['proto'] == proto].index[0]
        # Get corresponding encoded value
        encoded_value = encoded_df.loc[idx, 'proto']
        proto_mapping[proto] = encoded_value
    
    return proto_mapping

def preprocess_test_data(test_df, feature_names):
    """Apply the same preprocessing pipeline as training data"""
    print("🔄 Preprocessing test data...")
    
    # Step 1: Extract DoS vs Normal samples
    dos_mask = test_df['attack_cat'] == 'DoS'
    normal_mask = test_df['attack_cat'] == 'Normal'
    binary_test = test_df[dos_mask | normal_mask].copy()
    
    print(f"📊 Extracted binary data: {len(binary_test)} samples")
    print(f"   - DoS: {dos_mask.sum()}, Normal: {normal_mask.sum()}")
    
    # Step 2: Create binary labels
    binary_test['binary_label'] = (binary_test['attack_cat'] == 'DoS').astype(int)
    
    # Step 3: Protocol encoding
    proto_mapping = create_protocol_mapping()
    binary_test['proto_encoded'] = binary_test['proto'].map(proto_mapping)
    
    # Handle unmapped protocols
    unmapped = binary_test['proto_encoded'].isna().sum()
    if unmapped > 0:
        print(f"⚠️  {unmapped} unmapped protocols, using mode value")
        mode_value = binary_test['proto_encoded'].mode()[0] if not binary_test['proto_encoded'].mode().empty else 0
        binary_test['proto_encoded'] = binary_test['proto_encoded'].fillna(mode_value)
    
    # Step 4: Update proto column
    binary_test['proto'] = binary_test['proto_encoded']
    
    # Step 5: Select features
    available_features = [f for f in feature_names if f in binary_test.columns]
    X_test = binary_test[available_features].copy()
    y_test = binary_test['binary_label'].copy()
    
    print(f"✅ Using features: {available_features}")
    
    return X_test, y_test

def scale_features_properly(X_test):
    """Apply proper scaling based on training data statistics"""
    print("🔄 Applying proper feature scaling...")
    
    # Load original training data before scaling
    dos_df = pd.read_csv('01_data_preparation/data/dos_detection_dataset.csv')
    
    # Create protocol mapping
    proto_mapping = create_protocol_mapping()
    dos_df['proto'] = dos_df['proto'].map(proto_mapping)
    
    # Handle any unmapped protocols in training data
    if dos_df['proto'].isna().any():
        dos_df['proto'] = dos_df['proto'].fillna(dos_df['proto'].mode()[0])
    
    # Get the same features from training data
    feature_names = X_test.columns.tolist()
    X_train_orig = dos_df[feature_names]
    
    # Create and fit scaler on original training data
    scaler = StandardScaler()
    scaler.fit(X_train_orig)
    
    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    print("✅ Features scaled successfully")
    return X_test_scaled_df

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data"""
    print("🔄 Making predictions...")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    
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
        'total_samples': len(y_test),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def display_results(results, training_results=None):
    """Display benchmarking results"""
    print("\n" + "="*60)
    print("🎯 FIXED EXTERNAL BENCHMARKING RESULTS")
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
    print(f"   • True Negatives (Normal correctly classified):  {cm[0,0]:,}")
    print(f"   • False Positives (Normal classified as DoS):    {cm[0,1]:,}")
    print(f"   • False Negatives (DoS classified as Normal):    {cm[1,0]:,}")
    print(f"   • True Positives (DoS correctly classified):     {cm[1,1]:,}")
    
    # Calculate rates
    total_normal = cm[0,0] + cm[0,1]
    total_dos = cm[1,0] + cm[1,1]
    
    print(f"\n📋 Classification Breakdown:")
    print(f"   • Normal Traffic: {total_normal:,} samples")
    print(f"     - Correctly identified: {cm[0,0]:,} ({cm[0,0]/total_normal*100:.1f}%)")
    print(f"     - Misclassified as DoS: {cm[0,1]:,} ({cm[0,1]/total_normal*100:.1f}%)")
    print(f"   • DoS Attacks: {total_dos:,} samples")
    print(f"     - Correctly identified: {cm[1,1]:,} ({cm[1,1]/total_dos*100:.1f}%)")
    print(f"     - Missed (false negative): {cm[1,0]:,} ({cm[1,0]/total_dos*100:.1f}%)")
    
    if training_results:
        print(f"\n🔄 Training vs Testing Comparison:")
        print(f"   • Training Accuracy: {training_results['accuracy']:.4f} ({training_results['accuracy']*100:.2f}%)")
        print(f"   • Testing Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
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
    print("🚀 Starting Fixed DoS Detection Benchmarking")
    print("="*60)
    
    try:
        # Load model
        model, feature_names = load_trained_model()
        
        # Load test data
        test_df = pd.read_csv('01_data_preparation/data/UNSW_NB15_testing-set.csv')
        print(f"📊 Loaded test dataset: {test_df.shape}")
        
        # Preprocess test data
        X_test, y_test = preprocess_test_data(test_df, feature_names)
        
        # Scale features properly
        X_test_scaled = scale_features_properly(X_test)
        
        # Evaluate model
        results = evaluate_model(model, X_test_scaled, y_test)
        
        # Load training results for comparison
        training_results = load_training_results()
        
        # Display results
        display_results(results, training_results)
        
        print("\n✅ Fixed benchmarking completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
