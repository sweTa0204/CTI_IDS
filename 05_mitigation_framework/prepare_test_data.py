"""
Prepare Test/Benchmark Data for DoS Detection
==============================================

This script prepares the benchmark test data for evaluation:
1. Load the UNSW-NB15 benchmark CSV
2. Filter for DoS and Normal traffic
3. Select the 10 features we use
4. Scale using the same scaler as training
5. Save for testing

Author: Research Project
Date: 2026-01-30
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# UNSW-NB15 Official Dataset Files (renamed for clarity):
# UNSW_NB15_TRAINING_175341.csv = Official Training Set (175,341 records) - used to train the model
# UNSW_NB15_TESTING_82332.csv = Official Testing Set (82,332 records) - used for benchmark evaluation
BENCHMARK_PATH = os.path.join(BASE_DIR, '01_data_preparation', 'data', 'official_datasets',
                               'UNSW_NB15_TESTING_82332.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '03_model_training', 'proper_training', 'data')
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, 'X_train_scaled.csv')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'feature_scaler.pkl')
PROTO_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'proto_encoder.pkl')

# 10 features we use (must match training)
FEATURE_NAMES = ['rate', 'sload', 'sbytes', 'dload', 'proto',
                 'dtcpb', 'stcpb', 'dmean', 'tcprtt', 'dur']


def prepare_test_data():
    """Prepare benchmark test data."""

    print("=" * 70)
    print("PREPARING BENCHMARK TEST DATA")
    print("=" * 70)

    # Step 1: Load benchmark data
    print("\n[Step 1] Loading benchmark data...")
    df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    print(f"  Loaded {len(df)} records")

    # Step 2: Check attack categories
    print("\n[Step 2] Analyzing attack categories...")
    print(f"  Attack categories: {df['attack_cat'].unique()}")
    print(f"  Value counts:\n{df['attack_cat'].value_counts()}")

    # Step 3: Filter for DoS and Normal (binary classification)
    print("\n[Step 3] Filtering for DoS detection (DoS vs Normal)...")

    # Create binary label: 1 for DoS, 0 for Normal
    # Note: 'attack_cat' contains attack types, 'label' contains 0/1
    # We want specifically DoS attacks vs Normal traffic

    # Filter for DoS attacks and Normal traffic only
    dos_df = df[df['attack_cat'] == 'DoS'].copy()
    normal_df = df[df['attack_cat'] == 'Normal'].copy()

    print(f"  DoS records: {len(dos_df)}")
    print(f"  Normal records: {len(normal_df)}")

    # Combine
    test_df = pd.concat([dos_df, normal_df], ignore_index=True)
    print(f"  Total test records: {len(test_df)}")

    # Create binary labels
    test_df['binary_label'] = (test_df['attack_cat'] == 'DoS').astype(int)

    # Step 4: Select features
    print("\n[Step 4] Selecting features...")

    # Check if all features exist
    missing_features = [f for f in FEATURE_NAMES if f not in test_df.columns]
    if missing_features:
        print(f"  [WARNING] Missing features: {missing_features}")

    # Handle 'proto' encoding using SAVED encoder from training
    if 'proto' in test_df.columns and test_df['proto'].dtype == 'object':
        print("  Encoding 'proto' column using saved encoder...")
        print(f"  Loading proto encoder from: {PROTO_ENCODER_PATH}")
        with open(PROTO_ENCODER_PATH, 'rb') as f:
            proto_encoder = pickle.load(f)

        # Transform using saved encoder - handle unseen labels
        proto_values = test_df['proto'].astype(str)
        known_classes = set(proto_encoder.classes_)

        # Map unknown protocols to a known one (fallback)
        def safe_transform(val):
            if val in known_classes:
                return proto_encoder.transform([val])[0]
            else:
                # Fallback to most common protocol (tcp)
                return proto_encoder.transform(['tcp'])[0] if 'tcp' in known_classes else 0

        test_df['proto'] = proto_values.apply(safe_transform)

    # Select features
    X_test = test_df[FEATURE_NAMES].copy()
    y_test = test_df['binary_label'].copy()

    print(f"  Feature shape: {X_test.shape}")
    print(f"  Label distribution: Normal={sum(y_test==0)}, DoS={sum(y_test==1)}")

    # Step 5: Handle missing values
    print("\n[Step 5] Handling missing values...")
    missing_before = X_test.isnull().sum().sum()
    print(f"  Missing values before: {missing_before}")

    # Fill with median (same approach as training)
    for col in X_test.columns:
        if X_test[col].isnull().sum() > 0:
            X_test[col] = X_test[col].fillna(X_test[col].median())

    missing_after = X_test.isnull().sum().sum()
    print(f"  Missing values after: {missing_after}")

    # Step 6: Scale features using SAVED SCALER from training
    print("\n[Step 6] Scaling features...")

    # IMPORTANT: Use the SAME scaler that was used during training!
    # This is critical - using a different scaler will give wrong results.
    print(f"  Loading saved scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Use transform() NOT fit_transform() - we use the training scaler
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES)

    print(f"  Scaled shape: {X_test_scaled.shape}")

    # Step 7: Save test data
    print("\n[Step 7] Saving test data...")

    X_test_path = os.path.join(OUTPUT_DIR, 'X_test_scaled.csv')
    y_test_path = os.path.join(OUTPUT_DIR, 'y_test.csv')

    X_test_scaled.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False, header=['label'])

    print(f"  Saved: {X_test_path}")
    print(f"  Saved: {y_test_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nTest Dataset Summary:")
    print(f"  - Total records: {len(X_test_scaled)}")
    print(f"  - Normal: {sum(y_test==0)} ({sum(y_test==0)/len(y_test)*100:.1f}%)")
    print(f"  - DoS: {sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.1f}%)")
    print(f"  - Features: {FEATURE_NAMES}")

    return X_test_scaled, y_test


if __name__ == "__main__":
    X_test, y_test = prepare_test_data()
