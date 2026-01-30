#!/usr/bin/env python3
"""
Model Inspector - View contents of .pkl model files
====================================================
Usage: python inspect_model.py <model_folder>
Example: python inspect_model.py xgboost
"""

import joblib
import sys
import os

def inspect_model(model_folder):
    """Inspect the contents of a trained model"""

    # Find the pkl file
    pkl_files = [f for f in os.listdir(model_folder) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"No .pkl files found in {model_folder}")
        return

    pkl_path = os.path.join(model_folder, pkl_files[0])
    print(f"\n{'='*60}")
    print(f"INSPECTING: {pkl_path}")
    print(f"{'='*60}\n")

    # Load the model
    model = joblib.load(pkl_path)

    # Basic info
    print(f"Model Type: {type(model).__name__}")
    print(f"Module: {type(model).__module__}")
    print()

    # Get all attributes
    print("MODEL ATTRIBUTES:")
    print("-" * 40)

    for attr in dir(model):
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(model, attr)
                if not callable(value):  # Skip methods
                    # Format the value nicely
                    if hasattr(value, 'shape'):
                        print(f"  {attr}: array with shape {value.shape}")
                    elif isinstance(value, (list, tuple)) and len(value) > 5:
                        print(f"  {attr}: {type(value).__name__} with {len(value)} items")
                    else:
                        val_str = str(value)
                        if len(val_str) > 60:
                            val_str = val_str[:60] + "..."
                        print(f"  {attr}: {val_str}")
            except:
                pass

    print()

    # Model-specific details
    print("KEY INFORMATION:")
    print("-" * 40)

    # XGBoost
    if hasattr(model, 'n_estimators'):
        print(f"  Number of trees/estimators: {model.n_estimators}")

    if hasattr(model, 'max_depth'):
        print(f"  Max depth: {model.max_depth}")

    # Feature importance (XGBoost, Random Forest)
    if hasattr(model, 'feature_importances_'):
        print(f"  Feature importances: {model.feature_importances_}")

    # Coefficients (Logistic Regression)
    if hasattr(model, 'coef_'):
        print(f"  Coefficients shape: {model.coef_.shape}")
        print(f"  Coefficients: {model.coef_[0]}")

    if hasattr(model, 'intercept_'):
        print(f"  Intercept: {model.intercept_}")

    # Neural Network (MLP)
    if hasattr(model, 'n_layers_'):
        print(f"  Number of layers: {model.n_layers_}")

    if hasattr(model, 'hidden_layer_sizes'):
        print(f"  Hidden layer sizes: {model.hidden_layer_sizes}")

    if hasattr(model, 'n_iter_'):
        print(f"  Iterations to converge: {model.n_iter_}")

    if hasattr(model, 'coefs_'):
        print(f"  Weight matrices: {len(model.coefs_)} layers")
        for i, coef in enumerate(model.coefs_):
            print(f"    Layer {i+1}: {coef.shape}")

    # SVM
    if hasattr(model, 'support_vectors_'):
        print(f"  Number of support vectors: {len(model.support_vectors_)}")

    if hasattr(model, 'kernel'):
        print(f"  Kernel: {model.kernel}")

    if hasattr(model, 'C'):
        print(f"  C (regularization): {model.C}")

    print()
    print(f"{'='*60}")
    print("Inspection complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # If no argument, inspect all models
        models = ['xgboost', 'random_forest', 'svm', 'mlp', 'logistic_regression']
        print("No model specified. Inspecting all models...\n")
        for model in models:
            if os.path.exists(model):
                inspect_model(model)
    else:
        model_folder = sys.argv[1]
        if os.path.exists(model_folder):
            inspect_model(model_folder)
        else:
            print(f"Folder not found: {model_folder}")
