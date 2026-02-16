"""Model loading and results aggregation."""

import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "03_model_training" / "proper_training" / "data"
MODEL_DIR = PROJECT_ROOT / "03_model_training" / "proper_training" / "models"
RESULTS_DIR = PROJECT_ROOT / "03_model_training" / "proper_training" / "results"

FEATURE_NAMES = ["rate", "sload", "sbytes", "dload", "proto",
                 "dtcpb", "stcpb", "dmean", "tcprtt", "dur"]
THRESHOLD = 0.8517


@st.cache_resource
def load_model():
    with open(MODEL_DIR / "xgboost" / "xgboost_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(DATA_DIR / "feature_scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoder():
    with open(DATA_DIR / "proto_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_shap_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


@st.cache_data
def load_sample_data():
    X = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    if list(X.columns) == list(range(len(X.columns))):
        X.columns = FEATURE_NAMES
    y = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
    return X, y


def load_all_model_results():
    """Load results for all 7 models from various result files."""
    results = {}

    # 1. Load optimized benchmark results (XGBoost, RF, MLP, SVM, LR)
    opt_path = RESULTS_DIR / "benchmark_results_optimized.json"
    if opt_path.exists():
        with open(opt_path) as f:
            benchmarks = json.load(f)
        for name, d in benchmarks.items():
            results[name] = {
                "accuracy": round(d["accuracy"] * 100, 2),
                "precision": round(d["precision"] * 100, 2),
                "recall": round(d["recall"] * 100, 2),
                "f1": round(d["f1_score"] * 100, 2),
                "threshold": round(d["optimal_threshold"], 4),
                "tp": d.get("true_positives", 0),
                "fp": d.get("false_positives", 0),
                "fn": d.get("false_negatives", 0),
                "tn": d.get("true_negatives", 0),
            }

    # 2. Load training results for CV scores
    train_path = RESULTS_DIR / "training_results.json"
    if train_path.exists():
        with open(train_path) as f:
            training = json.load(f)
        for name, d in training.items():
            if name in results:
                results[name]["cv_f1"] = round(d["cv_f1_mean"] * 100, 2)

    # 3. Load LSTM results
    lstm_path = MODEL_DIR / "lstm" / "results" / "lstm_results.json"
    if lstm_path.exists():
        with open(lstm_path) as f:
            d = json.load(f)
        opt = d.get("optimized", d.get("optimized_threshold", {}))
        cm = opt.get("confusion_matrix", [[0, 0], [0, 0]])
        results["LSTM"] = {
            "accuracy": round(opt.get("accuracy", 0) * 100, 2),
            "precision": round(opt.get("precision", 0) * 100, 2),
            "recall": round(opt.get("recall", 0) * 100, 2),
            "f1": round(opt.get("f1_score", 0) * 100, 2),
            "threshold": round(opt.get("threshold", 0.5), 4),
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
            "cv_f1": None,
        }

    # 4. Load 1D-CNN results
    cnn_path = MODEL_DIR / "cnn1d" / "results" / "cnn1d_results.json"
    if cnn_path.exists():
        with open(cnn_path) as f:
            d = json.load(f)
        opt = d.get("optimized", {})
        cm = opt.get("confusion_matrix", [[0, 0], [0, 0]])
        results["1D-CNN"] = {
            "accuracy": round(opt.get("accuracy", 0) * 100, 2),
            "precision": round(opt.get("precision", 0) * 100, 2),
            "recall": round(opt.get("recall", 0) * 100, 2),
            "f1": round(opt.get("f1_score", 0) * 100, 2),
            "threshold": round(opt.get("threshold", 0.5), 4),
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
            "cv_f1": None,
        }

    # Sort by F1 descending
    results = dict(sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True))
    return results
