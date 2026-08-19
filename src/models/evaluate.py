"""Evaluation Metrics and Visualizations Module for EMIPredict AI.

Calculates comprehensive performance metrics for Classification and Regression,
and generates publication-ready diagnostic charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import label_binarize


def evaluate_classification_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    target_names: List[str] = ["Eligible", "High_Risk", "Not_Eligible"]
) -> Dict[str, Any]:
    """Compute complete classification evaluation metrics."""
    y_pred = model.predict(X)
    
    # Check if predict_proba is available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X)
        exp_vals = np.exp(df_vals - np.max(df_vals, axis=1, keepdims=True))
        y_prob = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    else:
        y_prob = None

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    roc_auc_macro = None
    roc_auc_weighted = None
    if y_prob is not None:
        try:
            n_classes = len(target_names)
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            if y_prob.shape[1] == n_classes:
                roc_auc_macro = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
                roc_auc_weighted = float(roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr"))
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def evaluate_regression_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, Any]:
    """Compute complete regression evaluation metrics."""
    y_pred = model.predict(X)
    
    # Clip negative predictions to 0 for financial validity
    y_pred = np.maximum(0.0, y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    max_err = float(np.max(np.abs(y_true - y_pred)))
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "r2_score": float(r2),
        "mape": float(mape),
        "max_error": max_err,
        "y_pred": y_pred
    }


def plot_confusion_matrix(
    cm: List[List[int]], 
    target_names: List[str], 
    save_path: Optional[str] = None
):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=target_names, 
        yticklabels=target_names,
        ax=ax,
        cbar=False
    )
    ax.set_title("Confusion Matrix (EMI Eligibility)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return fig


def plot_feature_importances(
    feature_names: List[str], 
    importances: np.ndarray, 
    top_n: int = 15,
    title: str = "Top Feature Importances",
    save_path: Optional[str] = None
):
    """Plot top N feature importances."""
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_idx][::-1]
    top_scores = importances[sorted_idx][::-1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top_names, top_scores, color="#1f77b4", edgecolor="#0e4b75")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return fig


def plot_regression_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: Optional[str] = None
):
    """Plot predicted vs actual and residual distribution."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Actual vs Predicted
    ax1.scatter(y_true[:2000], y_pred[:2000], alpha=0.3, color="#2ca02c", s=15)
    max_val = max(np.max(y_true[:2000]), np.max(y_pred[:2000]))
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2)
    ax1.set_title("Actual vs Predicted Max EMI", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Actual EMI (INR)")
    ax1.set_ylabel("Predicted EMI (INR)")
    
    # Subplot 2: Residuals Distribution
    sns.histplot(residuals, kde=True, ax=ax2, color="#ff7f0e", bins=40)
    ax2.axvline(0, color="black", linestyle="--")
    ax2.set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return fig
