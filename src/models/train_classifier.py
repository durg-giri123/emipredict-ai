"""Classification Model Development for EMI Eligibility Prediction.

Trains minimum 3 models:
1. Logistic Regression (Baseline Interpretable)
2. Random Forest Classifier (Ensemble Feature Importance)
3. XGBoost Classifier (Gradient Boosting State-of-the-Art)
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.evaluate import evaluate_classification_model


def get_classification_models() -> Dict[str, Any]:
    """Instantiate classification models with optimized hyperparameters."""
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, 
            C=1.0, 
            solver="lbfgs", 
            random_state=42
        ),
        "Random_Forest_Classifier": RandomForestClassifier(
            n_estimators=100, 
            max_depth=16, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost_Classifier": XGBClassifier(
            n_estimators=150, 
            max_depth=6, 
            learning_rate=0.08, 
            subsample=0.85, 
            colsample_bytree=0.85, 
            eval_metric="mlogloss", 
            random_state=42, 
            n_jobs=-1
        )
    }


def train_and_evaluate_classifier(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Train single classifier and evaluate on train, val, test splits."""
    print(f"--- Training Classifier: {model_name} ---")
    start_time = time.time()
    
    # Train
    model.fit(X_train, y_train)
    train_duration = round(time.time() - start_time, 2)
    
    # Evaluate
    val_metrics = evaluate_classification_model(model, X_val, y_val)
    test_metrics = evaluate_classification_model(model, X_test, y_test)
    train_metrics = evaluate_classification_model(model, X_train, y_train)
    
    print(f"[{model_name}] Completed in {train_duration}s | "
          f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_weighted']:.4f} | "
          f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1_weighted']:.4f}")
    
    return {
        "model_name": model_name,
        "model": model,
        "train_time_sec": train_duration,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "hyperparameters": model.get_params()
    }


def train_all_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Train all 3 classification candidate models."""
    models = get_classification_models()
    results = {}
    
    for name, model in models.items():
        results[name] = train_and_evaluate_classifier(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
    return results
