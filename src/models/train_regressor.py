"""Regression Model Development for Maximum Monthly EMI Prediction.

Trains minimum 3 models:
1. Ridge / Linear Regression (Baseline)
2. Random Forest Regressor (Ensemble-based)
3. XGBoost Regressor (Advanced Gradient Boosting)
"""

import time
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.evaluate import evaluate_regression_model


def get_regression_models() -> Dict[str, Any]:
    """Instantiate regression models with tuned hyperparameters."""
    return {
        "Linear_Ridge_Regression": Ridge(
            alpha=10.0, 
            random_state=42
        ),
        "Random_Forest_Regressor": RandomForestRegressor(
            n_estimators=100, 
            max_depth=16, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost_Regressor": XGBRegressor(
            n_estimators=150, 
            max_depth=6, 
            learning_rate=0.08, 
            subsample=0.85, 
            colsample_bytree=0.85, 
            random_state=42, 
            n_jobs=-1
        )
    }


def train_and_evaluate_regressor(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Train single regressor and evaluate on train, val, test splits."""
    print(f"--- Training Regressor: {model_name} ---")
    start_time = time.time()
    
    # Train
    model.fit(X_train, y_train)
    train_duration = round(time.time() - start_time, 2)
    
    # Evaluate
    val_metrics = evaluate_regression_model(model, X_val, y_val)
    test_metrics = evaluate_regression_model(model, X_test, y_test)
    train_metrics = evaluate_regression_model(model, X_train, y_train)
    
    print(f"[{model_name}] Completed in {train_duration}s | "
          f"Val RMSE: INR {val_metrics['rmse']:.2f}, Val R²: {val_metrics['r2_score']:.4f} | "
          f"Test RMSE: INR {test_metrics['rmse']:.2f}, Test R²: {test_metrics['r2_score']:.4f}")
    
    return {
        "model_name": model_name,
        "model": model,
        "train_time_sec": train_duration,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "hyperparameters": model.get_params()
    }


def train_all_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Train all 3 regression candidate models."""
    models = get_regression_models()
    results = {}
    
    for name, model in models.items():
        results[name] = train_and_evaluate_regressor(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
    return results
