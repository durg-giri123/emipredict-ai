"""MLflow Experiment Tracking and Model Registry Management for EMIPredict AI.

Handles logging parameters, metrics, artifact visualizations, candidate comparison,
best model selection, and model registry lifecycle management.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Enable MLflow file store compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.models.evaluate import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_regression_residuals
)


class MLflowTracker:
    """Manages MLflow experiment tracking and model registration."""
    
    def __init__(
        self, 
        tracking_uri: str = "sqlite:///mlflow.db",
        clf_experiment_name: str = "EMIPredict_Classification_Suite",
        reg_experiment_name: str = "EMIPredict_Regression_Suite"
    ):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.tracking_uri = tracking_uri
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        except Exception as e:
            print(f"[MLflow] Note: Setting tracking URI fallback: {e}")
            self.tracking_uri = "file:./mlruns"
            mlflow.set_tracking_uri(self.tracking_uri)
            
        self.clf_experiment_name = clf_experiment_name
        self.reg_experiment_name = reg_experiment_name
        
        self.clf_exp_id = self._get_or_create_experiment(self.clf_experiment_name)
        self.reg_exp_id = self._get_or_create_experiment(self.reg_experiment_name)

    def _get_or_create_experiment(self, exp_name: str) -> Optional[str]:
        """Create experiment or return existing ID safely."""
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                return mlflow.create_experiment(exp_name)
            return exp.experiment_id
        except Exception as e:
            print(f"[MLflow] Info getting experiment '{exp_name}': {e}")
            return "0"

    def log_classification_run(
        self,
        model_name: str,
        model: Any,
        hyperparams: Dict[str, Any],
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        feature_names: List[str],
        artifacts_dir: str = "artifacts/visualizations"
    ) -> str:
        """Log a classification experiment run to MLflow."""
        os.makedirs(artifacts_dir, exist_ok=True)
        
        try:
            mlflow.set_experiment(self.clf_experiment_name)
            with mlflow.start_run(run_name=model_name) as run:
                # 1. Log Hyperparameters
                filtered_params = {
                    k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                    for k, v in hyperparams.items()
                    if v is not None
                }
                mlflow.log_params(dict(list(filtered_params.items())[:50]))
                
                # 2. Log Metrics
                mlflow.log_metrics({
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_weighted": train_metrics["f1_weighted"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_weighted": val_metrics["precision_weighted"],
                    "val_recall_weighted": val_metrics["recall_weighted"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision_weighted": test_metrics["precision_weighted"],
                    "test_recall_weighted": test_metrics["recall_weighted"],
                    "test_f1_weighted": test_metrics["f1_weighted"],
                    "test_f1_macro": test_metrics["f1_macro"]
                })
                
                if val_metrics.get("roc_auc_weighted") is not None:
                    mlflow.log_metric("val_roc_auc_weighted", val_metrics["roc_auc_weighted"])
                if test_metrics.get("roc_auc_weighted") is not None:
                    mlflow.log_metric("test_roc_auc_weighted", test_metrics["roc_auc_weighted"])
                    
                # 3. Generate and Log Visual Artifacts
                target_names = ["Eligible", "High_Risk", "Not_Eligible"]
                cm_path = os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png")
                plot_confusion_matrix(test_metrics["confusion_matrix"], target_names, save_path=cm_path)
                try:
                    mlflow.log_artifact(cm_path, artifact_path="visualizations")
                except Exception:
                    pass
                
                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    fi_path = os.path.join(artifacts_dir, f"{model_name}_feature_importance.png")
                    plot_feature_importances(
                        feature_names, 
                        model.feature_importances_, 
                        title=f"{model_name} Feature Importances", 
                        save_path=fi_path
                    )
                    try:
                        mlflow.log_artifact(fi_path, artifact_path="visualizations")
                    except Exception:
                        pass
                elif hasattr(model, "coef_"):
                    fi_path = os.path.join(artifacts_dir, f"{model_name}_coef_importance.png")
                    avg_coef = np.mean(np.abs(model.coef_), axis=0)
                    plot_feature_importances(
                        feature_names, 
                        avg_coef, 
                        title=f"{model_name} Absolute Coefficients", 
                        save_path=fi_path
                    )
                    try:
                        mlflow.log_artifact(fi_path, artifact_path="visualizations")
                    except Exception:
                        pass
                    
                # 4. Log Model with MLflow
                try:
                    if "XGB" in model_name:
                        mlflow.xgboost.log_model(model, artifact_path="model")
                    else:
                        mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception:
                    pass
                    
                return run.info.run_id
        except Exception as e:
            print(f"[MLflow] Warning during logging classification run: {e}")
            return "run_clf_completed"

    def log_regression_run(
        self,
        model_name: str,
        model: Any,
        hyperparams: Dict[str, Any],
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        y_test_true: np.ndarray,
        feature_names: List[str],
        artifacts_dir: str = "artifacts/visualizations"
    ) -> str:
        """Log a regression experiment run to MLflow."""
        os.makedirs(artifacts_dir, exist_ok=True)
        
        try:
            mlflow.set_experiment(self.reg_experiment_name)
            with mlflow.start_run(run_name=model_name) as run:
                filtered_params = {
                    k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                    for k, v in hyperparams.items()
                    if v is not None
                }
                mlflow.log_params(dict(list(filtered_params.items())[:50]))
                
                # Log Metrics
                mlflow.log_metrics({
                    "train_rmse": train_metrics["rmse"],
                    "train_r2_score": train_metrics["r2_score"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2_score": val_metrics["r2_score"],
                    "val_mape": val_metrics["mape"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2_score": test_metrics["r2_score"],
                    "test_mape": test_metrics["mape"],
                    "test_max_error": test_metrics["max_error"]
                })
                
                # Generate and Log Residual Plots
                resid_path = os.path.join(artifacts_dir, f"{model_name}_residuals.png")
                plot_regression_residuals(y_test_true, test_metrics["y_pred"], save_path=resid_path)
                try:
                    mlflow.log_artifact(resid_path, artifact_path="visualizations")
                except Exception:
                    pass
                
                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    fi_path = os.path.join(artifacts_dir, f"{model_name}_feature_importance.png")
                    plot_feature_importances(
                        feature_names, 
                        model.feature_importances_, 
                        title=f"{model_name} Feature Importances", 
                        save_path=fi_path
                    )
                    try:
                        mlflow.log_artifact(fi_path, artifact_path="visualizations")
                    except Exception:
                        pass
                elif hasattr(model, "coef_"):
                    fi_path = os.path.join(artifacts_dir, f"{model_name}_coef_importance.png")
                    plot_feature_importances(
                        feature_names, 
                        np.abs(model.coef_), 
                        title=f"{model_name} Absolute Coefficients", 
                        save_path=fi_path
                    )
                    try:
                        mlflow.log_artifact(fi_path, artifact_path="visualizations")
                    except Exception:
                        pass
                    
                # Log Model
                try:
                    if "XGB" in model_name:
                        mlflow.xgboost.log_model(model, artifact_path="model")
                    else:
                        mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception:
                    pass
                    
                return run.info.run_id
        except Exception as e:
            print(f"[MLflow] Warning during logging regression run: {e}")
            return "run_reg_completed"

    def get_leaderboard(self, experiment_type: str = "classification") -> pd.DataFrame:
        """Fetch all runs for an experiment and return formatted leaderboard safely."""
        try:
            exp_name = self.clf_experiment_name if experiment_type == "classification" else self.reg_experiment_name
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                return pd.DataFrame()
                
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if runs is None or runs.empty:
                return pd.DataFrame()
                
            return runs
        except Exception as e:
            print(f"[MLflow] Leaderboard search exception: {e}")
            return pd.DataFrame()

    def select_best_models(
        self,
        clf_results: Dict[str, Dict[str, Any]],
        reg_results: Dict[str, Dict[str, Any]],
        export_dir: str = "artifacts/production_models"
    ) -> Dict[str, Any]:
        """
        Evaluate and select best performing models:
        - Classification: Highest Validation F1-Weighted / Accuracy
        - Regression: Lowest Validation RMSE / Highest R²
        Saves production models and registry metadata.
        """
        os.makedirs(export_dir, exist_ok=True)
        
        # Best Classifier Selection
        best_clf_name = max(
            clf_results.keys(), 
            key=lambda k: clf_results[k]["val_metrics"]["f1_weighted"]
        )
        best_clf_obj = clf_results[best_clf_name]["model"]
        
        # Best Regressor Selection
        best_reg_name = min(
            reg_results.keys(), 
            key=lambda k: reg_results[k]["val_metrics"]["rmse"]
        )
        best_reg_obj = reg_results[best_reg_name]["model"]
        
        # Save to disk for production application inference
        clf_save_path = os.path.join(export_dir, "best_classifier.joblib")
        reg_save_path = os.path.join(export_dir, "best_regressor.joblib")
        
        joblib.dump(best_clf_obj, clf_save_path)
        joblib.dump(best_reg_obj, reg_save_path)
        
        # Model Registry summary
        registry_meta = {
            "best_classifier": {
                "name": best_clf_name,
                "val_f1_weighted": clf_results[best_clf_name]["val_metrics"]["f1_weighted"],
                "test_accuracy": clf_results[best_clf_name]["test_metrics"]["accuracy"],
                "path": clf_save_path,
                "stage": "Production"
            },
            "best_regressor": {
                "name": best_reg_name,
                "val_rmse": reg_results[best_reg_name]["val_metrics"]["rmse"],
                "test_r2": reg_results[best_reg_name]["test_metrics"]["r2_score"],
                "path": reg_save_path,
                "stage": "Production"
            }
        }
        
        meta_path = os.path.join(export_dir, "model_registry.json")
        with open(meta_path, "w") as f:
            json.dump(registry_meta, f, indent=4)
            
        print("\n=======================================================")
        print("★ MODEL REGISTRY: PRODUCTION SELECTION COMPLETE ★")
        print(f"-> Production Classifier: {best_clf_name} (Val F1: {registry_meta['best_classifier']['val_f1_weighted']:.4f}, Test Acc: {registry_meta['best_classifier']['test_accuracy']:.4f})")
        print(f"-> Production Regressor : {best_reg_name} (Val RMSE: INR {registry_meta['best_regressor']['val_rmse']:.2f}, Test R²: {registry_meta['best_regressor']['test_r2']:.4f})")
        print("=======================================================\n")
        
        return registry_meta
