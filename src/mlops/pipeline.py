"""
Main pipeline script — runs the full training workflow from data generation to model export.
Run this once before launching the app to generate models and artifacts.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.data_generator import generate_emi_dataset
from src.data.preprocessor import DataPreprocessor, DataQualityAuditor, load_raw_dataset
from src.models.train_classifier import train_all_classifiers
from src.models.train_regressor import train_all_regressors
from src.mlops.mlflow_tracking import MLflowTracker


def run_full_mlops_pipeline(
    records_count: int = 400000,
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    artifacts_dir: str = "artifacts",
    force_regenerate: bool = False
):
    print("===================================================================")
    print("LAUNCHING EMIPREDICT AI END-TO-END MLOPS PIPELINE")
    print(f"Dataset Target: {records_count:,} Financial Profiles across 5 EMI Scenarios")
    print("===================================================================\n")
    
    total_start_time = time.time()
    
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    parquet_path = os.path.join(raw_data_dir, "emi_dataset_400k.parquet")
    csv_path = os.path.join(raw_data_dir, "emi_dataset_400k.csv")
    
    # -------------------------------------------------------------
    # Step 1: Data Generation & Loading
    # -------------------------------------------------------------
    if os.path.exists(parquet_path) and not force_regenerate:
        print(f"[1/6] Loading existing raw dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"[1/6] Generating {records_count:,} synthetic financial records...")
        df = generate_emi_dataset(
            total_records=records_count, 
            output_dir=raw_data_dir, 
            random_state=42
        )
    print(f"Dataset Ready: {df.shape[0]:,} rows x {df.shape[1]} columns.\n")
    
    # -------------------------------------------------------------
    # Step 2: Data Quality Assessment & Validation Audit
    # -------------------------------------------------------------
    print("[2/6] Running Data Quality Assessment & Audit...")
    audit_report = DataQualityAuditor.audit(df)
    audit_path = os.path.join(artifacts_dir, "data_quality_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit_report, f, indent=4)
    print(f"[OK] Data Quality Score: {audit_report['quality_score']}/100. Audit report saved to {audit_path}.\n")
    
    # -------------------------------------------------------------
    # Step 3: Feature Engineering & Preprocessing
    # -------------------------------------------------------------
    print("[3/6] Fitting Preprocessing & Feature Engineering Pipeline (70/15/15 split)...")
    preprocessor = DataPreprocessor()
    split_data = preprocessor.prepare_train_test_val_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    
    # Save a lightweight sample for UI quick exploration
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    sample_path = os.path.join(processed_data_dir, "dataset_sample_50k.parquet")
    sample_df.to_parquet(sample_path, index=False)
    print(f"[OK] Feature transformations fitted. {len(split_data['feature_names'])} input features generated.")
    print(f"[OK] Preprocessor saved to {preprocessor_path}\n")
    
    # -------------------------------------------------------------
    # Step 4: MLflow Tracker Initialization
    # -------------------------------------------------------------
    print("[4/6] Initializing MLflow Tracking Server...")
    # Initialize MLflow Tracker with default SQLite backend
    tracker = MLflowTracker()
    print(f"[OK] MLflow Experiments Configured: '{tracker.clf_experiment_name}' & '{tracker.reg_experiment_name}'\n")
    
    # -------------------------------------------------------------
    # Step 5: Model Training & MLflow Logging
    # -------------------------------------------------------------
    # 5A. Classification Suite (3 Models)
    print("[5A/6] Training Candidate Classification Models (EMI Eligibility)...")
    clf_results = train_all_classifiers(
        X_train=split_data["X_train"],
        y_train=split_data["y_train_clf"],
        X_val=split_data["X_val"],
        y_val=split_data["y_val_clf"],
        X_test=split_data["X_test"],
        y_test=split_data["y_test_clf"]
    )
    
    print("Logging Classification Models & Artifacts to MLflow...")
    for model_name, res in clf_results.items():
        run_id = tracker.log_classification_run(
            model_name=model_name,
            model=res["model"],
            hyperparams=res["hyperparameters"],
            train_metrics=res["train_metrics"],
            val_metrics=res["val_metrics"],
            test_metrics=res["test_metrics"],
            feature_names=split_data["feature_names"],
            artifacts_dir=os.path.join(artifacts_dir, "visualizations")
        )
        print(f"  -> Logged {model_name} to MLflow Run: {run_id}")
    print()
    
    # 5B. Regression Suite (3 Models)
    print("[5B/6] Training Candidate Regression Models (Max Monthly EMI Amount)...")
    reg_results = train_all_regressors(
        X_train=split_data["X_train"],
        y_train=split_data["y_train_reg"],
        X_val=split_data["X_val"],
        y_val=split_data["y_val_reg"],
        X_test=split_data["X_test"],
        y_test=split_data["y_test_reg"]
    )
    
    print("Logging Regression Models & Artifacts to MLflow...")
    for model_name, res in reg_results.items():
        run_id = tracker.log_regression_run(
            model_name=model_name,
            model=res["model"],
            hyperparams=res["hyperparameters"],
            train_metrics=res["train_metrics"],
            val_metrics=res["val_metrics"],
            test_metrics=res["test_metrics"],
            y_test_true=split_data["y_test_reg"],
            feature_names=split_data["feature_names"],
            artifacts_dir=os.path.join(artifacts_dir, "visualizations")
        )
        print(f"  -> Logged {model_name} to MLflow Run: {run_id}")
    print()
    
    # -------------------------------------------------------------
    # Step 6: Model Selection & Registry Promotion
    # -------------------------------------------------------------
    print("[6/6] Selecting Best Models and Staging to Model Registry...")
    registry_meta = tracker.select_best_models(
        clf_results=clf_results,
        reg_results=reg_results,
        export_dir=os.path.join(artifacts_dir, "production_models")
    )
    
    # Export Benchmark Summary JSON
    benchmark_summary = {
        "dataset_size": len(df),
        "pipeline_execution_time_sec": round(time.time() - total_start_time, 2),
        "classification_models": {
            k: {
                "test_accuracy": round(v["test_metrics"]["accuracy"], 4),
                "test_f1_weighted": round(v["test_metrics"]["f1_weighted"], 4),
                "val_f1_weighted": round(v["val_metrics"]["f1_weighted"], 4),
                "train_time_sec": v["train_time_sec"]
            }
            for k, v in clf_results.items()
        },
        "regression_models": {
            k: {
                "test_rmse": round(v["test_metrics"]["rmse"], 2),
                "test_mae": round(v["test_metrics"]["mae"], 2),
                "test_r2": round(v["test_metrics"]["r2_score"], 4),
                "val_rmse": round(v["val_metrics"]["rmse"], 2),
                "train_time_sec": v["train_time_sec"]
            }
            for k, v in reg_results.items()
        },
        "production_registry": registry_meta
    }
    
    benchmark_path = os.path.join(artifacts_dir, "benchmark_summary.json")
    with open(benchmark_path, "w") as f:
        json.dump(benchmark_summary, f, indent=4)
        
    print(f"[OK] Pipeline Finished Successfully in {benchmark_summary['pipeline_execution_time_sec']} seconds!")
    print(f"[OK] Benchmark Report Saved to {benchmark_path}")
    print("===================================================================\n")
    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full EMIPredict AI MLOps Pipeline")
    parser.add_argument("--records", type=int, default=400000, help="Number of records to process")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate raw dataset")
    args = parser.parse_args()
    
    run_full_mlops_pipeline(records_count=args.records, force_regenerate=args.force_regenerate)
