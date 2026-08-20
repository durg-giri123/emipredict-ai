"""
Cloud startup script — runs automatically when models are missing.
Used on Streamlit Cloud where pre-trained models aren't available.
Trains on 50K records (takes ~2-3 min) instead of 400K to stay within cloud limits.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


def models_exist():
    """Check if production models are already available."""
    clf = os.path.join(PROJECT_ROOT, "artifacts/production_models/best_classifier.joblib")
    reg = os.path.join(PROJECT_ROOT, "artifacts/production_models/best_regressor.joblib")
    pre = os.path.join(PROJECT_ROOT, "artifacts/preprocessor.joblib")
    return os.path.exists(clf) and os.path.exists(reg) and os.path.exists(pre)


def run_cloud_setup(records: int = 10000):
    """
    Runs a lightweight version of the pipeline suitable for cloud environments.
    Trains on 50K records by default — fast enough to complete on Streamlit Cloud.
    """
    print(f"[STARTUP] Models not found. Running cloud setup with {records:,} records...")
    start = time.time()

    from src.data.data_generator import generate_emi_dataset
    from src.data.preprocessor import DataPreprocessor, DataQualityAuditor
    from src.models.train_classifier import train_all_classifiers
    from src.models.train_regressor import train_all_regressors
    from src.mlops.mlflow_tracking import MLflowTracker

    import joblib
    import numpy as np

    # Dirs
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts/production_models", exist_ok=True)
    os.makedirs("artifacts/visualizations", exist_ok=True)

    # 1. Generate data
    print(f"[STARTUP] Generating {records:,} synthetic records...")
    raw_path = "data/raw/emi_dataset_400k.parquet"
    sample_path = "data/processed/dataset_sample_50k.parquet"

    df = generate_emi_dataset(total_records=records, output_dir="data/raw", random_state=42)

    # Save sample for UI
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    sample_df.to_parquet(sample_path, index=False)
    print(f"[STARTUP] Dataset ready: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Preprocess
    print("[STARTUP] Fitting preprocessor...")
    preprocessor = DataPreprocessor()
    split_data = preprocessor.prepare_train_test_val_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    preprocessor.save("artifacts/preprocessor.joblib")

    # 3. Data quality audit
    audit = DataQualityAuditor.audit(df)
    with open("artifacts/data_quality_audit.json", "w") as f:
        json.dump(audit, f, indent=4)

    # 4. Train models
    print("[STARTUP] Training classification models...")
    clf_results = train_all_classifiers(
        X_train=split_data["X_train"], y_train=split_data["y_train_clf"],
        X_val=split_data["X_val"],   y_val=split_data["y_val_clf"],
        X_test=split_data["X_test"], y_test=split_data["y_test_clf"]
    )

    print("[STARTUP] Training regression models...")
    reg_results = train_all_regressors(
        X_train=split_data["X_train"], y_train=split_data["y_train_reg"],
        X_val=split_data["X_val"],   y_val=split_data["y_val_reg"],
        X_test=split_data["X_test"], y_test=split_data["y_test_reg"]
    )

    # 5. Select best models
    best_clf_name = max(clf_results, key=lambda m: clf_results[m]["val_metrics"]["f1_weighted"])
    best_reg_name = min(reg_results, key=lambda m: reg_results[m]["val_metrics"]["rmse"])
    best_clf = clf_results[best_clf_name]["model"]
    best_reg = reg_results[best_reg_name]["model"]

    joblib.dump(best_clf, "artifacts/production_models/best_classifier.joblib")
    joblib.dump(best_reg, "artifacts/production_models/best_regressor.joblib")

    # 6. Save registry + benchmark
    registry = {
        "best_classifier": {
            "model_name": best_clf_name,
            "val_f1_weighted": clf_results[best_clf_name]["val_metrics"]["f1_weighted"],
            "test_accuracy": clf_results[best_clf_name]["test_metrics"]["accuracy"],
        },
        "best_regressor": {
            "model_name": best_reg_name,
            "val_rmse": reg_results[best_reg_name]["val_metrics"]["rmse"],
            "test_r2": reg_results[best_reg_name]["test_metrics"]["r2_score"],
        }
    }
    with open("artifacts/production_models/model_registry.json", "w") as f:
        json.dump(registry, f, indent=4)

    benchmark = {
        "dataset_size": len(df),
        "pipeline_execution_time_sec": round(time.time() - start, 2),
        "classification_models": {
            k: {
                "test_accuracy": round(v["test_metrics"]["accuracy"], 4),
                "test_f1_weighted": round(v["test_metrics"]["f1_weighted"], 4),
                "val_f1_weighted": round(v["val_metrics"]["f1_weighted"], 4),
                "train_time_sec": v["train_time_sec"]
            } for k, v in clf_results.items()
        },
        "regression_models": {
            k: {
                "test_rmse": round(v["test_metrics"]["rmse"], 2),
                "test_mae": round(v["test_metrics"]["mae"], 2),
                "test_r2": round(v["test_metrics"]["r2_score"], 4),
                "val_rmse": round(v["val_metrics"]["rmse"], 2),
                "train_time_sec": v["train_time_sec"]
            } for k, v in reg_results.items()
        },
        "production_registry": registry
    }
    with open("artifacts/benchmark_summary.json", "w") as f:
        json.dump(benchmark, f, indent=4)

    elapsed = round(time.time() - start, 1)
    print(f"[STARTUP] Setup complete in {elapsed}s — best classifier: {best_clf_name}, best regressor: {best_reg_name}")
    return True


if __name__ == "__main__":
    run_cloud_setup()
