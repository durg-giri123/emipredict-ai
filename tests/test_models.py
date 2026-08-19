"""Tests for ML Model Training and Evaluation."""

import pytest
import numpy as np
from src.data.data_generator import generate_emi_dataset
from src.data.preprocessor import DataPreprocessor
from src.models.train_classifier import get_classification_models, train_and_evaluate_classifier
from src.models.train_regressor import get_regression_models, train_and_evaluate_regressor
from src.models.evaluate import evaluate_classification_model, evaluate_regression_model


def test_classification_and_regression_models():
    """Verify training and evaluation of baseline and ensemble models on mini sample."""
    df = generate_emi_dataset(total_records=300, random_state=42)
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_train_test_val_splits(df, test_size=0.2, val_size=0.2, random_state=42)
    
    # 1. Test Classifier
    clf_models = get_classification_models()
    lr = clf_models["Logistic_Regression"]
    clf_res = train_and_evaluate_classifier(
        "Logistic_Regression", lr,
        splits["X_train"], splits["y_train_clf"],
        splits["X_val"], splits["y_val_clf"],
        splits["X_test"], splits["y_test_clf"]
    )
    assert clf_res["test_metrics"]["accuracy"] > 0.70
    assert len(clf_res["test_metrics"]["confusion_matrix"]) == 3
    
    # 2. Test Regressor
    reg_models = get_regression_models()
    ridge = reg_models["Linear_Ridge_Regression"]
    reg_res = train_and_evaluate_regressor(
        "Linear_Ridge_Regression", ridge,
        splits["X_train"], splits["y_train_reg"],
        splits["X_val"], splits["y_val_reg"],
        splits["X_test"], splits["y_test_reg"]
    )
    assert reg_res["test_metrics"]["r2_score"] > 0.70
    assert reg_res["test_metrics"]["rmse"] > 0
