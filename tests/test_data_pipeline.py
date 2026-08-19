"""Tests for Data Generator and Preprocessing Pipeline."""

import os
import pytest
import pandas as pd
import numpy as np

from src.data.data_generator import generate_emi_dataset
from src.data.preprocessor import DataPreprocessor, DataQualityAuditor


def test_data_generator_shape_and_columns():
    """Verify generated dataset contains all required 22 features and 2 targets."""
    df = generate_emi_dataset(total_records=500, random_state=42)
    assert len(df) == 500
    
    # 22 input features + 2 targets
    expected_cols = [
        "age", "gender", "marital_status", "education",
        "monthly_salary", "employment_type", "years_of_employment", "company_type",
        "house_type", "monthly_rent", "family_size", "dependents",
        "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses",
        "existing_loans", "current_emi_amount", "credit_score",
        "bank_balance", "emergency_fund", "emi_scenario",
        "requested_amount", "requested_tenure",
        "emi_eligibility", "max_monthly_emi"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_scenario_distribution():
    """Verify all 5 EMI scenarios are evenly generated."""
    df = generate_emi_dataset(total_records=1000, random_state=42)
    scenarios = df["emi_scenario"].unique()
    assert len(scenarios) == 5
    assert "E-commerce Shopping" in scenarios
    assert "Home Appliances" in scenarios
    assert "Vehicle" in scenarios
    assert "Personal Loan" in scenarios
    assert "Education" in scenarios


def test_data_quality_audit():
    """Verify quality auditor flags no critical errors on clean data."""
    df = generate_emi_dataset(total_records=200, random_state=42)
    audit = DataQualityAuditor.audit(df)
    assert audit["quality_score"] >= 95.0
    assert audit["total_records"] == 200


def test_preprocessor_splits_and_transformation():
    """Verify train-val-test splits and preprocessing transformations."""
    df = generate_emi_dataset(total_records=1000, random_state=42)
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_train_test_val_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    
    assert splits["X_train"].shape[0] == 700
    assert splits["X_val"].shape[0] == 150
    assert splits["X_test"].shape[0] == 150
    assert splits["y_train_clf"].shape[0] == 700
    assert splits["y_train_reg"].shape[0] == 700
    assert preprocessor.is_fitted
