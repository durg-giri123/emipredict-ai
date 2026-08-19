"""Tests for Feature Engineering and Financial Ratios."""

import pytest
import pandas as pd
import numpy as np

from src.data.data_generator import generate_emi_dataset
from src.features.feature_engineering import FinancialFeatureEngineer, compute_estimated_emi


def test_feature_engineering_transforms():
    """Verify all engineered financial ratios and metrics are calculated without NaN."""
    df = generate_emi_dataset(total_records=200, random_state=42)
    fe = FinancialFeatureEngineer()
    transformed = fe.transform(df)
    
    expected_engineered = [
        "total_monthly_expenses", "disposable_income", "estimated_requested_emi",
        "debt_to_income_ratio", "expense_to_income_ratio", "foir",
        "affordability_ratio", "savings_to_income_ratio", "emergency_runway_months",
        "employment_stability_score", "loan_to_income_ratio",
        "interaction_salary_credit", "interaction_dti_credit", "credit_score_normalized"
    ]
    
    for col in expected_engineered:
        assert col in transformed.columns, f"Engineered column missing: {col}"
        assert not transformed[col].isnull().any(), f"NaN found in column {col}"


def test_foir_calculation_bounds():
    """Verify FOIR calculation produces non-negative values."""
    df = generate_emi_dataset(total_records=100, random_state=42)
    fe = FinancialFeatureEngineer()
    transformed = fe.transform(df)
    assert (transformed["foir"] >= 0).all()
    assert (transformed["debt_to_income_ratio"] >= 0).all()
