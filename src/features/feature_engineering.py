"""Feature Engineering Module for EMIPredict AI.

Calculates domain-specific FinTech financial ratios, credit risk indicators,
stability scores, and interaction terms from the 22 baseline features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def compute_estimated_emi(
    amount: np.ndarray, 
    tenure: np.ndarray, 
    scenario: Optional[np.ndarray] = None,
    default_rate: float = 0.12
) -> np.ndarray:
    """Calculate estimated monthly EMI based on scenario interest rates or default rate."""
    rate_map = {
        "E-commerce Shopping": 0.14,
        "Home Appliances": 0.13,
        "Vehicle": 0.095,
        "Personal Loan": 0.125,
        "Education": 0.105
    }
    
    if scenario is not None and len(scenario) > 0:
        rates = np.array([rate_map.get(str(s), default_rate) for s in scenario])
    else:
        rates = np.full(len(amount), default_rate)
        
    monthly_rates = rates / 12.0
    tenures = np.maximum(1, tenure)
    
    r_factor = (1 + monthly_rates) ** tenures
    emi = amount * (monthly_rates * r_factor) / (r_factor - 1)
    return np.nan_to_num(emi, nan=0.0)


class FinancialFeatureEngineer:
    """Computes advanced banking feature ratios and interaction metrics."""
    
    def __init__(self):
        self.engineered_feature_names: List[str] = [
            "total_monthly_expenses",
            "disposable_income",
            "estimated_requested_emi",
            "debt_to_income_ratio",
            "expense_to_income_ratio",
            "foir",
            "affordability_ratio",
            "savings_to_income_ratio",
            "emergency_runway_months",
            "employment_stability_score",
            "loan_to_income_ratio",
            "interaction_salary_credit",
            "interaction_dti_credit",
            "credit_score_normalized"
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer derived features from raw dataframe without modifying original inplace."""
        data = df.copy()
        
        # 1. Total monthly non-loan living expenses
        living_expenses = (
            data["monthly_rent"].fillna(0) +
            data["school_fees"].fillna(0) +
            data["college_fees"].fillna(0) +
            data["travel_expenses"].fillna(0) +
            data["groceries_utilities"].fillna(0) +
            data["other_monthly_expenses"].fillna(0)
        )
        data["total_monthly_expenses"] = living_expenses
        
        # 2. Total Outflow & Disposable Income
        current_emi = data["current_emi_amount"].fillna(0)
        salary = np.maximum(1.0, data["monthly_salary"].values)
        
        data["disposable_income"] = salary - (living_expenses + current_emi)
        
        # 3. Estimated requested EMI
        req_amount = data["requested_amount"].values
        req_tenure = data["requested_tenure"].values
        scenarios = data["emi_scenario"].values if "emi_scenario" in data.columns else None
        
        est_emi = compute_estimated_emi(req_amount, req_tenure, scenarios)
        data["estimated_requested_emi"] = est_emi
        
        # 4. Debt-to-Income (DTI)
        total_projected_emi = current_emi + est_emi
        data["debt_to_income_ratio"] = np.clip(total_projected_emi / salary, 0.0, 5.0)
        
        # 5. Expense-to-Income (ETI)
        data["expense_to_income_ratio"] = np.clip(living_expenses / salary, 0.0, 5.0)
        
        # 6. Fixed Obligation to Income Ratio (FOIR)
        # In banking: (All EMIs + Rent) / Net Salary
        foir_obligations = current_emi + est_emi + data["monthly_rent"].fillna(0)
        data["foir"] = np.clip(foir_obligations / salary, 0.0, 5.0)
        
        # 7. Affordability Ratio: Disposable Income / Requested EMI
        data["affordability_ratio"] = np.clip(
            data["disposable_income"] / np.maximum(100.0, est_emi), 
            -5.0, 10.0
        )
        
        # 8. Savings & Liquidity Ratios
        total_savings = data["bank_balance"].fillna(0) + data["emergency_fund"].fillna(0)
        annual_salary = salary * 12.0
        data["savings_to_income_ratio"] = np.clip(total_savings / annual_salary, 0.0, 20.0)
        
        total_monthly_burn = np.maximum(100.0, living_expenses + current_emi)
        data["emergency_runway_months"] = np.clip(
            data["emergency_fund"].fillna(0) / total_monthly_burn, 
            0.0, 36.0
        )
        
        # 9. Employment Stability Score
        working_age_span = np.maximum(1.0, data["age"] - 20.0)
        data["employment_stability_score"] = np.clip(
            data["years_of_employment"].fillna(0) / working_age_span,
            0.0, 1.0
        )
        
        # 10. Loan to Annual Income Ratio
        data["loan_to_income_ratio"] = np.clip(req_amount / annual_salary, 0.0, 10.0)
        
        # 11. Credit Score Normalized & Interactions
        credit = data["credit_score"].fillna(650).values
        data["credit_score_normalized"] = np.clip((credit - 300.0) / 550.0, 0.0, 1.0)
        
        data["interaction_salary_credit"] = (salary / 100000.0) * data["credit_score_normalized"]
        data["interaction_dti_credit"] = data["debt_to_income_ratio"] * (1.0 - data["credit_score_normalized"])
        
        return data

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


def get_feature_groups() -> Dict[str, List[str]]:
    """Returns categorizations of all 22 raw features + engineered features."""
    return {
        "demographics": ["age", "gender", "marital_status", "education"],
        "employment": ["monthly_salary", "employment_type", "years_of_employment", "company_type"],
        "housing_family": ["house_type", "monthly_rent", "family_size", "dependents"],
        "obligations": ["school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses"],
        "financial_status": ["existing_loans", "current_emi_amount", "credit_score", "bank_balance", "emergency_fund"],
        "loan_details": ["emi_scenario", "requested_amount", "requested_tenure"],
        "engineered_ratios": [
            "total_monthly_expenses", "disposable_income", "estimated_requested_emi",
            "debt_to_income_ratio", "expense_to_income_ratio", "foir",
            "affordability_ratio", "savings_to_income_ratio", "emergency_runway_months",
            "employment_stability_score", "loan_to_income_ratio",
            "interaction_salary_credit", "interaction_dti_credit", "credit_score_normalized"
        ]
    }
