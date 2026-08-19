"""Data Generator for EMIPredict AI Platform.

Generates a realistic, statistically coherent financial dataset of 400,000 records
across 5 lending scenarios with 22 financial and demographic variables.
"""

import numpy as np
import pandas as pd
import os
import argparse
from typing import Optional, Tuple


def calculate_amortized_emi(principal: np.ndarray, tenure_months: np.ndarray, annual_rate: float = 0.12) -> np.ndarray:
    """Calculate standard monthly EMI using annuity formula: E = P * r * (1+r)^n / ((1+r)^n - 1)"""
    monthly_rate = annual_rate / 12.0
    r_factor = (1 + monthly_rate) ** tenure_months
    emi = principal * (monthly_rate * r_factor) / (r_factor - 1)
    return np.round(emi, 2)


def generate_emi_dataset(
    total_records: int = 400000,
    random_state: int = 42,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate 400,000 realistic financial profiles across 5 distinct EMI scenarios.
    
    Scenarios (80k each):
    1. E-commerce Shopping: 10K-200K INR, 3-24 months
    2. Home Appliances: 20K-300K INR, 6-36 months
    3. Vehicle: 80K-1500K INR, 12-84 months
    4. Personal Loan: 50K-1000K INR, 12-60 months
    5. Education: 50K-500K INR, 6-48 months
    """
    np.random.seed(random_state)
    records_per_scenario = total_records // 5

    scenarios_config = [
        {
            "scenario": "E-commerce Shopping",
            "count": records_per_scenario,
            "min_amt": 10000, "max_amt": 200000,
            "min_tenure": 3, "max_tenure": 24,
            "interest_rate": 0.14
        },
        {
            "scenario": "Home Appliances",
            "count": records_per_scenario,
            "min_amt": 20000, "max_amt": 300000,
            "min_tenure": 6, "max_tenure": 36,
            "interest_rate": 0.13
        },
        {
            "scenario": "Vehicle",
            "count": records_per_scenario,
            "min_amt": 80000, "max_amt": 1500000,
            "min_tenure": 12, "max_tenure": 84,
            "interest_rate": 0.095
        },
        {
            "scenario": "Personal Loan",
            "count": records_per_scenario,
            "min_amt": 50000, "max_amt": 1000000,
            "min_tenure": 12, "max_tenure": 60,
            "interest_rate": 0.125
        },
        {
            "scenario": "Education",
            "count": records_per_scenario,
            "min_amt": 50000, "max_amt": 500000,
            "min_tenure": 6, "max_tenure": 48,
            "interest_rate": 0.105
        }
    ]

    dfs = []
    
    for cfg in scenarios_config:
        n = cfg["count"]
        
        # 1. Personal Demographics
        age = np.random.randint(25, 61, size=n)
        gender = np.random.choice(["Male", "Female"], size=n, p=[0.62, 0.38])
        marital_status = np.where(age > 30, 
                                  np.random.choice(["Married", "Single"], size=n, p=[0.75, 0.25]),
                                  np.random.choice(["Single", "Married"], size=n, p=[0.65, 0.35]))
        
        education_levels = ["High School", "Graduate", "Post Graduate", "Professional"]
        education = np.random.choice(education_levels, size=n, p=[0.15, 0.50, 0.25, 0.10])
        
        # 2. Employment and Income
        # Base salary driven by education, age, and log-normal variance
        edu_salary_multiplier = {
            "High School": 0.65,
            "Graduate": 1.0,
            "Post Graduate": 1.45,
            "Professional": 1.85
        }
        edu_mult = np.array([edu_salary_multiplier[e] for e in education])
        exp_factor = 1 + (age - 25) * 0.025
        
        raw_salary = np.random.lognormal(mean=10.2, sigma=0.45, size=n) * edu_mult * exp_factor
        monthly_salary = np.clip(np.round(raw_salary / 500) * 500, 15000, 200000).astype(int)
        
        employment_type = np.random.choice(
            ["Private", "Government", "Self-employed"], 
            size=n, 
            p=[0.65, 0.20, 0.15]
        )
        
        # Years of employment bounded by age
        max_possible_exp = np.maximum(1, age - 21)
        years_of_employment = np.clip(
            np.random.randint(0, 35, size=n), 
            0, 
            max_possible_exp
        )
        
        company_type = np.random.choice(
            ["Startup", "Small-Scale", "Mid-Size", "MNC", "Government"],
            size=n,
            p=[0.20, 0.25, 0.25, 0.20, 0.10]
        )
        
        # 3. Housing and Family
        house_type = np.random.choice(["Rented", "Own", "Family"], size=n, p=[0.45, 0.35, 0.20])
        
        # Rent logic: Rented pays 10-25% of salary, Own/Family = 0
        rent_ratio = np.random.uniform(0.10, 0.25, size=n)
        monthly_rent = np.where(
            house_type == "Rented",
            np.clip(np.round((monthly_salary * rent_ratio) / 500) * 500, 3000, 45000),
            0
        ).astype(int)
        
        family_size = np.where(
            marital_status == "Married",
            np.random.choice([2, 3, 4, 5, 6, 7], size=n, p=[0.20, 0.35, 0.30, 0.10, 0.03, 0.02]),
            np.random.choice([1, 2, 3, 4], size=n, p=[0.55, 0.25, 0.15, 0.05])
        )
        
        dependents = np.clip(
            np.where(
                marital_status == "Married",
                np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.15, 0.40, 0.35, 0.08, 0.02]),
                np.random.choice([0, 1, 2], size=n, p=[0.75, 0.20, 0.05])
            ),
            0,
            np.maximum(0, family_size - 1)
        )
        
        # 4. Monthly Financial Obligations
        # School fees: only if dependents > 0
        school_fees = np.where(
            dependents > 0,
            np.clip(np.round(np.random.uniform(1500, 8000, size=n) * dependents / 500) * 500, 1000, 25000),
            0
        ).astype(int)
        
        # College fees: higher education (probabilistic for older parents / higher education)
        has_college = (age >= 40) & (dependents > 0) & (np.random.rand(n) < 0.25)
        college_fees = np.where(
            has_college,
            np.clip(np.round(np.random.uniform(4000, 20000, size=n) / 500) * 500, 2000, 35000),
            0
        ).astype(int)
        
        # Travel expenses
        travel_base = np.random.uniform(0.02, 0.08, size=n) * monthly_salary
        travel_expenses = np.clip(np.round(travel_base / 100) * 100, 500, 15000).astype(int)
        
        # Groceries and utilities (scales with family size)
        grocery_base = (3000 + family_size * 2200 + monthly_salary * 0.05) * np.random.uniform(0.85, 1.15, size=n)
        groceries_utilities = np.clip(np.round(grocery_base / 500) * 500, 3000, 30000).astype(int)
        
        # Other miscellaneous expenses
        other_exp_base = (monthly_salary * np.random.uniform(0.02, 0.08, size=n)) + 500
        other_monthly_expenses = np.clip(np.round(other_exp_base / 500) * 500, 1000, 20000).astype(int)
        
        # 5. Financial Status and Credit History
        existing_loans_prob = np.where(age > 30, 0.55, 0.35)
        existing_loans = np.where(np.random.rand(n) < existing_loans_prob, "Yes", "No")
        
        # Current EMI burden
        current_emi_ratio = np.random.uniform(0.10, 0.35, size=n)
        current_emi_amount = np.where(
            existing_loans == "Yes",
            np.clip(np.round((monthly_salary * current_emi_ratio) / 500) * 500, 2000, 40000),
            0
        ).astype(int)
        
        # Credit Score (300-850) - correlated with income stability, age, and loan history
        score_mean = 680 + (monthly_salary / 200000.0) * 80 + (years_of_employment * 1.5)
        score_mean = np.where(existing_loans == "Yes", score_mean - 15, score_mean)
        raw_credit_score = np.random.normal(loc=score_mean, scale=75, size=n)
        credit_score = np.clip(np.round(raw_credit_score), 300, 850).astype(int)
        
        # Bank balance and emergency fund
        savings_propensity = np.random.uniform(0.5, 6.0, size=n)
        bank_balance = np.clip(
            np.round((monthly_salary * savings_propensity + np.random.uniform(5000, 100000, size=n)) / 1000) * 1000,
            5000, 1500000
        ).astype(int)
        
        emergency_fund = np.clip(
            np.round((bank_balance * np.random.uniform(0.15, 0.65, size=n)) / 1000) * 1000,
            0, 600000
        ).astype(int)
        
        # 6. Loan Application Details
        requested_amount = np.clip(
            np.round(np.random.uniform(cfg["min_amt"], cfg["max_amt"], size=n) / 1000) * 1000,
            cfg["min_amt"], cfg["max_amt"]
        ).astype(int)
        
        requested_tenure = np.random.randint(cfg["min_tenure"], cfg["max_tenure"] + 1, size=n)
        
        # 7. Dual Targets Computation
        # Calculate expected EMI for this requested loan
        req_emi = calculate_amortized_emi(requested_amount, requested_tenure, annual_rate=cfg["interest_rate"])
        
        total_mandatory_expenses = (
            monthly_rent + school_fees + college_fees + travel_expenses + 
            groceries_utilities + other_monthly_expenses + current_emi_amount
        )
        disposable_surplus = monthly_salary - total_mandatory_expenses
        total_projected_emi = current_emi_amount + req_emi
        foir = total_projected_emi / monthly_salary
        dti = total_projected_emi / monthly_salary
        
        # Classification Target: emi_eligibility (3 classes)
        # Eligibility Score [0 to 100]
        credit_score_norm = (credit_score - 300) / 550.0  # 0 to 1
        surplus_ratio = np.clip(disposable_surplus / (req_emi + 1e-5), -2.0, 5.0)
        
        # Comprehensive banking risk score
        risk_score = (
            (1.0 - credit_score_norm) * 35.0 +
            np.clip(foir * 50.0, 0, 45) +
            np.where(disposable_surplus < req_emi, 30.0, 0.0) +
            np.where(monthly_salary < 20000, 15.0, 0.0) +
            np.where(years_of_employment < 1, 10.0, 0.0) -
            np.clip(emergency_fund / (req_emi * 6 + 1e-5) * 10.0, 0, 15.0)
        )
        
        # Add slight statistical noise
        risk_score += np.random.normal(0, 3.5, size=n)
        
        # Assign 3 Classes
        eligibility = np.empty(n, dtype=object)
        eligibility[risk_score < 38] = "Eligible"
        eligibility[(risk_score >= 38) & (risk_score < 58)] = "High_Risk"
        eligibility[risk_score >= 58] = "Not_Eligible"
        
        # Hard regulatory constraints
        eligibility[credit_score < 550] = "Not_Eligible"
        eligibility[disposable_surplus < (req_emi * 0.7)] = "Not_Eligible"
        eligibility[(credit_score >= 750) & (foir < 0.40) & (disposable_surplus > req_emi * 2)] = "Eligible"
        
        # Regression Target: max_monthly_emi (Continuous 500 - 50,000 INR)
        # Standard Banking FOIR limit: 40% - 60% based on salary and credit score
        base_foir_limit = 0.35 + (credit_score_norm * 0.20) + np.clip(monthly_salary / 400000.0, 0, 0.10)
        max_allowed_total_emi = monthly_salary * base_foir_limit
        raw_max_emi = max_allowed_total_emi - current_emi_amount
        
        # Cash flow cap: max EMI shouldn't exceed 65% of current disposable surplus + current_emi
        cash_flow_cap = np.maximum(0, disposable_surplus) * 0.65
        calculated_max_emi = np.minimum(raw_max_emi, cash_flow_cap)
        
        # Realistic calibration noise and bounds [500, 50000]
        calculated_max_emi += np.random.normal(0, 250, size=n)
        max_monthly_emi = np.clip(np.round(calculated_max_emi / 50) * 50, 500, 50000)
        
        scenario_df = pd.DataFrame({
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": cfg["scenario"],
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "emi_eligibility": eligibility,
            "max_monthly_emi": max_monthly_emi
        })
        
        dfs.append(scenario_df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    # Shuffle records to ensure mixed scenario distribution
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "emi_dataset_400k.csv")
        parquet_path = os.path.join(output_dir, "emi_dataset_400k.parquet")
        
        print(f"Saving dataset to {parquet_path} and {csv_path}...")
        full_df.to_parquet(parquet_path, index=False)
        full_df.to_csv(csv_path, index=False)
        print(f"Successfully generated and saved {len(full_df):,} records!")
        
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 400K EMI Financial Records")
    parser.add_argument("--records", type=int, default=400000, help="Total records to generate")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    generate_emi_dataset(total_records=args.records, output_dir=args.output_dir)
