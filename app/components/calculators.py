"""Loan Amortization & Affordability Calculators for EMIPredict AI."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def calculate_loan_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Calculate monthly EMI using standard banking amortization formula."""
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    r_factor = (1 + monthly_rate) ** tenure_months
    emi = principal * (monthly_rate * r_factor) / (r_factor - 1)
    return round(float(emi), 2)


def generate_amortization_schedule(
    principal: float, 
    annual_rate_pct: float, 
    tenure_months: int
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate detailed monthly amortization schedule and cost summary."""
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    emi = calculate_loan_emi(principal, annual_rate_pct, tenure_months)
    
    schedule = []
    balance = principal
    total_interest = 0.0
    
    for month in range(1, tenure_months + 1):
        interest_payment = balance * monthly_rate
        principal_payment = emi - interest_payment
        balance = max(0.0, balance - principal_payment)
        total_interest += interest_payment
        
        schedule.append({
            "Month": month,
            "EMI (INR)": round(emi, 2),
            "Principal (INR)": round(principal_payment, 2),
            "Interest (INR)": round(interest_payment, 2),
            "Remaining Balance (INR)": round(balance, 2)
        })
        
    df_schedule = pd.DataFrame(schedule)
    summary = {
        "monthly_emi": emi,
        "total_principal": principal,
        "total_interest": round(total_interest, 2),
        "total_payment": round(principal + total_interest, 2),
        "interest_ratio_pct": round((total_interest / (principal + total_interest)) * 100, 1)
    }
    
    return df_schedule, summary


def calculate_max_loan_capacity(
    max_monthly_emi: float, 
    annual_rate_pct: float, 
    tenure_months: int
) -> float:
    """Calculate maximum loan principal an applicant can borrow given their safe max EMI."""
    if max_monthly_emi <= 0 or tenure_months <= 0:
        return 0.0
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    r_factor = (1 + monthly_rate) ** tenure_months
    principal = max_monthly_emi * (r_factor - 1) / (monthly_rate * r_factor)
    return round(float(principal), 2)
