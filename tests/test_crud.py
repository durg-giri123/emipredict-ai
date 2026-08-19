"""Tests for Database CRUD Operations."""

import os
import pytest
from src.database.crud_manager import FinancialDatabaseManager


@pytest.fixture
def test_db(tmp_path):
    db_file = os.path.join(tmp_path, "test_applications.db")
    manager = FinancialDatabaseManager(db_path=db_file)
    return manager


def test_crud_lifecycle(test_db):
    """Verify complete Create, Read, Update, and Delete operations."""
    # 1. CREATE
    record = {
        "applicant_name": "Test User",
        "age": 30,
        "gender": "Male",
        "marital_status": "Single",
        "education": "Graduate",
        "monthly_salary": 50000,
        "employment_type": "Private",
        "years_of_employment": 4.0,
        "company_type": "MNC",
        "house_type": "Rented",
        "monthly_rent": 10000,
        "family_size": 2,
        "dependents": 0,
        "school_fees": 0,
        "college_fees": 0,
        "travel_expenses": 2000,
        "groceries_utilities": 6000,
        "other_monthly_expenses": 1000,
        "existing_loans": "No",
        "current_emi_amount": 0,
        "credit_score": 750,
        "bank_balance": 150000,
        "emergency_fund": 80000,
        "emi_scenario": "Vehicle",
        "requested_amount": 200000,
        "requested_tenure": 24,
        "predicted_eligibility": "Eligible",
        "confidence_score": 0.95,
        "predicted_max_emi": 20000,
        "foir_percentage": 18.5,
        "underwriting_status": "Under Review",
        "underwriter_notes": "Test application notes"
    }
    
    app_id = test_db.create_application(record)
    assert app_id.startswith("APP-")
    
    # 2. READ
    fetched = test_db.get_application_by_id(app_id)
    assert fetched is not None
    assert fetched["applicant_name"] == "Test User"
    assert fetched["credit_score"] == 750
    
    # Search
    search_res = test_db.search_applications(query_text="Test User")
    assert len(search_res) == 1
    
    # 3. UPDATE
    updated = test_db.update_application(app_id, {
        "underwriting_status": "Approved",
        "underwriter_notes": "Manually verified income tax returns."
    })
    assert updated is True
    
    refetched = test_db.get_application_by_id(app_id)
    assert refetched["underwriting_status"] == "Approved"
    assert "tax returns" in refetched["underwriter_notes"]
    
    # 4. DELETE
    deleted = test_db.delete_application(app_id)
    assert deleted is True
    assert test_db.get_application_by_id(app_id) is None
