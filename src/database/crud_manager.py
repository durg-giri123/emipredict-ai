"""Database CRUD Operations Manager for EMIPredict AI Platform.

Provides SQLite persistence for customer loan applications, real-time predictions,
underwriting audit logs, and status transitions.
"""

import os
import sqlite3
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


class FinancialDatabaseManager:
    """Manages SQLite database for Loan Applications and Risk Assessments."""
    
    def __init__(self, db_path: str = "data/database/emipredict_applications.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema with indexes and initial sample records if empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_applications (
                    application_id TEXT PRIMARY KEY,
                    applicant_name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    marital_status TEXT NOT NULL,
                    education TEXT NOT NULL,
                    monthly_salary REAL NOT NULL,
                    employment_type TEXT NOT NULL,
                    years_of_employment REAL NOT NULL,
                    company_type TEXT NOT NULL,
                    house_type TEXT NOT NULL,
                    monthly_rent REAL NOT NULL,
                    family_size INTEGER NOT NULL,
                    dependents INTEGER NOT NULL,
                    school_fees REAL NOT NULL,
                    college_fees REAL NOT NULL,
                    travel_expenses REAL NOT NULL,
                    groceries_utilities REAL NOT NULL,
                    other_monthly_expenses REAL NOT NULL,
                    existing_loans TEXT NOT NULL,
                    current_emi_amount REAL NOT NULL,
                    credit_score INTEGER NOT NULL,
                    bank_balance REAL NOT NULL,
                    emergency_fund REAL NOT NULL,
                    emi_scenario TEXT NOT NULL,
                    requested_amount REAL NOT NULL,
                    requested_tenure INTEGER NOT NULL,
                    predicted_eligibility TEXT,
                    confidence_score REAL,
                    predicted_max_emi REAL,
                    foir_percentage REAL,
                    underwriting_status TEXT DEFAULT 'Under Review',
                    underwriter_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Indexes for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON loan_applications(underwriting_status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenario ON loan_applications(emi_scenario);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON loan_applications(created_at);")
            conn.commit()

    def create_application(self, data: Dict[str, Any]) -> str:
        """Insert a new loan application record."""
        app_id = data.get("application_id")
        if not app_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            random_suffix = os.urandom(2).hex().upper()
            app_id = f"APP-{timestamp}-{random_suffix}"
            data["application_id"] = app_id

        columns = [
            "application_id", "applicant_name", "age", "gender", "marital_status", "education",
            "monthly_salary", "employment_type", "years_of_employment", "company_type",
            "house_type", "monthly_rent", "family_size", "dependents",
            "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
            "other_monthly_expenses", "existing_loans", "current_emi_amount",
            "credit_score", "bank_balance", "emergency_fund",
            "emi_scenario", "requested_amount", "requested_tenure",
            "predicted_eligibility", "confidence_score", "predicted_max_emi",
            "foir_percentage", "underwriting_status", "underwriter_notes"
        ]

        values = [data.get(col, None) for col in columns]
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)

        query = f"INSERT INTO loan_applications ({col_names}) VALUES ({placeholders})"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()

        return app_id

    def get_application_by_id(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single application by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM loan_applications WHERE application_id = ?", (app_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def search_applications(
        self,
        query_text: Optional[str] = None,
        scenario: Optional[str] = None,
        eligibility: Optional[str] = None,
        status: Optional[str] = None,
        min_credit: Optional[int] = None,
        max_credit: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Search and filter loan application records."""
        conditions = ["1=1"]
        params = []

        if query_text:
            conditions.append("(application_id LIKE ? OR applicant_name LIKE ?)")
            params.extend([f"%{query_text}%", f"%{query_text}%"])

        if scenario and scenario != "All Scenarios":
            conditions.append("emi_scenario = ?")
            params.append(scenario)

        if eligibility and eligibility != "All":
            conditions.append("predicted_eligibility = ?")
            params.append(eligibility)

        if status and status != "All Statuses":
            conditions.append("underwriting_status = ?")
            params.append(status)

        if min_credit is not None:
            conditions.append("credit_score >= ?")
            params.append(min_credit)

        if max_credit is not None:
            conditions.append("credit_score <= ?")
            params.append(max_credit)

        sql = f"""
            SELECT * FROM loan_applications 
            WHERE {" AND ".join(conditions)} 
            ORDER BY created_at DESC 
            LIMIT ?
        """
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
            return df

    def update_application(self, app_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields of an existing loan application."""
        if not updates:
            return False

        updates["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        set_clauses = [f"{k} = ?" for k in updates.keys()]
        values = list(updates.values()) + [app_id]

        query = f"UPDATE loan_applications SET {', '.join(set_clauses)} WHERE application_id = ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_application(self, app_id: str) -> bool:
        """Delete an application record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM loan_applications WHERE application_id = ?", (app_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Aggregate high-level loan metrics for dashboards."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM loan_applications")
            total_apps = cursor.fetchone()[0]

            if total_apps == 0:
                return {
                    "total_applications": 0,
                    "approved_count": 0,
                    "rejected_count": 0,
                    "under_review_count": 0,
                    "total_disbursed_volume": 0.0,
                    "avg_credit_score": 0.0,
                    "approval_rate": 0.0
                }

            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN underwriting_status = 'Approved' THEN 1 END) as approved,
                    COUNT(CASE WHEN underwriting_status = 'Rejected' THEN 1 END) as rejected,
                    COUNT(CASE WHEN underwriting_status = 'Under Review' THEN 1 END) as under_review,
                    SUM(CASE WHEN underwriting_status = 'Approved' THEN requested_amount ELSE 0 END) as total_volume,
                    AVG(credit_score) as avg_score
                FROM loan_applications
            """)
            row = cursor.fetchone()
            
            return {
                "total_applications": total_apps,
                "approved_count": row[0],
                "rejected_count": row[1],
                "under_review_count": row[2],
                "total_disbursed_volume": float(row[3] or 0),
                "avg_credit_score": round(float(row[4] or 0), 1),
                "approval_rate": round((row[0] / total_apps) * 100, 1)
            }

    def seed_initial_demo_records(self, n: int = 25):
        """Populate realistic sample records if database is freshly created."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM loan_applications")
            if cursor.fetchone()[0] >= n:
                return

        sample_names = [
            "Aarav Sharma", "Priya Patel", "Vikram Malhotra", "Ananya Iyer", "Rahul Verma",
            "Sneha Kulkarni", "Rohan Gupta", "Deepika Nair", "Arjun Reddy", "Pooja Banerjee",
            "Kavita Mehta", "Siddharth Rao", "Neha Saxena", "Aditya Joshi", "Divya Menon",
            "Gaurav Choudhury", "Meera Nambiar", "Karan Kapoor", "Simran Kaur", "Nikhil Sen"
        ]
        
        scenarios = ["Vehicle", "Home Appliances", "Personal Loan", "E-commerce Shopping", "Education"]
        
        for i in range(min(n, len(sample_names))):
            name = sample_names[i]
            age = 26 + (i * 2) % 30
            salary = 35000 + (i * 7500)
            score = 620 + (i * 12) % 220
            scenario = scenarios[i % len(scenarios)]
            req_amt = 50000 + (i * 30000)
            tenure = 12 + (i * 6) % 36
            
            eligibility = "Eligible" if score >= 700 and salary >= 45000 else ("High_Risk" if score >= 600 else "Not_Eligible")
            status = "Approved" if eligibility == "Eligible" else ("Under Review" if eligibility == "High_Risk" else "Rejected")
            
            self.create_application({
                "applicant_name": name,
                "age": age,
                "gender": "Male" if i % 2 == 0 else "Female",
                "marital_status": "Married" if age > 30 else "Single",
                "education": "Graduate" if i % 3 == 0 else "Post Graduate",
                "monthly_salary": salary,
                "employment_type": "Private" if i % 4 != 0 else "Government",
                "years_of_employment": max(1, age - 23),
                "company_type": "MNC" if i % 2 == 0 else "Mid-Size",
                "house_type": "Rented" if i % 2 == 0 else "Own",
                "monthly_rent": 12000 if i % 2 == 0 else 0,
                "family_size": 3,
                "dependents": 1,
                "school_fees": 3000,
                "college_fees": 0,
                "travel_expenses": 2500,
                "groceries_utilities": 8500,
                "other_monthly_expenses": 2000,
                "existing_loans": "Yes" if i % 3 == 0 else "No",
                "current_emi_amount": 5000 if i % 3 == 0 else 0,
                "credit_score": score,
                "bank_balance": salary * 3,
                "emergency_fund": salary * 1.5,
                "emi_scenario": scenario,
                "requested_amount": req_amt,
                "requested_tenure": tenure,
                "predicted_eligibility": eligibility,
                "confidence_score": 0.94 if eligibility == "Eligible" else 0.88,
                "predicted_max_emi": round(salary * 0.40),
                "foir_percentage": round((req_amt / tenure) / salary * 100, 1),
                "underwriting_status": status,
                "underwriter_notes": f"Automated risk evaluation completed for {scenario} loan."
            })
