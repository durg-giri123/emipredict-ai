"""Page 5: Application Data Management (Complete CRUD Operations)."""

import os
import sys
import datetime
import streamlit as st
import pandas as pd

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from components.cards import render_header, render_metric_card
except ImportError:
    from app.components.cards import render_header, render_metric_card

from src.database.crud_manager import FinancialDatabaseManager

st.set_page_config(page_title="Data Management (CRUD) - EMIPredict AI", page_icon="📁", layout="wide")

render_header(
    title="📁 Financial Application Data Management (CRUD)",
    subtitle="Create new loan applications, Read & filter portfolio records, Update underwriting decisions, and Delete cases",
    badge="Database CRUD"
)

db = FinancialDatabaseManager(db_path=os.path.join(PROJECT_ROOT, "data/database/emipredict_applications.db"))
db.seed_initial_demo_records(20)
stats = db.get_summary_statistics()

# Top Stats
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Total Stored Cases", f"{stats['total_applications']}", delta="In SQLite", color="#2563eb", icon="🗄️")
with c2:
    render_metric_card("Approved Volume", f"₹ {stats['total_disbursed_volume']:,.0f}", delta=f"{stats['approved_count']} loans", color="#10b981", icon="💵")
with c3:
    render_metric_card("Under Review", f"{stats['under_review_count']}", delta="Pending Action", color="#f59e0b", icon="⏳")
with c4:
    render_metric_card("Rejected Cases", f"{stats['rejected_count']}", delta="High Risk", color="#ef4444", icon="❌")

st.markdown("---")

crud_tab1, crud_tab2, crud_tab3, crud_tab4 = st.tabs([
    "🔍 [READ] View & Search Applications",
    "➕ [CREATE] New Loan Application",
    "✏️ [UPDATE] Edit Application & Decision",
    "🗑️ [DELETE] Remove Application Records"
])

# -------------------------------------------------------------
# TAB 1: READ / SEARCH
# -------------------------------------------------------------
with crud_tab1:
    st.markdown("### 🔍 Search & Filter Stored Loan Applications")
    
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        search_query = st.text_input("Search Applicant Name or ID", placeholder="e.g. Aarav, APP-2026...")
    with f_col2:
        filter_scenario = st.selectbox(
            "Filter Scenario", 
            ["All Scenarios", "Vehicle", "Home Appliances", "Personal Loan", "E-commerce Shopping", "Education"]
        )
    with f_col3:
        filter_status = st.selectbox(
            "Filter Underwriting Status", 
            ["All Statuses", "Approved", "Under Review", "Rejected"]
        )
    with f_col4:
        filter_elig = st.selectbox(
            "Filter AI Eligibility", 
            ["All", "Eligible", "High_Risk", "Not_Eligible"]
        )
        
    records_df = db.search_applications(
        query_text=search_query,
        scenario=filter_scenario,
        eligibility=filter_elig,
        status=filter_status,
        limit=500
    )
    
    st.markdown(f"**Found {len(records_df)} records matching filter criteria.**")
    
    if not records_df.empty:
        # Display table with formatting
        display_df = records_df[[
            "application_id", "applicant_name", "age", "monthly_salary",
            "credit_score", "emi_scenario", "requested_amount", "requested_tenure",
            "predicted_eligibility", "predicted_max_emi", "underwriting_status", "created_at"
        ]].copy()
        
        display_df["monthly_salary"] = display_df["monthly_salary"].apply(lambda v: f"₹ {v:,.0f}")
        display_df["requested_amount"] = display_df["requested_amount"].apply(lambda v: f"₹ {v:,.0f}")
        display_df["predicted_max_emi"] = display_df["predicted_max_emi"].apply(lambda v: f"₹ {v:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export Actions
        st.markdown("##### 📤 Export Data")
        exp_col1, exp_col2, _ = st.columns([1, 1, 2])
        with exp_col1:
            csv_data = records_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name=f"emipredict_applications_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with exp_col2:
            json_data = records_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                label="📥 Export to JSON",
                data=json_data,
                file_name=f"emipredict_applications_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.warning("No applications found matching the selected filter criteria.")

# -------------------------------------------------------------
# TAB 2: CREATE
# -------------------------------------------------------------
with crud_tab2:
    st.markdown("### ➕ Add New Customer Loan Application")
    with st.form("create_app_form"):
        c_name = st.text_input("Applicant Full Name", value="Rohan Singhania")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            c_age = st.number_input("Age", 21, 75, 32)
            c_gender = st.selectbox("Gender", ["Male", "Female"], key="c_gender")
            c_marital = st.selectbox("Marital Status", ["Single", "Married"], key="c_marital")
            c_edu = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"], key="c_edu")
        with col_c2:
            c_salary = st.number_input("Monthly Salary (INR)", 15000, 1000000, 75000, step=5000)
            c_emp_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"], key="c_emp")
            c_exp = st.number_input("Experience (Years)", 0.0, 40.0, 6.0, step=0.5)
            c_comp = st.selectbox("Company Type", ["Startup", "Small-Scale", "Mid-Size", "MNC", "Government"], index=3, key="c_comp")
        with col_c3:
            c_house = st.selectbox("House Status", ["Rented", "Own", "Family"], key="c_house")
            c_rent = st.number_input("Monthly Rent", 0, 100000, 15000 if c_house == "Rented" else 0)
            c_family = st.number_input("Family Size", 1, 10, 3)
            c_deps = st.number_input("Dependents", 0, 8, 1)
            
        st.markdown("###### Loan Details & Financials")
        col_c4, col_c5, col_c6 = st.columns(3)
        with col_c4:
            c_scenario = st.selectbox("EMI Scenario", ["Vehicle", "Home Appliances", "Personal Loan", "E-commerce Shopping", "Education"], key="c_scen")
            c_req_amt = st.number_input("Requested Loan Amount (INR)", 10000, 2000000, 350000, step=10000)
            c_tenure = st.number_input("Requested Tenure (Months)", 3, 84, 36)
        with col_c5:
            c_credit = st.number_input("Credit Score", 300, 850, 760)
            c_existing = st.selectbox("Existing Loans?", ["No", "Yes"], key="c_exist")
            c_curr_emi = st.number_input("Current EMI Amount", 0, 100000, 0)
        with col_c6:
            c_bank = st.number_input("Bank Balance", 0, 5000000, 200000, step=10000)
            c_emergency = st.number_input("Emergency Fund", 0, 2000000, 100000, step=5000)
            c_status = st.selectbox("Initial Status", ["Approved", "Under Review", "Rejected"], key="c_init_status")

        create_btn = st.form_submit_button("🚀 Submit & Save Application", use_container_width=True)
        if create_btn:
            new_id = db.create_application({
                "applicant_name": c_name, "age": c_age, "gender": c_gender, "marital_status": c_marital,
                "education": c_edu, "monthly_salary": c_salary, "employment_type": c_emp_type,
                "years_of_employment": c_exp, "company_type": c_comp, "house_type": c_house,
                "monthly_rent": c_rent, "family_size": c_family, "dependents": c_deps,
                "school_fees": 3000 if c_deps > 0 else 0, "college_fees": 0,
                "travel_expenses": 3000, "groceries_utilities": 10000, "other_monthly_expenses": 2000,
                "existing_loans": c_existing, "current_emi_amount": c_curr_emi,
                "credit_score": c_credit, "bank_balance": c_bank, "emergency_fund": c_emergency,
                "emi_scenario": c_scenario, "requested_amount": c_req_amt, "requested_tenure": c_tenure,
                "predicted_eligibility": "Eligible" if c_credit >= 700 else "High_Risk",
                "confidence_score": 0.95, "predicted_max_emi": round(c_salary * 0.40),
                "foir_percentage": 25.0, "underwriting_status": c_status,
                "underwriter_notes": "Created manually via Administrative Data Management CRUD interface."
            })
            st.success(f"🎉 Created loan application successfully with ID: **{new_id}**")
            st.rerun()

# -------------------------------------------------------------
# TAB 3: UPDATE
# -------------------------------------------------------------
with crud_tab3:
    st.markdown("### ✏️ Update Existing Loan Application")
    
    edit_app_id = st.text_input("Enter Application ID to Edit", placeholder="e.g. APP-...")
    if edit_app_id:
        app_record = db.get_application_by_id(edit_app_id.strip())
        if app_record:
            st.info(f"Loaded record for: **{app_record['applicant_name']}** ({app_record['emi_scenario']} Loan)")
            
            with st.form("edit_app_form"):
                u_col1, u_col2 = st.columns(2)
                with u_col1:
                    u_salary = st.number_input("Monthly Salary (INR)", value=float(app_record["monthly_salary"]))
                    u_credit = st.number_input("Credit Score", 300, 850, int(app_record["credit_score"]))
                    u_req_amt = st.number_input("Requested Loan Amount (INR)", value=float(app_record["requested_amount"]))
                    u_tenure = st.number_input("Requested Tenure (Months)", value=int(app_record["requested_tenure"]))
                with u_col2:
                    current_status = app_record["underwriting_status"]
                    status_idx = ["Approved", "Under Review", "Rejected"].index(current_status) if current_status in ["Approved", "Under Review", "Rejected"] else 0
                    u_status = st.selectbox("Underwriting Decision", ["Approved", "Under Review", "Rejected"], index=status_idx)
                    u_notes = st.text_area("Underwriter Audit Notes", value=app_record["underwriter_notes"] or "")
                    
                update_btn = st.form_submit_button("💾 Save Changes", use_container_width=True)
                if update_btn:
                    success = db.update_application(edit_app_id.strip(), {
                        "monthly_salary": u_salary,
                        "credit_score": u_credit,
                        "requested_amount": u_req_amt,
                        "requested_tenure": u_tenure,
                        "underwriting_status": u_status,
                        "underwriter_notes": u_notes
                    })
                    if success:
                        st.success(f"✅ Application **{edit_app_id}** updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update application.")
        else:
            st.error(f"Application ID '{edit_app_id}' not found in database.")

# -------------------------------------------------------------
# TAB 4: DELETE
# -------------------------------------------------------------
with crud_tab4:
    st.markdown("### 🗑️ Delete Loan Application Record")
    st.warning("⚠️ Deleting an application record is irreversible. Please verify the Application ID before confirming.")
    
    del_app_id = st.text_input("Enter Application ID to Delete", key="del_input")
    if del_app_id:
        record_to_del = db.get_application_by_id(del_app_id.strip())
        if record_to_del:
            st.write(f"Applicant Name: **{record_to_del['applicant_name']}** | Status: **{record_to_del['underwriting_status']}** | Requested Amount: ₹ {record_to_del['requested_amount']:,.0f}")
            confirm_check = st.checkbox(f"I confirm that I want to delete record {del_app_id.strip()}")
            if confirm_check and st.button("🚨 Permanently Delete Record", type="primary"):
                deleted = db.delete_application(del_app_id.strip())
                if deleted:
                    st.success(f"Record **{del_app_id}** permanently deleted from database.")
                    st.rerun()
                else:
                    st.error("Failed to delete record.")
        else:
            st.error(f"Application ID '{del_app_id}' not found.")
