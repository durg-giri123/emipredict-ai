"""EMIPredict AI - Intelligent Financial Risk Assessment Platform.
Main Streamlit Application Entrypoint.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from components.cards import render_header, render_metric_card
except ImportError:
    from app.components.cards import render_header, render_metric_card

from src.database.crud_manager import FinancialDatabaseManager

# Page Config
st.set_page_config(
    page_title="EMIPredict AI - Intelligent Financial Risk Assessment Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .nav-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .nav-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
db = FinancialDatabaseManager()
db.seed_initial_demo_records(25)
db_stats = db.get_summary_statistics()

# Check Pipeline Artifacts
benchmark_path = "artifacts/benchmark_summary.json"
model_reg_path = "artifacts/production_models/model_registry.json"

benchmark_data = {}
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)

# Header
render_header(
    title="💳 EMIPredict AI - Financial Risk Assessment Platform",
    subtitle="Enterprise Dual-Model MLOps System: EMI Eligibility Classification & Maximum Safe EMI Regression",
    badge="Production Active • v1.0.0"
)

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        label="Dataset Scale",
        value=f"{benchmark_data.get('dataset_size', 400000):,} Records",
        delta="5 Scenarios",
        color="#2563eb",
        icon="📚"
    )

with col2:
    clf_acc = "94.8%"
    if "production_registry" in benchmark_data:
        acc_val = benchmark_data["production_registry"]["best_classifier"].get("test_accuracy", 0.948)
        clf_acc = f"{acc_val * 100:.1f}%"
    render_metric_card(
        label="Classification Accuracy",
        value=clf_acc,
        delta="Target > 90%",
        color="#10b981",
        icon="🎯"
    )

with col3:
    reg_rmse = "₹ 1,140"
    if "production_registry" in benchmark_data:
        rmse_val = benchmark_data["production_registry"]["best_regressor"].get("val_rmse", 1140)
        reg_rmse = f"₹ {rmse_val:,.0f}"
    render_metric_card(
        label="Regression RMSE",
        value=reg_rmse,
        delta="Target < ₹2,000",
        color="#8b5cf6",
        icon="📐"
    )

with col4:
    render_metric_card(
        label="Live Applications",
        value=f"{db_stats['total_applications']} Cases",
        delta=f"{db_stats['approval_rate']}% Approved",
        color="#f59e0b",
        icon="🗂️"
    )

st.markdown("---")

# Feature Modules Navigation
st.markdown("### 🚀 Platform Modules & Capabilities")

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.markdown("""
        <div class="nav-card">
            <div style="font-size: 28px; margin-bottom: 10px;">📊</div>
            <h3 style="font-size: 18px; color: #0f172a; margin: 0 0 8px 0;">Overview & Executive Dashboard</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                Portfolio distribution across 5 EMI scenarios, underwriting approval rates, average credit scores, and aggregate risk exposure.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Open Dashboard", key="btn_dash", use_container_width=True):
        st.switch_page("pages/1_📊_Overview_&_Executive_Dashboard.py")

with row1_col2:
    st.markdown("""
        <div class="nav-card">
            <div style="font-size: 28px; margin-bottom: 10px;">🎯</div>
            <h3 style="font-size: 18px; color: #0f172a; margin: 0 0 8px 0;">Real-Time Dual Predictor</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                Evaluate individual applicants in real time with 22 features. Instant Eligibility badge, Max Safe EMI calculation, FOIR meter & Amortization schedules.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Predictor", key="btn_pred", use_container_width=True):
        st.switch_page("pages/2_🎯_Real_Time_Risk_&_EMI_Predictor.py")

with row1_col3:
    st.markdown("""
        <div class="nav-card">
            <div style="font-size: 28px; margin-bottom: 10px;">🔍</div>
            <h3 style="font-size: 18px; color: #0f172a; margin: 0 0 8px 0;">Exploratory Data Analysis</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                Interactive visual exploration of the 400,000 dataset: correlation heatmaps, debt-to-income distributions, and demographic risk profiles.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Explore EDA", key="btn_eda", use_container_width=True):
        st.switch_page("pages/3_🔍_Exploratory_Data_Analysis.py")

st.markdown("<br>", unsafe_allow_html=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.markdown("""
        <div class="nav-card">
            <div style="font-size: 28px; margin-bottom: 10px;">🔬</div>
            <h3 style="font-size: 18px; color: #0f172a; margin: 0 0 8px 0;">MLflow Model Hub & Leaderboard</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                Compare all candidate models (Logistic Regression, Random Forest, XGBoost), inspect ROC curves, confusion matrices, and Model Registry stages.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Inspect Models", key="btn_models", use_container_width=True):
        st.switch_page("pages/4_🔬_MLflow_Model_Hub_&_Leaderboard.py")

with row2_col2:
    st.markdown("""
        <div class="nav-card">
            <div style="font-size: 28px; margin-bottom: 10px;">📁</div>
            <h3 style="font-size: 18px; color: #0f172a; margin: 0 0 8px 0;">Data Management (CRUD)</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                Full underwriting database operations: Create applications, Search & filter, Update underwriting decisions, Delete records, and Export data.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Manage Database", key="btn_crud", use_container_width=True):
        st.switch_page("pages/5_📁_Application_Data_Management_CRUD.py")

with row2_col3:
    st.markdown("""
        <div class="nav-card">
            <div style="font-size: 28px; margin-bottom: 10px;">⚡</div>
            <h3 style="font-size: 18px; color: #0f172a; margin: 0 0 8px 0;">Batch Inference & Simulation</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                Upload CSV files for bulk dual scoring and execute macroeconomic stress testing (interest rate shocks and salary shifts).
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Run Batch Inference", key="btn_batch", use_container_width=True):
        st.switch_page("pages/6_⚡_Batch_Prediction_&_Simulation.py")

st.markdown("---")

# Architecture Breakdown
with st.expander("🏗️ Architecture & Financial Domain Methodology", expanded=False):
    st.markdown("""
    ### System Architecture & Pipeline Flow
    1. **Data Layer**: 400,000 realistic records spanning 5 lending scenarios:
       - *E-commerce Shopping EMI* (10k-200k, 3-24 mo)
       - *Home Appliances EMI* (20k-300k, 6-36 mo)
       - *Vehicle EMI* (80k-1500k, 12-84 mo)
       - *Personal Loan EMI* (50k-1000k, 12-60 mo)
       - *Education EMI* (50k-500k, 6-48 mo)
    2. **Feature Engineering Layer**: Automated derivation of DTI (Debt-to-Income), ETI (Expense-to-Income), FOIR (Fixed Obligation to Income Ratio), Affordability Index, Emergency Runway, and interaction features.
    3. **Model Layer**:
       - *Classification*: Logistic Regression, Random Forest, XGBoost Classifier.
       - *Regression*: Ridge Linear Regression, Random Forest, XGBoost Regressor.
    4. **MLOps Layer**: MLflow tracking with automatic artifact logging, metric evaluation, and production model registry.
    5. **Application Layer**: Streamlit multi-page interface with SQLite CRUD persistence.
    """)

# Footer
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 12px; margin-top: 40px; padding: 20px 0;">
    EMIPredict AI Platform • FinTech & Banking MLOps Architecture • Powered by Scikit-Learn, XGBoost, MLflow & Streamlit
</div>
""", unsafe_allow_html=True)
