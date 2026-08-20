"""Page 3: Exploratory Data Analysis (EDA) for EMIPredict AI."""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from components.cards import render_header
except ImportError:
    from app.components.cards import render_header

from src.features.feature_engineering import FinancialFeatureEngineer

st.set_page_config(page_title="Exploratory Data Analysis - EMIPredict AI", page_icon="🔍", layout="wide")

render_header(
    title="🔍 Exploratory Data Analysis & Financial Patterns",
    subtitle="Statistical insights, correlation analysis, demographic risk profiling, and multi-scenario distributions",
    badge="400,000 Dataset Insights"
)

# Load Sample or Raw Dataset
sample_path = os.path.join(PROJECT_ROOT, "data/processed/dataset_sample_50k.parquet")
raw_parquet_path = os.path.join(PROJECT_ROOT, "data/raw/emi_dataset_400k.parquet")

@st.cache_data
def load_eda_data():
    from src.data.data_generator import generate_emi_dataset
    # Try sample parquet
    if os.path.exists(sample_path):
        try:
            return pd.read_parquet(sample_path)
        except Exception:
            try:
                os.remove(sample_path)
            except Exception:
                pass
    # Try raw parquet
    if os.path.exists(raw_parquet_path):
        try:
            df_raw = pd.read_parquet(raw_parquet_path)
            return df_raw.sample(n=min(30000, len(df_raw)), random_state=42)
        except Exception:
            pass
    # Generate fresh — lightweight for cloud
    df_gen = generate_emi_dataset(total_records=10000, random_state=42)
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    try:
        df_gen.to_parquet(sample_path, index=False)
    except Exception:
        pass
    return df_gen

df_raw = load_eda_data()
fe = FinancialFeatureEngineer()
df = fe.transform(df_raw)

# Sidebar Filter Controls
st.sidebar.markdown("### 🎛️ Interactive Filters")
scenario_filter = st.sidebar.multiselect(
    "EMI Scenarios",
    options=df["emi_scenario"].unique().tolist(),
    default=df["emi_scenario"].unique().tolist()
)

eligibility_filter = st.sidebar.multiselect(
    "Eligibility Status",
    options=df["emi_eligibility"].unique().tolist(),
    default=df["emi_eligibility"].unique().tolist()
)

min_salary, max_salary = int(df["monthly_salary"].min()), int(df["monthly_salary"].max())
salary_range = st.sidebar.slider("Salary Range (INR)", min_salary, max_salary, (min_salary, max_salary), step=5000)

min_score, max_score = int(df["credit_score"].min()), int(df["credit_score"].max())
score_range = st.sidebar.slider("Credit Score Range", min_score, max_score, (min_score, max_score), step=10)

# Filter Data
filtered_df = df[
    (df["emi_scenario"].isin(scenario_filter)) &
    (df["emi_eligibility"].isin(eligibility_filter)) &
    (df["monthly_salary"] >= salary_range[0]) &
    (df["monthly_salary"] <= salary_range[1]) &
    (df["credit_score"] >= score_range[0]) &
    (df["credit_score"] <= score_range[1])
]

st.markdown(f"**Showing {len(filtered_df):,} out of {len(df):,} records** based on active filters.")

# EDA Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Scenario & Loan Distribution",
    "👥 Demographics & Employment",
    "💳 Financial Obligations & FOIR",
    "🔥 Correlation Matrix",
    "📈 Statistical Summaries"
])

# Tab 1: Scenarios & Loan Distribution
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Requested Amount Distribution by Scenario")
        fig1 = px.box(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="emi_scenario",
            y="requested_amount",
            color="emi_scenario",
            labels={"requested_amount": "Requested Amount (INR)", "emi_scenario": "Scenario"}
        )
        fig1.update_layout(template="plotly_white", showlegend=False, height=350)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown("##### Requested Tenure Distribution by Scenario")
        fig2 = px.violin(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="emi_scenario",
            y="requested_tenure",
            color="emi_scenario",
            box=True,
            points=False,
            labels={"requested_tenure": "Tenure (Months)", "emi_scenario": "Scenario"}
        )
        fig2.update_layout(template="plotly_white", showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Demographics & Employment
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Salary vs. Education & Company Type")
        fig3 = px.box(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="education",
            y="monthly_salary",
            color="company_type",
            labels={"monthly_salary": "Monthly Salary (INR)", "education": "Education Level"}
        )
        fig3.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col2:
        st.markdown("##### Risk Tier by Employment Type")
        emp_risk = pd.crosstab(filtered_df["employment_type"], filtered_df["emi_eligibility"], normalize="index") * 100
        fig4 = px.bar(
            emp_risk,
            barmode="stack",
            color_discrete_map={"Eligible": "#10b981", "High_Risk": "#f59e0b", "Not_Eligible": "#ef4444"},
            labels={"value": "Percentage (%)", "employment_type": "Employment Type"}
        )
        fig4.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Financial Obligations & FOIR
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Fixed Obligation to Income Ratio (FOIR) Distribution")
        fig5 = px.histogram(
            filtered_df.sample(min(4000, len(filtered_df)), random_state=42),
            x="foir",
            color="emi_eligibility",
            nbins=40,
            color_discrete_map={"Eligible": "#10b981", "High_Risk": "#f59e0b", "Not_Eligible": "#ef4444"},
            labels={"foir": "FOIR Ratio"}
        )
        fig5.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig5, use_container_width=True)
        
    with col2:
        st.markdown("##### Disposable Income vs. Credit Score")
        fig6 = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df)), random_state=42),
            x="credit_score",
            y="disposable_income",
            color="emi_eligibility",
            color_discrete_map={"Eligible": "#10b981", "High_Risk": "#f59e0b", "Not_Eligible": "#ef4444"},
            labels={"credit_score": "Credit Score (300-850)", "disposable_income": "Monthly Disposable Income (INR)"},
            opacity=0.6
        )
        fig6.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig6, use_container_width=True)

# Tab 4: Correlation Matrix
with tab4:
    st.markdown("##### 🔥 Correlation Heatmap across Numerical & Ratio Variables")
    corr_cols = [
        "monthly_salary", "age", "years_of_employment", "credit_score",
        "current_emi_amount", "bank_balance", "emergency_fund",
        "requested_amount", "requested_tenure", "total_monthly_expenses",
        "disposable_income", "foir", "debt_to_income_ratio", "max_monthly_emi"
    ]
    corr_matrix = filtered_df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="<b>Pearson Correlation Heatmap</b>"
    )
    fig_corr.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab 5: Statistical Summaries
with tab5:
    st.markdown("##### 📋 Comprehensive Statistical Summary (Filtered Population)")
    st.dataframe(filtered_df[corr_cols].describe().T.style.format("{:,.2f}"), use_container_width=True)
