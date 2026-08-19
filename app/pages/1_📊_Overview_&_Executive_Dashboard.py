"""Page 1: Overview & Executive Dashboard for EMIPredict AI."""

import os
import sys
import json
import streamlit as st
import pandas as pd
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
    from components.cards import render_header, render_metric_card
except ImportError:
    from app.components.cards import render_header, render_metric_card

from src.database.crud_manager import FinancialDatabaseManager

st.set_page_config(page_title="Executive Dashboard - EMIPredict AI", page_icon="📊", layout="wide")

render_header(
    title="📊 Executive Portfolio & Underwriting Dashboard",
    subtitle="High-level financial risk metrics, portfolio distributions across 5 EMI scenarios, and underwriting analytics",
    badge="Executive Analytics"
)

# Load sample dataset or database
db = FinancialDatabaseManager(db_path=os.path.join(PROJECT_ROOT, "data/database/emipredict_applications.db"))
db.seed_initial_demo_records(25)
db_stats = db.get_summary_statistics()

# Load 50k sample dataset for rich visualizations if available
sample_path = os.path.join(PROJECT_ROOT, "data/processed/dataset_sample_50k.parquet")
raw_parquet_path = os.path.join(PROJECT_ROOT, "data/raw/emi_dataset_400k.parquet")

df = None
if os.path.exists(sample_path):
    df = pd.read_parquet(sample_path)
elif os.path.exists(raw_parquet_path):
    df = pd.read_parquet(raw_parquet_path)
    df = df.sample(n=min(30000, len(df)), random_state=42)
else:
    try:
        from src.data.data_generator import generate_emi_dataset
        df = generate_emi_dataset(total_records=25000, random_state=42)
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df.to_parquet(sample_path, index=False)
    except Exception:
        df = db.search_applications(limit=1000)

if df is not None:
    # Harmonize database columns with analytics columns
    if "predicted_max_emi" in df.columns and "max_monthly_emi" not in df.columns:
        df["max_monthly_emi"] = df["predicted_max_emi"]
    if "predicted_eligibility" in df.columns and "emi_eligibility" not in df.columns:
        df["emi_eligibility"] = df["predicted_eligibility"]

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_cases = len(df) if df is not None else 400000
with kpi1:
    render_metric_card("Analyzed Records", f"{total_cases:,}", delta="400K Target", color="#2563eb", icon="📑")

with kpi2:
    if df is not None and "emi_eligibility" in df.columns:
        eligible_pct = round((df["emi_eligibility"] == "Eligible").mean() * 100, 1)
    else:
        eligible_pct = 54.2
    render_metric_card("Eligible Approval Rate", f"{eligible_pct}%", delta="Prime Borrowers", color="#10b981", icon="✅")

with kpi3:
    if df is not None and "credit_score" in df.columns:
        avg_score = round(df["credit_score"].mean(), 0)
    else:
        avg_score = 692
    render_metric_card("Avg Credit Score", f"{avg_score:.0f}", delta="Good Rating", color="#8b5cf6", icon="⭐")

with kpi4:
    if df is not None and "max_monthly_emi" in df.columns:
        avg_max_emi = round(df["max_monthly_emi"].mean(), 0)
    else:
        avg_max_emi = 18450
    render_metric_card("Avg Safe Max EMI", f"₹ {avg_max_emi:,.0f}", delta="Per Month", color="#f59e0b", icon="💰")

st.markdown("---")

if df is not None and len(df) > 0:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 🎯 EMI Eligibility Breakdown across Portfolio")
        if "emi_eligibility" in df.columns:
            elig_counts = df["emi_eligibility"].value_counts()
            fig_elig = px.pie(
                values=elig_counts.values,
                names=elig_counts.index,
                hole=0.55,
                color=elig_counts.index,
                color_discrete_map={
                    "Eligible": "#10b981",
                    "High_Risk": "#f59e0b",
                    "Not_Eligible": "#ef4444"
                }
            )
            fig_elig.update_traces(textposition="inside", textinfo="percent+label")
            fig_elig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_elig, use_container_width=True)
            
    with col_right:
        st.markdown("#### 📦 Volume Distribution across 5 EMI Scenarios")
        if "emi_scenario" in df.columns:
            scen_counts = df["emi_scenario"].value_counts()
            fig_scen = px.bar(
                x=scen_counts.index,
                y=scen_counts.values,
                labels={"x": "EMI Scenario", "y": "Number of Profiles"},
                color=scen_counts.index,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_scen.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
            st.plotly_chart(fig_scen, use_container_width=True)
            
    st.markdown("---")
    
    # Financial Relationship Charts
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("#### 📈 Monthly Salary vs. Maximum Safe EMI Amount")
        sample_scatter = df[["monthly_salary", "max_monthly_emi"] + 
                            (["emi_eligibility"] if "emi_eligibility" in df.columns else [])
                           ].dropna().sample(n=min(2000, len(df)), random_state=42)
        
        color_col = "emi_eligibility" if "emi_eligibility" in sample_scatter.columns else None
        color_map = {"Eligible": "#10b981", "High_Risk": "#f59e0b", "Not_Eligible": "#ef4444"}
        
        try:
            import statsmodels  # noqa: F401
            fig_scatter = px.scatter(
                sample_scatter,
                x="monthly_salary", y="max_monthly_emi",
                color=color_col,
                color_discrete_map=color_map,
                labels={"monthly_salary": "Monthly Gross Salary (INR)", "max_monthly_emi": "Max Safe EMI (INR)"},
                opacity=0.7,
                trendline="ols",
                trendline_scope="overall",
            )
        except Exception:
            fig_scatter = px.scatter(
                sample_scatter,
                x="monthly_salary", y="max_monthly_emi",
                color=color_col,
                color_discrete_map=color_map,
                labels={"monthly_salary": "Monthly Gross Salary (INR)", "max_monthly_emi": "Max Safe EMI (INR)"},
                opacity=0.7,
            )
        fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
        fig_scatter.update_layout(
            height=400,
            template="plotly_white",
            margin=dict(l=30, r=20, t=20, b=40),
            xaxis=dict(title="Monthly Gross Salary (INR)", tickformat=","),
            yaxis=dict(title="Max Safe EMI (INR)", tickformat=","),
            legend=dict(title="Eligibility", orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_b2:
        st.markdown("#### 🛡️ Credit Score Distribution by Eligibility Status")
        fig_box = px.box(
            df.sample(n=min(5000, len(df)), random_state=42),
            x="emi_eligibility" if "emi_eligibility" in df.columns else None,
            y="credit_score",
            color="emi_eligibility" if "emi_eligibility" in df.columns else None,
            color_discrete_map={
                "Eligible": "#10b981",
                "High_Risk": "#f59e0b",
                "Not_Eligible": "#ef4444"
            },
            labels={"credit_score": "Credit Score (300-850)", "emi_eligibility": "Risk Tier"}
        )
        fig_box.update_layout(height=340, template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
        
    # Scenario Summary Table
    st.markdown("### 📋 Lending Scenario Risk & Affordability Breakdown")
    if "emi_scenario" in df.columns:
        summary_table = df.groupby("emi_scenario").agg(
            Avg_Requested_Amount=("requested_amount", "mean"),
            Avg_Salary=("monthly_salary", "mean"),
            Avg_Credit_Score=("credit_score", "mean"),
            Avg_Max_EMI=("max_monthly_emi", "mean"),
            Eligible_Ratio=("emi_eligibility", lambda x: f"{(x == 'Eligible').mean() * 100:.1f}%")
        ).reset_index()
        
        summary_table["Avg_Requested_Amount"] = summary_table["Avg_Requested_Amount"].apply(lambda v: f"₹ {v:,.0f}")
        summary_table["Avg_Salary"] = summary_table["Avg_Salary"].apply(lambda v: f"₹ {v:,.0f}")
        summary_table["Avg_Credit_Score"] = summary_table["Avg_Credit_Score"].apply(lambda v: f"{v:.0f}")
        summary_table["Avg_Max_EMI"] = summary_table["Avg_Max_EMI"].apply(lambda v: f"₹ {v:,.0f}")
        
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
