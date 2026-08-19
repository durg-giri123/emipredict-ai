"""Interactive Plotly Chart Generators for EMIPredict AI."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_foir_gauge(foir_value: float) -> go.Figure:
    """Create a gauge meter for Fixed Obligation to Income Ratio (FOIR)."""
    val_pct = min(100.0, max(0.0, foir_value * 100))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_pct,
        number={"suffix": "%", "font": {"size": 28, "color": "#0f172a"}},
        title={"text": "<b>FOIR (Debt Burden)</b><br><span style='font-size:12px;color:#64748b'>Safe Limit ≤ 50%</span>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
            "bar": {"color": "#1e293b", "thickness": 0.25},
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},      # Safe (Green)
                {"range": [40, 55], "color": "#fef9c3"},     # Moderate (Yellow)
                {"range": [55, 100], "color": "#fee2e2"}     # Risky (Red)
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_probability_donut(probabilities: Dict[str, float]) -> go.Figure:
    """Create probability distribution donut chart for classification classes."""
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textposition="outside"
    )])
    fig.update_layout(
        showlegend=False,
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_amortization_chart(df_schedule: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of Monthly Principal vs Interest breakdown."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_schedule["Month"],
        y=df_schedule["Principal (INR)"],
        name="Principal",
        marker_color="#2563eb"
    ))
    fig.add_trace(go.Bar(
        x=df_schedule["Month"],
        y=df_schedule["Interest (INR)"],
        name="Interest",
        marker_color="#f97316"
    ))
    fig.update_layout(
        barmode="stack",
        title="<b>Monthly Payment Breakdown (Principal vs Interest)</b>",
        xaxis_title="Repayment Month",
        yaxis_title="Payment Amount (INR)",
        height=320,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_scenario_pie_chart(scenario_counts: pd.Series) -> go.Figure:
    """Donut chart for scenario distribution."""
    fig = px.pie(
        values=scenario_counts.values,
        names=scenario_counts.index,
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    return fig
