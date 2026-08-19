"""UI Cards and Metric Components for EMIPredict AI Streamlit Application."""

import streamlit as st


def render_header(title: str, subtitle: str, badge: str = "MLOps v1.0"):
    """Render a premium header section with gradient text and badge."""
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                    padding: 24px 30px; border-radius: 12px; margin-bottom: 24px; 
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25); border-left: 5px solid #2563eb;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;">
                        {title}
                    </h1>
                    <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 14px;">
                        {subtitle}
                    </p>
                </div>
                <div style="background: rgba(37, 99, 235, 0.2); border: 1px solid #3b82f6; 
                            color: #60a5fa; padding: 6px 14px; border-radius: 20px; 
                            font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">
                    {badge}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, color: str = "#2563eb", icon: str = "📈"):
    """Render a clean KPI metric card."""
    delta_html = ""
    if delta:
        delta_color = "#10b981" if "+" in delta or "▲" in delta else "#ef4444"
        delta_html = f'<span style="color: {delta_color}; font-size: 12px; font-weight: 600; margin-left: 8px;">{delta}</span>'

    st.markdown(f"""
        <div style="background: #ffffff; padding: 18px 20px; border-radius: 10px; 
                    border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    margin-bottom: 15px; border-top: 3px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #64748b; font-size: 13px; font-weight: 600; text-transform: uppercase;">{label}</span>
                <span style="font-size: 18px;">{icon}</span>
            </div>
            <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 8px; display: flex; align-items: baseline;">
                {value} {delta_html}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_risk_badge(eligibility: str, confidence: float = 0.95):
    """Render a risk classification badge with appropriate styling."""
    configs = {
        "Eligible": {
            "bg": "#ecfdf5", "border": "#10b981", "text": "#047857",
            "icon": "✅", "desc": "Low Risk • High Credit Worthiness • Approved"
        },
        "High_Risk": {
            "bg": "#fffbeb", "border": "#f59e0b", "text": "#b45309",
            "icon": "⚠️", "desc": "Moderate Risk • Marginal Debt Burden • Manual Review Required"
        },
        "Not_Eligible": {
            "bg": "#fef2f2", "border": "#ef4444", "text": "#b91c1c",
            "icon": "❌", "desc": "High Risk • Low Affordability or Poor Score • Rejected"
        }
    }
    
    cfg = configs.get(eligibility, configs["High_Risk"])
    
    st.markdown(f"""
        <div style="background: {cfg['bg']}; border: 1.5px solid {cfg['border']}; 
                    border-radius: 10px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">{cfg['icon']}</span>
                    <div>
                        <div style="color: {cfg['text']}; font-size: 18px; font-weight: 700; text-transform: uppercase;">
                            {eligibility.replace('_', ' ')}
                        </div>
                        <div style="color: #64748b; font-size: 12px; margin-top: 2px;">
                            {cfg['desc']}
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #64748b; font-size: 11px; font-weight: 600; text-transform: uppercase;">Model Confidence</div>
                    <div style="color: {cfg['text']}; font-size: 18px; font-weight: 700;">{confidence * 100:.1f}%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
