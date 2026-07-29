"""
src/ui_components.py
====================
Reusable UI primitives for the Optima Dashboard.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def inject_global_css() -> None:
    """Inject the global Optima Dashboard CSS."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global resets ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Main area background ── */
        .stApp {
            background: #0f172a;
        }

        /* ── Metric cards (responsive) ── */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
            box-sizing: border-box;
        }
        [data-testid="stMetric"]:hover {
            border-color: #38bdf8;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.75rem !important;
            font-weight: 500;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: normal !important; /* Allow wrapping */
        }
        [data-testid="stMetricValue"] {
            color: #e2e8f0 !important;
            font-size: clamp(1.2rem, 2vw, 1.8rem) !important; /* Responsive scaling */
            font-weight: 700;
        }
        [data-testid="stMetricDelta"] svg { display: none; }
        [data-testid="stMetricDelta"] > div {
            font-size: 0.75rem;
            font-weight: 500;
        }

        /* ── Hide native sidebar nav ── */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style='margin-bottom: 2rem; border-bottom: 1px solid #334155; padding-bottom: 1rem;'>
            <h1 style='color: #f8fafc; font-size: 2.2rem; font-weight: 700; margin: 0 0 0.5rem 0;'>
                {f'<span style="margin-right:0.5rem;">{icon}</span>' if icon else ''}{title}
            </h1>
            <p style='color: #94a3b8; font-size: 1rem; margin: 0;'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str, icon: str = "") -> None:
    st.markdown(
        f"""
        <div style='margin-bottom: 1rem;'>
            <h3 style='color: #e2e8f0; font-size: 1.25rem; font-weight: 600; margin: 0 0 0.2rem 0;'>
                {f'<span style="margin-right:0.4rem;">{icon}</span>' if icon else ''}{title}
            </h3>
            <p style='color: #64748b; font-size: 0.85rem; margin: 0;'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(content: str) -> None:
    st.markdown(
        f"""
        <div style='background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; color: #94a3b8; font-size: 0.85rem; line-height: 1.6;'>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_gauge(value: float, max_val: float, label: str, unit: str, color_thresholds: tuple[float, float]) -> go.Figure:
    """Plotly indicator gauge for resource predictions. Fully responsive."""
    low, high = color_thresholds
    if value > high:
        bar_color = "#ef4444"
    elif value > low:
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickfont": {"color": "#64748b", "size": 10},
                "tickcolor": "#334155",
            },
            "bar":        {"color": bar_color, "thickness": 0.2},
            "bgcolor":    "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0,        low],    "color": "#052e16"},
                {"range": [low,      high],   "color": "#422006"},
                {"range": [high,     max_val], "color": "#3f0f0f"},
            ],
            "threshold": {
                "line": {"color": "#e2e8f0", "width": 2},
                "thickness": 0.75,
                "value": value,
            },
        },
        title={"text": f"{label}<br><span style='font-size:0.7em;color:#64748b'>{unit}</span>", "font": {"color": "#94a3b8", "size": 14, "family": "Inter"}},
    ))

    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=20, r=20, t=60, b=10),
        height=200,
        font={"color": "#e2e8f0", "family": "Inter"},
    )
    return fig


def feature_importance_bar(importances: dict[str, float], title: str = "Feature Importances") -> go.Figure:
    features = list(importances.keys())
    values   = list(importances.values())

    paired   = sorted(zip(values, features), reverse=True)
    values_s = [p[0] for p in paired]
    feats_s  = [p[1] for p in paired]

    colors = ["#38bdf8" if v > 0.3 else "#818cf8" if v > 0.1 else "#475569" for v in values_s]

    fig = go.Figure(go.Bar(
        x=values_s, y=feats_s, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in values_s], textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        title=dict(text=title, font=dict(color="#e2e8f0", size=14, family="Inter"), x=0),
        xaxis=dict(showgrid=False, zeroline=False, tickformat=".0%", title=""),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=10, r=60, t=40, b=10),
        height=max(250, len(features) * 32),
        font=dict(color="#94a3b8", family="Inter"), bargap=0.35,
    )
    return fig


def live_sparkline(values: list[float], label: str = "Cluster Load") -> go.Figure:
    x = list(range(len(values)))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=values, fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="rgba(56,189,248,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=values, mode="lines", line=dict(color="#38bdf8", width=2),
        showlegend=False, hovertemplate="t=%{x}  load=%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        title=dict(text=f"<span style='color:#94a3b8; font-size:12px'>● {label}</span>", font=dict(family="Inter"), x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, tickfont=dict(size=10, color="#64748b")),
        margin=dict(l=10, r=10, t=35, b=10), height=160,
    )
    return fig


def render_sidebar_brand() -> None:
    """Render the Optima logo, brand, and custom navigation explicitly in the sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 1rem 0 0.5rem 0; margin-bottom: 0.5rem;">
          <h1 style="color: #e2e8f0; font-size: 1.5rem; font-weight: 700; margin: 0; line-height: 1.2;">
            &#9889; Project Optima
          </h1>
          <div style="color: #64748b; font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 0.3rem;">
            Cluster Intelligence
          </div>
        </div>
        <hr style="border-top: 1px solid #334155; margin-top: 0; margin-bottom: 1rem;">
        """,
        unsafe_allow_html=True,
    )
    
    st.sidebar.page_link("1_🏠_Home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/2_📊_Interactive_EDA.py", label="Interactive EDA", icon="📊")
    st.sidebar.page_link("pages/3_⚙️_Prediction_Engine.py", label="Prediction Engine", icon="⚙️")

