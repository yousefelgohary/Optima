"""
1_🏠_Home.py
===========
Optima Dashboard - Home / Landing Page

Displays a clean hero section with project summary, model KPIs,
and a feature overview.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    BEST_CPU_METRICS, BEST_MEM_METRICS,
    TRAIN_ROWS, TEST_ROWS, N_FEATURES,
    CPU_CLIP_UPPER, MEM_CLIP_UPPER,
    CPI_MEDIAN, MAPI_MEDIAN,
    CPU_FEATURE_IMPORTANCES, MEM_FEATURE_IMPORTANCES,
)
from src.ui_components import (
    inject_global_css, render_sidebar_brand,
    page_header, section_header, info_box,
    feature_importance_bar, live_sparkline,
)

st.set_page_config(
    page_title="Optima Dashboard - Home",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_global_css()
render_sidebar_brand()

page_header(
    title="Cluster Overview",
    subtitle="Pipeline health &mdash; Model performance &mdash; Dataset statistics",
    icon="&#9881;",
)

# ── Hero KPI Row ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
r2_cpu = BEST_CPU_METRICS.get("R\u00b2", BEST_CPU_METRICS.get("R2", 0.9158))
r2_mem = BEST_MEM_METRICS.get("R\u00b2", BEST_MEM_METRICS.get("R2", 0.9520))

with k1:
    st.metric("Training Samples", f"{TRAIN_ROWS:,}", "80% split")
with k2:
    st.metric("Test Samples", f"{TEST_ROWS:,}", "20% split")
with k3:
    st.metric("Engineered Features", str(N_FEATURES), "42 columns")
with k4:
    st.metric("Best CPU R2", f"{r2_cpu:.3f}", "HGBR model")
with k5:
    st.metric("Best MEM R2", f"{r2_mem:.3f}", "HGBR model")

st.divider()

# ── Model Performance Summary ────────────────────────────────────────────────
section_header("Best Model Performance", "HGBR evaluated on 81,024 held-out test samples", "")

col_cpu, col_mem = st.columns(2)
with col_cpu:
    st.markdown(
        """
        <div style="background:#1e293b; border:1px solid #34d399; border-radius:12px;
             padding:1.25rem 1.5rem;">
            <div style="color:#34d399; font-size:0.72rem; font-weight:600;
                 text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">
                CPU Request &mdash; HGBR
            </div>
            <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
                <div>
                    <div style="color:#64748b; font-size:0.7rem;">R2</div>
                    <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">0.9158</div>
                </div>
                <div>
                    <div style="color:#64748b; font-size:0.7rem;">RMSE</div>
                    <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">0.005969</div>
                </div>
                <div>
                    <div style="color:#64748b; font-size:0.7rem;">MAE</div>
                    <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">0.002216</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_mem:
    st.markdown(
        """
        <div style="background:#1e293b; border:1px solid #818cf8; border-radius:12px;
             padding:1.25rem 1.5rem;">
            <div style="color:#818cf8; font-size:0.72rem; font-weight:600;
                 text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">
                MEM Request &mdash; HGBR
            </div>
            <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
                <div>
                    <div style="color:#64748b; font-size:0.7rem;">R2</div>
                    <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">0.9520</div>
                </div>
                <div>
                    <div style="color:#64748b; font-size:0.7rem;">RMSE</div>
                    <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">0.003936</div>
                </div>
                <div>
                    <div style="color:#64748b; font-size:0.7rem;">MAE</div>
                    <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">0.001109</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Feature Importances ───────────────────────────────────────────────────────
section_header("Feature Importances", "Top drivers from the HGBR models", "")

fi1, fi2 = st.columns(2)
with fi1:
    st.caption("CPU Request — HGBR")
    st.plotly_chart(
        feature_importance_bar(CPU_FEATURE_IMPORTANCES, "CPU HGBR Feature Importances"),
        width="stretch",
    )
with fi2:
    st.caption("Memory Request — HGBR")
    st.plotly_chart(
        feature_importance_bar(MEM_FEATURE_IMPORTANCES, "MEM HGBR Feature Importances"),
        width="stretch",
    )

st.divider()

# ── Dataset Health ────────────────────────────────────────────────────────────
section_header("Dataset & Pipeline Health", "Preprocessing constants and data quality indicators", "")

dh1, dh2, dh3 = st.columns(3)
with dh1:
    st.markdown("**Dataset Statistics**")
    info_box(
        "Total Rows: <strong>405,894</strong><br>"
        "Raw Columns: <strong>33</strong><br>"
        "Engineered Features: <strong>42</strong><br>"
        "Targets: <strong>CPU &amp; MEM request</strong><br>"
        "Cluster IDs: <strong>1 &ndash; 8</strong>"
    )
with dh2:
    st.markdown("**Imputation Constants**")
    info_box(
        f"CPI Median Fill: <code>{CPI_MEDIAN:.6f}</code><br>"
        f"MAPI Median Fill: <code>{MAPI_MEDIAN:.6f}</code><br>"
        "CPI Missing Rate: <strong>30.72%</strong><br>"
        "MAPI Missing Rate: <strong>30.72%</strong><br>"
        "Missingness flags added before fill"
    )
with dh3:
    st.markdown("**Clip Bounds (99th pct)**")
    info_box(
        f"CPU Clip Upper: <code>{CPU_CLIP_UPPER:.8f}</code><br>"
        f"MEM Clip Upper: <code>{MEM_CLIP_UPPER:.8f}</code><br>"
        "Scaler: <strong>StandardScaler</strong><br>"
        "Fitted on training set only<br>"
        "No retraining inside the app"
    )

st.divider()

# ── Simulated Live Load Sparklines ────────────────────────────────────────────
section_header("Simulated Cluster Load", "Random-walk signal refreshed each minute", "")

rng = np.random.default_rng(seed=int(pd.Timestamp.utcnow().timestamp()) // 60)
cpu_signal = np.clip(0.04 + np.cumsum(rng.normal(0, 0.003, 80)), 0.005, 0.12)
rng2 = np.random.default_rng(seed=int(pd.Timestamp.utcnow().timestamp()) // 60 + 1)
mem_signal = np.clip(0.034 + np.cumsum(rng2.normal(0, 0.002, 80)), 0.003, 0.10)

sp1, sp2 = st.columns(2)
with sp1:
    st.plotly_chart(live_sparkline(cpu_signal.tolist(), "CPU Request Load"), width="stretch")
with sp2:
    st.plotly_chart(live_sparkline(mem_signal.tolist(), "MEM Request Load"), width="stretch")

st.caption("Refresh the page to update the signal. Use the Prediction Engine for real inference.")
