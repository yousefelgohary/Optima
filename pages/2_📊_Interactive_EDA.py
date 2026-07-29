"""
pages/2_Interactive_EDA.py
===========================
Optima Dashboard - Interactive Server EDA

Includes robust try-except rendering and Seaborn-based correlation heatmap.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from src.data_loader import load_sample_data
from src.ui_components import inject_global_css, render_sidebar_brand, page_header, section_header

st.set_page_config(page_title="Interactive EDA - Optima", page_icon="⚡", layout="wide")
inject_global_css()
render_sidebar_brand()

page_header(
    title="Interactive Server EDA",
    subtitle="Explore CPU & RAM usage patterns across scheduling classes, clusters, and time",
    icon="&#128202;",
)

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading cluster trace data..."):
    try:
        df_full = load_sample_data(n=50_000, seed=42)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

# ── Inline filter panel ──────────────────────────────────────────────────────
with st.container():
    st.markdown(
        "<div style='background:#1e293b; border:1px solid #334155; border-radius:12px;"
        " padding:1rem 1.25rem; margin-bottom:1rem;'>"
        "<p style='color:#64748b; font-size:0.72rem; text-transform:uppercase;"
        " letter-spacing:0.09em; margin:0 0 0.6rem 0;'>Chart Filters</p>",
        unsafe_allow_html=True,
    )

    sched_labels  = {0: "0 - Batch", 1: "1 - Best Effort", 2: "2 - Production", 3: "3 - Monitoring"}
    ctype_labels  = {0: "0 - Job", 1: "1 - Alloc Set"}
    all_sched     = sorted(df_full["scheduling_class"].unique().tolist())
    all_ctype     = sorted(df_full["collection_type"].unique().tolist())
    all_clusters  = sorted(df_full["cluster"].dropna().unique().tolist())

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    with fc1:
        sel_sched = st.multiselect(
            "Scheduling Class", options=all_sched, default=all_sched,
            format_func=lambda x: sched_labels.get(x, str(x)),
        )
    with fc2:
        sel_ctype = st.multiselect(
            "Collection Type", options=all_ctype, default=all_ctype,
            format_func=lambda x: ctype_labels.get(x, str(x)),
        )
    with fc3:
        sel_clusters = st.multiselect("Cluster IDs", options=all_clusters, default=all_clusters)
    with fc4:
        sample_n = st.slider("Sample size", 1_000, 50_000, 20_000, 1_000)

    st.markdown("</div>", unsafe_allow_html=True)

# Guard
if not sel_sched or not sel_ctype or not sel_clusters:
    st.warning("Please select at least one option in every filter.")
    st.stop()

mask = (
    df_full["scheduling_class"].isin(sel_sched)
    & df_full["collection_type"].isin(sel_ctype)
    & df_full["cluster"].isin(sel_clusters)
)
n_avail = int(mask.sum())
if n_avail == 0:
    st.warning("No rows match the current filters. Adjust your selections.")
    st.stop()

df = df_full[mask].sample(n=min(sample_n, n_avail), random_state=42).copy()
df["sched_label"] = df["scheduling_class"].map(sched_labels).fillna(df["scheduling_class"].astype(str))
df["ctype_label"] = df["collection_type"].map(ctype_labels).fillna(df["collection_type"].astype(str))

st.caption(f"Showing {len(df):,} rows after filtering.")

_DL = dict(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=10, r=10, t=50, b=10),
)
COLOR_SEQ = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171"]

# ── Chart 1: Violin ─────────────────────────────────────────────────────────
section_header("Resource Request Distributions", "CPU & RAM by scheduling class", "")
vc1, vc2 = st.columns(2)
try:
    with vc1:
        fig1 = px.violin(df, x="sched_label", y="resource_request_cpus", color="sched_label", box=True, points=False, color_discrete_sequence=COLOR_SEQ, title="CPU Request")
        fig1.update_layout(**_DL, showlegend=False)
        st.plotly_chart(fig1, width="stretch")
    with vc2:
        fig2 = px.violin(df, x="sched_label", y="resource_request_memory", color="sched_label", box=True, points=False, color_discrete_sequence=COLOR_SEQ, title="RAM Request")
        fig2.update_layout(**_DL, showlegend=False)
        st.plotly_chart(fig2, width="stretch")
except Exception as e:
    st.warning(f"Could not render Violin charts: {e}")

st.divider()

# ── Chart 2: Box ────────────────────────────────────────────────────────────
section_header("Priority vs Resource Requests", "Resource allocation across priority tiers", "")
try:
    df["priority_bin"] = pd.cut(
        df["priority"],
        bins=[0, 50, 100, 200, 300, 450],
        labels=["0-50 (Low)", "51-100", "101-200", "201-300", "301-450 (High)"],
        include_lowest=True,
    )
    valid_df = df.dropna(subset=["priority_bin"])
    bc1, bc2 = st.columns(2)
    with bc1:
        fig_b1 = px.box(valid_df, x="priority_bin", y="resource_request_cpus", color="priority_bin", color_discrete_sequence=COLOR_SEQ, title="CPU Request by Priority Range")
        fig_b1.update_layout(**_DL, showlegend=False)
        st.plotly_chart(fig_b1, width="stretch")
    with bc2:
        fig_b2 = px.box(valid_df, x="priority_bin", y="resource_request_memory", color="priority_bin", color_discrete_sequence=COLOR_SEQ, title="RAM Request by Priority Range")
        fig_b2.update_layout(**_DL, showlegend=False)
        st.plotly_chart(fig_b2, width="stretch")
except Exception as e:
    st.warning(f"Could not render Box charts: {e}")

st.divider()

# ── Chart 3: Correlation heatmap (SEABORN MATPLOTLIB) ────────────────────────
section_header("Feature Correlation Matrix", "Pearson correlation across numerical features", "")
try:
    num_cols = [c for c in [
        "resource_request_cpus", "resource_request_memory",
        "avg_usage_cpus", "avg_usage_memory",
        "max_usage_cpus", "max_usage_memory",
        "sample_cpus", "sample_memory",
        "assigned_memory", "page_cache_memory",
        "priority", "scheduling_class",
        "hour_of_day", "day_of_week",
        "cycles_per_instruction", "memory_accesses_per_instruction",
        "sample_rate",
    ] if c in df.columns]

    corr = df[num_cols].corr()

    # Dark background config for matplotlib to match Streamlit
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    # explicitly setting cmap to coolwarm, annot=True
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        linecolor="#1e293b",
        cbar_kws={"shrink": .8},
        ax=ax,
        annot_kws={"size": 9}
    )

    ax.tick_params(colors='#94a3b8', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.title("Pearson Correlation Matrix", color='#e2e8f0', pad=20, size=16, weight='bold')
    
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not render Heatmap: {e}")

st.divider()

# ── Chart 4: Peak hour heatmap ───────────────────────────────────────────────
section_header("Peak Hour Load Map", "Avg CPU request by hour of day vs. day of week", "")

if "hour_of_day" in df.columns and "day_of_week" in df.columns:
    pivot = (
        df.groupby(["day_of_week", "hour_of_day"])["resource_request_cpus"]
        .mean()
        .reset_index()
        .pivot(index="day_of_week", columns="hour_of_day", values="resource_request_cpus")
    )
    
    # Force a full 7x24 grid to prevent Plotly from failing on missing combinations (fixes the blank render)
    pivot = pivot.reindex(index=range(7), columns=range(24)).fillna(0)
    
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_ph = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_labels,
        colorscale="Blues",
        hovertemplate="Hour=%{x}  Day=%{y}<br>Avg CPU=%{z:.4f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#64748b"), title=dict(text="CPU", font=dict(color="#64748b"))),
    ))
    fig_ph.update_layout(
        **_DL,
        title="Avg CPU Request - Hour of Day vs. Day of Week",
        xaxis=dict(title="Hour (UTC)", tickfont=dict(color="#64748b"), tickangle=-45),
        yaxis=dict(title="", tickfont=dict(color="#64748b")),
        height=310,
    )
    st.plotly_chart(fig_ph, width="stretch")
