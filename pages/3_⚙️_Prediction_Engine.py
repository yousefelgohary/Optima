"""
pages/3_Prediction_Engine.py
=============================
Optima Dashboard - Resource Prediction Engine

Responsive inference metrics utilizing st.columns and st.metric.
"""

import datetime

import pandas as pd
import streamlit as st

from src.config import (
    SCHEDULING_CLASS_LABELS, COLLECTION_TYPE_LABELS,
    VERTICAL_SCALING_LABELS, CLUSTER_IDS, EVENT_NAMES,
    CPU_CLIP_UPPER, MEM_CLIP_UPPER,
    CPI_MEDIAN, MAPI_MEDIAN,
)
from src.model_inference import (
    load_models, predict_both,
    CPU_R2, MEM_R2, CPU_RMSE, MEM_RMSE,
)
from src.preprocessing import build_inference_input
from src.ui_components import inject_global_css, render_sidebar_brand, page_header, section_header, prediction_gauge

st.set_page_config(page_title="Prediction Engine - Optima", page_icon="⚡", layout="wide")
inject_global_css()
render_sidebar_brand()

page_header(
    title="Prediction Engine",
    subtitle="HGBR inference &mdash; Input job specs to predict CPU & RAM resource requests",
    icon="&#9889;",
)

models = load_models()

if "pred_cpu" not in st.session_state:
    st.session_state.pred_cpu = None
    st.session_state.pred_mem = None
    st.session_state.pred_error = None

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1.0], gap="large")

with left_col:
    section_header("Job Specification", "Define the cluster job parameters", "")

    c1, c2 = st.columns(2)
    with c1:
        input_date = st.date_input("Start Date", value=datetime.date.today(), key="sd")
        end_date   = st.date_input("End Date",   value=datetime.date.today(), key="ed")
    with c2:
        input_time = st.time_input("Start Time (UTC)", value=datetime.time(10, 0), key="st_t")
        end_time_v = st.time_input("End Time (UTC)",   value=datetime.time(11, 0), key="et_t")

    job_dur = st.number_input(
        "Job Duration (seconds) — 0 = derive from Start/End times",
        min_value=0.0, value=0.0, step=60.0, format="%.1f",
    )

    st.divider()

    m1, m2 = st.columns(2)
    with m1:
        sched_class = st.selectbox(
            "Scheduling Class",
            options=list(SCHEDULING_CLASS_LABELS.keys()),
            format_func=lambda k: SCHEDULING_CLASS_LABELS[k],
            index=2,
        )
        vert_scaling = st.selectbox(
            "Vertical Scaling",
            options=list(VERTICAL_SCALING_LABELS.keys()),
            format_func=lambda k: VERTICAL_SCALING_LABELS[k],
            index=0,
        )
    with m2:
        coll_type = st.selectbox(
            "Collection Type",
            options=list(COLLECTION_TYPE_LABELS.keys()),
            format_func=lambda k: COLLECTION_TYPE_LABELS[k],
            index=0,
        )
        scheduler = st.selectbox("Scheduler ID", options=[0, 1, 2, 3, 4], index=0)

    m3, m4 = st.columns(2)
    with m3:
        priority = st.slider("Priority", 0, 450, 100, 1)
    with m4:
        inst_idx = st.slider("Instance Index", 0, 200, 0, 1)

    st.divider()

    e1, e2 = st.columns(2)
    with e1:
        event_sel   = st.selectbox("Event Type", options=EVENT_NAMES, index=EVENT_NAMES.index("SCHEDULE"))
    with e2:
        cluster_sel = st.selectbox("Cluster ID", options=CLUSTER_IDS, index=2)

    st.divider()

    with st.expander("Telemetry — Historical Usage Signals", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            avg_cpu  = st.number_input("avg_usage_cpus",    0.0, 0.5, 0.010, 0.001, "%.4f")
            max_cpu  = st.number_input("max_usage_cpus",    0.0, 0.5, 0.020, 0.001, "%.4f")
            smp_cpu  = st.number_input("sample_cpus",       0.0, 0.5, 0.010, 0.001, "%.4f")
            asgn_mem = st.number_input("assigned_memory",   0.0, 0.5, 0.050, 0.001, "%.4f")
            smp_rate = st.number_input("sample_rate",       0.0, 1.0, 0.100, 0.010, "%.3f")
        with t2:
            avg_mem  = st.number_input("avg_usage_memory",  0.0, 0.5, 0.020, 0.001, "%.4f")
            max_mem  = st.number_input("max_usage_memory",  0.0, 0.5, 0.030, 0.001, "%.4f")
            smp_mem  = st.number_input("sample_memory",     0.0, 0.5, 0.020, 0.001, "%.4f")
            pg_cache = st.number_input("page_cache_memory", 0.0, 0.5, 0.010, 0.001, "%.4f")

    with st.expander("CPU Efficiency Metrics (CPI / MAPI)", expanded=False):
        cpi_use_med  = st.checkbox("CPI missing — use training median", value=True)
        mapi_use_med = st.checkbox("MAPI missing — use training median", value=True)

        cpi_val = mapi_val = None
        if not cpi_use_med:
            cpi_val = st.number_input(f"CPI (training median: {CPI_MEDIAN:.4f})", 0.0, 10.0, float(CPI_MEDIAN), 0.01, "%.4f")
        else:
            st.caption(f"Median fill: {CPI_MEDIAN:.6f} | cpi_missing = 1")

        if not mapi_use_med:
            mapi_val = st.number_input(f"MAPI (training median: {MAPI_MEDIAN:.6f})", 0.0, 1.0, float(MAPI_MEDIAN), 0.0001, "%.6f")
        else:
            st.caption(f"Median fill: {MAPI_MEDIAN:.6f} | mapi_missing = 1")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    run_btn = st.button("Run HGBR Prediction", type="primary", use_container_width=True)

with right_col:
    section_header("Prediction Results", "HistGradientBoosting — production model", "")

    if run_btn:
        start_dt = datetime.datetime.combine(input_date, input_time)
        end_dt   = datetime.datetime.combine(end_date,   end_time_v)
        form_data = dict(
            start_dt         = start_dt,
            end_dt           = end_dt,
            job_duration_sec = job_dur if job_dur > 0 else None,
            scheduling_class = sched_class,
            collection_type  = coll_type,
            priority         = priority,
            instance_index   = inst_idx,
            vertical_scaling = vert_scaling,
            scheduler        = scheduler,
            event            = event_sel,
            cluster          = cluster_sel,
            avg_usage_cpus   = avg_cpu,
            avg_usage_memory = avg_mem,
            max_usage_cpus   = max_cpu,
            max_usage_memory = max_mem,
            sample_cpus      = smp_cpu,
            sample_memory    = smp_mem,
            assigned_memory  = asgn_mem,
            page_cache_memory= pg_cache,
            sample_rate      = smp_rate,
            cpi              = None if cpi_use_med  else cpi_val,
            mapi             = None if mapi_use_med else mapi_val,
        )

        try:
            scaled_X = build_inference_input(form_data, models["scaler"])
            preds    = predict_both(scaled_X, models)
            st.session_state.pred_cpu   = preds["cpu"]
            st.session_state.pred_mem   = preds["mem"]
            st.session_state.pred_error = None
        except Exception as exc:
            st.session_state.pred_cpu   = None
            st.session_state.pred_mem   = None
            st.session_state.pred_error = str(exc)

    if st.session_state.pred_error:
        st.error(f"Prediction failed: {st.session_state.pred_error}")

    elif st.session_state.pred_cpu is not None:
        cpu_pred = st.session_state.pred_cpu
        mem_pred = st.session_state.pred_mem

        st.markdown(
            "<p style='color:#34d399; font-size:0.8rem; margin:0 0 0.5rem 0;'>"
            "&#9679; HGBR prediction ready</p>", unsafe_allow_html=True
        )

        # ── Gauges ───────────────────────────────────────────────────────────
        g1, g2 = st.columns(2)
        try:
            with g1:
                st.plotly_chart(prediction_gauge(cpu_pred, CPU_CLIP_UPPER, "CPU Request", "x cap", (0.04, 0.09)), use_container_width=True)
            with g2:
                st.plotly_chart(prediction_gauge(mem_pred, MEM_CLIP_UPPER, "RAM Request", "x cap", (0.03, 0.07)), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render gauges: {e}")

        st.divider()

        # ── Fully responsive metrics using native st.metric ──────────────────
        CPU_AVG, MEM_AVG = 0.018, 0.025
        ma1, ma2 = st.columns(2)
        with ma1:
            st.metric(label="CPU Request (HGBR)", value=f"{cpu_pred:.5f}", delta=f"{cpu_pred - CPU_AVG:+.5f} vs avg")
        with ma2:
            st.metric(label="MEM Request (HGBR)", value=f"{mem_pred:.5f}", delta=f"{mem_pred - MEM_AVG:+.5f} vs avg")

        mb1, mb2 = st.columns(2)
        with mb1:
            st.metric("CPU Clip Bound", f"{CPU_CLIP_UPPER:.5f}")
        with mb2:
            st.metric("MEM Clip Bound", f"{MEM_CLIP_UPPER:.5f}")

        st.divider()
        section_header("Capacity Utilisation", "Predicted request as fraction of max", "")
        
        cpu_frac = min(1.0, cpu_pred / CPU_CLIP_UPPER) if CPU_CLIP_UPPER > 0 else 0.0
        mem_frac = min(1.0, mem_pred / MEM_CLIP_UPPER) if MEM_CLIP_UPPER > 0 else 0.0
        
        p1, p2 = st.columns(2)
        with p1:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.caption("CPU utilisation")
            with c2:
                st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 0.85rem; font-weight: 500; padding-top: 0.1rem;'>{cpu_frac*100:.1f}%</div>", unsafe_allow_html=True)
            st.progress(cpu_frac)
        with p2:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.caption("MEM utilisation")
            with c2:
                st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 0.85rem; font-weight: 500; padding-top: 0.1rem;'>{mem_frac*100:.1f}%</div>", unsafe_allow_html=True)
            st.progress(mem_frac)

    else:
        st.info("Configure parameters on the left, then click Run HGBR Prediction.")
