"""
src/data_loader.py
==================
Cached data loading utilities for the Optima Dashboard EDA page.

Uses @st.cache_data so the 314MB CSV is read and sampled only once per
session (or once per hour if TTL expires), keeping EDA charts snappy.

The parsed DataFrame exposes all columns needed for Plotly charts:
    - resource_request_cpus / resource_request_memory  (targets)
    - avg_usage_cpus / avg_usage_memory                (telemetry)
    - scheduling_class, collection_type, priority      (metadata)
    - hour_of_day, day_of_week                         (temporal)
    - event (raw string), cluster (int)                (categorical)
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.config import DATA_PATH, CPI_MEDIAN, MAPI_MEDIAN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_parse_dict(val, key: str, default: float = 0.0) -> float:
    """Safely extract a float from a stringified Python dict."""
    if pd.isna(val) or val is None:
        return default
    try:
        if isinstance(val, dict):
            return float(val.get(key, default))
        parsed = ast.literal_eval(str(val))
        return float(parsed.get(key, default))
    except (ValueError, SyntaxError, TypeError):
        return default


def _derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour_of_day, day_of_week, is_peak_hour from start_time nanoseconds."""
    df = df.copy()
    start_sec = df["start_time"].fillna(0) / 1e6
    start_dt = pd.to_datetime(start_sec, unit="s", utc=True, errors="coerce")
    end_sec = df["end_time"].fillna(0) / 1e6

    df["hour_of_day"]     = start_dt.dt.hour.fillna(0).astype(int)
    df["day_of_week"]     = start_dt.dt.dayofweek.fillna(0).astype(int)
    df["is_peak_hour"]    = df["hour_of_day"].between(9, 17).astype(int)
    df["job_duration_sec"] = (end_sec - start_sec).clip(lower=0).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Public cached loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="🔄 Loading cluster trace data…")
def load_sample_data(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Load and enrich a random sample of the Borg traces CSV.

    Parameters
    ----------
    n    : Number of rows to sample (default 50,000).
    seed : Random state for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame with enriched columns ready for EDA visualisations.
    """
    logger.info("Reading CSV from %s", DATA_PATH)

    # Read only columns we actually need for EDA to save memory
    usecols = [
        "scheduling_class", "collection_type", "priority", "instance_index",
        "vertical_scaling", "scheduler", "start_time", "end_time",
        "average_usage", "maximum_usage", "random_sample_usage",
        "assigned_memory", "page_cache_memory",
        "cycles_per_instruction", "memory_accesses_per_instruction",
        "sample_rate", "cluster", "event", "resource_request",
    ]

    df_full = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    logger.info("Full dataset shape: %s", df_full.shape)

    # Sample
    df = df_full.sample(n=min(n, len(df_full)), random_state=seed).reset_index(drop=True)
    del df_full

    # -----------------------------------------------------------------------
    # Parse stringified dicts → numeric columns
    # -----------------------------------------------------------------------
    df["resource_request_cpus"]   = df["resource_request"].apply(lambda v: _safe_parse_dict(v, "cpus"))
    df["resource_request_memory"] = df["resource_request"].apply(lambda v: _safe_parse_dict(v, "memory"))
    df["avg_usage_cpus"]          = df["average_usage"].apply(lambda v: _safe_parse_dict(v, "cpus"))
    df["avg_usage_memory"]        = df["average_usage"].apply(lambda v: _safe_parse_dict(v, "memory"))
    df["max_usage_cpus"]          = df["maximum_usage"].apply(lambda v: _safe_parse_dict(v, "cpus"))
    df["max_usage_memory"]        = df["maximum_usage"].apply(lambda v: _safe_parse_dict(v, "memory"))
    df["sample_cpus"]             = df["random_sample_usage"].apply(lambda v: _safe_parse_dict(v, "cpus"))
    df["sample_memory"]           = df["random_sample_usage"].apply(lambda v: _safe_parse_dict(v, "memory"))

    # Drop raw dict columns
    df.drop(columns=["resource_request", "average_usage", "maximum_usage", "random_sample_usage"],
            inplace=True, errors="ignore")

    # -----------------------------------------------------------------------
    # Temporal features
    # -----------------------------------------------------------------------
    df = _derive_temporal_features(df)

    # -----------------------------------------------------------------------
    # Fill missing CPI / MAPI
    # -----------------------------------------------------------------------
    df["cycles_per_instruction"]          = df["cycles_per_instruction"].fillna(CPI_MEDIAN)
    df["memory_accesses_per_instruction"] = df["memory_accesses_per_instruction"].fillna(MAPI_MEDIAN)

    # -----------------------------------------------------------------------
    # Type cleanup
    # -----------------------------------------------------------------------
    for col in ["scheduling_class", "collection_type", "priority",
                "instance_index", "cluster", "hour_of_day", "day_of_week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["vertical_scaling", "scheduler", "assigned_memory",
                "page_cache_memory", "sample_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Event column: ensure string
    df["event"] = df["event"].fillna("UNKNOWN").astype(str)

    logger.info("Sample shape after enrichment: %s", df.shape)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_dataset_stats() -> dict:
    """Return high-level stats for the Overview page without loading full sample."""
    df = load_sample_data()
    return {
        "n_sample":            len(df),
        "cpu_mean":            float(df["resource_request_cpus"].mean()),
        "cpu_median":          float(df["resource_request_cpus"].median()),
        "mem_mean":            float(df["resource_request_memory"].mean()),
        "mem_median":          float(df["resource_request_memory"].median()),
        "scheduling_classes":  sorted(df["scheduling_class"].unique().tolist()),
        "collection_types":    sorted(df["collection_type"].unique().tolist()),
        "event_types":         sorted(df["event"].unique().tolist()),
        "cluster_ids":         sorted(df["cluster"].unique().tolist()),
    }
