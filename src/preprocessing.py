"""
src/preprocessing.py
====================
Inference-time feature engineering pipeline for the Optima Dashboard.

build_inference_input() mirrors the notebook's preprocessing pipeline
exactly — including time feature derivation, one-hot encoding, CPI/MAPI
missing indicators, and StandardScaler transformation.

The output is a scaled numpy array ready for model.predict().
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    EVENT_COLS,
    CLUSTER_COLS,
    CPI_MEDIAN,
    MAPI_MEDIAN,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_inference_input(
    form_data: dict,
    scaler,
) -> np.ndarray:
    """
    Transform a user-supplied form_data dict into a scaled (1, 42) numpy array
    ready to be passed directly to model.predict().

    Parameters
    ----------
    form_data : dict
        Keys defined below. All optional keys default gracefully.
    scaler : sklearn StandardScaler
        The fitted scaler loaded from optima_scaler.joblib.

    Returns
    -------
    np.ndarray of shape (1, 42), dtype float64.

    Expected form_data keys
    -----------------------
    start_dt        : datetime.datetime  — job start datetime (local or UTC)
    end_dt          : datetime.datetime  — job end datetime
    scheduling_class: int  (0–3)
    collection_type : int  (0 or 1)
    priority        : int  (0–450)
    instance_index  : int  (0–N)
    vertical_scaling: int  (0, 1, or 2)
    scheduler       : int  (0–4)
    event           : str  — one of EVENT_NAMES (e.g. "SCHEDULE", "EVICT")
    cluster         : int  (1–8)

    Telemetry floats (default 0.0 if absent):
    avg_usage_cpus, avg_usage_memory, max_usage_cpus, max_usage_memory,
    sample_cpus, sample_memory, assigned_memory, page_cache_memory, sample_rate

    CPI / MAPI (pass None to trigger median fill + missing flag):
    cpi  : float | None
    mapi : float | None

    job_duration_sec : float | None  — if None, derived from start_dt and end_dt
    """

    row: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 1 — Scheduling metadata
    # ------------------------------------------------------------------
    row["scheduling_class"]  = float(form_data.get("scheduling_class", 0))
    row["collection_type"]   = float(form_data.get("collection_type", 0))
    row["priority"]          = float(form_data.get("priority", 0))
    row["instance_index"]    = float(form_data.get("instance_index", 0))
    row["vertical_scaling"]  = float(form_data.get("vertical_scaling", 0))
    row["scheduler"]         = float(form_data.get("scheduler", 0))

    # ------------------------------------------------------------------
    # Step 2 — Historical usage telemetry (default 0.0 = new job)
    # ------------------------------------------------------------------
    for col in [
        "avg_usage_cpus", "avg_usage_memory",
        "max_usage_cpus", "max_usage_memory",
        "sample_cpus",    "sample_memory",
        "assigned_memory", "page_cache_memory",
        "sample_rate",
    ]:
        row[col] = float(form_data.get(col, 0.0))

    # ------------------------------------------------------------------
    # Step 3 — CPI / MAPI with missing indicators
    # ------------------------------------------------------------------
    cpi_raw  = form_data.get("cpi",  None)
    mapi_raw = form_data.get("mapi", None)

    row["cpi_missing"]  = 1.0 if cpi_raw is None else 0.0
    row["mapi_missing"] = 1.0 if mapi_raw is None else 0.0
    row["cycles_per_instruction"]          = float(cpi_raw)  if cpi_raw  is not None else CPI_MEDIAN
    row["memory_accesses_per_instruction"] = float(mapi_raw) if mapi_raw is not None else MAPI_MEDIAN

    # ------------------------------------------------------------------
    # Step 4 — Temporal features from start_dt / end_dt
    # ------------------------------------------------------------------
    start_dt: datetime.datetime = form_data.get("start_dt", datetime.datetime.utcnow())
    end_dt:   datetime.datetime = form_data.get("end_dt",   start_dt + datetime.timedelta(hours=1))

    # Convert to nanosecond-equivalent seconds for time_of_week
    start_ts = start_dt.timestamp()
    end_ts   = end_dt.timestamp()

    row["hour_of_day"]     = float(start_dt.hour)
    row["day_of_week"]     = float(start_dt.weekday())   # 0=Monday, 6=Sunday
    row["is_peak_hour"]    = 1.0 if 9 <= start_dt.hour <= 17 else 0.0
    row["time_of_week_sec"] = float(int(start_ts) % 604_800)   # 0 → 604,800 seconds

    # job_duration_sec: use explicit value if supplied, else derive
    explicit_duration = form_data.get("job_duration_sec", None)
    if explicit_duration is not None:
        row["job_duration_sec"] = max(0.0, float(explicit_duration))
    else:
        row["job_duration_sec"] = max(0.0, end_ts - start_ts)

    # ------------------------------------------------------------------
    # Step 5 — Event one-hot encoding (10 columns)
    # ------------------------------------------------------------------
    selected_event: str = str(form_data.get("event", "SCHEDULE")).upper().strip()
    for col in EVENT_COLS:
        event_name = col.replace("event_", "")
        row[col] = 1.0 if event_name == selected_event else 0.0

    # ------------------------------------------------------------------
    # Step 6 — Cluster one-hot encoding (8 columns)
    # ------------------------------------------------------------------
    selected_cluster: int = int(form_data.get("cluster", 1))
    for col in CLUSTER_COLS:
        cluster_id = int(col.replace("cluster_", ""))
        row[col] = 1.0 if cluster_id == selected_cluster else 0.0

    # ------------------------------------------------------------------
    # Step 7 — Assemble DataFrame in exact FEATURE_COLUMNS order
    # ------------------------------------------------------------------
    feature_df = pd.DataFrame([row])[FEATURE_COLUMNS]

    # Sanity check
    assert feature_df.shape == (1, 42), (
        f"Feature vector shape mismatch: expected (1, 42), got {feature_df.shape}. "
        f"Missing columns: {set(FEATURE_COLUMNS) - set(feature_df.columns)}"
    )

    # ------------------------------------------------------------------
    # Step 8 — StandardScaler transform
    # ------------------------------------------------------------------
    scaled = scaler.transform(feature_df.values.astype(float))
    return scaled


def get_feature_df_unscaled(form_data: dict) -> pd.DataFrame:
    """
    Return the unscaled 42-feature DataFrame (for display/debugging purposes only).
    Does NOT apply the scaler.
    """
    # Temporarily mock a passthrough scaler
    class _Identity:
        def transform(self, X):
            return X

    _ = build_inference_input(form_data, _Identity())

    # Re-run without scaler to get the DataFrame
    row: dict[str, float] = {}
    row["scheduling_class"]  = float(form_data.get("scheduling_class", 0))
    row["collection_type"]   = float(form_data.get("collection_type", 0))
    row["priority"]          = float(form_data.get("priority", 0))
    row["instance_index"]    = float(form_data.get("instance_index", 0))
    row["vertical_scaling"]  = float(form_data.get("vertical_scaling", 0))
    row["scheduler"]         = float(form_data.get("scheduler", 0))
    for col in ["avg_usage_cpus","avg_usage_memory","max_usage_cpus","max_usage_memory",
                "sample_cpus","sample_memory","assigned_memory","page_cache_memory","sample_rate"]:
        row[col] = float(form_data.get(col, 0.0))
    cpi_raw  = form_data.get("cpi",  None)
    mapi_raw = form_data.get("mapi", None)
    row["cpi_missing"]  = 1.0 if cpi_raw is None else 0.0
    row["mapi_missing"] = 1.0 if mapi_raw is None else 0.0
    row["cycles_per_instruction"]          = float(cpi_raw)  if cpi_raw  is not None else CPI_MEDIAN
    row["memory_accesses_per_instruction"] = float(mapi_raw) if mapi_raw is not None else MAPI_MEDIAN
    start_dt = form_data.get("start_dt", datetime.datetime.utcnow())
    end_dt   = form_data.get("end_dt",   start_dt + datetime.timedelta(hours=1))
    start_ts = start_dt.timestamp()
    end_ts   = end_dt.timestamp()
    row["hour_of_day"]      = float(start_dt.hour)
    row["day_of_week"]      = float(start_dt.weekday())
    row["is_peak_hour"]     = 1.0 if 9 <= start_dt.hour <= 17 else 0.0
    row["time_of_week_sec"] = float(int(start_ts) % 604_800)
    explicit_duration = form_data.get("job_duration_sec", None)
    row["job_duration_sec"] = max(0.0, float(explicit_duration)) if explicit_duration is not None else max(0.0, end_ts - start_ts)
    selected_event = str(form_data.get("event", "SCHEDULE")).upper().strip()
    for col in EVENT_COLS:
        row[col] = 1.0 if col.replace("event_", "") == selected_event else 0.0
    selected_cluster = int(form_data.get("cluster", 1))
    for col in CLUSTER_COLS:
        row[col] = 1.0 if int(col.replace("cluster_", "")) == selected_cluster else 0.0
    return pd.DataFrame([row])[FEATURE_COLUMNS]
