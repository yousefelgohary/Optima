"""
src/config.py
=============
Central configuration module for Optima Dashboard.
Loads the feature manifest once and exposes all typed constants used
across the backend pipeline. No business logic lives here.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — resolves correctly regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "optima_models"
DATA_PATH: Path = PROJECT_ROOT / "datasets" / "borg_traces_data.csv"
MANIFEST_PATH: Path = MODELS_DIR / "optima_feature_manifest.json"

# ---------------------------------------------------------------------------
# Load feature manifest
# ---------------------------------------------------------------------------
with open(MANIFEST_PATH, "r", encoding="utf-8") as _f:
    _MANIFEST: dict = json.load(_f)

# ---------------------------------------------------------------------------
# Feature schema constants (sourced strictly from manifest)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = _MANIFEST["feature_columns"]       # 42 features, exact order
N_FEATURES: int             = _MANIFEST["n_features"]            # 42
EVENT_COLS: list[str]       = _MANIFEST["event_columns"]         # 10 event_* one-hot cols
CLUSTER_COLS: list[str]     = _MANIFEST["cluster_columns"]       # 8 cluster_* one-hot cols

TARGET_CPU: str    = _MANIFEST["target_cpu"]     # "resource_request_cpus"
TARGET_MEM: str    = _MANIFEST["target_memory"]  # "resource_request_memory"

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
CPU_CLIP_UPPER: float = _MANIFEST["cpu_clip_upper"]       # 0.12548828125
MEM_CLIP_UPPER: float = _MANIFEST["mem_clip_upper"]       # 0.1041259765625
CPI_MEDIAN: float     = _MANIFEST["cpi_median_fill"]      # 1.918680549
MAPI_MEDIAN: float    = _MANIFEST["mapi_median_fill"]     # 0.009505982

# ---------------------------------------------------------------------------
# Model file references
# ---------------------------------------------------------------------------
BEST_CPU_MODEL_FILE: str = _MANIFEST["best_cpu_model_file"]  # optima_cpu_hgbr.joblib
BEST_MEM_MODEL_FILE: str = _MANIFEST["best_mem_model_file"]  # optima_mem_hgbr.joblib

SCALER_FILE: str = "optima_scaler.joblib"

# Full path shortcuts
BEST_CPU_MODEL_PATH: Path = MODELS_DIR / BEST_CPU_MODEL_FILE
BEST_MEM_MODEL_PATH: Path = MODELS_DIR / BEST_MEM_MODEL_FILE
SCALER_PATH: Path          = MODELS_DIR / SCALER_FILE

# ---------------------------------------------------------------------------
# Benchmark metrics (from manifest — for display on overview page)
# ---------------------------------------------------------------------------
BEST_CPU_METRICS: dict = _MANIFEST["best_cpu_metrics"]
BEST_MEM_METRICS: dict = _MANIFEST["best_mem_metrics"]
TRAIN_ROWS: int         = _MANIFEST["train_rows"]   # 324096
TEST_ROWS: int          = _MANIFEST["test_rows"]    # 81024

# ---------------------------------------------------------------------------
# UI / domain constants
# ---------------------------------------------------------------------------

# Scheduling class labels for form dropdowns
SCHEDULING_CLASS_LABELS: dict[int, str] = {
    0: "0 — Batch (Free-tier)",
    1: "1 — Best-Effort",
    2: "2 — Production",
    3: "3 — Monitoring (Critical)",
}

# Collection type labels
COLLECTION_TYPE_LABELS: dict[int, str] = {
    0: "0 — Job",
    1: "1 — Alloc Set",
}

# Vertical scaling labels
VERTICAL_SCALING_LABELS: dict[int, str] = {
    0: "0 — Disabled",
    1: "1 — Enabled",
    2: "2 — Partial",
}

# Cluster IDs for one-hot encoding
CLUSTER_IDS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]

# Event type names (stripped of "event_" prefix for UI)
EVENT_NAMES: list[str] = [col.replace("event_", "") for col in EVENT_COLS]

# All model keys — cpu first, then mem
ALL_MODEL_KEYS: list[str] = [
    "cpu_hgbr", "cpu_rf", "cpu_ridge",
    "mem_hgbr", "mem_rf", "mem_ridge",
]

# Human-readable model name map
MODEL_NAME_MAP: dict[str, str] = {
    "cpu_hgbr":  "HGBR (Best)",
    "cpu_rf":    "Random Forest",
    "cpu_ridge": "Ridge L2",
    "mem_hgbr":  "HGBR (Best)",
    "mem_rf":    "Random Forest",
    "mem_ridge": "Ridge L2",
}

# Per-model R² scores for display badges (from earlier research)
MODEL_METRICS: dict[str, dict] = {
    "cpu_hgbr":  {"RMSE": 0.005969, "MAE": 0.002216, "R2": 0.915760},
    "cpu_rf":    {"RMSE": 0.007514, "MAE": 0.003571, "R2": 0.866523},
    "cpu_ridge": {"RMSE": 0.015179, "MAE": 0.008568, "R2": 0.455257},
    "mem_hgbr":  {"RMSE": 0.003936, "MAE": 0.001109, "R2": 0.951990},
    "mem_rf":    {"RMSE": 0.004424, "MAE": 0.001374, "R2": 0.939348},
    "mem_ridge": {"RMSE": 0.008283, "MAE": 0.003750, "R2": 0.787410},
}

# Feature importance scores (from HGBR, top drivers)
CPU_FEATURE_IMPORTANCES: dict[str, float] = {
    "avg_usage_cpus":     0.395,
    "assigned_memory":    0.311,
    "page_cache_memory":  0.044,
    "max_usage_memory":   0.033,
    "max_usage_cpus":     0.031,
    "sample_cpus":        0.028,
    "avg_usage_memory":   0.022,
    "priority":           0.015,
    "scheduling_class":   0.012,
    "other":              0.109,
}

MEM_FEATURE_IMPORTANCES: dict[str, float] = {
    "assigned_memory":    0.925,
    "avg_usage_memory":   0.014,
    "page_cache_memory":  0.012,
    "max_usage_memory":   0.010,
    "sample_memory":      0.008,
    "other":              0.031,
}
