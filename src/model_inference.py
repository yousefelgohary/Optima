"""
src/model_inference.py
======================
Production inference module — HGBR models only.

Loads optima_cpu_hgbr.joblib and optima_mem_hgbr.joblib.
All other models are excluded from the loading pipeline.

Best-model metrics (from optima_feature_manifest.json):
  CPU HGBR  ->  RMSE=0.005969  MAE=0.002216  R2=0.9158
  MEM HGBR  ->  RMSE=0.003936  MAE=0.001109  R2=0.9520
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import streamlit as st

from src.config import MODELS_DIR, CPU_CLIP_UPPER, MEM_CLIP_UPPER

logger = logging.getLogger(__name__)

_HGBR_FILES: dict[str, str] = {
    "cpu_hgbr": "optima_cpu_hgbr.joblib",
    "mem_hgbr": "optima_mem_hgbr.joblib",
    "scaler":   "optima_scaler.joblib",
}

CPU_R2  = 0.9158
MEM_R2  = 0.9520
CPU_RMSE = 0.005969
MEM_RMSE = 0.003936


@st.cache_resource(show_spinner="Loading HGBR models...")
def load_models() -> dict:
    """
    Load the two HGBR production models and the StandardScaler.
    Called once per Streamlit server process.

    Returns
    -------
    dict  keys: "cpu_hgbr", "mem_hgbr", "scaler"
    """
    loaded: dict = {}
    for key, fname in _HGBR_FILES.items():
        path = MODELS_DIR / fname
        logger.info("Loading %s from %s", key, path)
        loaded[key] = joblib.load(path)
    logger.info("HGBR models loaded successfully.")
    return loaded


def predict_cpu(scaled_X: np.ndarray, models: dict) -> float:
    """Predict CPU resource request with HGBR, clipped to training bounds."""
    raw = float(models["cpu_hgbr"].predict(scaled_X)[0])
    return float(np.clip(raw, 0.0, CPU_CLIP_UPPER))


def predict_mem(scaled_X: np.ndarray, models: dict) -> float:
    """Predict MEM resource request with HGBR, clipped to training bounds."""
    raw = float(models["mem_hgbr"].predict(scaled_X)[0])
    return float(np.clip(raw, 0.0, MEM_CLIP_UPPER))


def predict_both(scaled_X: np.ndarray, models: dict) -> dict[str, float]:
    """Run both HGBR models and return a dict with cpu and mem predictions."""
    return {
        "cpu": predict_cpu(scaled_X, models),
        "mem": predict_mem(scaled_X, models),
    }
