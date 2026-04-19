"""
Application Configuration
==========================
Centralised settings loaded once at startup.
Uses pydantic-settings for env-var override support.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent   # project root
MODEL_DIR   = BASE_DIR / "backend" / "models"
MODEL_PATH  = MODEL_DIR / "best_model.pkl"
META_PATH   = MODEL_DIR / "model_metadata.json"

# ── API ───────────────────────────────────────────────────────────────────────
API_TITLE       = "Diabetes Prediction API"
API_VERSION     = "1.0.0"
API_DESCRIPTION = (
    "Production-grade ML API for diabetes risk assessment. "
    "Powered by a scikit-learn ensemble pipeline trained on the Pima Indians dataset."
)

# ── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1",
    "null",   # file:// origin
    "*",      # allow all in dev — tighten in production
]

# ── Model feature order (must match training) ──────────────────────────────
FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
