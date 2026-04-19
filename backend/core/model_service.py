"""
Model Service
=============
Singleton that loads the trained sklearn Pipeline once at startup
and exposes a clean inference interface.

Thread-safe: FastAPI workers share the same process; joblib-loaded
models are read-only after loading, so no locking is required.
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple

from backend.core.config import MODEL_PATH, META_PATH, FEATURE_ORDER

logger = logging.getLogger(__name__)


class ModelService:

    def __init__(self):
        self._pipeline = None
        self._metadata: dict = {}
        self._feature_names = FEATURE_ORDER

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Called once during FastAPI startup."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )

        logger.info(f"Loading model from {MODEL_PATH} …")

        self._pipeline = joblib.load(MODEL_PATH)

        logger.info("Model loaded ✓")

        if META_PATH.exists():
            with open(META_PATH) as f:
                self._metadata = json.load(f)

            logger.info(
                f"Metadata loaded → best model: {self._metadata.get('best_model')}"
            )
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        return self._metadata.get("best_model", "Unknown")

    @property
    def training_metrics(self) -> dict:
        return self._metadata.get("metrics", {})

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, patient_data: dict) -> dict:
        """
        Args:
            patient_data: dict matching FEATURE_ORDER keys

        Returns:
            dict with prediction, probabilities, confidence band,
            risk level, and optional feature importances.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call ModelService.load() first.")

        # ── Build input DataFrame in correct column order ─────────────────────
        X = pd.DataFrame([patient_data], columns=self._feature_names)

        # ── Inference ─────────────────────────────────────────────────────────
        prediction    = int(self._pipeline.predict(X)[0])
        probabilities = self._pipeline.predict_proba(X)[0]        # [P(0), P(1)]
        prob_diabetic = float(probabilities[1])
        prob_healthy  = float(probabilities[0])

        # ── Confidence band ───────────────────────────────────────────────────
        max_prob = max(prob_diabetic, prob_healthy)
        if max_prob >= 0.80:
            confidence = "HIGH"
        elif max_prob >= 0.65:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ── Risk level (clinical-style tiering) ───────────────────────────────
        risk_level = self._compute_risk_level(prob_diabetic)

        # ── Feature importances (RF / GB only) ───────────────────────────────
        feature_importances = self._get_feature_importances()

        return {
            "prediction":           prediction,
            "label":               "Diabetic" if prediction == 1 else "Not Diabetic",
            "probability_diabetic": round(prob_diabetic, 4),
            "probability_healthy":  round(prob_healthy,  4),
            "confidence":           confidence,
            "risk_level":           risk_level,
            "model_used":           self.model_name,
            "feature_importances":  feature_importances,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_risk_level(prob_diabetic: float) -> dict:
        if prob_diabetic < 0.25:
            return {"label": "Low Risk",      "color": "#22c55e", "icon": "✅"}
        elif prob_diabetic < 0.50:
            return {"label": "Moderate Risk", "color": "#f59e0b", "icon": "⚠️"}
        elif prob_diabetic < 0.75:
            return {"label": "High Risk",     "color": "#f97316", "icon": "🔶"}
        else:
            return {"label": "Very High Risk","color": "#ef4444", "icon": "🚨"}

    def _get_feature_importances(self) -> Optional[dict]:
        """Extract feature importances from tree-based models."""
        try:
            clf = self._pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                ranked = sorted(
                    zip(self._feature_names, importances),
                    key=lambda x: x[1], reverse=True
                )
                return {name: round(float(val), 4) for name, val in ranked}
        except Exception:
            pass
        return None


# ── Module-level singleton ────────────────────────────────────────────────────
model_service = ModelService()
