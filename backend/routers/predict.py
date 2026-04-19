"""
Prediction Router
"""

import logging
from fastapi import APIRouter, HTTPException, Request

# ✅ FIXED IMPORTS (RELATIVE)
from ..core.schemas import PatientInput, PredictionResponse
from ..core.model_service import model_service
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: PatientInput, request: Request):

    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        patient_dict = payload.model_dump()

        logger.info(f"Request from {request.client.host} → {patient_dict}")

        result = model_service.predict(patient_dict)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/model-info")
async def model_info():

    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_service.model_name,
        "metrics": model_service.training_metrics,
    }