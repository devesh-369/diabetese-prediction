"""
Pydantic Schemas
================
Strict input validation with domain-specific constraints.
FastAPI uses these to auto-generate OpenAPI docs and reject bad inputs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PatientInput(BaseModel):
    """
    All 8 clinical features required by the Pima Indians model.
    Each field carries realistic physiological bounds derived from
    published literature — invalid values are rejected at the API layer,
    before any ML inference occurs.
    """

    Pregnancies: int = Field(
        ..., ge=0, le=20,
        description="Number of times pregnant",
        example=3
    )
    Glucose: float = Field(
        ..., ge=40, le=300,
        description="Plasma glucose concentration (mg/dL) — 2-hour OGTT",
        example=120.0
    )
    BloodPressure: float = Field(
        ..., ge=20, le=180,
        description="Diastolic blood pressure (mm Hg)",
        example=70.0
    )
    SkinThickness: float = Field(
        ..., ge=0, le=110,
        description="Triceps skinfold thickness (mm)",
        example=23.0
    )
    Insulin: float = Field(
        ..., ge=0, le=900,
        description="2-Hour serum insulin (mu U/ml)",
        example=85.0
    )
    BMI: float = Field(
        ..., ge=10.0, le=75.0,
        description="Body mass index (kg/m²)",
        example=28.5
    )
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0.0, le=3.0,
        description="Diabetes pedigree function (genetic risk score)",
        example=0.452
    )
    Age: int = Field(
        ..., ge=18, le=120,
        description="Age in years",
        example=35
    )

    @field_validator("Glucose")
    @classmethod
    def glucose_not_critically_low(cls, v):
        if v < 44:
            raise ValueError("Glucose below 44 mg/dL indicates severe hypoglycaemia — check input.")
        return v

    @field_validator("BMI")
    @classmethod
    def bmi_sanity(cls, v):
        if v < 10:
            raise ValueError("BMI below 10 is incompatible with life — check input.")
        return round(v, 1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 3,
                "Glucose": 128.0,
                "BloodPressure": 74.0,
                "SkinThickness": 23.0,
                "Insulin": 94.0,
                "BMI": 29.6,
                "DiabetesPedigreeFunction": 0.537,
                "Age": 36,
            }
        }
    }


class RiskLevel(BaseModel):
    label: str
    color: str
    icon: str


class PredictionResponse(BaseModel):
    """Structured API response sent back to the frontend."""
    prediction:       int           = Field(..., description="0 = No Diabetes, 1 = Diabetes")
    label:            str           = Field(..., description="Human-readable prediction label")
    probability_diabetic: float     = Field(..., description="Probability of diabetes [0–1]")
    probability_healthy:  float     = Field(..., description="Probability of no diabetes [0–1]")
    confidence:       str           = Field(..., description="HIGH / MEDIUM / LOW confidence band")
    risk_level:       RiskLevel
    model_used:       str
    feature_importances: Optional[dict] = None


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    model_name:  str
    api_version: str


class ErrorResponse(BaseModel):
    detail: str
    code:   int
