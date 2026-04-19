# 🩺 GlycoSense — Diabetes Risk Intelligence Platform

> A production-grade, end-to-end machine learning system for diabetes risk assessment, featuring a premium glassmorphism UI, FastAPI backend, and a rigorously-trained sklearn ensemble pipeline.

---

## 📐 Architecture Overview

```
diabetes-prediction-system/
├── backend/
│   ├── core/
│   │   ├── config.py          # Centralised app settings & paths
│   │   ├── schemas.py         # Pydantic input/output models (strict validation)
│   │   └── model_service.py   # ML singleton — load, predict, metadata
│   ├── routers/
│   │   └── predict.py         # POST /api/v1/predict endpoint
│   ├── models/
│   │   ├── best_model.pkl     # Trained sklearn Pipeline (auto-generated)
│   │   └── model_metadata.json
│   └── main.py                # FastAPI app factory + lifespan
│
├── frontend/
│   ├── index.html             # Premium glassmorphism SPA
│   └── js/app.js              # Fetch integration, animations, validation
│
├── ml/
│   ├── train.py               # Full ML training pipeline
│   └── reports/               # Confusion matrix + comparison charts
│
├── data/
│   └── diabetes.csv           # Pima Indians Diabetes dataset
│
├── scripts/
│   └── run.sh                 # One-command launcher
│
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Pipeline

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Split BEFORE scaling** | Prevents data leakage — scaler never sees test data during fit |
| **Sklearn Pipeline** | Scaler/imputer embedded per model; safe in cross-validation |
| **Median imputation** | Physiologically impossible zeros (Glucose, BMI) replaced with NaN, then imputed |
| **Stratified K-Fold (k=5)** | Preserves class balance in every fold |
| **Best model = highest ROC-AUC** | More robust than accuracy for imbalanced classes |

### Models Trained

| Model | Description |
|---|---|
| Logistic Regression | Baseline — interpretable, fast, calibrated probabilities |
| Random Forest | 300 trees, balanced class weights, max_features=sqrt |
| Gradient Boosting | 200 estimators, learning_rate=0.08, subsample=0.85 |
| SVM (RBF kernel) | C=10, gamma=scale, probability=True |

### What Gets Saved
- `best_model.pkl` — the **entire sklearn Pipeline** (imputer → scaler → classifier). No separate scaler file needed; the pipeline handles transforms atomically.
- `model_metadata.json` — training metrics, confusion matrix, classification report

---

## ⚙️ Backend API

### Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Health check + model status |
| `GET` | `/health` | Detailed health check |
| `POST` | `/api/v1/predict` | Run ML inference |
| `GET` | `/api/v1/model-info` | Training metrics |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### POST `/api/v1/predict` — Request Body

```json
{
  "Pregnancies": 3,
  "Glucose": 128.0,
  "BloodPressure": 74.0,
  "SkinThickness": 23.0,
  "Insulin": 94.0,
  "BMI": 29.6,
  "DiabetesPedigreeFunction": 0.537,
  "Age": 36
}
```

### Response

```json
{
  "prediction": 1,
  "label": "Diabetic",
  "probability_diabetic": 0.7842,
  "probability_healthy": 0.2158,
  "confidence": "HIGH",
  "risk_level": {
    "label": "High Risk",
    "color": "#f97316",
    "icon": "🔶"
  },
  "model_used": "GradientBoosting",
  "feature_importances": {
    "Glucose": 0.3421,
    "BMI": 0.1876,
    "Age": 0.1543,
    ...
  }
}
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Option A — One Command
```bash
chmod +x scripts/run.sh
./scripts/run.sh --train --open
```

### Option B — Manual Steps

**1. Create virtual environment & install deps**
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the ML model**
```bash
python3 ml/train.py
```
Expected output:
```
══════════════════════════════════════════════════════
  DIABETES PREDICTION — ML TRAINING PIPELINE
══════════════════════════════════════════════════════
[1/5] Loading & cleaning data …
[2/5] Splitting data (stratified, no leakage) …
[3/5] Training & evaluating models …
  ── LogisticRegression
  ── RandomForest
  ── GradientBoosting
  ── SVM_RBF
[4/5] Selecting best model …
  🏆 Winner : GradientBoosting
     Accuracy : 0.8117
     F1 Score : 0.8654
     ROC-AUC  : 0.8892
[5/5] Generating report visuals …
✅ Training complete!
```

**3. Start FastAPI backend**
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Open the frontend**
```bash
# Simply open frontend/index.html in your browser
open frontend/index.html          # macOS
xdg-open frontend/index.html      # Linux
start frontend/index.html         # Windows
```

Or, the frontend is also served by FastAPI at `http://127.0.0.1:8000/app`

---

## 🎨 Frontend Features

| Feature | Implementation |
|---|---|
| Glassmorphism design | `backdrop-filter: blur()` + semi-transparent surfaces |
| Animated mesh background | CSS `radial-gradient` + floating orbs with `keyframes` |
| Shimmer button effect | CSS `::after` pseudo-element sweep |
| Probability bars | CSS transition on `width` with 1.1s ease |
| Feature importance chart | Dynamically generated bars with staggered animation |
| API status pill | Live health-check every 30 seconds |
| Demo data loader | 3 reference cases cycling on button click |
| Error toast | Slide-in notification with auto-dismiss |
| Form validation | Client-side before any API call |

---

## 🔒 Production Hardening Checklist

- [ ] Replace `ALLOWED_ORIGINS = ["*"]` with specific domain(s)
- [ ] Add rate limiting (e.g., `slowapi`)
- [ ] Add JWT authentication for protected endpoints
- [ ] Store predictions in PostgreSQL for audit trail
- [ ] Add Prometheus metrics middleware
- [ ] Containerise with Docker + docker-compose
- [ ] Set up CI/CD pipeline with model retraining triggers
- [ ] Add model monitoring (data drift detection)

---

## 📊 API Testing with curl

```bash
# Health check
curl http://127.0.0.1:8000/

# Predict (high-risk profile)
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

---

## ⚠️ Medical Disclaimer

This system is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
