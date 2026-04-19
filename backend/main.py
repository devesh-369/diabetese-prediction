"""
Diabetes Prediction API — Main Application
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ✅ FIXED IMPORTS (RELATIVE)
from .core.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, ALLOWED_ORIGINS
)
from .core.model_service import model_service
from .core.schemas import HealthResponse
from .routers.predict import router as predict_router


# ── Logging ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading ML model …")
    try:
        model_service.load()
        logger.info(f"Model ready: {model_service.model_name}")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.warning("Model not found. Train model first.")
    yield
    logger.info("Shutting down …")


# ── App ────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


# ── CORS ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 ye change karo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware ─────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter() - start)*1000:.2f}ms"
    return response


# ── Exception Handler ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Routes ────────────────────────────────────────
app.include_router(predict_router)


# ── Health Check ──────────────────────────────────
@app.get("/")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded,
        "model_name": model_service.model_name if model_service.is_loaded else "not loaded",
        "version": API_VERSION,
    }


# ── Static Frontend ───────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logger.info(f"Frontend available at http://127.0.0.1:8000/app")


# ── Run Server ────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )