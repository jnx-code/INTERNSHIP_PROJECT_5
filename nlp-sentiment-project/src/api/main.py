"""
src/api/main.py
FastAPI application for sentiment analysis.

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from loguru import logger

from src.inference.predictor import SentimentPredictor
from src.monitoring.tracker import tracker

# ------------------------------------------------------------------ #
# Globals
# ------------------------------------------------------------------ #
predictor: Optional[SentimentPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model …")
    try:
        predictor = SentimentPredictor()
        logger.success("Model loaded ✅")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model — train first with: python src/training/train.py")
    yield
    logger.info("Shutting down …")


# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Sentiment Analysis API",
    description="Bidirectional LSTM sentiment classifier (Negative / Neutral / Positive)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #
class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def validate_list(cls, v):
        if not v:
            raise ValueError("texts list must not be empty")
        if len(v) > 100:
            raise ValueError("Batch size limit is 100")
        return [t.strip() for t in v if t.strip()]


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #
@app.get("/", tags=["Root"])
def root():
    return {"message": "Sentiment Analysis API is running 🚀", "docs": "/docs"}


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok" if predictor else "model_not_loaded",
        "model_ready": predictor is not None,
    }


@app.get("/metrics", tags=["System"])
def metrics():
    return tracker.snapshot()


@app.post("/predict", tags=["Inference"])
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Run training first.")
    t0 = time.perf_counter()
    try:
        result = predictor.predict_one(req.text)
        latency = (time.perf_counter() - t0) * 1000
        tracker.record(result["label"], latency)
        result["latency_ms"] = round(latency, 2)
        return result
    except Exception as e:
        tracker.record("error", 0, error=True)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))


@app.post("/batch_predict", tags=["Inference"])
def batch_predict(req: BatchPredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Run training first.")
    t0 = time.perf_counter()
    try:
        results  = predictor.predict_batch(req.texts)
        latency  = (time.perf_counter() - t0) * 1000
        for r in results:
            tracker.record(r["label"], latency / len(results))
        return {"results": results, "count": len(results), "total_latency_ms": round(latency, 2)}
    except Exception as e:
        tracker.record("error", 0, error=True)
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(500, str(e))
