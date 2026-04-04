"""FastAPI service for weather image classification.

Endpoints:
  POST /classify  — classify an uploaded image
  GET  /health    — service health check
"""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from classifier.model_loader import WeatherClassifier

logger = logging.getLogger(__name__)

# Global classifier instance (loaded once at startup)
classifier: WeatherClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, clean up on shutdown."""
    global classifier
    logger.info("Loading SigLIP2 Weather Classification model...")
    classifier = WeatherClassifier()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down classifier service.")


app = FastAPI(
    title="WeatherWear Classifier",
    description="SigLIP2-based weather image classification service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    """Service health endpoint."""
    status = "ok" if classifier is not None else "loading"
    return {"status": status, "model": "prithivMLmods/Weather-Image-Classification"}


@app.post("/classify")
async def classify_image(photo: UploadFile = File(...)) -> dict:
    """Classify an uploaded image and return weather type."""
    if classifier is None:
        return {"error": "Model is still loading, try again in a moment"}

    image_data = await photo.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    result = classifier.predict(image)
    logger.info(
        "Classified image: %s (confidence: %.4f)",
        result["weather_type"],
        result["confidence"],
    )
    return result
