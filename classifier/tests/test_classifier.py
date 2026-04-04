"""Tests for the weather classifier."""

from PIL import Image
import pytest

from classifier.model_loader import WeatherClassifier, WEATHER_DISPLAY


@pytest.fixture(scope="module")
def classifier() -> WeatherClassifier:
    """Load the model once for all tests in this module."""
    return WeatherClassifier()


def test_predict_returns_weather_type(classifier: WeatherClassifier) -> None:
    """Prediction should return a valid weather type."""
    # Create a simple blue image (sky-like)
    img = Image.new("RGB", (224, 224), color=(135, 206, 250))
    result = classifier.predict(img)

    assert "weather_type" in result
    assert "confidence" in result
    assert "all_scores" in result
    assert result["weather_type"] in WEATHER_DISPLAY.values()
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_returns_all_scores(classifier: WeatherClassifier) -> None:
    """All 5 weather classes should be present in scores."""
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    result = classifier.predict(img)

    assert len(result["all_scores"]) == 5
    for weather_name in WEATHER_DISPLAY.values():
        assert weather_name in result["all_scores"]


def test_health_endpoint() -> None:
    """Test the FastAPI health endpoint."""
    from fastapi.testclient import TestClient
    from classifier.app import app, classifier as app_classifier

    # Skip if model not loaded (this test is meant to run with model loaded)
    if app_classifier is None:
        pytest.skip("Model not loaded in test environment")

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Weather-Image-Classification" in data["model"]
