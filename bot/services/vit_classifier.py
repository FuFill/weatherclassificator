"""HTTP client for the Weather Classifier service.

Sends images to the classifier microservice and receives
weather type predictions.
"""

import logging

import httpx

from bot.config import settings

logger = logging.getLogger(__name__)


class ClassifierError(Exception):
    """Raised when the classifier service is unreachable or returns an error."""


async def classify_image(image_bytes: bytes, filename: str = "photo.jpg") -> dict:
    """Send an image to the classifier and return the prediction result.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG).
        filename: Original filename (for logging).

    Returns:
        dict with keys:
          - weather_type (str): sunny / cloudy / rainy / snowy / foggy
          - confidence (float): 0.0 - 1.0
          - all_scores (dict): probabilities for all classes

    Raises:
        ClassifierError: If the service is unreachable or returns an error.
    """
    url = f"{settings.classifier_api_base_url}/classify"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {
                "photo": (filename, image_bytes, "image/jpeg"),
            }
            response = await client.post(url, files=files)
            response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise ClassifierError(result["error"])

        logger.info(
            "Classified %s → %s (confidence: %.4f)",
            filename,
            result["weather_type"],
            result["confidence"],
        )
        return result

    except httpx.ConnectError as e:
        raise ClassifierError(
            f"Cannot connect to classifier service at {url}. "
            f"Is the classifier container running?"
        ) from e
    except httpx.TimeoutException as e:
        raise ClassifierError(
            "Classifier service timed out. The model might be busy."
        ) from e
    except httpx.HTTPStatusError as e:
        raise ClassifierError(
            f"Classifier returned HTTP {e.response.status_code}: {e.response.text}"
        ) from e


async def check_classifier_health() -> dict:
    """Check if the classifier service is alive.

    Returns:
        dict with status info, e.g. {"status": "ok", "model": "..."}.

    Raises:
        ClassifierError: If the service is unreachable.
    """
    url = f"{settings.classifier_api_base_url}/health"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()
    except httpx.ConnectError as e:
        raise ClassifierError(
            f"Classifier service unreachable at {url}"
        ) from e
