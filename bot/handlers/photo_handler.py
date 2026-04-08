"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → full image analysis
  4. Try Qwen LLM → unique contextual recommendation
  5. If LLM fails, return detailed classifier analysis summary
  6. Return clean text response to user
"""

import logging

from bot.services import llm_client as llm
from bot.services import vit_classifier as classifier

logger = logging.getLogger(__name__)

WEATHER_EMOJI = {
    "sunny": "☀️",
    "cloudy": "☁️",
    "rainy": "🌧️",
    "snowy": "❄️",
    "foggy": "🌫️",
    "night": "🌙",
}


def _clean_response(text: str) -> str:
    """Remove markdown artifacts and ensure clean plain text."""
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", "")
    text = text.replace("`", "")
    text = text.replace("<|", "").replace("|>", "")
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a UNIQUE clothing recommendation from LLM.

    The classifier provides full visual analysis. This context is sent to
    LLM which generates a fresh, non-deterministic response every time.
    No pre-written fallbacks — if LLM fails, return an error message.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Clean text recommendation (no markdown, English only).
    """
    try:
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]
        context = result["context_for_llm"]

        if user_message:
            context += f"\nUser also said: {user_message}"

        # LLM generates unique response — no cached/pre-written answers
        try:
            recommendation = await llm.get_clothing_recommendation(context)
            recommendation = _clean_response(recommendation)
            emoji = WEATHER_EMOJI.get(weather_type, "🌤️")
            return f"{emoji} Weather: {weather_type} (confidence {confidence:.0%})\n\n{recommendation}"

        except llm.LLMError as e:
            logger.error("LLM failed: %s", e)
            return (
                f"AI clothing recommendation is temporarily unavailable. "
                f"The weather looks {weather_type} though — dress accordingly!"
            )

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "Could not analyze the photo. "
            "Make sure it's a street photo and try again."
        )

    except Exception as e:
        import traceback
        logger.error("Unexpected error: %s", traceback.format_exc())
        return (
            "An error occurred while processing the photo. "
            "Try again or send a different photo."
        )


def handle_photo() -> str:
    """Sync placeholder for --test mode."""
    return (
        "Photo received! Analyzing weather and finding clothing advice...\n\n"
        "This is a stub — connect the classifier and LLM for real functionality."
    )
