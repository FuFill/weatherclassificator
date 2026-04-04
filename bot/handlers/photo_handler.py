"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → full image analysis
  4. Send analysis to Qwen LLM → unique contextual recommendation
  5. Return clean text response to user

NO rule-based fallbacks. If LLM fails, return an error message.
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
    """Process a photo and return a unique clothing recommendation from LLM.

    The classifier provides full visual analysis (weather, brightness,
    color temperature, contrast, saturation). This context is sent to
    LLM for a contextual, unique response every time.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Clean text recommendation (no markdown, Russian only).

    Raises:
        Returns error message if classifier or LLM fails.
    """
    try:
        # Step 1: Full image analysis from ViT
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]
        context = result["context_for_llm"]

        # Add user message context if provided
        if user_message:
            context += f"\nUser also said: {user_message}"

        # Step 2: LLM generates unique advice from full context
        try:
            recommendation = await llm.get_clothing_recommendation(context)
            recommendation = _clean_response(recommendation)
        except llm.LLMError as e:
            logger.error("LLM error: %s", e)
            return (
                "ИИ-сервис временно недоступен. "
                "Попробуйте через пару минут или отправьте фото позже."
            )

        # Step 3: Format response
        emoji = WEATHER_EMOJI.get(weather_type, "🌤️")

        return f"{emoji} Определено: {weather_type} (уверенность {confidence:.0%})\n\n{recommendation}"

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "Не удалось проанализировать фото. "
            "Убедитесь что это фотография улицы и попробуйте снова."
        )

    except Exception as e:
        logger.error("Unexpected error processing photo: %s", e)
        return (
            "Произошла ошибка при обработке фото. "
            "Попробуйте ещё раз или отправьте другое фото."
        )


def handle_photo() -> str:
    """Sync placeholder for --test mode."""
    return (
        "Фото получено! Анализирую погоду и подбираю одежду...\n\n"
        "Это заглушка — подключите classifier и LLM для работы."
    )
