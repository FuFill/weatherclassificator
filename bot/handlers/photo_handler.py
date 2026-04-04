"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → full image analysis
  4. Try Qwen LLM → unique contextual recommendation
  5. If LLM fails, return classifier analysis summary
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

WEATHER_ADVICE = {
    "sunny": "Ясно и солнечно — легкая одежда, очки, SPF крем.",
    "cloudy": "Облачно — возьмите легкую куртку или кофту.",
    "rainy": "Дождь — водонепроницаемая куртка, зонт, непромокаемая обувь.",
    "snowy": "Снег — теплая куртка, шапка, шарф, перчатки.",
    "foggy": "Туман — одевайтесь теплее, зонт не помешает.",
    "night": "Ночь — теплые слои, куртка, шапка.",
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


def _build_fallback_response(result: dict) -> str:
    """Build a response from classifier analysis when LLM is unavailable."""
    weather_type = result["weather_type"]
    confidence = result["confidence"]
    emoji = WEATHER_EMOJI.get(weather_type, "🌤️")

    features = result.get("visual_features", {})
    brightness = features.get("brightness", {})
    color = features.get("color_temperature", {})
    contrast = features.get("contrast", {})

    details = []
    details.append(f"Яркость: {brightness.get('brightness_0_255', '?')}/255")
    details.append(f"Тона: {color.get('dominant_tone', '?')}")
    details.append(f"Контраст: {contrast.get('contrast_level', '?')}")

    advice = WEATHER_ADVICE.get(
        weather_type, "Одевайтесь по погоде!"
    )

    return (
        f"{emoji} Определено: {weather_type} (уверенность {confidence:.0%})\n"
        f"Детали: {', '.join(details)}\n\n"
        f"{advice}"
    )


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a clothing recommendation from LLM.

    Falls back to classifier-based analysis if LLM is unavailable.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Clean text recommendation (no markdown, Russian only).
    """
    try:
        # Step 1: Full image analysis from ViT
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]
        context = result["context_for_llm"]

        if user_message:
            context += f"\nUser also said: {user_message}"

        # Step 2: Try LLM for unique advice
        try:
            recommendation = await llm.get_clothing_recommendation(context)
            recommendation = _clean_response(recommendation)
            emoji = WEATHER_EMOJI.get(weather_type, "🌤️")
            return f"{emoji} Определено: {weather_type} (уверенность {confidence:.0%})\n\n{recommendation}"

        except llm.LLMError:
            # Fallback to classifier analysis
            return _build_fallback_response(result)

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
