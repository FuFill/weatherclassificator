"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → get weather_type + confidence
  4. Send weather_type to Qwen LLM → get clothing recommendation
  5. Return formatted response to user
"""

import logging

from bot.services import llm_client as llm
from bot.services import vit_classifier as classifier

logger = logging.getLogger(__name__)


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a clothing recommendation.

    This is the async version that actually calls the services.
    The sync handle_photo() is a placeholder for --test mode.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Formatted clothing recommendation string.
    """
    try:
        # Step 1: Classify weather
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]

        # Step 2: Get clothing recommendation from LLM
        recommendation = await llm.get_clothing_recommendation(weather_type, user_message)

        # Step 3: Format response
        weather_emoji = {
            "sunny": "☀️",
            "cloudy": "☁️",
            "rainy": "🌧️",
            "snowy": "❄️",
            "foggy": "🌫️",
        }.get(weather_type, "🌤️")

        return (
            f"{weather_emoji} **Погода:** {weather_type} "
            f"(уверенность: {confidence:.0%})\n\n"
            f"👔 {recommendation}"
        )

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "🔬 Не удалось распознать погоду.\n\n"
            "Сервис классификации временно недоступен. "
            "Попробуйте позже или опишите погоду текстом "
            '(например, "на улице дождь").'
        )

    except llm.LLMError as e:
        logger.error("LLM error: %s", e)
        # Fallback: give a rule-based recommendation
        result_fallback = result if "result" in dir() else None
        if result_fallback:
            weather_type = result_fallback.get("weather_type", "unknown")
            return _rule_based_advice(weather_type)
        return (
            "🧠 Не удалось получить рекомендацию от ИИ.\n\n"
            "Сервис временно недоступен. "
            "Попробуйте позже!"
        )


def _rule_based_advice(weather_type: str) -> str:
    """Fallback clothing recommendation without LLM."""
    advice = {
        "sunny": (
            "☀️ Солнечно! Лёгкая футболка, шорты или платье. "
            "Не забудьте солнцезащитные очки и SPF."
        ),
        "cloudy": (
            "☁️ Облачно. Лёгкая куртка или кофта. "
            "Возьмите зонт на всякий случай."
        ),
        "rainy": (
            "🌧️ Дождь! Водонепроницаемая куртка, зонт, "
            "непромокаемая обувь. Избегайте светлых тканей."
        ),
        "snowy": (
            "❄️ Снег! Тёплая куртка, шапка, шарф, перчатки. "
            "Утеплённая обувь с нескользящей подошвой."
        ),
        "foggy": (
            "🌫️ Туман. Одевайтесь теплее — видимость низкая, "
            "ветра нет, но сырость проникает. Зонт не помешает."
        ),
    }
    return advice.get(
        weather_type,
        "Не удалось определить погоду. "
        "Попробуйте другое фото или опишите погоду текстом.",
    )


def handle_photo() -> str:
    """Sync placeholder for --test mode.

    The real async version (handle_photo_async) is called
    from the Telegram handler where we have the event loop.
    """
    return (
        "📸 Фото получено!\n\n"
        "⏳ Анализирую погоду и подбираю одежду...\n\n"
        "🔧 Это заглушка — подключите classifier и LLM для работы."
    )
