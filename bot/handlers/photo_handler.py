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

WEATHER_EMOJI = {
    "sunny": "☀️",
    "cloudy": "☁️",
    "rainy": "🌧️",
    "snowy": "❄️",
    "foggy": "🌫️",
    "night": "🌙",
}

# Fallback rule-based recommendations
_RULE_BASED_ADVICE = {
    "sunny": (
        "☀️ Солнечно! Лёгкая одежда — футболка, шорты или платье. "
        "Не забудьте солнцезащитные очки и крем с SPF."
    ),
    "cloudy": (
        "☁️ Облачно. Возьмите лёгкую куртку или кофту. "
        "Зонт не помешает на всякий случай."
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
        "но сырость проникает. Зонт не помешает."
    ),
    "night": (
        "🌙 Ночь. Тёплые слои, куртка потеплее. "
        "Если холодно — шапка и перчатки не помешают."
    ),
}


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a clothing recommendation.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Formatted clothing recommendation string.
    """
    result = None

    try:
        # Step 1: Classify weather
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]

        # Step 2: Get clothing recommendation from LLM
        try:
            recommendation = await llm.get_clothing_recommendation(
                weather_type, user_message
            )
        except llm.LLMError as e:
            logger.warning("LLM unavailable, using fallback: %s", e)
            recommendation = _rule_based_advice.get(
                weather_type,
                f"Погода: {weather_type}. Одевайтесь по погоде!",
            )

        # Step 3: Format response
        emoji = WEATHER_EMOJI.get(weather_type, "🌤️")

        # Add analysis context if available
        extra = ""
        analysis = result.get("analysis", {})
        if result.get("is_night"):
            extra = "\n🌑 Определено ночное время."
        elif analysis.get("warmth", {}).get("is_warm"):
            extra = "\n🌅 Заметны тёплые тона (рассвет/закат)."

        return (
            f"{emoji} **Погода:** {weather_type} "
            f"(уверенность: {confidence:.0%}){extra}\n\n"
            f"👔 {recommendation}"
        )

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "🔬 Не удалось распознать погоду.\n\n"
            "Возможно, сервис классификации временно недоступен. "
            "Попробуйте позже или опишите погоду текстом "
            '(например, "на улице дождь").'
        )

    except Exception as e:
        logger.error("Unexpected error processing photo: %s", e)
        # Try to give rule-based advice even if things fail
        if result:
            weather_type = result.get("weather_type", "unknown")
            advice = _RULE_BASED_ADVICE.get(
                weather_type, "Одевайтесь по погоде и не забудьте зонтик!"
            )
            return f"⚠️ Обработка с ошибкой, но вот совет:\n\n{advice}"
        return (
            "⚠️ Произошла ошибка при обработке фото. "
            "Попробуйте ещё раз или опишите погоду текстом."
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
