"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → get weather_type + confidence
  4. Try LLM for a contextual recommendation, fallback to rule-based
  5. Return clean text response (no markdown, single language)
"""

import logging
import random

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

# Varied rule-based recommendations
_RULE_BASED_ADVICE = {
    "sunny": [
        "Ясная погода, отлично! Футболка, шорты или лёгкое платье — самое то. Не забудь солнцезащитные очки и крем от загара.",
        "Солнечно и тепло — одевайся полегче. Кепка и очки спасут от яркого света, а лёгкая одежда не даст перегреться.",
        "Отличный день! Худи не понадобится — хватит футболки и шорт. SPF обязателен.",
    ],
    "cloudy": [
        "Небо затянуло облаками — лучше захвати лёгкую куртку. Зонт брось в сумку, мало ли.",
        "Облачно и прохладно. Кофта или ветровка будут в самый раз. Если планируешь быть на улице долго — тёплый шарф не помешает.",
        "Пасмурно, но без дождя. Одевайся в несколько слоёв — так можно регулировать температуру по ощущениям.",
    ],
    "rainy": [
        "Дождь — без вариантов, нужна водонепроницаемая куртка и зонт. Обувь выбирай резиновую или непромокаемую.",
        "Ливень! Ветровка с капюшоном, зонт-трость, сапоги. Никаких тканевых сумок — только рюкзак с водозащитой.",
        "Мокро и сыро. Дождевик, резиновые ботинки, зонт. Избегай светлой одежды — пятна будут видны.",
    ],
    "snowy": [
        "Снег и холод! Тёплая куртка, шапка, шарф, перчатки — полный комплект. Обувь с утеплением и нескользящей подошвой.",
        "Зимняя погода — одевайся как капуста. Термобельё, свитер, пуховик. Шапку и варежки не забудь.",
        "Морозно и снежно. Тёплые вещи обязательны, особенно шапка и шарф. Если идёшь далеко — термос с чаем спасёт.",
    ],
    "foggy": [
        "Туман — сыро и зябко. Одевайся теплее, чем кажется. Водоотталкивающая куртка будет кстати.",
        "Видимость низкая, влажность высокая. Флисовая кофта, ветровка, может пригодиться зонт.",
        "Туманно и промозгло. Многослойная одежда — лучший вариант. Шарф защитит шею от холода.",
    ],
    "night": [
        "Темно и, скорее всего, холодно. Тёплая куртка, шапка, шарф. Если светоотражающие элементы на одежде — ещё лучше.",
        "Ночная прогулка — одевайся теплее днём. Куртка потеплее, перчатки, может термобельё. Светлая одежда для видимости.",
        "Ночь — холодно. Пуховик, тёплые штаны, шапка. Если есть фонарик или светоотражатели — прикрепи к одежде.",
    ],
}


def _clean_response(text: str) -> str:
    """Remove markdown artifacts and ensure clean plain text.

    Strips *, _, **, __, backticks, and extra whitespace.
    """
    # Remove markdown bold/italic
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", "")
    # Remove backticks
    text = text.replace("`", "")
    # Remove HTML-like tags that sometimes leak
    text = text.replace("<|", "").replace("|>", "")
    # Collapse multiple spaces
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a clothing recommendation.

    Tries LLM first for contextual advice. Falls back to varied
    rule-based recommendations if LLM is unavailable.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Clean text recommendation (no markdown, Russian only).
    """
    result = None

    try:
        # Step 1: Classify weather
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]

        # Step 2: Try LLM for contextual advice
        try:
            recommendation = await llm.get_clothing_recommendation(
                weather_type, user_message
            )
            recommendation = _clean_response(recommendation)
        except llm.LLMError as e:
            logger.warning("LLM unavailable, using fallback: %s", e)
            options = _RULE_BASED_ADVICE.get(weather_type, [])
            recommendation = random.choice(options) if options else (
                "Одевайтесь по погоде и не забудьте зонтик!"
            )

        # Step 3: Format response — emoji + clean text
        emoji = WEATHER_EMOJI.get(weather_type, "🌤️")
        analysis = result.get("analysis", {})

        extra = ""
        if result.get("is_night"):
            extra = " Определено ночное время."
        elif analysis.get("warmth", {}).get("is_warm"):
            extra = " Заметны тёплые тона — возможно закат или рассвет."

        return f"{emoji} Погода: {weather_type} (уверенность {confidence:.0%}).{extra}\n\n{recommendation}"

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "Не удалось распознать погоду. Возможно, сервис классификации "
            "временно недоступен. Попробуйте позже или опишите погоду текстом."
        )

    except Exception as e:
        logger.error("Unexpected error processing photo: %s", e)
        if result:
            weather_type = result.get("weather_type", "unknown")
            options = _RULE_BASED_ADVICE.get(weather_type, [])
            advice = random.choice(options) if options else (
                "Одевайтесь по погоде и не забудьте зонтик!"
            )
            return f"Обработка с ошибкой, но вот совет: {advice}"
        return (
            "Произошла ошибка при обработке фото. Попробуйте ещё раз."
        )


def handle_photo() -> str:
    """Sync placeholder for --test mode."""
    return (
        "Фото получено! Анализирую погоду и подбираю одежду...\n\n"
        "Это заглушка — подключите classifier и LLM для работы."
    )
