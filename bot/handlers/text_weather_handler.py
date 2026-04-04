"""Handler for text-based weather descriptions.

When user describes weather in text (e.g., "на улице дождь и ветер"),
this handler detects the weather type and provides clothing advice.
"""

import random
import logging

from bot.handlers.text_weather import detect_weather_from_text
from bot.services import llm_client as llm

logger = logging.getLogger(__name__)

WEATHER_EMOJI = {
    "sunny": "☀️",
    "cloudy": "☁️",
    "rainy": "🌧️",
    "snowy": "❄️",
    "foggy": "🌫️",
    "night": "🌙",
}

_DETAILED_ADVICE = {
    "sunny": [
        "Солнечно и тепло — одевайтесь полегче. Хлопковая футболка и шорты, сандалии или лёгкие кеды. Кепка или панама от солнца, солнцезащитные очки SPF 30+. Возьмите воду с собой.",
        "Ясный день! Лёгкая одежда — футболка, шорты или платье. На ноги — сандалии, мокасины или дышащие кроссовки. Не забудьте крем от загара и очки. Светлая одежда поможет от перегрева.",
    ],
    "cloudy": [
        "Облачно и прохладно. Многослойность — ваш друг: футболка + флисовая кофта + лёгкая куртка. Джинсы или брюки, кроссовки. Зонт в сумке на всякий случай. Если ветер — добавьте тонкий шарф.",
        "Пасмурно. Наденьте ветровку или лёгкую куртку поверх кофты. Удобные кроссовки или ботинки. Зонт обязателен — погода может измениться. Если долго на улице, возьмите тёплый шарф.",
    ],
    "rainy": [
        "Дождь — экипируйтесь полностью. Водонепроницаемая куртка с капюшоном, штаны из синтетики (джинсы долго сохнут), резиновые сапоги или непромокаемые ботинки. Зонт большой, рюкзак с водозащитой. Телефон в водонепроницаемый чехол.",
        "Ливень! Дождевик или плащ с капюшоном. Под низ — синтетические штаны, избегайте хлопка. Резиновые ботинки, зонт-трость. Избегайте светлой одежды — грязь видна. Носки с запасом — промокнут.",
    ],
    "snowy": [
        "Снег и мороз. Термобельё + флисовая кофта + тёплый пуховик. Шапка закрывает уши, шарф вокруг шеи, варежки (тёплее перчаток). Утеплённые ботинки с нескользящей подошвой, шерстяные носки. Если гололёд — ледоступы на обувь.",
        "Зимняя погода. Пуховик или парка, свитер, термобельё. Шапка-ушанка или вязаная, шарф, варежки. Тёплые брюки, зимние ботинки с протектором. Термос с горячим чаем не помешает.",
    ],
    "foggy": [
        "Туман и сырость. Водоотталкивающая куртка, флисовая кофта, шарф на шею. Джинсы или брюки, непромокаемые ботинки. Туман конденсируется на ткани — избегайте хлопка, синтетика сохнет быстрее. Светлая одежда для видимости.",
        "Мгла и влажность. Куртка с мембраной, тёплая кофта, тонкие перчатки. Обувь закрытая, с водоотталкивающей пропиткой. Если есть светоотражатели — отлично, в тумане вас плохо видно.",
    ],
    "night": [
        "Ночью холодно. Тёплая куртка или пуховик, шапка, шарф, перчатки. Под куртку — свитер или флиска. Утеплённые ботинки с толстой подошвой. Если светоотражатели на одежде — обязательно. Фонарик лучше телефонного.",
        "Тёмное время, температура падает. Минимум на один слой больше чем днём. Пуховик, шапка закрывает уши, варежки. Термобельё если долго на улице. Светлая одежда или жилет со светоотражателями.",
    ],
}


async def handle_text_weather_async(user_text: str) -> str:
    """Handle user's text description of weather.

    Detects weather type from keywords, then tries LLM for contextual
    advice. Falls back to detailed rule-based recommendations.

    Args:
        user_text: User's text describing the weather.

    Returns:
        Clothing recommendation text.
    """
    # Detect weather type from text
    weather_type = detect_weather_from_text(user_text)

    if weather_type is None:
        return (
            "Я не смог определить погоду по вашему описанию.\n\n"
            "Попробуйте использовать ключевые слова:\n"
            "☀️ солнце, ясно, жарко\n"
            "☁️ облачно, пасмурно, тучи\n"
            "🌧️ дождь, ливень, мокро\n"
            "❄️ снег, мороз, холодно\n"
            "🌫️ туман, мгла, дымка\n"
            "🌙 ночь, темно, вечер\n\n"
            "Или просто отправьте фото улицы!"
        )

    emoji = WEATHER_EMOJI.get(weather_type, "🌤️")

    # Try LLM first
    try:
        context = f"User says the weather is: {weather_type}. Details: {user_text}"
        recommendation = await llm.get_clothing_recommendation(context)
        # Clean response
        recommendation = recommendation.replace("**", "").replace("*", "")
        recommendation = recommendation.replace("`", "")
        import re
        recommendation = re.sub(r"\s+", " ", recommendation).strip()
        return f"{emoji} Определено по описанию: {weather_type}\n\n{recommendation}"

    except llm.LLMError:
        # Fallback to detailed advice
        options = _DETAILED_ADVICE.get(weather_type, [
            "Одевайтесь многослойно — куртка, удобная обувь, зонт на всякий случай."
        ])
        advice = random.choice(options)
        return f"{emoji} Определено по описанию: {weather_type}\n\n{advice}"
