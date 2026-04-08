"""Handler for text-based weather descriptions.

When user describes weather in text (e.g., "it's raining and windy"),
this handler detects the weather type and provides clothing advice.
"""

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


async def handle_text_weather_async(user_text: str) -> str:
    """Handle user's text description of weather.

    Detects weather type from keywords, then tries LLM for contextual
    advice. Falls back to detailed rule-based recommendations.

    Args:
        user_text: User's text describing the weather.

    Returns:
        Clothing recommendation text.
    """
    weather_type = detect_weather_from_text(user_text)

    if weather_type is None:
        return (
            "I couldn't determine the weather from your description.\n\n"
            "Try using keywords like:\n"
            "☀️ sunny, clear, hot\n"
            "☁️ cloudy, overcast, grey\n"
            "🌧️ rain, wet, storm\n"
            "❄️ snow, cold, frost\n"
            "🌫️ fog, mist, hazy\n"
            "🌙 night, dark, evening\n\n"
            "Or just send a street photo!"
        )

    emoji = WEATHER_EMOJI.get(weather_type, "🌤️")

    try:
        context = f"User says the weather is: {weather_type}. Details: {user_text}"
        recommendation = await llm.get_clothing_recommendation(context)
        recommendation = recommendation.replace("**", "").replace("*", "")
        recommendation = recommendation.replace("`", "")
        import re
        recommendation = re.sub(r"\s+", " ", recommendation).strip()
        return f"{emoji} Detected from description: {weather_type}\n\n{recommendation}"

    except llm.LLMError:
        return f"{emoji} Detected from description: {weather_type}\n\nAI recommendation is unavailable right now. Try sending a street photo instead!"
