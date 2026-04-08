"""Text-based weather intent detection.

When users describe weather in text instead of sending a photo,
this module detects the weather type and provides clothing advice.
"""

# Keywords that indicate each weather type in English
WEATHER_KEYWORDS = {
    "sunny": [
        "sun", "sunny", "clear", "bright", "hot", "warm",
        "heat", "shine", "blue sky", "no clouds",
    ],
    "cloudy": [
        "cloud", "cloudy", "overcast", "grey", "gray",
        "gloomy", "dull", "hazy sky",
    ],
    "rainy": [
        "rain", "raining", "wet", "drizzle", "pour",
        "storm", "shower", "downpour", "thunder", "soaked",
    ],
    "snowy": [
        "snow", "snowing", "frost", "cold", "freezing",
        "ice", "blizzard", "winter", "flaky", "slush",
    ],
    "foggy": [
        "fog", "foggy", "mist", "haze", "hazy",
        "murky", "can't see", "poor visibility",
    ],
    "night": [
        "night", "dark", "evening", "late", "midnight",
        "after dark", "dim", "twilight", "dusk",
    ],
}


def detect_weather_from_text(text: str) -> str | None:
    """Detect weather type from user's text description.

    Args:
        text: User message in English.

    Returns:
        Weather type string (sunny/cloudy/rainy/snowy/foggy/night) or None.
    """
    text_lower = text.lower().strip()

    scores = {}
    for weather_type, keywords in WEATHER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[weather_type] = score

    if not scores:
        return None

    return max(scores, key=scores.get)
