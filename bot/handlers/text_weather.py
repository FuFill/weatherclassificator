"""Text-based weather intent detection.

When users describe weather in text instead of sending a photo,
this module detects the weather type and provides clothing advice.
"""

import re

# Keywords that indicate each weather type in Russian and English
WEATHER_KEYWORDS = {
    "sunny": [
        "солнц", "sun", "ясн", "жарк", "тепл", "свет",
        "bright", "clear sky", "без облак",
    ],
    "cloudy": [
        "облачн", "пасмурн", "туч", "cloud", "overcast",
        "хмур", "серо", "grey sky",
    ],
    "rainy": [
        "дожд", "rain", "ливн", "мокр", "капел",
        "wet", "drizzle", "слякот", "гроза", "storm",
    ],
    "snowy": [
        "снег", "snow", "мороз", "холод", "зим",
        "frost", "лед", "ice", "метель", "blizzard",
    ],
    "foggy": [
        "туман", "fog", "дымк", "мгл", "haze",
        "мутн", "не видн", "плох вид",
    ],
    "night": [
        "ноч", "night", "темн", "dark", "вечер",
        "evening", "поздно", "late", "сумерк",
    ],
}


def detect_weather_from_text(text: str) -> str | None:
    """Detect weather type from user's text description.

    Args:
        text: User message in Russian or English.

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

    # Return the type with highest keyword matches
    return max(scores, key=scores.get)
