"""Weather descriptions and advice for inline keyboard buttons."""

# Weather type → (emoji, description, advice)
WEATHER_INFO = {
    "sunny": {
        "emoji": "☀️",
        "name": "Sunny",
        "advice": "Light clothing, SPF, sunglasses.",
    },
    "cloudy": {
        "emoji": "☁️",
        "name": "Cloudy",
        "advice": "Light jacket or sweater. Bring an umbrella.",
    },
    "rainy": {
        "emoji": "🌧️",
        "name": "Rainy",
        "advice": "Waterproof jacket, umbrella, rubber boots.",
    },
    "snowy": {
        "emoji": "❄️",
        "name": "Snowy",
        "advice": "Warm jacket, hat, scarf, gloves.",
    },
    "foggy": {
        "emoji": "🌫️",
        "name": "Foggy",
        "advice": "Dress warmer, visibility is low. An umbrella helps.",
    },
    "night": {
        "emoji": "🌙",
        "name": "Night",
        "advice": "Warm layers, heavier jacket. Hat and gloves.",
    },
}
