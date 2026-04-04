"""Weather descriptions and advice for inline keyboard buttons."""

# Weather type → (emoji, description, advice)
WEATHER_INFO = {
    "sunny": {
        "emoji": "☀️",
        "name": "Солнечно",
        "advice": "Лёгкая одежда, SPF, солнцезащитные очки.",
    },
    "cloudy": {
        "emoji": "☁️",
        "name": "Облачно",
        "advice": "Лёгкая куртка или кофта. Возьмите зонт.",
    },
    "rainy": {
        "emoji": "🌧️",
        "name": "Дождь",
        "advice": "Водонепроницаемая куртка, зонт, резиновые сапоги.",
    },
    "snowy": {
        "emoji": "❄️",
        "name": "Снег",
        "advice": "Тёплая куртка, шапка, шарф, перчатки.",
    },
    "foggy": {
        "emoji": "🌫️",
        "name": "Туман",
        "advice": "Одевайтесь теплее, видимость низкая. Зонт не помешает.",
    },
    "night": {
        "emoji": "🌙",
        "name": "Ночь",
        "advice": "Тёплые слои, куртка потеплее. Шапка и перчатки.",
    },
}
