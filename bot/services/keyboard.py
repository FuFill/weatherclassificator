"""Inline keyboard builder and static weather advice."""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Pre-written advice for each weather type — used ONLY for keyboard button clicks
WEATHER_ADVICE = {
    "sunny": "☀️ Sunny — Light clothing, SPF, sunglasses. Cotton t-shirt and shorts, breathable sneakers. A cap for sun protection and SPF 30+ on exposed skin.",
    "cloudy": "☁️ Cloudy — Light jacket or sweater over a t-shirt. Jeans or trousers, comfortable sneakers. Bring an umbrella just in case.",
    "rainy": "🌧️ Rainy — Waterproof jacket with hood, synthetic pants, rubber boots. Big umbrella and water-resistant backpack. Phone in a waterproof case.",
    "snowy": "❄️ Snowy — Warm down jacket, thermal layers, hat covering ears, scarf, mittens. Insulated boots with non-slip soles and woolen socks.",
    "foggy": "🌫️ Foggy — Water-resistant jacket, fleece sweater, scarf. Waterproof boots. Light-colored or reflective clothing for visibility.",
    "night": "🌙 Night — Heavy jacket or down jacket, warm hat, scarf, gloves. One extra layer compared to daytime. Reflective vest for safety.",
}


def get_weather_keyboard() -> InlineKeyboardMarkup:
    """Build the main weather selection keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="☀️ Sunny", callback_data="weather_sunny"),
            InlineKeyboardButton(text="☁️ Cloudy", callback_data="weather_cloudy"),
        ],
        [
            InlineKeyboardButton(text="🌧️ Rainy", callback_data="weather_rainy"),
            InlineKeyboardButton(text="❄️ Snowy", callback_data="weather_snowy"),
        ],
        [
            InlineKeyboardButton(text="🌫️ Foggy", callback_data="weather_foggy"),
            InlineKeyboardButton(text="🌙 Night", callback_data="weather_night"),
        ],
    ])
