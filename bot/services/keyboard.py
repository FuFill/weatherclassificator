"""Inline keyboard builder for the bot."""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_weather_keyboard() -> InlineKeyboardMarkup:
    """Build the main weather selection keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="☀️ Солнечно", callback_data="weather_sunny"),
            InlineKeyboardButton(text="☁️ Облачно", callback_data="weather_cloudy"),
        ],
        [
            InlineKeyboardButton(text="🌧️ Дождь", callback_data="weather_rainy"),
            InlineKeyboardButton(text="❄️ Снег", callback_data="weather_snowy"),
        ],
        [
            InlineKeyboardButton(text="🌫️ Туман", callback_data="weather_foggy"),
            InlineKeyboardButton(text="🌙 Ночь", callback_data="weather_night"),
        ],
    ])
