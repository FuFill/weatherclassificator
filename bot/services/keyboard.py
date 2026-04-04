"""Inline keyboard builder and callback handlers."""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import CallbackQuery


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


async def handle_weather_callback(callback: CallbackQuery) -> str:
    """Handle weather button click and return formatted response.

    Returns:
        Text to send as a new message (not edit).
    """
    weather_type = callback.data.replace("weather_", "")

    responses = {
        "sunny": [
            "Ясная погода, отлично! Футболка, шорты или лёгкое платье — самое то. Не забудь солнцезащитные очки и крем от загара.",
            "Солнечно и тепло — одевайся полегче. Кепка и очки спасут от яркого света, а лёгкая одежда не даст перегреться.",
            "Отличный день для прогулки! Худи не понадобится — хватит футболки и шорт. SPF обязателен.",
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

    import random

    options = responses.get(weather_type, ["Одевайся по погоде и не забудь зонт!"])
    return random.choice(options)
