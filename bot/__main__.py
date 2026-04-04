"""WeatherWear Bot — entry point.

Supports two modes:
  --test <message>  Print handler response to stdout (no Telegram connection)
  (default)          Run as a Telegram bot using aiogram
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so we can import from bot/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.config import settings
from bot.handlers.health import handle_health, handle_health_async
from bot.handlers.help import handle_help
from bot.handlers.photo_handler import handle_photo_async, handle_photo
from bot.handlers.start import handle_start
from bot.handlers.weather_info import WEATHER_INFO
from bot.services.keyboard import get_weather_keyboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeatherWear Telegram Bot")
    parser.add_argument(
        "--test",
        type=str,
        metavar="MESSAGE",
        help="Run in test mode — print response to stdout and exit",
    )
    return parser.parse_args()


def route_message(text: str) -> str:
    """Route a text message to the appropriate handler."""
    if text == "/start":
        return handle_start()
    if text == "/help":
        return handle_help()
    if text == "/health":
        return handle_health()
    if text == "/weather":
        lines = ["Все типы погоды и рекомендации:"]
        for wt, info in WEATHER_INFO.items():
            lines.append(f"{info['emoji']} {info['name']} — {info['advice']}")
        return "\n\n".join(lines)
    if text == "/photo" or text.startswith("/photo "):
        return handle_photo()
    return (
        "Я пока не понимаю эту команду.\n\n"
        "Используй /start для начала или отправь фото улицы, "
        "и я подскажу, что надеть!"
    )


def run_test_mode(message: str) -> None:
    """Run in test mode — route message and print response."""
    response = route_message(message)
    print(response)


async def run_telegram_bot() -> None:
    """Run the Telegram bot with aiogram dispatcher."""
    if not settings.bot_token:
        print("Error: BOT_TOKEN is not set. Set it in .env or environment.")
        sys.exit(1)

    bot_instance = Bot(token=settings.bot_token)
    dp = Dispatcher()
    router = Router()

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        text = handle_start()
        await message.answer(text, reply_markup=get_weather_keyboard())

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(handle_help())

    @router.message(Command("health"))
    async def cmd_health(message: Message) -> None:
        result = await handle_health_async()
        await message.answer(result)

    @router.message(Command("weather"))
    async def cmd_weather(message: Message) -> None:
        """Show all weather types and their advice."""
        lines = ["Все типы погоды и рекомендации:"]
        for wt, info in WEATHER_INFO.items():
            lines.append(f"{info['emoji']} {info['name']} — {info['advice']}")
        await message.answer("\n\n".join(lines))

    @router.callback_query(F.data.startswith("weather_"))
    async def on_weather_callback(callback: CallbackQuery) -> None:
        """Handle inline keyboard weather button clicks.

        Sends a short info message + main keyboard.
        For real advice — send a photo of the street.
        """
        weather_type = callback.data.replace("weather_", "")

        if weather_type in WEATHER_INFO:
            info = WEATHER_INFO[weather_type]
            await callback.message.answer(
                f"{info['emoji']} {info['name']}\n\n"
                f"Отправь мне фото улицы с такой погодой, "
                f"и я подберу что надеть!",
                reply_markup=get_weather_keyboard(),
            )
        else:
            await callback.answer("Неизвестный тип погоды", show_alert=True)

        await callback.answer()

    @router.message(F.photo)
    async def on_photo(message: Message) -> None:
        """Download photo from Telegram and process it."""
        # Get the highest resolution photo
        photo = message.photo[-1]
        file_info = await message.bot.get_file(photo.file_id)
        file_bytes = await message.bot.download_file(file_info.file_path)

        # Get any caption text
        user_message = message.caption or ""

        result = await handle_photo_async(file_bytes, user_message)
        await message.answer(
            result,
            reply_markup=get_weather_keyboard(),
        )

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        response = route_message(message.text)
        await message.answer(response, reply_markup=get_weather_keyboard())

    dp.include_router(router)

    print("WeatherWear Bot started. Polling for updates...")
    await dp.start_polling(bot_instance)


def main() -> None:
    args = parse_args()

    if args.test:
        run_test_mode(args.test)
    else:
        asyncio.run(run_telegram_bot())


if __name__ == "__main__":
    main()
