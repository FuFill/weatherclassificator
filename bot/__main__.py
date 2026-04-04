"""WeatherWear Bot — entry point.

Supports two modes:
  --test <message>  Print handler response to stdout (no Telegram connection)
  (default)          Run as a Telegram bot using aiogram
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so we can import from bot/ and classifier/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message

from bot.config import settings
from bot.handlers.health import handle_health, handle_health_async
from bot.handlers.help import handle_help
from bot.handlers.photo_handler import handle_photo_async, handle_photo
from bot.handlers.start import handle_start


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
    """Route a text message to the appropriate handler.

    This is the core routing function — it maps commands to handlers
    without any Telegram dependency. Same function works in --test mode
    and in the real Telegram dispatcher.
    """
    if text == "/start":
        return handle_start()
    if text == "/help":
        return handle_help()
    if text == "/health":
        return handle_health()
    if text == "/photo" or text.startswith("/photo "):
        return handle_photo()
    return (
        "🤔 Я пока не понимаю эту команду.\n\n"
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

    @router.message(F.text == "/start")
    async def cmd_start(message: Message) -> None:
        await message.answer(handle_start())

    @router.message(F.text == "/help")
    async def cmd_help(message: Message) -> None:
        await message.answer(handle_help())

    @router.message(F.text == "/health")
    async def cmd_health(message: Message) -> None:
        result = await handle_health_async()
        await message.answer(result)

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
        await message.answer(result)

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        response = route_message(message.text)
        await message.answer(response)

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
