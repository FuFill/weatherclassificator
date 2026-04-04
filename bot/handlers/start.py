"""Handler for the /start command."""


def handle_start() -> str:
    """Return a welcome message for the /start command.

    This handler has no Telegram dependency — it just takes input
    and returns text. Can be called from --test mode, unit tests,
    or the Telegram dispatcher.
    """
    return (
        "👋 Привет! Я WeatherWear — твой помощник по выбору одежды.\n\n"
        "Отправь мне фото улицы, и я подскажу, что надеть, "
        "учитывая погоду!\n\n"
        "Используй /help, чтобы узнать больше."
    )
