"""Handler for the /help command."""


def handle_help() -> str:
    """Return a list of available commands."""
    return (
        "📋 Доступные команды:\n\n"
        "/start — Начать работу с ботом\n"
        "/help — Показать эту справку\n"
        "/health — Проверить статус сервисов\n\n"
        "💡 Просто отправь фото улицы, и я подскажу, что надеть!"
    )
