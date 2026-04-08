"""Handler for the /help command."""


def handle_help() -> str:
    """Return a list of available commands."""
    return (
        "📋 Available commands:\n\n"
        "/start — Get started with the bot\n"
        "/help — Show this help message\n"
        "/health — Check service status\n"
        "/weather — Show clothing advice for all weather types\n\n"
        "💡 Just send a photo of the street and I'll tell you what to wear!"
    )
