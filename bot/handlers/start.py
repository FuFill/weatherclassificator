"""Handler for the /start command."""


def handle_start() -> str:
    """Return a welcome message for the /start command."""
    return (
        "👋 Hi! I'm WeatherWear — your personal clothing advisor.\n\n"
        "Send me a photo of the street, and I'll tell you what to wear "
        "based on the weather!\n\n"
        "Use /help to learn more."
    )
