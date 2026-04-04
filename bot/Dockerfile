FROM python:3.14-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Install bot-only deps
RUN uv pip install --system \
    "aiogram>=3.20" "httpx>=0.28" "pydantic-settings>=2.12"

# Copy bot code
COPY bot/ bot/

# Run the bot
CMD ["python", "-m", "bot"]
