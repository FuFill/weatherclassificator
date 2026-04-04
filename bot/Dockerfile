FROM python:3.14-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy root workspace files for dependency resolution
COPY pyproject.toml uv.lock ./
COPY bot/pyproject.toml bot/pyproject.toml

# Sync dependencies using uv (workspace includes bot + classifier)
RUN uv sync --frozen

# Copy the rest of the code
COPY bot/ bot/
COPY classifier/ classifier/

# Run the bot
CMD ["uv", "run", "-m", "bot"]
