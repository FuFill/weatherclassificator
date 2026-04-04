FROM python:3.14-slim

WORKDIR /app

# Install curl for healthcheck + uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

# Copy root workspace files for dependency resolution
COPY pyproject.toml uv.lock ./
COPY classifier/pyproject.toml classifier/pyproject.toml

# Sync dependencies using uv (workspace includes bot + classifier)
RUN uv sync --frozen

# Copy the rest of the code
COPY bot/ bot/
COPY classifier/ classifier/

# Expose the service port
EXPOSE 8000

# Run the service
CMD ["uv", "run", "uvicorn", "classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
