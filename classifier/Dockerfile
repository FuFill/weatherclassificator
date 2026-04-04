FROM python:3.14-slim

WORKDIR /app

# Install curl for healthcheck + uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

# Copy root workspace files for dependency resolution
COPY pyproject.toml uv.lock ./
COPY classifier/pyproject.toml classifier/pyproject.toml

# Sync dependencies — no dev deps, CPU-only torch
RUN uv sync --frozen --no-dev && \
    # Replace CUDA torch with CPU-only version
    uv pip install torch==2.11.0+cpu torchvision==0.26.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the code
COPY bot/ bot/
COPY classifier/ classifier/

# Expose the service port
EXPOSE 8000

# Run the service
CMD ["uv", "run", "uvicorn", "classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
