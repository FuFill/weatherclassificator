FROM python:3.14-slim

WORKDIR /app

# Install curl for healthcheck + uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

# Install CPU-only torch first (no CUDA, ~200MB instead of ~2GB)
RUN uv pip install --system \
    "torch>=2.5" "torchvision>=0.20" \
    --index-url https://download.pytorch.org/whl/cpu

# Install classifier deps
RUN uv pip install --system \
    "fastapi>=0.115" "uvicorn[standard]>=0.34" \
    "transformers>=4.48" "pillow>=11.0" "python-multipart>=0.0.18" \
    "onnxruntime>=1.20" "numpy>=2.0"

# Export model to ONNX (runs once at build time)
RUN mkdir -p /app/classifier/models && \
    uv pip install --system "torch>=2.5" "onnx>=1.18" && \
    python -c "from classifier.export_onnx import main; main()" 2>/dev/null || \
    echo "ONNX export failed at build time, will use PyTorch fallback"

# Copy classifier code
COPY classifier/ classifier/

# Expose the service port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
