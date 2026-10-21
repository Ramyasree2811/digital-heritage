# ==============================================================================
# Production Dockerfile for Digital Heritage Restoration Platform
# Optimized for Cloud Deployment (Render, Cloud Run, Hugging Face, AWS)
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000

# Install essential system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install PyTorch CPU and Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import easyocr; easyocr.Reader(['hi', 'en'], gpu=False, verbose=False)"

# Copy application source code and models
COPY . /app

# Create required cache and upload directories
RUN mkdir -p static/uploads static/outputs outputs/checkpoints

# Expose service port
EXPOSE 10000

# Start production WSGI server with Gunicorn (1 worker + 4 threads to prevent OOM on 512MB RAM)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 180 app:app"]
