# ==============================================================================
# Production Dockerfile for Digital Heritage Restoration Platform
# Compatible with Hugging Face Spaces (16GB RAM) & Cloud Services (Render, AWS)
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install essential system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (Standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install PyTorch CPU and Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import easyocr; easyocr.Reader(['hi', 'en'], gpu=False, verbose=False)" || true

# Copy application source code and models
COPY . /app

# Create required cache/output directories and set ownership
RUN mkdir -p static/uploads static/outputs outputs/checkpoints && \
    chown -R user:user /app

USER user

# Expose service port
EXPOSE 7860

# Start production WSGI server with Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 180 app:app"]
