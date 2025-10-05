# Railway-optimized Dockerfile for wound segmentation API
# Python 3.10 + TensorFlow 2.12 compatible

FROM python:3.10-slim

# Keep Python output unbuffered and avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (OpenCV + build chain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Python deps (upgrade pip/setuptools/wheel first)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source (includes model + static files)
COPY . .

# Create necessary directories
RUN mkdir -p uploads reports logs models

# Keras/TensorFlow runtime knobs
ENV KERAS_BACKEND=tensorflow \
    TF_USE_LEGACY_KERAS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

# Expose port (Railway provides $PORT)
EXPOSE 8080

# Healthcheck (hits your Flask /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD wget -qO- http://127.0.0.1:${PORT:-8080}/health || exit 1

# Start the app with Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} \
  --workers 1 --threads 2 --timeout 180 app_free:app
