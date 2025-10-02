# Use Python 3.10 slim image for TensorFlow-friendly stack
FROM python:3.10-slim

# System libs for OpenCV/Pillow (no tkinter, GUI stuff)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.server.txt /app/

# Avoid TF oneDNN perf warnings in some hosts
ENV TF_ENABLE_ONEDNN_OPTS=0 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.server.txt

# Copy the application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/uploads /app/reports /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app_clean.py
ENV FLASK_ENV=production
ENV MPLBACKEND=Agg

# Expose Railway PORT
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run with gunicorn for production
CMD gunicorn -w 1 -b 0.0.0.0:$PORT app_clean:app --timeout 180