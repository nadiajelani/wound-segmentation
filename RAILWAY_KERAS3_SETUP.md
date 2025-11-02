# Railway Deployment Guide - Keras 3 Upgrade

## Overview
This guide covers the complete setup for deploying the wound segmentation API on Railway with Keras 3 and TensorFlow 2.16.

## 1. Requirements Update

The `requirements.txt` has been updated to use Keras 3:

```txt
# --- Web ---
flask==3.0.3
flask-cors==4.0.1
gunicorn==21.2.0

# --- ML stack (Keras 3 + TF 2.16) ---
tensorflow==2.16.1
keras==3.3.3
numpy==1.26.4
typing-extensions>=4.6.0
protobuf==4.25.3
h5py==3.11.0

# --- Image / utils ---
opencv-python-headless==4.10.0.84
pillow==10.4.0
```

## 2. Railway Environment Variables

Go to your Railway project → **Variables** tab and add/update these variables:

### Required Variables:

```bash
# Keras Backend Configuration
KERAS_BACKEND=tensorflow

# Gunicorn Configuration (increased timeout for model download)
GUNICORN_CMD_ARGS=--timeout 600 --workers 1 --threads 1

# TensorFlow Thread Configuration (optimize for Railway free tier)
TF_NUM_INTRAOP_THREADS=1
TF_NUM_INTEROP_THREADS=1

# Model Configuration
SIMCLR_MODEL_PATH=/app/models/simclr_unet_patch_wound.keras
SIMCLR_MODEL_URL=https://github.com/nadiajelani/wound-segmentation/releases/download/v1.0.0/simclr_unet_patch_wound.keras
SIMCLR_MODEL_TAG=v1.0.0
SIMCLR_MODEL_REPO=nadiajelani/wound-segmentation
SIMCLR_MODEL_ASSET=simclr_unet_patch_wound.keras

# GitHub Token (for private repos or rate limit issues)
GITHUB_TOKEN=your_github_token_here

# File Integrity Checks (optional but recommended)
SIMCLR_MODEL_MIN_BYTES=400000000
SIMCLR_MODEL_SHA256=a9c15bbd4f5a5967660044907f8a2faad485ff83811a3420231d718bb75306ec

# Python Runtime
NIXPACKS_PYTHON_VERSION=3.10

# General Configuration
PYTHONUNBUFFERED=1
SECRET_KEY=railway-wound-seg-2024-secret
CORS_ORIGINS=*
```

### Variable Explanations:

- **KERAS_BACKEND**: Forces Keras 3 to use TensorFlow as the backend
- **GUNICORN_CMD_ARGS**: 
  - `--timeout 600`: 10-minute timeout for large model downloads
  - `--workers 1`: Single worker to minimize memory usage
  - `--threads 1`: Single thread per worker
- **TF_NUM_INTRAOP_THREADS** & **TF_NUM_INTEROP_THREADS**: Limit TensorFlow parallelism to reduce memory footprint
- **SIMCLR_MODEL_***: Configuration for model download and validation
- **GITHUB_TOKEN**: Required if your repository is private or to avoid GitHub API rate limits
- **SIMCLR_MODEL_MIN_BYTES**: Minimum file size to detect partial downloads (400MB)
- **SIMCLR_MODEL_SHA256**: Expected SHA256 hash for integrity verification

## 3. Railway Service Configuration

### Builder Settings:
- **Builder**: Nixpacks (default)
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Health Check Path**: `/health`

### Runtime Configuration:
- **Python Version**: 3.10 (via `runtime.txt` or `NIXPACKS_PYTHON_VERSION`)
- **Region**: Choose closest to your users (e.g., `us-west1`, `europe-west4`)

## 4. Deployment Process

1. **Commit and push changes**:
   ```bash
   git add requirements.txt app.py
   git commit -m "Upgrade to Keras 3 and TensorFlow 2.16"
   git push origin main-fixed
   ```

2. **Configure Railway variables** (as listed above)

3. **Trigger deployment** in Railway dashboard

4. **Monitor build logs** for:
   ```
   ✅ pip install -r requirements.txt
   ✅ Successfully installed tensorflow-2.16.1 keras-3.3.3
   ```

5. **Monitor startup logs** for:
   ```
   TensorFlow version: 2.16.1
   [MODEL] Using Keras version: 3.3.3
   📦 Loading wound segmentation model...
   [MODEL] Model file exists at /app/models/simclr_unet_patch_wound.keras, size: 527123456 bytes
   [MODEL] Input shape: (None, 128, 128, 3)
   [MODEL] Output shape: (None, 128, 128, 1)
   ✅ Model loaded successfully
   ```

## 5. Verification Steps

After deployment completes:

### Test Health Endpoint:
```bash
curl https://your-app.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2025-10-05T16:30:00.000000"
}
```

### Test Ready Endpoint:
```bash
curl https://your-app.up.railway.app/ready
```

Expected response:
```json
{
  "ready": true,
  "model_loaded": true,
  "message": "Service ready to process requests",
  "timestamp": "2025-10-05T16:30:00.000000"
}
```

### Test Debug Endpoint:
```bash
curl https://your-app.up.railway.app/debug
```

Expected response:
```json
{
  "model_loaded": true,
  "model_exists": true,
  "model_path": "/app/models/simclr_unet_patch_wound.keras",
  "model_path_exists": true,
  "model_input_shape": [null, 128, 128, 3],
  "model_output_shape": [null, 128, 128, 1],
  "timestamp": "2025-10-05T16:30:00.000000"
}
```

## 6. Troubleshooting

### Issue: Model download times out
**Solution**: Increase `GUNICORN_CMD_ARGS` timeout to 900 seconds (15 minutes)

### Issue: Out of memory errors
**Solution**: 
- Ensure `--workers 1 --threads 1` in `GUNICORN_CMD_ARGS`
- Set `TF_NUM_INTRAOP_THREADS=1` and `TF_NUM_INTEROP_THREADS=1`

### Issue: GitHub download fails with 404
**Solution**: 
- Verify the GitHub release exists and is public
- If private, ensure `GITHUB_TOKEN` is set with correct permissions
- Check `SIMCLR_MODEL_URL` points to the correct release

### Issue: Model deserialization error
**Solution**: 
- Verify model was saved with Keras 3
- Check logs for Keras version mismatch
- Ensure `KERAS_BACKEND=tensorflow` is set

### Issue: Partial file download
**Solution**: 
- The app automatically detects and removes files < 400MB
- Set `SIMCLR_MODEL_MIN_BYTES` to expected model size
- Add `SIMCLR_MODEL_SHA256` for integrity verification

## 7. Performance Optimization

### Memory Usage:
- Single worker configuration: ~1.5GB RAM
- Model size: ~527MB
- TensorFlow overhead: ~500MB
- Flask app: ~100MB

### Startup Time:
- Build: ~2-3 minutes
- Model download (if needed): ~2-5 minutes (depending on network)
- Model loading: ~30-60 seconds
- Total: ~5-10 minutes for cold start

### Request Latency:
- Health check: <50ms
- Image analysis: ~500ms-2s (depending on image size)

## 8. Cost Optimization

Railway free tier limits:
- 500 hours/month execution time
- 512MB RAM (upgrade recommended to 1GB for stability)
- 1GB disk space

Recommendations:
- Use model download instead of including in repo (saves build time)
- Enable health checks to prevent unnecessary restarts
- Set appropriate timeout values to avoid premature worker kills

## 9. Security Best Practices

- Never commit `GITHUB_TOKEN` to repository
- Use Railway's secret variables for sensitive data
- Set `CORS_ORIGINS` to specific domains in production
- Rotate `SECRET_KEY` regularly
- Keep dependencies updated for security patches

## 10. Next Steps

After successful deployment:
1. Test the `/analyze` endpoint with sample wound images
2. Monitor Railway metrics for memory and CPU usage
3. Set up custom domain (optional)
4. Configure alerts for deployment failures
5. Document API endpoints for your users

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Review the `/debug` endpoint output
3. Verify all environment variables are set correctly
4. Ensure GitHub release is accessible
5. Check Railway service status page
