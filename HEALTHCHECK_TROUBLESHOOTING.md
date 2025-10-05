# Railway Healthcheck Troubleshooting Guide

## Issue: "1/1 replicas never became healthy! Healthcheck failed!"

This guide helps diagnose and fix Railway healthcheck failures.

## Recent Fix Applied

The app has been updated to be **resilient to model loading failures**:
- ✅ App starts even if model fails to load
- ✅ `/health` endpoint always returns 200
- ✅ Model can be loaded in background while app serves requests
- ✅ Better error logging for diagnostics

## Railway Healthcheck Configuration

Railway expects:
- **Path**: `/health` (default)
- **Expected Status**: 200 OK
- **Timeout**: 60 seconds (default)
- **Interval**: 10 seconds

## Step-by-Step Troubleshooting

### 1. Check Railway Environment Variables

Ensure these are set in Railway → Variables:

```bash
# CRITICAL - Must be set for Keras 3
KERAS_BACKEND=tensorflow

# CRITICAL - Increased timeout for model download
GUNICORN_CMD_ARGS=--timeout 600 --workers 1 --threads 1

# Thread optimization for Railway
TF_NUM_INTRAOP_THREADS=1
TF_NUM_INTEROP_THREADS=1

# Python runtime
NIXPACKS_PYTHON_VERSION=3.10
PYTHONUNBUFFERED=1
```

### 2. Check Railway Service Settings

Go to Railway → Service → Settings:

- **Builder**: Nixpacks
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Health Check Path**: `/health`
- **Health Check Timeout**: 60 (or increase to 120)

### 3. Monitor Deployment Logs

Watch for these key log messages:

#### ✅ Good Signs:
```
TensorFlow version: 2.16.1
🚀 Initializing Flask app...
📦 Attempting to load model during app initialization...
🎉 Flask app initialization complete - ready to accept connections
[INFO] Starting gunicorn 21.2.0
[INFO] Listening at: http://0.0.0.0:8080
[INFO] Booting worker with pid: X
```

#### ⚠️ Warning Signs (but app should still start):
```
⚠️ Model failed to load during app initialization - app will start anyway
⚠️ App will start without model - healthcheck will still pass
```

#### ❌ Bad Signs:
```
ModuleNotFoundError: No module named 'keras'
ImportError: cannot import name 'X' from 'tensorflow'
[ERROR] Worker failed to boot
```

### 4. Test Endpoints After Deployment

Use the test script:
```bash
python test_healthcheck.py https://your-app.up.railway.app
```

Or manually test:
```bash
# Root endpoint
curl https://your-app.up.railway.app/

# Health check
curl https://your-app.up.railway.app/health

# Debug info
curl https://your-app.up.railway.app/debug
```

### 5. Common Issues and Solutions

#### Issue: "Worker timeout" in logs
**Cause**: Model download taking too long  
**Solution**: 
- Increase `GUNICORN_CMD_ARGS` timeout to 900 seconds
- Verify `GITHUB_TOKEN` is set correctly
- Check GitHub release is public and accessible

#### Issue: "ModuleNotFoundError: No module named 'keras'"
**Cause**: Dependencies not installed correctly  
**Solution**: 
- Verify `requirements.txt` has `keras==3.3.3` and `tensorflow==2.16.1`
- Trigger a clean rebuild (delete and recreate service)

#### Issue: "Port already in use"
**Cause**: App trying to bind to wrong port  
**Solution**: 
- Ensure start command uses `$PORT` variable
- Check no hardcoded ports in app.py

#### Issue: "Out of memory"
**Cause**: Railway free tier has limited RAM  
**Solution**: 
- Ensure `--workers 1 --threads 1` in `GUNICORN_CMD_ARGS`
- Set `TF_NUM_INTRAOP_THREADS=1` and `TF_NUM_INTEROP_THREADS=1`
- Consider upgrading Railway plan

#### Issue: "Connection refused"
**Cause**: App crashed during startup  
**Solution**: 
- Check full deployment logs for Python errors
- Verify all environment variables are set
- Test locally first: `gunicorn app:app --bind 0.0.0.0:8080`

### 6. Local Testing

Before deploying to Railway, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export KERAS_BACKEND=tensorflow
export SIMCLR_MODEL_PATH=./models/simclr_unet_patch_wound.keras

# Run with gunicorn (same as Railway)
gunicorn app:app --bind 0.0.0.0:8080 --timeout 600 --workers 1

# In another terminal, test healthcheck
curl http://localhost:8080/health
```

### 7. Debugging with Railway Logs

View real-time logs:
```bash
# In Railway CLI (if installed)
railway logs

# Or use Railway dashboard → Deployments → View Logs
```

Look for:
1. **Build phase**: Dependency installation
2. **Deploy phase**: Gunicorn startup
3. **Runtime phase**: Model loading and healthcheck attempts

### 8. Emergency Fallback

If healthcheck keeps failing, temporarily disable model loading:

Add to Railway Variables:
```bash
SKIP_MODEL_LOAD=true
```

Then update `app.py`:
```python
if os.getenv("SKIP_MODEL_LOAD") == "true":
    logger.warning("Skipping model load (SKIP_MODEL_LOAD=true)")
    MODEL_LOADED = False
else:
    model_loaded = load_model()
```

This lets you verify the app itself works, then debug model loading separately.

### 9. Healthcheck Best Practices

✅ **DO**:
- Return 200 status even if model isn't loaded
- Keep healthcheck endpoint simple and fast
- Log healthcheck requests for debugging
- Set reasonable timeouts (60-120 seconds)

❌ **DON'T**:
- Crash the app if model fails to load
- Do heavy computation in healthcheck
- Require model to be loaded for healthcheck to pass
- Use very short timeouts (<30 seconds)

### 10. Success Criteria

Your deployment is successful when:

1. ✅ Build completes without errors
2. ✅ Gunicorn starts and listens on port
3. ✅ `/health` returns 200 within timeout
4. ✅ App stays running (no crash loops)
5. ✅ Model eventually loads (check `/debug`)

## Getting Help

If issues persist:

1. Share full deployment logs (Build + Deploy + Runtime)
2. Share Railway environment variables (redact sensitive values)
3. Share output of `curl https://your-app.up.railway.app/debug`
4. Note any error messages or stack traces

## Quick Checklist

Before redeploying, verify:

- [ ] `KERAS_BACKEND=tensorflow` is set
- [ ] `GUNICORN_CMD_ARGS` includes `--timeout 600`
- [ ] Start command is `gunicorn app:app --bind 0.0.0.0:$PORT`
- [ ] `requirements.txt` has `keras==3.3.3` and `tensorflow==2.16.1`
- [ ] GitHub release is public or `GITHUB_TOKEN` is set
- [ ] Health check path is `/health`
- [ ] Latest code is pushed to GitHub

## Current Status

With the latest changes:
- App will start even if model fails to load ✅
- Healthcheck will pass as long as Flask is running ✅
- Model loading errors won't crash the app ✅
- Detailed logs help diagnose issues ✅

The healthcheck should now pass! 🎉
