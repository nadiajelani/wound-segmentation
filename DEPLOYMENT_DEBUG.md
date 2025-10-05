# Deployment Debugging - What to Check Now

## The healthcheck is still failing. Let's diagnose the exact cause.

## Step 1: Get the Full Error Logs

Go to Railway → Your Service → Deployments → Click on the failed deployment → View Logs

### What to look for:

#### A. Build Phase Logs
Look for errors during `pip install`:
```
ERROR: Could not find a version that satisfies the requirement...
ERROR: No matching distribution found for...
```

#### B. Deploy Phase Logs
Look for Gunicorn startup errors:
```
[ERROR] Exception in worker process
ModuleNotFoundError: No module named 'X'
ImportError: cannot import name 'X'
```

#### C. Runtime Phase Logs
Look for application errors:
```
❌ Model load error: ...
Traceback (most recent call last):
```

## Step 2: Check Railway Configuration

### A. Environment Variables
Go to Railway → Variables and verify these are set:

**CRITICAL - Must have these:**
```
KERAS_BACKEND=tensorflow
GUNICORN_CMD_ARGS=--timeout 600 --workers 1 --threads 1
```

**Important:**
```
NIXPACKS_PYTHON_VERSION=3.10
PYTHONUNBUFFERED=1
TF_NUM_INTRAOP_THREADS=1
TF_NUM_INTEROP_THREADS=1
```

### B. Service Settings
Go to Railway → Settings → Deploy:

- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Health Check Path**: `/health`
- **Health Check Timeout**: 60 (try increasing to 120)

## Step 3: Common Failure Scenarios

### Scenario 1: Build Fails (Dependencies)
**Symptoms**: Build phase shows errors, never reaches deploy
**Logs show**: `ERROR: Could not find a version...`
**Fix**: 
- Check `requirements.txt` is committed
- Verify Python version compatibility
- Try removing version pins temporarily

### Scenario 2: Import Errors
**Symptoms**: Gunicorn starts but worker crashes immediately
**Logs show**: `ModuleNotFoundError` or `ImportError`
**Fix**:
- Ensure all dependencies in `requirements.txt`
- Check for typos in import statements
- Verify Keras 3 compatibility

### Scenario 3: Port Binding Issues
**Symptoms**: App starts but healthcheck times out
**Logs show**: Gunicorn listening on wrong port
**Fix**:
- Ensure start command uses `$PORT` variable
- Check no hardcoded ports in code

### Scenario 4: Memory/Timeout Issues
**Symptoms**: App starts but gets killed during model loading
**Logs show**: Worker timeout or OOM killed
**Fix**:
- Increase `GUNICORN_CMD_ARGS` timeout
- Reduce workers/threads to 1
- Set TensorFlow thread limits

### Scenario 5: Keras Backend Issues
**Symptoms**: Import errors related to Keras
**Logs show**: `Could not import keras.src...`
**Fix**:
- Ensure `KERAS_BACKEND=tensorflow` is set
- Verify TensorFlow 2.16.1 and Keras 3.3.3 installed

## Step 4: Quick Fixes to Try

### Fix 1: Simplify Gunicorn Command
In Railway → Settings → Deploy → Start Command:
```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1
```

### Fix 2: Increase Healthcheck Timeout
In Railway → Settings → Deploy:
- Health Check Timeout: 120 seconds

### Fix 3: Add Explicit Port
In Railway → Variables:
```
PORT=8080
```

### Fix 4: Force Python 3.10
Ensure `runtime.txt` exists with:
```
python-3.10.14
```

## Step 5: Emergency Minimal Test

Let's verify the basic app works by temporarily simplifying it.

Create a minimal test version:

```python
# test_minimal.py
from flask import Flask, jsonify
import sys

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "python": sys.version}), 200

@app.route('/')
def root():
    return jsonify({"message": "minimal test"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Then temporarily change Railway start command to:
```
gunicorn test_minimal:app --bind 0.0.0.0:$PORT
```

If this works, the issue is in the main app. If not, it's a Railway configuration issue.

## Step 6: What I Need to Help You

Please share:

1. **Full deployment logs** (Build + Deploy + Runtime)
   - Copy from Railway logs panel
   - Include the last 50-100 lines

2. **Railway environment variables** (redact sensitive values)
   - List all variables currently set

3. **Service settings**
   - Start command
   - Health check configuration
   - Builder type

4. **Error message**
   - The exact error from Railway
   - Any stack traces

## Step 7: Alternative Approaches

If Railway keeps failing, we can try:

### Option A: Use Docker Builder Instead
Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1
```

Then in Railway → Settings → Builder: Switch to Dockerfile

### Option B: Split Model Loading
Don't load model at startup, load on first request:
```python
@app.before_request
def load_model_lazy():
    global MODEL, MODEL_LOADED
    if not MODEL_LOADED:
        load_model()
```

### Option C: Use Different Platform
Try deploying to:
- Render.com (similar to Railway)
- Fly.io (more control)
- Google Cloud Run (more resources)

## Next Steps

1. **Share the logs** - I need to see the actual error
2. **Verify variables** - Double-check all environment variables are set
3. **Try quick fixes** - Start with the simplest fixes above
4. **Test locally** - Ensure it works on your machine first

## Local Testing Command

Before deploying, test locally:
```bash
# Set environment
export KERAS_BACKEND=tensorflow
export PORT=8080

# Install dependencies
pip install -r requirements.txt

# Run with gunicorn (exactly as Railway does)
gunicorn app:app --bind 0.0.0.0:8080 --timeout 600 --workers 1

# In another terminal
curl http://localhost:8080/health
```

If this works locally but fails on Railway, it's a Railway-specific issue.

---

**Please share the deployment logs and I'll provide a targeted fix!** 🎯
