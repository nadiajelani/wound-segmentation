# Railway Gunicorn Timeout Configuration

## Problem
The large model download (400MB+) can take several minutes, but Railway's default Gunicorn timeout is too short, causing the application to crash during startup.

## Solution
Set the `GUNICORN_CMD_ARGS` environment variable in Railway to increase the timeout.

## Railway Environment Variables to Add/Update

Go to your Railway project → Variables tab and add/update these:

```
GUNICORN_CMD_ARGS=--timeout 600 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100
```

This sets:
- `--timeout 600`: 10 minutes timeout for worker processes
- `--keep-alive 2`: Keep connections alive for 2 seconds
- `--max-requests 1000`: Restart workers after 1000 requests
- `--max-requests-jitter 100`: Add randomness to restart timing

## Alternative: Railway.json Configuration

If you prefer to use a configuration file, create `railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "command": "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

## Verification

After setting the timeout:

1. Redeploy your service
2. Check the logs - you should see Gunicorn starting with the new timeout settings
3. The model download should complete without timing out
4. Test the `/health` endpoint to verify the model loaded successfully

## Expected Log Output

```
[INFO] Starting gunicorn 21.2.0
[INFO] Listening at: http://0.0.0.0:8080 (1)
[INFO] Using worker: sync
[INFO] Booting worker with pid: X
[MODEL] Downloading model from https://github.com/... -> /app/models/simclr_unet_patch_wound.keras
[MODEL] Download completed successfully
✅ Model loaded successfully
```
