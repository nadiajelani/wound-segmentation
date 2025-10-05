# 🧪 Wound Segmentation API Testing Guide

## 🎯 **Quick Setup**

1. **Get your Railway URL:**
   - Go to Railway Dashboard → Your Project → Your Service → Settings → Domains
   - Copy the URL (e.g., `https://your-app-name.up.railway.app`)

2. **Replace `<your-railway-url>` in the commands below with your actual URL**

---

## 📋 **Test 1: Health Check**

```bash
curl -s https://<your-railway-url>.up.railway.app/health | jq
```

**✅ Expected Success:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-XX T XX:XX:XX",
  "version": "1.0.0"
}
```

**❌ If `model_loaded: false`:**
- Check Railway Variables: `SIMCLR_MODEL_PATH = /app/models/simclr_unet_patch_wound.keras`
- Check Railway Logs for model loading errors

---

## 📋 **Test 2: Ready Check**

```bash
curl -s https://<your-railway-url>.up.railway.app/ready | jq
```

**✅ Expected Success:**
```json
{
  "ready": true,
  "model_loaded": true,
  "message": "Service ready to process requests"
}
```

---

## 📋 **Test 3: Root Endpoint**

```bash
curl -s https://<your-railway-url>.up.railway.app/ | jq
```

**✅ Expected Success:**
```json
{
  "message": "Wound Segmentation API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "ready": "/ready",
    "analyze": "/analyze"
  }
}
```

---

## 📋 **Test 4: Image Analysis**

```bash
# Test with a wound image
curl -s -X POST \
  -F "image=@uploads/wound_20250928_163620.jpg" \
  https://<your-railway-url>.up.railway.app/analyze | jq
```

**✅ Expected Success:**
```json
{
  "success": true,
  "analysis": {
    "wound_area_pixels": 1234,
    "wound_area_percentage": 5.67,
    "confidence": 0.89
  },
  "mask_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "processing_time": 2.34
}
```

---

## 🚨 **Troubleshooting**

### **If Health Check Fails:**
1. Check Railway Logs for startup errors
2. Verify all environment variables are set
3. Check if model file exists at `/app/models/simclr_unet_patch_wound.keras`

### **If Model Not Loading:**
1. **Railway Variables to check:**
   ```
   NIXPACKS_PYTHON_VERSION = 3.10
   PYTHONUNBUFFERED = 1
   SECRET_KEY = railway-wound-seg-2024-secret
   SIMCLR_MODEL_PATH = /app/models/simclr_unet_patch_wound.keras
   ```

2. **Check Railway Logs for:**
   - "Loading wound segmentation model..."
   - "Model loaded successfully (tf.keras)"
   - Any TensorFlow/Keras errors

### **If Analysis Fails:**
1. Check image file size (must be ≤ 8MB)
2. Check image format (JPG, PNG supported)
3. Check Railway Logs for processing errors

---

## 🎉 **Success Indicators**

✅ **API is healthy:** `/health` returns `"status": "healthy"`  
✅ **Model loaded:** `"model_loaded": true`  
✅ **Ready for requests:** `/ready` returns `"ready": true`  
✅ **Can process images:** `/analyze` returns analysis results  
✅ **No errors in Railway Logs**  

---

## 📊 **Available Test Images**

Your repository has these test images:
- `uploads/wound_20250928_163620.jpg`
- `uploads/wound_20250928_154105.jpg`
- `uploads/wound_20250928_155839.jpg`
- `uploads/wound_20250928_163435.jpg`
- `uploads/wound_20250928_164004.jpg`

---

## 🔧 **Automated Testing**

Use the provided test script:
```bash
# Edit test_api.sh and replace <your-railway-url> with your actual URL
./test_api.sh
```
