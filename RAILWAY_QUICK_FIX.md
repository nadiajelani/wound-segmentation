# 🚀 Railway Quick Fix - Python 3.10 + Server-Only Stack

## ❌ **Problem Fixed:**
- Railway was defaulting to Python 3.13
- Trying to pip install tkinter (not available on Linux)
- Many packages don't have wheels for Python 3.13

## ✅ **Solution Applied:**

### **1. Railway Variables to Set:**
```bash
NIXPACKS_PYTHON_VERSION=3.10
PYTHONUNBUFFERED=1
```

### **2. Updated requirements.txt:**
- **Removed tkinter**: Not pip-installable on Linux
- **Removed GUI deps**: Server-only stack
- **Python 3.10 compatible**: TF 2.12.0 + NumPy 1.23.5
- **Headless OpenCV**: opencv-python-headless

### **3. Start Command:**
```bash
gunicorn -w 1 -b 0.0.0.0:$PORT app_clean:app --timeout 180
```

## 🚀 **Deploy Steps:**

### **Step 1: Set Railway Variables**
In Railway dashboard → Variables:
```bash
NIXPACKS_PYTHON_VERSION=3.10
PYTHONUNBUFFERED=1
CORS_ORIGINS=https://your-app-name.railway.app
SIMCLR_MODEL_PATH=models/simclr_unet_patch_wound.keras
```

### **Step 2: Set Start Command**
In Railway Settings → Deploy → Start Command:
```bash
gunicorn -w 1 -b 0.0.0.0:$PORT app_clean:app --timeout 180
```

### **Step 3: Redeploy**
- Commit and push the updated requirements.txt
- Railway will rebuild with Python 3.10
- No more tkinter errors!

## 🧪 **Test After Deployment:**

### **Health Checks:**
```bash
curl -s https://your-app-name.railway.app/health
curl -s https://your-app-name.railway.app/ready
curl -s https://your-app-name.railway.app/diag
```

### **Expected Results:**
- **Health**: `{"status": "healthy", "models_loaded": true}`
- **Ready**: `{"status": "ready"}`
- **Diag**: Shows model info and parameters

## ✅ **This Fix Ensures:**
- ✅ **Python 3.10**: Railway uses correct version
- ✅ **No tkinter**: Server-only dependencies
- ✅ **Compatible stack**: TF 2.12.0 + NumPy 1.23.5
- ✅ **Clean build**: No more pip install errors
- ✅ **Production ready**: Gunicorn + proper port binding

**Your Railway deployment will now succeed!** 🚀