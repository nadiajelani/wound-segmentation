# 📁 Railway Deployment - Files Used

## 🎯 **Core Files for Railway Deployment**

### **1. Main Application**
- **`app_clean.py`** - Main production API (SimCLR + U-Net only)
  - Flask web server
  - SimCLR model integration
  - Image upload and analysis
  - PDF report generation
  - Rate limiting and security

### **2. Model System**
- **`model_loader.py`** - SimCLR model loading system
  - Loads 527MB trained SimCLR model
  - Fallback mechanisms
  - Custom loss functions
  - Model health checks

### **3. Frontend Website**
- **`wound_whisperer.html`** - Modern responsive website
  - User interface for wound upload
  - Real-time results display
  - Mobile-friendly design
  - API integration

### **4. Deployment Configuration**
- **`Dockerfile`** - Container configuration
  - Python 3.9 slim base image
  - System dependencies
  - Environment variables
  - Health checks

- **`railway.json`** - Railway deployment config
  - Build settings
  - Health check endpoints
  - Restart policies

### **5. Dependencies**
- **`requirements_production.txt`** - Python packages
  - Flask and extensions
  - TensorFlow for AI
  - OpenCV for image processing
  - PDF generation
  - All production dependencies

### **6. AI Model**
- **`models/simclr_unet_patch_wound.keras`** - Trained SimCLR model (527MB)
  - Pre-trained wound segmentation model
  - Self-supervised learning
  - High accuracy wound detection

## 📋 **File Structure for Railway**

```
wound-segmentation/
├── app_clean.py                    # 🎯 MAIN API (SimCLR + U-Net)
├── model_loader.py                 # 🧠 Model loading system
├── wound_whisperer.html           # 🌐 Frontend website
├── Dockerfile                      # 🐳 Container config
├── railway.json                    # 🚀 Railway deployment
├── requirements_production.txt     # 📦 Dependencies
├── test_deployment.py             # 🧪 Testing script
└── models/
    └── simclr_unet_patch_wound.keras  # 🤖 AI Model (527MB)
```

## 🔧 **What Each File Does**

### **app_clean.py** (Main API)
```python
# Flask web server with SimCLR integration
- Health check endpoints (/health, /ready)
- Image upload and analysis (/upload)
- PDF report generation
- Rate limiting (10 requests/minute)
- CORS configuration
- Automatic file cleanup
```

### **model_loader.py** (AI Model System)
```python
# SimCLR model loading and management
- Loads 527MB trained model
- Fallback to building from scratch
- Custom loss functions (Dice + Binary Crossentropy)
- Model health monitoring
- Prediction interface
```

### **wound_whisperer.html** (Frontend)
```html
<!-- Modern responsive website -->
- Image upload interface
- Real-time analysis results
- Mobile-friendly design
- API communication
- Results visualization
```

### **Dockerfile** (Container)
```dockerfile
# Production container configuration
FROM python:3.9-slim
# System dependencies for OpenCV, TensorFlow
# Python packages installation
# Environment variables
# Health checks
```

### **railway.json** (Deployment Config)
```json
{
  "build": {"builder": "DOCKERFILE"},
  "deploy": {
    "startCommand": "python app_clean.py",
    "healthcheckPath": "/health"
  }
}
```

## 🚫 **Files NOT Used for Railway**

### **Old/Alternative Files (Not Used):**
- `app.py` - Old basic version
- `app_production.py` - Has MedSAM dependencies
- `app_production_simple.py` - Alternative version
- `wound_medsam.py` - MedSAM integration (removed)
- `analyze_wound.py` - Old analysis system
- `train_*.py` - Training scripts (not needed for deployment)
- `test_*.py` - Test files (except test_deployment.py)

### **Why These Are Not Used:**
- **MedSAM Dependencies**: Removed to simplify deployment
- **Training Scripts**: Not needed for production
- **Old Versions**: Replaced with clean, optimized versions
- **Test Files**: Only test_deployment.py is needed

## 🎯 **Railway Deployment Process**

### **1. Railway Reads These Files:**
1. **`Dockerfile`** - How to build the container
2. **`railway.json`** - Deployment configuration
3. **`requirements_production.txt`** - Python dependencies
4. **`app_clean.py`** - Main application to run

### **2. Railway Builds:**
1. **Downloads dependencies** from requirements_production.txt
2. **Copies all files** to container
3. **Loads SimCLR model** (527MB)
4. **Starts Flask app** on port 8080

### **3. Railway Serves:**
- **Website**: `https://your-app.railway.app/`
- **API**: `https://your-app.railway.app/upload`
- **Health**: `https://your-app.railway.app/health`

## ✅ **Summary**

**Only 6 core files are used for Railway deployment:**

1. **`app_clean.py`** - Main API (SimCLR + U-Net)
2. **`model_loader.py`** - AI model system
3. **`wound_whisperer.html`** - Frontend website
4. **`Dockerfile`** - Container configuration
5. **`railway.json`** - Railway deployment config
6. **`requirements_production.txt`** - Dependencies
7. **`models/simclr_unet_patch_wound.keras`** - AI model (527MB)

**Everything else is not used for the Railway deployment!** 🚀