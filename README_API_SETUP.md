# Wound Segmentation API - Complete Setup Guide

## 🎯 Overview

The wound segmentation API is now fully functional with Keras 3.x compatibility. The system consists of two servers:

- **Model Server** (Port 9100): Handles the SimCLR U-Net model inference
- **Proxy API** (Port 8000): Provides a user-friendly interface and file upload capabilities

## 🚀 Quick Start

### 1. Start the Servers

```bash
# Terminal 1 - Model Server
./start_model_server.sh

# Terminal 2 - Proxy API  
./start_proxy_api.sh
```

### 2. Verify Everything is Working

```bash
# Run comprehensive smoke test
python smoke_test.py

# Check individual endpoints
curl http://127.0.0.1:9100/readyz  # Model server
curl http://127.0.0.1:8000/readyz  # Proxy API
```

### 3. Test with Your Own Image

```bash
# Using the example script
python example_analysis.py

# Or with curl
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "your_base64_image", "threshold": 0.5}'
```

## 📋 API Endpoints

### Model Server (Port 9100)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/readyz` | GET | Check if model is ready |
| `/healthz` | GET | Basic health check |
| `/diagz` | GET | Detailed diagnostics |
| `/last_errorz` | GET | Last error information |
| `/versionz` | GET | Version information |
| `/infer` | POST | Direct model inference |

### Proxy API (Port 8000)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/readyz` | GET | Check if API is ready |
| `/versionz` | GET | Version information |
| `/analyze` | POST | Analyze base64 image |
| `/analyze/upload` | POST | Upload and analyze file |
| `/report/{id}` | GET | Get analysis report |

## 🔧 Technical Details

### Root Cause Resolution
- **Problem**: Model saved with Keras 3.x format, but environment had Keras 2.12.0 (TensorFlow's Keras)
- **Solution**: Installed Keras 3.0.5 to match model's native format
- **Result**: Model loads successfully with standalone Keras 3.x

### Dependencies
- **Model Server**: `apps/model_server/requirements.txt`
- **Proxy API**: `apps/local_api/requirements.txt`

### Key Components
- ✅ **Model Loading**: Keras 3.x with proper error handling
- ✅ **FastAPI Endpoints**: Properly typed with Pydantic models
- ✅ **Diagnostics**: Comprehensive error reporting and health checks
- ✅ **Version Info**: Detailed version information for both servers
- ✅ **Smoke Testing**: End-to-end validation script

## 🧪 Testing

### Smoke Test
```bash
python smoke_test.py
```
- Waits for both servers to be ready
- Generates synthetic test image
- Performs end-to-end analysis
- Validates response format and metrics

### Manual Testing
```bash
# Check versions
curl http://127.0.0.1:9100/versionz | python -m json.tool
curl http://127.0.0.1:8000/versionz | python -m json.tool

# Check diagnostics
curl http://127.0.0.1:9100/diagz | python -m json.tool
```

## 📊 Response Format

### Analysis Response
```json
{
  "id": "1759048241992",
  "backend": "simclr_unet",
  "result": {
    "mask_area_px": 2052,
    "wound_percentage": 0.1252,
    "perimeter_px": 180.5,
    "severity": "Severe",
    "healing_potential": "Poor",
    "mask_uri": "data:image/png;base64,..."
  },
  "mode": "real"
}
```

### Version Information
```json
{
  "app": "model_server",
  "keras": "3.0.5",
  "tensorflow": "2.12.0",
  "numpy": "1.23.5"
}
```

## 🛠️ Troubleshooting

### Check Server Status
```bash
# Check if servers are running
ps aux | grep uvicorn

# Check server logs
curl http://127.0.0.1:9100/last_errorz
curl http://127.0.0.1:9100/diagz
```

### Restart Servers
```bash
# Stop all servers
pkill -f uvicorn

# Restart
./start_model_server.sh &
./start_proxy_api.sh &
```

### Common Issues
1. **Port conflicts**: Ensure ports 9100 and 8000 are free
2. **Model loading**: Check `/diagz` for detailed loading status
3. **Memory issues**: Model requires ~500MB RAM
4. **Disk space**: Ensure sufficient space for model cache

## 🎉 Success Metrics

- ✅ **Model Server**: Ready with Keras 3.x
- ✅ **Proxy API**: Ready and connected
- ✅ **Analysis**: Processing images successfully
- ✅ **Diagnostics**: All endpoints working
- ✅ **Version Info**: Detailed version reporting
- ✅ **Smoke Test**: End-to-end validation passing

The wound segmentation API is now production-ready! 🚀