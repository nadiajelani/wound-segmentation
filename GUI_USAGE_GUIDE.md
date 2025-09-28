# Wound Segmentation GUI - Usage Guide

## 🎯 Overview

The Wound Whisperer GUI is now connected to your local API and can perform **real wound analysis** using the SimCLR U-Net model. No more dummy data!

## 🚀 Quick Start

### Option 1: Start Everything at Once
```bash
./start_everything.sh
```
This will start:
- Model server (port 9100)
- Proxy API (port 8000) 
- GUI server (port 3000)
- Open your browser automatically

### Option 2: Start Manually
```bash
# Terminal 1 - Model Server
./start_model_server.sh

# Terminal 2 - Proxy API
./start_proxy_api.sh

# Terminal 3 - GUI Server
python start_gui.py
```

## 🌐 Access the GUI

Once everything is running, open your browser to:
**http://localhost:3000**

## 📱 How to Use the GUI

### 1. **Upload an Image**
- Click the upload area or drag & drop an image
- Supported formats: JPG, PNG, JPEG
- Recommended size: 128x128 to 512x512 pixels

### 2. **Adjust Settings**
- **Threshold**: Segmentation sensitivity (0.1 - 0.9)
  - Lower = more sensitive (detects smaller wounds)
  - Higher = less sensitive (only large wounds)
- **Default**: 0.5 (balanced)

### 3. **Analyze**
- Click "Analyze Wound" button
- Wait for processing (usually 2-5 seconds)
- View results in real-time

### 4. **View Results**
The GUI will show:
- **Wound Area**: Size in pixels
- **Wound Percentage**: Percentage of total image
- **Severity Assessment**: Mild/Moderate/Severe
- **Healing Potential**: Good/Fair/Poor
- **Visual Mask**: Overlay showing detected wound area

## 🔧 Features

### Real-Time Analysis
- ✅ **Real Model**: Uses your SimCLR U-Net model
- ✅ **Live Processing**: No dummy data
- ✅ **Accurate Results**: Based on actual wound segmentation
- ✅ **Visual Feedback**: See the detected wound area

### Interactive Interface
- 🖼️ **Drag & Drop**: Easy image upload
- 🎛️ **Adjustable Threshold**: Fine-tune sensitivity
- 📊 **Detailed Metrics**: Comprehensive wound analysis
- 🖼️ **Mask Visualization**: See exactly what was detected

### Health Monitoring
- 🔍 **API Status**: Real-time connection status
- 📊 **Performance Metrics**: Processing time and accuracy
- 🔗 **Direct Links**: Access to API diagnostics

## 🧪 Testing the GUI

### Test with Sample Images
1. **Red Square Test**: Upload a simple red square on green background
2. **Real Wound Images**: Use actual wound photos (if available)
3. **Different Thresholds**: Try 0.3, 0.5, 0.7 to see sensitivity changes

### Expected Results
- **Red Square**: Should detect as "Severe" wound
- **Real Wounds**: Should show appropriate severity based on size
- **Healthy Skin**: Should show "Mild" or no detection

## 🔍 Troubleshooting

### GUI Not Loading
```bash
# Check if GUI server is running
curl http://localhost:3000

# Restart GUI
python start_gui.py
```

### Analysis Failing
```bash
# Check API status
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:9100/readyz

# Check detailed diagnostics
curl http://127.0.0.1:9100/diagz | python -m json.tool
```

### Slow Performance
- Use smaller images (128x128 to 256x256)
- Check system resources
- Ensure model server is ready

## 📊 Understanding Results

### Wound Area (pixels)
- **0-100**: Very small wound
- **100-1000**: Small wound  
- **1000-5000**: Medium wound
- **5000+**: Large wound

### Wound Percentage
- **0-1%**: Minimal wound
- **1-5%**: Small wound
- **5-15%**: Medium wound
- **15%+**: Large wound

### Severity Assessment
- **Mild**: < 1% of image area
- **Moderate**: 1-5% of image area
- **Severe**: > 5% of image area

### Healing Potential
- **Good**: < 1% wound area
- **Fair**: 1-5% wound area
- **Poor**: > 5% wound area

## 🎉 Success!

Your GUI is now fully functional with:
- ✅ **Real Model Inference**: No more dummy data
- ✅ **Live API Connection**: Direct to your local servers
- ✅ **Accurate Analysis**: Based on SimCLR U-Net model
- ✅ **Interactive Interface**: Easy to use and understand

**Enjoy analyzing wounds with your AI-powered GUI!** 🚀