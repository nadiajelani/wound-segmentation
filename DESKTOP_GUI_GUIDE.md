# Desktop GUI for Wound Segmentation

## 🖥️ Overview

You now have a **simple desktop GUI application** that connects to your local SimCLR U-Net API for real wound analysis. This is a native desktop application (not a web browser) that provides an easy-to-use interface for wound segmentation.

## 🚀 How to Run the Desktop GUI

### **Option 1: Quick Start (Recommended)**
```bash
./start_desktop_gui.sh
```
This will:
- Check if API servers are running
- Start them if needed
- Launch the desktop GUI

### **Option 2: Manual Start**
```bash
# Terminal 1 - Start API servers first
./start_model_server.sh
./start_proxy_api.sh

# Terminal 2 - Start desktop GUI
python wound_gui.py
```

## 📱 Desktop GUI Features

### **Main Interface**
- **📁 File Selection**: Click "Choose Image" to select wound images
- **⚙️ Threshold Slider**: Adjust sensitivity (0.1 to 0.9)
- **🔬 Analyze Button**: Start wound analysis
- **📊 Results Panel**: View detailed analysis results

### **Supported Image Formats**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### **Real-Time Status**
- **✅ API Ready**: Green status when connected to SimCLR U-Net model
- **❌ API Error**: Red status if servers are not running
- **🔄 Analyzing**: Progress bar during analysis

## 🎯 How to Use

### **Step 1: Start the GUI**
```bash
./start_desktop_gui.sh
```

### **Step 2: Select an Image**
1. Click "📂 Choose Image" button
2. Browse and select a wound image
3. The filename will appear in the interface

### **Step 3: Adjust Settings (Optional)**
- **Threshold**: Drag the slider to adjust sensitivity
  - **Lower (0.1-0.3)**: More sensitive, detects smaller wounds
  - **Higher (0.7-0.9)**: Less sensitive, only large wounds
  - **Default (0.5)**: Balanced detection

### **Step 4: Analyze**
1. Click "🔬 Analyze Wound" button
2. Wait for processing (2-5 seconds)
3. View results in the right panel

## 📊 Understanding Results

### **Wound Metrics**
- **📏 Wound Area**: Size in pixels
- **📊 Wound Percentage**: Percentage of total image
- **📐 Perimeter**: Wound boundary length

### **Assessment**
- **⚠️ Severity**: Mild/Moderate/Severe
- **💊 Healing Potential**: Good/Fair/Poor

### **Technical Details**
- **Model**: SimCLR U-Net
- **Threshold**: Your selected sensitivity
- **Report ID**: Unique analysis identifier

## 🔧 Troubleshooting

### **GUI Won't Start**
```bash
# Check if Python tkinter is available
python -c "import tkinter; print('Tkinter OK')"

# Install tkinter if missing (Ubuntu/Debian)
sudo apt-get install python3-tk
```

### **"API Not Ready" Error**
```bash
# Check if API servers are running
curl http://127.0.0.1:9100/readyz
curl http://127.0.0.1:8000/readyz

# Start servers manually
./start_model_server.sh &
./start_proxy_api.sh &
```

### **Analysis Fails**
- Check that the image file is valid
- Ensure API servers are running
- Try a different threshold setting
- Check the error message in the results panel

## 🎉 Advantages of Desktop GUI

### **vs Web Interface**
- ✅ **Native Application**: No browser needed
- ✅ **Faster**: Direct API connection
- ✅ **Offline Ready**: Works without internet
- ✅ **Better Performance**: Native UI rendering

### **vs Command Line**
- ✅ **User Friendly**: Point-and-click interface
- ✅ **Visual Feedback**: Real-time status updates
- ✅ **Easy File Selection**: File browser integration
- ✅ **Results Display**: Formatted output

## 📋 System Requirements

### **Required**
- Python 3.8+
- tkinter (usually included with Python)
- API servers running (model server + proxy API)

### **Optional**
- PIL/Pillow for image processing
- requests for API communication

## 🚀 Quick Test

1. **Start the GUI**:
   ```bash
   ./start_desktop_gui.sh
   ```

2. **Test with a sample image**:
   - Create a simple red square on green background
   - Save as JPEG
   - Load in the GUI
   - Analyze with threshold 0.5

3. **Expected results**:
   - Should detect the red area as a wound
   - Show severity and healing potential
   - Display technical details

## 🎯 Success!

Your desktop GUI is now ready to use with:
- ✅ **Real SimCLR U-Net Model**: No dummy data
- ✅ **Native Desktop Interface**: Easy to use
- ✅ **Live API Connection**: Direct to your servers
- ✅ **Professional Results**: Detailed wound analysis

**Enjoy analyzing wounds with your desktop GUI!** 🖥️🔬