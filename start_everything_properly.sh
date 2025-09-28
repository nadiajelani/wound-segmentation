#!/usr/bin/env bash
"""
Start everything properly for the desktop GUI
"""
set -euo pipefail

echo "🚀 Starting Wound Segmentation System"
echo "====================================="

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f uvicorn 2>/dev/null || true
pkill -f "python.*wound_gui" 2>/dev/null || true
sleep 2

# Start Model Server
echo "🔄 Starting Model Server (port 9100)..."
source /Users/nadiajelani/projects/wound-segmentation/api_env_clean/bin/activate
export UNET_WEIGHTS_PATH="/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
uvicorn apps.model_server.server:create_app --factory --host 127.0.0.1 --port 9100 --workers 1 --log-level info &
MODEL_PID=$!

# Wait for model server to start
echo "⏳ Waiting for model server to start..."
sleep 5

# Check if model server is ready
if curl -s http://127.0.0.1:9100/readyz | grep -q '"ready":true'; then
    echo "✅ Model server ready!"
else
    echo "❌ Model server failed to start"
    kill $MODEL_PID 2>/dev/null || true
    exit 1
fi

# Start Proxy API
echo "🔄 Starting Proxy API (port 8000)..."
export MODEL_SERVER_URL="http://127.0.0.1:9100"
python -m uvicorn apps.local_api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info &
PROXY_PID=$!

# Wait for proxy API to start
echo "⏳ Waiting for proxy API to start..."
sleep 3

# Check if proxy API is ready
if curl -s http://127.0.0.1:8000/readyz | grep -q '"ready":true'; then
    echo "✅ Proxy API ready!"
else
    echo "❌ Proxy API failed to start"
    kill $MODEL_PID $PROXY_PID 2>/dev/null || true
    exit 1
fi

# Test the connection
echo "🧪 Testing API connection..."
python test_gui_api.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 All systems ready!"
    echo "==================="
    echo "✅ Model Server: http://127.0.0.1:9100"
    echo "✅ Proxy API: http://127.0.0.1:8000"
    echo ""
    echo "🖥️  Starting Desktop GUI..."
    echo "   (Close the GUI window to stop all services)"
    echo ""
    
    # Start the desktop GUI
    python wound_gui.py
    
    # Cleanup when GUI closes
    echo ""
    echo "🧹 Cleaning up..."
    kill $MODEL_PID $PROXY_PID 2>/dev/null || true
    echo "✅ All services stopped"
else
    echo "❌ API test failed"
    kill $MODEL_PID $PROXY_PID 2>/dev/null || true
    exit 1
fi