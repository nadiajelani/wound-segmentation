#!/usr/bin/env bash
"""
Start the complete wound segmentation system:
1. Model server (port 9100)
2. Proxy API (port 8000) 
3. GUI server (port 3000)
"""
set -euo pipefail

echo "🚀 Starting Wound Segmentation System..."
echo "========================================"

# Check if API servers are already running
if curl -s http://127.0.0.1:9100/readyz > /dev/null 2>&1; then
    echo "✅ Model server already running on port 9100"
else
    echo "🔄 Starting model server..."
    ./start_model_server.sh &
    sleep 3
fi

if curl -s http://127.0.0.1:8000/readyz > /dev/null 2>&1; then
    echo "✅ Proxy API already running on port 8000"
else
    echo "🔄 Starting proxy API..."
    ./start_proxy_api.sh &
    sleep 3
fi

# Wait for APIs to be ready
echo "⏳ Waiting for APIs to be ready..."
timeout=30
while [ $timeout -gt 0 ]; do
    if curl -s http://127.0.0.1:9100/readyz | grep -q '"ready":true' && \
       curl -s http://127.0.0.1:8000/readyz | grep -q '"ready":true'; then
        echo "✅ All APIs ready!"
        break
    fi
    sleep 1
    timeout=$((timeout - 1))
done

if [ $timeout -eq 0 ]; then
    echo "❌ APIs not ready after 30 seconds"
    echo "🔍 Check logs:"
    echo "   Model server: curl http://127.0.0.1:9100/diagz"
    echo "   Proxy API: curl http://127.0.0.1:8000/readyz"
    exit 1
fi

# Start GUI server
echo "🌐 Starting GUI server..."
python start_gui.py &

echo ""
echo "🎉 Wound Segmentation System is running!"
echo "========================================"
echo "📱 GUI: http://localhost:3000"
echo "🔗 API: http://127.0.0.1:8000"
echo "🧠 Model: http://127.0.0.1:9100"
echo ""
echo "⏹️  Press Ctrl+C to stop all services"
echo ""

# Wait for user to stop
wait