#!/usr/bin/env bash
"""
Start the desktop GUI for wound segmentation
"""
set -euo pipefail

echo "🖥️  Starting Wound Segmentation Desktop GUI"
echo "=========================================="

# Check if API servers are running
echo "🔍 Checking API servers..."

if ! curl -s http://127.0.0.1:9100/readyz > /dev/null 2>&1; then
    echo "❌ Model server not running on port 9100"
    echo "🔄 Starting model server..."
    ./start_model_server.sh &
    sleep 5
fi

if ! curl -s http://127.0.0.1:8000/readyz > /dev/null 2>&1; then
    echo "❌ Proxy API not running on port 8000"
    echo "🔄 Starting proxy API..."
    ./start_proxy_api.sh &
    sleep 5
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
    echo "🔍 Check if servers are running:"
    echo "   curl http://127.0.0.1:9100/readyz"
    echo "   curl http://127.0.0.1:8000/readyz"
    exit 1
fi

# Start the desktop GUI
echo "🚀 Starting desktop GUI..."
python wound_gui.py