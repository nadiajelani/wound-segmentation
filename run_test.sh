#!/bin/bash
# Quick test script for your wound segmentation API

echo "🏥 Wound Segmentation API - Quick Test"
echo "======================================"
echo ""

# Check if API URL is provided
if [ -z "$1" ]; then
    echo "❌ Please provide your Railway API URL"
    echo ""
    echo "Usage: ./run_test.sh <api_url> [image_path]"
    echo ""
    echo "Examples:"
    echo "  ./run_test.sh https://your-app.up.railway.app"
    echo "  ./run_test.sh https://your-app.up.railway.app uploads/wound_20250928_163620.jpg"
    echo ""
    echo "To find your Railway URL:"
    echo "  1. Go to Railway dashboard"
    echo "  2. Click on your service"
    echo "  3. Go to Settings → Networking"
    echo "  4. Copy the public domain URL"
    echo ""
    exit 1
fi

API_URL="$1"
IMAGE_PATH="$2"

echo "🎯 Testing API at: $API_URL"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check if requests library is available
python3 -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing requests library..."
    pip3 install requests pillow
fi

# Run the test
if [ -z "$IMAGE_PATH" ]; then
    echo "💡 No image provided, will use first available image or create test image"
    # Find first wound image
    FIRST_IMAGE=$(find uploads -name "*.jpg" -o -name "*.png" | head -1)
    if [ -n "$FIRST_IMAGE" ]; then
        echo "📸 Using image: $FIRST_IMAGE"
        python3 test_api.py "$API_URL" "$FIRST_IMAGE"
    else
        echo "📸 No images found, will create test image"
        python3 test_api.py "$API_URL"
    fi
else
    echo "📸 Using image: $IMAGE_PATH"
    python3 test_api.py "$API_URL" "$IMAGE_PATH"
fi

