#!/bin/bash

# Wound Segmentation API Testing Script
# Replace <your-railway-url> with your actual Railway URL

RAILWAY_URL="https://<your-railway-url>.up.railway.app"
TEST_IMAGE="uploads/wound_20250928_163620.jpg"

echo "🧪 Testing Wound Segmentation API at: $RAILWAY_URL"
echo "=================================================="

# Test 1: Health Check
echo "📋 Test 1: Health Check"
echo "----------------------"
curl -s "$RAILWAY_URL/health" | jq .
echo ""

# Test 2: Ready Check
echo "📋 Test 2: Ready Check"
echo "----------------------"
curl -s "$RAILWAY_URL/ready" | jq .
echo ""

# Test 3: Root Endpoint
echo "📋 Test 3: Root Endpoint"
echo "------------------------"
curl -s "$RAILWAY_URL/" | jq .
echo ""

# Test 4: Image Analysis (if test image exists)
if [ -f "$TEST_IMAGE" ]; then
    echo "📋 Test 4: Image Analysis"
    echo "------------------------"
    echo "Testing with image: $TEST_IMAGE"
    curl -s -X POST -F "image=@$TEST_IMAGE" "$RAILWAY_URL/analyze" | jq .
else
    echo "📋 Test 4: Image Analysis"
    echo "------------------------"
    echo "❌ Test image not found: $TEST_IMAGE"
    echo "Available images in uploads/:"
    ls -la uploads/*.jpg 2>/dev/null | head -3
fi

echo ""
echo "🎉 Testing Complete!"
echo "===================="
