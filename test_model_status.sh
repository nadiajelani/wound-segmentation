#!/bin/bash

# Model Status Testing Script
# Replace <your-railway-url> with your actual Railway URL

RAILWAY_URL="https://<your-railway-url>.up.railway.app"

echo "🔍 Testing Model Status on Railway"
echo "=================================="
echo "URL: $RAILWAY_URL"
echo ""

# Test 1: Debug endpoint
echo "📋 Test 1: Debug Information"
echo "----------------------------"
curl -s "$RAILWAY_URL/debug" | jq .
echo ""

# Test 2: Health endpoint
echo "📋 Test 2: Health Check"
echo "-----------------------"
curl -s "$RAILWAY_URL/health" | jq .
echo ""

# Test 3: Ready endpoint
echo "📋 Test 3: Ready Check"
echo "----------------------"
curl -s "$RAILWAY_URL/ready" | jq .
echo ""

# Test 4: Image analysis (if test image exists)
if [ -f "uploads/wound_20250928_163620.jpg" ]; then
    echo "📋 Test 4: Image Analysis"
    echo "------------------------"
    echo "Testing with: uploads/wound_20250928_163620.jpg"
    curl -s -X POST -F "image=@uploads/wound_20250928_163620.jpg" "$RAILWAY_URL/analyze" | jq .
else
    echo "📋 Test 4: Image Analysis"
    echo "------------------------"
    echo "❌ Test image not found: uploads/wound_20250928_163620.jpg"
fi

echo ""
echo "🎯 Summary:"
echo "==========="
echo "1. Check 'model_loaded' field in debug/health/ready responses"
echo "2. If model_loaded: false, check Railway logs for model loading errors"
echo "3. If image analysis returns same results for different images, model is using fallback"
echo "4. Real model results should vary based on actual wound content"
