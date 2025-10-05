#!/usr/bin/env python3
"""
Comprehensive API Test Script
Tests all endpoints of the wound segmentation API
"""
import requests
import json
import sys
import base64
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_health(api_url):
    """Test health endpoint"""
    print_header("Testing Health Check")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        print(f"✅ Response:")
        print(json.dumps(data, indent=2))
        
        if data.get('model_loaded'):
            print("\n🎉 Model is loaded and ready!")
            return True
        else:
            print("\n⚠️  Model not loaded yet - wait a moment and retry")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ready(api_url):
    """Test ready endpoint"""
    print_header("Testing Ready Check")
    try:
        response = requests.get(f"{api_url}/ready", timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        print(f"✅ Response:")
        print(json.dumps(data, indent=2))
        return data.get('ready', False)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_debug(api_url):
    """Test debug endpoint"""
    print_header("Testing Debug Info")
    try:
        response = requests.get(f"{api_url}/debug", timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        print(f"✅ Response:")
        print(json.dumps(data, indent=2))
        
        # Check critical info
        if data.get('model_loaded'):
            print(f"\n✅ Model Shape: {data.get('model_input_shape')} → {data.get('model_output_shape')}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root(api_url):
    """Test root endpoint"""
    print_header("Testing Root Endpoint")
    try:
        response = requests.get(f"{api_url}/", timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        print(f"✅ Response:")
        print(json.dumps(data, indent=2))
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze_with_image(api_url, image_path):
    """Test analyze endpoint with actual image"""
    print_header(f"Testing Wound Analysis: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        print(f"📤 Uploading image...")
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{api_url}/analyze", files=files, timeout=30)
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print(f"\n🎉 Analysis Successful!")
                print(f"\n📊 Metrics:")
                metrics = data.get('metrics', {})
                print(f"  • Area (pixels): {metrics.get('area_pixels')}")
                print(f"  • Area (%): {metrics.get('area_percentage')}%")
                print(f"  • Perimeter: {metrics.get('perimeter')} pixels")
                print(f"  • Severity: {metrics.get('severity')}")
                
                # Save mask
                mask_data = data.get('mask_image', '')
                if mask_data.startswith('data:image/png;base64,'):
                    mask_bytes = base64.b64decode(mask_data.split(',')[1])
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    mask_filename = f"wound_mask_{timestamp}.png"
                    with open(mask_filename, 'wb') as f:
                        f.write(mask_bytes)
                    print(f"\n💾 Mask saved to: {mask_filename}")
                
                # Save full result
                result_filename = f"wound_result_{timestamp}.json"
                with open(result_filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"💾 Full result saved to: {result_filename}")
                
                return True
            else:
                print(f"❌ Analysis failed: {data}")
                return False
        else:
            print(f"❌ Error Response:")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image():
    """Create a simple test image if none provided"""
    print_header("Creating Test Image")
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple 256x256 image with a circular "wound"
        img = Image.new('RGB', (256, 256), color=(255, 200, 180))  # Skin-like color
        draw = ImageDraw.Draw(img)
        
        # Draw a "wound" (dark red circle)
        draw.ellipse([80, 80, 180, 180], fill=(139, 0, 0))
        
        filename = "test_wound_image.jpg"
        img.save(filename)
        print(f"✅ Created test image: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("  🏥 WOUND SEGMENTATION API - COMPREHENSIVE TEST")
    print("="*70)
    
    # Get API URL
    if len(sys.argv) < 2:
        print("\n❌ Usage: python test_api.py <api_url> [image_path]")
        print("\nExamples:")
        print("  python test_api.py https://your-app.up.railway.app")
        print("  python test_api.py https://your-app.up.railway.app wound.jpg")
        print("  python test_api.py http://localhost:8080 wound.jpg")
        sys.exit(1)
    
    api_url = sys.argv[1].rstrip('/')
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\n🎯 Testing API at: {api_url}")
    
    # Test all endpoints
    results = {}
    
    # 1. Root
    results['root'] = test_root(api_url)
    
    # 2. Health
    results['health'] = test_health(api_url)
    
    # 3. Ready
    results['ready'] = test_ready(api_url)
    
    # 4. Debug
    results['debug'] = test_debug(api_url)
    
    # 5. Analyze (if image provided or create test)
    if image_path or True:  # Always try to test
        if not image_path:
            print("\n💡 No image provided, creating a test image...")
            image_path = create_test_image()
        
        if image_path:
            results['analyze'] = test_analyze_with_image(api_url, image_path)
        else:
            results['analyze'] = None
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name.upper()}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {test_name.upper()}: FAILED")
            failed += 1
        else:
            print(f"⏭️  {test_name.upper()}: SKIPPED")
            skipped += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 ALL TESTS PASSED! Your API is working perfectly! 🎉")
        return 0
    elif passed > 0:
        print(f"\n⚠️  Some tests passed, but {failed} failed. Check the logs above.")
        return 1
    else:
        print("\n❌ All tests failed. Your API may not be running or accessible.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

