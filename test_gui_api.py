#!/usr/bin/env python3
"""
Test that the GUI can connect to the local API
"""
import requests
import base64
import io
from PIL import Image

def test_gui_api_connection():
    """Test that the GUI API endpoints are working"""
    
    print("🧪 Testing GUI API Connection")
    print("=" * 40)
    
    # Test 1: Check if APIs are running
    try:
        model_response = requests.get("http://127.0.0.1:9100/readyz", timeout=5)
        proxy_response = requests.get("http://127.0.0.1:8000/readyz", timeout=5)
        
        print(f"✅ Model Server: {model_response.json()}")
        print(f"✅ Proxy API: {proxy_response.json()}")
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False
    
    # Test 2: Create a test image and analyze it
    print("\n🖼️  Testing image analysis...")
    
    # Create a simple test image
    img = Image.new('RGB', (128, 128), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    # Test the analyze endpoint (same as GUI uses)
    try:
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            json={"image_b64": img_b64, "threshold": 0.5},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"📊 Wound area: {result['result']['mask_area_px']} pixels")
            print(f"📊 Wound percentage: {result['result']['wound_percentage']:.4f}")
            print(f"📊 Severity: {result['result']['severity']}")
            print(f"📊 Healing potential: {result['result']['healing_potential']}")
            print(f"🆔 Report ID: {result['id']}")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_api_connection()
    
    if success:
        print("\n🎉 GUI API Connection Test PASSED!")
        print("🌐 You can now use the GUI at: http://localhost:3000")
        print("📱 The GUI will connect to your local API for real wound analysis")
    else:
        print("\n❌ GUI API Connection Test FAILED!")
        print("🔧 Make sure both API servers are running:")
        print("   ./start_model_server.sh")
        print("   ./start_proxy_api.sh")