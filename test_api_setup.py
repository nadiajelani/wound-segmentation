#!/usr/bin/env python3
"""
Test script to verify the API setup
"""
import requests
import json
import base64
from PIL import Image
import io
import numpy as np

def test_model_server():
    """Test the model server directly"""
    print("Testing model server...")
    try:
        response = requests.get("http://127.0.0.1:9100/readyz", timeout=5)
        print(f"Model server readyz: {response.json()}")
        return response.json().get("ready", False)
    except Exception as e:
        print(f"Model server error: {e}")
        return False

def test_proxy_api():
    """Test the proxy API"""
    print("Testing proxy API...")
    try:
        response = requests.get("http://127.0.0.1:8000/readyz", timeout=5)
        print(f"Proxy API readyz: {response.json()}")
        return response.json().get("ready", False)
    except Exception as e:
        print(f"Proxy API error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple 128x128 RGB image
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    return img_b64

def test_analysis():
    """Test the analysis endpoint"""
    print("Testing analysis endpoint...")
    
    # Check if proxy API is ready
    if not test_proxy_api():
        print("Proxy API not ready, skipping analysis test")
        return False
    
    # Create test image
    img_b64 = create_test_image()
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            json={"image_b64": img_b64},
            timeout=30
        )
        print(f"Analysis response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Analysis error: {response.text}")
            return False
    except Exception as e:
        print(f"Analysis error: {e}")
        return False

if __name__ == "__main__":
    print("=== API Setup Test ===")
    
    model_ready = test_model_server()
    proxy_ready = test_proxy_api()
    
    print(f"\nModel server ready: {model_ready}")
    print(f"Proxy API ready: {proxy_ready}")
    
    if proxy_ready:
        print("\nTesting analysis...")
        test_analysis()
    else:
        print("\nSkipping analysis test - API not ready")