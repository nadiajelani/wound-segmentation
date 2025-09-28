#!/usr/bin/env python3
"""
Example: How to use the wound segmentation API
"""
import requests
import base64
import json
from PIL import Image
import io

def analyze_wound_image(image_path: str, threshold: float = 0.5):
    """
    Analyze a wound image using the API
    
    Args:
        image_path: Path to the image file
        threshold: Segmentation threshold (0.0-1.0)
    
    Returns:
        dict: Analysis results
    """
    
    # Convert image to base64
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    # Prepare request
    payload = {
        "image_b64": img_b64,
        "threshold": threshold
    }
    
    # Send request to API
    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Analysis successful!")
        print(f"📊 Wound area: {result['result']['mask_area_px']} pixels")
        print(f"📊 Wound percentage: {result['result']['wound_percentage']:.4f}")
        print(f"📊 Severity: {result['result']['severity']}")
        print(f"📊 Healing potential: {result['result']['healing_potential']}")
        print(f"🆔 Report ID: {result['id']}")
        return result
    else:
        print(f"❌ Analysis failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_with_sample_image():
    """Test with a sample image"""
    print("🧪 Testing with sample image...")
    
    # Create a simple test image
    img = Image.new('RGB', (128, 128), color='red')
    img.save('test_wound.jpg')
    
    # Analyze it
    result = analyze_wound_image('test_wound.jpg')
    
    if result:
        # Save the mask if you want to see it
        mask_b64 = result['result']['mask_uri'].split(',')[1]
        mask_bytes = base64.b64decode(mask_b64)
        with open('wound_mask.png', 'wb') as f:
            f.write(mask_bytes)
        print("💾 Mask saved as 'wound_mask.png'")

if __name__ == "__main__":
    print("🔬 Wound Segmentation API Example")
    print("=" * 40)
    
    # Check if API is ready
    try:
        response = requests.get("http://127.0.0.1:8000/readyz", timeout=5)
        if response.json().get('ready'):
            print("✅ API is ready!")
            test_with_sample_image()
        else:
            print("❌ API is not ready")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure both servers are running:")
        print("  - Model server on port 9100")
        print("  - Proxy API on port 8000")