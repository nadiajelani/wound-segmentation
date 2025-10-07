#!/usr/bin/env python3
"""
Quick test to check what the Railway API returns
"""

import requests
import json
import sys
from PIL import Image
import io

def test_api(url):
    """Test the API and print what it returns"""
    
    # Create a simple red test image
    img = Image.new('RGB', (128, 128), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    print(f"Testing: {url}/analyze")
    print("Sending test image...")
    
    try:
        files = {'image': ('test.png', img_bytes, 'image/png')}
        response = requests.post(f"{url}/analyze", files=files, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n" + "="*60)
            print("API RESPONSE STRUCTURE:")
            print("="*60)
            
            for key in data.keys():
                if key.endswith('_image'):
                    # Image data
                    img_data = data[key]
                    if img_data:
                        print(f"\n✅ {key}:")
                        print(f"   - Type: {type(img_data)}")
                        print(f"   - Length: {len(img_data)} characters")
                        print(f"   - Starts with: {img_data[:50]}")
                        print(f"   - Is base64 PNG: {img_data.startswith('data:image/png;base64,')}")
                    else:
                        print(f"\n❌ {key}: EMPTY or MISSING")
                else:
                    # Other data
                    print(f"\n✅ {key}: {type(data[key])}")
                    if isinstance(data[key], dict):
                        print(f"   Keys: {list(data[key].keys())}")
            
            # Check specific images
            print("\n" + "="*60)
            print("IMAGE CHECK:")
            print("="*60)
            
            images = {
                'mask_image': 'Segmentation Mask',
                'heatmap_image': 'Confidence Heatmap',
                'overlay_image': 'Overlay'
            }
            
            for img_key, img_name in images.items():
                if img_key in data and data[img_key]:
                    status = "✅ PRESENT"
                    size = len(data[img_key])
                else:
                    status = "❌ MISSING"
                    size = 0
                
                print(f"{status} {img_name}: {size} chars")
            
            # Check report structures
            print("\n" + "="*60)
            print("REPORT CHECK:")
            print("="*60)
            
            if 'healing_stage' in data:
                hs = data['healing_stage']
                print(f"✅ Healing Stage: {hs.get('stage')}")
                print(f"   Confidence: {hs.get('confidence')}")
                print(f"   Has recommendations: {len(hs.get('recommendations', []))} items")
            else:
                print("❌ healing_stage: MISSING")
            
            if 'doctor_report' in data:
                dr = data['doctor_report']
                print(f"✅ Doctor Report ID: {dr.get('report_id')}")
                print(f"   Has clinical_findings: {'clinical_findings' in dr}")
                print(f"   Has follow_up: {'follow_up' in dr}")
            else:
                print("❌ doctor_report: MISSING")
            
            if 'metrics' in data:
                m = data['metrics']
                print(f"✅ Metrics:")
                print(f"   Area: {m.get('area_pixels')} pixels ({m.get('area_percentage')}%)")
                print(f"   Severity: {m.get('severity')}")
            else:
                print("❌ metrics: MISSING")
            
            # Summary
            print("\n" + "="*60)
            print("SUMMARY:")
            print("="*60)
            
            required_keys = ['success', 'mask_image', 'heatmap_image', 'overlay_image', 
                           'metrics', 'healing_stage', 'doctor_report']
            
            missing = []
            for key in required_keys:
                if key not in data or (isinstance(data[key], str) and not data[key]):
                    missing.append(key)
            
            if not missing:
                print("✅ ALL REQUIRED DATA PRESENT!")
                print("\nThe API is returning:")
                print("  • Segmentation mask ✅")
                print("  • Confidence heatmap ✅")
                print("  • Overlay image ✅")
                print("  • Clinical metrics ✅")
                print("  • Healing stage assessment ✅")
                print("  • Doctor's report ✅")
            else:
                print(f"❌ MISSING: {', '.join(missing)}")
                print("\nThe API is NOT returning all required data!")
        else:
            print(f"\n❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_api_response.py <RAILWAY_URL>")
        print("Example: python test_api_response.py https://your-app.up.railway.app")
        sys.exit(1)
    
    url = sys.argv[1].rstrip('/')
    test_api(url)
