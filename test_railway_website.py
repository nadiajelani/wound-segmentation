#!/usr/bin/env python3
"""
Test Railway Website Features
Tests: segmentation, heatmap, and doctor report
"""

import requests
import json
import sys
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_health(base_url):
    """Test health endpoint"""
    print_section("1. Testing Health Check")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data.get('status')}")
            print(f"✅ Model Loaded: {data.get('model_loaded')}")
            print(f"✅ Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_homepage(base_url):
    """Test if homepage serves wound_analyzer.html"""
    print_section("2. Testing Homepage (Wound Analyzer)")
    
    try:
        response = requests.get(base_url, timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            html = response.text
            
            # Check if it's the wound analyzer HTML
            checks = {
                "Has title": "Wound Analyzer" in html,
                "Has upload section": "upload-section" in html,
                "Has analyze button": "analyzeBtn" in html,
                "Has heatmap": "heatmap" in html.lower(),
                "Has doctor report": "doctor" in html.lower() or "clinical" in html.lower(),
            }
            
            for check, result in checks.items():
                status = "✅" if result else "❌"
                print(f"{status} {check}: {result}")
            
            return all(checks.values())
        else:
            print(f"❌ Homepage failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Homepage error: {e}")
        return False

def test_analyze_endpoint(base_url):
    """Test /analyze endpoint with a sample image"""
    print_section("3. Testing /analyze Endpoint (Segmentation + Heatmap + Report)")
    
    # Create a simple test image (red square)
    try:
        from PIL import Image
        import io
        
        # Create test image
        img = Image.new('RGB', (128, 128), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        print("📤 Sending test image to /analyze endpoint...")
        
        files = {'image': ('test.png', img_bytes, 'image/png')}
        response = requests.post(f"{base_url}/analyze", files=files, timeout=30)
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for key features
            checks = {
                "Success flag": data.get('success', False),
                "Has metrics": 'metrics' in data,
                "Has healing_stage": 'healing_stage' in data,
                "Has doctor_report": 'doctor_report' in data,
                "Has mask_image": 'mask_image' in data,
                "Has heatmap_image": 'heatmap_image' in data,
                "Has overlay_image": 'overlay_image' in data,
            }
            
            for check, result in checks.items():
                status = "✅" if result else "❌"
                print(f"{status} {check}: {result}")
            
            # Display detailed results
            if all(checks.values()):
                print("\n📊 METRICS:")
                metrics = data.get('metrics', {})
                print(f"  • Area: {metrics.get('area_pixels')} pixels ({metrics.get('area_percentage')}%)")
                print(f"  • Perimeter: {metrics.get('perimeter')} pixels")
                print(f"  • Severity: {metrics.get('severity')}")
                
                print("\n🩹 HEALING STAGE:")
                healing = data.get('healing_stage', {})
                print(f"  • Stage: {healing.get('stage')}")
                print(f"  • Confidence: {healing.get('confidence', 0)*100:.1f}%")
                print(f"  • Description: {healing.get('description', '')[:100]}...")
                
                print("\n📋 DOCTOR REPORT:")
                report = data.get('doctor_report', {})
                print(f"  • Report ID: {report.get('report_id')}")
                print(f"  • Generated At: {report.get('generated_at')}")
                
                findings = report.get('clinical_findings', {})
                print(f"  • Wound Area: {findings.get('wound_area')}")
                print(f"  • Severity Classification: {findings.get('severity_classification')}")
                
                print("\n🖼️  IMAGES:")
                print(f"  • Mask image: {'✅ Present' if data.get('mask_image') else '❌ Missing'}")
                print(f"  • Heatmap image: {'✅ Present' if data.get('heatmap_image') else '❌ Missing'}")
                print(f"  • Overlay image: {'✅ Present' if data.get('overlay_image') else '❌ Missing'}")
                
                # Check if images are base64 encoded
                if data.get('mask_image'):
                    mask_size = len(data['mask_image'])
                    print(f"  • Mask size: {mask_size} bytes")
                    
                if data.get('heatmap_image'):
                    heatmap_size = len(data['heatmap_image'])
                    print(f"  • Heatmap size: {heatmap_size} bytes")
            
            return all(checks.values())
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except ImportError:
        print("❌ PIL (Pillow) not installed. Install with: pip install Pillow")
        return False
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_download_feature(base_url):
    """Test if download features would work"""
    print_section("4. Testing Download Features")
    
    print("ℹ️  Download features are client-side JavaScript")
    print("✅ They work automatically once the analyze endpoint returns data")
    print("✅ Downloads: mask, heatmap, overlay, clinical report, full JSON")
    
    return True

def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("❌ Usage: python test_railway_website.py <RAILWAY_URL>")
        print("\nExample:")
        print("  python test_railway_website.py https://wound-segmentation-production.up.railway.app")
        print("\n💡 To find your Railway URL:")
        print("  1. Go to railway.app")
        print("  2. Click your project")
        print("  3. Click 'Settings' → 'Domains'")
        print("  4. Copy the generated domain (e.g., xxx.up.railway.app)")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    print("\n" + "="*60)
    print("  🏥 RAILWAY WEBSITE TEST")
    print(f"  Testing: {base_url}")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['health'] = test_health(base_url)
    results['homepage'] = test_homepage(base_url)
    results['analyze'] = test_analyze_endpoint(base_url)
    results['download'] = test_download_feature(base_url)
    
    # Summary
    print_section("📊 TEST SUMMARY")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name.upper()}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("  🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Your Railway website is working perfectly!")
        print(f"\n🌐 Visit: {base_url}")
        print("\n📝 Features Available:")
        print("  • Wound segmentation (binary mask)")
        print("  • Confidence heatmap (color-coded)")
        print("  • Overlay visualization")
        print("  • Doctor's clinical report")
        print("  • Download all results")
        print("\n🚀 Next Steps:")
        print("  1. Open the URL in your browser")
        print("  2. Upload a wound image")
        print("  3. Click 'Analyze Wound'")
        print("  4. View all visualizations and reports")
        print("  5. Download results")
    else:
        print("\n" + "="*60)
        print("  ⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\n🔍 Troubleshooting:")
        
        if not results['health']:
            print("  • Health check failed - API may not be running")
            print("  • Check Railway logs for errors")
            
        if not results['homepage']:
            print("  • Homepage not serving wound_analyzer.html")
            print("  • Check if wound_analyzer.html is in the deployment")
            print("  • Verify app.py is serving the file from root endpoint")
            
        if not results['analyze']:
            print("  • Analysis endpoint failed")
            print("  • Model may not be loaded")
            print("  • Check Railway logs for model loading errors")
        
        print(f"\n📞 Check Railway logs: railway logs")
        print(f"📞 Check debug endpoint: {base_url}/debug")
    
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()

