#!/usr/bin/env python3
"""
Test script for wound detection API deployment
"""
import requests
import json
import os
import time
from typing import Dict, Any

class DeploymentTester:
    """Test the deployed wound detection API"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('API_BASE_URL', 'http://localhost:8080')
        self.test_image_path = 'test_wound.jpg'
        
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_readiness_check(self) -> bool:
        """Test readiness check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/ready", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Readiness check passed: {data}")
                return True
            else:
                print(f"❌ Readiness check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Readiness check error: {str(e)}")
            return False
    
    def test_image_upload(self) -> bool:
        """Test image upload and analysis"""
        if not os.path.exists(self.test_image_path):
            print(f"❌ Test image not found: {self.test_image_path}")
            return False
        
        try:
            with open(self.test_image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'use_medsam': 'false',
                    'generate_report': 'true'
                }
                
                print(f"📤 Uploading image to {self.base_url}/upload...")
                response = requests.post(
                    f"{self.base_url}/upload", 
                    files=files, 
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Image analysis successful!")
                    print(f"   - Severity: {result.get('analysis', {}).get('severity', 'N/A')}")
                    print(f"   - Healing Potential: {result.get('analysis', {}).get('healing_potential', 'N/A')}")
                    print(f"   - Wound Area: {result.get('analysis', {}).get('wound_area_mm2', 'N/A')} mm²")
                    print(f"   - Confidence: {result.get('analysis', {}).get('confidence', 'N/A')}")
                    return True
                else:
                    print(f"❌ Image upload failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Image upload error: {str(e)}")
            return False
    
    def test_website_access(self) -> bool:
        """Test website access"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                print("✅ Website accessible")
                return True
            else:
                print(f"❌ Website access failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Website access error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🧪 Starting deployment tests...")
        print(f"📍 Testing API at: {self.base_url}")
        print("-" * 50)
        
        results = {}
        
        # Test health check
        print("1. Testing health check...")
        results['health'] = self.test_health_check()
        
        # Test readiness check
        print("\n2. Testing readiness check...")
        results['readiness'] = self.test_readiness_check()
        
        # Test website access
        print("\n3. Testing website access...")
        results['website'] = self.test_website_access()
        
        # Test image upload (only if we have a test image)
        if os.path.exists(self.test_image_path):
            print("\n4. Testing image upload and analysis...")
            results['image_upload'] = self.test_image_upload()
        else:
            print(f"\n4. Skipping image upload test (no test image: {self.test_image_path})")
            results['image_upload'] = None
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = 0
        total = 0
        
        for test_name, result in results.items():
            if result is not None:
                total += 1
                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            else:
                print(f"⏭️  {test_name}: SKIPPED")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Deployment is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the deployment.")
        
        return results

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test wound detection API deployment')
    parser.add_argument('--url', default=None, help='API base URL to test')
    parser.add_argument('--image', default='test_wound.jpg', help='Test image path')
    
    args = parser.parse_args()
    
    tester = DeploymentTester(args.url)
    tester.test_image_path = args.image
    
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if all(result for result in results.values() if result is not None):
        exit(0)
    else:
        exit(1)

if __name__ == "__main__":
    main()