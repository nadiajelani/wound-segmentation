#!/usr/bin/env python3
"""
Test script to verify healthcheck endpoints work correctly
"""
import requests
import json
import sys

def test_endpoint(url, endpoint_name):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing {endpoint_name}: {url}")
    print('='*60)
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print(f"✅ {endpoint_name} passed")
            return True
        else:
            print(f"❌ {endpoint_name} failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {endpoint_name} failed with error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_healthcheck.py <base_url>")
        print("Example: python test_healthcheck.py https://your-app.up.railway.app")
        print("Or for local: python test_healthcheck.py http://localhost:8080")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    print(f"\n🔍 Testing Wound Segmentation API at: {base_url}")
    
    endpoints = [
        ('/', 'Root'),
        ('/health', 'Health Check'),
        ('/ready', 'Ready Check'),
        ('/debug', 'Debug Info'),
        ('/diag', 'Diagnostics'),
    ]
    
    results = []
    for path, name in endpoints:
        url = f"{base_url}{path}"
        passed = test_endpoint(url, name)
        results.append((name, passed))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
