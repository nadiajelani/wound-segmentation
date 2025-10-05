#!/usr/bin/env python3
"""
Test different GitHub release URL formats
"""
import urllib.request
import ssl

def test_url(url, description):
    """Test a URL and return status"""
    print(f"\n🔍 Testing: {description}")
    print(f"URL: {url}")
    
    try:
        # Use proper headers like the app does
        headers = {
            "User-Agent": "wound-segmentation/1.0 (+https://github.com/nadiajelani/wound-segmentation)",
            "Accept": "application/octet-stream"
        }
        req = urllib.request.Request(url, headers=headers)
        ctx = ssl.create_default_context()
        
        with urllib.request.urlopen(req, context=ctx) as response:
            content_length = response.headers.get('Content-Length', 'Unknown')
            print(f"✅ Status: {response.status}")
            print(f"📦 Size: {content_length} bytes")
            return True
            
    except urllib.request.HTTPError as e:
        print(f"❌ HTTP Error: {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🧪 Testing GitHub Release URL Formats")
    print("=" * 50)
    
    # Test different URL formats
    urls_to_test = [
        {
            "url": "https://github.com/nadiajelani/wound-segmentation/releases/download/v1.0.0/simclr_unet_patch_wound.keras",
            "description": "Standard GitHub release download URL"
        },
        {
            "url": "https://github.com/nadiajelani/wound-segmentation/releases/download/v1.0/simclr_unet_patch_wound.keras",
            "description": "Alternative tag format (v1.0)"
        },
        {
            "url": "https://github.com/nadiajelani/wound-segmentation/archive/refs/tags/v1.0.0.zip",
            "description": "Source code archive URL"
        }
    ]
    
    working_urls = []
    
    for test_case in urls_to_test:
        if test_url(test_case["url"], test_case["description"]):
            working_urls.append(test_case)
    
    print("\n" + "=" * 50)
    if working_urls:
        print("✅ Working URLs found:")
        for url_info in working_urls:
            print(f"  - {url_info['description']}")
            print(f"    {url_info['url']}")
    else:
        print("❌ No working URLs found")
        print("\n🔧 Possible issues:")
        print("1. Release might not be public")
        print("2. File might not be attached to the release")
        print("3. Repository might be private")
        print("4. Tag name might be different")

if __name__ == "__main__":
    main()
