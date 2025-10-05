#!/usr/bin/env python3
"""
Test SharePoint model download
"""
import urllib.request
import os
from pathlib import Path

def test_download():
    # Test the GitHub release URL
    url = "https://github.com/nadiajelani/wound-segmentation/releases/download/v1.0.0/simclr_unet_patch_wound.keras"
    local_path = "test_model_download.keras"
    
    print(f"Testing download from: {url}")
    print(f"Downloading to: {local_path}")
    
    try:
        # Create directory if needed
        Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) / total_size)
                print(f"\rDownloaded: {downloaded:,} / {total_size:,} bytes ({percent:.1f}%)", end="")
        
        urllib.request.urlretrieve(url, local_path, show_progress)
        
        # Check file
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            print(f"\n✅ Download successful!")
            print(f"File size: {size:,} bytes ({size / (1024*1024):.1f} MB)")
            
            # Check if it looks like a Keras file
            with open(local_path, 'rb') as f:
                header = f.read(100)
                if b'keras' in header.lower() or b'hdf5' in header.lower():
                    print("✅ File appears to be a valid Keras model")
                else:
                    print("⚠️  File doesn't appear to be a Keras model")
        else:
            print("❌ Download failed - file not created")
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")

if __name__ == "__main__":
    test_download()
