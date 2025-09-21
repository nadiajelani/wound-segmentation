#!/usr/bin/env python3
"""
Test image format to debug the blue color issue.
"""

import cv2
import numpy as np
from PIL import Image
import io

# Load image the same way as the CLI
image_path = '/Users/nadiajelani/Desktop/wounds-whisperer/wound pics/ww4.png'
img_bgr = cv2.imread(image_path)
print(f"OpenCV loaded image shape: {img_bgr.shape}")
print(f"OpenCV loaded image dtype: {img_bgr.dtype}")
print(f"First few pixels (BGR): {img_bgr[0, 0, :]}")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"After BGR->RGB conversion: {img_rgb[0, 0, :]}")

# Test PIL conversion
pil_img = Image.fromarray(img_rgb)
print(f"PIL image mode: {pil_img.mode}")

# Save as PNG to test
pil_img.save('test_rgb.png')
print("Saved test_rgb.png")

# Test the analysis pipeline way
from woundseg.pipelines.preprocess import ImagePreprocessor
preprocessor = ImagePreprocessor()
analysis_img = preprocessor.load_image(image_path)
print(f"Analysis pipeline image shape: {analysis_img.shape}")
print(f"Analysis pipeline first pixels: {analysis_img[0, 0, :]}")

# Test PDF generation format
img_buffer = io.BytesIO()
pil_img.save(img_buffer, format='PNG')
img_data = img_buffer.getvalue()
print(f"PNG data length: {len(img_data)}")
print(f"PNG data starts with: {img_data[:10]}")