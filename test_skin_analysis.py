#!/usr/bin/env python3
"""
Test Skin Color Analysis - Command line version to debug issues
"""
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import os
import sys

def test_image_loading():
    """Test if we can load images properly"""
    print("Testing image loading...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_image)
    test_path = "test_image.jpg"
    test_pil.save(test_path)
    print(f"Created test image: {test_path}")
    
    try:
        # Try to load it back
        img = Image.open(test_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        print(f"✅ Successfully loaded test image: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"❌ Failed to load test image: {e}")
        return None

def analyze_skin_color_simple(image_array):
    """Simple skin color analysis"""
    try:
        print("Starting simple skin color analysis...")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        print(f"Image shape: {image_array.shape}")
        
        # Calculate average color
        avg_color = np.mean(image_array, axis=(0, 1))
        print(f"Average color (RGB): {avg_color}")
        
        # Calculate brightness
        brightness = np.mean(avg_color)
        print(f"Brightness: {brightness}")
        
        # Simple skin tone classification based on brightness
        if brightness > 0.8:
            skin_tone_level = 1
            skin_tone_desc = "Very Light (Type I)"
        elif brightness > 0.7:
            skin_tone_level = 2
            skin_tone_desc = "Light (Type II)"
        elif brightness > 0.6:
            skin_tone_level = 3
            skin_tone_desc = "Light-Medium (Type III)"
        elif brightness > 0.5:
            skin_tone_level = 4
            skin_tone_desc = "Medium (Type IV)"
        elif brightness > 0.4:
            skin_tone_level = 5
            skin_tone_desc = "Medium-Dark (Type V)"
        elif brightness > 0.3:
            skin_tone_level = 6
            skin_tone_desc = "Dark (Type VI)"
        else:
            skin_tone_level = 7
            skin_tone_desc = "Very Dark (Type VII)"
        
        print(f"✅ Skin tone level: {skin_tone_level}")
        print(f"✅ Skin tone description: {skin_tone_desc}")
        
        return {
            'skin_color_rgb': avg_color,
            'skin_tone_level': skin_tone_level,
            'skin_tone_description': skin_tone_desc,
            'brightness': brightness,
            'method': 'simple_analysis'
        }
        
    except Exception as e:
        print(f"❌ Error in simple analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_skin_color_kmeans(image_array):
    """Analyze skin color using K-means clustering"""
    try:
        print("Starting K-means skin color analysis...")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        print(f"Image shape: {image_array.shape}")
        
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        print(f"Pixels shape: {pixels.shape}")
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        print(f"Found {len(colors)} dominant colors")
        
        # Find the most dominant skin-like color
        skin_scores = []
        for i, color in enumerate(colors):
            r, g, b = color
            # Simple skin detection: higher red, moderate green, lower blue
            skin_score = r * 0.4 + g * 0.3 + b * 0.3
            skin_scores.append(skin_score)
            print(f"Color {i}: RGB({r:.2f}, {g:.2f}, {b:.2f}) - Skin score: {skin_score:.2f}")
        
        # Get the most skin-like color
        dominant_skin_idx = np.argmax(skin_scores)
        dominant_skin_color = colors[dominant_skin_idx]
        
        print(f"✅ Dominant skin color: RGB({dominant_skin_color[0]:.2f}, {dominant_skin_color[1]:.2f}, {dominant_skin_color[2]:.2f})")
        
        # Classify skin tone based on RGB values
        r, g, b = dominant_skin_color
        
        # Calculate skin tone level (0-10 scale)
        brightness = (r + g + b) / 3 / 255.0
        
        if brightness > 0.8:
            skin_tone_level = 1
            skin_tone_desc = "Very Light (Type I)"
        elif brightness > 0.7:
            skin_tone_level = 2
            skin_tone_desc = "Light (Type II)"
        elif brightness > 0.6:
            skin_tone_level = 3
            skin_tone_desc = "Light-Medium (Type III)"
        elif brightness > 0.5:
            skin_tone_level = 4
            skin_tone_desc = "Medium (Type IV)"
        elif brightness > 0.4:
            skin_tone_level = 5
            skin_tone_desc = "Medium-Dark (Type V)"
        elif brightness > 0.3:
            skin_tone_level = 6
            skin_tone_desc = "Dark (Type VI)"
        else:
            skin_tone_level = 7
            skin_tone_desc = "Very Dark (Type VII)"
        
        print(f"✅ Skin tone level: {skin_tone_level}")
        print(f"✅ Skin tone description: {skin_tone_desc}")
        
        return {
            'skin_color_rgb': dominant_skin_color,
            'skin_tone_level': skin_tone_level,
            'skin_tone_description': skin_tone_desc,
            'all_colors': colors,
            'method': 'kmeans_clustering'
        }
        
    except Exception as e:
        print(f"❌ Error in K-means analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🧪 Testing Skin Color Analysis")
    print("=" * 50)
    
    # Test image loading
    img_array = test_image_loading()
    if img_array is None:
        print("❌ Cannot proceed without working image loading")
        return
    
    print("\n🔍 Testing Simple Analysis")
    print("-" * 30)
    simple_result = analyze_skin_color_simple(img_array)
    if simple_result:
        print("✅ Simple analysis successful!")
        print(f"   Skin tone: {simple_result['skin_tone_description']}")
        print(f"   Color: RGB({int(simple_result['skin_color_rgb'][0])}, {int(simple_result['skin_color_rgb'][1])}, {int(simple_result['skin_color_rgb'][2])})")
    else:
        print("❌ Simple analysis failed")
    
    print("\n🔍 Testing K-means Analysis")
    print("-" * 30)
    kmeans_result = analyze_skin_color_kmeans(img_array)
    if kmeans_result:
        print("✅ K-means analysis successful!")
        print(f"   Skin tone: {kmeans_result['skin_tone_description']}")
        print(f"   Color: RGB({int(kmeans_result['skin_color_rgb'][0])}, {int(kmeans_result['skin_color_rgb'][1])}, {int(kmeans_result['skin_color_rgb'][2])})")
    else:
        print("❌ K-means analysis failed")
    
    # Clean up test image
    try:
        os.remove("test_image.jpg")
        print("\n🧹 Cleaned up test image")
    except:
        pass
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()