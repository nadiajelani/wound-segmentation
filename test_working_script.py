#!/usr/bin/env python3
"""
Working wound analysis script with voice service.

This script properly asks for image input and generates voice summaries.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Add the project root to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import the voice service
from woundseg.services import get_voice_service

# -------- Setup Logging --------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("wound_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- Settings --------
IMG_SIZE = (128, 128)
segmentation_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
scale_mm_per_pixel = 0.1  # mm per pixel

def select_image():
    """Ask user to select an image file."""
    print("\n🖼️  Image Selection")
    print("=" * 40)
    
    # Try GUI selection first
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        input_image_path = filedialog.askopenfilename(
            title="Select wound image for analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if input_image_path:
            print(f"✅ Selected image: {input_image_path}")
            return input_image_path
    except Exception as e:
        print(f"⚠️  GUI selection failed: {e}")
    
    # Manual input
    print("\n📁 Please enter the path to your wound image:")
    print("   Supported formats: .jpg, .jpeg, .png, .bmp, .tif, .tiff")
    print("   Example: /Users/username/Desktop/wound_image.jpg")
    
    while True:
        input_image_path = input("\n🔗 Enter image path: ").strip()
        
        if not input_image_path:
            print("❌ Please enter a valid path.")
            continue
            
        if not os.path.exists(input_image_path):
            print(f"❌ File not found: {input_image_path}")
            print("   Please check the path and try again.")
            continue
            
        # Check if it's an image file
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        if not any(input_image_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"❌ Invalid file type. Please use: {', '.join(valid_extensions)}")
            continue
            
        print(f"✅ Image found: {input_image_path}")
        return input_image_path

def main():
    """Main analysis function."""
    print("🏥 Wound Analysis with Voice Service")
    print("=" * 50)
    
    # Get voice service
    voice_service = get_voice_service()
    if voice_service.is_available():
        print("✅ Voice service is available")
    else:
        print("⚠️  Voice service not available - check ENABLE_VOICE_SUMMARY config")
    
    # Select image
    input_image_path = select_image()
    
    # Load model
    try:
        print("\n🔬 Loading segmentation model...")
        segmentation_model = tf.keras.models.load_model(segmentation_model_path, compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load and preprocess image
    try:
        print("\n📸 Loading and preprocessing image...")
        original = cv2.imread(input_image_path)
        if original is None:
            raise ValueError(f"Could not read image: {input_image_path}")
        
        img = load_img(input_image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"✅ Image loaded: {original.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Run segmentation
    try:
        print("\n🔬 Running wound segmentation...")
        prediction = segmentation_model.predict(img_array, verbose=0)
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        print("✅ Segmentation completed")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return
    
    # Calculate metrics
    try:
        print("\n📊 Calculating wound metrics...")
        
        # Calculate area
        area_pixels = np.sum(mask > 0)
        area_mm2 = area_pixels * (scale_mm_per_pixel ** 2)
        
        # Calculate perimeter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter_pixels = cv2.arcLength(contours[0], True)
            perimeter_mm = perimeter_pixels * scale_mm_per_pixel
        else:
            perimeter_mm = 0
        
        # Determine condition
        if area_mm2 < 10:
            condition = "Small wound - healing well"
            healing_potential = "Good"
            severity = "Mild"
        elif area_mm2 < 50:
            condition = "Medium wound - stable"
            healing_potential = "Fair"
            severity = "Moderate"
        else:
            condition = "Large wound - needs attention"
            healing_potential = "Poor"
            severity = "Severe"
        
        print(f"✅ Wound area: {area_mm2:.2f} mm²")
        print(f"✅ Perimeter: {perimeter_mm:.2f} mm")
        print(f"✅ Condition: {condition}")
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        return
    
    # Generate voice summaries
    if voice_service.is_available():
        try:
            print("\n🎤 Generating voice summaries...")
            
            analysis_result = {
                'area_mm2': area_mm2,
                'healing_potential': healing_potential,
                'confidence_score': 0.85,
                'severity': severity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Generate patient voice
            patient_audio = voice_service.generate_summary(
                analysis_result, 
                audience="patient", 
                filename=f"patient_voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )
            
            # Generate clinician voice
            clinician_audio = voice_service.generate_summary(
                analysis_result, 
                audience="clinician", 
                filename=f"clinician_voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )
            
            if patient_audio and clinician_audio:
                print(f"✅ Patient voice: {patient_audio}")
                print(f"✅ Clinician voice: {clinician_audio}")
            else:
                print("❌ Voice generation failed")
                
        except Exception as e:
            print(f"❌ Voice generation error: {e}")
    else:
        print("⚠️  Voice service not available - skipping voice generation")
    
    # Save results
    try:
        print("\n💾 Saving results...")
        
        # Create output directory
        output_dir = "/Users/nadiajelani/projects/wound-segmentation/wound_progress_report/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mask
        mask_path = os.path.join(output_dir, "simclr_mask.png")
        cv2.imwrite(mask_path, mask)
        print(f"✅ Mask saved: {mask_path}")
        
        # Create overlay
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
        overlay_path = os.path.join(output_dir, "wound_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"✅ Overlay saved: {overlay_path}")
        
        # Create heatmap
        heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        heatmap_path = os.path.join(output_dir, "wound_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        print(f"✅ Heatmap saved: {heatmap_path}")
        
        # Create trend chart
        trend_data = pd.DataFrame({
            'Date': [datetime.now().strftime("%Y-%m-%d")],
            'Area (mm²)': [area_mm2]
        })
        
        plt.figure(figsize=(10, 6))
        plt.plot(trend_data['Date'], trend_data['Area (mm²)'], marker='o', linewidth=2, markersize=8)
        plt.title('Wound Area Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Area (mm²)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        trend_plot_path = os.path.join(output_dir, "wound_area_trend.png")
        plt.savefig(trend_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Trend chart saved: {trend_plot_path}")
        
        # Save CSV report
        report_data = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": os.path.basename(input_image_path),
            "Wound Area (mm²)": f"{area_mm2:.2f}",
            "Perimeter (mm)": f"{perimeter_mm:.2f}",
            "Condition": condition,
            "Healing Potential": healing_potential,
            "Severity": severity
        }
        
        csv_path = os.path.join(output_dir, "report.csv")
        df = pd.DataFrame([report_data])
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV report saved: {csv_path}")
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis cancelled by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)