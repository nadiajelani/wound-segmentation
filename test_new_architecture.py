#!/usr/bin/env python3
"""
Test script using the new modular architecture.

This script demonstrates how to use the new modular services
to save files to the organized outputs/ directory structure.
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

# Import the new modular services
from woundseg.services import (
    get_storage_service,
    get_voice_service,
    get_reporting_service,
    get_validation_service
)
from woundseg.types import Patient, AnalysisResult

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
    """Main function using new modular architecture."""
    print("🏥 Wound Analysis with New Modular Architecture")
    print("=" * 60)
    
    # Get services
    storage_service = get_storage_service()
    voice_service = get_voice_service()
    reporting_service = get_reporting_service()
    validation_service = get_validation_service()
    
    print(f"✅ Storage service: {storage_service.base_dir}")
    print(f"✅ Voice service: {'Available' if voice_service.is_available() else 'Not available'}")
    print(f"✅ Reporting service: Available")
    print(f"✅ Validation service: Available")
    
    # Select image
    input_image_path = select_image()
    
    # Load model
    try:
        segmentation_model = tf.keras.models.load_model(segmentation_model_path, compile=False)
        logger.info(f"Loaded segmentation model from {segmentation_model_path}")
    except Exception as e:
        logger.error(f"Failed to load segmentation model: {e}")
        raise
    
    # Load and preprocess image
    try:
        original = cv2.imread(input_image_path)
        if original is None:
            raise ValueError(f"Could not read image: {input_image_path}")
        
        img = load_img(input_image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"✅ Image loaded: {original.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise
    
    # Run segmentation
    try:
        print("\n🔬 Running wound segmentation...")
        prediction = segmentation_model.predict(img_array, verbose=0)
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        print("✅ Segmentation completed")
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise
    
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
        logger.error(f"Metrics calculation failed: {e}")
        raise
    
    # Save files using new modular services
    try:
        print("\n💾 Saving files using new modular architecture...")
        
        # 1. Save original image to uploads/
        with open(input_image_path, 'rb') as f:
            image_data = f.read()
        upload_path = storage_service.save_image(
            image_data, "uploads", f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        print(f"✅ Original image saved to: {upload_path}")
        
        # 2. Save mask to masks/
        _, mask_encoded = cv2.imencode('.png', mask)
        mask_path = storage_service.save_image(
            mask_encoded.tobytes(), "masks", f"segmentation_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        print(f"✅ Segmentation mask saved to: {mask_path}")
        
        # 3. Create and save overlay to visualizations/
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
        _, overlay_encoded = cv2.imencode('.png', overlay)
        overlay_path = storage_service.save_image(
            overlay_encoded.tobytes(), "visualizations", f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        print(f"✅ Overlay saved to: {overlay_path}")
        
        # 4. Create and save heatmap to visualizations/
        heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        _, heatmap_encoded = cv2.imencode('.png', heatmap)
        heatmap_path = storage_service.save_image(
            heatmap_encoded.tobytes(), "visualizations", f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        print(f"✅ Heatmap saved to: {heatmap_path}")
        
        # 5. Create and save trend chart to visualizations/
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
        
        # Save plot to bytes
        import io
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        trend_path = storage_service.save_image(
            plot_data, "visualizations", f"trend_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        print(f"✅ Trend chart saved to: {trend_path}")
        
        # 6. Save analysis data to reports/
        analysis_data = f"""Wound Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {os.path.basename(input_image_path)}

Results:
- Wound Area: {area_mm2:.2f} mm²
- Perimeter: {perimeter_mm:.2f} mm
- Condition: {condition}
- Healing Potential: {healing_potential}
- Severity: {severity}

Recommendations:
- Continue current treatment plan
- Monitor progress regularly
- Consult healthcare provider if condition changes
"""
        
        report_path = storage_service.save_text(
            analysis_data, "reports", f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        print(f"✅ Analysis report saved to: {report_path}")
        
        # 7. Generate voice summaries
        if voice_service.is_available():
            print("\n🎤 Generating voice summaries...")
            
            analysis_result_dict = {
                'area_mm2': area_mm2,
                'healing_potential': healing_potential,
                'confidence_score': 0.85,
                'severity': severity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Generate patient voice
            patient_audio = voice_service.generate_summary(
                analysis_result_dict, 
                audience="patient", 
                filename=f"patient_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )
            
            # Generate clinician voice
            clinician_audio = voice_service.generate_summary(
                analysis_result_dict, 
                audience="clinician", 
                filename=f"clinician_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )
            
            if patient_audio and clinician_audio:
                print(f"✅ Patient voice: {patient_audio}")
                print(f"✅ Clinician voice: {clinician_audio}")
            else:
                print("❌ Voice generation failed")
        else:
            print("⚠️  Voice service not available - skipping voice generation")
        
        # 8. Generate PDF reports using reporting service
        try:
            print("\n📄 Generating PDF reports...")
            
            # Create patient and analysis result objects
            patient = Patient(
                name="Patient",
                age=45
            )
            
            analysis_result = AnalysisResult(
                area_mm2=area_mm2,
                healing_potential=healing_potential,
                confidence_score=0.85,
                severity=severity,
                original_image=original,
                mask=mask
            )
            
            # Generate patient report
            patient_report = reporting_service.generate_patient_report(
                patient, analysis_result, f"patient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            # Generate clinician report
            clinician_report = reporting_service.generate_clinician_report(
                patient, analysis_result, f"clinician_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            if patient_report:
                print(f"✅ Patient PDF report: {patient_report}")
            if clinician_report:
                print(f"✅ Clinician PDF report: {clinician_report}")
                
        except Exception as e:
            print(f"⚠️  PDF report generation failed: {e}")
        
        # Show final file organization
        print("\n📁 Files organized in outputs/ directory:")
        print("=" * 50)
        
        for name, path in storage_service.dirs.items():
            if path.exists():
                files = list(path.glob("*"))
                if files:
                    print(f"\n📂 {name.upper()}:")
                    for file in files:
                        print(f"   - {file.name}")
                else:
                    print(f"\n📂 {name.upper()}: (empty)")
        
        print("\n✅ Analysis completed successfully!")
        print("🎯 All files are organized in the outputs/ directory structure")
        
    except Exception as e:
        logger.error(f"Failed to save files: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis cancelled by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)