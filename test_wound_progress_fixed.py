#!/usr/bin/env python3
"""
Fixed version of test_wound_progress.py that properly asks for image input.

This version includes better image selection and the new voice service integration.
"""

import os
import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# Add the project root to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import the new modular services
from woundseg.services import get_voice_service

# -------- Setup Logging --------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("wound_progress.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- Settings --------
IMG_SIZE = (128, 128)
segmentation_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
report_dir = "/Users/nadiajelani/projects/wound-segmentation/wound_progress_report/"
os.makedirs(report_dir, exist_ok=True)

scale_mm_per_pixel = 0.1  # mm per pixel (set according to your setup)
scale_bar_length_mm = 10

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

def generate_voice_summary(analysis_result, voice_service):
    """Generate voice summary if voice service is available."""
    if not voice_service or not voice_service.is_available():
        print("⚠️  Voice service not available - skipping voice generation")
        return None
    
    try:
        print("\n🎤 Generating voice summary...")
        
        # Generate patient voice
        patient_audio = voice_service.generate_summary(
            analysis_result, 
            audience="patient", 
            filename="wound_analysis_patient.mp3"
        )
        
        # Generate clinician voice
        clinician_audio = voice_service.generate_summary(
            analysis_result, 
            audience="clinician", 
            filename="wound_analysis_clinician.mp3"
        )
        
        if patient_audio and clinician_audio:
            print(f"✅ Patient voice: {patient_audio}")
            print(f"✅ Clinician voice: {clinician_audio}")
            return {"patient": patient_audio, "clinician": clinician_audio}
        else:
            print("❌ Voice generation failed")
            return None
            
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
        return None

def main():
    """Main function."""
    print("🏥 Wound Analysis with Voice Service")
    print("=" * 50)
    
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
        elif area_mm2 < 50:
            condition = "Medium wound - stable"
            healing_potential = "Fair"
        else:
            condition = "Large wound - needs attention"
            healing_potential = "Poor"
        
        print(f"✅ Wound area: {area_mm2:.2f} mm²")
        print(f"✅ Perimeter: {perimeter_mm:.2f} mm")
        print(f"✅ Condition: {condition}")
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise
    
    # Create analysis result for voice service
    analysis_result = {
        'area_mm2': area_mm2,
        'healing_potential': healing_potential,
        'confidence_score': 0.85,  # Mock confidence
        'severity': 'Mild' if area_mm2 < 10 else 'Moderate' if area_mm2 < 50 else 'Severe',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate voice summary
    voice_service = get_voice_service()
    voice_files = generate_voice_summary(analysis_result, voice_service)
    
    # Save results
    try:
        print("\n💾 Saving results...")
        
        # Save mask
        mask_path = os.path.join(report_dir, "simclr_mask.png")
        cv2.imwrite(mask_path, mask)
        logger.info(f"Saved mask at {mask_path}")
        
        # Create overlay - resize mask to match original image
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
        overlay_path = os.path.join(report_dir, "wound_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        logger.info(f"Overlay image saved at {overlay_path}")
        
        # Create heatmap
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap_path = os.path.join(report_dir, "wound_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        logger.info(f"Segmentation heatmap saved at {heatmap_path}")
        
        # Create trend plot
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
        
        trend_plot_path = os.path.join(report_dir, "wound_area_trend.png")
        plt.savefig(trend_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Wound area trend plot saved at {trend_plot_path}")
        
        # Create report data
        report = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": os.path.basename(input_image_path),
            "Wound Area (mm²)": f"{area_mm2:.2f}",
            "Perimeter (mm)": f"{perimeter_mm:.2f}",
            "Shape Irregularity": "Regular",
            "Condition": condition,
            "Instructions": "Continue current treatment plan and monitor progress."
        }
        
        # Save CSV report
        csv_report_path = os.path.join(report_dir, "report.csv")
        df = pd.DataFrame([report])
        df.to_csv(csv_report_path, index=False)
        
        # Generate PDF report
        pdf_report_path = os.path.join(report_dir, "wound_report.pdf")
        doc = SimpleDocTemplate(pdf_report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        elements.append(Paragraph("Wound Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # Add report details
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Date: {current_date}", styles['Normal']))
        elements.append(Paragraph(f"Image: {report['Image']}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Add data table
        data = [
            ["Parameter", "Value"],
            ["Wound Area (mm²)", str(report.get("Wound Area (mm²)", "N/A"))],
            ["Perimeter (mm)", str(report.get("Perimeter (mm)", "N/A"))],
            ["Shape Irregularity", str(report.get("Shape Irregularity", "N/A"))],
            ["Condition", report.get("Condition", "N/A")],
            ["Instructions", report.get("Instructions", "N/A")]
        ]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        # Add images
        for img_path, title in [
            (input_image_path, "Original Image"),
            (mask_path, "Predicted Mask"),
            (overlay_path, "Overlay"),
            (heatmap_path, "Heatmap"),
            (trend_plot_path, "Area Trend"),
        ]:
            if os.path.exists(img_path):
                elements.append(Paragraph(title, styles['Heading2']))
                elements.append(RLImage(img_path, width=2*inch, height=2*inch))
                elements.append(Spacer(1, 12))
        
        doc.build(elements)
        logger.info(f"PDF report saved at {pdf_report_path}")
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {report_dir}")
        print(f"📄 PDF report: {pdf_report_path}")
        
        if voice_files:
            print(f"🎤 Voice files:")
            print(f"   Patient: {voice_files['patient']}")
            print(f"   Clinician: {voice_files['clinician']}")
        
        print("\n🎉 Done! See your report for wound details and advice.")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
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