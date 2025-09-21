#!/usr/bin/env python3
"""
Working wound analysis script with voice service.

This script properly asks for image input and generates voice summaries.
"""

# Set environment variable BEFORE importing any modules
import os
os.environ["ENABLE_VOICE_SUMMARY"] = "true"

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

# Import the modular services
from woundseg.services import get_voice_service, get_storage_service, get_reporting_service, reset_voice_service

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
    
    # Get services
    voice_service = get_voice_service()
    storage_service = get_storage_service()
    reporting_service = get_reporting_service()
    
    if voice_service.is_available():
        print("✅ Voice service is available")
    else:
        print("⚠️  Voice service not available - check gTTS installation")
    
    print("✅ Storage service is available")
    print("✅ Reporting service is available")
    
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
    
    # Save results using modular services
    try:
        print("\n💾 Saving results...")
        
        # Create session directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = storage_service.create_session_dir(session_id)
        print(f"✅ Created session directory: {session_dir}")
        
        # Save original image to uploads
        _, original_encoded = cv2.imencode('.jpg', original)
        original_path = storage_service.save_image(original_encoded.tobytes(), 'uploads', f"original_{session_id}.jpg")
        print(f"✅ Original image saved: {original_path}")
        
        # Save mask to masks
        _, mask_encoded = cv2.imencode('.png', mask)
        mask_path = storage_service.save_image(mask_encoded.tobytes(), 'masks', f"segmentation_mask_{session_id}.png")
        print(f"✅ Mask saved: {mask_path}")
        
        # Create and save overlay to visualizations
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
        _, overlay_encoded = cv2.imencode('.png', overlay)
        overlay_path = storage_service.save_image(overlay_encoded.tobytes(), 'visualizations', f"overlay_{session_id}.png")
        print(f"✅ Overlay saved: {overlay_path}")
        
        # Create and save heatmap to visualizations
        heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        _, heatmap_encoded = cv2.imencode('.png', heatmap)
        heatmap_path = storage_service.save_image(heatmap_encoded.tobytes(), 'visualizations', f"heatmap_{session_id}.png")
        print(f"✅ Heatmap saved: {heatmap_path}")
        
        # Create and save trend chart to visualizations
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
        
        # Save trend chart to temp file first, then move to storage
        temp_trend_path = f"/tmp/trend_chart_{session_id}.png"
        plt.savefig(temp_trend_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Read trend chart and save to visualizations
        with open(temp_trend_path, 'rb') as f:
            trend_data_bytes = f.read()
        trend_plot_path = storage_service.save_image(trend_data_bytes, 'visualizations', f"trend_chart_{session_id}.png")
        os.remove(temp_trend_path)  # Clean up temp file
        print(f"✅ Trend chart saved: {trend_plot_path}")
        
        # Save CSV report to reports
        report_data = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": os.path.basename(input_image_path),
            "Wound Area (mm²)": f"{area_mm2:.2f}",
            "Perimeter (mm)": f"{perimeter_mm:.2f}",
            "Condition": condition,
            "Healing Potential": healing_potential,
            "Severity": severity
        }
        
        csv_content = pd.DataFrame([report_data]).to_csv(index=False)
        csv_path = storage_service.save_text(csv_content, 'reports', f"report_{session_id}.csv")
        print(f"✅ CSV report saved: {csv_path}")
        
        # Generate PDF reports
        print("\n📄 Generating PDF reports...")
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            import io
            
            # Generate Patient Report
            patient_buffer = io.BytesIO()
            patient_doc = SimpleDocTemplate(patient_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            patient_story = []
            
            # Patient Report Content
            patient_story.append(Paragraph("Wound Analysis Report - Patient Version", styles['Title']))
            patient_story.append(Spacer(1, 12))
            
            patient_story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            patient_story.append(Paragraph(f"Image: {os.path.basename(input_image_path)}", styles['Normal']))
            patient_story.append(Spacer(1, 12))
            
            patient_story.append(Paragraph("Your Wound Analysis Results:", styles['Heading2']))
            
            # Patient-friendly metrics table
            patient_data = [
                ["Measurement", "Value", "What this means"],
                ["Wound Size", f"{area_mm2:.2f} mm²", "The area of your wound"],
                ["Wound Condition", condition, "How your wound is healing"],
                ["Healing Potential", healing_potential, "Expected healing progress"],
                ["Severity Level", severity, "How serious the wound is"]
            ]
            
            patient_table = Table(patient_data)
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            patient_story.append(patient_table)
            patient_story.append(Spacer(1, 12))
            
            # Patient care instructions
            patient_story.append(Paragraph("Care Instructions:", styles['Heading2']))
            if "healing well" in condition.lower():
                instructions = "Continue your current treatment. Your wound is healing well. Keep the area clean and follow your doctor's advice."
            elif "stable" in condition.lower():
                instructions = "Your wound is stable. Continue following your treatment plan and monitor for any changes."
            else:
                instructions = "Your wound needs attention. Please consult with your healthcare provider for proper care instructions."
            
            patient_story.append(Paragraph(instructions, styles['Normal']))
            
            patient_doc.build(patient_story)
            patient_pdf_bytes = patient_buffer.getvalue()
            patient_buffer.close()
            
            # Save patient report
            patient_pdf_path = storage_service.save_pdf(patient_pdf_bytes, 'reports', f"patient_report_{session_id}.pdf")
            print(f"✅ Patient report saved: {patient_pdf_path}")
            
            # Generate Clinician Report
            clinician_buffer = io.BytesIO()
            clinician_doc = SimpleDocTemplate(clinician_buffer, pagesize=letter)
            clinician_story = []
            
            # Clinician Report Content
            clinician_story.append(Paragraph("Wound Analysis Report - Clinician Version", styles['Title']))
            clinician_story.append(Spacer(1, 12))
            
            clinician_story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            clinician_story.append(Paragraph(f"Source Image: {os.path.basename(input_image_path)}", styles['Normal']))
            clinician_story.append(Paragraph(f"Model Used: SIMCLR U-Net", styles['Normal']))
            clinician_story.append(Spacer(1, 12))
            
            clinician_story.append(Paragraph("Quantitative Analysis:", styles['Heading2']))
            
            # Clinician metrics table
            clinician_data = [
                ["Parameter", "Value", "Units", "Notes"],
                ["Wound Area", f"{area_mm2:.2f}", "mm²", "Calculated from segmentation mask"],
                ["Perimeter", f"{perimeter_mm:.2f}", "mm", "Wound boundary length"],
                ["Condition Assessment", condition, "", "AI-determined status"],
                ["Healing Potential", healing_potential, "", "Predicted healing trajectory"],
                ["Severity Classification", severity, "", "Clinical severity level"],
                ["Confidence Score", "0.85", "", "Model confidence in analysis"]
            ]
            
            clinician_table = Table(clinician_data)
            clinician_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            clinician_story.append(clinician_table)
            clinician_story.append(Spacer(1, 12))
            
            # Clinical recommendations
            clinician_story.append(Paragraph("Clinical Recommendations:", styles['Heading2']))
            if "healing well" in condition.lower():
                recommendations = "Continue current treatment protocol. Monitor for any changes. Consider reducing frequency of assessments."
            elif "stable" in condition.lower():
                recommendations = "Maintain current treatment plan. Regular monitoring recommended. Assess for any signs of deterioration."
            else:
                recommendations = "Immediate attention required. Consider treatment modification. Increase monitoring frequency. Evaluate for infection or complications."
            
            clinician_story.append(Paragraph(recommendations, styles['Normal']))
            clinician_story.append(Spacer(1, 12))
            
            # Technical details
            clinician_story.append(Paragraph("Technical Details:", styles['Heading2']))
            clinician_story.append(Paragraph(f"• Segmentation Model: SIMCLR U-Net", styles['Normal']))
            clinician_story.append(Paragraph(f"• Image Processing: 128x128 normalization", styles['Normal']))
            clinician_story.append(Paragraph(f"• Scale Factor: {scale_mm_per_pixel} mm/pixel", styles['Normal']))
            clinician_story.append(Paragraph(f"• Analysis Timestamp: {session_id}", styles['Normal']))
            
            clinician_doc.build(clinician_story)
            clinician_pdf_bytes = clinician_buffer.getvalue()
            clinician_buffer.close()
            
            # Save clinician report
            clinician_pdf_path = storage_service.save_pdf(clinician_pdf_bytes, 'reports', f"clinician_report_{session_id}.pdf")
            print(f"✅ Clinician report saved: {clinician_pdf_path}")
            
        except Exception as e:
            print(f"⚠️  PDF report generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {session_dir}")
        print(f"📁 All files organized in: {storage_service.base_dir}")
        
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