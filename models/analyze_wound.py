import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import json
from datetime import datetime

from wound_segmentation import build_unet, predict_healing_potential, load_medsam_model, medsam_segment

def generate_patient_report(image, pred_mask, patient_info, severity, healing_potential, wound_area_mm2, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Calculate estimated diameter
    wound_area_px = np.sum(pred_mask > 0)
    estimated_diameter = np.sqrt(wound_area_px / np.pi) * 0.264  # 96 DPI, mm/pixel

    # Patient-friendly wound description
    wound_description = f"""
    Wound Description:
    - Size: The wound is about {estimated_diameter:.2f} millimeters wide, roughly {'smaller than a US dime' if estimated_diameter < 18 else 'about the size of a US dime' if estimated_diameter < 22 else 'larger than a US dime'}.
    - Severity: {severity}
    - Healing Potential: {healing_potential}
    """

    # Simplified instructions
    patient_instructions = """
    What to Do:
    - Keep the wound clean and dry.
    - Change the dressing daily or as told by your doctor.
    - Watch for signs like redness, swelling, or pain.
    - Contact your doctor if the wound doesn't improve in 3–5 days.
    """

    # Report text
    report_text = f"""
    About You:
    - Name: {patient_info.get('name', 'Unknown')}
    - Age: {patient_info.get('age', 'Unknown')}

    {wound_description}

    {patient_instructions}
    """

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Wound Assessment Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    # Add report text
    for line in report_text.strip().split("\n"):
        pdf.multi_cell(0, 10, line.strip().encode('latin-1', 'replace').decode('latin-1'))

    # Add visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vis_path = os.path.join(output_dir, f"wound_vis_{timestamp}.png")
    create_visualization(image, pred_mask, vis_path)
    pdf.ln(10)
    pdf.image(vis_path, x=10, w=190)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Explanation: The left image shows your wound. The middle image shows the detected wound area. The right image highlights the wound with a green outline.")

    # Save PDF
    report_path = os.path.join(output_dir, f"patient_wound_report_{timestamp}.pdf")
    pdf.output(report_path)
    return report_path, vis_path

def create_visualization(image, pred_mask, output_path):
    img_rgb = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.copy()
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Your Wound")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Detected Wound Area")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(contour_img)
    plt.title("Wound Outline")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def analyze_image(image_path, unet_model_path, medsam_model_path, patient_info, output_dir='analysis_output'):
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_resized = tf.image.resize(img_rgb, (128, 128))[None, ...]

    # Load U-Net model
    try:
        unet_model = build_unet(input_shape=(128, 128, 3))
        unet_model.load_weights(unet_model_path)
    except Exception as e:
        raise ValueError(f"Error loading U-Net model: {e}")

    # Load MedSAM model
    try:
        medsam_model = load_medsam_model(medsam_model_path)
    except Exception as e:
        raise ValueError(f"Error loading MedSAM model: {e}")

    # Predict with U-Net
    unet_pred = unet_model.predict(img_resized, verbose=0)[0, ..., 0]
    unet_mask = (unet_pred > 0.5).astype(np.uint8)

    # Predict with MedSAM
    medsam_mask = medsam_segment(img_rgb, medsam_model)
    medsam_mask = tf.image.resize(medsam_mask[..., None], (128, 128), method='nearest').numpy().squeeze().astype(np.uint8)

    # Combine predictions (union)
    combined_mask = np.logical_or(unet_mask, medsam_mask).astype(np.uint8)

    # Restore original size
    pred_mask_resized = tf.image.resize(combined_mask[..., None], img_rgb.shape[:2], method='nearest').numpy().squeeze().astype(np.uint8)

    # Estimate clinical info
    severity, healing_potential, wound_area_mm2 = predict_healing_potential(pred_mask_resized, img_rgb)

    # Generate report
    report_path, vis_path = generate_patient_report(img_rgb, pred_mask_resized, patient_info, severity, healing_potential, wound_area_mm2, output_dir)

    print(f"\n✅ Wound analysis complete.")
    print(f"- Report saved at: {report_path}")
    print(f"- Visualization saved at: {vis_path}")
    print(f"- Severity: {severity}")
    print(f"- Healing Potential: {healing_potential}")
    print(f"- Wound Area: {wound_area_mm2:.2f} mm²")

def main():
    # Directories and paths
    input_dir = "/Users/nadiajelani/Desktop/patient_input"
    unet_model_path = "/Users/nadiajelani/Desktop/Medical Pics/wound-segmentation/models/best_unet_wound_model.h5"
    medsam_model_path = "/Users/nadiajelani/Desktop/Medical Pics/wound-segmentation/models/medsam_vit_b.pth"
    output_dir = "/Users/nadiajelani/Desktop/wound_analysis_output"
    
    # Instructions for patient
    print("Please follow these steps to generate your wound report:")
    print(f"1. Save your wound image (PNG or JPG) in: {input_dir}")
    print(f"2. Create a file named 'patient_info.json' in {input_dir} with the following format:")
    print('   ```json')
    print('   {"name": "Your Name", "age": Your Age}')
    print('   ```')
    print("3. Run this script again after placing the files.")
    print(f"   Example: If your name is John Doe and you're 50, the JSON should be:")
    print('   {"name": "John Doe", "age": 50}')

    # Check for patient input
    patient_info_path = os.path.join(input_dir, "patient_info.json")
    image_path = None
    for file in os.listdir(input_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, file)
            break

    if not os.path.exists(patient_info_path) or not image_path:
        print("Error: Missing patient_info.json or wound image. Please provide both and try again.")
        exit()

    # Load patient info
    try:
        with open(patient_info_path, "r") as f:
            patient_info = json.load(f)
        patient_name = patient_info.get("name", "Unknown")
        patient_age = patient_info.get("age", "Unknown")
    except Exception as e:
        print(f"Error reading patient_info.json: {e}")
        exit()

    # Run analysis
    try:
        analyze_image(image_path, unet_model_path, medsam_model_path, patient_info, output_dir)
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()