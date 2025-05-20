from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask import render_template

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import json
import base64
import uuid
from datetime import datetime
from wound_medsam import build_unet, predict_healing_potential, load_medsam_model, medsam_segment

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

UNET_MODEL_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/best_unet_wound_model.h5"
MEDSAM_MODEL_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/best_medsam_model.pth"

model = build_unet(input_shape=(128, 128, 3))
model.load_weights(UNET_MODEL_PATH)

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

def generate_patient_report(image, pred_mask, patient_info, severity, healing_potential, wound_area_mm2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Check write permissions
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"No write permissions for output directory: {output_dir}")

    wound_area_px = np.sum(pred_mask > 0)
    estimated_diameter = np.sqrt(wound_area_px / np.pi) * 0.264
    wound_description = f"""
    Wound Description:
    - Size: The wound is about {estimated_diameter:.2f} millimeters wide, roughly {'smaller than a US dime' if estimated_diameter < 18 else 'about the size of a US dime' if estimated_diameter < 22 else 'larger than a US dime'}.
    - Severity: {severity}
    - Healing Potential: {healing_potential}
    """
    patient_instructions = """
    What to Do:
    - Keep the wound clean and dry.
    - Change the dressing daily or as told by your doctor.
    - Watch for signs like redness, swelling, or pain.
    - Contact your doctor if the wound doesn't improve in 3–5 days.
    """
    report_text = f"""
    About You:
    - Name: {patient_info.get('name', 'Unknown')}
    - Age: {patient_info.get('age', 'Unknown')}

    {wound_description}
    {patient_instructions}
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Wound Assessment Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    for line in report_text.strip().split("\n"):
        pdf.multi_cell(0, 10, line.strip().encode('latin-1', 'replace').decode('latin-1'))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vis_path = os.path.join(output_dir, f"wound_vis_{timestamp}.png")
    create_visualization(image, pred_mask, vis_path)
    pdf.ln(10)
    pdf.image(vis_path, x=10, w=190)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Explanation: The left image shows your wound. The middle image shows the detected wound area. The right image highlights the wound with a green outline.")
    report_path = os.path.join(output_dir, f"patient_wound_report_{timestamp}.pdf")
    pdf.output(report_path)
    return report_path, vis_path

def analyze_image(image_path, unet_model_path, medsam_model_path, patient_info, output_dir='analysis_output'):
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Ensure 3-channel input (handle grayscale images)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Explicitly resize for model input
    img_resized = tf.image.resize(img_rgb, (128, 128), method='bilinear')
    img_resized = tf.expand_dims(img_resized, 0)  # Add batch dimension
    if img_resized.shape != (1, 128, 128, 3):
        raise ValueError(f"Resized image shape {img_resized.shape} does not match expected shape (1, 128, 128, 3)")

    # Load models
    unet_model = build_unet(input_shape=(128, 128, 3))
    unet_model.load_weights(unet_model_path)
    medsam_model = load_medsam_model(medsam_model_path)

    # Predict masks
    unet_pred = unet_model.predict([img_resized], verbose=0)[0, ..., 0]  # Pass as list to avoid Keras warning
    unet_mask = (unet_pred > 0.5).astype(np.uint8)

    medsam_mask = medsam_segment(img_rgb, medsam_model)
    medsam_mask = tf.image.resize(medsam_mask[..., None], (128, 128), method='nearest').numpy().squeeze().astype(np.uint8)

    combined_mask = np.logical_or(unet_mask, medsam_mask).astype(np.uint8)

    # Resize mask back to original size
    pred_mask_resized = tf.image.resize(combined_mask[..., None], img_rgb.shape[:2], method='nearest').numpy().squeeze().astype(np.uint8)

    # Clinical analysis
    severity, healing_potential, wound_area_mm2 = predict_healing_potential(pred_mask_resized, img_rgb)

    # Generate PDF report
    report_path, vis_path = generate_patient_report(img_rgb, pred_mask_resized, patient_info, severity, healing_potential, wound_area_mm2, output_dir)

    print(f"\n✅ Wound analysis complete.")
    print(f"- Report saved at: {report_path}")
    print(f"- Visualization saved at: {vis_path}")
    print(f"- Severity: {severity}")
    print(f"- Healing Potential: {healing_potential}")
    print(f"- Wound Area: {wound_area_mm2:.2f} mm²")

    return report_path, vis_path

@app.route("/report/<filename>")
def serve_report(filename):
    return send_from_directory(REPORT_FOLDER, filename)

@app.route("/upload", methods=["POST"])
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            raise ValueError("No image file provided in the request")
        file = request.files["image"]
        name = request.form.get("name", "Unknown")
        age = request.form.get("age", "Unknown")
        use_medsam = request.form.get("use_medsam", "false").lower() == "true"

        filename = str(uuid.uuid4()) + ".png"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # Validate the saved image
        img_check = cv2.imread(image_path)
        if img_check is None:
            raise ValueError(f"Failed to load the uploaded image: {image_path}. Ensure the file is a valid image (e.g., PNG, JPEG).")

        patient_info = {"name": name, "age": age}

        report_path, vis_path = analyze_image(
            image_path=image_path,
            unet_model_path=UNET_MODEL_PATH,
            medsam_model_path=MEDSAM_MODEL_PATH,
            patient_info=patient_info,
            output_dir=REPORT_FOLDER
        )

        return jsonify({
            "report_url": f"/report/{os.path.basename(report_path)}",
        })
    
    except Exception as e:
        import traceback
        print("❌ Error during image processing:", str(e))
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        import traceback
        print("❌ Error during image processing:", str(e))
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("wound-wisperer.html")

if __name__ == "__main__":
    app.run(debug=True)
