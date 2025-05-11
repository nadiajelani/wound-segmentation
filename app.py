import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wound-segmentation', 'models')))

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import base64
import time
import cv2
import numpy as np
import tensorflow as tf
from fpdf import FPDF
from gtts import gTTS
from wound_segmentation import build_unet, predict_healing_potential

# Optional: placeholder for MedSAM integration
# from segment_anything import build_sam_vit_b

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
VOICE_FOLDER = "voice_summaries"
MODEL_PATH = "best_unet_wound_model.h5"


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(VOICE_FOLDER, exist_ok=True)

# Load model
model = build_unet(input_shape=(128, 128, 3))
model.load_weights(MODEL_PATH)

# Optional: load MedSAM model if available
# medsam_model = build_sam_vit_b(checkpoint="medsam_vit_b.pth").to("cpu").eval()

def generate_voice_summary(text, output_dir=VOICE_FOLDER):
    from datetime import datetime
    filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    filepath = os.path.join(output_dir, filename)
    try:
        tts = gTTS(text)
        tts.save(filepath)
        return f"/voice/{filename}"
    except Exception as e:
        print(f"Voice generation failed: {e}")
        return None

app = Flask(__name__, template_folder='.')

@app.route("/")
def home():
    return render_template("process_image.html")

@app.route("/voice/<filename>")
def serve_voice(filename):
    return send_from_directory(VOICE_FOLDER, filename)

@app.route("/report/<filename>")
def serve_report(filename):
    return send_from_directory(REPORT_FOLDER, filename)

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_resized = tf.image.resize(img_rgb, (128, 128))[None, ...]

    # Read form flags
    # Check toggles
    use_medsam = request.form.get("use_medsam", "false").lower() == "true"
    voice_summary = request.form.get("voice_summary", "false").lower() == "true"

    # Use U-Net or MedSAM
    if use_medsam:
        # Placeholder logic (fallback to U-Net)
        print("MedSAM selected, but not implemented. Using U-Net fallback.")
    pred = model.predict(img_resized)[0, ..., 0]
    pred_mask = (pred > 0.5).astype(np.uint8)
    pred_mask_resized = tf.image.resize(pred_mask[..., None], img.shape[:2], method='nearest').numpy().squeeze().astype(np.uint8)

    # Clinical analysis
    severity, healing_potential, wound_area_mm2 = predict_healing_potential(pred_mask_resized, img_rgb)

    # PDF report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"wound_report_{timestamp}.pdf"
    report_path = os.path.join(REPORT_FOLDER, report_filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Wound Assessment Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    for line in [
        f"Severity: {severity}",
        f"Healing Potential: {healing_potential}",
        f"Wound Area: {wound_area_mm2:.2f} mm²"
    ]:
        pdf.multi_cell(0, 10, line.strip().encode('latin-1', 'replace').decode('latin-1'))
    pdf.output(report_path)

    # Encode original image
    _, img_buffer = cv2.imencode('.png', cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(img_buffer).decode('utf-8')

    # Encode mask
    _, mask_buffer = cv2.imencode('.png', pred_mask_resized * 255)
    mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')

    # Optional voice
    voice_url = None
    if voice_summary:
        from gtts import gTTS
        summary_text = f"The wound is classified as {severity} with a healing potential of {healing_potential}."
        voice_filename = f"summary_{timestamp}.mp3"
        voice_path = os.path.join(VOICE_FOLDER, voice_filename)
        try:
            tts = gTTS(summary_text)
            tts.save(voice_path)
            voice_url = f"/voice/{voice_filename}"
        except Exception as e:
            print("Voice generation failed:", e)

    return jsonify({
        "pdf_url": f"/report/{report_filename}",
        "voice_summary_url": voice_url,
        "severity": severity,
        "healing_potential": healing_potential,
        "area_mm2": wound_area_mm2,
        "image_base64": image_base64,
        "mask_base64": mask_base64
    })
if __name__ == "__main__":
    app.run(debug=True)
