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
import subprocess

# ------------ Logger ------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("wound_progress.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------ Settings -----------
IMG_SIZE = (128, 128)
segmentation_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"  # UPDATE to your actual model
report_dir = "/Users/nadiajelani/projects/wound-segmentation/models/wound_progress_report/Skin_report"
os.makedirs(report_dir, exist_ok=True)
csv_report_path = os.path.join(report_dir, "report.csv")
pdf_report_path = os.path.join(report_dir, "wound_report.pdf")
mask_path = os.path.join(report_dir, "simclr_mask.png")
overlay_path = os.path.join(report_dir, "wound_overlay.png")
heatmap_path = os.path.join(report_dir, "wound_heatmap.png")
trend_plot_path = os.path.join(report_dir, "wound_area_trend.png")
skin_palette_img = os.path.join(report_dir, "skin_palette_debug.png")
skin_face_img = os.path.join(report_dir, "skin_face_result.png")

scale_mm_per_pixel = 0.1
scale_bar_length_mm = 10

TONE_LABEL_TO_FITZPATRICK = {
    'CA': ('Type I',    'Very light'),
    'CB': ('Type II',   'Light'),
    'CC': ('Type II-III', 'Light-medium'),
    'CD': ('Type III',  'Medium'),
    'CE': ('Type IV',   'Medium-dark'),
    'CF': ('Type V',    'Dark'),
    'CG': ('Type V',    'Dark'),
    'CH': ('Type VI',   'Very dark'),
    'CI': ('Type VI',   'Very dark'),
    'CJ': ('Type VI',   'Deep brown'),
    'CK': ('Type VI',   'Black'),
}
def get_fitzpatrick_from_label(label):
    return TONE_LABEL_TO_FITZPATRICK.get(label, ('Unknown', 'Unknown'))

# ------------ User Input ------------
def ask_for_images():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        print("Please select the WOUND image (with the wound):")
        wound_image_path = filedialog.askopenfilename(
            title="Select wound image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not wound_image_path:
            raise Exception("No wound image selected.")
        print("Please select a reference SKIN image (clear skin, ideally near the wound):")
        skin_image_path = filedialog.askopenfilename(
            title="Select skin image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not skin_image_path:
            raise Exception("No skin image selected.")
        print(f"Wound image: {wound_image_path}")
        print(f"Skin image: {skin_image_path}")
    except Exception as e:
        print("Automatic file dialog failed. Please enter full paths manually.")
        wound_image_path = input("Enter the full path to the wound image: ").strip()
        skin_image_path = input("Enter the full path to the skin reference image: ").strip()
    if not os.path.exists(wound_image_path):
        raise FileNotFoundError("Wound image not found.")
    if not os.path.exists(skin_image_path):
        raise FileNotFoundError("Skin reference image not found.")
    return wound_image_path, skin_image_path

# ------------ Skin Tone Detection via Stone ------------
def run_stone_on_skin(skin_image_path, report_dir, logger=None):
    result_dir = os.path.join(report_dir, "skin_tone_results")
    os.makedirs(result_dir, exist_ok=True)
    # Remove old result.csv if present
    try:
        os.remove(os.path.join(result_dir, "result.csv"))
    except Exception:
        pass
    try:
        # Run stone classifier (make sure it's installed and in your PATH!)
        subprocess.run([
            "stone", "-i", skin_image_path, "-o", result_dir, "-d"
        ], check=True)
        result_csv = os.path.join(result_dir, "result.csv")
        debug_dir = os.path.join(result_dir, "debug", "color")
        # Try to find the palette/face result debug image
        debug_img = None
        for root, dirs, files in os.walk(debug_dir):
            for f in files:
                if "faces" in root and f.endswith((".png", ".jpg", ".jpeg")):
                    debug_img = os.path.join(root, f)
                    break
        # Try to copy palette result image to our report folder
        if debug_img and os.path.exists(debug_img):
            import shutil
            shutil.copy(debug_img, skin_palette_img)
        # Return CSV info
        if os.path.exists(result_csv):
            df = pd.read_csv(result_csv)
            if len(df) > 0:
                row = df.iloc[0]
                tone_label = row['tone label']
                dominant_color = row['dominant 1']
                percent = float(row['percent 1'])*100
                skin_tone_hex = row['skin tone']
                acc = row['accuracy(0-100)']
                fitz_type, description = get_fitzpatrick_from_label(tone_label)
                text = f"Dominant color: {dominant_color} ({percent:.1f}%)<br/>" \
                       f"Skin tone: {skin_tone_hex} (label: {tone_label}, accuracy: {acc}%)"
                return {
                    "tone_label": tone_label,
                    "fitz_type": fitz_type,
                    "fitz_desc": description,
                    "dominant_color": dominant_color,
                    "percent": percent,
                    "skin_tone_hex": skin_tone_hex,
                    "accuracy": acc,
                    "palette_img": skin_palette_img if os.path.exists(skin_palette_img) else None,
                    "text": text
                }
    except Exception as e:
        if logger:
            logger.warning(f"Skin tone classifier failed: {e}")
    return None

# ------------ Main Pipeline ------------
def main():
    wound_image_path, skin_image_path = ask_for_images()

    # Load Wound Model
    try:
        segmentation_model = tf.keras.models.load_model(segmentation_model_path, compile=False)
        logger.info(f"Loaded segmentation model from {segmentation_model_path}")
    except Exception as e:
        logger.error(f"Failed to load segmentation model: {e}")
        raise

    # ----------- Load Images -----------
    original = cv2.imread(wound_image_path)
    if original is None:
        raise ValueError(f"Could not read wound image: {wound_image_path}")
    img = load_img(wound_image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # ----------- Predict Mask ----------
    pred_mask = segmentation_model.predict(img_input, verbose=0)[0, :, :, 0]
    thresh_val = np.percentile(pred_mask, 80) * 0.8 + 0.2
    pred_mask_bin = (pred_mask > thresh_val).astype(np.uint8) * 255
    mask_resized = cv2.resize(pred_mask_bin, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(mask_path, mask_resized)
    logger.info(f"Saved mask at {mask_path}")

    # ----------- Mask Analysis ---------
    mask = mask_resized
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    report = {}
    current_date = datetime.now().strftime("%Y-%m-%d")
    report["Date"] = current_date
    report["Image"] = os.path.basename(wound_image_path)

    if not contours:
        logger.warning("No wound detected in mask!")
        for k in ["Wound Area (px²)", "Perimeter (px)", "Bounding Box", "Centroid", "Shape Irregularity", "Wound Area (mm²)", "Perimeter (mm)"]:
            report[k] = "N/A"
        report["Condition"] = "No wound detected."
        report["Instructions"] = "Verify image or consult a healthcare provider."
    else:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
        irregularity = (perimeter ** 2) / (4 * math.pi * area + 1e-6)
        area_mm = area * (scale_mm_per_pixel ** 2)
        perimeter_mm = perimeter * scale_mm_per_pixel

        report.update({
            "Wound Area (px²)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2),
            "Bounding Box": f"x={x}, y={y}, w={w}, h={h}",
            "Centroid": f"({cx}, {cy})",
            "Shape Irregularity": round(irregularity, 3),
            "Wound Area (mm²)": round(area_mm, 2),
            "Perimeter (mm)": round(perimeter_mm, 2)
        })
        condition = ""
        if area > 10000:
            condition = "⚠️ Large wound — likely chronic or ulcerative"
        elif area < 3000:
            condition = "✅ Small wound — early or healing"
        else:
            condition = "➖ Moderate wound — monitor size"
        if irregularity > 2.0:
            condition += " | Edge irregularity may indicate inflammation or infection"
        elif irregularity < 1.5:
            condition += " | Regular edges — likely healing"
        report["Condition"] = condition
        instructions = (
            "Consult a doctor if area increases or if red, swollen, or exudative. "
            "Continue monitoring weekly."
        )
        report["Instructions"] = instructions

    # ----------- Generate Overlay -------
    overlay = original.copy()
    color_mask = np.zeros_like(original)
    color_mask[:,:,1] = mask
    overlay = cv2.addWeighted(color_mask, 0.5, overlay, 1-0.5, 0)
    if contours:
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
    scale_bar_px = int(scale_bar_length_mm / scale_mm_per_pixel)
    bar_start = (10, overlay.shape[0] - 20)
    bar_end = (10 + scale_bar_px, overlay.shape[0] - 20)
    cv2.line(overlay, bar_start, bar_end, (255, 255, 255), 3)
    cv2.putText(overlay, f"{scale_bar_length_mm} mm", (bar_start[0], bar_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(overlay_path, overlay)
    logger.info(f"Overlay image saved at {overlay_path}")

    # ----------- Segmentation Heatmap ---
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
    cv2.imwrite(heatmap_path, heatmap)
    logger.info(f"Segmentation heatmap saved at {heatmap_path}")

    # ----------- Area Trend Plot --------
    if os.path.exists(csv_report_path):
        df_hist = pd.read_csv(csv_report_path)
    else:
        df_hist = pd.DataFrame(columns=["Date", "Wound Area (mm²)"])
    df_hist = pd.concat([df_hist, pd.DataFrame([{"Date": current_date, "Wound Area (mm²)": report.get("Wound Area (mm²)", np.nan)}])], ignore_index=True)
    df_hist.to_csv(csv_report_path, index=False)
    if len(df_hist) > 1:
        plt.figure(figsize=(7, 4))
        plt.plot(df_hist["Date"], df_hist["Wound Area (mm²)"], marker="o", color="b")
        plt.title("Wound Area Over Time")
        plt.xlabel("Date")
        plt.ylabel("Area (mm²)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(trend_plot_path)
        plt.close()
        logger.info(f"Wound area trend plot saved at {trend_plot_path}")

    # ----------- Run Stone on Skin Reference -----------
    skin_tone_data = run_stone_on_skin(skin_image_path, report_dir, logger)
    skin_text = "N/A"
    fitz_type = "N/A"
    fitz_desc = ""
    if skin_tone_data:
        skin_text = skin_tone_data["text"]
        fitz_type = skin_tone_data["fitz_type"]
        fitz_desc = skin_tone_data["fitz_desc"]

    # ----------- PDF Report -----------
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_report_path, pagesize=letter)
    elements = []

    elements.append(Paragraph("Wound Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Date:</b> {current_date}", styles['Normal']))
    elements.append(Paragraph(f"<b>Wound Image:</b> {report['Image']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Table
    data = [
        ["Parameter", "Value"],
        ["Wound Area (mm²)", str(report.get("Wound Area (mm²)", "N/A"))],
        ["Perimeter (mm)", str(report.get("Perimeter (mm)", "N/A"))],
        ["Shape Irregularity", str(report.get("Shape Irregularity", "N/A"))],
        ["Skin Tone (AI-Palette)", skin_text],
        ["Fitzpatrick Type", f"{fitz_type} {fitz_desc}"],
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

    # Visuals
    def add_image_section(path, title, size=2*inch):
        if path and os.path.exists(path):
            elements.append(Paragraph(title, styles['Heading2']))
            elements.append(RLImage(path, width=size, height=size))
            elements.append(Spacer(1, 12))

    add_image_section(wound_image_path, "Original Wound Image")
    add_image_section(mask_path, "Predicted Mask")
    add_image_section(overlay_path, "Overlay")
    add_image_section(heatmap_path, "Segmentation Heatmap")
    add_image_section(trend_plot_path, "Area Trend")
    add_image_section(skin_image_path, "Skin Reference Image")
    add_image_section(skin_palette_img, "Skin Tone Palette (AI result)")

    doc.build(elements)
    logger.info(f"PDF report saved at {pdf_report_path}")
    logger.info("Done. See your report for wound, segmentation, and skin tone context.")

if __name__ == "__main__":
    main()
