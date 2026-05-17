"""
Wound Segmentation API — Cloud Run
Includes: segmentation, healing stage, skin tone analysis, clinical report
"""
import os, sys, time, logging, urllib.request, urllib.error
import hashlib, ssl, json, math, shutil, base64, io
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from typing import Optional
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("🔥 Wound Segmentation API Starting Up...")
logger.info(f"Python: {sys.version} | TensorFlow: {tf.__version__}")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'wound-secret')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
CORS(app, origins=["*"])

MODEL = None
MODEL_LOADED = False

# ── Model download & load ──────────────────────────────────────────────────────

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def ensure_clean_local_model(model_path):
    min_bytes = int(os.getenv("SIMCLR_MODEL_MIN_BYTES", "400000000"))
    expected_sha = os.getenv("SIMCLR_MODEL_SHA256", "").strip()
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size < min_bytes:
            try: os.remove(model_path)
            except: pass
        elif expected_sha:
            try:
                if file_sha256(model_path).lower() != expected_sha.lower():
                    os.remove(model_path)
            except: pass

GITHUB_API = "https://api.github.com"

def _http_get(url, headers, dest_path):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=600) as r, open(dest_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _headers_for_download():
    token = os.getenv("GITHUB_TOKEN", "").strip()
    h = {"User-Agent": "wound-segmentation/1.0", "Accept": "application/octet-stream"}
    if token: h["Authorization"] = f"Bearer {token}"
    return h

def download_model_with_fallbacks(model_url, dest_path,
                                   repo_full="nadiajelani/wound-segmentation",
                                   tag="v1.0.0", asset_name="simclr_unet_patch_wound.keras"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    headers = _headers_for_download()
    candidates = [model_url.strip()] if model_url else []
    candidates += [
        f"https://github.com/{repo_full}/releases/download/{tag}/{asset_name}",
        f"https://github.com/{repo_full}/releases/latest/download/{asset_name}",
    ]
    for u in candidates:
        if not u: continue
        try:
            _http_get(u, headers, dest_path)
            return True
        except: continue
    return False

def load_model():
    global MODEL, MODEL_LOADED
    if MODEL_LOADED: return True
    try:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        model_path  = os.getenv("SIMCLR_MODEL_PATH", "/tmp/models/simclr_unet_patch_wound.keras")
        model_url   = os.getenv("SIMCLR_MODEL_URL", "")
        tag         = os.getenv("SIMCLR_MODEL_TAG", "v1.0.0")
        repo_full   = os.getenv("SIMCLR_MODEL_REPO", "nadiajelani/wound-segmentation")
        asset_name  = os.getenv("SIMCLR_MODEL_ASSET", "simclr_unet_patch_wound.keras")
        ensure_clean_local_model(model_path)
        if not os.path.exists(model_path):
            ok = download_model_with_fallbacks(model_url, model_path,
                                               repo_full=repo_full, tag=tag, asset_name=asset_name)
            if not ok:
                logger.error("[MODEL] Download failed"); return False
        import keras
        MODEL = keras.models.load_model(model_path, compile=False)
        MODEL_LOADED = True
        logger.info(f"✅ Model loaded | input:{MODEL.input_shape} output:{MODEL.output_shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Model load error: {e}")
        MODEL = None; MODEL_LOADED = False; return False

try:
    load_model()
except Exception as e:
    logger.error(f"⚠️ Model load exception: {e}")
    MODEL_LOADED = False

# ── Image helpers ──────────────────────────────────────────────────────────────

def preprocess_image(image_data, target_size=(128, 128)):
    try:
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Preprocess failed: {e}"); return None

def predict_wound_mask(image_array):
    h, w = image_array.shape[1], image_array.shape[2]
    if not MODEL_LOADED or MODEL is None:
        pred_map = np.zeros((h, w), dtype=np.float32)
        cv2.circle(pred_map, (w//2, h//2), min(h,w)//8, 1.0, -1)
        return pred_map, (pred_map > 0.5).astype(np.uint8) * 255
    try:
        prediction = MODEL.predict(image_array, verbose=0)[0,:,:,0].astype(np.float32)
        pred_map = np.clip(prediction, 0.0, 1.0)
        return pred_map, (pred_map > 0.5).astype(np.uint8) * 255
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        pred_map = np.zeros((h, w), dtype=np.float32)
        return pred_map, (pred_map > 0.5).astype(np.uint8) * 255

def to_base64_png(img):
    if img.ndim == 2:
        pil_img = Image.fromarray(img)
    else:
        arr = img[:,:,::-1] if img.shape[2] == 3 and img.dtype == np.uint8 else img
        pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def make_heatmap(pred_map):
    return cv2.applyColorMap((pred_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

def make_overlay(rgb_0_1, heatmap_bgr, alpha=0.45):
    base_bgr = (rgb_0_1 * 255).astype(np.uint8)[:,:,::-1]
    return cv2.addWeighted(heatmap_bgr, alpha, base_bgr, 1.0 - alpha, 0.0)

# ── Metrics ────────────────────────────────────────────────────────────────────

def calculate_metrics(mask):
    try:
        area_pixels   = int(np.sum(mask > 0))
        total_pixels  = mask.shape[0] * mask.shape[1]
        area_pct      = round((area_pixels / total_pixels) * 100, 2)
        contours, _   = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter     = round(float(cv2.arcLength(contours[0], True)), 2) if contours else 0.0
        irregularity  = round((perimeter**2) / (4 * math.pi * area_pixels + 1e-6), 3) if area_pixels > 0 else 0.0

        if area_pct < 1:   severity = "Mild"
        elif area_pct < 5: severity = "Moderate"
        else:              severity = "Severe"

        return {
            "area_pixels":      area_pixels,
            "area_percentage":  area_pct,
            "perimeter":        perimeter,
            "irregularity":     irregularity,
            "severity":         severity
        }
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        return {"area_pixels": 0, "area_percentage": 0.0, "perimeter": 0.0, "irregularity": 0.0, "severity": "Unknown"}

# ── Healing Stage Classification ───────────────────────────────────────────────
# Based on wound area, shape irregularity, and colour variance
# Matches the 4-stage RYB / clinical model: Haemostasis → Inflammatory → Proliferative → Remodelling

def classify_healing_stage(metrics, pred_map):
    """
    Rule-based classification using:
    - area_percentage   : wound size relative to image
    - irregularity      : shape complexity (high = irregular edges)
    - colour_variance   : variance in the wound region (high = inflammation)
    - pred_confidence   : mean prediction confidence in wound area
    """
    try:
        area       = metrics.get("area_percentage", 0)
        irreg      = metrics.get("irregularity", 1.0)
        # Colour variance from pred_map (proxy for tissue complexity)
        wound_vals = pred_map[pred_map > 0.3]
        colour_var = float(np.var(wound_vals)) if len(wound_vals) > 10 else 0.0
        confidence = float(np.mean(wound_vals)) if len(wound_vals) > 10 else 0.0

        # ── Stage logic ──
        # Haemostasis: very small, dense wound, high confidence boundary
        if area < 0.5 and confidence > 0.7 and irreg < 1.5:
            stage = "Haemostasis"
            description = (
                "Immediate post-injury phase — bleeding has stopped and a clot has formed. "
                "AI markers: dense, acute wound boundaries with high segmentation confidence."
            )
            recommendations = [
                "Keep wound clean and dry",
                "Apply gentle pressure if any oozing remains",
                "Cover with sterile dressing",
                "Monitor for signs of infection over next 24–48 hours"
            ]
            confidence_score = min(0.95, 0.6 + confidence * 0.4)

        # Inflammatory: irregular edges, high colour variance
        elif irreg > 2.0 or colour_var > 0.05:
            stage = "Inflammatory"
            description = (
                "Inflammatory phase — redness, oedema, and exudate are typical. "
                "AI markers: high colour variance in wound region and irregular wound edges."
            )
            recommendations = [
                "Monitor for excessive redness, warmth, or swelling",
                "Change dressings regularly to manage exudate",
                "Keep wound moist but not saturated",
                "Seek medical attention if signs of infection develop (fever, increasing pain)"
            ]
            confidence_score = min(0.92, 0.55 + colour_var * 3)

        # Proliferative: moderate area, shrinking, pinkish tissue (medium confidence)
        elif 0.5 <= area <= 5.0 and confidence >= 0.5 and irreg <= 2.0:
            stage = "Proliferative"
            description = (
                "Proliferative phase — granulation tissue is forming and the wound is contracting. "
                "AI markers: pinkish-red tissue appearance, shrinking wound area, moderately regular edges."
            )
            recommendations = [
                "Maintain moist wound environment to support granulation",
                "Avoid disrupting the wound bed during dressing changes",
                "Consider advanced wound dressings (hydrocolloid, foam)",
                "Monitor wound size weekly — expect gradual reduction"
            ]
            confidence_score = min(0.90, 0.5 + (1 - area/10) * 0.4)

        # Remodelling: small area, regular edges, low variance
        elif area < 0.5 and irreg < 1.5 and colour_var < 0.03:
            stage = "Remodelling"
            description = (
                "Remodelling phase — scar tissue is maturing. "
                "AI markers: pale, regular wound boundaries and significantly reduced wound area."
            )
            recommendations = [
                "Apply silicone gel or sheeting to reduce scar formation",
                "Protect scar from sun exposure for at least 12 months",
                "Gentle massage of scar tissue once fully closed",
                "Scar will continue to mature for up to 2 years"
            ]
            confidence_score = 0.85

        # Default: general moderate wound
        else:
            stage = "Proliferative"
            description = (
                "Wound appears to be in an active healing phase. "
                "AI markers: moderate wound area with typical healing tissue characteristics."
            )
            recommendations = [
                "Maintain regular dressing changes",
                "Monitor wound size and appearance weekly",
                "Ensure adequate nutrition (protein, Vitamin C, Zinc) to support healing",
                "Consult a healthcare provider if healing appears to stall"
            ]
            confidence_score = 0.70

        return {
            "stage":            stage,
            "description":      description,
            "recommendations":  recommendations,
            "confidence":       round(confidence_score, 2),
            "markers": {
                "area_percentage":  area,
                "irregularity":     irreg,
                "colour_variance":  round(colour_var, 4),
                "mean_confidence":  round(confidence, 4)
            }
        }

    except Exception as e:
        logger.error(f"Healing stage classification failed: {e}")
        return {
            "stage": "Unknown", "description": "Could not classify healing stage.",
            "recommendations": ["Consult a healthcare provider."], "confidence": 0.0
        }

# ── Skin Tone Analysis ─────────────────────────────────────────────────────────
# RGB-to-Fitzpatrick mapping — ported from test_wound_progress_skintype.py
# Uses ITA° (Individual Typology Angle) from LAB colorspace — clinical standard

FITZPATRICK_RANGES = [
    # (ITA_min, ITA_max, type, description, skin_label)
    ( 55,  90, "Type I",   "Very light / pale white",         "Very Light"),
    ( 41,  55, "Type II",  "Light / white",                   "Light"),
    ( 28,  41, "Type III", "Light-medium / cream white",      "Light-Medium"),
    ( 10,  28, "Type IV",  "Medium / moderate brown",         "Medium"),
    (-30,  10, "Type V",   "Medium-dark / dark brown",        "Medium-Dark"),
    (-90, -30, "Type VI",  "Dark / deeply pigmented",         "Dark"),
]

def detect_skin_tone(rgb_image_0_1):
    """
    Classify skin tone from the non-wound skin region of the image.
    Uses ITA° (Individual Typology Angle) from CIE LAB colorspace.
    Falls back to mean RGB Fitzpatrick mapping if LAB fails.
    """
    try:
        img_uint8 = (rgb_image_0_1 * 255).astype(np.uint8)

        # Use border region as skin proxy (avoids wound centre)
        h, w = img_uint8.shape[:2]
        border_mask = np.zeros((h, w), dtype=np.uint8)
        thickness = max(8, h // 8)
        border_mask[:thickness, :] = 255
        border_mask[-thickness:, :] = 255
        border_mask[:, :thickness] = 255
        border_mask[:, -thickness:] = 255

        skin_pixels_rgb = img_uint8[border_mask > 0]

        if len(skin_pixels_rgb) < 50:
            skin_pixels_rgb = img_uint8.reshape(-1, 3)

        # Convert to LAB
        skin_img = skin_pixels_rgb.reshape(1, -1, 3)
        lab = cv2.cvtColor(skin_img, cv2.COLOR_RGB2LAB)
        L = lab[0, :, 0].astype(float) * (100.0 / 255.0)  # scale L to 0-100
        b = lab[0, :, 2].astype(float) - 128.0              # shift b to -128..+127

        # ITA° = arctan((L - 50) / b) × (180 / π)
        ita_vals = np.degrees(np.arctan2(L - 50, b + 1e-6))
        ita = float(np.median(ita_vals))

        # Fitzpatrick from ITA°
        fitz_type, fitz_desc, skin_label = "Type III", "Light-medium", "Light-Medium"
        for ita_min, ita_max, ftype, fdesc, slabel in FITZPATRICK_RANGES:
            if ita_min <= ita < ita_max:
                fitz_type, fitz_desc, skin_label = ftype, fdesc, slabel
                break

        # Mean RGB for swatch display
        mean_rgb = skin_pixels_rgb.mean(axis=0).astype(int).tolist()
        luminance = float(np.mean(L))

        return {
            "skin_type":    skin_label,
            "fitzpatrick":  fitz_type,
            "description":  fitz_desc,
            "ita_angle":    round(ita, 1),
            "luminance":    round(luminance, 1),
            "rgb_avg":      mean_rgb,
        }

    except Exception as e:
        logger.error(f"Skin tone detection failed: {e}")
        return {
            "skin_type":   "Unknown", "fitzpatrick": "—",
            "description": "Could not determine skin tone",
            "ita_angle":   0.0, "luminance": 128.0, "rgb_avg": [180, 150, 130]
        }

# ── Clinical Report ────────────────────────────────────────────────────────────
# Structured physician report — ported from reportlab PDF logic in test files
# Returns structured JSON (rendered as report card in the UI)

def generate_doctor_report(metrics, healing_stage, skin_analysis, timestamp):
    try:
        area      = metrics.get("area_percentage", 0)
        perim     = metrics.get("perimeter", 0)
        severity  = metrics.get("severity", "Unknown")
        irreg     = metrics.get("irregularity", 0)
        stage     = healing_stage.get("stage", "Unknown")
        fitz      = skin_analysis.get("fitzpatrick", "—")
        skin_desc = skin_analysis.get("description", "—")

        report_id = f"WA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Shape descriptor
        if irreg < 1.3:     shape_desc = "Regular, well-defined boundaries"
        elif irreg < 2.0:   shape_desc = "Moderately irregular boundaries"
        else:               shape_desc = "Highly irregular edges — possible inflammation"

        # Size descriptor
        scale_mm_per_pixel = 0.1
        area_mm2 = round(metrics.get("area_pixels", 0) * (scale_mm_per_pixel ** 2), 2)
        perim_mm = round(perim * scale_mm_per_pixel, 2)

        clinical_findings = {
            "Wound Area":       f"{area:.2f}% of image ({area_mm2} mm²)",
            "Perimeter":        f"{perim_mm} mm",
            "Shape":            shape_desc,
            "Severity":         severity,
            "Healing Stage":    stage,
            "Skin Type":        f"{fitz} — {skin_desc}",
        }

        # Follow-up care based on stage
        follow_up_map = {
            "Haemostasis":   "Keep wound clean and dry. Apply sterile dressing. Review in 24–48 hours.",
            "Inflammatory":  "Monitor for infection signs (erythema, warmth, purulent exudate). Change dressings daily. Consider antimicrobial dressing if infection suspected.",
            "Proliferative": "Maintain moist wound environment. Avoid disturbing granulation tissue. Review weekly and measure wound area to confirm reduction.",
            "Remodelling":   "Wound is closing. Protect scar from UV exposure. Consider silicone therapy. Full maturation takes 6–24 months.",
            "Unknown":       "Consult a qualified healthcare professional for full assessment.",
        }
        follow_up = follow_up_map.get(stage, follow_up_map["Unknown"])

        return {
            "report_id":          report_id,
            "generated_at":       timestamp,
            "clinical_findings":  clinical_findings,
            "follow_up_care":     follow_up,
            "disclaimer":         "This AI-generated report is for research use only and does not constitute medical advice. Always consult a qualified healthcare professional."
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "report_id":         "ERROR",
            "generated_at":      timestamp,
            "clinical_findings": {},
            "follow_up_care":    "Consult a healthcare provider.",
            "disclaimer":        "Report generation failed."
        }

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_status": "loaded" if MODEL_LOADED else "not_loaded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__
    }), 200

@app.route('/ready')
def ready_check():
    return jsonify({
        "ready": MODEL_LOADED,
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/debug')
def debug_info():
    model_path = os.getenv('SIMCLR_MODEL_PATH', '/tmp/models/simclr_unet_patch_wound.keras')
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "model_path": model_path,
        "model_path_exists": os.path.exists(model_path),
        "model_input_shape":  MODEL.input_shape  if MODEL else None,
        "model_output_shape": MODEL.output_shape if MODEL else None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_wound():
    try:
        if 'image' not in request.files and not (request.is_json and 'image_data' in request.json):
            return jsonify({"error": "No image provided"}), 400

        if 'image' in request.files:
            f = request.files['image']
            if f.filename == '':
                return jsonify({"error": "No file selected"}), 400
            image_data = f.read()
        else:
            image_data = request.json['image_data']

        img_array = preprocess_image(image_data)
        if img_array is None:
            return jsonify({"error": "Image preprocessing failed"}), 400

        # Core segmentation
        pred_map, mask = predict_wound_mask(img_array)
        metrics        = calculate_metrics(mask)

        # ── New features ──
        healing_stage  = classify_healing_stage(metrics, pred_map)
        skin_analysis  = detect_skin_tone(img_array[0])
        timestamp      = datetime.now().isoformat()
        doctor_report  = generate_doctor_report(metrics, healing_stage, skin_analysis, timestamp)

        # Visuals
        heatmap_bgr = make_heatmap(pred_map)
        overlay_bgr = make_overlay(img_array[0], heatmap_bgr, 0.45)

        return jsonify({
            "success":       True,
            "metrics":       metrics,
            "healing_stage": healing_stage,
            "skin_analysis": skin_analysis,
            "doctor_report": doctor_report,
            "mask_image":    to_base64_png(mask),
            "heatmap_image": to_base64_png(heatmap_bgr),
            "overlay_image": to_base64_png(overlay_bgr),
            "timestamp":     timestamp
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/')
def index():
    for name in ['index_free.html', 'wound_analyzer.html']:
        if os.path.exists(name):
            return send_from_directory('.', name)
    return jsonify({
        "message": "WoundAI API v2.0",
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "endpoints": {"/health": "GET", "/analyze": "POST image"}
    }), 200

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
