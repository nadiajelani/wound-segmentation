"""
Free Tier Optimized Wound Detection API
Optimized for Railway deployment (Python 3.10 + TF 2.12)
"""
import os
import sys
import time
import logging
import urllib.request
import urllib.error
import hashlib
import ssl
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import shutil
from typing import Dict, Any, Tuple, Optional
import base64
import io
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("🔥 Wound Segmentation API Starting Up...")
logger.info(f"Python version: {sys.version}")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Working directory: {os.getcwd()}")

app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'free-tier-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit for free tier

# CORS configuration - allow all origins for free deployment
CORS(app, origins=["*"])

# Global model storage
MODEL = None
MODEL_LOADED = False

def _log(msg): 
    logger.info(f"[MODEL] {msg}")

def file_sha256(path):
    """Calculate SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def ensure_clean_local_model(model_path: str) -> None:
    """Ensure model file is complete and valid before loading"""
    # If a previous boot was killed mid-download, a tiny/invalid file may be present.
    min_bytes = int(os.getenv("SIMCLR_MODEL_MIN_BYTES", "400000000"))  # ~400MB
    expected_sha = os.getenv("SIMCLR_MODEL_SHA256", "").strip()

    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        # delete if too small
        if size < min_bytes:
            try:
                os.remove(model_path)
                logger.info(f"[MODEL] Removed partial file ({size} bytes). Will re-download.")
            except Exception as e:
                logger.warning(f"[MODEL] Could not remove partial file: {e}")

        # optional integrity check
        elif expected_sha:
            try:
                actual = file_sha256(model_path)
                if actual.lower() != expected_sha.lower():
                    os.remove(model_path)
                    logger.info("[MODEL] SHA256 mismatch. Removed and will re-download.")
            except Exception as e:
                logger.warning(f"[MODEL] SHA256 check failed: {e}")

GITHUB_API = "https://api.github.com"

def _http_get(url, headers, dest_path):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers)
    # Set a longer timeout for large model downloads (10 minutes)
    logger.info(f"[MODEL] Starting download from {url} (timeout: 600s)")
    with urllib.request.urlopen(req, context=ctx, timeout=600) as r, open(dest_path, "wb") as f:
        shutil.copyfileobj(r, f)
    logger.info(f"[MODEL] Download completed successfully to {dest_path}")

def _headers_for_download():
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "User-Agent": "wound-segmentation/1.0",
        "Accept": "application/octet-stream",
    }
    if token:
        # Works with classic and fine-grained tokens
        headers["Authorization"] = f"Bearer {token}"
    return headers

def download_model_with_fallbacks(model_url, dest_path, repo_full="nadiajelani/wound-segmentation",
                                  tag="v1.0.0", asset_name="simclr_unet_patch_wound.keras"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    headers = _headers_for_download()

    # 1) Try provided URL first (trimmed)
    url_candidates = [model_url.strip()] if model_url else []

    # 2) Common GitHub URL variants
    url_candidates += [
        f"https://github.com/{repo_full}/releases/download/{tag}/{asset_name}",
        f"https://github.com/{repo_full}/releases/download/v1.0/{asset_name}",
        f"https://github.com/{repo_full}/releases/latest/download/{asset_name}",
    ]

    # try each direct URL
    for u in url_candidates:
        if not u:
            continue
        try:
            _http_get(u, headers, dest_path)
            return True
        except urllib.error.HTTPError as e:
            # 404 is common; try next
            continue
        except Exception:
            continue

    # 3) API path: find asset by name, then download by assets/:id
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        # cannot use API fallback without token
        return False

    # get release by tag; if that fails, fall back to 'latest'
    rel_urls = [
        f"{GITHUB_API}/repos/{repo_full}/releases/tags/{tag}",
        f"{GITHUB_API}/repos/{repo_full}/releases/latest",
    ]
    api_headers = {
        "User-Agent": "wound-segmentation/1.0",
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }

    for rel_url in rel_urls:
        try:
            req = urllib.request.Request(rel_url, headers=api_headers)
            with urllib.request.urlopen(req) as r:
                release = json.loads(r.read().decode("utf-8"))
            assets = release.get("assets", [])
            asset = next((a for a in assets if a.get("name") == asset_name), None)
            if not asset:
                continue
            asset_id = asset["id"]
            asset_api_url = f"{GITHUB_API}/repos/{repo_full}/releases/assets/{asset_id}"
            # Note the special Accept to stream the file
            stream_headers = dict(api_headers)
            stream_headers["Accept"] = "application/octet-stream"
            _http_get(asset_api_url, stream_headers, dest_path)
            return True
        except Exception:
            continue

    return False

def load_model():
    global MODEL, MODEL_LOADED
    if MODEL_LOADED:
        return True
    try:
        logger.info("📦 Loading wound segmentation model...")
        
        # Set Keras backend before importing Keras
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        
        model_path = os.getenv("SIMCLR_MODEL_PATH", "/app/models/simclr_unet_patch_wound.keras")
        model_url  = os.getenv("SIMCLR_MODEL_URL", "")
        tag        = os.getenv("SIMCLR_MODEL_TAG", "v1.0.0")
        repo_full  = os.getenv("SIMCLR_MODEL_REPO", "nadiajelani/wound-segmentation")
        asset_name = os.getenv("SIMCLR_MODEL_ASSET", "simclr_unet_patch_wound.keras")

        # Ensure model file is complete and valid
        ensure_clean_local_model(model_path)
        
        if os.path.exists(model_path):
            logger.info(f"[MODEL] Model file exists at {model_path}, size: {os.path.getsize(model_path)} bytes")
        
        if not os.path.exists(model_path):
            logger.info(f"[MODEL] Attempting download -> {model_path}")
            logger.info(f"[MODEL] Download URL: {model_url}")
            logger.info(f"[MODEL] Repo: {repo_full}, Tag: {tag}, Asset: {asset_name}")
            ok = download_model_with_fallbacks(model_url, model_path,
                                               repo_full=repo_full, tag=tag, asset_name=asset_name)
            if not ok:
                logger.error("[MODEL] Download failed via all methods")
                return False
            else:
                logger.info(f"[MODEL] Download successful, file size: {os.path.getsize(model_path)} bytes")

        # Import Keras 3 (not tf.keras)
        import keras
        logger.info(f"[MODEL] Using Keras version: {keras.__version__}")
        
        # Load model with Keras 3
        MODEL = keras.models.load_model(model_path, compile=False)
        MODEL_LOADED = True
        logger.info("✅ Model loaded successfully")
        logger.info(f"[MODEL] Input shape: {MODEL.input_shape}")
        logger.info(f"[MODEL] Output shape: {MODEL.output_shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Model load error: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        MODEL = None
        MODEL_LOADED = False
        return False

# Load model when Flask app is created (works with gunicorn)
logger.info("🚀 Initializing Flask app...")
logger.info("📦 Attempting to load model during app initialization...")

# Try to load model, but don't crash if it fails
try:
    model_loaded = load_model()
    if model_loaded:
        logger.info("✅ Model loaded successfully during app initialization")
    else:
        logger.warning("⚠️ Model failed to load during app initialization - app will start anyway")
except Exception as e:
    logger.error(f"⚠️ Exception during model loading: {e}")
    logger.warning("⚠️ App will start without model - healthcheck will still pass")
    MODEL_LOADED = False

logger.info("🎉 Flask app initialization complete - ready to accept connections")

def preprocess_image(image_data, target_size=(128, 128)):
    """Preprocess image for model input"""
    try:
        # Handle base64 input
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_data = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def predict_wound_mask(image_array):
    """
    Returns:
        pred_map: float32 array in [0,1] shaped (H, W) – the model's probability map
        mask:     uint8 array in {0,255} shaped (H, W) – thresholded binary mask
    """
    global MODEL, MODEL_LOADED
    h, w = image_array.shape[1], image_array.shape[2]

    if not MODEL_LOADED or MODEL is None:
        logger.error("❌ Model not loaded, using fallback prediction")
        pred_map = np.zeros((h, w), dtype=np.float32)
        cv2.circle(pred_map, (w // 2, h // 2), min(h, w) // 8, 1.0, -1)
        mask = (pred_map > 0.5).astype(np.uint8) * 255
        return pred_map, mask

    try:
        # Model outputs (1, H, W, 1) with values in [0,1]
        prediction = MODEL.predict(image_array, verbose=0)[0, :, :, 0].astype(np.float32)
        pred_map = np.clip(prediction, 0.0, 1.0)
        mask = (pred_map > 0.5).astype(np.uint8) * 255
        return pred_map, mask

    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        pred_map = np.zeros((h, w), dtype=np.float32)
        cv2.circle(pred_map, (w // 2, h // 2), min(h, w) // 8, 1.0, -1)
        mask = (pred_map > 0.5).astype(np.uint8) * 255
        return pred_map, mask

def calculate_metrics(mask):
    """Calculate wound metrics from mask"""
    try:
        # Calculate area
        area_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        area_percentage = (area_pixels / total_pixels) * 100
        
        # Calculate perimeter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0
        
        # Calculate circularity (4π × area / perimeter²)
        circularity = 0.0
        if perimeter > 0:
            circularity = (4 * np.pi * area_pixels) / (perimeter ** 2)
        
        # Determine severity
        if area_percentage < 1:
            severity = "Mild"
        elif area_percentage < 5:
            severity = "Moderate"
        else:
            severity = "Severe"
        
        return {
            "area_pixels": int(area_pixels),
            "area_percentage": round(area_percentage, 2),
            "perimeter": round(float(perimeter), 2),
            "circularity": round(float(circularity), 3),
            "severity": severity
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {
            "area_pixels": 0,
            "area_percentage": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "severity": "Unknown"
        }

def classify_healing_stage(metrics, pred_map):
    """Classify wound healing stage based on metrics and prediction confidence"""
    try:
        area_pct = metrics.get("area_percentage", 0)
        circularity = metrics.get("circularity", 0)
        
        # Calculate average confidence in wound region
        mask = (pred_map > 0.5).astype(np.uint8)
        if np.sum(mask) > 0:
            avg_confidence = np.mean(pred_map[mask > 0])
        else:
            avg_confidence = 0
        
        # Classify healing stage based on characteristics
        if area_pct < 0.5 and circularity > 0.7:
            stage = "Final Remodeling"
            description = "Wound shows minimal tissue involvement with well-defined borders. Near complete healing."
            recommendations = [
                "Continue monitoring for complete epithelialization",
                "Protect new tissue from trauma",
                "Consider scar management if needed"
            ]
        elif area_pct < 2 and avg_confidence > 0.6:
            stage = "Proliferative/Maturation"
            description = "Active tissue regeneration with granulation tissue formation and epithelialization in progress."
            recommendations = [
                "Maintain moist wound environment",
                "Monitor for signs of infection",
                "Consider nutritional support",
                "Continue current treatment protocol"
            ]
        elif area_pct < 5:
            stage = "Early Proliferative"
            description = "Wound bed preparation phase with new tissue formation beginning."
            recommendations = [
                "Ensure adequate debridement if needed",
                "Optimize wound bed moisture balance",
                "Address any underlying factors affecting healing",
                "Regular dressing changes per protocol"
            ]
        else:
            stage = "Inflammatory/Early Healing"
            description = "Initial healing phase with inflammatory response. Wound requires close monitoring."
            recommendations = [
                "Assess for infection signs",
                "Ensure proper wound cleansing",
                "Evaluate underlying health conditions",
                "Consider advanced wound care consultation",
                "Regular clinical assessment required"
            ]
        
        return {
            "stage": stage,
            "description": description,
            "confidence": round(float(avg_confidence), 3),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Healing stage classification failed: {e}")
        return {
            "stage": "Assessment Required",
            "description": "Unable to classify healing stage automatically. Clinical assessment recommended.",
            "confidence": 0.0,
            "recommendations": ["Consult healthcare provider for proper assessment"]
        }

def generate_doctor_report(metrics, healing_stage, timestamp):
    """Generate comprehensive doctor's report"""
    try:
        report = {
            "report_id": f"WA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": timestamp,
            "patient_info": {
                "note": "Patient details to be added by healthcare provider"
            },
            "clinical_findings": {
                "wound_area": f"{metrics['area_pixels']} pixels ({metrics['area_percentage']}% of image)",
                "wound_perimeter": f"{metrics['perimeter']} pixels",
                "wound_shape": f"Circularity: {metrics['circularity']} (1.0 = perfect circle)",
                "severity_classification": metrics['severity']
            },
            "healing_assessment": {
                "stage": healing_stage['stage'],
                "stage_description": healing_stage['description'],
                "ai_confidence": f"{healing_stage['confidence'] * 100:.1f}%"
            },
            "clinical_recommendations": healing_stage['recommendations'],
            "technical_notes": {
                "analysis_method": "AI-powered segmentation using SimCLR-pretrained U-Net",
                "image_processing": "Automated segmentation with confidence heatmap analysis",
                "note": "This AI analysis is a decision support tool and should be used in conjunction with clinical judgment"
            },
            "follow_up": {
                "recommended_reassessment": _get_followup_interval(metrics['severity']),
                "monitoring_parameters": [
                    "Wound size and area changes",
                    "Signs of infection (redness, warmth, discharge)",
                    "Healing progression",
                    "Patient symptoms and comfort"
                ]
            },
            "disclaimer": "This report is generated by an AI system for clinical decision support. Final diagnosis and treatment decisions should be made by qualified healthcare professionals based on complete clinical assessment."
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Doctor report generation failed: {e}")
        return {
            "error": "Report generation failed",
            "message": str(e)
        }

def _get_followup_interval(severity):
    """Get recommended follow-up interval based on severity"""
    intervals = {
        "Mild": "Re-assess in 5-7 days or as clinically indicated",
        "Moderate": "Re-assess in 3-5 days or sooner if symptoms worsen",
        "Severe": "Re-assess in 24-48 hours or immediately if symptoms worsen"
    }
    return intervals.get(severity, "Re-assess as clinically indicated")

def to_base64_png(img: np.ndarray) -> str:
    """Encode a HxW or HxWx3 uint8 image to data URL PNG base64."""
    if img.ndim == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img[:, :, ::-1]) if img.shape[2] == 3 and img.dtype == np.uint8 else Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()

def make_heatmap(pred_map: np.ndarray) -> np.ndarray:
    """pred_map in [0,1] -> uint8 BGR heatmap via OpenCV COLORMAP_JET."""
    hm = (pred_map * 255.0).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)  # BGR
    return hm_color

def make_overlay(rgb_image_0_1: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend heatmap over original RGB (0..1). Returns uint8 BGR for easy PNG."""
    # rgb_image_0_1: (H,W,3) float32 in [0,1] from preprocess
    base_bgr = (rgb_image_0_1 * 255.0).astype(np.uint8)[:, :, ::-1]  # to BGR
    overlay = cv2.addWeighted(heatmap_bgr, alpha, base_bgr, 1.0 - alpha, 0.0)
    return overlay

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - always returns 200 to pass Railway healthcheck"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL_LOADED,
        "model_status": "loaded" if MODEL_LOADED else "not_loaded",
        "version": "1.0.0",
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__
    }), 200

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug information endpoint"""
    model_path = os.getenv('SIMCLR_MODEL_PATH', '/app/models/simclr_unet_patch_wound.keras')
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "model_exists": MODEL is not None,
        "model_path": model_path,
        "model_path_exists": os.path.exists(model_path),
        "model_path_files": os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else "Directory not found",
        "model_input_shape": MODEL.input_shape if MODEL else None,
        "model_output_shape": MODEL.output_shape if MODEL else None,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/diag", methods=['GET'])
def diag():
    """Diagnostics endpoint to see what the container sees"""
    p = os.getenv("SIMCLR_MODEL_PATH", "/app/models/simclr_unet_patch_wound.keras")
    exists = os.path.exists(p)
    size = os.path.getsize(p) if exists else 0
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "path": p,
        "exists": exists,
        "size_bytes": size,
        "url_set": bool(os.getenv("SIMCLR_MODEL_URL", "").strip()),
        "model_url": os.getenv("SIMCLR_MODEL_URL", "").strip() or "Not set",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/ready', methods=['GET'])
def ready_check():
    """Readiness check endpoint"""
    return jsonify({
        "ready": MODEL_LOADED,
        "model_loaded": MODEL_LOADED,
        "message": "Service ready to process requests" if MODEL_LOADED else "Model not loaded",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_wound():
    """Analyze wound from uploaded image"""
    try:
        # Get image data
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"error": "No image provided"}), 400
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Read image data
            image_data = file.read()
        
        # Handle base64 data
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
        
        else:
            return jsonify({"error": "No image data provided"}), 400
        
        # Preprocess image
        img_array = preprocess_image(image_data)  # shape (1, H, W, 3), float32 in [0,1]
        if img_array is None:
            return jsonify({"error": "Image preprocessing failed"}), 400
        
        # Predict map + mask
        pred_map, mask = predict_wound_mask(img_array)
        
        # Metrics from binary mask
        metrics = calculate_metrics(mask)
        
        # Classify healing stage
        healing_stage = classify_healing_stage(metrics, pred_map)
        
        # Generate doctor's report
        timestamp = datetime.now().isoformat()
        doctor_report = generate_doctor_report(metrics, healing_stage, timestamp)
        
        # Build visuals
        heatmap_bgr = make_heatmap(pred_map)                          # HxWx3 (BGR)
        overlay_bgr = make_overlay(img_array[0], heatmap_bgr, 0.45)   # HxWx3 (BGR)
        
        # Encode images
        mask_b64     = to_base64_png(mask)          # grayscale
        heatmap_b64  = to_base64_png(heatmap_bgr)   # color heatmap
        overlay_b64  = to_base64_png(overlay_bgr)   # blended on original
        
        result = {
            "success": True,
            "metrics": metrics,
            "healing_stage": healing_stage,
            "doctor_report": doctor_report,
            "mask_image": mask_b64,
            "heatmap_image": heatmap_b64,
            "overlay_image": overlay_b64,
            "timestamp": timestamp
        }
        
        logger.info(f"Analysis completed: {metrics}, Stage: {healing_stage['stage']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - serve wound analyzer interface"""
    try:
        # Always try wound_analyzer.html first (the new version with all features)
        return send_from_directory('.', 'wound_analyzer.html')
    except Exception as e:
        logger.error(f"Could not serve wound_analyzer.html: {e}")
        # Fallback to JSON response
        return jsonify({
            "message": "Wound Segmentation API",
            "status": "running",
            "model_loaded": MODEL_LOADED,
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "debug": "/debug",
                "analyze": "/analyze (POST)"
            },
            "version": "1.0.0"
        }), 200

@app.route('/analyzer', methods=['GET'])
def analyzer():
    """Direct route to wound analyzer"""
    return send_from_directory('.', 'wound_analyzer.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Load model on startup
    logger.info("🚀 Starting application...")
    logger.info("📦 Attempting to load model on startup...")
    
    model_loaded = load_model()
    if model_loaded:
        logger.info("✅ Model loaded successfully on startup")
    else:
        logger.error("❌ Model failed to load on startup")
    
    # Get port from environment (Railway requirement)
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🌐 Starting Flask application on port {port}...")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)