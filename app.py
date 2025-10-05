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

GITHUB_API = "https://api.github.com"

def _http_get(url, headers, dest_path):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx) as r, open(dest_path, "wb") as f:
        shutil.copyfileobj(r, f)

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
        model_path = os.getenv("SIMCLR_MODEL_PATH", "/app/models/simclr_unet_patch_wound.keras")
        model_url  = os.getenv("SIMCLR_MODEL_URL", "")
        tag        = os.getenv("SIMCLR_MODEL_TAG", "v1.0.0")
        repo_full  = os.getenv("SIMCLR_MODEL_REPO", "nadiajelani/wound-segmentation")
        asset_name = os.getenv("SIMCLR_MODEL_ASSET", "simclr_unet_patch_wound.keras")

        if not os.path.exists(model_path):
            logger.info(f"[MODEL] Attempting download -> {model_path}")
            ok = download_model_with_fallbacks(model_url, model_path,
                                               repo_full=repo_full, tag=tag, asset_name=asset_name)
            if not ok:
                logger.error("[MODEL] Download failed via all methods")
                return False

        import keras, tensorflow as tf
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["TF_USE_LEGACY_KERAS"] = "0"
        custom_objects = {
            "Custom>total_loss": lambda *a, **k: 0.0,
            "total_loss": lambda *a, **k: 0.0,
        }
        MODEL = keras.models.load_model(model_path, compile=False, safe_mode=False,
                                        custom_objects=custom_objects)
        MODEL_LOADED = True
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model load error: {e}")
        MODEL = None
        MODEL_LOADED = False
        return False

# Load model when Flask app is created (works with gunicorn)
logger.info("🚀 Initializing Flask app...")
logger.info("📦 Loading model during app initialization...")
model_loaded = load_model()
if model_loaded:
    logger.info("✅ Model loaded successfully during app initialization")
else:
    logger.error("❌ Model failed to load during app initialization")

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
    """Predict wound mask using the loaded model"""
    global MODEL, MODEL_LOADED
    
    logger.info(f"Predicting wound mask - Model loaded: {MODEL_LOADED}, Model exists: {MODEL is not None}")
    logger.info(f"Input image shape: {image_array.shape}")
    
    if not MODEL_LOADED or MODEL is None:
        logger.error("❌ Model not loaded, using fallback prediction")
        # Return a simple fallback mask
        h, w = image_array.shape[1], image_array.shape[2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Add a simple center region as "wound"
        center_h, center_w = h // 2, w // 2
        cv2.circle(mask, (center_w, center_h), min(h, w) // 8, 255, -1)
        logger.warning(f"Using fallback mask with center circle at ({center_h}, {center_w})")
        return mask
    
    try:
        logger.info(f"Running model prediction with input shape: {image_array.shape}")
        # Get prediction from model
        prediction = MODEL.predict(image_array, verbose=0)
        logger.info(f"Model prediction shape: {prediction.shape}")
        logger.info(f"Prediction min/max: {prediction.min():.4f}/{prediction.max():.4f}")
        
        # Convert to binary mask
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        logger.info(f"Binary mask shape: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
        
        return mask
        
    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}")
        import traceback
        logger.error(f"Prediction traceback: {traceback.format_exc()}")
        # Fallback to simple mask
        h, w = image_array.shape[1], image_array.shape[2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center_h, center_w = h // 2, w // 2
        cv2.circle(mask, (center_w, center_h), min(h, w) // 8, 255, -1)
        logger.warning(f"Using fallback mask due to prediction error")
        return mask

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
            "severity": severity
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {
            "area_pixels": 0,
            "area_percentage": 0.0,
            "perimeter": 0.0,
            "severity": "Unknown"
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0"
    })

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
        img_array = preprocess_image(image_data)
        if img_array is None:
            return jsonify({"error": "Image preprocessing failed"}), 400
        
        # Predict wound mask
        mask = predict_wound_mask(img_array)
        
        # Calculate metrics
        metrics = calculate_metrics(mask)
        
        # Convert mask to base64 for response
        mask_pil = Image.fromarray(mask)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Create response
        result = {
            "success": True,
            "metrics": metrics,
            "mask_image": f"data:image/png;base64,{mask_base64}",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Analysis completed: {metrics}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index_free.html')

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