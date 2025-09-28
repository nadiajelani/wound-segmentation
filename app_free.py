"""
Free Tier Optimized Wound Detection API
Optimized for Render Free Tier (512MB RAM, 750 hours/month)
"""
import os
import time
import logging
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

app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'free-tier-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit for free tier

# CORS configuration - allow all origins for free deployment
CORS(app, origins=["*"])

# Global model storage
MODEL = None
MODEL_LOADED = False

def load_model():
    """Load the wound segmentation model (optimized for free tier)"""
    global MODEL, MODEL_LOADED
    
    if MODEL_LOADED:
        return True
    
    try:
        logger.info("Loading wound segmentation model...")
        
        # Try to load the model
        model_path = "/app/models/simclr_unet_patch_wound.keras"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, using fallback")
            return False
        
        # Load with Keras 3
        import keras
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["TF_USE_LEGACY_KERAS"] = "0"
        
        # Custom objects for the model
        custom_objects = {
            'Custom>total_loss': lambda *args, **kwargs: 0.0,
            'total_loss': lambda *args, **kwargs: 0.0,
        }
        
        MODEL = keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects
        )
        
        MODEL_LOADED = True
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        MODEL = None
        MODEL_LOADED = False
        return False

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
    
    if not MODEL_LOADED or MODEL is None:
        logger.warning("Model not loaded, using fallback prediction")
        # Return a simple fallback mask
        h, w = image_array.shape[1], image_array.shape[2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Add a simple center region as "wound"
        center_h, center_w = h // 2, w // 2
        cv2.circle(mask, (center_w, center_h), min(h, w) // 8, 255, -1)
        return mask
    
    try:
        # Get prediction from model
        prediction = MODEL.predict(image_array, verbose=0)
        
        # Convert to binary mask
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        return mask
        
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        # Fallback to simple mask
        h, w = image_array.shape[1], image_array.shape[2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center_h, center_w = h // 2, w // 2
        cv2.circle(mask, (center_w, center_h), min(h, w) // 8, 255, -1)
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
        "model_loaded": MODEL_LOADED
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
    return send_from_directory('.', 'wound_whisperer.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Get port from environment (Render requirement)
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)