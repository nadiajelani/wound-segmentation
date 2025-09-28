#!/usr/bin/env python3
"""
Simple API server for wound detection using simclr_unet_patch_wound.keras
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import tempfile

app = Flask(__name__)
CORS(app)

# Model path
MODEL_PATH = "models/simclr_unet_patch_wound.keras"
model = None

def load_model():
    """Load the simclr model"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze wound image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Load and preprocess image
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Convert to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict with model
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        pred = model.predict(img_batch, verbose=0)[0, ..., 0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        
        # Calculate metrics
        wound_pixels = np.sum(pred_mask)
        total_pixels = pred_mask.size
        wound_percentage = (wound_pixels / total_pixels) * 100
        wound_area_mm2 = wound_pixels * 0.1  # Approximate mm²
        
        # Create visualization
        overlay = img_rgb.copy()
        overlay[pred_mask > 0] = [255, 0, 0]  # Red overlay
        
        # Convert visualization to base64
        vis_img = Image.fromarray(overlay)
        buffer = io.BytesIO()
        vis_img.save(buffer, format='PNG')
        vis_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'analysis': {
                'wound_detected': wound_percentage > 1.0,
                'wound_percentage': round(wound_percentage, 2),
                'wound_area_mm2': round(wound_area_mm2, 2),
                'confidence': round(float(np.mean(pred)), 3),
                'severity': 'High' if wound_percentage > 10 else 'Moderate' if wound_percentage > 5 else 'Low' if wound_percentage > 1 else 'None'
            },
            'visualization': f"data:image/png;base64,{vis_base64}",
            'timestamp': str(np.datetime64('now'))
        })
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Simple Wound Detection API")
    print("=" * 40)
    
    # Load model
    if load_model():
        print("🌐 Starting API server on http://127.0.0.1:8000")
        app.run(host='127.0.0.1', port=8000, debug=False)
    else:
        print("❌ Failed to load model. Exiting.")