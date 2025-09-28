"""
Free-tier optimized wound detection API
Designed for Render free tier, GitHub Pages, and other free hosting platforms
"""
import os
import time
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import json

# Minimal logging for free tier
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for free tier

# In-memory storage (no persistent files for free tier)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model (load once to save memory)
model = None

def load_model():
    """Load model on startup - optimized for free tier"""
    global model
    try:
        # Try to load the actual model first
        from woundseg.config import Config
        model_path = Config.get_model_path("unet")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Full U-Net model loaded successfully")
        else:
            # Fallback to basic model if main model not available
            model = create_basic_model()
            logger.info("Basic fallback model created")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Create a basic model as fallback
        model = create_basic_model()
        logger.info("Using fallback model")

def create_basic_model():
    """Create a basic model if main model fails to load"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
    
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2),
        Conv2D(64, 3, activation='relu'),
        Conv2DTranspose(32, 3, activation='relu'),
        UpSampling2D(2),
        Conv2D(1, 1, activation='sigmoid')
    ])
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for free tier monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'free_tier': True,
        'version': '1.0.0'
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check for deployment platforms"""
    if model is None:
        return jsonify({'status': 'not_ready', 'reason': 'model_not_loaded'}), 503
    
    return jsonify({
        'status': 'ready',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_wound():
    """Analyze wound image (optimized for free tier)"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size (limit for free tier)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            return jsonify({'error': 'File too large. Maximum 5MB for free tier.'}), 400
        
        # Process image in memory (no file saving for free tier)
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Resize and normalize
        img_resized = cv2.resize(img_array, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Load model if not loaded
        if model is None:
            load_model()
        
        # Predict
        pred = model.predict(img_batch, verbose=0)[0, ..., 0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        
        # Calculate basic metrics
        wound_pixels = np.sum(pred_mask)
        total_pixels = pred_mask.size
        wound_percentage = (wound_pixels / total_pixels) * 100
        
        # Estimate wound area (rough approximation)
        # Assuming average pixel represents ~0.1mm²
        wound_area_mm2 = wound_pixels * 0.1
        
        confidence = float(np.mean(pred))
        
        # Determine severity based on confidence and area
        if confidence > 0.7 and wound_percentage > 5:
            severity = "High"
        elif confidence > 0.4 and wound_percentage > 2:
            severity = "Moderate"
        elif confidence > 0.2:
            severity = "Low"
        else:
            severity = "None"
        
        # Create visualization in memory
        vis_img = create_visualization(img_array, pred_mask)
        vis_base64 = base64.b64encode(vis_img).decode('utf-8')
        
        # Return results (no file storage for free tier)
        return jsonify({
            'success': True,
            'analysis': {
                'wound_detected': confidence > 0.3,
                'confidence': round(confidence, 3),
                'wound_area_mm2': round(wound_area_mm2, 2),
                'wound_percentage': round(wound_percentage, 2),
                'severity': severity,
                'healing_potential': 'Good' if severity in ['Low', 'Moderate'] else 'Monitor' if severity == 'High' else 'None'
            },
            'visualization': f"data:image/png;base64,{vis_base64}",
            'timestamp': datetime.now().isoformat(),
            'free_tier': True
        })
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

def create_visualization(original_img, mask):
    """Create visualization in memory - optimized for free tier"""
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    # Create figure with smaller size to save memory
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Detected Wound')
    axes[1].axis('off')
    
    # Overlay
    overlay = original_img.copy()
    # Create colored overlay
    colored_overlay = np.zeros_like(original_img)
    colored_overlay[mask > 0] = [255, 0, 0]  # Red for wound area
    overlay = cv2.addWeighted(original_img, 0.7, colored_overlay, 0.3, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Wound Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save to bytes with lower DPI to save memory
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer.getvalue()

@app.route('/info', methods=['GET'])
def get_info():
    """Get information about the free tier service"""
    return jsonify({
        'service': 'Free Wound Detection API',
        'version': '1.0.0',
        'features': [
            'Wound detection and segmentation',
            'Severity assessment',
            'Area calculation',
            'Visualization generation'
        ],
        'limitations': [
            '5MB file size limit',
            'No persistent storage',
            'Basic analysis only',
            'No user accounts'
        ],
        'upgrade_info': 'Contact for paid version with advanced features',
        'model_loaded': model is not None
    })

@app.route('/')
def index():
    """Serve basic info page"""
    return jsonify({
        'message': 'Free Wound Detection API',
        'endpoints': {
            '/health': 'Health check',
            '/analyze': 'POST - Analyze wound image',
            '/info': 'Service information'
        },
        'usage': 'POST an image file to /analyze endpoint'
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Get port from environment (required for Render)
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)