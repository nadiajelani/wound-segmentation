"""
Production-ready wound detection API with security and rate limiting
"""
import os
import time
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import shutil
from typing import Dict, Any, Tuple, Optional

# Import your existing modules
from wound_medsam import build_unet, predict_healing_potential, load_medsam_model, medsam_segment
from woundseg.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB

# CORS configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=[f"{os.getenv('RATE_LIMIT_PER_MINUTE', '10')} per minute"]
)

# File upload configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/uploads')
REPORT_FOLDER = os.getenv('REPORT_FOLDER', '/app/reports')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs('/app/logs', exist_ok=True)

# Global model variables
unet_model = None
medsam_model = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load AI models on startup"""
    global unet_model, medsam_model
    
    try:
        # Load U-Net model
        unet_model_path = os.getenv('UNET_MODEL_PATH', Config.get_model_path("unet"))
        unet_model = build_unet(input_shape=(128, 128, 3))
        unet_model.load_weights(unet_model_path)
        logger.info("U-Net model loaded successfully")
        
        # Load MedSAM model (optional)
        medsam_model_path = os.getenv('MEDSAM_MODEL_PATH')
        if medsam_model_path and os.path.exists(medsam_model_path):
            medsam_model = load_medsam_model(medsam_model_path)
            logger.info("MedSAM model loaded successfully")
        else:
            logger.warning("MedSAM model not found, using U-Net only")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def cleanup_old_files():
    """Clean up old files to prevent disk space issues"""
    cleanup_hours = int(os.getenv('FILE_CLEANUP_HOURS', '24'))
    cutoff_time = datetime.now() - timedelta(hours=cleanup_hours)
    
    for folder in [UPLOAD_FOLDER, REPORT_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove {filename}: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': unet_model is not None
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint"""
    if unet_model is None:
        return jsonify({'status': 'not_ready', 'reason': 'models_not_loaded'}), 503
    
    return jsonify({
        'status': 'ready',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_and_analyze():
    """Upload and analyze wound image"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        
        # Process the image
        result = process_wound_image(filepath, filename)
        
        # Clean up uploaded file after processing
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove uploaded file {filepath}: {str(e)}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def process_wound_image(image_path: str, filename: str) -> Dict[str, Any]:
    """Process wound image and return analysis results"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_resized = tf.image.resize(img_rgb, (128, 128))[None, ...]
        
        # Predict with U-Net
        pred = unet_model.predict(img_resized, verbose=0)[0, ..., 0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        pred_mask_resized = tf.image.resize(
            pred_mask[..., None], 
            img.shape[:2], 
            method='nearest'
        ).numpy().squeeze().astype(np.uint8)
        
        # Analyze wound
        severity, healing_potential, wound_area_mm2 = predict_healing_potential(
            pred_mask_resized, img_rgb
        )
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"wound_report_{timestamp}.pdf"
        report_path = os.path.join(REPORT_FOLDER, report_filename)
        
        # Create visualization
        vis_filename = f"wound_vis_{timestamp}.png"
        vis_path = os.path.join(REPORT_FOLDER, vis_filename)
        create_visualization(img_rgb, pred_mask_resized, vis_path)
        
        # Generate PDF report
        generate_pdf_report(
            img_rgb, pred_mask_resized, severity, healing_potential, 
            wound_area_mm2, report_path, vis_path
        )
        
        # Return results
        return {
            'success': True,
            'filename': filename,
            'analysis': {
                'severity': severity,
                'healing_potential': healing_potential,
                'wound_area_mm2': float(wound_area_mm2),
                'confidence': float(np.mean(pred_mask))
            },
            'files': {
                'report_url': f'/report/{report_filename}',
                'visualization_url': f'/report/{vis_filename}'
            },
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"Error in process_wound_image: {str(e)}")
        raise

def create_visualization(image: np.ndarray, mask: np.ndarray, output_path: str):
    """Create visualization of wound analysis"""
    import matplotlib.pyplot as plt
    
    img_rgb = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Detected Wound Area")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(contour_img)
    plt.title("Wound Outline")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_pdf_report(image, mask, severity, healing_potential, wound_area, report_path, vis_path):
    """Generate PDF report"""
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Wound Analysis Report", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Severity: {severity}", 0, 1)
    pdf.cell(0, 10, f"Healing Potential: {healing_potential}", 0, 1)
    pdf.cell(0, 10, f"Wound Area: {wound_area:.2f} mm²", 0, 1)
    pdf.ln(10)
    
    pdf.cell(0, 10, "Visualization:", 0, 1)
    pdf.image(vis_path, x=10, w=190)
    
    pdf.ln(10)
    pdf.cell(0, 10, "Disclaimer: This analysis is for informational purposes only.", 0, 1)
    pdf.cell(0, 10, "Please consult a healthcare professional for medical advice.", 0, 1)
    
    pdf.output(report_path)

@app.route('/report/<filename>')
def serve_report(filename):
    """Serve generated reports"""
    return send_from_directory(REPORT_FOLDER, filename)

@app.route('/')
def index():
    """Serve the main application"""
    return send_from_directory('.', 'wound_whisperer.html')

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Start cleanup task
    import threading
    def periodic_cleanup():
        while True:
            time.sleep(3600)  # Run every hour
            cleanup_old_files()
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Run the app
    app.run(host='0.0.0.0', port=8080, debug=False)