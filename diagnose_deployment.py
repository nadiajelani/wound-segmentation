#!/usr/bin/env python3
"""
Railway Deployment Diagnostic Script
Run this locally to test model loading before deployment
"""

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check Python and TensorFlow environment"""
    logger.info("=== Environment Check ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Keras version: {tf.keras.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
        logger.info(f"Total memory: {memory.total / (1024**3):.2f} GB")
    except ImportError:
        logger.warning("psutil not available - cannot check memory")

def check_model_file():
    """Check if model file exists and is accessible"""
    logger.info("=== Model File Check ===")
    
    model_path = "models/simclr_unet_patch_wound.keras"
    logger.info(f"Looking for model at: {model_path}")
    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024**2)  # MB
        logger.info(f"Model file size: {file_size:.2f} MB")
        
        # Check if file is readable
        try:
            with open(model_path, 'rb') as f:
                f.read(1024)  # Read first 1KB
            logger.info("✅ Model file is readable")
        except Exception as e:
            logger.error(f"❌ Model file not readable: {e}")
    else:
        logger.error("❌ Model file not found")
        # List available files
        models_dir = "models"
        if os.path.exists(models_dir):
            logger.info(f"Available files in {models_dir}/:")
            for f in os.listdir(models_dir):
                logger.info(f"  - {f}")
        else:
            logger.error(f"❌ Models directory not found")

def test_model_loading():
    """Test model loading with detailed error reporting"""
    logger.info("=== Model Loading Test ===")
    
    model_path = "models/simclr_unet_patch_wound.keras"
    
    if not os.path.exists(model_path):
        logger.error("❌ Model file not found - skipping loading test")
        return False
    
    try:
        logger.info("Attempting to load model...")
        
        custom_objects = {
            'Custom>total_loss': lambda *args, **kwargs: 0.0,
            'total_loss': lambda *args, **kwargs: 0.0,
        }
        
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_prediction():
    """Test model prediction with dummy data"""
    logger.info("=== Prediction Test ===")
    
    try:
        # Create dummy image data
        dummy_image = np.random.random((1, 128, 128, 3)).astype(np.float32)
        logger.info(f"Created dummy image with shape: {dummy_image.shape}")
        
        # Load model
        model_path = "models/simclr_unet_patch_wound.keras"
        custom_objects = {
            'Custom>total_loss': lambda *args, **kwargs: 0.0,
            'total_loss': lambda *args, **kwargs: 0.0,
        }
        
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(dummy_image, verbose=0)
        logger.info(f"✅ Prediction successful")
        logger.info(f"Prediction shape: {prediction.shape}")
        logger.info(f"Prediction range: {prediction.min():.4f} to {prediction.max():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Railway Deployment Diagnostic")
    logger.info("=" * 50)
    
    check_environment()
    print()
    check_model_file()
    print()
    
    model_loaded = test_model_loading()
    print()
    
    if model_loaded:
        test_prediction()
    
    logger.info("=" * 50)
    logger.info("🏁 Diagnostic complete")
