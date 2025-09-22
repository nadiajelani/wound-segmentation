"""
Configuration management for wound segmentation package.

Centralizes all environment variables, settings, and constants.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize logging early
def _setup_initial_logging():
    """Setup basic logging before full configuration is available."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

_setup_initial_logging()

class Config:
    """
    Centralized configuration management for wound segmentation.
    
    All settings can be overridden via environment variables.
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    PROGRESS_REPORTS_DIR = PROJECT_ROOT / "wound_progress_report"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Model paths
    UNET_WEIGHTS_PATH = os.getenv("UNET_WEIGHTS_PATH", str(MODELS_DIR / "simple_unet_wound.keras"))
    CLASSIFIER_WEIGHTS_PATH = os.getenv("CLASSIFIER_WEIGHTS_PATH", str(MODELS_DIR / "resnet_classifier.h5"))
    
    # Device configuration
    DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, gpu, mps
    MIXED_PRECISION = os.getenv("MIXED_PRECISION", "true").lower() == "true"
    
    # Feature flags
    ENABLE_EXPLAINABILITY = os.getenv("ENABLE_EXPLAINABILITY", "true").lower() == "true"
    ENABLE_VOICE_SUMMARY = os.getenv("ENABLE_VOICE_SUMMARY", "false").lower() == "true"
    ENABLE_SYNTHETIC_DATA = os.getenv("ENABLE_SYNTHETIC_DATA", "false").lower() == "true"
    
    # Model configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "wound_segmentation.log"))
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "5000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    UPLOAD_MAX_SIZE = os.getenv("UPLOAD_MAX_SIZE", "10MB")
    
    # TensorFlow configuration
    TF_THREADS = int(os.getenv("TF_THREADS", "0"))  # 0 = auto
    
    @classmethod
    def setup_tensorflow(cls):
        """Configure TensorFlow settings based on environment."""
        import tensorflow as tf
        
        # Set threading
        if cls.TF_THREADS > 0:
            tf.config.threading.set_intra_op_parallelism_threads(cls.TF_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(cls.TF_THREADS)
        
        # Set mixed precision if enabled
        if cls.MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Set device
        if cls.DEVICE == "cpu":
            tf.config.set_visible_devices([], 'GPU')
        elif cls.DEVICE == "gpu":
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        logging.info(f"TensorFlow configured: device={cls.DEVICE}, mixed_precision={cls.MIXED_PRECISION}")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.MODELS_DIR,
            cls.OUTPUT_DIR,
            cls.REPORTS_DIR,
            cls.PROGRESS_REPORTS_DIR,
            cls.LOGS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured directory exists: {directory}")
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get the full path to a model file."""
        model_paths = {
            "unet": cls.UNET_WEIGHTS_PATH,
            "classifier": cls.CLASSIFIER_WEIGHTS_PATH,
        }
        
        if model_name not in model_paths:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_paths.keys())}")
        
        return model_paths[model_name]
    
    @classmethod
    def is_model_available(cls, model_name: str) -> bool:
        """Check if a model file exists."""
        try:
            model_path = cls.get_model_path(model_name)
            return Path(model_path).exists()
        except ValueError:
            return False
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_root": str(cls.PROJECT_ROOT),
            "models_dir": str(cls.MODELS_DIR),
            "output_dir": str(cls.OUTPUT_DIR),
            "device": cls.DEVICE,
            "mixed_precision": cls.MIXED_PRECISION,
            "enable_medsam": cls.ENABLE_MEDSAM,
            "enable_explainability": cls.ENABLE_EXPLAINABILITY,
            "batch_size": cls.BATCH_SIZE,
            "img_size": cls.IMG_SIZE,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
        }

# Initialize configuration
def init_config():
    """Initialize the configuration system."""
    # Ensure directories exist
    Config.ensure_directories()
    
    # Setup TensorFlow
    Config.setup_tensorflow()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Configuration initialized successfully")

# Auto-initialize when module is imported
if __name__ != "__main__":
    init_config()