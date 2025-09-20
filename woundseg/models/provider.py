"""
Model Provider for wound segmentation package.

Centralized model management with lazy loading for optimal memory usage.
"""

import logging
import os
from typing import Optional, Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)


class ModelProvider:
    """
    Central model provider with lazy loading for wound segmentation models.
    
    This class manages the loading and access to all ML models used in the system.
    Models are loaded only when first accessed (lazy loading) to optimize memory usage.
    """
    
    def __init__(self):
        """Initialize the model provider."""
        self._unet_model: Optional[tf.keras.Model] = None
        self._classifier_model: Optional[tf.keras.Model] = None
        self._models_info: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ModelProvider initialized")
    
    @property
    def unet_model(self) -> tf.keras.Model:
        """
        Get the U-Net segmentation model.
        
        Returns:
            tf.keras.Model: The loaded U-Net model
            
        Raises:
            FileNotFoundError: If the U-Net model file is not found
            Exception: If there's an error loading the model
        """
        if self._unet_model is None:
            self._unet_model = self._load_unet_model()
        return self._unet_model
    
    @property
    def classifier_model(self) -> Optional[tf.keras.Model]:
        """
        Get the ResNet classifier model (optional).
        
        Returns:
            tf.keras.Model or None: The loaded classifier model, or None if not available
        """
        if self._classifier_model is None:
            self._classifier_model = self._load_classifier_model()
        return self._classifier_model
    
    def _load_unet_model(self) -> tf.keras.Model:
        """
        Load the U-Net segmentation model.
        
        Returns:
            tf.keras.Model: The loaded U-Net model
            
        Raises:
            FileNotFoundError: If the model file is not found
            Exception: If there's an error loading the model
        """
        model_path = Config.get_model_path("unet")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"U-Net model not found: {model_path}")
        
        try:
            logger.info(f"Loading U-Net model from: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Store model info
            self._models_info["unet"] = {
                "path": model_path,
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "loaded_at": tf.timestamp()
            }
            
            logger.info(f"U-Net model loaded successfully. Input shape: {model.input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load U-Net model: {e}")
            raise
    
    def _load_classifier_model(self) -> Optional[tf.keras.Model]:
        """
        Load the ResNet classifier model (optional).
        
        Returns:
            tf.keras.Model or None: The loaded classifier model, or None if not available
        """
        try:
            model_path = Config.get_model_path("classifier")
            
            if not os.path.exists(model_path):
                logger.warning(f"Classifier model not found: {model_path}")
                return None
            
            logger.info(f"Loading classifier model from: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Store model info
            self._models_info["classifier"] = {
                "path": model_path,
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "loaded_at": tf.timestamp()
            }
            
            logger.info(f"Classifier model loaded successfully. Input shape: {model.input_shape}")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load classifier model: {e}")
            return None
    
    def predict_wound_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Predict wound segmentation mask using U-Net model.
        
        Args:
            image: Input image as numpy array (should be preprocessed)
            
        Returns:
            np.ndarray: Predicted segmentation mask
            
        Raises:
            ValueError: If image format is invalid
            Exception: If prediction fails
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        try:
            # Ensure image has correct shape for U-Net (batch dimension)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Get prediction
            pred_mask = self.unet_model.predict(image, verbose=0)[0, :, :, 0]
            
            logger.debug(f"U-Net prediction completed. Mask shape: {pred_mask.shape}")
            return pred_mask
            
        except Exception as e:
            logger.error(f"Failed to predict wound mask: {e}")
            raise
    
    def predict_wound_classification(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Predict wound classification using ResNet model.
        
        Args:
            image: Input image as numpy array (should be preprocessed)
            
        Returns:
            tuple: (is_wound: bool, confidence: float)
            
        Raises:
            ValueError: If image format is invalid
            Exception: If prediction fails
        """
        if self.classifier_model is None:
            raise RuntimeError("Classifier model not available")
        
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        try:
            # Ensure image has correct shape for classifier (batch dimension)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Get prediction
            prediction = self.classifier_model.predict(image, verbose=0)[0][0]
            is_wound = prediction > 0.5
            confidence = prediction if is_wound else 1 - prediction
            
            logger.debug(f"Classification prediction: is_wound={is_wound}, confidence={confidence:.3f}")
            return is_wound, confidence
            
        except Exception as e:
            logger.error(f"Failed to predict wound classification: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.
        
        Args:
            model_name: Name of the model ('unet' or 'classifier')
            
        Returns:
            Dict with model information or None if not found
        """
        return self._models_info.get(model_name)
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model to free memory.
        
        Args:
            model_name: Name of the model to unload ('unet' or 'classifier')
        """
        if model_name == "unet" and self._unet_model is not None:
            del self._unet_model
            self._unet_model = None
            logger.info("U-Net model unloaded")
        elif model_name == "classifier" and self._classifier_model is not None:
            del self._classifier_model
            self._classifier_model = None
            logger.info("Classifier model unloaded")
        else:
            logger.warning(f"Model '{model_name}' not found or already unloaded")
    
    def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        self.unload_model("unet")
        self.unload_model("classifier")
        logger.info("All models unloaded")


# Global model provider instance
_model_provider: Optional[ModelProvider] = None


def get_model_provider() -> ModelProvider:
    """
    Get the global model provider instance.
    
    Returns:
        ModelProvider: The global model provider instance
    """
    global _model_provider
    if _model_provider is None:
        _model_provider = ModelProvider()
    return _model_provider


def reset_model_provider() -> None:
    """Reset the global model provider (useful for testing)."""
    global _model_provider
    if _model_provider is not None:
        _model_provider.unload_all_models()
        _model_provider = None
    logger.info("Model provider reset")