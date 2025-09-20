"""
U-Net model implementation for wound segmentation.

This module provides the U-Net architecture and loading functionality
extracted from the main wound_medsam.py file.
"""

import logging
import os
from typing import Tuple, Optional
import tensorflow as tf
import numpy as np
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)


class UNetProvider:
    """
    U-Net model provider for wound segmentation.
    
    This class handles U-Net model building, loading, and prediction.
    It provides the U-Net architecture and loading logic extracted from wound_medsam.py.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the U-Net provider.
        
        Args:
            model_path: Optional path to the U-Net model file.
                       If None, uses the path from Config.
        """
        self.model_path = model_path or Config.get_model_path("unet")
        self._model: Optional[tf.keras.Model] = None
        self.input_shape = (128, 128, 3)  # Standard input shape for U-Net
        
        logger.info(f"UNetProvider initialized with model path: {self.model_path}")
    
    @property
    def model(self) -> tf.keras.Model:
        """
        Get the U-Net model.
        
        Returns:
            tf.keras.Model: The loaded U-Net model
            
        Raises:
            FileNotFoundError: If the model file is not found
            Exception: If there's an error loading the model
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self) -> tf.keras.Model:
        """
        Load the U-Net model from file.
        
        Returns:
            tf.keras.Model: The loaded U-Net model
            
        Raises:
            FileNotFoundError: If the model file is not found
            Exception: If there's an error loading the model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"U-Net model not found: {self.model_path}")
        
        try:
            logger.info(f"Loading U-Net model from: {self.model_path}")
            model = tf.keras.models.load_model(self.model_path, compile=False)
            
            logger.info(f"U-Net model loaded successfully. Input shape: {model.input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load U-Net model: {e}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int, int] = (128, 128, 3), dropout_rate: float = 0.1) -> tf.keras.Model:
        """
        Build a U-Net model from scratch.
        
        This is the build_unet() function extracted from wound_medsam.py.
        
        Args:
            input_shape: Input shape for the model (height, width, channels)
            dropout_rate: Dropout rate for regularization
            
        Returns:
            tf.keras.Model: The built U-Net model
        """
        logger.info(f"Building U-Net model with input shape: {input_shape}, dropout_rate: {dropout_rate}")
        
        # Input layer
        inputs = tf.keras.Input(shape=input_shape)
        
        # Encoder (Contracting Path)
        # Block 1
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bottleneck
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = tf.keras.layers.Dropout(dropout_rate)(conv4, training=True)
        
        # Decoder (Expanding Path)
        # Block 5
        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop4)
        up5 = tf.keras.layers.Concatenate()([up5, conv3])
        conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
        conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
        
        # Block 6
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = tf.keras.layers.Concatenate()([up6, conv2])
        conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
        conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
        
        # Block 7
        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = tf.keras.layers.Concatenate()([up7, conv1])
        conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
        conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
        
        # Output layer
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        logger.info(f"U-Net model built successfully. Total parameters: {model.count_params()}")
        return model
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation mask using the U-Net model.
        
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
            pred_mask = self.model.predict(image, verbose=0)[0, :, :, 0]
            
            logger.debug(f"U-Net prediction completed. Mask shape: {pred_mask.shape}")
            return pred_mask
            
        except Exception as e:
            logger.error(f"Failed to predict with U-Net: {e}")
            raise
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Preprocess an image for U-Net prediction.
        
        Args:
            image_path: Path to the input image
            target_size: Target size for the image
            
        Returns:
            np.ndarray: Preprocessed image array
            
        Raises:
            FileNotFoundError: If image file is not found
            Exception: If preprocessing fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            
            # Load and resize image
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            
            logger.debug(f"Image preprocessed. Shape: {img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def postprocess_mask(self, mask: np.ndarray, threshold_percentile: float = 80.0) -> np.ndarray:
        """
        Postprocess the predicted mask.
        
        This applies the same postprocessing logic used in test_wound_progress.py.
        
        Args:
            mask: Raw prediction mask
            threshold_percentile: Percentile for adaptive thresholding
            
        Returns:
            np.ndarray: Postprocessed binary mask
        """
        try:
            # Apply adaptive thresholding (same logic as test_wound_progress.py)
            thresh_val = np.percentile(mask, threshold_percentile) * 0.8 + 0.2
            mask_bin = (mask > thresh_val).astype(np.uint8) * 255
            
            logger.debug(f"Mask postprocessed. Threshold: {thresh_val:.3f}")
            return mask_bin
            
        except Exception as e:
            logger.error(f"Failed to postprocess mask: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the U-Net model.
        
        Returns:
            dict: Model information including path, input/output shapes
        """
        if self._model is None:
            return {"status": "not_loaded", "path": self.model_path}
        
        return {
            "status": "loaded",
            "path": self.model_path,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "parameters": self.model.count_params()
        }


# Convenience function for backward compatibility
def build_unet(input_shape: Tuple[int, int, int] = (128, 128, 3), dropout_rate: float = 0.1) -> tf.keras.Model:
    """
    Build a U-Net model (backward compatibility function).
    
    This function maintains compatibility with the original build_unet() function
    from wound_medsam.py.
    
    Args:
        input_shape: Input shape for the model
        dropout_rate: Dropout rate for regularization
        
    Returns:
        tf.keras.Model: The built U-Net model
    """
    provider = UNetProvider()
    return provider.build_model(input_shape, dropout_rate)