"""
Explainability service for model interpretability.

This module provides explainability features including Grad-CAM
and SHAP for understanding model decisions.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from ..config import Config
from ..models import get_model_provider
from .storage import get_storage_service

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    Explainability service for model interpretability.
    
    Provides Grad-CAM and SHAP-based explanations for model decisions.
    """
    
    def __init__(self, model_provider=None, storage_service=None):
        """
        Initialize the explainability service.
        
        Args:
            model_provider: Model provider instance (optional)
            storage_service: Storage service instance (optional)
        """
        self.model_provider = model_provider or get_model_provider()
        self.storage_service = storage_service or get_storage_service()
        self.enabled = Config.ENABLE_EXPLAINABILITY
        
        # Initialize explainability libraries
        self.gradcam_available = False
        self.shap_available = False
        
        if self.enabled:
            self._initialize_explainability_libraries()
        else:
            logger.info("ExplainabilityService initialized (disabled by config)")
    
    def _initialize_explainability_libraries(self):
        """Initialize explainability libraries with lazy imports."""
        try:
            # Try to import TensorFlow for Grad-CAM
            import tensorflow as tf
            self.tf = tf
            self.gradcam_available = True
            logger.info("Grad-CAM support available")
        except ImportError:
            logger.warning("TensorFlow not available - Grad-CAM disabled")
        
        try:
            # Try to import SHAP (lazy import to avoid dependency issues)
            import shap
            self.shap = shap
            self.shap_available = True
            logger.info("SHAP support available")
        except ImportError:
            logger.warning("SHAP not available - SHAP explanations disabled")
    
    def is_available(self) -> bool:
        """
        Check if explainability service is available.
        
        Returns:
            bool: True if explainability service is available and enabled
        """
        return self.enabled and (self.gradcam_available or self.shap_available)
    
    def generate_gradcam_explanation(self, image: np.ndarray, 
                                   layer_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM explanation for the model's prediction.
        
        Args:
            image: Input image as numpy array
            layer_name: Name of the layer to use for Grad-CAM (optional)
            
        Returns:
            Grad-CAM heatmap as numpy array, or None if generation failed
        """
        if not self.enabled or not self.gradcam_available:
            logger.warning("Grad-CAM not available - skipping explanation generation")
            return None
        
        try:
            # Get the model
            model = self.model_provider.unet_model
            
            # Find the target layer
            if layer_name is None:
                # Use the last convolutional layer
                target_layer = self._find_last_conv_layer(model)
            else:
                target_layer = model.get_layer(layer_name)
            
            if target_layer is None:
                logger.error("Could not find suitable layer for Grad-CAM")
                return None
            
            # Prepare image for prediction
            if len(image.shape) == 3:
                input_image = np.expand_dims(image, axis=0)
            else:
                input_image = image
            
            # Generate Grad-CAM
            gradcam = self._compute_gradcam(model, input_image, target_layer)
            
            logger.info("Grad-CAM explanation generated successfully")
            return gradcam
            
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM explanation: {e}")
            return None
    
    def generate_shap_explanation(self, image: np.ndarray, 
                                background_images: Optional[List[np.ndarray]] = None) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP explanation for the model's prediction.
        
        Args:
            image: Input image as numpy array
            background_images: Optional background images for SHAP (optional)
            
        Returns:
            SHAP explanation dictionary, or None if generation failed
        """
        if not self.enabled or not self.shap_available:
            logger.warning("SHAP not available - skipping explanation generation")
            return None
        
        try:
            # Get the model
            model = self.model_provider.unet_model
            
            # Prepare background data
            if background_images is None:
                # Create simple background (zeros or mean)
                background = np.zeros_like(image)
            else:
                background = np.mean(background_images, axis=0)
            
            # Create SHAP explainer
            explainer = self.shap.Explainer(model, background)
            
            # Generate SHAP values
            shap_values = explainer(image)
            
            # Process SHAP values
            explanation = {
                'shap_values': shap_values.values,
                'base_value': shap_values.base_values,
                'data': shap_values.data,
                'feature_names': [f'pixel_{i}' for i in range(image.size)]
            }
            
            logger.info("SHAP explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            return None
    
    def _find_last_conv_layer(self, model) -> Optional[Any]:
        """Find the last convolutional layer in the model."""
        try:
            conv_layers = []
            for layer in model.layers:
                if 'conv' in layer.name.lower() or 'Conv2D' in str(type(layer)):
                    conv_layers.append(layer)
            
            if conv_layers:
                return conv_layers[-1]
            else:
                # Fallback to the last layer
                return model.layers[-1]
                
        except Exception as e:
            logger.error(f"Failed to find last conv layer: {e}")
            return None
    
    def _compute_gradcam(self, model, image, target_layer) -> np.ndarray:
        """Compute Grad-CAM for the given model, image, and layer."""
        try:
            # Create a model that outputs the target layer and the final prediction
            grad_model = self.tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )
            
            # Compute gradients
            with self.tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                class_channel = predictions[0, :, :, 0]  # Assuming single channel output
            
            # Compute gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = self.tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Multiply each channel by its corresponding gradient
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., self.tf.newaxis]
            heatmap = self.tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = self.tf.maximum(heatmap, 0) / self.tf.math.reduce_max(heatmap)
            
            # Resize to original image size
            heatmap = self.tf.image.resize(
                heatmap[..., self.tf.newaxis], 
                (image.shape[1], image.shape[2])
            )
            
            return heatmap.numpy()
            
        except Exception as e:
            logger.error(f"Failed to compute Grad-CAM: {e}")
            return None
    
    def create_explanation_visualization(self, image: np.ndarray, 
                                       gradcam: Optional[np.ndarray] = None,
                                       shap_explanation: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Create visualization combining original image with explanations.
        
        Args:
            image: Original input image
            gradcam: Grad-CAM heatmap (optional)
            shap_explanation: SHAP explanation (optional)
            
        Returns:
            Combined visualization as numpy array, or None if creation failed
        """
        try:
            # Start with original image
            vis_image = image.copy()
            
            # Add Grad-CAM overlay if available
            if gradcam is not None:
                gradcam_resized = cv2.resize(gradcam, (image.shape[1], image.shape[0]))
                gradcam_colored = cv2.applyColorMap(
                    (gradcam_resized * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                vis_image = cv2.addWeighted(vis_image, 0.6, gradcam_colored, 0.4, 0)
            
            # Add SHAP overlay if available
            if shap_explanation is not None:
                shap_values = shap_explanation['shap_values']
                if len(shap_values.shape) == 4:  # Multi-channel
                    shap_values = np.mean(shap_values, axis=-1)
                
                shap_resized = cv2.resize(shap_values, (image.shape[1], image.shape[0]))
                shap_colored = cv2.applyColorMap(
                    (shap_resized * 255).astype(np.uint8), 
                    cv2.COLORMAP_VIRIDIS
                )
                vis_image = cv2.addWeighted(vis_image, 0.7, shap_colored, 0.3, 0)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Failed to create explanation visualization: {e}")
            return None
    
    def save_explanation(self, explanation_data: Dict[str, Any], 
                        filename_prefix: str) -> Dict[str, Path]:
        """
        Save explanation data to storage.
        
        Args:
            explanation_data: Dictionary containing explanation data
            filename_prefix: Prefix for saved files
            
        Returns:
            Dictionary mapping explanation types to file paths
        """
        saved_files = {}
        
        try:
            # Save Grad-CAM if available
            if 'gradcam' in explanation_data and explanation_data['gradcam'] is not None:
                gradcam = explanation_data['gradcam']
                gradcam_filename = f"{filename_prefix}_gradcam.png"
                gradcam_path = self.storage_service.save_image(
                    gradcam.tobytes(), 'visualizations', gradcam_filename
                )
                saved_files['gradcam'] = gradcam_path
            
            # Save SHAP explanation if available
            if 'shap' in explanation_data and explanation_data['shap'] is not None:
                shap_data = explanation_data['shap']
                shap_filename = f"{filename_prefix}_shap.npy"
                shap_path = self.storage_service.save_image(
                    np.array(shap_data).tobytes(), 'visualizations', shap_filename
                )
                saved_files['shap'] = shap_path
            
            # Save combined visualization if available
            if 'visualization' in explanation_data and explanation_data['visualization'] is not None:
                vis = explanation_data['visualization']
                vis_filename = f"{filename_prefix}_explanation.png"
                vis_path = self.storage_service.save_image(
                    vis.tobytes(), 'visualizations', vis_filename
                )
                saved_files['visualization'] = vis_path
            
            logger.info(f"Explanation saved: {list(saved_files.keys())}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save explanation: {e}")
            return {}
    
    def get_explanation_summary(self, explanation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of explanation data.
        
        Args:
            explanation_data: Dictionary containing explanation data
            
        Returns:
            Dictionary with explanation summary
        """
        summary = {
            'gradcam_available': 'gradcam' in explanation_data and explanation_data['gradcam'] is not None,
            'shap_available': 'shap' in explanation_data and explanation_data['shap'] is not None,
            'visualization_available': 'visualization' in explanation_data and explanation_data['visualization'] is not None
        }
        
        # Add Grad-CAM summary
        if summary['gradcam_available']:
            gradcam = explanation_data['gradcam']
            summary['gradcam_summary'] = {
                'shape': gradcam.shape,
                'min_value': float(np.min(gradcam)),
                'max_value': float(np.max(gradcam)),
                'mean_value': float(np.mean(gradcam))
            }
        
        # Add SHAP summary
        if summary['shap_available']:
            shap_data = explanation_data['shap']
            summary['shap_summary'] = {
                'keys': list(shap_data.keys()) if isinstance(shap_data, dict) else 'array',
                'type': type(shap_data).__name__
            }
        
        return summary
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the explainability service.
        
        Returns:
            dict: Service information
        """
        return {
            'enabled': self.enabled,
            'available': self.is_available(),
            'gradcam_available': self.gradcam_available,
            'shap_available': self.shap_available,
            'tensorflow_available': hasattr(self, 'tf'),
            'shap_imported': hasattr(self, 'shap')
        }


# Global explainability service instance
_explainability_service: Optional[ExplainabilityService] = None


def get_explainability_service() -> ExplainabilityService:
    """
    Get the global explainability service instance.
    
    Returns:
        ExplainabilityService: The global explainability service instance
    """
    global _explainability_service
    if _explainability_service is None:
        _explainability_service = ExplainabilityService()
    return _explainability_service


def reset_explainability_service() -> None:
    """Reset the global explainability service (useful for testing)."""
    global _explainability_service
    _explainability_service = None
    logger.info("Explainability service reset")