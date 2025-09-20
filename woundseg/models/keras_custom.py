"""
Custom Keras objects for wound segmentation.

This module contains custom loss functions, metrics, and other Keras objects
extracted from wound_medsam.py for centralized registration.
"""

import logging
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

logger = logging.getLogger(__name__)


class FocalTverskyLoss(tf.keras.losses.Loss):
    """
    Focal Tversky Loss for imbalanced segmentation.
    
    This is the FocalTverskyLoss class extracted from wound_medsam.py.
    It's particularly useful for wound segmentation where the wound area
    is typically much smaller than the background.
    
    Args:
        alpha: Weight for false negatives (default: 0.7)
        gamma: Focusing parameter (default: 0.75)
    """
    
    def __init__(self, alpha: float = 0.7, gamma: float = 0.75, name: str = 'focal_tversky_loss'):
        """
        Initialize the Focal Tversky Loss.
        
        Args:
            alpha: Weight for false negatives
            gamma: Focusing parameter
            name: Name of the loss function
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        
        logger.debug(f"FocalTverskyLoss initialized with alpha={alpha}, gamma={gamma}")
    
    def call(self, y_true, y_pred):
        """
        Compute the Focal Tversky Loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Loss value
        """
        # Flatten tensors
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # Calculate True Positives, False Negatives, False Positives
        tp = K.sum(y_true * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)
        
        # Calculate Tversky index
        tversky = (tp + 1e-7) / (tp + self.alpha * fn + (1 - self.alpha) * fp + 1e-7)
        
        # Apply focal weighting
        focal_tversky = K.pow(1 - tversky, self.gamma)
        
        return focal_tversky
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


class IOUScore(tf.keras.metrics.Metric):
    """
    Intersection over Union (IoU) metric for segmentation.
    
    This metric calculates the IoU score between predicted and ground truth masks.
    """
    
    def __init__(self, name: str = 'iou_score', dtype=None):
        """
        Initialize the IoU metric.
        
        Args:
            name: Name of the metric
            dtype: Data type for computations
        """
        super().__init__(name=name, dtype=dtype)
        self.intersection = self.add_weight(name='intersection', initializer='zeros', dtype=dtype)
        self.union = self.add_weight(name='union', initializer='zeros', dtype=dtype)
        
        logger.debug("IOUScore metric initialized")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sample_weight: Optional sample weights
        """
        # Flatten tensors
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # Calculate intersection and union
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        
        # Update state
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        """Calculate the final IoU score."""
        return self.intersection / (self.union + 1e-7)
    
    def reset_state(self):
        """Reset the metric state."""
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class DiceScore(tf.keras.metrics.Metric):
    """
    Dice coefficient metric for segmentation.
    
    This metric calculates the Dice score between predicted and ground truth masks.
    """
    
    def __init__(self, name: str = 'dice_score', dtype=None):
        """
        Initialize the Dice metric.
        
        Args:
            name: Name of the metric
            dtype: Data type for computations
        """
        super().__init__(name=name, dtype=dtype)
        self.intersection = self.add_weight(name='intersection', initializer='zeros', dtype=dtype)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=dtype)
        
        logger.debug("DiceScore metric initialized")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sample_weight: Optional sample weights
        """
        # Flatten tensors
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # Calculate intersection and total
        intersection = K.sum(y_true * y_pred)
        total = K.sum(y_true) + K.sum(y_pred)
        
        # Update state
        self.intersection.assign_add(intersection)
        self.total.assign_add(total)
    
    def result(self):
        """Calculate the final Dice score."""
        return (2.0 * self.intersection) / (self.total + 1e-7)
    
    def reset_state(self):
        """Reset the metric state."""
        self.intersection.assign(0.0)
        self.total.assign(0.0)


def register_custom_objects():
    """
    Register all custom Keras objects.
    
    This function registers the custom loss functions and metrics
    so they can be loaded when loading saved models.
    """
    try:
        # Register custom loss functions
        get_custom_objects().update({
            'FocalTverskyLoss': FocalTverskyLoss,
            'focal_tversky_loss': FocalTverskyLoss()
        })
        
        # Register custom metrics
        get_custom_objects().update({
            'IOUScore': IOUScore,
            'DiceScore': DiceScore,
            'iou_score': IOUScore(),
            'dice_score': DiceScore()
        })
        
        # Try to register segmentation_models metrics if available
        try:
            import segmentation_models as sm
            get_custom_objects().update({
                'iou_score': sm.metrics.IOUScore(),
                'f1_score': sm.metrics.FScore()
            })
            logger.info("Segmentation models metrics registered")
        except ImportError:
            logger.debug("Segmentation models not available, using custom metrics")
        
        logger.info("Custom Keras objects registered successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register custom objects: {e}")
        # Don't raise the exception, just log it as a warning


def get_custom_losses():
    """
    Get dictionary of available custom loss functions.
    
    Returns:
        dict: Dictionary of custom loss functions
    """
    return {
        'focal_tversky': FocalTverskyLoss(),
        'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
        'dice_loss': lambda y_true, y_pred: 1 - DiceScore()(y_true, y_pred).result()
    }


def get_custom_metrics():
    """
    Get dictionary of available custom metrics.
    
    Returns:
        dict: Dictionary of custom metrics
    """
    return {
        'iou_score': IOUScore(),
        'dice_score': DiceScore(),
        'accuracy': tf.keras.metrics.BinaryAccuracy(),
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall()
    }


# Auto-register custom objects when module is imported
try:
    register_custom_objects()
except Exception as e:
    logger.warning(f"Failed to auto-register custom objects: {e}")