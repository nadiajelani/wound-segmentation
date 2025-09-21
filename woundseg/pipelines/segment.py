"""
Segmentation pipeline for wound segmentation.

This module handles segmentation prediction, test-time augmentation,
mask combination, and uncertainty estimation.
"""

import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import albumentations as A
import tensorflow as tf

from ..models import ModelProvider
from ..types import SegmentationResult

logger = logging.getLogger(__name__)


class SegmentationPipeline:
    """
    Segmentation pipeline for wound segmentation.
    
    This class handles segmentation prediction, test-time augmentation,
    mask combination, and uncertainty estimation.
    """
    
    def __init__(self, model_provider: Optional[ModelProvider] = None):
        """
        Initialize the segmentation pipeline.
        
        Args:
            model_provider: Model provider instance (optional)
        """
        self.model_provider = model_provider
        logger.info("SegmentationPipeline initialized")
    
    def predict_mask(self, image: np.ndarray, use_tta: bool = False, 
                    num_tta_samples: int = 4) -> SegmentationResult:
        """
        Predict segmentation mask for an image.
        
        Args:
            image: Preprocessed input image
            use_tta: Whether to use test-time augmentation
            num_tta_samples: Number of TTA samples to use
            
        Returns:
            SegmentationResult: Segmentation result with mask and confidence
        """
        try:
            if use_tta:
                mask, uncertainty = self.test_time_augmentation(image, num_tta_samples)
                confidence = 1.0 - np.mean(uncertainty)  # Convert uncertainty to confidence
            else:
                mask = self._single_prediction(image)
                confidence = 0.8  # Default confidence for single prediction
            
            result = SegmentationResult(
                mask=mask,
                confidence=confidence,
                method="unet_tta" if use_tta else "unet"
            )
            
            logger.debug(f"Segmentation completed: method={result.method}, confidence={confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to predict mask: {e}")
            raise
    
    def _single_prediction(self, image: np.ndarray) -> np.ndarray:
        """
        Single prediction using U-Net model.
        
        Args:
            image: Preprocessed input image
            
        Returns:
            np.ndarray: Predicted mask
        """
        if self.model_provider is None:
            raise RuntimeError("Model provider not available")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get prediction
        pred_mask = self.model_provider.predict_wound_mask(image)
        
        return pred_mask
    
    def test_time_augmentation(self, image: np.ndarray, num_samples: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test-time augmentation for improved prediction.
        
        This is the test_time_augmentation function extracted from wound_medsam.py.
        
        Args:
            image: Input image
            num_samples: Number of augmentation samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        try:
            if self.model_provider is None:
                raise RuntimeError("Model provider not available")
            
            predictions = []
            
            # Define augmentations (same as wound_medsam.py)
            augmentations = [
                lambda x: x,  # Original
                lambda x: np.fliplr(x),  # Horizontal flip
                lambda x: np.flipud(x),  # Vertical flip
                lambda x: A.Rotate(limit=90, p=1)(image=x)['image']  # 90-degree rotation
            ]
            
            # Apply augmentations and collect predictions
            for i, aug in enumerate(augmentations[:num_samples]):
                aug_img = aug(image)
                pred = self._single_prediction(aug_img)
                
                # Reverse augmentation for prediction
                if i == 1:  # Horizontal flip
                    pred = np.fliplr(pred)
                elif i == 2:  # Vertical flip
                    pred = np.flipud(pred)
                elif i == 3:  # 90-degree rotation
                    pred = A.Rotate(limit=-90, p=1)(image=pred)['image']
                
                predictions.append(pred)
            
            # Calculate mean prediction and uncertainty
            predictions_array = np.array(predictions)
            mean_pred = np.mean(predictions_array, axis=0)
            uncertainty = np.var(predictions_array, axis=0)
            
            logger.debug(f"TTA completed: {num_samples} samples, uncertainty={np.mean(uncertainty):.3f}")
            return mean_pred, uncertainty
            
        except Exception as e:
            logger.error(f"Failed to perform TTA: {e}")
            raise
    
    def combine_masks(self, mask1: np.ndarray, mask2: np.ndarray, 
                     method: str = 'union') -> np.ndarray:
        """
        Combine two masks using specified method.
        
        This is the combine_masks function extracted from wound_medsam.py.
        
        Args:
            mask1: First mask
            mask2: Second mask
            method: Combination method ('union', 'intersection', 'average')
            
        Returns:
            np.ndarray: Combined mask
        """
        try:
            if method == 'union':
                combined = np.logical_or(mask1, mask2).astype(np.uint8)
            elif method == 'intersection':
                combined = np.logical_and(mask1, mask2).astype(np.uint8)
            elif method == 'average':
                combined = (mask1.astype(float) * 0.6 + mask2.astype(float) * 0.4)
                combined = (combined > 0.5).astype(np.uint8)
            else:
                raise ValueError(f"Unknown combination method: {method}")
            
            logger.debug(f"Masks combined using {method} method")
            return combined
            
        except Exception as e:
            logger.error(f"Failed to combine masks: {e}")
            raise
    
    def estimate_uncertainty(self, image: np.ndarray, num_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using multiple samples.
        
        This is the estimate_uncertainty function extracted from wound_medsam.py.
        
        Args:
            image: Input image
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        try:
            if self.model_provider is None:
                raise RuntimeError("Model provider not available")
            
            predictions = []
            
            # Generate multiple predictions
            for _ in range(num_samples):
                pred = self._single_prediction(image)
                predictions.append(pred)
            
            # Calculate statistics
            predictions_array = np.array(predictions)
            mean_pred = np.mean(predictions_array, axis=0)
            uncertainty = np.var(predictions_array, axis=0)
            
            logger.debug(f"Uncertainty estimation completed: {num_samples} samples, mean_uncertainty={np.mean(uncertainty):.3f}")
            return mean_pred, uncertainty
            
        except Exception as e:
            logger.error(f"Failed to estimate uncertainty: {e}")
            raise
    
    def predict_with_uncertainty(self, image: np.ndarray, num_samples: int = 10) -> SegmentationResult:
        """
        Predict mask with uncertainty estimation.
        
        Args:
            image: Input image
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            SegmentationResult: Segmentation result with uncertainty
        """
        try:
            mean_pred, uncertainty = self.estimate_uncertainty(image, num_samples)
            confidence = 1.0 - np.mean(uncertainty)
            
            result = SegmentationResult(
                mask=mean_pred,
                confidence=confidence,
                method="unet_uncertainty"
            )
            
            logger.debug(f"Uncertainty prediction completed: confidence={confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to predict with uncertainty: {e}")
            raise
    
    def batch_predict(self, images: List[np.ndarray], use_tta: bool = False) -> List[SegmentationResult]:
        """
        Predict masks for multiple images.
        
        Args:
            images: List of preprocessed images
            use_tta: Whether to use test-time augmentation
            
        Returns:
            List[SegmentationResult]: List of segmentation results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.predict_mask(image, use_tta=use_tta)
                results.append(result)
                logger.debug(f"Batch prediction {i+1}/{len(images)} completed")
            except Exception as e:
                logger.warning(f"Failed to predict image {i+1}: {e}")
                # Add a dummy result to maintain list length
                results.append(SegmentationResult(
                    mask=np.zeros((128, 128), dtype=np.float32),
                    confidence=0.0,
                    method="failed"
                ))
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
    
    def get_segmentation_summary(self) -> Dict[str, Any]:
        """
        Get summary of segmentation pipeline configuration.
        
        Returns:
            dict: Segmentation pipeline configuration summary
        """
        return {
            'model_available': self.model_provider is not None,
            'tta_enabled': True,
            'uncertainty_estimation_enabled': True,
            'mask_combination_methods': ['union', 'intersection', 'average'],
            'default_tta_samples': 4,
            'default_uncertainty_samples': 10
        }